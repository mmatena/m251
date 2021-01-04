"""Code for computing a diagonal approximation to the Fisher."""
import time
import datetime

from absl import logging
import bayes_opt as bayes
import tensorflow as tf

from del8.core.utils import hdf5_util

from .. import fisher_abcs


class DiagonalFisherComputer(fisher_abcs.FisherComputer):
    def __init__(self, model, total_examples, class_chunk_size=4096, y_samples=None):
        super().__init__()

        self.model = model
        self.total_examples = total_examples
        self.y_samples = y_samples
        self.class_chunk_size = class_chunk_size
        self.fisher_diagonals = [
            tf.Variable(tf.zeros(w.shape), trainable=False, name=f"fisher/{w.name}")
            for w in model.get_mergeable_variables()
        ]

    def train_step(self, data):
        if self.y_samples is None:
            return self.train_step_exact_y(data)
        else:
            return self.train_step_sample_y(data)

    @tf.function
    def train_step_exact_y(self, data):
        x, _ = data
        trainable_weights = self.model.get_mergeable_variables()

        with tf.GradientTape(persistent=True) as tape:
            logits = self.model.compute_logits(x, training=False)
            log_probs = tf.nn.log_softmax(logits, axis=-1)
        probs = tf.nn.softmax(logits)  # [batch, num_classes]
        batch_size = tf.cast(tf.shape(probs)[0], tf.float32)
        num_classes = tf.cast(tf.shape(probs)[1], tf.float32)

        num_chunks = tf.math.ceil(num_classes / float(self.class_chunk_size))
        num_chunks = tf.cast(num_chunks, tf.int32)
        for chunk_index in tf.range(num_chunks):
            with tape:
                log_probs_chunk = log_probs[
                    ...,
                    chunk_index
                    * self.class_chunk_size : (chunk_index + 1)
                    * self.class_chunk_size,
                ]
            probs_chunk = probs[
                ...,
                chunk_index
                * self.class_chunk_size : (chunk_index + 1)
                * self.class_chunk_size,
            ]
            actual_chunk_size = tf.cast(tf.shape(probs_chunk)[-1], tf.float32)

            grads = tape.jacobian(log_probs_chunk, trainable_weights)
            for g, fisher in zip(grads, self.fisher_diagonals):
                if g is None:
                    logging.info(
                        f"No gradients found for {fisher.name}. Skipping fisher "
                        "computing computation for those variables."
                    )
                    continue
                # g.shape = [batch, num_classes, *var.shape]
                update = tf.tensordot(probs_chunk, tf.square(g), [[0, 1], [0, 1]])
                fraction_of_total = batch_size / float(self.total_examples)
                fraction_of_total *= actual_chunk_size / num_classes
                fisher.assign_add(fraction_of_total * update)

        return {}

    @tf.function
    def train_step_sample_y(self, data):
        trainable_weights = self.model.get_mergeable_variables()

        with tf.GradientTape(persistent=True) as tape:
            # log_probs.shape = [y_samples, batch]
            log_probs = self.model.log_prob_of_y_samples(
                data, num_samples=self.y_samples, training=False
            )

            batch_size = tf.cast(tf.shape(log_probs)[1], tf.float32)

            log_prob_index = 0
            for log_prob in log_probs:
                with tape.stop_recording():
                    grads = tape.jacobian(log_prob, trainable_weights)
                    for g, fisher in zip(grads, self.fisher_diagonals):
                        if g is None:
                            if log_prob_index == 0:
                                logging.info(
                                    f"No gradients found for {fisher.name}. Skipping fisher "
                                    "computing computation for those variables."
                                )
                            continue
                        # g.shape = [batch, *var.shape]
                        update = tf.reduce_sum(tf.square(g), axis=0)
                        fraction_of_total = batch_size / tf.cast(
                            self.total_examples * self.y_samples, tf.float32
                        )
                        fisher.assign_add(fraction_of_total * update)
                log_prob_index += 1

        return {}

    def get_fisher_matrix(self):
        return DiagonalFisherMatrix(self.fisher_diagonals)

    def get_original_model(self):
        return self.model


class DiagonalFisherMatrix(fisher_abcs.FisherMatrix):
    def __init__(self, fisher_diagonals):
        # NOTE: Be careful that these aren't set to trainable. The FisherComputer
        # and the loader sets them to trainable=False, so we shouldn't have an
        # issue with that in our normal use-case.
        self.fisher_diagonals = fisher_diagonals

    @classmethod
    def load(cls, file):
        return cls(hdf5_util.load_variables_from_hdf5(file, trainable=False))

    def save(self, file):
        hdf5_util.save_variables_to_hdf5(self.fisher_diagonals, file)


@tf.function
def _merge_var(var, weighting, merge_vars, diags, min_fisher):
    diags = tf.maximum(diags, min_fisher)
    merge_vars = tf.stack(merge_vars)
    lhs = tf.einsum("m,m...->...", weighting, diags)
    rhs = tf.einsum("m,m...,m...->...", weighting, diags, merge_vars)
    var.assign(rhs / lhs)


def merge_models(
    merged_model, mergeable_models, weighting=None, single_task=True, min_fisher=1e-6
):
    # If single_task=True, then we only care about the score of the first model.
    if not weighting:
        weighting = len(mergeable_models) * [1.0]

    assert len(mergeable_models) == len(weighting)

    with tf.device("gpu"):
        for i, var in enumerate(merged_model.get_mergeable_variables()):
            lhs = []
            rhs = []
            for j, (weight, mm) in enumerate(zip(weighting, mergeable_models)):
                model = mm.model
                fisher_matrix = mm.fisher_matrix

                diag = fisher_matrix.fisher_diagonals[i]
                if not single_task or j == 0:
                    diag = tf.maximum(diag, min_fisher)

                tmp = weight * diag
                lhs.append(tmp)
                rhs.append(tmp * model.get_mergeable_variables()[i])
            rhs = tf.reduce_sum(rhs, axis=0)
            lhs = tf.reduce_sum(lhs, axis=0)
            var.assign(rhs / lhs)

    if single_task:
        heads = [mergeable_models[0].model.get_classifier_head()]
    else:
        heads = [m.model.get_classifier_head() for m in mergeable_models]
    merged_model.set_classifier_heads(heads)

    return merged_model


def merge_models_with_weightings(
    merged_model, mergeable_models, weightings, single_task=True, min_fisher=1e-6
):
    for weighting in weightings:
        merged = merge_models(
            merged_model,
            mergeable_models,
            weighting=weighting,
            single_task=single_task,
            min_fisher=min_fisher,
        )
        yield merged


###############################################################################


@tf.function
def _merge_body_fast_single_task(
    variables, weighting, merge_diags, merge_vars, min_fishers
):
    logging.info("Tracing _merge_body_fast_single_task")

    weighting = tf.unstack(weighting)
    min_fishers = tf.unstack(min_fishers)
    for var, merge_diag, merge_var in zip(variables, merge_diags, merge_vars):
        lhs = []
        rhs = []
        for w, d, mv, min_fisher in zip(weighting, merge_diag, merge_var, min_fishers):
            tmp = w * tf.maximum(d, min_fisher)
            lhs.append(tmp)
            rhs.append(tmp * mv)
        rhs = tf.reduce_sum(rhs, axis=0)
        lhs = tf.reduce_sum(lhs, axis=0)
        var.assign(rhs / lhs)


def _construct_fast_merge_assets(mergeable_models):
    num_vars = len(mergeable_models[0].model.get_mergeable_variables())

    merge_vars = [[] for _ in range(num_vars)]
    merge_diags = [[] for _ in range(num_vars)]

    for m in mergeable_models:
        model = m.model
        fisher_matrix = m.fisher_matrix
        for i, (v, d) in enumerate(
            zip(model.get_mergeable_variables(), fisher_matrix.fisher_diagonals)
        ):
            merge_vars[i].append(v)
            merge_diags[i].append(d)
    return merge_vars, merge_diags


# def _get_weighting(point, num_weights):
#     return [point[f'weight_{i}'] for i in range(num_weights)]


# def merge_search_best_weighting(
#     to_be_merged,
#     mergeable_models,
#     score_fn,
#     max_evals,
#     num_inits,
#     # algo=hyperopt.tpe.suggest,
#     min_fisher=1e-6
# ):
#     num_models = len(mergeable_models)
#     # NOTE: The first model in mergeable_models will be the one whose score
#     # we are attempting to maximize.
#     pbounds = {
#         f'weight_{i}': (0.0, 1.0) for i in range(num_models)
#     }

#     # Set up for merging.
#     to_be_merged.set_classifier_heads([mergeable_models[0].model.get_classifier_head()])
#     variables = to_be_merged.get_mergeable_variables()
#     merge_vars, merge_diags = _construct_fast_merge_assets(mergeable_models)

#     min_fishers = tf.constant([min_fisher] + (num_models - 1) * [0.0], dtype=tf.float32)

#     weightings = []
#     scores = []

#     time_marker = time.time()

#     def fn(**point):
#         nonlocal time_marker
#         new_time_marker = time.time()
#         elapsed_seconds = new_time_marker - time_marker
#         elapsed_nice = str(datetime.timedelta(seconds=elapsed_seconds))
#         logging.info(f"Hyperopt step took {elapsed_nice}.")
#         time_marker = new_time_marker

#         weighting = _get_weighting(point, num_weights=num_models)
#         weightings.append(weighting)

#         start_time = time.time()

#         weighting = tf.constant(weighting, dtype=tf.float32)
#         _merge_body_fast_single_task(variables, weighting, merge_diags, merge_vars, min_fishers)

#         elapsed_seconds = time.time() - start_time
#         elapsed_nice = str(datetime.timedelta(seconds=elapsed_seconds))
#         logging.info(f"Merging took {elapsed_nice}.")

#         ret = score_fn(to_be_merged)
#         scores.append(ret)

#         elapsed_seconds = time.time() - new_time_marker
#         elapsed_nice = str(datetime.timedelta(seconds=elapsed_seconds))
#         logging.info(f"Hyperopt fn call took {elapsed_nice}.")

#         return ret

#     # bounds_transformer = bayes.SequentialDomainReductionTransformer()
#     optimizer = bayes.BayesianOptimization(
#         f=fn,
#         pbounds=pbounds,
#         # verbose=0,
#         random_state=1,
#         # bounds_transformer=bounds_transformer
#     )
#     # Probe the unmerged model.
#     optimizer.probe(
#         params={
#             f'weight_{i}': float(not i) for i in range(num_models)
#         },
#         lazy=True,
#     )
#     optimizer.maximize(
#         init_points=num_inits,
#         n_iter=max_evals,
#     )

#     best_weighting = _get_weighting(optimizer.max['params'], num_weights=num_models)

#     merged_model = merge_models(
#         to_be_merged, mergeable_models, weighting=best_weighting, min_fisher=min_fisher, single_task=True
#     )

#     return merged_model, best_weighting, weightings, scores


#################################################################################################


# def _get_weighting(point, num_weights):
#     weights = [point[f'weight_{i}'] for i in range(num_weights - 1)]
#     last = 1.0 - sum(weights)
#     assert last >= 0.0
#     weights.append(last)
#     return weights


# def _is_in_simplex(point):
#     return sum(point.values()) <= 1.0


# def merge_search_best_weighting(
#     to_be_merged,
#     mergeable_models,
#     score_fn,
#     max_evals,
#     num_inits,
#     # algo=hyperopt.tpe.suggest,
#     min_fisher=1e-6
# ):
#     num_models = len(mergeable_models)
#     # NOTE: The first model in mergeable_models will be the one whose score
#     # we are attempting to maximize.
#     pbounds = {
#         f'weight_{i}': (0.0, 1.0) for i in range(num_models - 1)
#     }

#     # Set up for merging.
#     to_be_merged.set_classifier_heads([mergeable_models[0].model.get_classifier_head()])
#     variables = to_be_merged.get_mergeable_variables()
#     merge_vars, merge_diags = _construct_fast_merge_assets(mergeable_models)

#     min_fishers = tf.constant([min_fisher] + (num_models - 1) * [0.0], dtype=tf.float32)

#     weightings = []
#     scores = []

#     time_marker = time.time()

#     def fn(**point):
#         nonlocal time_marker
#         new_time_marker = time.time()
#         elapsed_seconds = new_time_marker - time_marker
#         elapsed_nice = str(datetime.timedelta(seconds=elapsed_seconds))
#         logging.info(f"Hyperopt step took {elapsed_nice}.")
#         time_marker = new_time_marker

#         if not _is_in_simplex(point):
#             return -100.0

#         weighting = _get_weighting(point, num_weights=num_models)
#         weightings.append(weighting)

#         start_time = time.time()

#         weighting = tf.constant(weighting, dtype=tf.float32)
#         _merge_body_fast_single_task(variables, weighting, merge_diags, merge_vars, min_fishers)

#         elapsed_seconds = time.time() - start_time
#         elapsed_nice = str(datetime.timedelta(seconds=elapsed_seconds))
#         logging.info(f"Merging took {elapsed_nice}.")

#         ret = score_fn(to_be_merged)
#         scores.append(ret)

#         elapsed_seconds = time.time() - new_time_marker
#         elapsed_nice = str(datetime.timedelta(seconds=elapsed_seconds))
#         logging.info(f"Hyperopt fn call took {elapsed_nice}.")

#         return ret

#     # bounds_transformer = bayes.SequentialDomainReductionTransformer()
#     optimizer = bayes.BayesianOptimization(
#         f=fn,
#         pbounds=pbounds,
#         # verbose=0,
#         random_state=1,
#         # bounds_transformer=bounds_transformer
#     )
#     # Probe the unmerged model.
#     optimizer.probe(
#         params={
#             f'weight_{i}': float(not i) for i in range(num_models - 1)
#         },
#         lazy=True,
#     )
#     optimizer.maximize(
#         init_points=num_inits,
#         n_iter=max_evals,
#     )

#     best_weighting = _get_weighting(optimizer.max['params'], num_weights=num_models)

#     merged_model = merge_models(
#         to_be_merged, mergeable_models, weighting=best_weighting, min_fisher=min_fisher, single_task=True
#     )

#     return merged_model, best_weighting, weightings, scores


#################################################################################################


def _get_weighting(point, num_weights, min_target_weight):
    weights = [min_target_weight] + [
        point[f"weight_{i}"] for i in range(1, num_weights)
    ]
    total = sum(weights)
    return [w / total for w in weights]


def _is_in_simplex(point):
    return sum(point.values()) <= 1.0


def merge_search_best_weighting(
    to_be_merged,
    mergeable_models,
    score_fn,
    max_evals,
    num_inits,
    # algo=hyperopt.tpe.suggest,
    min_fisher=1e-6,
    min_target_weight=0.25,
):
    num_models = len(mergeable_models)
    # NOTE: The first model in mergeable_models will be the one whose score
    # we are attempting to maximize.
    pbounds = {f"weight_{i}": (0.0, 1.0) for i in range(1, num_models)}

    # Set up for merging.
    to_be_merged.set_classifier_heads([mergeable_models[0].model.get_classifier_head()])
    variables = to_be_merged.get_mergeable_variables()
    merge_vars, merge_diags = _construct_fast_merge_assets(mergeable_models)

    min_fishers = tf.constant([min_fisher] + (num_models - 1) * [0.0], dtype=tf.float32)

    weightings = []
    scores = []

    time_marker = time.time()

    def fn(**point):
        nonlocal time_marker
        new_time_marker = time.time()
        elapsed_seconds = new_time_marker - time_marker
        elapsed_nice = str(datetime.timedelta(seconds=elapsed_seconds))
        logging.info(f"Hyperopt step took {elapsed_nice}.")
        time_marker = new_time_marker

        weighting = _get_weighting(
            point, num_weights=num_models, min_target_weight=min_target_weight
        )
        weightings.append(weighting)

        start_time = time.time()

        weighting = tf.constant(weighting, dtype=tf.float32)
        _merge_body_fast_single_task(
            variables, weighting, merge_diags, merge_vars, min_fishers
        )

        elapsed_seconds = time.time() - start_time
        elapsed_nice = str(datetime.timedelta(seconds=elapsed_seconds))
        logging.info(f"Merging took {elapsed_nice}.")

        ret = score_fn(to_be_merged)
        scores.append(ret)

        elapsed_seconds = time.time() - new_time_marker
        elapsed_nice = str(datetime.timedelta(seconds=elapsed_seconds))
        logging.info(f"Hyperopt fn call took {elapsed_nice}.")

        return ret

    bounds_transformer = bayes.SequentialDomainReductionTransformer()
    optimizer = bayes.BayesianOptimization(
        f=fn,
        pbounds=pbounds,
        # verbose=0,
        random_state=1,
        bounds_transformer=bounds_transformer,
    )
    # Probe the unmerged model.
    optimizer.probe(
        params={f"weight_{i}": float(0.0) for i in range(1, num_models)},
        lazy=True,
    )
    optimizer.maximize(
        init_points=num_inits,
        n_iter=max_evals,
    )

    best_weighting = _get_weighting(
        optimizer.max["params"],
        num_weights=num_models,
        min_target_weight=min_target_weight,
    )

    merged_model = merge_models(
        to_be_merged,
        mergeable_models,
        weighting=best_weighting,
        min_fisher=min_fisher,
        single_task=True,
    )

    return merged_model, best_weighting, weightings, scores
