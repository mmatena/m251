"""Abstract base classes for models."""
import abc


class FisherableModel(abc.ABC):
    """ABC for model we can compute Fisher of.

    NOTE: A lot of times I conflate mergeability with Fisher-ability.
    """

    @abc.abstractmethod
    def get_mergeable_body(self):
        # Probably should return something like a keras layer.
        raise NotImplementedError

    def get_mergeable_variables(self):
        return self.get_mergeable_body().trainable_variables

    #############################################

    @abc.abstractmethod
    def compute_logits(self, x, training=None, mask=None):
        raise NotImplementedError

    @abc.abstractmethod
    def log_prob_of_y_samples(self, data, num_samples, training=None, mask=None):
        raise NotImplementedError


class MergeableModel(FisherableModel):
    # NOTE: Not all of these are required for mergeability per se, but
    # our code and experiments related to merging require them.
    #
    # NOTE: We also assume that we are dealing with classifiers.

    # The key to use in multtask dicts when we have a single task model.
    # If each task has multiple entries in the dict, then this may not
    # actually correspond to an entry in the dict. However, try to use
    # this whenever possible.
    SINGLE_TASK_KEY = "task_0"

    @abc.abstractmethod
    def assert_single_task(self):
        # Useful when we have a multi-task model but want to use it to represent
        # single-task models in certain cases.
        raise NotImplementedError

    @abc.abstractmethod
    def add_regularizer(self, regularizer):
        raise NotImplementedError

    #############################################

    @abc.abstractmethod
    def get_classifier_head(self):
        raise NotImplementedError

    @abc.abstractmethod
    def set_classifier_heads(self, heads):
        raise NotImplementedError

    #############################################

    @abc.abstractmethod
    def compute_task_logits(self, task_inputs, task, training=False):
        raise NotImplementedError

    @abc.abstractmethod
    def get_num_classes_for_task(self, task):
        pass
