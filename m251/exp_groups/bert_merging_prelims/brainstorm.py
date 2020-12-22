"""Just thinking in code."""

"""
ExperimentTypes:
- Computation of Fisher matrix for pretrained BERT checkpoints and MLM task.
- Fine-tuning on GLUE (with different regularizers...)
- Computing Fisher matrices for the fine-tuned models on their respective GLUE tasks.
- Fisher merging of the checkpoints + evaluation on different tasks.
#
- Fine-tuning with frozen body on GLUE and then evaluation.
- Multi-task fine-tuning from the start (with different regularizers...?)
- Merging via distillation (definitely some more params here.)

NOTE: I'll probably want to do something (almost the same) for image models.
- Can I have an ExperimentGroup class that I have multiple instances of?
- Anything with subclassing or composition?

Misc:
- Should groups be able to be nested? For example, should the whole m251 project
  be a group itself with nested groups corresponding to different groups of experiments
  done for it?
- Should I rename ExperimentGroup to something like Project? (Maybe not Project itself, I'd
  want something that fits in with the nesting structure better.)
- Should I rename Experiment to something like Procedure?
    - Can make sense as the experiments are more like procedures.
    - Experiment could then refer to a grouping of procedure executions.

Storage: writing, updating, reading, and querying.
- Need to figure out how to apportion responsibilities between ExperimentGroup, Experiment, and Procedure.
- Should interactions with storage during a run only be mediated through instances of these three classes,
  which would injected instead of injecting storage directly?

Might want to move those classes outside of the `storage` folder in del8 core. Probably create own
subfolder. Probably call it experiment.

I think EGEPs should be mostly specific and mostly non-reusable. Put general and reusable stuff in @executables.
- One counterpoint to this might be repeating all the experiments for image models.
- However, writing code for this purpose is different than writing production code for some industrial purpose.
  (Think writing test code vs production code.) It's more like a lab journal if anything, where the
  point is to get across what you did clearly.
- So it's probably OK just to mostly copy and paster for that purpose.

Experiments need a good way of generating lists of procedure parameters from concise descriptions.
- At least in the future, we might want to read/query from storage to affect what parameters we produce.
    - Actually, we probably need that now or very soon.
"""

# "Pretend" imports
exp = None  # would be `from ... import experiment as exp`
data_class = None


@data_class.data_class()
class DiagonalFisherParams(object):
    # NOTE: This class will probably be defined in some general, re-useable place.
    # NOTE: These are ONLY the params that are SPECIFIC to DIAGONAL Fisher computation.
    def __init__(
        self,
        y_samples,
    ):
        pass


@data_class.data_class()
class ComputeMlmFisherParams(object):
    def __init__(
        self,
        tfds_task,
        pretrained_model,
        fisher_type,
        fisher_params,
        train_examples,
        sequence_length,
        batch_size,
    ):
        pass


@exp.procedure(
    params_cls=ComputeMlmFisherParams,
)
class ComputeMlmFisherProcedure(object):
    def __init__(self, params):
        self.params = params

    @classmethod
    def from_params(cls, params):
        # NOTE: Probably put this on the procedure base class.
        return cls(params)

    def create_execution_item(self):
        # NOTE: This is where we'll handle setting up the injection config
        # and maybe some kwargs for the executable class.
        pass


@exp.experiment(
    # The user is to manually generate the uuid (maybe provide a script to do so) and
    # paste it in here.
    uuid="0693ceb02f4048f3afd0b561fd2cb351",
    # NOTE: The idea is that you can create all the injections just from the params_cls instance.
    # IDK if there is some good way to enforce this.
    # NOTE: Can set only procedure_cls and use the class and its associated
    # params. Or you can only set params_cls and then assume that experiment
    # is its own procedure. If so, then you need to provide a `create_execution_item(self, params)`
    # method. If both, then use procedure_cls and verify that stuff matches.
    params_cls=None,
    procedure_cls=ComputeMlmFisherProcedure,
    ###
    # params_cls=ComputeMlmFisherParams,
    # procedure_cls=None,
    # NOTE: Maybe have user specify AT THE CLASS LEVEL what they want as "partial" params.
    # NOTE: That's probably best for a first pass, can create utility methods for generating
    # the partials from concise descriptors later.
    varying_params=[
        # NOTE: As a first pass, I'll just have this be a list of dicts of kwargs to
        # the params class. Eventually, I'd want a way to specific attributes of nested params.
        {"pretrained_model": "large", "batch_size": 1},
        {"pretrained_model": "base", "batch_size": 2},
        {"pretrained_model": "small", "batch_size": 4},
    ],
    fixed_params={
        "tfds_task": "wiki40b/Wiki40B.en",
        "fisher_type": "DIAGONAL",
        "fisher_params": DiagonalFisherParams(y_samples=8),
        "train_examples": 32768,
        "sequence_length": 256,
    },
    # Params that uniquely identifies each procedure run.
    # Given the list of varying "partial" params, we check that these fields are indeed unique
    # for each params instance.
    # Defaults all the non-empty fields in the "partial" params.
    #
    # A @data_class RunKey item will be inserted into storage for each run to facilitate
    # retrieval of that run's assets in the future.
    key_fields={"pretrained_model"},
)
class ComputeMlmFishersFirstExperiment(object):
    # NOTE: This class will be fairly specific and all of its specificities
    # probably are not going to be reflected in its name.
    #
    # NOTE: These probably should be allowed to have arguments in their __init__ method.
    # They might pretty much end up being a singleton.

    def get_procedure_parameters(self):
        params_cls = self.procedure_cls.params_cls
        params = []
        for varying in self.varying_params:
            params.append(params_cls(**self.fixed_params, **varying))
        return params

    # # IDK if we want something like this.
    # @exp.stores("Some @data_class class.")
    # def store_mlm(self, ...):
    #     pass


@exp.experiment_group(
    uuid="70d7ee6c70734a0e963f2bece3a73c8d",
    experiments=[
        ComputeMlmFishersFirstExperiment,
    ],
)
class FirstExperimentsGroup(object):
    # NOTE: These probably should be allowed to have arguments in their __init__ method.
    # They might pretty much end up being a singleton.
    pass

    def get_glue_ft_run_id(self, pretrained_model, task, reg_type=None, reg_str=0):
        pass


# - Computation of Fisher matrix for pretrained BERT checkpoints and MLM task.
# - Fine-tuning on GLUE (with different regularizers...)
# - Computing Fisher matrices for the fine-tuned models on their respective GLUE tasks.
# - Fisher merging of the checkpoints + evaluation on different tasks.
# #
# - Fine-tuning with frozen body on GLUE and then evaluation.
# - Multi-task fine-tuning from the start (with different regularizers...?)
# - Merging via distillation (definitely some more params here.
