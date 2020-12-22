"""Abstract base classes for models."""
import abc


class MergeableModel(abc.ABC):
    # NOTE: Not all of these are required for mergeability per se, but
    # our code and experiments related to merging require them.
    @abc.abstractmethod
    def get_body(self):
        raise NotImplementedError

    @abc.abstractmethod
    def get_head(self):
        raise NotImplementedError

    @abc.abstractmethod
    def add_regularizer(self, regularizer):
        raise NotImplementedError
