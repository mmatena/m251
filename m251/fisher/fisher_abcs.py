"""TODO: Add title."""
import abc
import tensorflow as tf


class FisherMatrix(abc.ABC):
    @classmethod
    def get_file_suffix(cls):
        # Suffix of the saved file.
        return ".h5"

    @classmethod
    @abc.abstractmethod
    def load(cls, file: str):
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, file: str):
        raise NotImplementedError


class FisherComputer(tf.keras.Model, abc.ABC):
    @abc.abstractmethod
    def get_fisher_matrix(self) -> FisherMatrix:
        # NOTE: This should return the already computed Fisher matrix,
        # not compute the Fisher itself.
        raise NotImplementedError

    @abc.abstractmethod
    def get_original_model(self):
        raise NotImplementedError
