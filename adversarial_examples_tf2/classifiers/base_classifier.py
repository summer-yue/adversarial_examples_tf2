import tensorflow as tf

from abc import ABCMeta, abstractmethod


class BaseClassifier(object):
    __metaclass__ = ABCMeta
    """
    Defines the interface for classifiers in the adversarial examples library.
    """
    def __init__(self, input_shape, output_shape, model_params,
                 optimizer="adam",
                 loss=tf.keras.losses.SparseCategoricalCrossentropy()):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.model_params = model_params
        self.optimizer = optimizer
        # Convert the input loss to a callable loss function if needed.
        self.loss = tf.losses.get(loss)

    @abstractmethod
    def validate_data(self, data):
        """
        Check whether a given dataset is a valid input for the classifier.

        Args:
            data (tf.data.Dataset): the dataset to be fed into the classifier.

        Returns:
            (boolean): whether the data is valid.

        """
        raise NotImplementedError

    @abstractmethod
    def train(self, data, epochs):
        """
        Train the classifier for a given number of epochs on a given training
        dataset.

        Args:
            data (tf.data.Dataset): the dataset the model uses for training. The
                dataset shape needs to match the input shape set in the
                initializer. For example, for MNIST, self.input_shape = (28, 28)
                and the dataset feature shape needs to be (None, 28, 28) and
                the dataset label shape needs to be (None, ).
            epochs (int): the number of epochs, the number of times the given
                dataset was iterated over during training.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, data):
        """
        Evaluate the trained model on a test dataset.

        Args:
            data (tf.data.Dataset): the dataset the model uses for evaluation.
                The dataset shape needs to match the input shape set in the
                initializer. For example, for MNIST, self.input_shape = (28, 28)
                and the dataset feature shape needs to be (None, 28, 28) and
                the dataset label shape needs to be (None, ).

        """
        raise NotImplementedError

    @abstractmethod
    def get_loss_fn(self, reduction=tf.keras.losses.Reduction.AUTO):
        """

        Args:
            reduction: (tf.keras.losses.Reduction)(Optional) Type of
                reduction to apply to loss. Default value is `AUTO`. `AUTO`
                indicates that the reduction option will be determined by the
                usage context.
                For almost all cases this defaults to `SUM_OVER_BATCH_SIZE`.
                When used in taking the gradient of the loss with respect to
                individual input such as in FGSMAttacker, reduction should
                be set to tf.keras.losses.Reduction.NONE.

        Returns:
            (LossFunctionWrapper or a callable loss function): should take in (
            label_tensor, logit_tensor) and return the loss.
        """
        raise NotImplementedError

    @abstractmethod
    def save_to_file(self, path):
        """
        Save self.model on disk at a specified path.

        Args:
            path: The path to which the model is saved. The model file needs to
                be a XXX file type.

        """
        raise NotImplementedError

    @abstractmethod
    def load_from_file(self, path):
        """
        Load an existing model from disk into self.model.

        Args:
            path: The path to which the model is saved. The model file needs to
                be a XXX file type.

        Throws:
            ValueError if the loaded model is incompatible with existing
                instance variables related to model parameters.

        """
        raise NotImplementedError


