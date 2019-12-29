import tensorflow as tf

from abc import ABCMeta, abstractmethod


class BaseClassifier(object):
    __metaclass__ = ABCMeta
    """
    Defines the interface for classifiers in the adversarial examples library.
    """

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
                and the dataset feature shape needs to be (None, 28, 28).
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
                and the dataset feature shape needs to be (None, 28, 28).

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


