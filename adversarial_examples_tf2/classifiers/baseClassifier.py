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
            data: tf.data.Dataset object representing the data to be fed into
                the classifier.

        Returns:
            a boolean flag indicating whether the data is valid.

        """
        raise NotImplementedError

    @abstractmethod
    def train(self, data, epochs):
        """
        Train the classifier for a given number of epochs on a given training
        dataset.

        Args:
            data: tf.data.Dataset object representing the data the classifier
                uses for training.
            epochs: the number of epochs, the number of times the given dataset
                was iterated over during training.

        Raises:
            ValueError:
                if the training data is invalid - if validate_data(data) fails.
        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, data):
        """
        Evaluate the trained model on a test dataset.

        Args:
            data: tf.data.Dataset object representing the data the classifier
                uses for evaluation.

        Raises:
            ValueError:
                if the eval data shape does not match input_shape.
        """
        raise NotImplementedError

