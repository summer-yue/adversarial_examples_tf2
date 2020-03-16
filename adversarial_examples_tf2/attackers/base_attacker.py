from abc import ABCMeta, abstractmethod


class BaseAttacker(object):
    __metaclass__ = ABCMeta
    """
    Defines the interface for attackers in the adversarial examples library.
    """

    @abstractmethod
    def generate_adversarial_examples(self, input_tensor, label_tensor):
        """
        Generate a list of adversarial examples from some original input
        examples.

        Args:
            input_tensor (tf.Tensor): input data to be modified to become
                adversarial examples. The shape of base_input is (None,
                input_shape) where None is the dimension representing the
                number of examples.
            label_tensor (tf.Tensor): labels of shape (None, output_shape)
                where None is the dimension representing the number of examples.

        Returns:
            (tf.Tensor): perturbed examples of shape (None, input_shape)
                where None is the dimension representing the number of examples.
        """
        raise NotImplementedError()

    def generate_adversarial_examples_dataset(self, data):
        """
        Generate a dataset of adversarial examples from some original dataset.

        Args:
            data (tf.Dataset): Input dataset to be modified to generate
                adversarial examples. Each element in the dataset is a
                (features, label) tuple of Tensors. The shape of features is
                (None, input_shape) where None is the dimension representing the
                number of examples. The shape of labels is (None, output_shape).

        Returns:
            (tf.Dataset): Perturbed dataset of (perturbed_features, label)
                Tensors. The shape of the Tensors are the same as the shapes of
                the original dataset.
        """
        def perturb_fn(features, label):
            pertubed_features = self.generate_adversarial_examples(
                features, label)
            return (pertubed_features, label)

        perturbed_dataset = data.map(perturb_fn, num_parallel_calls=3)
        return perturbed_dataset
