from abc import ABCMeta, abstractmethod


class BaseAttacker(object):
    __metaclass__ = ABCMeta
    """
    Defines the interface for attackers in the adversarial examples library.
    """

    @abstractmethod
    def generate_adversarial_examples(self, base_input):
        """
        Generate a list of adversarial examples from some original input
        examples.

        Args:
            base_input (tf.Tensor): Input data to be modified to become
                adversarial examples. The shape of base_input is (None,
                input_shape) where None is the dimension representing the
                number of examples.

        Returns:
            (tf.Tensor): perturbed examples of shape (None, input_shape)
                where None is the dimension representing the number of examples.
        """
        raise NotImplementedError()
