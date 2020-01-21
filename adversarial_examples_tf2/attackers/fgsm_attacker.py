import tensorflow as tf

from adversarial_examples_tf2.attackers.base_attacker import BaseAttacker


class FGSMAttacker(BaseAttacker):
    def __init__(self, model, epsilon):
        """
        Initialize a fast gradient sign method attacker.

        Args:
            model (BaseClassifier): the targeted classifier under attack.
            epsilon (double): The epsilon parameter used in FGSM indicating the
                scale of the perturbation.
        """
        self.model = model
        self.epsilon = epsilon

    def generate_adversarial_examples(self, input_tensor, label_tensor):
        """
        Generate a list of adversarial examples with the fast gradient sign
            method attack.

        Args:
            input_tensor (tf.Tensor): input data to be modified to become
                adversarial examples. The shape of base_input is (None,
                input_shape) where None is the dimension representing the
                number of examples.
            label_tensor (tf.Tensor): labels of shape (None, output_shape)
                where None is the dimension representing the number of examples.

        Returns:
            (tf.Tensor): adversarial examples of shape (None, input_shape).
        """

        loss_fn = self.model.get_loss_fn(
            reduction=tf.keras.losses.Reduction.NONE)
        with tf.GradientTape() as tape:
            tape.watch(input_tensor)
            output_tensor = self.model.model(input_tensor)
            loss = loss_fn(label_tensor, output_tensor)

        gradient = tape.gradient(loss, input_tensor)
        perturbed_input_tensor = input_tensor + self.epsilon * tf.sign(gradient)
        return perturbed_input_tensor
