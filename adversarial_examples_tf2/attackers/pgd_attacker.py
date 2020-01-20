import tensorflow as tf

from adversarial_examples_tf2.attackers.base_attacker import BaseAttacker


class PGDAttacker(BaseAttacker):
    def __init__(self, model, epsilon, n_iter):
        """
        Initialize a projected gradient descent attacker.

        Args:
            model (BaseClassifier): the targeted classifier under attack.
            epsilon (double): the epsilon parameter used in FGSM indicating the
                scale of the perturbation.
            n_iter (int): the number of iterations of perturbations applied.
        """
        self.model = model
        self.epsilon = epsilon
        self.n_iter = n_iter

    def generate_adversarial_examples(self, input_tensor, label_tensor):
        """
        Generate a list of adversarial examples with the projected gradient
            descent attack.

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

        # TODO(summeryue): look up the appropriate initialization procedure.
        perturbed_tensor = tf.Variable(input_tensor)
        for i in range(self.n_iter):
            # TODO(summeryue): Make this into a tf.function to improve speed.
            with tf.GradientTape() as tape:
                logit_tensor = self.model.model(perturbed_tensor)
                loss = loss_fn(label_tensor, logit_tensor)

            gradient = tape.gradient(loss, perturbed_tensor)
            perturbed_tensor.assign_add(gradient)

            # Project perturbed_tensor onto the L-infinity ball around
            # input_tensor.
            perturbed_tensor.assign(self.epsilon * tf.sign(
                perturbed_tensor - input_tensor) + input_tensor)

        return perturbed_tensor.value()
