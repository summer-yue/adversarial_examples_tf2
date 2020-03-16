import unittest
import tensorflow as tf

from adversarial_examples_tf2.attackers.base_attacker import BaseAttacker


class DummyAttacker(BaseAttacker):

    def __init__(self):
        pass

    def generate_adversarial_examples(self, input_tensor, label_tensor):
        return input_tensor + 1


class BaseAttackerTest(tf.test.TestCase):

    def testGenerateAdversarialExamplesDataset(self):
        attacker = DummyAttacker()
        features = [1, 2, 3]
        labels = [0, 0, 1]

        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        perturbed_dataset = attacker.generate_adversarial_examples_dataset(
            dataset)
        perturbed_dataset_list = list(perturbed_dataset.as_numpy_iterator())

        expected_data = [(2, 0), (3, 0), (4, 1)]
        self.assertAllClose(expected_data, perturbed_dataset_list)


if __name__ == '__main__':
    tf.test.main()
