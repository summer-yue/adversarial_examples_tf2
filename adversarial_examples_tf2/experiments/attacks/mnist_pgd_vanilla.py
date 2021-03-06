"""
Train and evaluate a simple fully connected MNIST classifier.
Then apply Projected Gradient Descent attack on the model and re-evaluate its
accuracy.

"""

import tensorflow as tf

from adversarial_examples_tf2.attackers.pgd_attacker import PGDAttacker
from adversarial_examples_tf2.experiments.classifier_utils.mnist_vanilla import load_existing_model
from adversarial_examples_tf2.experiments.data_utils.mnist import load_mnist

def main():
    # Load MNIST data as tf.data.Dataset.
    batch_size = 32
    train_data, test_data = load_mnist(batch_size)

    model_num_for_experiment = 3
    total_model_num = 10
    paths = ["../models/mnist/fully_connected/model-{}-{}"
        .format(i+1, total_model_num) for i in range(model_num_for_experiment)]

    for path in paths:
        print("Evaluating model from {}".format(path))
        classifier = load_existing_model(path)
        # Evaluate the trained classifier on test data.
        # Uncomment to find out the accuracy and loss for the entire test set.
        # loss, accuracy = classifier.evaluate(test_data)
        # print("Test loss :", loss)
        # print("Test accuracy :", accuracy)
        sample_batch_num = 2

        sample_data = test_data.take(sample_batch_num)
        sample_loss, sample_accuracy = classifier.evaluate(sample_data)
        print("Accuracy on sampled test data:", sample_accuracy)

        sample_features = tf.stack([features for features, _ in sample_data])
        sample_features = tf.reshape(sample_features, [-1, 28, 28])
        sample_labels = tf.stack([label for _, label in sample_data])
        sample_labels = tf.reshape(sample_labels, [-1, ])

        # Perform fast gradient sign method attack on a sample of the test data.
        attacker = PGDAttacker(classifier, epsilon=0.1, n_iter=20)
        perturbed_features = attacker.generate_adversarial_examples(sample_features,
                                                                    sample_labels)

        # Evaluate the trained classifier on perturbed sample data.
        perturbed_data = tf.data.Dataset.from_tensor_slices(
            (perturbed_features, sample_labels))
        perturbed_data = perturbed_data.shuffle(buffer_size=500).batch(batch_size)
        print("Perturbed data: {}".format(perturbed_data))
        perturbed_loss, perturbed_accuracy = classifier.evaluate(perturbed_data)
        print("Accuracy on perturbed test data:", perturbed_accuracy)


if __name__ == "__main__":
    main()
