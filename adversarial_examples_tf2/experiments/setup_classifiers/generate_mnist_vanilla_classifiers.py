"""
Train and save 10 simple fully connected MNIST classifiers for evaluating
attack methods under the experiments/models folder.

"""

from __future__ import absolute_import

from adversarial_examples_tf2.experiments.classifier_utils.mnist_vanilla import get_trained_model
from adversarial_examples_tf2.experiments.data_utils.mnist import load_mnist


def main():
    # Load MNIST data as tf.data.Dataset.
    batch_size = 32
    train_data, test_data = load_mnist(batch_size)

    model_num = 10
    paths = ["../models/mnist/fully_connected/model-{}-{}"
        .format(i+1, model_num) for i in range(model_num)]

    print("Paths: {}".format(paths))

    epochs = 5
    for path in paths:
        # Compile and train a simple fully connected MNIST classifier.
        classifier = get_trained_model(train_data, epochs=epochs)
        print("Saving model to path {}.".format(path))
        classifier.save_to_file(path)

if __name__ == "__main__":
    main()
