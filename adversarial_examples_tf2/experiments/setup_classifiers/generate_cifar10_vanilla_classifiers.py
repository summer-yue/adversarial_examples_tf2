"""
Train and save 10 simple fully connected CIFAR10 classifiers for evaluating
attack methods under the experiments/models folder.

"""

from __future__ import absolute_import

from adversarial_examples_tf2.experiments.classifier_utils.cifar10_vanilla \
    import get_trained_model
from adversarial_examples_tf2.experiments.data_utils.cifar10 import load_cifar10


def main():
    # Load CIFAR10 data as tf.data.Dataset.
    batch_size = 16
    train_data, test_data = load_cifar10(batch_size)

    model_num = 5
    paths = ["../models/cifar10/fully_connected/model-{}-{}"
                 .format(i + 1, model_num) for i in range(model_num)]

    print("Paths: {}".format(paths))

    epochs = 20
    for path in paths:
        # Compile and train a simple fully connected CIFAR10 classifier.
        classifier = get_trained_model(train_data, epochs=epochs)
        print("Saving model to path {}.".format(path))
        classifier.save_to_file(path)


if __name__ == "__main__":
    main()
