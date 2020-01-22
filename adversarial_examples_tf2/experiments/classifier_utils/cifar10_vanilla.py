from adversarial_examples_tf2.classifiers.vanilla_dnn import VanillaDNN

cifar10_input_shape = (32, 32, 3)
cifar10_output_shape = 1

model_params = [{"neuron_num": 1024, "activation": "relu",
                 "dropout_ratio": 0.2},
                {"neuron_num": 600, "activation": "relu",
                 "dropout_ratio": 0.2},
                {"neuron_num": 100, "activation": "relu",
                 "dropout_ratio": 0.2},
                {"neuron_num": 10, "activation":
                    "softmax"}]


def get_trained_model(train_data, epochs):
    """
    Compile and train a simple 3 layered fully connected CIFAR10 classifier.

    Args:
        train_data: (tf.data.Dataset): the dataset the model uses for training.
            The dataset feature and label shape needs to be (None, 32, 32) and
            (None, ).
        epochs (int): the number of epochs, the number of times the given
                dataset was iterated over during training.

    Returns:
        (BaseClassifier): a trained classifier.

    """
    classifier = VanillaDNN(cifar10_input_shape, cifar10_output_shape,
                            model_params)
    classifier.train(train_data, epochs)
    return classifier


def load_existing_model(path):
    """
    Load an existing classifier as a VanillaDNN classifier.

    Args:
        path: The path to which the model is saved.
    """
    classifier = VanillaDNN(cifar10_input_shape, cifar10_output_shape,
                            model_params)
    classifier.load_from_file(path)
    return classifier
