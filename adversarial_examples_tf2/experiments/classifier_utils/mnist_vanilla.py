from adversarial_examples_tf2.classifiers.vanilla_dnn import VanillaDNN

mnist_input_shape = (28, 28)
mnist_output_shape = ()

def get_trained_model(train_data, epochs):
    """
    Compile and train a simple 2 layered fully connected MNIST classifier.

    Args:
        train_data: (tf.data.Dataset): the dataset the model uses for training.
            The dataset feature and label shape needs to be (None, 28, 28) and
            (None, ).
        epochs (int): the number of epochs, the number of times the given
                dataset was iterated over during training.

    Returns:
        (BaseClassifier): a trained classifier.

    """
    model_params = [{"neuron_num": 128, "activation": "relu",
                     "dropout_ratio": 0.2},
                    {"neuron_num": 10, "activation":
                    "softmax"}]
    classifier = VanillaDNN(mnist_input_shape, mnist_output_shape, model_params)
    classifier.train(train_data, epochs)
    return classifier

def load_existing_model(path):
    """
    Load an existing classifier as a VanillaDNN classifier.

    Args:
        path: The path to which the model is saved. The model file needs to
                be a XXX file type.
    """
    model_params = [{"neuron_num": 128, "activation": "relu",
                     "dropout_ratio": 0.2},
                    {"neuron_num": 10, "activation":
                    "softmax"}]
    classifier = VanillaDNN(mnist_input_shape, mnist_output_shape, model_params)
    classifier.load_from_file(path)
    return classifier
