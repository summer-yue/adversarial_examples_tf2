from adversarial_examples_tf2.classifiers.vanilla_dnn import VanillaDNN


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
    mnist_input_shape = (28, 28)
    mnist_output_shape = ()
    model_params = [{"neuron_num": 128, "activation": "relu",
                     "dropout_ratio": 0.2},
                    {"neuron_num": 10, "activation":
                    "softmax"}]
    classifier = VanillaDNN(mnist_input_shape, mnist_output_shape, model_params)
    classifier.train(train_data, epochs)
    return classifier

