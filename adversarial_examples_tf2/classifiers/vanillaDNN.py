import tensorflow as tf

from adversarial_examples_tf2.classifiers.baseClassifier import BaseClassifier


class VanillaDNN(BaseClassifier):
    """
    Defines a simple fully connected classifier with dropout regularization.
    """

    def __init__(self, input_shape, output_shape, model_params,
                 optimizer="adam",
                 loss=tf.keras.losses.SparseCategoricalCrossentropy()):
        """
        Initializes and compiles a fully connected neural net model with
        specified activation functions and dropout ratio for each layer.

        Args:
            input_shape: The shape of the input data without the batch size
                dimension. For example, MNIST's input shape would be (28, 28).
            output_shape: The shape of the output data without the batch size
                dimension. For example, MNIST's output shape would be ().
            model_params: A list representing the parameters of the fully
                connected model. Each element is a dictionary representing one
                layer. For example, a 2 layered network is represented as
                [{"neuron_num": 128, "activation": "relu", "dropout_ratio":
                0.2}, {"neuron_num": 10, "activation": "softmax"}] The
                default activation function is relu if unspecified. The default
                dropout ratio is 0 if unspecified.
            optimizer: string (name of optimizer) or tf.keras.optimizers
                instance. The optimizer used for model training.
            loss: string (name of objective function), objective function or
                tf.losses.Loss instance.

        Raises:
            ValueError:
                If model_params miss required hyperparameters such as
                    "neural_num".

        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.optimizer = optimizer
        self.loss = loss

        self.model = tf.keras.models.Sequential([tf.keras.layers.Flatten(
            input_shape=self.input_shape)])
        for layer_params in model_params:
            if "neuron_num" not in layer_params  \
                    or layer_params["neuron_num"] <= 0:
                raise ValueError("Invalid neuron_num specification in model "
                                 "params: {}".format(model_params))
            if "activation" not in layer_params:
                layer_params["activation"] = "relu"
            self.model.add(tf.keras.layers.Dense(layer_params["neuron_num"],
                                                 activation=layer_params[
                                                     "activation"]))
            if "dropout_ratio" in layer_params:
                self.model.add(tf.keras.layers.Dropout(layer_params[
                                                           "dropout_ratio"]))

        self.model.compile(optimizer=self.optimizer,
                loss=self.loss, metrics=["accuracy"])

    def validate_data(self, data):
        """
        Checks that whether the data shape is compatible with the shape
        specified in the classifier"s initializer.

        For example, for the MNIST dataset, the dataset feature shape is
        (None, 28, 28) and the dataset label shape is (None, ). self.input_shape
        is (28, 28) and self.output_shape is (). The None represents the
        batch size.

        Args:
            data: tf.data.Dataset object representing the data to be fed into
                the classifier for training or eval.

        Returns:
            a boolean flag indicating whether the data is valid.

        """
        return data.element_spec[0].shape[1:] == self.input_shape and \
            data.element_spec[1].shape[1:] == self.output_shape

    def train(self, data, epochs):
        """
        Train the classifier for a given number of epochs on a given training
        dataset.

        Args:
            data: tf.data.Dataset object representing the data the classifier
                uses for training.
            epochs: the number of epochs, the number of times the given dataset
                was iterated over during training.

        Raises:
            ValueError:
                if the training data shape does not match input_shape.
        """
        if not self.validate_data(data):
            raise ValueError("The training data is not a valid classifier "
                             "input: {}".format(data))
        self.model.fit(data, epochs=epochs)

    def evaluate(self, data):
        """
        Evaluate the trained model on a test dataset.

        Args:
            data: tf.data.Dataset object representing the data the classifier
                uses for evaluation.

        Returns:
            The loss and accuracy on the input test dataset.

        Raises:
            ValueError:
                if the evaluation data shape does not match input_shape.
        """
        if not self.validate_data(data):
            raise ValueError("The eval data is not a valid classifier "
                             "input: {}".format(data))
        loss, accuracy = self.model.evaluate(data)
        return loss, accuracy
