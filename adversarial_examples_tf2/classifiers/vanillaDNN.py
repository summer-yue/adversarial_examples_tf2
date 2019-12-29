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
            input_shape (tuple(int)): The shape of the input data without the
                batch size dimension. For example, MNIST's input shape would be
                (28, 28).
            output_shape (tuple(int)): The shape of the output data without the
                batch size dimension. For example, MNIST's output shape would be
                ().
            model_params (list of dict): the parameters of the fully
                connected model. Each element is a dictionary representing one
                layer. For example, a 2 layered network is represented as
                [{"neuron_num": 128, "activation": "relu", "dropout_ratio": 0.2},
                {"neuron_num": 10, "activation": "softmax"}] The default
                activation function is relu if unspecified. The default
                dropout ratio is 0 if unspecified.
            optimizer (string or tf.keras.optimizers): The optimizer used for
                model training.
            loss: (string name of objective function, objective function or
                tf.losses.Loss instance): the loss function used for training.

        Raises:
            ValueError:
                If model_params miss required hyperparameters such as
                    "neural_num".

        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.optimizer = optimizer
        # Convert the input loss to a callable loss function if needed.
        self.loss = tf.losses.get(loss)

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

        self.model.compile(optimizer=self.optimizer, loss=self.loss,
                           metrics=["accuracy"])

    def validate_data(self, data):
        """
        Checks that whether the data shape is compatible with the shape
        specified in the classifier"s initializer.

        For example, for the MNIST dataset, the dataset feature shape is
        (None, 28, 28) and the dataset label shape is (None, ). self.input_shape
        is (28, 28) and self.output_shape is (). The None represents the
        batch size.

        Args:
            data (tf.data.Dataset): the dataset to be fed into
                the model for training or eval.

        Returns:
            (boolean): whether the input dataset is valid.

        """
        return data.element_spec[0].shape[1:] == self.input_shape and \
            data.element_spec[1].shape[1:] == self.output_shape

    def train(self, data, epochs):
        """
        Train the classifier for a given number of epochs on a given training
        dataset.

        Args:
            data (tf.data.Dataset): the dataset the model uses for training.
            epochs (int): the number of epochs, the number of times the given
                dataset was iterated over during training.

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
            data (tf.data.Dataset): the dataset the model uses for evaluation.

        Returns:
            (int or list of ints): the loss for the test dataset.
            (double or list of double): the accuracy for the test dataset.

        Raises:
            ValueError:
                if the evaluation data shape does not match input_shape.
        """
        if not self.validate_data(data):
            raise ValueError("The eval data is not a valid classifier "
                             "input: {}".format(data))
        loss, accuracy = self.model.evaluate(data)
        return loss, accuracy

    def get_loss_fn(self, reduction=tf.keras.losses.Reduction.AUTO):
        """

        Args:
            reduction: (tf.keras.losses.Reduction)(Optional) type of
                reduction to apply to loss. Default value is `AUTO`. `AUTO`
                indicates that the reduction option will be determined by the
                usage context. For almost all cases this defaults to
                `SUM_OVER_BATCH_SIZE`. When used in taking the gradient of
                the loss with respect to individual input such as in
                FGSMAttacker, reduction should be set to
                tf.keras.losses.Reduction.NONE.

        Returns:
            (LossFunctionWrapper or a callable loss function): the loss
                function the classifier uses, should take in (label_tensor,
                logit_tensor) and return the loss.

        TODO(summeryue): Output self.loss with a modified reduction function
            here instead.

        """
        return tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE)

