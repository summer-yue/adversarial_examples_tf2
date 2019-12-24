"""
Train and evaluate a simple fully connected MNIST classifier.

TODO(summeryue): attack the classifier with adversarial examples generated \
     with FDSM (Fast Gradient Sign Method) and re-evaluate the performace on \
     perturbed data.
"""

import tensorflow as tf

from adversarial_examples_tf2.classifiers.vanillaDNN import VanillaDNN


# Load MNIST data as tf.data.Dataset.
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

batch_size = 32
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.shuffle(buffer_size=5000).batch(batch_size)
test_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
test_data = test_data.shuffle(buffer_size=500).batch(batch_size)

# Compile and train the fully connected MNIST classifier.
mnist_input_shape = (28, 28)
mnist_output_shape = ()
model_params = [{"neuron_num": 128, "activation": "relu", "dropout_ratio":
                0.2}, {"neuron_num": 10, "activation": "softmax"}]
classifier = VanillaDNN(mnist_input_shape, mnist_output_shape, model_params)
classifier.train(train_data, epochs=2)

# Evaluate the trained classifier.
loss, accuracy = classifier.evaluate(test_data)
print("Test loss :", loss)
print("Test accuracy :", accuracy)
