"""
Train and evaluate a simple fully connected MNIST classifier.

TODO(summeryue): attack the classifier with adversarial examples generated \
     with FDSM (Fast Gradient Sign Method) and re-evaluate the performance on \
     perturbed data.

# TODO(summeryue): Move the value errors into args documentation.
"""

import tensorflow as tf

from adversarial_examples_tf2.attackers.fgsmAttacker import FGSMAttacker
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
classifier.train(train_data, epochs=1)

# Evaluate the trained classifier on sample data.
loss, accuracy = classifier.evaluate(test_data)
print("Test loss :", loss)
print("Test accuracy :", accuracy)

# Perform fast gradient sign method attack on a sample of the test data.
sample_batch_num = 2
# The epsilon parameter used in FGSM indicating the scale of the perturbation.
epsilon = 0.1
attacker = FGSMAttacker(classifier, epsilon=epsilon)
sample_data = test_data.take(sample_batch_num)

sample_loss, sample_accuracy = classifier.evaluate(sample_data)
print("Loss on sampled test data:", sample_loss)
print("Accuracy on sampled test data:", sample_accuracy)

sample_features = tf.stack([features for features, _ in sample_data])
sample_features = tf.reshape(sample_features, [-1] + list(mnist_input_shape))
sample_labels = tf.stack([label for _, label in sample_data])
sample_labels = tf.reshape(sample_labels, [-1] + list(mnist_output_shape))

perturbed_features = attacker.generate_adversarial_examples(sample_features,
                                                            sample_labels)

# Evaluate the trained classifier on perturbed sample data.
perturbed_data = tf.data.Dataset.from_tensor_slices(
    (perturbed_features, sample_labels))
perturbed_data = perturbed_data.shuffle(buffer_size=500).batch(batch_size)
print("Perturbed data: {}".format(perturbed_data))
perturbed_loss, perturbed_accuracy = classifier.evaluate(perturbed_data)
print("Loss on perturbed test data:", perturbed_loss)
print("Accuracy on perturbed test data:", perturbed_accuracy)
