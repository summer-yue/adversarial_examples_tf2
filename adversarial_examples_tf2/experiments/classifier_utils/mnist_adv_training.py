from adversarial_examples_tf2.attackers.pgd_attacker import PGDAttacker
from adversarial_examples_tf2.classifiers.vanilla_dnn import VanillaDNN
from adversarial_examples_tf2.defenders.adversarial_training_defender import AdversarialTrainingDefender

mnist_input_shape = (28, 28)
mnist_output_shape = ()

def get_trained_model(train_data, epochs):
    """
    Compile and train a 2 layered fully connected MNIST classifier, with
        adversarial training on data generated by a projected gradient descent
        attacker.

    Args:
        train_data: (tf.data.Dataset): the dataset the model uses for training.
            Each element is a (features, label) tuple of Tensors.
            The feature and label shape needs to be (28, 28) and ().
        epochs (int): the number of epochs, the number of times the given
            dataset was iterated over during training. We apply the same number
            of epochs on the original training data s well as the adversarially
            generated training data.

    Returns:
        (VanillaDNN): a trained classifier.

    """
    model_params = [{"neuron_num": 128, "activation": "relu",
                     "dropout_ratio": 0.2},
                    {"neuron_num": 10, "activation":
                    "softmax"}]

    classifier = VanillaDNN(mnist_input_shape, mnist_output_shape, model_params)
    attacker = PGDAttacker(classifier, epsilon=0.1, n_iter=20)
    defender = AdversarialTrainingDefender(classifier, [attacker])

    perturbed_data = defender.generate_adversarial_training_data(train_data)
    all_data = train_data.concatenate(perturbed_data)

    classifier.train(all_data, epochs)
    return classifier
