# Adversarial Attacks and Defenses Library in TF2

We provide a lightweight and beginner-friendly library for:
- Training simple image classifiers.
- Generating adversarial examples - perturbed inputs to a neural network that
results in incorrect outputs.
- Building more robust classifiers by defending against those attacks.

This library is a simple starting point to check out some code and experiment
with adversarial examples yourself. For a more complete coverage of state of
the art techniques of the field, we encourage you to check out [CleverHans](https://github.com/tensorflow/cleverhans).

Supported datasets:
- [MNIST handwritten digit dataset](http://yann.lecun.com/exdb/mnist/)
- [CIFAR-10 image dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

Supported attacks:
- [Fast Gradient Sign Method (FGSM)](https://arxiv.org/pdf/1412.6572.pdf)
- [Projected Gradient Descent (PGD)](https://arxiv.org/pdf/1706.06083.pdf)

Supported defenses:
- [Adversarial training](https://arxiv.org/pdf/1412.6572.pdf)

## Installation

First, clone this repository on your local machine.
```
git clone https://github.com/summer-yue/adversarial_examples_tf2.git
```

Navigate to this directory with the *setup.py* file, and install the packages
required to run our code.
```
pip install -e .
```
Note that this library runs on [Tensorflow 2.0](https://www.tensorflow.org/) and
above.

## Get Started

Try running our simple examples in *adversarial_examples_tf2/experiments*.

### Set up trained models

First, run *experiments/setup_classifiers/...* to train some classifiers for
experimentation. Classifier training could take a while, so we will first
generate some trained models to be used in our examples.

Navigate to *adversarial_examples_tf2/experiments/setup_classifiers*.

```
python generate_mnist_vanilla_classifiers.py
python generate_mnist_adv_training_classifiers.py
```

Trained models will be saved in the experiments/models/ folder.

### Run the adversarial attacks examples

After the models are created, let's generate some adversarial examples,
and feed them into our trained classifiers.

Navigate to *adversarial_examples_tf2/experiments/attacks*.

```
python mnist_fgsm_vanilla.py
python mnist_pgd_vanilla.py
```

Feel free to tweak the parameters in the code, and see how the performance
changes. For example, you may change the `epsilon` parameter from 0.1 to 0.3,
which means that you are applying a larger perturbation to the original inputs
when generating adversarial examples. You would expect the accuracy on the
perturbed examples to drop even further.

### Verify that adversarial training makes classifiers more robust

Now let's verify that adversarial training indeed makes your classifiers more
robust towards adversarial examples.

Navigate to the *adversarial_examples_tf2/experiments/defenses*.
```
python mnist_adv_training.py
```

In this example, we observe that the FGSM attack lowers the accuracy of a
vanilla classifier severely.

However, if this classifier is trained with adversarial training based on the
stronger PGD attack, you notice that its accuracy drops much less when attacked.

## Contributing
If you'd like to contribute, feel free to send me PRs directly. I can be reached
at summeryue@google.com for questions. Please note that this is not a Google
product.
