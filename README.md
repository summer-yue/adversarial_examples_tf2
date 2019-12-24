# adversarial_attacks_tf2

A library for adversarial attacks and defenses implemented in Tensorflow 2.0. This library implements existing
adversarial attacks and defenses as benchmarks, and allow users to test their new attack/defense mechanisms on various
data sets.

TODO(summeryue): Document supported datasets, attacks and defenses.

Milestones:
- Build a vanila MNIST classifier.
- Build an FGSM attacker which attacks the vanila MNIST classifier.
- Build a vanila CIFAR10 classifier.
- Extend the FGSM attacker to be able to attack an CIFAR10 classifier.
- Set up a classifier which increases its robustness with adversarial training. The adversarial examples are generated
via FGSM.
- Add more attacks and defenses. TBD.
