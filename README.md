# TF2 Adversarial Attacks and Defenses Library

A light weight library for adversarial attacks and defenses implemented in TF2.
This library implements existing adversarial attacks and defenses as benchmarks,
and allow users to test their new attack/defense mechanisms on various data sets.

TODO(summeryue): Document supported datasets, attacks and defenses.

Milestones:
- (Done) Build a vanilla MNIST classifier.
- (Done) Build an FGSM attacker which attacks the vanilla MNIST classifier.
- (Done) Build an Projected Gradient Descent attacker which attacks the vanilla
MNIST classifier.
- (Done) Build a vanilla CIFAR 10 classifier.
- (Done) Extend the FGSM attacker to be able to attack an CIFAR 10 classifier.
- Set up a classifier which increases its robustness with adversarial training.
The adversarial examples are generated via PGD.
- Add more attacks and defenses. TBD.

Other TODOs:
- Throw a more meaningful error when user attempts to run experiments without
setting up classifiers.
