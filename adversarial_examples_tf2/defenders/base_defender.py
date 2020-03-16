import tensorflow as tf

from abc import ABCMeta, abstractmethod


class BaseDefender(object):
    __metaclass__ = ABCMeta
    """
    Defines the interface for defenders in the adversarial examples library.
    """

    def __init__(self, classifier, attackers):
        """
        Args:
            classifier (classifiers.BaseClassifier): the classifier to be
                modified to become more robust.
            attackers: the attacking mechanisms against which the defender is
                trying to become more robust.
        """
        self.classifier = classifier
        self.attackers = attackers

