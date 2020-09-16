"""Copyright © 2020-present, Swisscom (Schweiz) AG.
All rights reserved.

Abstract Metric class, used for evaluating results.

The abstract Metric class contains a basic skeleton and some implementation
details that will be shared among its children classes. Its function is to
evaluate results obtained using a certain model.
"""
from abc import ABC, abstractmethod
import torch


class Metric(ABC):
    """Abstract MetricAtK class.

    The Abstract MetricAtK class represents the parent class that is inherited
    if all concrete metric implementations.

    Attributes:
        _name: A string indicating the name of the metric.
        _k: An integer denoting the first N items upon which to calculate
            the given metric.
    """
    def __init__(self, name):
        """Inits MetricAtK with its name and k value.
        Raises:
            TypeError: The k value is not an integer or is not set.
            ValueError: The k value is smaller than 1.
        """
        super().__init__()
        self._name = name

    def get_name(self):
        """Returns the name of the MetricAtK class."""
        return self._name


    @abstractmethod
    def evaluate(self, y_true, y_pred, model):
        """Evaluates the given predictions with the implemented metric.

        Calculates the implemented metric on the passed predicted and true
        values at k.

        Args:
            y_true: A PyTorch tensor of true values.
            y_pred: A PyTorch tensor of predicted values.

        Returns:
            Will return a float with the calculated metric value, currently
            unimplemented.
        Raises:
            TypeError: An error occured while accessing the arguments -
                one of the arguments is NoneType.
            ValueError: An error occured when checking the dimensions of the
                y_pred and y_true arguments.
        """
        pass
