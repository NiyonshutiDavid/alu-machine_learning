#!/usr/bin/env python3
"""Class Neuron that defines a single neuron performing binary classification
"""


import numpy as np


class Neuron:
    """ Class Neuron
    """

    def __init__(self, nx):
        """ Instantiation function of the neuron

        Args:
            nx (_type_): _description_

        Raises:
            TypeError: _description_
            ValueError: _description_
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be positive')

        # initialize private instance attributes
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

        # getter function
    @property
    def W(self):
        """Return weights"""
        return self.__W

    @property
    def b(self):
        """Return bias"""
        return self.__b

    @property
    def A(self):
        """Return output"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron

        Args:
            X (numpy.ndarray): matrix with the input data of shape (nx, m)

        Returns:
            numpy.ndarray: The output of the neural network.
        """
        z = np.matmul(self.__W, X) + self.__b
        sigmoid = 1 / (1 + np.exp(-z))
        self.__A = sigmoid
        return self.__A

    def cost(self, Y, A):
        """ Compute the of the model using logistic regression

        Args:
            Y (np.array): True values
            A (np.array): Prediction valuesss

        Returns:
            float: cost function
        """
        # calculate
        loss = - (Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = np.mean(loss)
        return cost
