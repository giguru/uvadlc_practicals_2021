################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2021
# Date Created: 2021-11-01
################################################################################
"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization. 
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # Kaiming HE initialization is a normal distribution with mu=0, but sigma depends on whether ReLu was applied
        # on the layer before.
        # For the first layer (l = 1), we should have n(l) * Var[w] = 1, because there is no ReLU applied on the
        # input signal. This can be rewritten to Var[w] = 1/n(l).
        if input_layer:
            kaimingHeStandardDevication = np.sqrt(1/in_features)
        else:
            kaimingHeStandardDevication = np.sqrt(2/in_features)

        self.in_features = in_features
        self.out_features = out_features
        self.params = {
            'weight': np.random.normal(loc=0, scale=kaimingHeStandardDevication, size=(in_features, out_features)),
            'bias': np.zeros(shape=out_features, dtype=np.float64)
        }
        self.grads = {
            'weight': np.zeros(shape=(self.in_features, self.out_features), dtype=np.float64),
            'bias': np.zeros(self.out_features, dtype=np.float64)
        }
        self.last_out = None
        self.last_input = None
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        out = np.matmul(x, self.params['weight']) + self.params['bias']
        self.last_out = out.copy()
        self.last_input = x.copy()
        #######################
        # END OF YOUR CODE    #
        #######################
        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.grads['weight'] += np.matmul(self.last_input.T, dout)
        self.grads['bias'] += np.sum(dout, axis=0, keepdims=False)
        dx = np.matmul(dout, self.params['weight'].T)
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.last_out = None
        self.last_input = None

        # Reset the gradients as well.
        for k in self.grads.keys():
            self.grads[k] = np.zeros(shape=self.grads[k].shape, dtype=self.grads[k].dtype)

        #######################
        # END OF YOUR CODE    #
        #######################


class ReLUModule(object):
    """
    ReLU activation module.
    """

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        out = np.maximum(0, x)
        self.last_out = out.copy()
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        dx = np.where(self.last_out > 0, dout, 0)
        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.last_out = None
        #######################
        # END OF YOUR CODE    #
        #######################


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        diff = x - x.max(axis=1, keepdims=True)
        y = np.exp(diff)
        out = y / y.sum(axis=1, keepdims=True)
        self.last_out = out.copy()
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        I = np.eye(dout.shape[1], dout.shape[1])
        outer_product = np.einsum('ij,ik->ijk', self.last_out, self.last_out)
        d_softmax = np.einsum('ij,jk->ijk', self.last_out, I) - outer_product
        dx = np.einsum('ijk,ik->ij', d_softmax, dout)
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.last_out = None
        #######################
        # END OF YOUR CODE    #
        #######################


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        n_samples = x.shape[0]
        T = np.zeros(x.shape, dtype=np.float64)
        T[np.arange(y.shape[0]), y] = 1.0
        out = -np.sum(np.multiply(T, np.log(x)))
        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        num_samples = y.shape[0]
        T = np.zeros(x.shape, dtype=np.float64)
        T[np.arange(num_samples), y] = 1.
        # dx = -np.divide(T, x + 0.0001)
        dx = (x - T) / -num_samples
        #######################
        # END OF YOUR CODE    #
        #######################

        return dx