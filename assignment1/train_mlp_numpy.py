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
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from tqdm.auto import tqdm
from copy import deepcopy
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils
import matplotlib.pyplot as plt
import torch


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.

    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 1D int array of size [batch_size]. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions between 0 and 1,
                i.e. the average correct predictions over the whole batch

    TODO:
    Implement accuracy computation.
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    # Use 'sum()' to count the number of True values in the resulting array
    accuracy = (np.argmax(predictions, axis=1) == targets).sum()
    accuracy /= len(targets)
    #######################
    # END OF YOUR CODE    #
    #######################

    return accuracy


def evaluate_model(model, data_loader):
    """
    Performs the evaluation of the MLP model on a given dataset.

    Args:
      model: An instance of 'MLP', the model to evaluate.
      data_loader: The data loader of the dataset to evaluate.
    Returns:
      avg_accuracy: scalar float, the average accuracy of the model on the dataset.

    TODO:
    Implement evaluation of the MLP model on a given dataset.

    Hint: make sure to return the average accuracy of the whole dataset, 
          independent of batch sizes (not all batches might be the same size).
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    preds_val, labels_val = None, np.array([])
    for batch_inputs, batch_labels in tqdm(data_loader):
        out = model.forward(batch_inputs)
        preds_val = np.vstack((preds_val, out)) if preds_val is not None else out
        labels_val = np.concatenate((labels_val, batch_labels))
    avg_accuracy = accuracy(preds_val, labels_val)
    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_accuracy


def train(hidden_dims, lr, batch_size, epochs, seed, data_dir):
    """
    Performs a full training cycle of MLP model.

    Args:
      hidden_dims: A list of ints, specificying the hidden dimensionalities to use in the MLP.
      lr: Learning rate of the SGD to apply.
      batch_size: Minibatch size for the data loaders.
      epochs: Number of training epochs to perform.
      seed: Seed to use for reproducible results.
      data_dir: Directory where to store/find the CIFAR10 dataset.
    Returns:
      model: An instance of 'MLP', the trained model that performed best on the validation set.
      val_accuracies: A list of scalar floats, containing the accuracies of the model on the
                      validation set per epoch (element 0 - performance after epoch 1)
      test_accuracy: scalar float, average accuracy on the test dataset of the model that 
                     performed best on the validation. Between 0.0 and 1.0
      logging_info: An arbitrary object containing logging information. This is for you to 
                    decide what to put in here.

    TODO:
    - Implement the training of the MLP model. 
    - Evaluate your model on the whole validation set each epoch.
    - After finishing training, evaluate your model that performed best on the validation set, 
      on the whole test dataset.
    - Integrate _all_ input arguments of this function in your training. You are allowed to add
      additional input argument if you assign it a default value that represents the plain training
      (e.g. '..., new_param=False')

    Hint: you can save your best model by deepcopy-ing it.
    """

    # Set the random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    ## Loading the dataset
    cifar10 = cifar10_utils.get_cifar10(data_dir)
    cifar10_loader = cifar10_utils.get_dataloader(cifar10, batch_size=batch_size,
                                                  return_numpy=True)

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # TODO: Initialize model and loss module
    model = MLP(n_inputs=32*32*3,
                n_hidden=hidden_dims,
                n_classes=len(cifar10['train'].dataset.classes)
                )
    loss_module = CrossEntropyModule()

    # TODO: Training loop including validation
    val_accuracies = []

    # TODO: Add any information you might want to save for plotting
    logging_dict = {
        'loss_per_batch': [],
    }
    best_model = None
    for epoch_number in range(0, epochs):
        logging_dict['loss_per_batch'].append([])
        for batch_inputs, batch_labels in tqdm(cifar10_loader['train'], desc=f"Epoch {epoch_number}"):
            out = model.forward(batch_inputs)

            # Compute loss
            loss = loss_module.forward(out, batch_labels)
            logging_dict['loss_per_batch'][epoch_number].append(loss)

            # Do backpropagation
            dout = loss_module.backward(out, batch_labels)
            model.backward(dout)

            # Update parameters
            for layer in model.layers:
                grads = getattr(layer, 'grads', None)
                params = getattr(layer, 'params', None)

                if params is not None and grads is not None:
                    for k in params.keys():
                        layer.params[k] = layer.params[k] + lr * layer.grads[k]
            # Clean up
            model.clear_cache()

        # Do validation
        acc = evaluate_model(model, cifar10_loader['validation'])
        if best_model is None or acc > np.max(val_accuracies):
            best_model = deepcopy(model)
        val_accuracies.append(acc)

    # TODO: Test best model
    test_accuracy = evaluate_model(best_model, cifar10_loader['test'])
    #######################
    # END OF YOUR CODE    #
    #######################

    return model, val_accuracies, test_accuracy, logging_dict


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--hidden_dims', default=[128], type=int, nargs='+',
                        help='Hidden dimensionalities to use inside the network. To specify multiple, use " " to separate them. Example: "256 128"')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.1, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=10, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)

    model, val_accuracies, test_accuracy, logging_dict = train(**kwargs)
    # Feel free to add any additional functions, such as plotting of the loss curve here
    x_point = np.arange(1, kwargs['epochs'] + 1)
    plt.xlabel("Epoch no.")
    plt.title("Numpy model - mean loss per epoch")
    plt.ylabel("Mean Loss")
    plt.plot(x_point, np.array(logging_dict['loss_per_batch']).mean(axis=1))
    plt.show()

    plt.xlabel("Epoch no.")
    plt.title("Numpy model - validation accuracy")
    plt.ylabel("Accuracy")
    plt.plot(x_point, val_accuracies)
    plt.show()
    print(f"Test accuracy is: {test_accuracy}")
