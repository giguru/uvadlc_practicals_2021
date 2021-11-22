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
This file implements the execution of different hyperparameter configurations with
respect to using batch norm or not, and plots the results
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import train_mlp_pytorch

import time
import torch.nn as nn
import torch.optim as optim
# Hint: you might want to import some plotting libraries or similar
# You are also allowed to use libraries here which are not in the provided environment.
import matplotlib.pyplot as plt


def train_models(results_filename):
    """
    Executes all requested hyperparameter configurations and stores all results in a file.
    Note that we split the running of the model and the plotting, since you might want to 
    try out different plotting configurations without re-running your models every time.

    Args:
      results_filename - string which specifies the name of the file to which the results
                         should be saved.

    TODO:
    - Loop over all requested hyperparameter configurations and train the models accordingly.
    - Store the results in a file. The form of the file is left up to you (numpy, json, pickle, etc.)
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    lr = 0.1
    epochs = 20
    # TODO: Run all hyperparameter configurations as requested
    n_hiddens = [
        [128],
        [256, 128],
        [512, 256, 128],
    ]
    results = {}
    for n_hidden in n_hiddens:
        for use_batch_norm in [True, False]:
            start = time.time()
            save_key = f"hidden-{n_hidden}-batch_norm-{use_batch_norm}-lr-{lr}-epochs-{epochs}"
            print(f"Training {save_key}")
            res = train_mlp_pytorch.train(hidden_dims=n_hidden,
                                          lr=lr,
                                          epochs=epochs,
                                          use_batch_norm=use_batch_norm,
                                          batch_size=128,
                                          seed=42,
                                          data_dir='data/')
            end = time.time()
            # Skip saving the model in the results file
            results[save_key] = list(res[1:]) + [end - start]

    with open(results_filename, 'wb') as f:
        np.save(f, results)
        f.close()
    #######################
    # END OF YOUR CODE    #
    #######################


def plot_results(results_filename):
    """
    Plots the results that were exported into the given file.

    Args:
      results_filename - string which specifies the name of the file from which the results
                         are loaded.

    TODO:
    - Visualize the results in plots

    Hint: you are allowed to add additional input arguments if needed.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    with open(results_filename, 'rb') as f:
        results = np.load(f, allow_pickle=True).item()
        legend_kwargs = {
            'loc': 'upper center',
            'bbox_to_anchor': (0.5, -0.05),
            'fancybox': True,
            'shadow': True,
            'ncol': 1
        }
        plt.title("Torch model - mean loss per epoch")
        plt.xlabel("Epoch no.")
        plt.ylabel("Mean loss")
        x_point = np.arange(1, 20 + 1)
        for legend_name, data in results.items():
            plt.plot(x_point,
                     np.array(data[2]['loss_per_batch']).mean(axis=1),
                     label=legend_name)
        plt.legend(**legend_kwargs)
        plt.show()

        plt.xlabel("Epoch no.")
        plt.title("Torch model - validation accuracy")
        plt.ylabel("Accuracy")
        for legend_name, data in results.items():
            plt.plot(x_point, data[0], label=legend_name)
        plt.legend(**legend_kwargs)
        plt.show()

        plt.xlabel("Epoch no.")
        plt.title("Torch model - training accuracy")
        plt.ylabel("Accuracy")
        for legend_name, data in results.items():
            plt.plot(x_point, data[2]['training_acc'], label=legend_name)
        plt.legend(**legend_kwargs)
        plt.show()

        for legend_name, data in results.items():
            print(f"Test accuracy for {legend_name} is: {data[1]}")
    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == '__main__':
    # Feel free to change the code below as you need it.
    FILENAME = 'results.txt' 
    if not os.path.isfile(FILENAME):
        train_models(FILENAME)
    plot_results(FILENAME)