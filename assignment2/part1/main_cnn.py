###############################################################################
# MIT License
#
# Copyright (c) 2021
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2021
# Date Created: 2021-11-11
###############################################################################
"""
Main file for Question 1.2 of the assignment. You are allowed to add additional
imports if you want.
"""
import os
import json
import argparse
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

import torchvision
import torchvision.models as models
from torchvision.datasets import CIFAR10
from torchvision import transforms
from tqdm import tqdm
from augmentations import gaussian_noise_transform, gaussian_blur_transform, contrast_transform, jpeg_transform
from cifar10_utils import get_train_validation_set, get_test_set
from copy import deepcopy


logger = logging.getLogger(__name__)


def set_seed(seed):
    """
    Function for setting the seed for reproducibility.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.determinstic = True
    torch.backends.cudnn.benchmark = False


def get_model(model_name, num_classes=10):
    """
    Returns the model architecture for the provided model_name. 

    Args:
        model_name: Name of the model architecture to be returned. 
                    Options: ['debug', 'vgg11', 'vgg11_bn', 'resnet18', 
                              'resnet34', 'densenet121']
                    All models except debug are taking from the torchvision library.
        num_classes: Number of classes for the final layer (for CIFAR10 by default 10)
    Returns:
        cnn_model: nn.Module object representing the model architecture.
    """
    if model_name == 'debug':  # Use this model for debugging
        cnn_model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(32*32*3, num_classes)
            )
    elif model_name == 'vgg11':
        cnn_model = models.vgg11(num_classes=num_classes)
    elif model_name == 'vgg11_bn':
            cnn_model = models.vgg11_bn(num_classes=num_classes)
    elif model_name == 'resnet18':
        cnn_model = models.resnet18(num_classes=num_classes)
    elif model_name == 'resnet34':
        cnn_model = models.resnet34(num_classes=num_classes)
    elif model_name == 'densenet121':
        cnn_model = models.densenet121(num_classes=num_classes)
    else:
        assert False, f'Unknown network architecture \"{model_name}\"'
    return cnn_model


def train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device):
    """
    Trains a given model architecture for the specified hyperparameters.

    Args:
        model: Model architecture to train.
        lr: Learning rate to use in the optimizer.
        batch_size: Batch size to train the model with.
        epochs: Number of epochs to train the model for.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        checkpoint_name: Filename to save the best model on validation to.
        device: Device to use for training.
    Returns:
        model: Model that has performed best on the validation set.

    TODO:
    Implement the training of the model with the specified hyperparameters
    Save the best model to disk so you can load it later.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    model = model.to(device)

    # Load the datasets
    train_dataset, val_dataset = get_train_validation_set(data_dir=data_dir)
    train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validation_dataloader = data.DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Initialize the optimizers and learning rate scheduler. 
    # We provide a recommend setup, which you are allowed to change if interested.
    optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 135], gamma=0.1)

    # Training loop with validation after each epoch. and remember to use the lr scheduler.
    loss_module = nn.CrossEntropyLoss()
    loss_module.to(device)

    logging_info = {
        'loss_per_batch': [],
        'training_acc': []
    }
    best_model = None
    val_accuracies = []
    logger.info('Start training...')
    for epoch_number in range(0, epochs):
        model.train()
        model = model.to(device)
        logging_info['loss_per_batch'].append([])
        for batch_inputs, batch_labels in tqdm(train_dataloader, desc=f"Epoch {epoch_number}"):
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            output = model.forward(batch_inputs)
            loss = loss_module(output, batch_labels)
            logging_info['loss_per_batch'][epoch_number].append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Optionally compute accuracy on training set
        # logging_info['training_acc'].append(evaluate_model(model, train_dataloader, device))
        # Do validation
        acc = evaluate_model(model, validation_dataloader, device)
        if best_model is None or acc > np.max(val_accuracies):
            logger.info(f"New best model at epoch {epoch_number} with validation acc {acc}")
            best_model = deepcopy(model.to('cpu'))
        val_accuracies.append(acc)

    # Save the best model
    torch.save(best_model.state_dict(), checkpoint_name)
    logging_info['validation_acc'] = val_accuracies
    with open(f"{checkpoint_name}-train-logging.json", 'w') as f:
        json.dump(logging_info, f)

    # Load best model and return it.
    model = best_model

    #######################
    # END OF YOUR CODE    #
    #######################
    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluates a trained model on a given dataset.

    Args:
        model: Model architecture to evaluate.
        data_loader: The data loader of the dataset to evaluate on.
        device: Device to use for training.
    Returns:
        accuracy: The accuracy on the dataset.

    TODO:
    Implement the evaluation of the model on the dataset.
    Remember to set the model in evaluation mode and back to training mode in the training loop.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    model.eval()
    model.to(device)
    with torch.no_grad():
        preds_val, labels_val = None, torch.tensor([]).to(device)
        for batch_inputs, batch_labels in tqdm(data_loader):
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            out = model.forward(batch_inputs)
            preds_val = torch.vstack((preds_val, out)) if preds_val is not None else out
            labels_val = torch.concat((labels_val, batch_labels))

        # Use 'sum()' to count the number of True values in the resulting array
        accuracy = (torch.argmax(preds_val, dim=1) == labels_val).sum()
        accuracy = accuracy.item() / len(labels_val)

    #######################
    # END OF YOUR CODE    #
    #######################
    return accuracy


def test_model(model, batch_size, data_dir, device, seed):
    """
    Tests a trained model on the test set with all corruption functions.

    Args:
        model: Model architecture to test.
        batch_size: Batch size to use in the test.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        device: Device to use for training.
        seed: The seed to set before testing to ensure a reproducible test.
    Returns:
        test_results: Dictionary containing an overview of the accuracies achieved on the different
                      corruption functions and the plain test set.

    TODO:
    Evaluate the model on the plain test set. Make use of the evaluate_model function.
    For each corruption function and severity, repeat the test. 
    Summarize the results in a dictionary (the structure inside the dict is up to you.)
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    set_seed(seed)
    test_results = {}
    corruptions = {
        'gaussian_noise': gaussian_noise_transform,
        'gaussian_blur': gaussian_blur_transform,
        'contrast': contrast_transform,
        'jpeg': jpeg_transform
    }
    data_loader_args = {'batch_size': batch_size, 'shuffle': False, 'drop_last': False}

    # First test without corruptions
    test_results['none'] = evaluate_model(model, data.DataLoader(dataset=get_test_set(data_dir), **data_loader_args), device)

    # Then test with corruptions over different severities
    for k, c_func in corruptions.items():
        for severity in [1, 2, 3, 4, 5]:
            augmentation = c_func(severity) if c_func else None
            test_dataloader = data.DataLoader(dataset=get_test_set(data_dir, augmentation),
                                              **data_loader_args)
            test_results[f"{k}-{severity}"] = evaluate_model(model, test_dataloader, device)
    #######################
    # END OF YOUR CODE    #
    #######################
    return test_results


def main(model_name, lr, batch_size, epochs, data_dir, seed):
    """
    Function that summarizes the training and testing of a model.

    Args:
        model: Model architecture to test.
        batch_size: Batch size to use in the test.
        data_dir: Directory where the CIFAR10 dataset should be loaded from or downloaded to.
        device: Device to use for training.
        seed: The seed to set before testing to ensure a reproducible test.
    Returns:
        test_results: Dictionary containing an overview of the accuracies achieved on the different
                      corruption functions and the plain test set.

    TODO:
    Load model according to the model name.
    Train the model (recommendation: check if you already have a saved model. If so, skip training and load it)
    Test the model using the test_model function.
    Save the results to disk.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    set_seed(seed)
    checkpoint_name = f'saved-{model_name}'
    has_checkpoint = os.path.exists(checkpoint_name)
    model = get_model(model_name).to(device)
    if has_checkpoint:
        model.load_state_dict(torch.load(checkpoint_name, map_location=device))
    else:
        model = train_model(model, lr, batch_size, epochs, data_dir, checkpoint_name, device)

    test_results = test_model(model=model, batch_size=batch_size, data_dir=data_dir, device=device, seed=seed)
    print(test_results)

    #######################
    # END OF YOUR CODE    #
    #######################





if __name__ == '__main__':
    """
    The given hyperparameters below should give good results for all models.
    However, you are allowed to change the hyperparameters if you want.
    Further, feel free to add any additional functions you might need, e.g. one for calculating the RCE and CE metrics.
    """
    # Command line arguments
    parser = argparse.ArgumentParser()
    
    # Model hyperparameters
    parser.add_argument('--model_name', default='debug', type=str,
                        help='Name of the model to train.')
    
    # Optimizer hyperparameters
    parser.add_argument('--lr', default=0.01, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=128, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--epochs', default=150, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--data_dir', default='data/', type=str,
                        help='Data directory where to store/find the CIFAR10 dataset.')

    args = parser.parse_args()
    kwargs = vars(args)
    print(kwargs)
    main(**kwargs)