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
# Date Created: 2021-11-17
################################################################################
from __future__ import absolute_import, division, print_function

import argparse
from copy import deepcopy
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data.batch import Batch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from assignment2.part3.data import FLAT_INPUT_DIM, Z_ONE_HOT_DIM, EDGE_ATTR_DIM, get_qm9
from assignment2.part3.networks import MLP, GNN
from data import *
from networks import *
import matplotlib.pyplot as plt


def permute_indices(molecules: Batch) -> Batch:
    """permute the atoms within a molecule, but not across molecules

    Args:
        molecules: batch of molecules from pytorch geometric

    Returns:
        permuted molecules
    """
    # Permute the node indices within a molecule, but not across them.
    ranges = [
        (i, j) for i, j in zip(molecules.ptr.tolist(), molecules.ptr[1:].tolist())
    ]
    permu = torch.cat([torch.arange(i, j)[torch.randperm(j - i)] for i, j in ranges])

    n_nodes = molecules.x.size(0)
    inits = torch.arange(n_nodes)
    # For the edge_index to work, this must be an inverse permutation map.
    translation = {k: v for k, v in zip(permu.tolist(), inits.tolist())}

    permuted = deepcopy(molecules)
    permuted.x = permuted.x[permu]
    # Below is the identity transform, by construction of our permutation.
    permuted.batch = permuted.batch[permu]
    permuted.edge_index = (
        permuted.edge_index.cpu()
        .apply_(translation.get)
        .to(molecules.edge_index.device)
    )
    return permuted


def compute_loss(
    model: nn.Module, molecules: Batch, criterion: Callable
) -> torch.Tensor:
    """use the model to predict the target determined by molecules. loss computed by criterion.

    Args:
        model: trainable network
        molecules: batch of molecules from pytorch geometric
        criterion: callable which takes a prediction and the ground truth 

    Returns:
        loss

    TODO: 
    - conditionally compute loss based on model type
    - make sure there are no warnings / errors
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    labels = get_labels(molecules)
    if isinstance(model, MLP):
        output = model(x=get_mlp_features(molecules))
        output = output.view(-1)
        loss = criterion(output, labels)
    elif isinstance(model, GNN):
        output = model(x=get_node_features(molecules),
                       edge_index=molecules.edge_index,
                       edge_attr=molecules.edge_attr,
                       batch_idx=molecules.batch)
        output = output.view(-1)
        loss = criterion(output, labels)
    #######################
    # END OF YOUR CODE    #
    #######################
    return loss


def evaluate_model(
    model: nn.Module, data_loader: DataLoader, criterion: Callable, permute: bool
) -> float:
    """
    Performs the evaluation of the model on a given dataset.

    Args:
        model: trainable network
        data_loader: The data loader of the dataset to evaluate.
        criterion: loss module, i.e. torch.nn.MSELoss()
        permute: whether to permute the atoms within a molecule
    Returns:
        avg_loss: scalar float, the average loss of the model on the dataset.

    Hint: make sure to return the average loss of the whole dataset, 
          independent of batch sizes (not all batches might be the same size).
    
    TODO: conditionally permute indices
          calculate loss
          average loss independent of batch sizes
          make sure the model is in the correct mode
    """

    #######################
    # PUT YOUR CODE HERE  #
    #######################
    losses = []
    total_items = 0
    model.eval()
    with torch.no_grad():
        for molecules in tqdm(data_loader, f"Evaluate with permute={permute}"):
            loss = compute_loss(model=model,
                                criterion=criterion,
                                molecules=permute_indices(molecules) if permute else molecules)
            n_items = len(molecules.y)
            total_items += n_items
            losses.append(loss.item() * n_items)

    avg_loss = sum(losses) / total_items
    #######################
    # END OF YOUR CODE    #
    #######################

    return avg_loss


def train(
    model: nn.Module, lr: float, batch_size: int, epochs: int, seed: int, data_dir: str
):
    """a full training cycle of an mlp / gnn on qm9.

    Args:
        model: a differentiable pytorch module which estimates the U0 quantity
        lr: learning rate of optimizer
        batch_size: batch size of molecules
        epochs: number of epochs to optimize over
        seed: random seed
        data_dir: where to place the qm9 data

    Returns:
        model: the trained model which performed best on the validation set
        test_loss: the loss over the test set
        permuted_test_loss: the loss over the test set where atomic indices have been permuted
        val_losses: the losses over the validation set at every epoch
        logging_info: general object with information for making plots or whatever you'd like to do with it

    TODO:
    - Implement the training of both the mlp and the gnn in the same function
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
    if torch.cuda.is_available():  # GPU operation have separate seed
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False

    # Loading the dataset
    train, valid, test = get_qm9(data_dir, model.device)
    train_dataloader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        exclude_keys=["pos", "idx", "z", "name"],
    )
    valid_dataloader = DataLoader(
        valid, batch_size=batch_size, exclude_keys=["pos", "idx", "z", "name"]
    )
    test_dataloader = DataLoader(
        test, batch_size=batch_size, exclude_keys=["pos", "idx", "z", "name"]
    )

    #######################
    # PUT YOUR CODE HERE  #
    #######################

    # TODO: Initialize loss module and optimizer
    # TODO: Do optimization, we used adam with amsgrad. (many should work)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, amsgrad=True)

    logging_info = {
        'loss_per_batch': [],
    }
    val_losses = []
    best_model = None

    # TODO: Training loop including validation, using evaluate_model
    for epoch_number in range(0, epochs):
        model.train()
        logging_info['loss_per_batch'].append([])
        for data_batch in tqdm(train_dataloader, desc=f"Epoch {epoch_number}"):
            data_batch = data_batch.to(model.device)
            loss = compute_loss(model=model, criterion=criterion, molecules=data_batch)
            loss = loss.mean()
            logging_info['loss_per_batch'][epoch_number].append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Do validation
        val_loss = evaluate_model(model, valid_dataloader, criterion=criterion, permute=False)
        if best_model is None or val_loss > np.max(val_losses):
            # Store best model on CPU to save space on CUDA if CUDA is used
            best_model = deepcopy(model).to('cpu')
        val_losses.append(val_loss)

    # TODO: Test best model
    test_loss = evaluate_model(best_model.to(model.device), test_dataloader, criterion=criterion, permute=False)

    # TODO: Test best model against permuted indices
    permuted_test_loss = evaluate_model(best_model.to(model.device), test_dataloader, criterion=criterion, permute=True)

    #######################
    # END OF YOUR CODE    #
    #######################
    return model, test_loss, permuted_test_loss, val_losses, logging_info


def main(**kwargs):
    """main handles the arguments, instantiates the correct model, tracks the results, and saves them."""
    which_model = kwargs.pop("model")
    mlp_hidden_dims = kwargs.pop("mlp_hidden_dims")
    gnn_hidden_dims = kwargs.pop("gnn_hidden_dims")
    gnn_num_blocks = kwargs.pop("gnn_num_blocks")

    # Set default device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if which_model == "mlp":
        model = MLP(FLAT_INPUT_DIM, mlp_hidden_dims, 1)
    elif which_model == "gnn":
        model = GNN(
            n_node_features=Z_ONE_HOT_DIM,
            n_edge_features=EDGE_ATTR_DIM,
            n_hidden=gnn_hidden_dims,
            n_output=1,
            num_convolution_blocks=gnn_num_blocks,
        )
    else:
        raise NotImplementedError("only mlp and gnn are possible models.")

    model.to(device)
    model, test_loss, permuted_test_loss, val_losses, logging_info = train(
        model, **kwargs
    )
    print("Test loss", test_loss, "Permutated test loss", permuted_test_loss)

    # plot the loss curve, etc. below.
    loss_per_batch = logging_info['loss_per_batch']

    x_point = np.arange(1, len(loss_per_batch) + 1)
    plt.xlabel("Epoch number")
    plt.ylabel("Loss")

    plt.plot(x_point, np.mean(loss_per_batch, axis=1))
    plt.xticks(x_point)

    plt.title(f"{which_model} training loss")
    plt.show()

    x_point = np.arange(1, len(val_losses) + 1)
    plt.xlabel("Epoch number")
    plt.ylabel("Loss")

    plt.plot(x_point, val_losses)
    plt.xticks(x_point)

    plt.title(f"{which_model} validation loss")
    plt.show()


if __name__ == "__main__":
    # Command line arguments
    parser = argparse.ArgumentParser()

    # Model hyperparameters
    parser.add_argument(
        "--model",
        default="mlp",
        type=str,
        choices=["mlp", "gnn"],
        help="Select between training an mlp or a gnn.",
    )
    parser.add_argument(
        "--mlp_hidden_dims",
        default=[128, 128, 128, 128],
        type=int,
        nargs="+",
        help='Hidden dimensionalities to use inside the mlp. To specify multiple, use " " to separate them. Example: "256 128"',
    )
    parser.add_argument(
        "--gnn_hidden_dims",
        default=64,
        type=int,
        help="Hidden dimensionalities to use inside the mlp. The same number of hidden features are used at every layer.",
    )
    parser.add_argument(
        "--gnn_num_blocks",
        default=2,
        type=int,
        help="Number of blocks of GNN convolutions. A block may include multiple different kinds of convolutions (see GNN comments)!",
    )

    # Optimizer hyperparameters
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate to use")
    parser.add_argument("--batch_size", default=128, type=int, help="Minibatch size")

    # Other hyperparameters
    parser.add_argument("--epochs", default=10, type=int, help="Max number of epochs")

    # Technical
    parser.add_argument(
        "--seed", default=42, type=int, help="Seed to use for reproducing results"
    )
    parser.add_argument(
        "--data_dir",
        default="data/",
        type=str,
        help="Data directory where to store/find the qm9 dataset.",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    main(**kwargs)
