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
# Date Adapted: 2021-11-11
###############################################################################

from datetime import datetime
import argparse
from tqdm.auto import tqdm

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset, text_collate_fn
from model import TextGenerationModel
import json


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


def train(args):
    """
    Trains an LSTM model on a text dataset
    
    Args:
        args: Namespace object of the command line arguments as 
              specified in the main function.
        
    TODO:
    Create the dataset.
    Create the model and optimizer (we recommend Adam as optimizer).
    Define the operations for the training loop here. 
    Call the model forward function on the inputs, 
    calculate the loss with the targets and back-propagate, 
    Also make use of gradient clipping before the gradient step.
    Recommendation: you might want to try out Tensorboard for logging your experiments.
    """
    #######################
    # PUT YOUR CODE HERE  #
    #######################
    set_seed(args.seed)
    # Load dataset
    # The data loader returns pairs of tensors (input, targets) where inputs are the
    # input characters, and targets the labels, i.e. the text shifted by one.
    dataset = TextDataset(args.txt_file, args.input_seq_length)
    data_loader = DataLoader(dataset, args.batch_size, 
                             shuffle=True, drop_last=True, pin_memory=True,
                             collate_fn=text_collate_fn)

    args.vocabulary_size = dataset.vocabulary_size
    device = args.device
    # Create model
    model = TextGenerationModel(args)
    model = model.to(device)
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Training loop
    loss_module = nn.CrossEntropyLoss()
    loss_module.to(device)

    logging_info = {
        'loss_per_batch': [],
        'training_acc': []
    }
    torch.autograd.set_detect_anomaly(True)
    for epoch_number in range(0, args.num_epochs):
        model.train()
        logging_info['loss_per_batch'].append([])
        for batch_inputs, batch_labels in tqdm(data_loader, desc=f"Epoch {epoch_number}"):
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            output = model.forward(batch_inputs)

            # Take the mean of the losses over the 30 time steps
            losses = []
            for t in range(0, output.shape[0]):
                losses.append(loss_module(output[t], batch_labels[t]))
            loss = sum(losses) / len(batch_inputs)

            logging_info['loss_per_batch'][epoch_number].append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        # get training accuracy
        model.eval()
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
            logging_info['training_acc'].append(accuracy.item() / len(labels_val))


    torch.save(model.state_dict(), "lstm-model")
    with open(f"lstm-train-logging.json", 'w') as f:
        json.dump(logging_info, f)
    #######################
    # END OF YOUR CODE    #
    #######################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # Parse training configuration

    # Model
    parser.add_argument('--txt_file', type=str, required=True, help="Path to a .txt file to train on")
    parser.add_argument('--input_seq_length', type=int, default=30, help='Length of an input sequence')
    parser.add_argument('--lstm_hidden_dim', type=int, default=1024, help='Number of hidden units in the LSTM')
    parser.add_argument('--embedding_size', type=int, default=256, help='Dimensionality of the embeddings.')

    # Training
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size to train with.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train for.')
    parser.add_argument('--clip_grad_norm', type=float, default=5.0, help='Gradient clipping norm')

    # Additional arguments. Feel free to add more arguments
    parser.add_argument('--seed', type=int, default=0, help='Seed for pseudo-random number generator')

    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else use CPU
    train(args)
