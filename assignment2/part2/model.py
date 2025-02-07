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

import math
import random
import torch
import torch.nn as nn


class LSTM(nn.Module):
    """
    Own implementation of LSTM cell.
    """

    def __init__(self, lstm_hidden_dim, embedding_size):
        """
        Initialize all parameters of the LSTM class.

        Args:
            lstm_hidden_dim: hidden state dimension.
            embedding_size: size of embedding (and hence input sequence).

        TODO:
        Define all necessary parameters in the init function as properties of the LSTM class.
        """
        super(LSTM, self).__init__()
        self.hidden_dim = lstm_hidden_dim
        self.embed_dim = embedding_size
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        # gate weights
        tensor_for_x = lambda: torch.zeros(size=(self.embed_dim, self.hidden_dim), requires_grad=True)
        tensor_for_h = lambda: torch.zeros(size=(self.hidden_dim, self.hidden_dim), requires_grad=True)
        tensor_for_bias = lambda: torch.zeros(self.hidden_dim, requires_grad=True)

        self.w_gx = nn.Parameter(tensor_for_x())
        self.w_gh = nn.Parameter(tensor_for_h())
        self.bias_g = nn.Parameter(tensor_for_bias())
        # input weights
        self.w_ix = nn.Parameter(tensor_for_x())
        self.w_ih = nn.Parameter(tensor_for_h())
        self.bias_i = nn.Parameter(tensor_for_bias())
        # forget weights
        self.w_fx = nn.Parameter(tensor_for_x())
        self.w_fh = nn.Parameter(tensor_for_h())
        self.bias_f = nn.Parameter(tensor_for_bias())
        # output weights
        self.w_ox = nn.Parameter(tensor_for_x())
        self.w_oh = nn.Parameter(tensor_for_h())
        self.bias_o = nn.Parameter(tensor_for_bias())

        #######################
        # END OF YOUR CODE    #
        #######################
        self.init_parameters()

    def init_parameters(self):
        """
        Parameters initialization.

        Args:
            self.parameters(): list of all parameters.
            self.hidden_dim: hidden state dimension.

        TODO:
        Initialize all your above-defined parameters,
        with a uniform distribution with desired bounds (see exercise sheet).
        Also, add one (1.) to the uniformly initialized forget gate-bias.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        a, b = -1 / math.sqrt(self.hidden_dim), 1 / math.sqrt(self.hidden_dim)
        for name, params in self.named_parameters():
            # continue
            if name == 'bias_f':
                # The reason for this is that for learning long-term dependencies, it is good practice to initialize the
                # bias of the forget gate to a larger value, such that the model starts off with remembering old states
                # and learns what to forget (rather than vice versa).
                nn.init.uniform_(self.bias_f.data, a + 1, b + 1)
            else:
                nn.init.uniform_(params.data, a, b)
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, embeds):
        """
        Forward pass of LSTM.

        Args:
            embeds: embedded input sequence with shape [input length, batch size, embedding size].

        TODO:
          Specify the LSTM calculations on the input sequence.
        Hint:
        The output needs to span all time steps, (not just the last one),
        so the output shape is [input length, batch size, hidden dimension].
        """
        #
        #
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        I, B, H = embeds.shape
        self.h = [None] * I
        self.c = [None] * I

        h0 = torch.zeros((B, self.hidden_dim)).to(embeds.device)
        c0 = torch.zeros((B, self.hidden_dim)).to(embeds.device)

        for i in range(I):  # loop over input length
            x = embeds[i]
            prev_h = self.h[i-1] if i > 0 else h0
            prev_c = self.c[i-1] if i > 0 else c0
            g = torch.tanh(torch.matmul(x, self.w_gx) + torch.matmul(prev_h, self.w_gh) + self.bias_g)
            i_2 = torch.sigmoid(torch.matmul(x, self.w_ix) + torch.matmul(prev_h, self.w_ih) + self.bias_i)
            f = torch.sigmoid(torch.matmul(x, self.w_fx) + torch.matmul(prev_h, self.w_fh) + self.bias_f)
            o = torch.sigmoid(torch.matmul(x, self.w_ox) + torch.matmul(prev_h, self.w_oh) + self.bias_o)
            self.c[i] = g * i_2 + prev_c * f
            self.h[i] = torch.tanh(self.c[i]) * o

        return torch.cat(self.h).view(I, B, self.hidden_dim).to(embeds.device)
        #######################
        # END OF YOUR CODE    #
        #######################


class TextGenerationModel(nn.Module):
    """
    This module uses your implemented LSTM cell for text modelling.
    It should take care of the character embedding,
    and linearly maps the output of the LSTM to your vocabulary.
    """

    def __init__(self, args):
        """
        Initializing the components of the TextGenerationModel.

        Args:
            args.vocabulary_size: The size of the vocabulary.
            args.embedding_size: The size of the embedding.
            args.lstm_hidden_dim: The dimension of the hidden state in the LSTM cell.

        TODO:
        Define the components of the TextGenerationModel,
        namely the embedding, the LSTM cell and the linear classifier.
        """
        super(TextGenerationModel, self).__init__()
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.args = args
        self.embedding = nn.Embedding(args.vocabulary_size, args.embedding_size)
        self.lstm = LSTM(lstm_hidden_dim=args.lstm_hidden_dim, embedding_size=args.embedding_size)

        # Used the nn.Linear here. In the assignment, it seems like you only cannot used it for the LSTM component
        self.linear = nn.Linear(in_features=args.lstm_hidden_dim, out_features=args.vocabulary_size)

        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: input

        TODO:
        Embed the input,
        apply the LSTM cell
        and linearly map to vocabulary size.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        one = self.embedding(x)
        two = self.lstm(one)
        three = self.linear(two)
        return three
        #######################
        # END OF YOUR CODE    #
        #######################

    def sample(self, batch_size=5, sample_length=30, temperature=0.):
        """
        Sampling from the text generation model.

        Args:
            batch_size: Number of samples to return
            sample_length: length of desired sample.
            temperature: temperature of the sampling process (see exercise sheet for definition).

        TODO:
        Generate sentences by sampling from the model, starting with a random character.
        If the temperature is 0, the function should default to argmax sampling,
        else to softmax sampling with specified temperature.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        from dataset import TextDataset
        dataset = TextDataset(self.args.txt_file, self.args.input_seq_length)

        with torch.no_grad():
            sentences = []
            for b in range(batch_size):
                s = random.choice(range(dataset.vocabulary_size))
                chars = [[s]]
                for i in range(sample_length - 1):
                    tens = torch.LongTensor(chars).to(self.args.device)

                    res = self.forward(tens)[i][0]
                    if temperature == 0:
                        new_char_idx = torch.argmax(res).item()
                    else:
                        new_char_idx = torch.argmax((res / temperature).softmax(dim=0)).item()
                    chars.append([new_char_idx])
                sentences.append(dataset.convert_to_string([c[0] for c in chars]))

        return sentences
        #######################
        # END OF YOUR CODE    #
        #######################
