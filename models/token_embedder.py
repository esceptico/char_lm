from typing import Callable

import torch.nn as nn
import torch

from .highway import Highway


class Conv1dMaxPooling(nn.Module):
    """Conv1d with max-over-time pooling."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        activation: Callable[[torch.Tensor], torch.Tensor] = torch.relu
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size
        )
        self.activation = activation

    def forward(self, inputs):
        convolved = self.conv(inputs)
        # max over time pooling
        convolved, _ = torch.max(convolved, dim=-1)
        convolved = self.activation(convolved)
        return convolved


class TokenEmbedder(nn.Module):
    """
    Compute context insensitive token representation

    This embedder has input character ids of size
    (batch_size, sequence_length, vocab_size)
    and returns (batch_size, sequence_length, projection_size).

    Parameters
    ----------
    char_embedding_size : ``int``, (default = 16).
        Size of each char embedding vector.
    vocab_size : ``int``, (default = 264).
        Size of character vocabulary.
    highway_layers : ``int``, (default = 1).
        The number of highway layers.
    filters : ``list``, (default = [[1, 4], [2, 8], [3, 16], [4, 32], [5, 64], [6, 128]]).
        pass
    projection_size : ``int``, (default = 256).
        pass

    """
    def __init__(
        self,
        char_embedding_size: int = 16,
        vocab_size: int = 264,
        highway_layers: int = 1,
        filters: list = [[1, 4], [2, 8], [3, 16], [4, 32], [5, 64], [6, 128]],
        projection_size: int = 256
    ):
        super().__init__()
        self.char_embedding = nn.Embedding(vocab_size, char_embedding_size)
        self.convolutions = nn.ModuleList([
            Conv1dMaxPooling(in_channels=char_embedding_size,
                             out_channels=num,
                             kernel_size=width)
            for width, num in filters
        ])

        highway_input_size = sum(
            num_filters for _, num_filters in filters
        )
        self.highway = Highway(highway_input_size, highway_layers)
        self.projection = nn.Linear(highway_input_size, projection_size)

    def forward(self, inputs):
        batch_size, seq_len, token_len = inputs.size()

        # (batch_size * seq_len, token_len)
        inputs = inputs.view(-1, token_len)

        # (batch_size * seq_len, token_len, embedding_size)
        char_embedding = self.char_embedding(inputs)

        # (batch_size * seq_len, embedding_size, token_len)
        char_embedding = char_embedding.permute(0, 2, 1)

        # (batch_size * sequence_length, n_filters) for each convolution
        convs = [conv(char_embedding) for conv in self.convolutions]

        # (batch_size * sequence_length, n_filters)
        token_embedding = torch.cat(convs, dim=-1)
        token_embedding = self.highway(token_embedding)

        # (batch_size * sequence_length, projection_size)
        token_embedding = self.projection(token_embedding)

        # (batch_size, sequence_length, projection_size)
        return token_embedding.view(batch_size, seq_len, -1)
