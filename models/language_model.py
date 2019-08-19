import torch
import torch.nn as nn

from .token_embedder import TokenEmbedder


# TODO: residual/highway lstm connections
class LanguageModel(nn.Module):
    def __init__(
        self,
        char_vocab_size: int,
        word_vocab_size: int,
        char_embedding_size: int = 16,
        embedding_size: int = 256,
        hidden_size: int = 512,
        dropout: float = 0.1,
        lstm_layers: int = 1,
        highway_layers: int = 1,
        filters: list = [[1, 4], [2, 8], [3, 16], [4, 32], [5, 64], [6, 128]]
    ):
        super().__init__()
        lstm_dropout = 0 if lstm_layers == 1 else dropout
        self.lstm_layers = lstm_layers
        self.hidden_size = hidden_size

        self.token_embedder = TokenEmbedder(
            char_embedding_size=char_embedding_size,
            vocab_size=char_vocab_size,
            filters=filters,
            projection_size=hidden_size,
            highway_layers=highway_layers
        )
        self.dropout = nn.Dropout(dropout)
        self.bilstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            bidirectional=True,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
            batch_first=True
        )
        self.projection = nn.Linear(hidden_size * 2, word_vocab_size)

    def forward(self, inputs, hidden):
        token_embedding = self.token_embedder(inputs)
        output, hidden = self.bilstm(token_embedding, hidden)
        output = self.dropout(output)
        projected = self.projection(
            output.view(output.size(0) * output.size(1), output.size(2))
        )
        return (
            projected.view(output.size(0), output.size(1), projected.size(1)),
            hidden
        )

    def init_hidden(self, batch_size):
        zeros = torch.zeros(self.lstm_layers * 2,
                            batch_size, self.hidden_size)
        return zeros, zeros
