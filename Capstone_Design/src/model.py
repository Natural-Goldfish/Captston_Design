import torch
from torch.nn import LSTM, Linear
from src.utils import Embedding

class ASPModel(torch.nn.Module):
    """
    This is our LSTM model. It will predict blood sugar values of the future
        It will predict ? days/months of the future.
        Which could be diferenct depends on how do you define those parameters below
        - Sequence length : ?
        - Input dimension : ?

    """

    def __init__(self, seq_len = 96*3, input_dim = 1, hidden_dim = 1):
        self.SEQ_LEN = seq_len
        self.INPUT_DIM = input_dim
        self.HIDDEN_DIM = hidden_dim

        self.embedding = Embedding(x_min= 0, x_max= 250, batch_num= -1, input_dim= 1)
        self.lstm = LSTM(input_size = self.INPUT_DIM, hidden_size = self.HIDDEN_DIM)
        self.linear = Linear(in_features= self.HIDDEN_DIM, out_features= self.INPUT_DIM)


    def forward(self, x):
        input_vectors = self.embedding(x)
        input_vectors = input_vectors.view(len(x), -1, self.input_dim).contiguous()
        output, _ = self.lstm(input_vectors)
        output = self.linear(output)
        return output

