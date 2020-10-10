import torch
from torch.nn import LSTM


class ASPModel(torch.nn.Module):
    """
    This is our LSTM model. It will predict blood sugar values of the future
        It will predict ? days/months of the future.
        Which could be diferenct depends on how do you define those parameters below
        - Sequence length : ?
        - Input dimension : ?

    """
    def __init__(self, seq_len, input_dim):
        # TODO

        self.SEQ_LEN = seq_len
        self.INPUT_DIM = input_dim
        

    def forward(self, x):
        # TODO

        return 0
