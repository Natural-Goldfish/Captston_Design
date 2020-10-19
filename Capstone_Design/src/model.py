import torch

_SEQUENCE_LENGTH = 96*3
_NUM_EMBEDDINGS= 300
_EMBEDDING_DIM = 25
_INPUT_DIM = 25
_HIDDEN_DIM = 25

class ASPModel(torch.nn.Module):
    """
    This is our LSTM model. It will predict glucose values of the future
        It will predict 3 days/months of the future.
        Which could be diferenct depends on how do you define those parameters below
        - Sequence length : 96*3
        - Input dimension : 5 (Experimental)

    """
    def __init__(self, seq_len = _SEQUENCE_LENGTH, input_dim = _INPUT_DIM, hidden_dim = _HIDDEN_DIM, num_embeddings = _NUM_EMBEDDINGS, embedding_dim = _EMBEDDING_DIM):
        super().__init__()
        # Attributes
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        # Model
        self.embedding = torch.nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.lstm = torch.nn.LSTM(input_size = self.input_dim, hidden_size = self.hidden_dim)
        self.linear = torch.nn.Linear(in_features = self.hidden_dim, out_features = 1)

    def forward(self, x):
        if type(x) != 'torch.long':
            x = x.long()
        input_vectors = self.embedding(x)
        input_vectors = input_vectors.view(len(x), -1, self.input_dim).contiguous()
        output, _ = self.lstm(input_vectors)
        output = self.linear(output)
        return output

