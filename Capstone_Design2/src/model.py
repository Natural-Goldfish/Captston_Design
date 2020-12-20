import torch


_SEQUENCE_LENGTH =96*3
_NUM_EMBEDDINGS = 300
_INPUT_DIM = 200
_HIDDEN_DIM = 1
_EMBEDDING_DIM = 200

class ASPModel(torch.nn.Module):

    def __init__(self, seq_len = _SEQUENCE_LENGTH, input_dim = _INPUT_DIM, hidden_dim = _HIDDEN_DIM, num_embeddings = _NUM_EMBEDDINGS, embedding_dim = _EMBEDDING_DIM):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embedding = torch.nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.lstm = torch.nn.LSTM(input_size = self.input_dim, hidden_size = self.hidden_dim, num_layers = 2, batch_first = True)

        self.layer1 = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(in_features = self.seq_len, out_features = 2048, bias = True),
            torch.nn.ReLU(),
            torch.nn.Dropout()
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(in_features = 2048, out_features = self.seq_len),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        if type(x) != 'torch.long':
            x = x.long()
        input_vectors = self.embedding(x)
        output, _ = self.lstm(input_vectors)

        output = output.view(-1, self.seq_len)
        """
        output = self.activiation0(output)
        output = self.linear1(output)
        output = self.activiation1(output)
        output = self.dropout(output)
        output = self.linear2(output)
        output = self.activiation2(output)
        """
        output = self.layer1(output)
        output = self.layer2(output)

        return output
