import torch


training_data = [
                torch.tensor([0, 1, 2, 3], dtype = torch.long, requires_grad =False),\
                torch.tensor([1, 2, 3, 4], dtype = torch.long, requires_grad =False),\
                torch.tensor([2, 3, 4, 5], dtype = torch.long, requires_grad =False),\
                torch.tensor([3, 4, 5, 6], dtype = torch.long, requires_grad =False)\
                ]
target = [
            # 0, 1, 2, 3, 4, 5, 6, 7
            torch.tensor([0, 0, 0, 0, 0, 0, 1], dtype = torch.float32, requires_grad = False),\
            torch.tensor([0, 0, 0, 0, 0, 1, 0], dtype = torch.float32, requires_grad = False),\
            torch.tensor([0, 0, 0, 0, 1, 0, 0], dtype = torch.float32, requires_grad = False),\
            torch.tensor([0, 0, 0, 1, 0, 0, 0], dtype = torch.float32, requires_grad = False),\
            torch.tensor([0, 0, 1, 0, 0, 0, 0], dtype = torch.float32, requires_grad = False),\
            torch.tensor([0, 1, 0, 0, 0, 0, 0], dtype = torch.float32, requires_grad = False),\
            torch.tensor([1, 0, 0, 0, 0, 0, 0], dtype = torch.float32, requires_grad = False)\
        ]



label_data = [] 
label_data.append(torch.cat((target[1], target[2], target[3], target[4])).view(4, 1, 7).contiguous())
label_data.append(torch.cat((target[2], target[3], target[4], target[5])).view(4, 1, 7).contiguous())
label_data.append(torch.cat((target[3], target[4], target[5], target[6])).view(4, 1, 7).contiguous())
label_data.append(torch.cat((target[4], target[5], target[6], target[0])).view(4, 1, 7).contiguous())

SEQ_LEN = 4
HIDDEN_DIM = 7
EMBEDDING_DIM = 7
INPUT_DIM = 7

class custom_LSTM(torch.nn.Module):
    def __init__(self, seq_len = SEQ_LEN, hidden_dim = HIDDEN_DIM, input_dim = INPUT_DIM, embedding_dim = EMBEDDING_DIM):
        super().__init__()
        
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.embedding = torch.nn.Embedding(self.input_dim, self.embedding_dim)
        self.lstm = torch.nn.LSTM(input_size = self.input_dim, hidden_size = self.hidden_dim)
        self.linear = torch.nn.Linear(in_features = self.hidden_dim, out_features = self.input_dim)

    def forward(self, x):
        input_vectors = self.embedding(x)
        input_vectors = input_vectors.view(len(x), -1, self.input_dim).contiguous()
        output, _ = self.lstm(input_vectors)
        output = self.linear(output)
        return output

if __name__ == "__main__":
    
    lstm = custom_LSTM(seq_len = SEQ_LEN, hidden_dim = HIDDEN_DIM, input_dim = INPUT_DIM, embedding_dim = EMBEDDING_DIM)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(lstm.parameters(), lr = 0.01)
    lstm.train()
    for j in range(2600):
        
        for i in range(len(training_data)):
            output = lstm(training_data[i])
            loss = criterion(output, label_data[i])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)
    
    lstm.eval()
    output = lstm(torch.tensor([0, 4, 3, 2], dtype = torch.long))
    print(output)