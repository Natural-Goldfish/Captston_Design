from src import model, dataset
import torch
"""
This is for training our model
    - SAVE_PATH                                 : This is a path to save our model's parameters
        models\{}check_point_{}.pth
        ㄴ{ } First : Keyword, Second : Epochs
    - LOAD_PATH                                 : This is a path to load our model's parameters
        models\{}check_point_{}.pth
        ㄴ{ } First : Keyword, Second : Epochs
    - EPOCHS                                    : The number of time to see the dataset
    - NUM_BATCH : Batch number                  : The number of data size to put in our model at once
    - SEQ_LEN                                   : The length of input x
    - INPUT_DIM                                 : The dimension of the input xt
    - HIDDEN_DIM                                : The dimension of the hidden state
    - NUM_LAYERS                                : The number of layers of LSTM
"""
SEQ_LEN = 96 * 3
HIDDEN_DIM = 1
INPUT_DIM = 1
EMBEDDING_DIM = 1
LEARNING_RATE = 0.01
EPOCHS = 2600

def train():
    
    lstm = model.ASPModel(seq_len = SEQ_LEN, hidden_dim = HIDDEN_DIM, input_dim = INPUT_DIM, embedding_dim = EMBEDDING_DIM)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(lstm.parameters(), lr = LEARNING_RATE)
    lstm.train()        
    
    for j in range(EPOCHS):
        for i in range(len(training_data)):
            output = lstm(training_data[i])

            loss = criterion(output, label_data[i])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss)
        
    lstm.eval()
    output = lstm(torch.tensor([6, 6, 6, 6], dtype = torch.float32))
    print(output)