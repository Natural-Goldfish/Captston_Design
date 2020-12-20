from src.model import ASPModel
from src.dataset import *
from src.utils import Normalization
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import sys
import os

_SEQUENCE_LENGTH = 96*3
_INPUT_DIM = 200
_HIDDEN_DIM = 1
_EMBEDDING_DIM = 200
_LEARNING_RATE = 0.001
_EPOCHS = 700
_BATCH_SIZE = 128
_CUDA_FLAG = torch.cuda.is_available()
_MODEL_LOAD_FLAG = False
_MODEL_PATH = "data\\models"
_MODEL_LOAD_NAME = "Temp35_{}_checkpoint.pth"

def train():
    #Load objects for training
    train_dataset = ASPDataset(mode = 'train')
    train_dataloader = DataLoader(train_dataset, batch_size = _BATCH_SIZE, shuffle = True)
    test_dataset = ASPDataset(mode = 'test')
    test_dataloader = DataLoader(test_dataset, batch_size = _BATCH_SIZE, shuffle = False)

    # Model load
    model = ASPModel(seq_len = _SEQUENCE_LENGTH, input_dim = _INPUT_DIM, hidden_dim = _HIDDEN_DIM)
    if _MODEL_LOAD_FLAG :
        model.load_state_dict(torch.load(os.path.join(_MODEL_PATH, _MODEL_LOAD_NAME)))
    if _CUDA_FLAG : model.cuda()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = _LEARNING_RATE)
    norm = Normalization()
    
    for cur_epoch in range(_EPOCHS):
        #Training
        model.train()
        train_total_loss = 0.0
        for cur_iter, train_data in enumerate(train_dataloader):
            # Data load
            train_inputs, train_labels = train_data
            if _CUDA_FLAG :
                train_inputs = train_inputs.cuda()
                train_labels = train_labels.cuda()
            
            # Update parameters
            train_outputs = model(train_inputs)
            train_labels = norm.normalize(train_labels)
            train_loss = criterion(train_outputs, train_labels)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            train_total_loss += train_loss.detach()
        print("TRAIN ::: EPOCH {}/{} Loss {:.6f}".format(cur_epoch + 1, _EPOCHS, train_total_loss/len(train_dataloader)))

        #Evalutation
        model.eval()
        with torch.no_grad():
            test_total_loss = 0.0
            for cur_iter, test_data in enumerate(test_dataloader):
                # Data Load
                test_inputs, test_labels = test_data
                if _CUDA_FLAG :
                    test_inputs = test_inputs.cuda()
                    test_labels = test_labels.cuda()
                test_outputs = model(test_inputs)
                test_total_loss += criterion(test_outputs, norm.normalize(test_labels))

            print("TEST ::: EPOCH {}/{} Loss {:.6f}".format(cur_epoch + 1, _EPOCHS, test_total_loss/len(test_dataloader)))
        if cur_epoch % 300 == 299 :
            torch.save(model.state_dict(), os.path.join(_MODEL_PATH, _MODEL_LOAD_NAME.format(cur_epoch)))
            break
        
if __name__ == "__main__":
    train()