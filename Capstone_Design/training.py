from src.model import ASPModel
from src.dataset import *
from src.utils import Normalization
from torch.utils.data import DataLoader
import torch

_SEQUENCE_LENGTH = 96*3
_INPUT_DIM = 25
_HIDDEN_DIM = 25
_EMBEDDING_DIM = 300
_LEARNING_RATE = 0.001
_EPOCHS = 2600
_BATCH_SIZE = 128
_CUDA_FLAG = torch.cuda.is_available()

_MODEL_LOAD_FLAG = False
_MODEL_PATH = "data\\models"
_MODEL_LOAD_NAME = "ASPModel_{}_checkpoint.pth".format("temp")

def train():
    train_dataset = ASPDataset(mode = "train")
    train_dataloader = DataLoader(train_dataset, batch_size = _BATCH_SIZE, shuffle = False)
    
    val_dataset = ASPDataset(mode = "val")
    val_dataloader = DataLoader(val_dataset, batch_size = _BATCH_SIZE, shuffle = False)
    
    norm = Normalization()
    # Model load
    model = ASPModel(seq_len = _SEQUENCE_LENGTH, input_dim = _INPUT_DIM, hidden_dim = _HIDDEN_DIM)
    if _MODEL_LOAD_FLAG :
        model.load_state_dict(torch.load(os.path.join(_MODEL_PATH, _MODEL_LOAD_NAME)))

    # Use GPU, if it is available
    if _CUDA_FLAG : model.cuda()

    # Loss function and Optimizer (Experimental) -> Optimizer : Adam
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = _LEARNING_RATE)

    for cur_epoch in range(_EPOCHS):
        # Training
        model.train()
        optimizer.zero_grad()
        for cur_iter, train_data in enumerate(train_dataloader):
            # Data load
            train_inputs, train_labels = train_data
            if _CUDA_FLAG :
                train_inputs = train_inputs.cuda()
                train_labels = train_labels.cuda()
            _, temp_length = train_inputs.shape

            # Update parameters
            train_outputs = model(train_inputs).view(-1, temp_length)
            train_labels = norm.normalize(train_labels)       # Experimental
            train_loss = criterion(train_outputs, train_labels)
            train_loss.backward()
            optimizer.step()
            print("TRAIN ::: EPOCH {}/{} Iteration {}/{} Loss {:.6f}".format(cur_epoch+1, _EPOCHS, cur_iter, len(train_dataloader), train_loss))

        # Evaludation
        model.eval()
        with torch.no_grad() :
            if cur_epoch % 300 == 0 :
                print("stop!")
            val_loss = 0.0
            for cur_iter, val_data in enumerate(val_dataloader):
                # Data load
                val_inputs, val_labels = val_data
                if _CUDA_FLAG :
                    val_inputs = val_inputs.cuda()
                    val_labels = val_labels.cuda()
                _, temp_length = val_inputs.shape
                val_outputs = model(val_inputs).view(-1, temp_length)
                val_labels = norm.normalize(val_labels)       # Experimental
                test_output = norm.de_normalize(val_outputs)
                test_max = torch.max(test_output)
                test_min = torch.min(test_output)
                val_loss += criterion(val_outputs, val_labels)
            print("VAL ::: EPOCH {}/{} Loss {:.6f}".format(cur_epoch+1, _EPOCHS, val_loss/len(val_dataloader)))

if __name__ == "__main__":
    train()
