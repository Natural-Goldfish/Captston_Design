from src.model import ASPModel
from src.dataset import *
from src.utils import Normalization
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import os

_SEQUENCE_LENGTH = 96*3
_INPUT_DIM = 50
_HIDDEN_DIM = 50
_EMBEDDING_DIM = 50
_LEARNING_RATE = 0.001
_EPOCHS = 2600
_BATCH_SIZE = 128
_CUDA_FLAG = torch.cuda.is_available()

_MODEL_LOAD_FLAG = False
_MODEL_PATH = "data\\models"
_MODEL_LOAD_NAME = "ASPModel_{}_checkpoint.pth"

def train():
    # Load objects for training
    train_dataset = ASPDataset(mode = "train")
    train_dataloader = DataLoader(train_dataset, batch_size = _BATCH_SIZE, shuffle = True)
    val_dataset = ASPDataset(mode = "test")
    val_dataloader = DataLoader(val_dataset, batch_size = _BATCH_SIZE, shuffle = False)
    model = ASPModel(seq_len = _SEQUENCE_LENGTH, input_dim = _INPUT_DIM, hidden_dim = _HIDDEN_DIM, embedding_dim = _EMBEDDING_DIM)

    if _MODEL_LOAD_FLAG :
        model.load_state_dict(torch.load(os.path.join(_MODEL_PATH, _MODEL_LOAD_NAME)))
    if _CUDA_FLAG : model.cuda()

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = _LEARNING_RATE)
    norm = Normalization()

    ######## TEMP CODE BLCOK ########
    writer = SummaryWriter("Tensorboard_test")
    ######## TEMP CODE BLCOK ########

    for cur_epoch in range(_EPOCHS):
        # Learning Rate Scheduler
        if cur_epoch == 300 :
            optimizer.param_groups[0]["lr"] = 0.0001
        elif cur_epoch == 500 :
            optimizer.param_groups[0]["lr"] = 0.00001
            
        # Training
        model.train()
        optimizer.zero_grad()
        train_total_loss = 0.0

        for cur_iter, train_data in enumerate(train_dataloader):
            # Data load
            train_inputs, train_labels = train_data
            if _CUDA_FLAG :
                train_inputs = train_inputs.cuda()
                train_labels = train_labels.cuda()

            # Update parameters
            train_outputs = model(train_inputs).view(-1, _SEQUENCE_LENGTH)
            train_labels = norm.normalize(train_labels)       # Experimental
            train_loss = criterion(train_outputs, train_labels)
            train_loss.backward()
            optimizer.step()
            train_total_loss += train_loss.detach()

            ######## TEMP CODE BLCOK ########
            if cur_epoch % 600 == 599 : 
                _test_sample("train_prediction", norm.de_normalize(train_labels), norm.de_normalize(train_outputs), train_inputs)
                break
            ######## TEMP CODE BLCOK ########

            print("TRAIN ::: EPOCH {}/{} Iteration {}/{} Loss {:.6f}".format(cur_epoch+1, _EPOCHS, cur_iter, len(train_dataloader), train_loss))
        
        # Evaludation
        with torch.no_grad() :
            model.eval()
            val_total_loss = 0.0

            for cur_iter, val_data in enumerate(val_dataloader):
                # Data load
                val_inputs, val_labels = val_data
                if _CUDA_FLAG :
                    val_inputs = val_inputs.cuda()
                    val_labels = val_labels.cuda()

                val_outputs = model(val_inputs).view(-1, _SEQUENCE_LENGTH)
                val_loss = criterion(val_outputs, norm.normalize(val_labels))
                val_total_loss += val_loss
                test_output = norm.de_normalize(val_outputs)

                ######## TEMP CODE BLCOK ########
                if cur_epoch % 600 == 599 : 
                    _test_sample("test_prediction", val_labels, test_output, val_inputs)
                    break
                ######## TEMP CODE BLCOK ########

            print("VAL ::: EPOCH {}/{} Loss {:.6f}".format(cur_epoch+1, _EPOCHS, val_total_loss/len(val_dataloader)))

        ######## TEMP CODE BLCOK ########
        writer.add_scalars("Loss", {"train_loss" : train_total_loss/len(train_dataloader), "val_loss" : val_total_loss/len(val_dataloader)}, cur_epoch)
        if cur_epoch% 600 == 599 :  
            torch.save(model.state_dict(), os.path.join(_MODEL_PATH, _MODEL_LOAD_NAME.format(cur_epoch)))
            break
        ######## TEMP CODE BLCOK ########

    ######## TEMP CODE BLCOK ########
    writer.close()
    ######## TEMP CODE BLCOK ########

######## TEMP CODE BLCOK ########
def _test_sample(name, val_label, val_output, val_input):
    writer_test = SummaryWriter("{}".format(name))
    test_input = val_input[0]
    prediction = val_output[0]
    test_label = val_label[0]
    predict = torch.cat((test_input, prediction), dim = 0)
    true = torch.cat((test_input, test_label), dim = 0)
    for i in range(288*2):
        writer_test.add_scalars("Glucose", {"True" : true[i], "prediction" : predict[i]}, i)
    writer_test.close()
######## TEMP CODE BLCOK ########

if __name__ == "__main__":
    train()
