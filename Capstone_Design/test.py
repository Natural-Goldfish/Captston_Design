from src.dataset import ASPDataset
from src.model import ASPModel
from src.utils import Normalization
from torch.utils.data import DataLoader
import argparse
import torch
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt

_CUDA_FLAG = torch.cuda.is_available()
_MODEL_PATH = "data\\models"
_MODEL_NAME = "test.pth"
_INDEX = 0

def test():
    """
    # For CMD
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type = str, required = True, help = "this is a file name to load")
    """
    with torch.no_grad():
        norm = Normalization()
        dataset = ASPDataset(mode = 'test')
        input_data, label = dataset[_INDEX]

        model = ASPModel()
        #model.load_state_dict(torch.load(os.path.join(_MODEL_PATH, _MODEL_NAME)))
        model.eval()

        if _CUDA_FLAG :
            input_data = input_data.cuda()
            label = label.cuda()
            model.cuda()

        output = model(input_data)
        prediction = norm.de_normalize(output).view(-1)

        x_axis = [i for i in range(1, 577)]
        predict_graph = np.array(torch.cat((input_data, prediction), dim = 0).cpu())
        true_graph = np.array(torch.cat((input_data, label), dim = 0).cpu())
        
        plt.subplot(2, 1, 1)
        plt.title("Prediction Glucose")
        plt.plot(x_axis, predict_graph, color = 'green')
        plt.axis([1, 576, 0, 300])
        
        plt.subplot(2, 1, 2)
        plt.title("Real Glucose")
        plt.plot(x_axis, true_graph, color = 'blue')
        plt.axis([1, 576, 0, 300])
        plt.show()

if __name__ == "__main__":
    test()