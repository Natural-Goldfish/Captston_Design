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
import pandas as pd

_CUDA_FLAG = torch.cuda.is_available()
_MODEL_PATH = "data\\models"
_MODEL_NAME = "test.pth"
_INDEX = 0
_DATA_LOAD_PATH = "./data/processed/"
_FILE_LOAD_NAME = "70man_test.csv"
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
        print(len(input_data)+len(prediction), input_data, label)
        print(len(input_data)+len(label))

        x_axis = [i for i in range(1, 577)]
        predict_graph = np.array(torch.cat((input_data, prediction), dim = 0).cpu())
        true_graph = np.array(torch.cat((input_data, label), dim = 0).cpu())
        

        #csv 부르기
        findindex = pd.read_csv(_DATA_LOAD_PATH+_FILE_LOAD_NAME)
        print(len(findindex))

        plt.subplot(2, 1, 1)
        plt.title("Prediction Glucose")
        plt.plot(x_axis, predict_graph, color = 'green')
        plt.axis([1, 576, 0, 300])
        
        plt.subplot(2, 1, 2)
        plt.title("Real Glucose")
        plt.plot(x_axis, true_graph, color = 'blue')
        plt.axis([1, 576, 0, 300])
        plt.show()

#def checkingTime(Labelgraph):#라벨 그래프의 

if __name__ == "__main__":
    test()