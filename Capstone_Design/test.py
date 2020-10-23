from src.dataset import ASPDataset
from src.model import ASPModel
from src.utils import Normalization
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
import argparse
import torch
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

_CUDA_FLAG = torch.cuda.is_available()
_MODEL_PATH = "data\\models"
_MODEL_NAME = "ASPModel_{}_checkpoint.pth"
_INDEX = 0
_DATA_LOAD_PATH = "./data/processed/"
_FILE_TEST_NAME = "70man_test2.csv"
_FILE_TRAIN_NAME = "70man_train2.csv"
def test():
    # For CMD
    #parser = argparse.ArgumentParser()
    #parser.add_argument("--model_name", type = str, required = True, help = "this is a csv file name to load")
    #parser.add_argument("--index", type = int, required = True, help = "this is a index you will load in the test csv file")
    #args = parser.parse_args()
    mode = ['train', 'test']

    with torch.no_grad():
        norm = Normalization()
        dataset = ASPDataset(mode = 'test')
        input_data, label = dataset[_INDEX]
        

        model = ASPModel()
        #model.load_state_dict(torch.load(os.path.join(_MODEL_PATH, _MODEL_NAME.format(args.model_name))))
        model.eval()

        if _CUDA_FLAG :
            input_data = input_data.cuda()
            label = label.cuda()
            model.cuda()

        output = model(input_data)
        prediction = norm.de_normalize(output).view(-1)
        

        
       

        for _, mode in enumerate(mode):
            dataset = ASPDataset(mode = mode)
            input_data, label = dataset[_INDEX]

            model = ASPModel()
            device = torch.device('cpu')
            
            #model.load_state_dict(torch.load(os.path.join(_MODEL_PATH, _MODEL_NAME.format(args.model_name)),map_location=device))
            model.eval()

            if _CUDA_FLAG :
                input_data = input_data.cuda()
                label = label.cuda()
                model.cuda()

            output = model(input_data)
            prediction = norm.de_normalize(output).view(-1)

            visual(input_data, prediction, label, mode)
        plt.show()
            

def visual(input_data, prediction, label, mode):

    predict_graph = np.array(torch.cat((input_data, prediction), dim = 0))
    true_graph = np.array(torch.cat((input_data, label), dim = 0))

    x_axis = [i for i in range(1, 577)]
     #csv 부르기
    test_time = pd.read_csv(_DATA_LOAD_PATH+_FILE_TEST_NAME)
    train_time = pd.read_csv(_DATA_LOAD_PATH+_FILE_TRAIN_NAME)
    test_predict_time = test_time['Time'][1:577]
    starttime = test_time['Time'][1]
    endtime = test_time['Time'][577]
    tpt = []
    tpt.append(starttime)
    for idx, value in enumerate(test_predict_time):
        if test_time['Glucose'][idx] >=200:
            tpt.append(test_time['Time'][idx]) 
    
    tpt.append(endtime)
    print(tpt)
    
    test_predict_time = np.array(test_predict_time)
    
    limit = [1, 576, 0, 300]
    danger  = [140 for i in range(1, 577)]
    fig = plt.figure("{} DATASET".format(mode.upper()))
    ax = plt.subplot(3,1,1)
    #ax.set_xticks(ax.get_xticks()[::20])
    plt.plot(test_predict_time, predict_graph, color = 'green')
    plt.plot(test_predict_time,danger ,color = 'red')
    plt.xticks(test_predict_time[0::80])
    plt.title("Prediction Glucose")
    plt.axis(limit)
    plt.grid()

    ax = plt.subplot(3,1,2)
    
    plt.title("Real Glucose")
    plt.plot(test_predict_time, true_graph, color = 'blue')
    plt.plot(test_predict_time,danger ,color = 'red')
    plt.xticks(test_predict_time[0::80])
    plt.axis(limit)
    plt.grid()

    ax = plt.subplot(3,1,3)
    
    
    plt.title("Together")
    plt.plot(test_predict_time, predict_graph, color = 'green')
    plt.plot(test_predict_time, true_graph, color = 'blue')
    plt.plot(test_predict_time,danger ,color = 'red')
    plt.xticks(test_predict_time[::80])
    plt.axis(limit)
    plt.grid()

#def checkingTime(Labelgraph):#라벨 그래프의 

if __name__ == "__main__":
    test()