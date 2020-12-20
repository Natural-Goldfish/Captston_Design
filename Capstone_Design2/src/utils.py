import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


_MIN = 0
_MAX = 300

class Normalization(object):
    def __init__(self, min = _MIN, max = _MAX):
        super().__init__()
        self.min = min
        self.max = max

    def normalize(self,data):
        # x- min/ max-min
        norm_data = (data-self.min) / (self.max - self.min)
        return norm_data

    def de_normalize(self, data):
        de_norm_data = data*(self.max-self.min)+self.min
        return de_norm_data

"""

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

def checkingdiabete(input_data,label,mode):
    #실제 사용자의 혈당그래프를 분석
    limit = [1, 576, 0, 300]
    fig = plt.figure("{} DATASET".format(mode.upper()))
    true_graph = np.array(torch.cat((input_data,label),dim =0))
    #200넘는 혈당 갯수 체크
    dangervalue = []
    dangertime = []
    countdanger=0
    for idx,above in enumerate(true_graph):
        if above >= 200:
            dangervalue.append(above)
            dangertime.append(idx)
            #true_graph[idx] = np.nan
            countdanger+=1
        else:
            dangervalue.append(np.nan)
            dangertime.append(idx)
            
    percent_of_danger = int(countdanger/len(true_graph)*100)

    test_time = pd.read_csv(_DATA_LOAD_PATH +_FILE_TEST_NAME)
    danger_period =[]
    for i in dangertime:
        danger_period.append(test_time['Time'][i])
    test_predict_time = test_time['Time'][1:577]
    
    plt.plot(test_predict_time, true_graph, color = 'blue',label = 'true')
    plt.plot(test_predict_time,dangervalue, color = 'red', label =str(percent_of_danger)+'%')
    plt.xticks(test_predict_time[::80])
    plt.title('Checking above 200(mg/dL)')
    plt.axis(limit)
    plt.legend()
    plt.grid()


def find_meattime(input_data,label,mode):
    #실제 사용자의 혈당그래프를 분석
    limit = [1, 576, 0, 300]
    fig = plt.figure("{} DATASET".format(mode.upper()))
    test_time = pd.read_csv(_DATA_LOAD_PATH+_FILE_TEST_NAME)
    true_graph = np.array(torch.cat((input_data,label),dim =0))
    #목적 : 식후1시간 혈당 구간 찾아내기
    afvalue = [] #식전 혈당
    aftime = [] #식전 혈당 시간
    countdanger=0
    for idx,above in enumerate(true_graph):
        if idx >= 4: #idx가 맨 마지막에서 4번째까지만 진행
            if true_graph[idx] >= 25 + true_graph[idx-4] and true_graph[idx-4] >= 80 and true_graph[idx] >=140:
                #1시간 뒤 혈당이 지금 혈당보다 35이상 높다면
                if idx+3 < len(true_graph)-1:
                    if true_graph[idx] > true_graph[idx+3]+25 or true_graph[idx]>=100:
                        afvalue.append(true_graph[idx])
                        if true_graph[idx] >=180:
                            countdanger+=1
                    else:
                        afvalue.append(np.nan)
                else:
                    if true_graph[idx]>140:
                        afvalue.append(true_graph[idx])
                    else:
                        afvalue.append(np.nan)
            elif true_graph[idx] >= 25 +true_graph[idx-3] and true_graph[idx-3] > 100:
                if true_graph[idx] >= 100:
                    afvalue.append(true_graph[idx])
                    if true_graph[idx] >= 180:
                        countdanger+=1
                else:
                    afvalue.append(np.nan)
            
            else:
                if true_graph[idx] >= 180:
                    #200이 넘는 혈당 구간은 반드시 식후 혈당의 피크 부분에 포함됨이 분명
                    afvalue.append(true_graph[idx])
                    countdanger+=1
                else:
                    afvalue.append(np.nan)
        else:
            afvalue.append(np.nan)
        
    test_predict_time = test_time['Time'][1:577]
    danger_afmeat = int(countdanger/len(true_graph)*100)
    danger_line = [180 for i in range(1,577)]
    plt.plot(test_predict_time,danger_line, color = 'purple', label = 'dangerline')
    plt.plot(test_predict_time, true_graph, color = 'blue',label = 'true')
    plt.plot(test_predict_time,afvalue, color = 'red', label =str(danger_afmeat)+'%')
    plt.xticks(test_predict_time[::80])
    plt.title('After meet 1~2hours\n'+'above 180(mg/dL) danger')
    plt.axis(limit)
    plt.legend()
    plt.grid()
"""