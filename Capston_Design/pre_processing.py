from src.utils import linear_interpolation
import pandas as pd
import numpy as np
from pandas import read_csv, Series, DataFrame
from datetime import datetime

# TODO

# 1. 데이터 불러오기.
def upload():
    input_file = './data/unprocessed/70man(3).csv'

    df = read_csv(input_file, header=None, index_col = None, delimiter = ',')
    #혈당 데이터가 첫번째 행이 string일때
    print(df[1][0])
    if str(df[1][0]) == "Glucose(mg/dL)":
        print("clear")
        df = df.drop([0])
        #print(df.head(1))
        print("upload and delete String values")
    return df
# 2. 선형 보간 ( 결측값에 대한 보간하기)
def interpolation():
    df = upload()
    
    #시계열 날짜 inddex를 기준으로 결측값 보간하자
    #print(df[0]) # 시간값은 1부터
    # 데이터의 0번째가 0시부터가 아니면 그냥 채워버리기
    dt_first = datetime.strptime(df[0][1],'%Y-%m-%d %H:%M')
    print(dt_first.hour, type(dt_first.hour))
    mcnt=0
    hcnt = 0
    first_hour = int(dt_first.hour)
    first_min = int(dt_first.minute)
    while(first_hour >= 0):
        while (first_min >0):
            first_min = first_min - 15
            mcnt = mcnt+1
            df = df.shift(1)
            df[0][1] = str(dt_first.year)
            if(dt_first.month < 10):
                df[0][1] += "-0"+str(dt_first.month)
            elif dt_first.month >= 10:
                df[0][1] += "-"+str(dt_first.month)
            
            if(dt_first.day < 10):
                df[0][1] += "-"+"0"+str(dt_first.day)
            elif dt_first.day >=10:
                df[0][1] += "-"+str(dt_first.day)
            if first_hour < 10:
                if first_hour == 0:
                    df[0][1] += " "+str(first_hour)
                else:
                    df[0][1] += " "+"0"+str(first_hour)
            elif first_hour >=10:
                df[0][1] += " "+str(first_hour)
            if first_min <10:
                df[0][1] += ":"+"0"+str(first_min)
            elif first_min >=10:
                df[0][1] += ":"+str(first_min)
        first_min = 60
        mcnt = mcnt -1
        first_hour = first_hour - 1
    print("맨 앞에 추가해야할 행의 갯수는 : ",mcnt-1)
    #초기화 시킨 앞에 행 완료.
    #print(df.head(30))
    print(dt_first.minute+40)
    for date in range(len(df[0])):
        if date > 2:
            dt_plus = datetime.strptime(df[0][date],'%Y-%m-%d %H:%M')
            dt_prev = datetime.strptime(df[0][date-1],'%Y-%m-%d %H:%M')
            #if(dt_plus.minute - dt_prev.minute > 20):
                #평범하게 중간에 빈 경우

            
    #for dt in df[0]:
        #dt = datetime.strptime(dt, '%Y-%m-%d %H:%M')
        #1. 
        #print(dt.hour, dt.minute) # 
    #맨앞부터 돌면서 맨앞이면 내 값부터 이전으로 0시까지 만들기
    #if(dt)
    #먼저 하루의 데이터들을 모두 0~23시로 채워버린다. 없으면 만든다.
    #for data in df[0]:
    #    if()
    dates = df[0]
    #ts = Series([ np.nan, np.nan, 10],index = dates)
    #ts_intp_by_time = 
#This file makes unprocessed csv file as a perfectly processed csv file for training
#The method of linear interpolation exist in the src\utils directories.
def main():
    interpolation()
if __name__ == "__main__":
    main()

