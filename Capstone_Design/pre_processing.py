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
    #print(dt_first.hour, type(dt_first.hour))
    '''
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
                df[0][1] += ":"+str(first_min)py
        first_min = 60
        mcnt = mcnt -1
        first_hour = first_hour - 1
    print("맨 앞에 추가해야할 행의 갯수는 : ",mcnt-1)
    #초기화 시킨 앞에 행 완료.
    '''
    realdate = '2'
    realvalue = '3'
    data=[] 
 
    raw_data = pd.DataFrame(data)
    raw_data.to_csv('./data/processed/sample.csv')
    
    
    csv = pd.read_csv('./data/unprocessed/70man.csv')

    df2 = pd.DataFrame(data=None, columns=csv.columns,index=csv.index) #초기화 시킨 csv
    df2 = df2.drop(index)
    print(df2)
    df2.to_csv('./data/processed/sample.csv')

    for date in df[0]:
        #조건 7.5 , 22.5, 37.5, 52.5로 기준을 나눈다.
        #상세 조건으로 23시 이고 분이 52.5보다 큰경우 해당 값의 시간은 0으로 바꾼다.
        realtime = datetime.strptime(date,'%Y-%m-%d %H:%M')
        realhour = realtime.hour
        realhour2 = int(realhour)
        realminute = realtime.minute
        realminute2 = int(realminute)
        #조건 1. 시간이 23시이냐 아니냐
        #조건 2. 분에 대해서 큰지 작은지 범위로 00 15 30 45로 기준화처리
        if realhour2 < 23:
            if realminute2 < 7.5:
                realminute2 = 0
            elif realminute2 > 7.5 and realminute2 < 22.5:
                realminute2 = 15
            elif realminute2 > 22.5 and realminute2 < 37.5:
                realminute2 = 30
            elif realminute2 > 37.5 and realminute2 < 52.5:
                realminute2 = 45
            elif realminute2 > 52.5:
                realminute2 = 0
                realhour2+=1
                #분이 52.5보다 크면 다음 시간으로 넘겨서 00분으로 처리
        elif realhour2 == 23:
            if realminute2 < 7.5:
                realminute2 = 0
            elif realminute2 > 7.5 and realminute2 < 22.5:
                realminute = 15
            elif realminute2 > 22.5 and realminute2 < 37.5:
                realminute2 = 30
            elif realminute2 > 37.5 and realminute2 < 52.5:
                realminute2 = 45
            elif realminute2 > 52.5:
                realminute2 = 0
                realhour2 = 0
        realhour = str(realhour2)
        realminute = str(realminute2)
        

        # 7. Excel 파일로 저장
        #df_att.to_excel('36_official_language.xlsx', index=False)
        #print(realhour," ",realminute)
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