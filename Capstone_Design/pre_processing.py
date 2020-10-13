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
    print(type(df[1][0]))
    if str(df[1][0]) == "Glucose(mg/dL)":
        print("clear")
        df = df.drop([0])
        #print(df.head(1))
        print(df[0]) # 1518 수정
        print("upload and delete String values")
    return df
# 2. 선형 보간 ( 결측값에 대한 보간하기)
def interpolation():
    df = upload()
    #시계열 날짜 inddex를 기준으로 결측값 보간하자
    #print(df[0]) # 시간값은 1부터
    # 데이터의 0번째가 0시부터가 아니면 그냥 채워버리기
    #print(dt_first.hour, type(dt_first.hour))
    
    cnt = 1
     
    for index,date in enumerate(df[0]):
        #조건 7.5 , 22.5, 37.5, 52.5로 기준을 나눈다.
        #상세 조건으로 23시 이고 분이 52.5보다 큰경우 해당 값의 시간은 0으로 바꾼다.
        #print(type(index[0]))
        realtime = datetime.strptime(date,'%Y-%m-%d %H:%M')
        realhour = realtime.hour

        realminute = realtime.minute
        
        
        #조건 1. 시간이 23시이냐 아니냐
        #조건 2. 분에 대해서 큰지 작은지 범위로 00 15 30 45로 기준화처리
        if realhour < 23:
            if realminute < 7.5:
                realminute = 0
            elif realminute > 7.5 and realminute < 22.5:
                realminute = 15
            elif realminute > 22.5 and realminute < 37.5:
                realminute = 30
            elif realminute > 37.5 and realminute < 52.5:
                realminute = 45
            elif realminute > 52.5:
                realminute = 0
                realhour+=1
                #분이 52.5보다 크면 다음 시간으로 넘겨서 00분으로 처리
        elif realhour == 23:
            if realminute < 7.5:
                realminute = 0
            elif realminute > 7.5 and realminute < 22.5:
                realminute = 15
            elif realminute > 22.5 and realminute < 37.5:
                realminute = 30
            elif realminute > 37.5 and realminute < 52.5:
                realminute = 45
            elif realminute > 52.5:
                realminute = 0
                realhour = 0
        if realhour < 10:
            realhour = '0'+str(realhour)
        elif realhour >= 10:
            realhour = str(realhour) 
        if realminute < 10:
            realminute = '0'+str(realminute)
        elif realminute >= 10:
            
            realminute = str(realminute)
        
        val = str(realtime.year)+'-'+str(realtime.month)+'-'+str(realtime.day)+' '+realhour+':'+realminute
        df[1][cnt] = int(df[1][cnt])
        df[0][cnt] = val
        df.to_csv('./data/processed/sample.csv')
        #print(df[0][cnt])
        #print(df[0][cnt])
        cnt+=1
        # 7. Excel 파일로 저장
        
    
            
    #for dt in df[0]:
        #dt = datetime.strptime(dt, '%Y-%m-%d %H:%M')
        #1. 
        #print(dt.hour, dt.minute) # 
    #맨앞부터 돌면서 맨앞이면 내 값부터 이전으로 0시까지 만들기
    #if(dt)
    #먼저 하루의 데이터들을 모두 0~23시로 채워버린다. 없으면 만든다.
    #for data in df[0]:
    #    if()

    #ts = Series([ np.nan, np.nan, 10],index = dates)
    #ts_intp_by_time = 
#This file makes unprocessed csv file as a perfectly processed csv file for training
#The method of linear interpolation exist in the src\utils directories.
def interpolation2():
    #interpolation()
    input_file= './data/processed/sample.csv'
    df = read_csv(input_file, header=None, index_col = None, delimiter = ',')
    
    # csv를 부른다.
    cnt = 0
    for stop,date in enumerate(df[1]):
        if stop == 0:
            continue
        else:
            if stop >=2:
                realtime = datetime.strptime(date,'%Y-%m-%d %H:%M')
                pasttime = datetime.strptime(df[1][stop-1],'%Y-%m-%d %H:%M')
            
                if 23 >pasttime.hour:
                    if realtime.hour > pasttime.hour:
                        if ((realtime.hour*60+realtime.minute)-(pasttime.hour*60+pasttime.minute)) > 15:
                            #csv 분리하기
                            
                            
                            print("past :",pasttime, 'stop',stop)
                            
                            cnt+=1
                            
                            rows = pd.read_csv(input_file)
                            
                            # split indexes
                            idxes = np.array_split(rows.index.values, stop)

                            chunks = [rows.loc[idx] for idx in idxes]
                            rows.to_csv('./data/processed/first.csv')
                            print("ya")
                            
                            second = pd.read_csv(input_file,chunksize=len(df[1]))
                            for j, data in enumerate(second):
                                
                                print("hi")
                                data.to_csv('./data/processed/second.csv'.format(j))
                            
                    elif realtime.hour < pasttime.hour:
                        if (((realtime.hour+24)*60+realtime.minute)-(pasttime.hour*60+pasttime.minute)) > 15:
                            cnt+=1
                            print("past :",pasttime, 'stop',stop)
                            
                            




                        




        #realminute = realtime.minute
        #print(df[0][cnt],df[0][cnt+1])
    print(cnt)
    #시간 차이가 15분 이상 나면  csv를 분리~~ 앞에꺼 1개 뒤에꺼 1개로 따로 저장 후 뒤에꺼에 부족한 갯수만큼 추가~~~
    
def main():
    interpolation2()
if __name__ == "__main__":
    main()