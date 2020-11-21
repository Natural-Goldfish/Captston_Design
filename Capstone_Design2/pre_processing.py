#from src.utils import linear_interpolation
import os
import pandas as pd
import numpy as np
from pandas import read_csv, Series, DataFrame
from datetime import datetime
# TODO
_DATA_SAVE_PATH = "data\\processed"
_DATA_LOAD_PATH = "data\\unprocessed"
_FILE_LOAD_NAME = "70man(3).csv"
_FILE_SAVE_NAME = "70man"
_FILE_RESULT_NAME = "result.csv"

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
        
    
            
   
#This file makes unprocessed csv file as a perfectly processed csv file for training
#The method of linear interpolation exist in the src\utils directories.
def interpolation2():
    #interpolation()
    input_file= './data/processed/sample.csv'
    df = read_csv(input_file, header=None, index_col = None, delimiter = ',')
    
    # csv를 부른다.
    cnt = 0

     #부족한 데이터 메꾸기 위한 새 dataframe을 만들어서 새로운 csv에 저장하기.
    newdf = pd.DataFrame(columns = ['Time','Gluco'])
    newdf.to_csv('./data/processed/newdf.csv') 
    newdf['Gluco'].astype('int')
    for stop,date in enumerate(df[1]):
        if stop == 0:
            continue
        else:
            if stop ==1:
                #부족한 데이터 메꾸기 위한 새 dataframe을 만들어서 새로운 csv에 저장하기.
                realdate1 = datetime.strptime(date,'%Y-%m-%d %H:%M')
                day = realdate1.day
                month = realdate1.month
                minute = realdate1.minute
                hour = realdate1.hour

                hour2 = str(hour)
                minute2 = str(minute)
                day2 = str(day)
                month2 = str(month)
                if int(minute2) < 10:
                    minute2 = '0'+minute2
                if int(day2) < 10:
                    day2 = '0'+day2
                if int(month2) < 10:
                    month2 = '0'+month2
                realdata1 = str(realdate1.year)+'-'+month2+'-'+day2+' '+hour2+':'+minute2
                newdf.loc[cnt] = {'Time':realdata1,'Gluco':str(df[2][stop])}
                #새로 생성되는 newdf csv 파일에 기존의 혈당데이터를 저장
                
                newdf.to_csv('./data/processed/newdf.csv')
                continue
                cnt+=1
            elif stop >=2:
                realtime = datetime.strptime(date,'%Y-%m-%d %H:%M')
                pasttime = datetime.strptime(df[1][stop-1],'%Y-%m-%d %H:%M')
                
                # 새로 추가될 데이터 이전에 기존 데이터들을 먼저 삽입
                day = pasttime.day
                month = pasttime.month
                minute = pasttime.minute
                hour = pasttime.hour

                hour2 = str(hour)
                minute2 = str(minute)
                day2 = str(day)
                month2 = str(month)
                if int(minute2) < 10:
                    minute2 = '0'+minute2
                if int(day2) < 10:
                    day2 = '0'+day2
                if int(month2) < 10:
                    month2 = '0'+month2
                realdata = str(pasttime.year)+'-'+month2+'-'+day2+' '+hour2+':'+minute2
                newdf.loc[cnt] = {'Time':realdata,'Gluco':str(df[2][stop])}
                #새로 생성되는 newdf csv 파일에 기존의 혈당데이터를 저장
                
                newdf.to_csv('./data/processed/newdf.csv')
                cnt += 1
                if 23 >pasttime.hour:
                    # 뒤에 시간이 전 시간보다 큰 경우
                    if realtime.hour > pasttime.hour:
                        if ((realtime.hour*60+realtime.minute)-(pasttime.hour*60+pasttime.minute)) > 15:
                            #csv 분리하기
                            #print("past :",pasttime, 'stop',stop, 'real',realtime)
                            #추가할 데이터 갯수 및 데이터들

                            #빈 부분 만들어내기
                            count = ((realtime.hour*60+realtime.minute)-(pasttime.hour*60+pasttime.minute))/15 -1
                            
                            #새 데이터 시작 일/ 시간/ 분
                            minute3 = pasttime.minute
                            hour3 = pasttime.hour
                            day3 = pasttime.day
                            month3 = pasttime.month
                            count = int(count)
                            
                            #pasttime시작으로 15분씩 추가하면서 새로운 데이터를 생성해낸다
                            for i in range(count):
                                minute3 +=15
                                if minute3 >= 60:
                                    minute3 -=60
                                    hour3 +=1
                                    if hour3 == 0:
                                        #날짜를 하루 올리기
                                        day3+=1
                                        hour3 -=24
                                        thirtyone = [1,3,5,7,8,10,12]
                                        thirty = [4,6,9,11]
                                        feb= [2]
                                        if day3 >31:
                                            if month3 in thirtyone:
                                                day3 -= 31
                                                month3+=1
                                            elif month3 in thirty:
                                                day3 -=30
                                                month3+=1
                                            else:
                                                day3 -=28
                                                month3+=1 
                                #day hour minute이 10보다 작으면 앞에 0붙여서 string으로 변환
                                day4 = str(day3)
                                hour4 = str(hour3)
                                minute4 = str(minute3)
                                month4 = str(month3)
                                if int(month4) < 10:
                                    month4 = '0'+month4
                                if int(minute4) < 10:
                                    minute4 = '0'+minute4
                                if int(hour4) < 10:
                                    hour4 = '0'+hour4
                                if int(day4) < 10:
                                    day4 = '0'+day4
                                
                                newdate4 = str(realtime.year)+'-'+month4+'-'+day4+' '+hour4+':'+minute4
                                #print("new date",newdate4)
                                #이제 새로 생성한 데이터를 저장한다.
                                newdf.loc[cnt,'Time'] = newdate4
                                newdf.loc[cnt,'Gluco'] = str(np.nan)
                                 #새로 생성되는 newdf csv 파일에 기존의 혈당데이터를 저장
                                newdf.to_csv('./data/processed/newdf.csv')
                                cnt += 1

                    
                    elif realtime.hour < pasttime.hour:
                        if (((realtime.hour+24)*60+realtime.minute)-(pasttime.hour*60+pasttime.minute)) > 15:
                            
                            #print("past :",pasttime, 'stop',stop, 'real',realtime)
                            count = (((realtime.hour+24)*60+realtime.minute)-(pasttime.hour*60+pasttime.minute))/15 -1
                            #print(count)
                            minute5 = pasttime.minute
                            hour5 = pasttime.hour
                            day5 = pasttime.day
                            month5 = pasttime.month
                            count = int(count)
                            
                            #pasttime시작으로 15분씩 추가하면서 새로운 데이터를 생성해낸다
                            for i in range(count):
                                minute5 +=15
                                if minute5 >= 60:
                                    minute5 -=60
                                    hour5 +=1
                                    if hour5 == 24:
                                        #날짜를 하루 올리기
                                        day5+=1
                                        hour5 -=24
                                        thirtyone = [1,3,5,7,8,10,12]
                                        thirty = [4,6,9,11]
                                        feb= [2]
                                        if day5 >31:
                                            if month5 in thirtyone:
                                                day5 -= 31
                                                month5+=1
                                            elif month5 in thirty:
                                                day5 -=30
                                                month5+=1
                                            else:
                                                day5 -=28
                                                month5+=1 

                                #day hour minute이 10보다 작으면 앞에 0붙여서 string으로 변환
                                
                                day6 = str(day5)
                                hour6 = str(hour5)
                                minute6 = str(minute5)
                                month6 = str(month5)
                                if int(minute6) < 10:
                                    minute6 = '0'+minute6
                                if int(hour6) < 10:
                                    hour6 = '0'+hour6
                                if int(day6) < 10:
                                    day6 = '0'+day6
                                if int(month6) < 10:
                                    month6 = '0'+month6
                                newdate6 = str(realtime.year)+'-'+month6+'-'+day6+' '+hour6+':'+minute6
                                #print("new date",newdate6)
                                
                                #이제 새로 생성한 데이터를 저장한다.
                                #newdf.loc[cnt,'Time'] = newdate
                                #newdf.loc[cnt,'Gluco'] = np.nan
                                newdf.loc[cnt] = {'Time':newdate6, 'Gluco': str(np.nan)}
                                #새로 생성되는 newdf csv 파일에 기존의 혈당데이터를 저장
                                newdf.to_csv('./data/processed/newdf.csv')
                                cnt = cnt+1

                elif pasttime.hour ==23:
                    if realtime.hour < pasttime.hour:
                        if (((realtime.hour+24)*60+realtime.minute)-(pasttime.hour*60+pasttime.minute)) > 15:
                            
                            #print("past :",pasttime, 'stop',stop, 'real',realtime)
                            count = (((realtime.hour+24)*60+realtime.minute)-(pasttime.hour*60+pasttime.minute))/15 -1
                            #print(count)
                            minute7 = pasttime.minute
                            hour7 = pasttime.hour
                            day7 = pasttime.day
                            month7 = pasttime.month
                            count = int(count)
                            
                            #pasttime시작으로 15분씩 추가하면서 새로운 데이터를 생성해낸다
                            for i in range(count):
                                minute7 +=15
                                if minute7 >= 60:
                                    minute7 -=60
                                    hour7 +=1
                                    if hour7 == 24:
                                        #날짜를 하루 올리기
                                        day7+=1
                                        hour7 -=24
                                        thirtyone = [1,3,5,7,8,10,12]
                                        thirty = [4,6,9,11]
                                        feb= [2]
                                        if day7 >31:
                                            if month7 in thirtyone:
                                                day7 -= 31
                                                month7+=1
                                            elif month7 in thirty:
                                                day7 -=30
                                                month7+=1
                                            else:
                                                day7 -=28
                                                month7+=1 

                                #day hour minute이 10보다 작으면 앞에 0붙여서 string으로 변환
                                
                                day8 = str(day7)
                                hour8 = str(hour7)
                                minute8 = str(minute7)
                                month8 = str(month7)
                                if int(minute8) < 10:
                                    minute8 = '0'+minute8
                                if int(hour8) < 10:
                                    hour8 = '0'+hour8
                                if int(day8) < 10:
                                    day8 = '0'+day8
                                if int(month8) < 10:
                                    month8 = '0'+month8
                                newdate8 = str(realtime.year)+'-'+month8+'-'+day8+' '+hour8+':'+minute8
                                #print("new date",newdate6)
                                
                                #이제 새로 생성한 데이터를 저장한다.
                                #newdf.loc[cnt,'Time'] = newdate
                                #newdf.loc[cnt,'Gluco'] = np.nan
                                newdf.loc[cnt] = {'Time':newdate8, 'Gluco': str(np.nan)}
                                #새로 생성되는 newdf csv 파일에 기존의 혈당데이터를 저장
                                newdf.to_csv('./data/processed/newdf.csv')
                                cnt = cnt+1
                    

       
    print(cnt)
    #시간 차이가 15분 이상 나면  csv를 분리~~ 앞에꺼 1개 뒤에꺼 1개로 따로 저장 후 뒤에꺼에 부족한 갯수만큼 추가~~~

def final_interpolate():
    #먼저 빈 곳에 nan값을입력
    df = pd.read_csv('./data/processed/newdf.csv')
    # empty frame with desired index
    df2 = df.interpolate()
    df2['Gluco'] = df2['Gluco'].astype(int)
    df2.to_csv('./data/processed/result.csv')

def main():
    #interpolation2()
    final_interpolate()
    load_path = _DATA_LOAD_PATH
    load_fname = _FILE_LOAD_NAME
    save_path = _DATA_SAVE_PATH
    save_fname = _FILE_SAVE_NAME
    cur_fpath = os.path.join(load_path, load_fname)
    train_new_data, val_new_data = _split_dataset()
    save_train_df = pd.DataFrame(train_new_data, columns = ["Time", "Glucose"])
    save_train_df.to_csv(os.path.join(_DATA_SAVE_PATH, "{}_{}.csv".format(save_fname, "train2")), index = False)

    # Save val dataset
    save_val_df = pd.DataFrame(val_new_data, columns = ["Time", "Glucose"])
    save_val_df.to_csv(os.path.join(_DATA_SAVE_PATH, "{}_{}.csv".format(save_fname, "test2")), index = False)

def _split_dataset():
    final_interpolate()
    tosplit = pd.read_csv('./data/processed/result.csv')
    data_length = len(tosplit['Time'])-1
    # Split whole dataset to the proper ratio 8 : 2 = Train : Test 
    min_length = 96*(3 + 3)
    data_length = data_length - min_length*2
    train_length = int(data_length*0.8) + min_length

    train_data = {"Time" : [], "Gluco" : []}
    train_data["Time"] = tosplit["Time"][:train_length]
    train_data["Glucose"] = tosplit["Gluco"][:train_length]

    val_data = {"Time" : [], "Gluco" : []}
    val_data["Time"] = tosplit["Time"][train_length:]
    val_data["Gluco"] = tosplit["Gluco"][train_length:]
    return train_data, val_data
if __name__ == "__main__":
    main()