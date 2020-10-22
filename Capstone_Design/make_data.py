import os
import pandas as pd
import numpy as np

_DATA_SAVE_PATH = "data\\processed"
_DATA_LOAD_PATH = "data\\unprocessed"
_FILE_LOAD_NAME = "70man(3).csv"
_FILE_SAVE_NAME = "70man"

class PreProcessing(object):
    def __init__(self, data_load_path, file_load_name, data_save_path, file_save_name):
        super().__init__()
        self.load_path = data_load_path
        self.load_fname = file_load_name
        self.save_path = data_save_path
        self.save_fname = file_save_name
        self.cur_fpath = os.path.join(self.load_path, self.load_fname)
        self.input_day = 3
        self.predict_day = 3
        self.minutes = ["00", "15", "30", "45"]
        self.compare_minutes = [7.5, 22.5, 37.5, 52.5]
        self.new_data = {"Time" : [], "Glucose" : []}

    def __call__(self):
        # Define how many day's data we gonna make
        dataframe = pd.read_csv(self.cur_fpath, header = 0, names = ["Time", "Glucose"])
<<<<<<< HEAD
        create_days = int(len(dataframe)/96)
        days = create_days*96
        print(np.array(dataframe['Glucose'][:]).max())
        print(np.array(dataframe['Glucose'][:]).min())
=======
        end_point = len(dataframe)

>>>>>>> 37d60bcea9054c300ede78e80713fcae00684393
        new_data_idx = 0
        idx = 0
        
        while(True):
            # Get data from unprocessed file
            cur_glucose = int(dataframe["Glucose"][idx])
            cur_hour, cur_minutes = self._get_time(dataframe["Time"][idx], True)
            new_hour, new_minutes = self._calcualte_newtime(cur_hour, cur_minutes, idx)

            # Check whether new data list is empty or not
            if not self.new_data["Time"]:
                self._add_data(new_hour, new_minutes, cur_glucose)
                idx += 1
                continue
            else :
                prev_hour, prev_minutes = self._get_time(self.new_data["Time"][new_data_idx], False)
                if self._compare_time(prev_hour, prev_minutes, new_hour, new_minutes):
                    self._add_data(new_hour, new_minutes, cur_glucose)
                    idx += 1
                    new_data_idx += 1
                else:
                    new_hour, new_minutes = self._calcualte_nexttime(prev_hour, prev_minutes)
                    self._add_data(new_hour, new_minutes, np.nan)
                    new_data_idx += 1
            if idx == end_point : break

        train_new_data, val_new_data = self._split_dataset(len(self.new_data["Time"]))
        # Save train dataset
        save_train_df = pd.DataFrame(train_new_data, columns = ["Time", "Glucose"])
        save_train_df = save_train_df.interpolate(method = "values")
        save_train_df.to_csv(os.path.join(_DATA_SAVE_PATH, "{}_{}.csv".format(self.save_fname, "train")), index = False)

        # Save val dataset
        save_val_df = pd.DataFrame(val_new_data, columns = ["Time", "Glucose"])
        save_val_df = save_val_df.interpolate(method = "values")
        save_val_df.to_csv(os.path.join(_DATA_SAVE_PATH, "{}_{}.csv".format(self.save_fname, "test")), index = False)

    def _split_dataset(self, data_length):
        # Split whole dataset to the proper ratio 8 : 2 = Train : Test 
        min_length = 96*(self.input_day + self.predict_day)
        data_length = data_length - min_length*2
        train_length = int(data_length*0.8) + min_length

        train_data = {"Time" : [], "Glucose" : []}
        train_data["Time"] = self.new_data["Time"][:train_length]
        train_data["Glucose"] = self.new_data["Glucose"][:train_length]

        val_data = {"Time" : [], "Glucose" : []}
        val_data["Time"] = self.new_data["Time"][train_length:]
        val_data["Glucose"] = self.new_data["Glucose"][train_length:]
        return train_data, val_data

    def _add_data(self, new_hour, new_minutes, new_glucose):
        new_time = new_hour + ":" + new_minutes
        self.new_data["Time"].append(new_time)
        self.new_data["Glucose"].append(new_glucose)

    def _get_time(self, time, flag):
        if flag :
            # When I get time data from unprocessed
            cur_time = time.split(" ")[-1].split(":")
        else :
            # When I get time data from new_data
            cur_time = time.split(":")
        return cur_time[0], cur_time[1]

    def _compare_time(self, prev_hour, prev_minutes, cur_hour, cur_minutes):
        prev_hour = int(prev_hour)
        prev_minutes = int(prev_minutes)
        cur_hour = int(cur_hour)
        cur_minutes = int(cur_minutes)

        if prev_hour == 23 and cur_hour == 0 :
            prev_hour = prev_hour*60
            cur_hour = 24*60
        else : 
            prev_hour = prev_hour*60
            cur_hour = cur_hour*60

        previous = prev_hour + prev_minutes
        current = cur_hour + cur_minutes

        if (current - previous) == 15 :
            return True
        else :
            return False

    def _calcualte_nexttime(self, prev_hour, prev_minutes):
        index = self.minutes.index(prev_minutes)
        if index == 3 :
            if int(prev_hour) == 23 : 
                new_hour = self.minutes[0]
                new_minutes = self.minutes[0]
            else : 
                new_hour = str(int(prev_hour)+1)
                new_minutes = self.minutes[0]
        else:
            new_hour = str(prev_hour)
            new_minutes = self.minutes[index+1]
        return new_hour, new_minutes

    def _calcualte_newtime(self, cur_hour, cur_minutes, idx):
        """
        Index means that cur_min exist in which section, when we divide minutes to 4 sections
        [0 ~ 15) : 0
        [15 ~ 30) : 1
        [30 ~ 45) : 2
        [45 ~ 60) : 3
        """
        cur_hour = int(cur_hour)
        cur_minutes = int(cur_minutes)
        index = int(cur_minutes / 15)

        if index == 3 :         # When minutes are 45
            if cur_minutes >= self.compare_minutes[index] : 
                if cur_hour == 23 : 
                    new_hour = self.minutes[0]
                    new_minutes = self.minutes[0]
                else : 
                    new_hour = str(cur_hour+1)
                    new_minutes = self.minutes[0]
            else : 
                new_hour = str(cur_hour)
                new_minutes = self.minutes[index]
        else:
            if cur_minutes >= self.compare_minutes[index] : 
                new_hour = str(cur_hour)
                new_minutes = self.minutes[index+1]
            else : 
                new_hour = str(cur_hour)
                new_minutes = self.minutes[index]
        return new_hour, new_minutes

if __name__ == "__main__":
    pp = PreProcessing(_DATA_LOAD_PATH, _FILE_LOAD_NAME, _DATA_SAVE_PATH, _FILE_SAVE_NAME)
    pp()
