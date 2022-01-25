from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch as T
from datetime import datetime, time
import torch.nn.functional as F

class GetData():
    def __init__(self, trade_frequency):
        filename = '/' + trade_frequency + '_Data_v1/'
        self.filepath = str(Path(__file__).parent) + filename
        self.years = 4
        if trade_frequency == 'Hourly':
            self.years = 6

    def make_DF(self):
        DFNEW = pd.DataFrame()
        for filename in os.listdir(self.filepath):
            DF = pd.read_csv(self.filepath + filename)
            DF = DF[['Local time', 'High', 'Low', 'Close', 'Volume']]
            try:
                DFNEW = DFNEW.merge(DF, left_on='Local time', right_on='Local time')
            except:
                DFNEW = DF
        DFNEW = DFNEW.replace(0, 1)
        return DFNEW

    def make_global_tensor_no_time(self):
        df = self.make_DF()
        arr = np.array(df)[:, 1:] # (time, features)
        Arr = np.array(arr[1:, :] / arr[:-1, :]).astype(float)
        X = T.ones((Arr.shape[-1] // 4, 4, Arr.shape[0]))
        for i in range(arr.shape[-1]):
            j = i // 4
            k = i % 4
            X[j, k, :] = T.tensor(Arr[:, i])
        return X

    def make_global_temporal_tensor(self):
        df = self.make_DF()
        arr = np.array(df)[:, 0]
        M = T.zeros((arr.shape[0], 36 + self.years))
        for i in range(1, arr.shape[0]):
            base = arr[i]

            half_hour = T.tensor(2 * (int(base[11:13]) - 9) + (int(base[14:16]) >= 30))
            if half_hour == 0:
                half_hour += 1
            day = T.tensor(datetime.strptime(base[:10].replace('.', ' '), '%d %m %Y').isoweekday())
            week = T.tensor(abs(int(base[:2]) - 4) // 7)
            month = T.tensor(int(base[3:5]))
            year = T.tensor(int(base[8:10]) - 17)

            half_hour = F.one_hot(half_hour-1, 15)
            day = F.one_hot(day-1, 5)
            week = F.one_hot(week, 4)
            month = F.one_hot(month-1, 12)
            year = F.one_hot(year, self.years)
            M[i, :] = T.cat((half_hour, day, week, month, year))
        return M[1:, :]
