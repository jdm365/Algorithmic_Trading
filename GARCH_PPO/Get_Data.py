from os import curdir
import torch as T
import numpy as np
import pandas as pd
from arch import arch_model
from datetime import timedelta
import datetime
from arch.__future__ import reindexing

class GetData():
    def __init__(self):
        filename_minutely = \
            'GARCH_PPO/AAPL_GARCH_PPO_v1/AAPL.USUSD_Candlestick_5_M_ASK_31.12.2018-31.12.2021.csv'
        filename_daily = \
            'GARCH_PPO/AAPL_GARCH_PPO_v1/AAPL.USUSD_Candlestick_1_D_ASK_01.04.2018-31.12.2021.csv'
        filename_weekly = \
            'GARCH_PPO/AAPL_GARCH_PPO_v1/AAPL.USUSD_Candlestick_1_W_ASK_01.04.2018-31.12.2021.csv'
        self.minutely_DF = pd.read_csv(filename_minutely)
        self.daily_DF = pd.read_csv(filename_daily)
        self.weekly_DF = pd.read_csv(filename_weekly)

        self.create_final_arrays()

    def make_arrays(self):
        minutely_arr = np.array(self.minutely_DF)
        daily_arr = np.array(self.daily_DF)
        weekly_arr = np.array(self.weekly_DF)

        minutely_arr = np.concatenate((minutely_arr[:, 0].reshape((*minutely_arr[:, 0].shape, 1)), minutely_arr[:, 2:]), axis=1)
        daily_arr = np.concatenate((daily_arr[:, 0].reshape((*daily_arr[:, 0].shape, 1)), daily_arr[:, 2:]), axis=1)
        weekly_arr = np.concatenate((weekly_arr[:, 0].reshape((*weekly_arr[:, 0].shape, 1)), weekly_arr[:, 2:]), axis=1)
    
        return minutely_arr, daily_arr, weekly_arr

    def calculate_garch(self):
        closes = self.make_arrays()[1][:, -2].astype(np.float32)
        returns = 100 * (closes[1:] / closes[:-1])
        garch = 31*[0]
        for i in range(returns.shape[-1] - 30):
            model = arch_model(returns[i:i+30])
            fit = model.fit(disp='off')
            pred = fit.forecast(horizon=1)
            garch.append(np.sqrt(pred.variance.values[-1, :][0]))
        garch = np.array((garch), dtype=np.float32)
        return garch.reshape((*garch.shape, 1))

    def create_final_arrays(self):
        X_m, X_d, X_w = self.make_arrays()
        X_d = np.concatenate((X_d, self.calculate_garch()), axis=1)
        self.X_m = X_m
        self.X_d = X_d
        self.X_w = X_w
   
    def min_max_norm(self, tensor):
        X = T.empty_like(tensor)
        for dim in range(tensor.shape[-1]):
            X[:, dim] = (tensor[:, dim] - T.min(tensor[:, dim])) \
                / (T.max(tensor[:, dim]) - T.min(tensor[:, dim]))
        return X

    def create_observation(self, time_step):
        X_m, X_d, X_w = self.X_m, self.X_d, self.X_w

        for i in range(X_d.shape[0]):
            X_d[i, 0] = X_d[i, 0][:10]
        for i in range(X_w.shape[0]):
             X_w[i, 0] = X_w[i, 0][:10]
        date = X_m[time_step, 0][:10]

        datetime_date = datetime.datetime(int(date[-4:]), int(date[3:5]), int(date[0:2]))
        daily_dates = [str((datetime_date - timedelta(i)))[:10] for i in range(-4, 0)]
        daily_dates = [x[8:10] + '.' + x[5:7] + '.' + x[:4] for x in daily_dates]
        for daily_date in daily_dates:
            if len(np.where(X_d == daily_date)[0]) > 0:
                current_daily_idx = int(np.where(X_d == daily_date)[0][0])

        datetime_date = datetime.datetime(int(date[-4:]), int(date[3:5]), int(date[0:2]))
        weekly_dates = [str((datetime_date - timedelta(i)))[:10] for i in range(-4, 4)]
        weekly_dates = [x[8:10] + '.' + x[5:7] + '.' + x[:4] for x in weekly_dates]
        for weekly_date in weekly_dates:
            if len(np.where(X_w == weekly_date)[0]) > 0:
                current_weekly_idx = int(np.where(X_w == weekly_date)[0][0])
        
        minutely_tensor = T.from_numpy(X_m[time_step-48:time_step, 1:].astype(np.float32))
        daily_tensor = T.from_numpy(X_d[current_daily_idx-30:current_daily_idx, 1:].astype(np.float32))
        weekly_tensor = T.from_numpy(X_w[current_weekly_idx-30:current_weekly_idx, 1:].astype(np.float32))

        X_m = self.min_max_norm(minutely_tensor)
        X_d = self.min_max_norm(daily_tensor)
        X_w = self.min_max_norm(weekly_tensor)

        return X_m, X_d, X_w


        