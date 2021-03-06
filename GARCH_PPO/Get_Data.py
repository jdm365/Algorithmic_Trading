from os import curdir
from sympy import convolution
import torch as T
import numpy as np
import pandas as pd
from arch import arch_model
from datetime import timedelta
import datetime
from arch.__future__ import reindexing
from zmq import device
from pathlib import Path

class GetData():
    def __init__(self, convolutional=False, recurrent=True, ticker='AAPL'):
        self.filepath = str(Path(__file__).parent)
        self.convolutional = convolutional
        self.recurrent = recurrent
        if ticker == 'AAPL':
            filename_minutely = self.filepath + \
                '/AAPL_GARCH_PPO_v1/AAPL.USUSD_Candlestick_5_M_ASK_31.12.2018-31.12.2021.csv'
            filename_daily =  self.filepath + \
                '/AAPL_GARCH_PPO_v1/AAPL.USUSD_Candlestick_1_D_ASK_01.04.2018-31.12.2021.csv'
            filename_weekly =  self.filepath + \
                '/AAPL_GARCH_PPO_v1/AAPL.USUSD_Candlestick_1_W_ASK_01.04.2018-31.12.2021.csv'
        elif ticker == 'TSLA':
            filename_minutely = self.filepath + \
                '/TSLA_GARCH_PPO_v1/TSLA.USUSD_Candlestick_5_M_ASK_31.12.2018-31.12.2021.csv'
            filename_daily =  self.filepath + \
                '/TSLA_GARCH_PPO_v1/TSLA.USUSD_Candlestick_1_D_ASK_31.12.2017-31.12.2021.csv'
            filename_weekly =  self.filepath + \
                '/TSLA_GARCH_PPO_v1/TSLA.USUSD_Candlestick_1_W_ASK_31.12.2017-31.12.2021.csv'
        elif ticker == '.INX':
            filename_minutely = self.filepath + \
                '/SNP_GARCH_PPO_v1/USA500.IDXUSD_Candlestick_5_M_ASK_31.12.2018-31.12.2021.csv'
            filename_daily =  self.filepath + \
                '/SNP_GARCH_PPO_v1/USA500.IDXUSD_Candlestick_1_D_ASK_31.12.2017-31.12.2021.csv'
            filename_weekly =  self.filepath + \
                '/SNP_GARCH_PPO_v1/USA500.IDXUSD_Candlestick_1_W_ASK_31.12.2017-31.12.2021.csv'
        elif ticker == '.INX2':
            filename_minutely = self.filepath + \
                '/SNP_GARCH_PPO_v2/USA500.IDXUSD_Candlestick_1_Hour_ASK_31.12.2014-31.12.2021.csv'
            filename_daily =  self.filepath + \
                '/SNP_GARCH_PPO_v2/USA500.IDXUSD_Candlestick_1_D_ASK_31.12.2013-31.12.2021.csv'
            filename_weekly =  self.filepath + \
                '/SNP_GARCH_PPO_v2/USA500.IDXUSD_Candlestick_1_W_ASK_31.12.2013-31.12.2021.csv'

        minutely_DF = pd.read_csv(filename_minutely)
        daily_DF = pd.read_csv(filename_daily)
        weekly_DF = pd.read_csv(filename_weekly)
        
        if ticker == 'AAPL':
            minutely_idx = minutely_DF.index[minutely_DF['Gmt time'] == '28.08.2020 19:55:00.000'][0]
            daily_idx = daily_DF.index[daily_DF['Gmt time'] == '27.08.2020 21:00:00.000'][0]
            weekly_idx = weekly_DF.index[weekly_DF['Gmt time'] == '23.08.2020 21:00:00.000'][0]
            
            minutely_DF.iloc[:minutely_idx+1, 1:5] = minutely_DF.iloc[:minutely_idx+1, 1:5] / 4
            daily_DF.iloc[:daily_idx+1, 1:5] = daily_DF.iloc[:daily_idx+1, 1:5] / 4
            weekly_DF.iloc[:weekly_idx+1, 1:5] = weekly_DF.iloc[:weekly_idx+1, 1:5] / 4

            minutely_DF.iloc[:minutely_idx+1, 5] = minutely_DF.iloc[:minutely_idx+1, 5] * 4
            daily_DF.iloc[:daily_idx+1, 5] = daily_DF.iloc[:daily_idx+1, 5] * 4
            weekly_DF.iloc[:weekly_idx+1, 5] = weekly_DF.iloc[:weekly_idx+1, 5] * 4
        elif ticker == 'TSLA':
            minutely_idx = minutely_DF.index[minutely_DF['Gmt time'] == '28.08.2020 19:55:00.000'][0]
            daily_idx = daily_DF.index[daily_DF['Gmt time'] == '27.08.2020 21:00:00.000'][0]
            weekly_idx = weekly_DF.index[weekly_DF['Gmt time'] == '23.08.2020 21:00:00.000'][0]
            
            minutely_DF.iloc[:minutely_idx+1, 1:5] = minutely_DF.iloc[:minutely_idx+1, 1:5] / 5
            daily_DF.iloc[:daily_idx+1, 1:5] = daily_DF.iloc[:daily_idx+1, 1:5] / 5
            weekly_DF.iloc[:weekly_idx+1, 1:5] = weekly_DF.iloc[:weekly_idx+1, 1:5] / 5

            minutely_DF.iloc[:minutely_idx+1, 5] = minutely_DF.iloc[:minutely_idx+1, 5] * 5
            daily_DF.iloc[:daily_idx+1, 5] = daily_DF.iloc[:daily_idx+1, 5] * 5
            weekly_DF.iloc[:weekly_idx+1, 5] = weekly_DF.iloc[:weekly_idx+1, 5] * 5
        elif ticker == '.INX' or ticker == '.INX2':
            pass

        self.minutely_DF = minutely_DF
        self.daily_DF = daily_DF
        self.weekly_DF = weekly_DF
        self.n_features = 4

        self.create_final_arrays()
        self.device = 'cpu' #T.device('cuda:0' if T.cuda.is_available() else 'cpu')

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
        if tensor.shape[-1] > self.n_features:
            X[:, self.n_features] = tensor[:, self.n_features]
        for dim in range(self.n_features):
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
                current_daily_idx = int(np.where(X_d == daily_date)[0][-1])

        datetime_date = datetime.datetime(int(date[-4:]), int(date[3:5]), int(date[0:2]))
        weekly_dates = [str((datetime_date - timedelta(i)))[:10] for i in range(-4, 4)]
        weekly_dates = [x[8:10] + '.' + x[5:7] + '.' + x[:4] for x in weekly_dates]
        for weekly_date in weekly_dates:
            if len(np.where(X_w == weekly_date)[0]) > 0:
                current_weekly_idx = int(np.where(X_w == weekly_date)[0][-1])
        
        minutely_tensor = T.from_numpy(X_m[time_step-48:time_step, 1:].astype(np.float32))
        daily_tensor = T.from_numpy(X_d[current_daily_idx-30:current_daily_idx, 1:].astype(np.float32))
        weekly_tensor = T.from_numpy(X_w[current_weekly_idx-30:current_weekly_idx, 1:].astype(np.float32))

        X_m = self.min_max_norm(minutely_tensor).to(self.device)
        X_d = self.min_max_norm(daily_tensor).to(self.device)
        X_w = self.min_max_norm(weekly_tensor).to(self.device)

        if self.recurrent:
            X_m = X_m.permute(1, 0).contiguous()
            X_d = X_d.permute(1, 0).contiguous()
            X_w = X_w.permute(1, 0).contiguous()

            if len(X_m.shape) != 4:
                X_m = X_m.reshape(1, *X_m.shape)
                X_d = X_d.reshape(1, *X_d.shape)
                X_w = X_w.reshape(1, *X_w.shape)
            else:
                X_m = X_m.reshape(*X_m.shape, 1)
                X_d = X_d.reshape(*X_d.shape, 1)
                X_w = X_w.reshape(*X_w.shape, 1)

        if self.convolutional:
            X_m = X_m.permute(1, 0).contiguous()
            X_d = X_d.permute(1, 0).contiguous()
            X_w = X_w.permute(1, 0).contiguous()

            if len(X_m.shape) != 4:
                X_m = X_m.reshape(1, *X_m.shape, 1)
                X_d = X_d.reshape(1, *X_d.shape, 1)
                X_w = X_w.reshape(1, *X_w.shape, 1)
            else:
                X_m = X_m.reshape(*X_m.shape, 1)
                X_d = X_d.reshape(*X_d.shape, 1)
                X_w = X_w.reshape(*X_w.shape, 1)
        return X_m, X_d, X_w


        
