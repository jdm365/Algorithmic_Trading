import numpy as np
import torch as T
import torch.nn.functional as F
import os
from pgportfolio.DDPG.utils import *
from pgportfolio.Market_Data.Finnhub import GetTrainData


class DataFeatures:
    def __init__(self, date):
        # Previous time step's portfolio weights
        train_data = GetTrainData(date)
        #test_data = GetTestData(date)
        self.price_features, self.y = train_data.loadTrainData()
        #self.price_features, self.y = test_data.loadTestData()
        
        self.batch_size = BATCH_SIZE
        self.N_ASSETS = len(self.price_features[0,1:,0])
        self.TIME_STEPS = len(self.price_features[0,0,:]) - BATCH_SIZE

    # How the portfolio weights changed in last period due to asset movement
    def calculateWPrime(self, time_step, last_action):
        last_action = T.reshape(last_action, (1, 1+self.N_ASSETS, 1))
        Y = T.reshape(self.y[:,:,time_step], (1, 1+self.N_ASSETS, 1))
        w_prime = T.reshape(((Y * last_action) / T.tensordot(T.flatten(Y), T.flatten(last_action), dims=1)), (1, 1+self.N_ASSETS, 1))
        return w_prime

    def calculateCommisionsFactor(self, time_step, action, last_action):
        delta = 5e-3
        c_factor = .0025
        done = False
        action = action.detach().clone()
        w_prime = self.calculateWPrime(time_step, last_action).detach().clone()[0]
        mu = c_factor * T.sum(T.abs(w_prime - action))
        while not done:
            mu_ = (1 - c_factor * w_prime[0] - (2*c_factor - c_factor**2) * T.sum(F.relu(w_prime[1:] - mu * action[1:]))) / (1 - c_factor * action[0])
            if T.abs(mu_-mu) < delta:
                done=True
            else:
                mu = mu_
        return mu_

    def computePriceTensor(self, time_step):
        X_ = T.zeros(size=(4, self.N_ASSETS, self.batch_size), dtype=T.float32)
        closes_ = T.flatten(self.price_features[1, 1:, time_step+2]).cpu().detach().numpy()
        volume_closes_ = T.flatten(self.price_features[0, 1:, time_step+2]).cpu().detach().numpy()

        for idx, close_ in enumerate(closes_):
            X_[1:, idx, :] = self.price_features[1:, 1 + idx, (time_step + 2 - self.batch_size): time_step + 2] / close_
        
        for idx, volume_close_ in enumerate(volume_closes_):
            X_[0, idx, :] = self.price_features[0, 1 + idx, (time_step + 2 - self.batch_size): time_step + 2] / volume_close_

        X_ = T.reshape(X_, (1, *X_.size()))
        
        return X_

class TradingEnv:
    def __init__(self, date):
        train_data = GetTrainData(date)
        #test_data = GetTestData(date)
        self.date = date
        self.price_features, self.y = train_data.loadTrainData() # Train Data
        #self.price_features, self.y = test_data.loadTestData() # Test Data
        self.batch_size = BATCH_SIZE
        self.time_step = BATCH_SIZE 
        self.N_ASSETS = len(self.price_features[0,1:,0])
        self.TIME_STEPS = len(self.price_features[0,0,:]) - BATCH_SIZE
        self.features = DataFeatures(self.date)                                                 

    def step(self, action, last_action, train_batch_window):
        # Create current price tensor and next price tensor
        X_ = self.features.computePriceTensor(self.time_step)

        # w_prime = features.calculateWPrime(self.time_step, action) # Not relevant without commisions
        action = T.tensor(action, dtype=T.float32)
        last_action = T.tensor(last_action, dtype=T.float32)
        mu = self.features.calculateCommisionsFactor(self.time_step, action, last_action)
        reward = T.log(mu * T.dot(T.flatten(self.y[:, :, self.time_step]), T.flatten(last_action))) / train_batch_window
        done = False
        if self.time_step == self.TIME_STEPS - 1:
            done = True

        self.time_step += 1
        last_action = action

        return X_.detach().numpy(), last_action.detach().numpy(), reward.detach().numpy(), done # Added last action

    def buyAndHoldReturns(self, buy_and_hold_action):
        return T.dot(T.flatten(self.y[:, :, self.time_step]), T.flatten(T.tensor(buy_and_hold_action, dtype=T.float32))).detach().numpy()

    def reset(self, time_step):
        self.time_step = time_step + self.batch_size - 1
        state = DataFeatures(self.date).computePriceTensor(self.time_step)
        last_action = np.random.rand(12,1) #Random initialization for exploration
        last_action = np.exp(last_action) / np.sum(np.exp(last_action), axis=0)

        return state.detach().numpy(), last_action


        