import os
import sys
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import itertools
import random
sys.path.append('/Users/jakemehlman/Desktop/Algorithmic_Trading/pgportfolio/ddpg_files/')
from utils import *

class ActorNetwork(nn.Module):
    def __init__(self, alpha, cl1_dims, cl2_dims, n_actions, lookback_window, name, chkpt_dir='trained_models'):
        super(ActorNetwork, self).__init__()
        self.cl1_dims = cl1_dims
        self.cl2_dims = cl2_dims
        self.N_ASSETS = n_actions
        try:
            os.mkdir(os.path.join('/Users/jakemehlman/Desktop/Algorithmic_Trading/pgportfolio/NewAttempt/', chkpt_dir))
            self.checkpoint_file = os.path.join(os.mkdir(os.path.join('/Users/jakemehlman/Desktop/Algorithmic_Trading/pgportfolio/NewAttempt/', chkpt_dir)), name + '_ddpg')
        except OSError as error:
            self.checkpoint_file = os.path.join(os.path.join('/Users/jakemehlman/Desktop/Algorithmic_Trading/pgportfolio/NewAttempt/', chkpt_dir), name + '_ddpg')
        
        self.cl1 = nn.Conv2d(in_channels=3, out_channels=self.cl1_dims, kernel_size=(1,3))
        self.bn1 = nn.BatchNorm2d(self.cl1_dims)

        self.cl2 = nn.Conv2d(in_channels=self.cl1_dims, out_channels=self.cl2_dims, kernel_size=(1, lookback_window - 2))
        self.bn2 = nn.BatchNorm2d(self.cl2_dims)

        self.mu = nn.Conv2d(in_channels=self.cl2_dims + 1, out_channels=1, kernel_size=1) # Plus one feature for previous action
        self.cash_bias = T.ones(1, 1, 1, 1)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation, last_action):
        x = F.relu(self.bn1(self.cl1(observation)))
        x = F.relu(self.bn2(self.cl2(x)))
        last_action = last_action[:, 1:, 0].reshape((len(observation[:,0,0,0]), 1, self.N_ASSETS, 1))
        x = T.cat((last_action, x), dim=1) # Add last_action

        CB = T.ones(len(observation[:,0,0,0]), 1, 1, 1) * self.cash_bias

        softmax = nn.Softmax(dim=2) # Asset dimension
        x = self.mu(x)
        action = softmax(T.cat((CB, x), dim=2)) 
        return action

    def save_checkpoint(self):
        print("...Saving checkpoint...")
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print("...Loading checkpoint...")
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, beta, cl1_dims, cl2_dims, n_actions, lookback_window, name, chkpt_dir='trained_models'):
        super(CriticNetwork, self).__init__()
        self.cl1_dims = cl1_dims
        self.cl2_dims = cl2_dims
        self.N_ASSETS = n_actions
        try:
            os.mkdir(os.path.join('/Users/jakemehlman/Desktop/Algorithmic_Trading/pgportfolio/NewAttempt/', chkpt_dir))
            self.checkpoint_file = os.path.join(os.mkdir(os.path.join('/Users/jakemehlman/Desktop/Algorithmic_Trading/pgportfolio/NewAttempt/', chkpt_dir)), name + '_ddpg')
        except OSError as error:
            self.checkpoint_file = os.path.join(os.path.join('/Users/jakemehlman/Desktop/Algorithmic_Trading/pgportfolio/NewAttempt/', chkpt_dir), name + '_ddpg')
        
        self.cl1 = nn.Conv2d(in_channels=3, out_channels=self.cl1_dims, kernel_size=(1,3))
        self.bn1 = nn.BatchNorm2d(self.cl1_dims)

        self.cl2 = nn.Conv2d(in_channels=self.cl1_dims, out_channels=self.cl2_dims, kernel_size=(1, lookback_window-2))
        self.bn2 = nn.BatchNorm2d(self.cl2_dims)

        self.action_value = nn.Conv2d(in_channels=1, out_channels=self.cl2_dims+1, kernel_size=1)
        self.q = nn.Conv2d(in_channels=self.cl2_dims+1, out_channels=1, kernel_size=1)

        self.cash_bias = T.ones(1, 1, 1, 1)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation, last_action, action):
        state_value = F.relu(self.bn1(self.cl1(observation)))
        state_value = F.relu(self.bn2(self.cl2(state_value))) # [1, 20, 11, 1]

        last_action = last_action[:, 1:, 0].reshape(len(observation[:,0,0,0]), 1, self.N_ASSETS, 1)
        state_value = T.cat((last_action, state_value), dim=1) # Add last_action; [1, 21, 11, 1]
        
        action = T.reshape(T.flatten(action)[64:], (len(observation[:,0,0,0]), 1, self.N_ASSETS, 1))
        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        self.cash_bias = T.ones(len(observation[:,0,0,0]), 1, 1, 1) * self.cash_bias
        
        state_action_value = T.cat((self.cash_bias, state_action_value), dim=2)

        return state_action_value

    def save_checkpoint(self):
        print("...Saving checkpoint...")
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        print("...Loading checkpoint...")
        self.load_state_dict(T.load(self.checkpoint_file))


class ReplayBuffer(object):
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def store_transition(self, state, last_action, action, reward, next_state, done):
        experience = (state, last_action, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample_buffer(self, batch_size):
        (state_batch, last_action_batch, action_batch, reward_batch, next_state_batch, 
            done_batch) = ([], [], [], [], [], [])
        
        i = random.randint(0, self.__len__() - batch_size)
        batch = list(itertools.islice(self.buffer, i, i + batch_size))
        # Sample recent events more often
        if self.__len__() > 200:
            sample_bias = 5e-2
            k = np.random.geometric(p=sample_bias, size=1)[0]
            i = int(self.__len__() + 1 - k - batch_size)
            if i + batch_size > self.__len__():
                i = self.__len__()
            batch = list(itertools.islice(self.buffer, i, i + batch_size))

        for experience in batch:
            state, last_action, action, reward, next_state, done = experience
            state_batch.append(state)
            last_action_batch.append(last_action)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        
        return state_batch, last_action_batch, action_batch, reward_batch, next_state_batch, done_batch
    
    def __len__(self):
        return len(self.buffer)

