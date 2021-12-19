import os
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import itertools
import random
from torch.distributions import Normal
from pgportfolio.DDPG.utils import *

class ActorNetwork(nn.Module):
    def __init__(self, alpha, cl1_dims, cl2_dims, n_actions, lookback_window, name, 
        chkpt_dir='trained_models_algo_trade'):
        super(ActorNetwork, self).__init__()
        self.cl1_dims = cl1_dims
        self.cl2_dims = cl2_dims
        self.N_ASSETS = n_actions
        try:
            os.mkdir(os.path.join(PATH, chkpt_dir))
            self.checkpoint_file = os.path.join(os.mkdir(os.path.join(PATH, chkpt_dir)), name + '_sac')
        except OSError as error:
            self.checkpoint_file = os.path.join(os.path.join(PATH, chkpt_dir), name + '_sac')
        self.reparam_noise = 1e-6

        self.cl1 = nn.Conv2d(in_channels=4, out_channels=self.cl1_dims, kernel_size=(1,3)) #Include volume this time
        self.bn1 = nn.BatchNorm2d(self.cl1_dims)

        self.cl2 = nn.Conv2d(in_channels=self.cl1_dims, out_channels=self.cl2_dims, kernel_size=(1, lookback_window - 2))
        self.bn2 = nn.BatchNorm2d(self.cl2_dims)

        self.mu = nn.Conv2d(in_channels=self.cl2_dims + 1, out_channels=1, kernel_size=1) # Plus one feature for previous action
        self.sigma = nn.Conv2d(in_channels=self.cl2_dims + 1, out_channels=1, kernel_size=1) # Plus one feature for previous action
        self.cash_bias = T.ones(1, 1, 1, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        
        self.to(self.device)

    def forward(self, observation, last_action):
        prob = F.relu(self.bn1(self.cl1(observation)))
        prob = F.relu(self.bn2(self.cl2(prob)))
        last_action = last_action[:, 1:, 0].reshape((len(observation[:,0,0,0]), 1, self.N_ASSETS, 1))
        prob = T.cat((last_action, prob), dim=1) # Add last_action

        CB = T.ones(len(observation[:,0,0,0]), 1, 1, 1) * self.cash_bias

        # Apply softmax at sample function
        mu = self.mu(prob)
        sigma = self.sigma(prob)
        mu = T.cat((CB, mu), dim=2)
        sigma = T.cat((CB, sigma), dim=2)
        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        return mu, sigma

    def sample_normal(self, observation, last_action, reparameterize=True):
        mu, sigma = self.forward(observation, last_action)
        probabilities = Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        softmax = nn.Softmax(dim=2) # Asset dimension
        action = softmax(actions).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= T.log(1-action.pow(2) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(self, beta, cl1_dims, cl2_dims, n_assets, lookback_window, name, 
        chkpt_dir='trained_models_algo_trade'):
        super(ValueNetwork, self).__init__()
        self.cl1_dims = cl1_dims
        self.cl2_dims = cl2_dims
        self.N_ASSETS = n_assets
        try:
            os.mkdir(os.path.join(PATH, chkpt_dir))
            self.checkpoint_file = os.path.join(os.mkdir(os.path.join(PATH, chkpt_dir)), name + '_sac')
        except OSError as error:
            self.checkpoint_file = os.path.join(os.path.join(PATH, chkpt_dir), name + '_sac')

        self.cl1 = nn.Conv2d(in_channels=4, out_channels=self.cl1_dims, kernel_size=(1,3)) #Include volume this time
        self.bn1 = nn.BatchNorm2d(self.cl1_dims)

        self.cl2 = nn.Conv2d(in_channels=self.cl1_dims, out_channels=self.cl2_dims, kernel_size=(1, lookback_window-2))
        self.bn2 = nn.BatchNorm2d(self.cl2_dims)

        self.v = nn.Conv2d(in_channels=self.cl2_dims+1, out_channels=1, kernel_size=1)

        self.cash_bias = T.ones(1, 1, 1, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, observation, last_action):
        state_value = F.relu(self.bn1(self.cl1(observation)))
        state_value = F.relu(self.bn2(self.cl2(state_value)))

        last_action = last_action[:, 1:, 0].reshape(len(observation[:,0,0,0]), 1, self.N_ASSETS, 1)
        state_value = T.cat((last_action, state_value), dim=1) # Add last_action

        value = self.v(state_value)

        self.cash_bias = T.ones(len(observation[:,0,0,0]), 1, 1, 1) * self.cash_bias
        
        value = T.cat((self.cash_bias, value), dim=2)

        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    def __init__(self, beta, cl1_dims, cl2_dims, n_actions, lookback_window, name, 
        chkpt_dir='trained_models_algo_trade'):
        super(CriticNetwork, self).__init__()
        self.cl1_dims = cl1_dims
        self.cl2_dims = cl2_dims
        self.N_ASSETS = n_actions
        try:
            os.mkdir(os.path.join(PATH, chkpt_dir))
            self.checkpoint_file = os.path.join(os.mkdir(os.path.join(PATH, chkpt_dir)), name + '_sac')
        except OSError as error:
            self.checkpoint_file = os.path.join(os.path.join(PATH, chkpt_dir), name + '_sac')

        self.cl1 = nn.Conv2d(in_channels=4, out_channels=self.cl1_dims, kernel_size=(1,3)) #Include volume this time
        self.bn1 = nn.BatchNorm2d(self.cl1_dims)

        self.cl2 = nn.Conv2d(in_channels=self.cl1_dims, out_channels=self.cl2_dims, kernel_size=(1, lookback_window-2))
        self.bn2 = nn.BatchNorm2d(self.cl2_dims)

        self.action_value = nn.Conv2d(in_channels=1, out_channels=self.cl2_dims+1, kernel_size=1)
        self.q = nn.Conv2d(in_channels=self.cl2_dims+1, out_channels=1, kernel_size=1)

        self.cash_bias = T.ones(1, 1, 1, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, observation, last_action, action):
        state_value = F.relu(self.bn1(self.cl1(observation)))
        state_value = F.relu(self.bn2(self.cl2(state_value)))

        last_action = last_action[:, 1:, 0].reshape(len(observation[:,0,0,0]), 1, self.N_ASSETS, 1)
        state_value = T.cat((last_action, state_value), dim=1) # Add last_action
        
        action = T.reshape(T.flatten(action)[64:], (len(observation[:,0,0,0]), 1, self.N_ASSETS, 1))
        action_value = F.relu(self.action_value(action))
        state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        self.cash_bias = T.ones(len(observation[:,0,0,0]), 1, 1, 1) * self.cash_bias
        
        state_action_value = T.cat((self.cash_bias, state_action_value), dim=2)

        return state_action_value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
    
    def load_checkpoint(self):
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
        sample_bias = 5e-3
        k = np.random.geometric(p=sample_bias, size=1)[0]
        i = int(self.__len__() + 1 - k - batch_size)
        if i + batch_size > self.__len__() or i + batch_size < 0:
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