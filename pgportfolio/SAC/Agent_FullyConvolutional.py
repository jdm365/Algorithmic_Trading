import os
import torch as T
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from Actor_CriticFullyConvolutional import ValueNetwork, ActorNetwork, CriticNetwork, ReplayBuffer
from pgportfolio.SAC.Trading_Env import TradingEnv, DataFeatures
from pgportfolio.DDPG.utils import *

class Agent():
    def __init__(self, env, alpha, beta, cl1_dims=3, cl2_dims=22, 
        gamma=.99, tau=.005, max_memory_size=50000, reward_scale=2):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_memory_size)
        self.batch_size = env.batch_size
        self.n_actions = env.N_ASSETS

        self.actor = ActorNetwork(alpha=alpha, cl1_dims=cl1_dims, cl2_dims=cl2_dims, n_actions=self.n_actions, 
            lookback_window=self.batch_size, name='Actor')

        self.critic_1 = CriticNetwork(beta=beta, cl1_dims=cl1_dims, cl2_dims=cl2_dims, n_actions=self.n_actions, 
            lookback_window=self.batch_size, name='Critic_1')

        self.critic_2 = CriticNetwork(beta=beta, cl1_dims=cl1_dims, cl2_dims=cl2_dims, n_actions=self.n_actions, 
            lookback_window=self.batch_size, name='Critic_2')

        self.value = ValueNetwork(beta=beta, cl1_dims=cl1_dims, cl2_dims=cl2_dims, n_assets=self.n_actions, 
            lookback_window=self.batch_size, name='Value')
        
        self.target_value = ValueNetwork(beta=beta, cl1_dims=cl1_dims, cl2_dims=cl2_dims, n_assets=self.n_actions, 
            lookback_window=self.batch_size, name='Target_Value')

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def choose_action(self, observation, last_action):
        state = Variable(T.from_numpy(observation).float().unsqueeze(0))[0]
        last_action = Variable(T.from_numpy(last_action).float().unsqueeze(0))

        actions, _ = self.actor.sample_normal(state, last_action, reparameterize=False)
        print(actions.clone().detach().numpy()[0,0])
        return actions.clone().detach().numpy()[0,0]

    def update(self, train_length):
        if self.memory.__len__() < self.batch_size:
            return

        state, last_action, action, reward, next_state, done = self.memory.sample_buffer(self.batch_size)
        state = T.squeeze(T.tensor(state, dtype=T.float32, requires_grad=True)).to(self.critic_1.device)
        last_action = T.tensor(last_action, dtype=T.float32, requires_grad=True).to(self.critic_1.device)
        action = T.tensor(action, dtype=T.float32, requires_grad=True).to(self.critic_1.device)
        # Make sure rewards are discounted for batch_length
        reward = T.tensor([rwrd / self.batch_size for rwrd in reward], dtype=T.float32, requires_grad=True).to(self.critic_1.device)
        next_state = T.squeeze(T.tensor(next_state, dtype=T.float32, requires_grad=True)).to(self.critic_1.device)
        done = T.tensor(done, dtype=T.bool).to(self.critic_1.device)

        value = self.value(state, last_action)
        target_value = self.target_value(next_state, action)
        target_value[done] = 0.0

        actions, log_probs = self.actor.sample_normal(state, last_action, reparameterize=False)
        q1_new_policy = self.critic_1.forward(state, last_action, actions)
        q2_new_policy = self.critic_2.forward(state, last_action, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)

        value_target = critic_value - log_probs # Different from target_value (network)
        value_loss = .5 * F.mse_loss(value, value_target)

        actions, log_probs = self.actor.sample_normal(state, last_action, reparameterize=True)
        q1_new_policy = self.critic_1.forward(state, last_action, actions)
        q2_new_policy = self.critic_2.forward(state, last_action, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)
        actor_loss = T.mean(log_probs - critic_value)

        q_hat = self.scale * T.reshape(reward, (self.batch_size, 1, 1, 1)) + self.gamma * target_value
        q1_old_policy = self.critic_1.forward(state, last_action, actions)
        q2_old_policy = self.critic_2.forward(state, last_action, actions)
        critic_1_loss = .5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = .5 * F.mse_loss(q2_old_policy, q_hat)
        critic_loss = critic_1_loss + critic_2_loss

        self.actor.optimizer.zero_grad()
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        self.value.optimizer.zero_grad()

        actor_loss.backward(retain_graph=True)
        critic_loss.backward(retain_graph=True)
        value_loss.backward()

        self.actor.optimizer.step()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        self.value.optimizer.step()
        
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        value_state_dict = self.value.state_dict()
        target_value_state_dict = self.target_value.state_dict()

        # update target networks
        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('...Saving Models...')
        self.actor.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()

    def load_models(self):
        print('...Loading Models...')
        self.actor.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()

