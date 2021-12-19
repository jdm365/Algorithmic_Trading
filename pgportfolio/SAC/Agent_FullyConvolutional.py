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
        gamma=.99, tau=.01, max_memory_size=50000, reward_scale=2):
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

        return actions.cpu().detach().numpy()[0,0]

    def update(self, train_length):
        if self.memory.__len__() < self.batch_size:
            return
        
        states, last_actions, actions, rewards, next_states, dones = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(states, dtype=T.float32, requires_grad=True).to(self.critic_1.device)
        last_actions = T.tensor(last_actions, dtype=T.float32, requires_grad=True).to(self.critic_1.device)
        actions = T.tensor(actions, dtype=T.float32, requires_grad=True).to(self.critic_1.device)
        # Make sure rewards are discounted for batch_length
        rewards = T.tensor([reward * train_length / self.batch_size for reward in rewards], dtype=T.float32, requires_grad=True).to(self.critic_1.device)
        next_states = T.tensor(next_states, dtype=T.float32, requires_grad=True).to(self.critic_1.device)
        dones = T.tensor(dones, dtype=T.float32).to(self.critic_1.device)

        values = self.value(states, last_actions)
        target_values = self.target_value(next_states, actions)
        target_values[dones] = 0.0

        actions, log_probs = self.actor.sample_normal(states, reparameterize=False)
        q1_new_policy = self.critic_1.forward(states, last_actions, actions)
        q2_new_policy = self.critic_2.forward(states, last_actions, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)

        self.value.optimizer.zero_grad()
        value_target = critic_value - log_probs # Different from target_value (network)
        value_loss = .5 * F.mse_loss(values, value_target)
        value_loss.backward(retain_graph=True)
        self.value.optimizer.step()

        actions, log_probs = self.actor.sample_normal(states, last_actions, reparameterize=True)
        q1_new_policy = self.critic_1.forward(states, last_actions, actions)
        q2_new_policy = self.critic_2.forward(states, last_actions, actions)
        critic_value = T.min(q1_new_policy, q2_new_policy)

        actor_loss = T.mean(log_probs - critic_value)
        self.actor.optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self.actor.optimizer.step()

        q_hat = self.scale * rewards + self.gamma * target_values
        q1_old_policy = self.critic_1.forward(states, last_actions, actions)
        q2_old_policy = self.critic_2.forward(states, last_actions, actions)
        critic_1_loss = .5 * F.mse_loss(q1_old_policy, q_hat)
        critic_2_loss = .5 * F.mse_loss(q2_old_policy, q_hat)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()
        
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

