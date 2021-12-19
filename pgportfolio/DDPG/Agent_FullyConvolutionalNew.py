import os
import torch as T
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from ActorCritic_FullyConvolutionalNew import ReplayBuffer, ActorNetwork, CriticNetwork
from Trading_env_no_commisions import TradingEnv, DataFeatures
from utils import *

class Agent(object):
    def __init__(self, env, cl1_dims=2, cl2_dims=20, alpha=3e-5, beta=3e-4, 
        gamma=.99, tau=.01, max_memory_size=50000):
        self.gamma = gamma
        self.tau = tau

        self.actor = ActorNetwork(alpha=alpha, cl1_dims=cl1_dims, cl2_dims=cl2_dims, n_actions=env.N_ASSETS, 
            lookback_window=env.batch_size, name='Actor')
        self.target_actor = ActorNetwork(alpha=alpha, cl1_dims=cl1_dims, cl2_dims=cl2_dims, n_actions=env.N_ASSETS, 
            lookback_window=env.batch_size, name='TargetActor')
        self.critic = CriticNetwork(beta=beta, cl1_dims=cl1_dims, cl2_dims=cl2_dims, n_actions=env.N_ASSETS, 
            lookback_window=env.batch_size, name='Critic')
        self.target_critic = CriticNetwork(beta=beta, cl1_dims=cl1_dims, cl2_dims=cl2_dims, n_actions=env.N_ASSETS, 
            lookback_window=env.batch_size, name='TargetCritic')

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.updateNetworkParameters(tau=1)

        # Train
        self.memory = ReplayBuffer(max_memory_size)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=beta)

    def choose_action(self, state, last_action):
        state = Variable(T.from_numpy(state).float().unsqueeze(0))[0]
        last_action = Variable(T.from_numpy(last_action).float().unsqueeze(0))
        action = self.actor.forward(state, last_action) # Added last_action
        action = action.detach().numpy()[0,0]
        return action

    def update(self, batch_size, train_length):
        # Don't update network until replay buffer is batch size
        if len(self.memory) < batch_size:
            return
        states, last_actions, actions, rewards, next_states, _ = self.memory.sample_buffer(batch_size)
        states = T.tensor(states, dtype=T.float32, requires_grad=True).to(self.critic.device)
        last_actions = T.tensor(last_actions, dtype=T.float32, requires_grad=True).to(self.critic.device)
        actions = T.tensor(actions, dtype=T.float32, requires_grad=True).to(self.critic.device)
        # Make sure rewards are discounted for batch_length
        rewards = T.tensor([reward * train_length / batch_size for reward in rewards], dtype=T.float32, requires_grad=True).to(self.critic.device) 
        next_states = T.tensor(next_states, dtype=T.float32, requires_grad=True).to(self.critic.device)

        # Critic Loss
        Qvals = self.critic.forward(T.squeeze(states), last_actions, actions)
        next_actions = self.target_actor.forward(T.squeeze(next_states), actions) # Next action after last action is current action
        next_Q = self.target_critic.forward(T.squeeze(next_states), actions, next_actions.detach())
        #print(next_Q.shape, Qvals.shape, rewards.shape)
        Qprime = T.reshape(rewards, (64, 1, 1, 1)) + self.gamma * next_Q
        critic_loss = F.mse_loss(Qvals, Qprime)

        # Actor Loss
        policy_loss = -self.critic.forward(T.squeeze(states), last_actions, self.actor.forward(T.squeeze(states), last_actions))

        # update networks
        self.actor_optimizer.zero_grad()
        T.mean(policy_loss).backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
    
        self.updateNetworkParameters()

    def updateNetworkParameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_state_dict = self.actor.state_dict()
        critic_state_dict = self.critic.state_dict()
        target_actor_state_dict = self.target_actor.state_dict()
        target_critic_state_dict = self.target_critic.state_dict()

        # update target networks
        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                (1-tau)*target_critic_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                (1-tau)*target_actor_state_dict[name].clone()

        self.target_actor.load_state_dict(actor_state_dict)

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()

