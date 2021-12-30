import os
import torch as T
import torch.nn.functional as F
import numpy as np
from Actor_Critic_ddpg import ActorNetwork, CriticNetwork, ReplayBuffer

class Agent():
    def __init__(self, alpha=3e-4, beta=3e-4, input_dims=[8], env=None, gamma=.99, 
        n_actions=1, max_size=100000, tau=.01, fc1_dims=256, fc2_dims=256, 
        hidden_dims=256, n_layers=20, batch_size=64):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions

        self.actor = ActorNetwork(alpha, input_dims, hidden_dims, n_layers, n_actions, name='Actor')
        self.target_actor = ActorNetwork(alpha, input_dims, hidden_dims, n_layers, n_actions, name='Target_Actor')
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, n_actions, name='Critic')
        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims, n_actions, name='Target_Critic')

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = T.tensor([observation]).to(self.actor.device)
        mu = self.actor.forward(state).to(self.actor.device)

        return mu.cpu().detach().numpy()

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        state, action, reward, state_, _ = self.memory.sample_buffer(self.batch_size)
        state = T.tensor(state, dtype=T.float).to(self.critic.device)
        action = T.tensor(action, dtype=T.float).to(self.critic.device)
        reward = T.tensor(reward, dtype=T.float).to(self.critic.device)
        state_ = T.tensor(state_, dtype=T.float).to(self.critic.device)

        target_actions = self.target_actor.forward(state_)
        critic_value_ = self.target_critic.forward(state_, target_actions)
        critic_value = self.critic.forward(state, action)
        
        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma*critic_value_[j])
        target = T.tensor(target).to(self.critic.device)
        target = target.view(self.batch_size, 1)

        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        mu = self.actor.forward(state)
        actor_loss = -self.critic.forward(state, mu)
        critic_loss = F.mse_loss(target, critic_value)
        
        T.mean(actor_loss).backward()
        critic_loss.backward()

        self.actor.optimizer.step()
        self.critic.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau
        
        actor_dict = self.actor.state_dict()
        critic_dict = self.critic.state_dict()
        target_actor_dict = self.target_actor.state_dict()
        target_critic_dict = self.target_critic.state_dict()

        for name in actor_dict:
            actor_dict[name] = tau*actor_dict[name].clone() + \
                (1-tau)*target_actor_dict[name].clone()

        self.target_actor.load_state_dict[actor_dict]

        for name in critic_dict:
            critic_dict[name] = tau*critic_dict[name].clone() + \
                (1-tau)*target_critic_dict[name].clone()
        
        self.target_critic.load_state_dict(critic_dict)

    def save_models(self):
        print('...Saving Models...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        print('...Loading Models...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.target_critic.load_checkpoint()

        

