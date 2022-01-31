import torch as T
from torch.cuda import is_available
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.distributions import Normal
from pathlib import Path

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        batches = [indices[i: i + self.batch_size] for i in batch_start]
        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state.cpu().detach().numpy())
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []


class Preproccess(nn.Module):
    def __init__(self, input_dims_minutely, input_dims_daily, 
        input_dims_weekly, lr=3e-4, fc1_dims=32, fc2_dims=16, output_dims=16):
        super(Preproccess, self).__init__()
        self.filepath = str(Path(__file__).parent)
        self.checkpoint_dir =  self.filepath + '/Trained_Models'
        self.filename = 'preproccess.pt'

        self.minutely_network = nn.Sequential(
            nn.Linear(input_dims_minutely, fc1_dims),
            nn.Dropout(p=.25),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.Dropout(p=.2),
            nn.ReLU(),
            nn.Linear(fc2_dims, output_dims)
        )

        self.daily_network = nn.Sequential(
            nn.Linear(input_dims_daily, fc1_dims),
            nn.Dropout(p=.25),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.Dropout(p=.2),
            nn.ReLU(),
            nn.Linear(fc2_dims, output_dims)
        )

        self.weekly_network = nn.Sequential(
            nn.Linear(input_dims_weekly, fc1_dims),
            nn.Dropout(p=.25),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.Dropout(p=.2),
            nn.ReLU(),
            nn.Linear(fc2_dims, output_dims)
        )

        minutely_M = T.randn((int(np.sqrt(output_dims)), int(np.sqrt(output_dims))))
        self.minutely_weight_M = nn.Parameter(minutely_M)

        daily_M = T.randn((int(np.sqrt(output_dims)), int(np.sqrt(output_dims))))
        self.daily_weight_M = nn.Parameter(daily_M)

        weekly_M = T.randn((int(np.sqrt(output_dims)), int(np.sqrt(output_dims))))
        self.weekly_weight_M = nn.Parameter(weekly_M)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, minutely_data, daily_data, weekly_data):
        M = self.minutely_network(T.flatten(minutely_data, start_dim=-2)).reshape(*self.minutely_weight_M.shape)
        D = self.daily_network(T.flatten(daily_data, start_dim=-2)).reshape(*self.daily_weight_M.shape)
        W = self.weekly_network(T.flatten(weekly_data, start_dim=-2)).reshape(*self.weekly_weight_M.shape)
        minute = T.mm(M, self.minutely_weight_M)
        day = T.mm(D, self.daily_weight_M)
        week = T.mm(W, self.weekly_weight_M)
        return T.cat((minute, day, week), dim=0)
    
    def save_checkpoint(self, reward_type):
        T.save(self.state_dict(), self.checkpoint_dir + '/' + reward_type + '_' + self.filename)

    def load_checkpoint(self, reward_type):
        self.load_state_dict(T.load(self.checkpoint_dir + '/' + reward_type + '_' + self.filename))

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, actor_lr, fc1_dims=256, fc2_dims=256):
        super(ActorNetwork, self).__init__()
        self.filepath = str(Path(__file__).parent)
        self.checkpoint_dir =  self.filepath + '/Trained_Models'
        self.filename = 'actor_model.pt'
        
        self.actor_network = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.LayerNorm(fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LayerNorm(fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
        self.sigma = nn.Parameter(T.ones(1))
        self.optimizer = optim.Adam(self.parameters(), lr=actor_lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        mu = self.actor_network(T.flatten(state, start_dim=-2))
        sigma = self.sigma.expand_as(mu)
        return mu, sigma

    def sample_normal(self, state):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)
        action = probabilities.sample()

        action = T.tanh(action).to(self.device)
        log_probs = probabilities.log_prob(action)
        return action, log_probs

    def save_checkpoint(self, reward_type):
        T.save(self.state_dict(), self.checkpoint_dir + '/' + reward_type + '_' + self.filename)

    def load_checkpoint(self, reward_type):
        self.load_state_dict(T.load(self.checkpoint_dir + '/' + reward_type + '_' + self.filename))

class CriticNetwork(nn.Module):
    def __init__(self, input_dims, critic_lr, fc1_dims=256, fc2_dims=256):
        super(CriticNetwork, self).__init__()
        self.filepath = str(Path(__file__).parent)
        self.checkpoint_dir =  self.filepath + '/Trained_Models'
        self.filename = 'critic_model.pt'

        self.critic_network = nn.Sequential(
            nn.Linear(input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )
        self.optimizer = T.optim.Adam(self.parameters(), lr=critic_lr)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.critic_network(T.flatten(state, start_dim=-2))

    def save_checkpoint(self, reward_type):
        T.save(self.state_dict(), self.checkpoint_dir + '/' + reward_type + '_' + self.filename)

    def load_checkpoint(self, reward_type):
        self.load_state_dict(T.load(self.checkpoint_dir + '/' + reward_type + '_' + self.filename))

class Agent:
    def __init__(self, input_dims_actorcritic=4*12, input_dims_minutely=48*4, 
        input_dims_daily=30*5, input_dims_weekly=30*4, discount=0.99, 
        actor_lr=3e-4, critic_lr=3e-4, gae_lambda=0.95, policy_clip=0.1, 
        batch_size=1024, N=2048, n_epochs=4):
        self.discount = discount
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.N = N

        self.preprocess = Preproccess(input_dims_minutely, input_dims_daily, input_dims_weekly)
        self.actor = ActorNetwork(input_dims_actorcritic, actor_lr)
        self.critic = CriticNetwork(input_dims_actorcritic, critic_lr)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def choose_action(self, minutely_data, daily_data, weekly_data):
        observation = self.preprocess.forward(minutely_data, daily_data, weekly_data)
        state = observation.to(self.actor.device)

        action, log_probs = self.actor.sample_normal(state)
        value = self.critic(state)

        log_probs = log_probs.detach().cpu().numpy().flatten()
        action = action.detach().cpu().numpy().flatten()
        value = value.detach().cpu().numpy().flatten()

        return action, log_probs, value, state

    def learn(self):
        for _ in range(self.n_epochs):
            states_arr, actions_arr, old_log_probs_arr, vals_arr, rewards_arr,\
                dones_arr, batches = self.memory.generate_batches()
            
            advantage = np.zeros(len(rewards_arr), dtype=np.float32)

            for t in range(len(rewards_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(rewards_arr) - 1):
                    a_t += discount * (rewards_arr[k] + self.discount * vals_arr[k+1] * \
                        (1 - int(dones_arr[k])) - vals_arr[k])
                    discount *= self.discount * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(vals_arr).to(self.actor.device)
            for batch in batches:
                states = T.nan_to_num(T.tensor(states_arr[batch], dtype=T.float), nan=2e-1).to(self.actor.device)
                old_log_probs = T.tensor(old_log_probs_arr[batch]).to(self.actor.device)

                critic_value = T.squeeze(self.critic(states))
                new_log_probs = self.actor.sample_normal(states)[1]
                prob_ratios = T.exp(new_log_probs - old_log_probs)

                weighted_probs = advantage[batch] * prob_ratios
                weighted_clipped_probs =T.clamp(prob_ratios, 1 - self.policy_clip, 
                    1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss

                self.preprocess.optimizer.zero_grad()
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                total_loss.backward()

                self.preprocess.optimizer.step()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

    def save_models(self, reward_type):
        print('...saving models...')
        self.preprocess.save_checkpoint(reward_type)
        self.actor.save_checkpoint(reward_type)
        self.critic.save_checkpoint(reward_type)

    def load_models(self, reward_type):
        print('...loading models...')
        self.preprocess.load_checkpoint(reward_type)
        self.actor.load_checkpoint(reward_type)
        self.critic.load_checkpoint(reward_type)
