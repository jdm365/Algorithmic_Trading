import torch as T
from torch._C import Value
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from datetime import datetime, time
from scipy.linalg import sqrtm
from pathlib import Path
import shutup

class GraphConstructor(nn.Module):
    def __init__(self, n_nodes, n_features, lookback_window, time_features=15+5+4+12, delta_min=0.05):
        super(GraphConstructor, self).__init__()
        ### 
        # n_nodes: int - number of nodes/assets
        # n_features: int - number of features after FC layers
        # time_features: int - dimension of one-hot encoded time vector
        # delta_min: float - minimum weight to consider in adjacency matrices

        self.n_nodes = n_nodes
        self.layer_initial = init.xavier_normal_(T.ones(lookback_window, n_nodes))
        self.lookback_window = lookback_window
        self.n_features = n_features
        self.time_features = time_features
        self.delta_min = delta_min

        fc1_dims = 256
        self.spatial = nn.Sequential(
            nn.Linear(n_nodes, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, n_nodes * n_features),
            nn.ReLU()
        )
        self.temporal = nn.Sequential(
            nn.Linear(time_features, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, n_nodes * n_features),
            nn.ReLU()
        )
        
        self.B = nn.Parameter(T.ones(n_features, n_features))

        self.optimizer = T.optim.Adam(self.parameters(), lr=1e-4)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def create_node_embedding(self, time_features):
        ###
        # time_features: Tensor (lookback_window, time_features) - Temporal node embedding

        # output: Tensor (n_nodes, n_features, lookback_window) - spatio-temporal embedding for each 
        #                                                         node at each time step.
        embedding = T.add(self.spatial(self.layer_initial), self.temporal(time_features))
        embedding = embedding.reshape(self.lookback_window, self.n_nodes, self.n_features)
        return embedding.permute(1, 0, 2).contiguous()

    def create_adjacency_matrix(self, time_features, idx, time_diff):
        ###
        # observation: Tensor (n_nodes, input_features + 1, lookback_window)
        # idx: int - current time step
        # U: Tensor (n_nodes, n_features, lookback_window) - spatio-temporal node embeddings
        # time_diff: int - time difference between spatial embedding 1 and 2

        # output: Tensor (n_nodes, n_nodes)
        U = self.create_node_embedding(time_features)
        U1 = T.squeeze(U[:, :, idx - time_diff])
        U2 = T.squeeze(U[:, :, idx])
        x = T.mm(T.mm(U1, self.B), T.transpose(U2, 0, 1))
        x = T.tensor([i if i >= self.delta_min else 0 \
            for i in T.flatten(x)]).reshape(x.shape)
        return F.softmax(x)


class DilatedGraphConvolutionCell(nn.Module):
    def __init__(self, kernel_size, n_data_features, dilation_list, 
    fc1_dims, fc2_dims, n_features, n_nodes, lookback_window):
        super(DilatedGraphConvolutionCell, self).__init__()
        ### 
        # kernel_size: int - size of kernel for convolution
        # n_data_features: int - number of node features in observed data
        # dilation_list: list - list of dilation factors for each layer
        # fc1_dims: int
        # fc2_dims: int
        # n_features: int - number of nodes features after FC layers
        # n_nodes: int
        # lookback_window: int
        
        # self.W_forward: Tensor (kernel_size, n_features, n_output_features) - parameter used in 
        #                                                                       convolutional layers
        # self.W_backward: Tensor (kernel_size, n_features, n_output_features) - parameter used in 
        #                                                                       convolutional layers
        # self.b: Tensor (n_output_features) - parameter used in convolutional layers

        self.kernel_size = kernel_size
        self.n_data_features = n_data_features
        self.dilation_list = dilation_list
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_features = n_features
        self.n_nodes = n_nodes
        self.lookback_window = lookback_window

        self.FC = nn.Sequential(
            nn.Linear(n_nodes * n_data_features, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_nodes * n_features),
            nn.ReLU()
        )

        self.W_forward = nn.Parameter(T.ones((self.kernel_size, n_features, n_features)))
        self.W_backward = nn.Parameter(T.ones((self.kernel_size, n_features, n_features)))
        self.b = nn.Parameter(T.ones((n_features)))

        self.optimizer = T.optim.Adam(self.parameters(), lr=1e-4)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def normalize_adjacency_matrix(self, time_features, idx, time_diff):
        ###
        # time_features: Tensor (n_nodes, 15+5+4+12, lookback_window)
        # idx: int - current time step
        # time_diff: int - difference between timesteps for ST graph construction

        # output: Tensor (n_nodes, n_nodes) - normalized adjacency matrix
        graph = GraphConstructor(
            self.n_nodes, 
            self.n_features, 
            self.lookback_window
        )
        adjacency_matrix = graph.create_adjacency_matrix(time_features, idx, time_diff)
        degree_matrix = T.eye(adjacency_matrix.shape[0]) * adjacency_matrix.sum(-1)
        D = T.inverse(T.tensor(sqrtm(degree_matrix), dtype=T.float))
        return T.mm(T.mm(D, adjacency_matrix), D)

    def fully_connected(self, observation):
        obs = T.flatten(observation.permute(2, 0, 1).contiguous(), start_dim=1)
        X = self.FC(obs).reshape(self.lookback_window, self.n_nodes, self.n_features)
        X = X.permute(1, 2, 0).contiguous()
        self.X = X

    def conv(self, input, time_features, idx):
        ###
        # input: Tensor (n_nodes, n_data_features, lookback_window)
        # time_features: Tensor (n_nodes, 15+5+4+12, lookback_window)
        # idx: int - current time step
        # X: Tensor (n_nodes, n_features, lookback_window)

        # output: Tensor (n_nodes, n_features, 1) - output of convolution operation
        Z = T.zeros((self.n_nodes, self.n_features))
        X = input
        for k in range(self.kernel_size):
            X_t = X[:, :, idx - k]
            L1 = self.normalize_adjacency_matrix(time_features, idx, k)
            L2 = self.normalize_adjacency_matrix(time_features, idx, -k)
            x = T.mm(T.mm(L1, X_t), T.squeeze(self.W_forward[k, :, :])) \
                + T.mm(T.mm(L2, X_t), T.squeeze(self.W_backward[k, :, :])) \
                + self.b
            Z += F.relu(x)
        return Z.reshape(*Z.shape, 1)

    def conv_layer(self, input, time_features, dilation_factor):
        ###
        # input: Tensor (n_nodes, n_data_features, lookback_window)
        # time_features: Tensor (n_nodes, 15+5+4+12, lookback_window)
        # dilation_factor: int - dilation for conv layer

        # output: Tensor (n_nodes, n_data_features, lookback_window) - output of convolution operation
        for t in range(1, self.lookback_window-1):
            if t % dilation_factor == 0:
                try:
                    Z = T.cat((Z, self.conv(input, time_features, t)), dim=-1)
                except UnboundLocalError:
                    Z = self.conv(input, time_features, t)
            else:
                try:
                    Z = T.cat((Z, T.zeros(self.n_nodes, self.n_features, 1)), dim=-1)
                except UnboundLocalError:
                    Z = T.zeros(self.n_nodes, self.n_features, 1)
        return Z

    def STJGN_module(self, observation, time_features):
        ###
        # observation: Tensor (n_nodes, n_data_features, lookback_window)
        # time_features: Tensor (n_nodes, 15+5+4+12, lookback_window)

        # output: list - hidden states of each STJGCN layer
        self.fully_connected(observation)
        Z0 = self.conv_layer(input=self.X, time_features=time_features, dilation_factor=self.dilation_list[0])
        Z1 = self.conv_layer(Z0, time_features, self.dilation_list[1])
        Z2 = self.conv_layer(Z1, time_features, self.dilation_list[2])
        Z3 = self.conv_layer(Z2, time_features, self.dilation_list[3])
        output = [Z0[:, :, -1], Z1[:, :, -1], Z2[:, :, -1], Z3[:, :, -1]]
        return output


class AttentionOutputModule(nn.Module):
    def __init__(self, kernel_size, n_data_features, dilation_list, 
    fc1_dims, fc2_dims, n_features, n_nodes, lookback_window):
        super(AttentionOutputModule, self).__init__()
        ### 
        # kernel_size: int - size of kernel for convolution
        # n_data_features: int - number of node features in observed data
        # dilation_list: list - list of dilation factors for each layer
        # fc1_dims: int
        # fc2_dims: int
        # n_features: int - number of nodes features after FC layers
        # n_nodes: int
        # lookback_window: int

        # self.W: Tensor (n_features, n_features) - parameter used in 
        #                                           calculating attention weights
        # self.v: Tensor (n_features) - parameter used in calculating attention weights
        # self.b: Tensor (n_features) - parameter used in calculating attention weights

        self.kernel_size = kernel_size
        self.n_data_features = n_data_features
        self.dilation_list = dilation_list
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_features = n_features
        self.n_nodes = n_nodes
        self.lookback_window = lookback_window

        self.STJGCN = DilatedGraphConvolutionCell(
            kernel_size=kernel_size,
            n_data_features=n_data_features,
            dilation_list=dilation_list,
            fc1_dims=fc1_dims,
            fc2_dims=fc2_dims,
            n_features=n_features,
            n_nodes=n_nodes,
            lookback_window=lookback_window
        )

        self.v = nn.Parameter(T.zeros(n_features, 1))

        self.state_layer = nn.Linear(n_features, 256)
        self.last_action_layer = nn.Linear(n_nodes, 256)

        self.FC = nn.Sequential(
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        self.optimizer = T.optim.Adam(self.parameters(), lr=1e-4)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def compute_att_weights(self, hidden_states):
        ###
        # hidden_states: List of Tensors - [n_conv_layers * (n_nodes, n_features)]
        # HS: Tensor (n_conv_layers, n_nodes, n_features)

        # output: Tensor (n_conv_layers, n_nodes) - attention weights
        HS = T.randn(4, *hidden_states[0].shape)
        Z = T.zeros(1, 23)
        alpha = T.zeros(4, 23)
        lin = nn.Linear(self.n_features, self.n_features)

        for idx, state in enumerate(hidden_states):
            HS[idx, :, :] = state
        
        for layer in range(HS.shape[0]):
            s = T.mm(T.transpose(self.v, 0, 1), T.transpose(T.tanh(lin(HS[layer, :, :])), 0, 1))
            Z += T.exp(s)
            alpha[layer, :] = T.exp(s)
        return (alpha / Z).reshape(*alpha.shape, 1), HS

    def compute_att_weighted_conv_output(self, observation, time_features):
        ###
        # observation: Tensor (n_nodes, n_data_features, lookback_window)
        # time_features: Tensor (n_nodes, 15+5+4+12, lookback_window)
        # HS: Tensor (n_conv_layers, n_nodes, n_features)
        
        # output: Tensor (n_nodes, n_features)
        hidden_states = self.STJGCN.STJGN_module(observation, time_features)
        alpha, HS = self.compute_att_weights(hidden_states)
        return T.sum(T.mul(alpha, HS), dim=0)

    def forward(self, observation, time_features, last_action):
        ###
        # observation: Tensor (n_nodes, n_data_features + 1, lookback_window)
        # last_action: Tensor (n_nodes) - last action (previous portfolio weights)
        
        # output: Tensor (n_nodes) - action (new portfloio weights)
        Y = self.compute_att_weighted_conv_output(observation, time_features)
        out = T.cat((self.state_layer(Y), T.ones(self.n_nodes, 256) * self.last_action_layer(last_action)), dim=1)
        action = self.FC(out)
        return T.squeeze(F.softmax(action, dim=0))


class Agent(nn.Module):
    def __init__(self, kernel_size, n_data_features, dilation_list, 
    fc1_dims, fc2_dims, n_features, n_nodes, lookback_window,
    minibatch_size):
        super(Agent, self).__init__()
        ###
        # minibatch_size: int
        # filename: location of market data
        self.minibatch_size = minibatch_size
        self.network = AttentionOutputModule(
            kernel_size=kernel_size, 
            n_data_features=n_data_features, 
            dilation_list=dilation_list, 
            fc1_dims=fc1_dims, 
            fc2_dims=fc2_dims, 
            n_features=n_features, 
            n_nodes=n_nodes, 
            lookback_window=lookback_window
        )
        
        self.optimizer = T.optim.Adam(self.parameters(), lr=1e-4)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_commisions_factor(self, observation, action, last_action):
        delta = 5e-3
        c_factor = .0025
        done = False
        action = action.detach().clone()
        price_change_vector = T.squeeze(observation[:, 2, -1])
        w_prime = T.mul(last_action, price_change_vector)
        mu = c_factor * T.sum(T.abs(w_prime - action))
        while not done:
            mu_ = (1 - c_factor * w_prime[0] - (2*c_factor - c_factor**2) * T.sum(F.relu(w_prime[1:] - mu * action[1:]))) / (1 - c_factor * action[0])
            if T.abs(mu_ - mu) < delta:
                done = True
            else:
                mu = mu_
        return mu_

    def step(self, observation, time_features, last_action):
        action = self.network.forward(observation, time_features, last_action)
        price_change_vector = observation[:, 2, -1]
        mu = self.calculate_commisions_factor(observation, action, last_action)
        reward = T.log(mu * T.dot(last_action, price_change_vector)) / self.minibatch_size
        return action, reward


class GetData():
    def __init__(self):
        self.filepath = str(Path(__file__).parent) + '/Minute_Data_v1/'

    def make_DF(self):
        DFNEW = pd.DataFrame()
        for filename in os.listdir(self.filepath):
            DF = pd.read_csv(self.filepath + filename)
            DF = DF[['Local time', 'High', 'Low', 'Close', 'Volume']]
            try:
                DFNEW = DFNEW.merge(DF, left_on='Local time', right_on='Local time')
            except:
                DFNEW = DF
        DFNEW = DFNEW.replace(0, 1)
        return DFNEW

    def make_global_tensor_no_time(self):
        df = self.make_DF()
        arr = np.array(df)[:, 1:] # (time, features)
        Arr = np.array(arr[1:, :] / arr[:-1, :]).astype(float)
        X = T.ones((Arr.shape[-1] // 4, 4, Arr.shape[0]))
        for i in range(arr.shape[-1]):
            j = i // 4
            k = i % 4
            X[j, k, :] = T.tensor(Arr[:, i])
        return X

    def make_global_temporal_tensor(self):
        df = self.make_DF()
        arr = np.array(df)[:, 0]
        M = T.zeros((arr.shape[0], 36))
        for i in range(1, arr.shape[0]):
            base = arr[i]

            half_hour = T.tensor(2 * (int(base[11:13]) - 9) + (int(base[14:16]) >= 30))
            day = T.tensor(datetime.strptime(base[:10].replace('.', ' '), '%d %m %Y').isoweekday())
            week = T.tensor(abs(int(base[:2]) - 4) // 7)
            month = T.tensor(int(base[3:5]))

            half_hour = F.one_hot(half_hour-1, 15)
            day = F.one_hot(day-1, 5)
            week = F.one_hot(week, 4)
            month = F.one_hot(month-1, 12)
            M[i, :] = T.cat((half_hour, day, week, month))
        return M[1:, :]


if __name__ == '__main__':
    shutup.please()
    n_epochs = 1000
    X = GetData().make_global_tensor_no_time()
    M = GetData().make_global_temporal_tensor()
    agent = Agent(
        kernel_size=2, 
        n_data_features=4, 
        dilation_list=[2, 3, 4, 6], 
        fc1_dims=256, 
        fc2_dims=512, 
        n_features=64, 
        n_nodes=X.shape[0], 
        lookback_window=64,
        minibatch_size=256
    )
    for epoch in tqdm(range(n_epochs)):
        done = False
        time_initial = np.random.randint(agent.network.lookback_window, \
            X.shape[-1] - agent.minibatch_size)
        Reward = 0
        cntr = 0
        capital = 10000
        last_action = nn.Softmax()(T.rand(X.shape[0]))
        while done is False:
            observation = X[:, :, time_initial + cntr - agent.network.lookback_window:cntr + time_initial]
            time_feature = M[time_initial + cntr - agent.network.lookback_window:cntr + time_initial, :]
            last_action, reward = agent.step(observation, time_feature, last_action)
            Reward += reward
            print(Reward)
            capital *= T.exp(reward * agent.minibatch_size)
            cntr += 1
            if cntr % agent.minibatch_size == 0:
                done = True
        Loss = -Reward
        Loss.backward()
        agent.optimizer.step()
        agent.optimizer.zero_grad()
        Profits = 10000 - capital

        print('Episode profits: {Profits}')

    T.save(agent.state_dict())