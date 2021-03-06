import torch as T
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import sqrtm
import time

class GraphConstructor(nn.Module):
    def __init__(self, n_nodes, n_features, lookback_window, n_time_features, delta_min=0.05):
        super(GraphConstructor, self).__init__()
        ### 
        # n_nodes: int - number of nodes/assets
        # n_features: int - number of features after FC layers
        # time_features: int - dimension of one-hot encoded time vector
        # delta_min: float - minimum weight to consider in adjacency matrices

        self.n_nodes = n_nodes
        self.layer_initial = T.randn(lookback_window, n_nodes)
        self.lookback_window = lookback_window
        self.n_features = n_features
        self.time_features = n_time_features
        self.delta_min = delta_min

        fc1_dims = 256
        self.spatial = nn.Sequential(
            nn.Linear(n_nodes, fc1_dims),
            nn.LayerNorm(fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, n_nodes * n_features)
        )
        self.temporal = nn.Sequential(
            nn.Linear(n_time_features, fc1_dims),
            nn.LayerNorm(fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, n_nodes * n_features)
        )
        
        self.B = nn.Parameter(T.ones(n_features, n_features))

        self.optimizer = T.optim.Adam(self.parameters(), lr=1e-3)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def create_node_embedding(self, time_features):
        ###
        # time_features: Tensor (lookback_window, time_features) - Temporal node embedding

        # output: Tensor (n_nodes, n_features, lookback_window) - spatio-temporal embedding for each 
        #                                                         node at each time step.
        layer_initial = T.randn(self.lookback_window, self.n_nodes, device=self.device)
        embedding = T.add(self.spatial(layer_initial), self.temporal(time_features))
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
        if time_diff >= 0:
            U1 = T.squeeze(U[:, :, idx - time_diff])
            U2 = T.squeeze(U[:, :, idx])
        else:
            U1 = T.squeeze(U[:, :, idx])
            U2 = T.squeeze(U[:, :, idx + time_diff])
        x = T.mm(T.mm(U1, self.B), T.transpose(U2, 0, 1)).detach().cpu()
        x = T.tensor([i if i >= self.delta_min else 0 \
            for i in T.flatten(x)]).reshape(x.shape)
        adj = x.float().softmax(dim=-1).to(self.device)
        return adj


class DilatedGraphConvolutionCell(nn.Module):
    def __init__(self, kernel_size, n_data_features, dilation_list, 
    fc1_dims, fc2_dims, n_features, n_nodes, lookback_window, n_time_features):
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
        self.graph = GraphConstructor(
            self.n_nodes, 
            self.n_features, 
            self.lookback_window,
            n_time_features
        )

        self.FC = nn.Sequential(
            nn.Linear(n_nodes * n_data_features, fc1_dims),
            nn.LayerNorm(fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.LayerNorm(fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_nodes * n_features)
        )

        self.W_forward = nn.Parameter(T.randn((self.kernel_size, n_features, n_features)))
        self.W_backward = nn.Parameter(T.randn((self.kernel_size, n_features, n_features)))
        self.b = nn.Parameter(T.randn((n_features)))

        self.optimizer = T.optim.Adam(self.parameters(), lr=1e-3)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def normalize_adjacency_matrix(self, time_features, idx, time_diff):
        ###
        # time_features: Tensor (n_nodes, n_time_features, lookback_window)
        # idx: int - current time step
        # time_diff: int - difference between timesteps for ST graph construction

        # output: Tensor (n_nodes, n_nodes) - normalized adjacency matrix
        adjacency_matrix = self.graph.create_adjacency_matrix(time_features, idx, time_diff)
        degree_matrix = T.eye(adjacency_matrix.shape[0]) * adjacency_matrix.clone().cpu().sum(-1)
        D = T.inverse(T.tensor(sqrtm(degree_matrix.cpu()), dtype=T.float, device=self.device))
        return T.mm(T.mm(D, adjacency_matrix), D)

    def fully_connected(self, observation):
        observation = observation.clone()
        obs = T.flatten(observation.permute(2, 0, 1).contiguous(), start_dim=1).to(self.device)
        X = self.FC(obs).reshape(self.lookback_window, self.n_nodes, self.n_features)
        X = X.permute(1, 2, 0).contiguous()
        self.X = X 

    def conv(self, input, time_features, idx, gamma):
        ###
        # input: Tensor (n_nodes, n_data_features, lookback_window)
        # time_features: Tensor (n_nodes, 15+5+4+12, lookback_window)
        # idx: int - current time step
        # X: Tensor (n_nodes, n_features, lookback_window)

        # output: Tensor (n_nodes, n_features, 1) - output of convolution operation
        Z = T.zeros((self.n_nodes, self.n_features), device=self.device)
        X = input
        for k in range(self.kernel_size):
            X_t = X[:, :, (idx // gamma) - k]
            L1 = self.normalize_adjacency_matrix(time_features, idx, k * gamma)
            L2 = self.normalize_adjacency_matrix(time_features, idx, -k * gamma)
            x = T.mm(T.mm(L1, X_t), T.squeeze(self.W_forward[k, :, :])) \
                + T.mm(T.mm(L2, X_t), T.squeeze(self.W_backward[k, :, :])) \
                + self.b
            Z += T.tanh(x)
        return Z

    def conv_layer(self, input, time_features, dilation_factor):
        ###
        # input: Tensor (n_nodes, n_data_features, lookback_window)
        # time_features: Tensor (n_nodes, 15+5+4+12, lookback_window)
        # dilation_factor: int - dilation for conv layer

        # output: Tensor (n_nodes, n_data_features, lookback_window) - output of convolution operation
        Z = T.zeros((self.n_nodes, self.n_features, input.shape[-1] // dilation_factor), device=self.device)
        for t in range(input.shape[-1]):
            if (t + 1) % dilation_factor == 0:
                Z[:, :, t // dilation_factor] = self.conv(input, time_features, self.gamma * t, self.gamma)
        self.gamma *= dilation_factor
        return Z

    def STJGN_module(self, observation, time_features):
        ###
        # observation: Tensor (n_nodes, n_data_features, lookback_window)
        # time_features: Tensor (n_nodes, 15+5+4+12, lookback_window)

        # output: list - hidden states of each STJGCN layer
        self.fully_connected(observation)
        self.gamma = 1
        output = []
        Z = self.X
        for dilation_factor in self.dilation_list:
            Z = self.conv_layer(input=Z, time_features=time_features, dilation_factor=dilation_factor)
            output.append(Z[:, :, -1])
        return output


class AttentionOutputModule(nn.Module):
    def __init__(self, kernel_size, n_data_features, dilation_list, 
    fc1_dims, fc2_dims, n_features, n_nodes, lookback_window, n_time_features):
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
            lookback_window=lookback_window,
            n_time_features=n_time_features
        )

        self.v = nn.Parameter(T.randn(n_features, 1))
        self.lin = nn.Sequential(
            nn.Linear(self.n_features, self.n_features),
            nn.LayerNorm(self.n_features),
            nn.Tanh(),
            nn.Linear(self.n_features, 1, bias=False),
            nn.LayerNorm(1)
        )

        self.conv_map_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=20, kernel_size=1),
            nn.GroupNorm(1, num_channels=20),
            nn.ReLU()
        )
        self.conv_map_2 = nn.Sequential(
            nn.Conv2d(in_channels=21, out_channels=6, kernel_size=1),
            nn.GroupNorm(1, num_channels=6),
            nn.ReLU(),
            nn.Conv2d(in_channels=6, out_channels=1, kernel_size=1),
            nn.GroupNorm(1, num_channels=1),
            nn.Softmax(dim=2)
        )
        self.optimizer = T.optim.Adam(self.parameters(), lr=1e-3)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def compute_att_weights(self, hidden_states):
        ###
        # hidden_states: List of Tensors - [n_conv_layers * (n_nodes, n_features)]
        # HS: Tensor (n_conv_layers, n_nodes, n_features)

        # output: Tensor (n_conv_layers, n_nodes) - attention weights
        HS = T.randn((len(hidden_states), *hidden_states[0].shape), device=self.device)
        alpha = T.zeros((len(hidden_states), self.n_nodes), device=self.device)

        for idx, state in enumerate(hidden_states):
            HS[idx, :, :] = state
        for node in range(HS.shape[1]):
            Z = T.zeros(1, device=self.device)
            for layer in range(HS.shape[0]):
                s = self.lin(HS[layer, node, :])
                Z += T.exp(s)
                alpha[layer, node] = T.exp(s)
            alpha[:, node] = alpha[:, node].clone() / Z
        return alpha.reshape(*alpha.shape, 1).to(self.device), HS

    def compute_att_weighted_conv_output(self, observation, time_features):
        ###
        # observation: Tensor (n_nodes, n_data_features, lookback_window)
        # time_features: Tensor (n_nodes, 15+5+4+12, lookback_window)
        # alpha: Tensor (n_conv_layers, n_nodes, 1)
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
        Y = self.compute_att_weighted_conv_output(observation, time_features).permute(1, 0).contiguous()
        Y = Y.reshape(1, *Y.shape, 1)
        last_action = last_action.reshape(1, 1, *last_action.shape, 1)

        action = T.cat((self.conv_map_1(Y), last_action), dim=1)
        action = self.conv_map_2(action)

        return T.squeeze(action)


class Agent(nn.Module):
    def __init__(self, kernel_size, n_data_features, dilation_list, 
    fc1_dims, fc2_dims, n_features, n_nodes, lookback_window,
    minibatch_size, n_time_features):
        super(Agent, self).__init__()
        ###
        # minibatch_size: int
        self.minibatch_size = minibatch_size
        self.network = AttentionOutputModule(
            kernel_size=kernel_size, 
            n_data_features=n_data_features, 
            dilation_list=dilation_list, 
            fc1_dims=fc1_dims, 
            fc2_dims=fc2_dims, 
            n_features=n_features, 
            n_nodes=n_nodes, 
            lookback_window=lookback_window,
            n_time_features=n_time_features
        )
        
        self.optimizer = T.optim.Adam(self.parameters(), lr=1e-3)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_commisions_factor(self, observation, action, last_action):
        delta = 5e-3
        c_factor = 0.00025
        done = False
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
        observation = observation.to(self.device)
        time_features = time_features.to(self.device)
        action = self.network.forward(observation, time_features, last_action).to('cpu')
        observation = observation.to('cpu')
        last_action = last_action.to('cpu')
        price_change_vector = observation[:, 2, -1]
        mu = self.calculate_commisions_factor(observation, action, last_action)
        reward = T.log(mu * T.dot(last_action, price_change_vector)) / self.minibatch_size
        return action.to(self.device), reward