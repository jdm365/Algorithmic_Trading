import torch as T
from torch._C import Value
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import opt


class GraphConstructor(nn.Module):
    def __init__(self, n_nodes, n_features, lookback_window, time_features=15*5*4*12, delta_min=0.05):
        super(GraphConstructor, self).__init__()
        ### 
        # n_nodes: int - number of nodes/assets
        # n_features: int - number of features after FC layers
        # time_features: int - dimension of one-hot encoded time vector
        # delta_min: float - minimum weight to consider in adjacency matrices

        self.n_nodes = n_nodes
        self.layer_initial = nn.init.xavier_normal((n_nodes, 1, lookback_window))
        self.lookback_window = lookback_window
        self.n_features = n_features
        self.time_features = time_features
        self.delta_min = delta_min

        fc1_dims = 256
        self.spatial = nn.Sequential(
            nn.Linear(n_nodes, fc1_dims),
            F.relu(),
            nn.Linear(fc1_dims, n_features),
            F.relu()
        )
        self.temporal = nn.Sequential(
            nn.Linear(n_nodes * time_features, fc1_dims),
            F.relu(),
            nn.Linear(fc1_dims, n_features),
            F.relu()
        )
        
        self.B = nn.Parameter(T.ones(n_features, n_features))

        self.optimizer = T.optim.Adam(self.parameters(), lr=1e-4)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def one_hot_encode_temporal(self, observation):
        ###
        # observation: Tensor - (n_nodes, input_features + 1, lookback_window)
        # timestamps: Tensor - (n_nodes, lookback_window)

        # output: Tensor (n_nodes, time_features, lookback_window)
        timestamps = T.squeeze(observation[:, -1, :])
        encoded_times = F.one_hot(timestamps) ## write after getting data
        return encoded_times

    def create_node_embedding(self, observation):
        ###
        # observation: Tensor - (n_nodes, input_features + 1, lookback_window)
        # space: Tensor (n_nodes, 1, lookback_window) - Spatial node embedding
        # time: Tensor (n_nodes * time_features, lookback_window) - Temporal node embedding

        # output: Tensor (n_nodes, n_features, lookback_window)
        space = self.layer_initial
        time = self.one_hot_encode_temporal(observation)
        return self.spatial(self.layer_initial) + self.temporal(time)

    def create_adjacency_matrix(self, observation, idx, time_diff):
        ###
        # observation: Tensor (n_nodes, input_features + 1, lookback_window)
        # idx: int - current time step
        # U: Tensor (n_nodes, n_features, lookback_window) - spatio-temporal node embeddings
        # time_diff: int - time difference between spatial embedding 1 and 2

        # output: Tensor (n_nodes, n_nodes)
        U = self.create_node_embedding(observation)

        U1 = T.squeeze(U[:, :, idx - time_diff])
        U2 = T.squeeze(U[:, :, idx])
        x = T.mm(T.mm(U1, self.B), T.transpose(U2, 0, 1))
        x = T.tensor([i if i >= self.delta_min else 0 \
            for i in T.flatten(x)]).reshape(x.shape)
        return F.softmax(x)


class DilatedGraphConvolutionCell(nn.Module):
    def __init__(self, kernel_size, n_data_features, n_output_features, 
    dilation_list, fc1_dims, fc2_dims, n_features, n_nodes, lookback_window):
        super(DilatedGraphConvolutionCell, self).__init__()
        ### 
        # kernel_size: int - size of kernel for convolution
        # n_data_features: int - number of node features in observed data
        # n_output_features: int - features after convolution
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
        self.n_output_features = n_output_features
        self.dilation_list = dilation_list
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_features = n_features
        self.n_nodes = n_nodes
        self.lookback_window = lookback_window

        self.FC = nn.Sequential(
            nn.Linear(n_nodes * n_data_features, fc1_dims),
            F.relu(),
            nn.Linear(fc1_dims, fc2_dims),
            F.relu(),
            nn.Linear(fc2_dims, n_nodes * n_features),
            F.relu()
        )

        self.W_forward = nn.Parameter(T.ones((self.kernel_size, n_features, n_output_features)))
        self.W_backward = nn.Parameter(T.ones((self.kernel_size, n_features, n_output_features)))
        self.b = nn.Parameter(T.ones((n_output_features)))

        self.optimizer = T.optim.Adam(self.parameters(), lr=1e-4)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def normalize_adjacency_matrix(self, observation, idx, time_diff):
        ###
        # observation: Tensor (n_nodes, n_data_features + 1, lookback_window)
        # idx: int - current time step
        # time_diff: int - difference between timesteps for ST graph construction

        # output: Tensor (n_nodes, n_nodes) - normalized adjacency matrix
        graph = GraphConstructor(
            self.n_nodes, 
            self.n_features, 
            self.lookback_window
        )
        adjacency_matrix = graph.create_adjacency_matrix(observation, idx, time_diff)
        degree_matrix = adjacency_matrix.sum(-1)
        return T.mm(T.mm(T.pow(degree_matrix, -1/2), adjacency_matrix), T.pow(degree_matrix, 1/2))

    def conv(self, observation, idx):
        ###
        # observation: Tensor (n_nodes, n_data_features + 1, lookback_window)
        # idx: int - current time step
        # X: Tensor (n_nodes, n_features, lookback_window)

        # output: Tensor (n_nodes, n_output_features, 1) - output of convolution operation
        Z = T.zeros((self.n_nodes, self.n_output_features))
        for k in range(self.kernel_size):
            L1 = self.normalize_adjacency_matrix(observation, idx, k)
            L2 = self.normalize_adjacency_matrix(observation, idx, -k)
            X = self.FC(observation)[:, :, idx - k]
            x = T.mm(T.mm(L1, X), T.squeeze(self.W_forward[k, :, :])) \
                + T.mm(T.mm(L2, X), T.squeeze(self.W_backward[k, :, :])) \
                + self.b
            Z += F.relu(x)
        return Z.reshape(*Z.shape, 1)

    def conv_layer(self, input, dilation_factor):
        ###
        # input: Tensor (n_nodes, n_features, lookback_window)
        # dilation_factor: int - dilation for conv layer

        # output: Tensor (n_nodes, n_output_features, lookback_window) - output of convolution operation
        for t in range(self.lookback_window):
            if t % dilation_factor == 0:
                try:
                    Z = T.cat(Z, self.conv(input, t), dim=-1)
                except:
                    Z = self.conv(input, t)
            else:
                try:
                    Z = T.cat(Z, T.zeros((self.n_nodes, self.n_output_features, 1)), dim=-1)
                except:
                    Z = T.zeros((self.n_nodes, self.n_output_features, 1))
        return Z

    def STJGN_module(self, observation):
        ###
        # observation: Tensor (n_nodes, n_data_features + 1, lookback_window)

        # output: list - hidden states of each STJGCN layer
        Z0 = self.conv_layer(observation, self.dilation_list[0])
        Z1 = self.conv_layer(Z0, self.dilation_list[1])
        Z2 = self.conv_layer(Z1, self.dilation_list[2])
        Z3 = self.conv_layer(Z2, self.dilation_list[3])

        output = [Z0[:, :, -1], Z1[:, :, -1], Z2[:, :, -1], Z3[:, :, -1]]
        return output


class AttentionOutputModule(nn.Module):
    def __init__(self, kernel_size, n_data_features, n_output_features, 
    dilation_list, fc1_dims, fc2_dims, n_features, n_nodes, lookback_window):
        super(AttentionOutputModule, self).__init__()
        ### 
        # kernel_size: int - size of kernel for convolution
        # n_data_features: int - number of node features in observed data
        # n_output_features: int - features after convolution
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
        self.n_output_features = n_output_features
        self.dilation_list = dilation_list
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_features = n_features
        self.n_nodes = n_nodes
        self.lookback_window = lookback_window

        self.STJGCN = DilatedGraphConvolutionCell(
            kernel_size=kernel_size,
            n_data_features=n_data_features,
            n_output_features=n_output_features,
            dilation_list=dilation_list,
            fc1_dims=fc1_dims,
            fc2_dims=fc2_dims,
            n_features=n_features,
            n_nodes=n_nodes,
            lookback_window=lookback_window
        )

        self.W = nn.Parameter(T.zeros(n_features, n_features))
        self.b = nn.Parameter(T.zeros(n_features))
        self.v = nn.Parameter(T.zeros(n_features))
        self.optimizer = T.optim.Adam(self.parameters(), lr=1e-4)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def compute_att_weighted_conv_output(self, observation):
        hidden_states = self.STJGCN.STJGN_module(observation)
        Z = T.sum(T.exp(hidden_states))
        for state in hidden_states:
            s = self.v.transpose * T.tanh(T.mm(self.W, state) + self.b)
            alpha = T.exp(s) / Z
        return T.dot(alpha, hidden_states)

    def forward(self, observation):
        