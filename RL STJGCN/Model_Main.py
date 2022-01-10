import torch as T
from torch._C import Value
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import opt


class GraphConstructor(nn.Module):
    def __init__(self, n_nodes, n_features, time_features=15*5*4*12, delta_min=0.05):
        super(GraphConstructor, self).__init__()
        ### 
        # n_nodes: int - number of nodes/assets
        # n_features: int - number of features after FC layers
        # time_features: int - dimension of one-hot encoded time vector
        # delta_min: float - minimum weight to consider in adjacency matrices

        self.n_nodes = n_nodes
        self.layer_initial = nn.init.xavier_normal((n_nodes, 1))
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

    def create_node_embedding(self, space, time):
        ###
        # space: Tensor (n_nodes, 1) - Spatial node embedding
        # time: Tensor (n_nodes * time_features, 1) - Temporal node embedding
        if space is None:
            space = self.layer_initial
        return self.spatial(space) + self.temporal(time)

    def create_adjacency_matrix(self, space, time1, time2):
        U1 = self.create_node_embedding(space, time1)
        U2 = self.create_node_embedding(space, time2)
        x = T.mm(T.mm(U1, self.B), T.transpose(U2, 0, 1))
        x = T.tensor([i if i >= self.delta_min else 0 \
            for i in T.flatten(x)]).reshape(x.shape)
        return F.softmax(x)


class DilatedGraphConvolutionCell(nn.Module):
    def __init__(self, kernel_size, n_data_features, n_output_features, 
    dilation_list, fc1_dims, fc2_dims, n_features, n_nodes, window_size):
        super(DilatedGraphConvolutionCell, self).__init__()
        self.kernel_size = kernel_size
        self.n_data_features = n_data_features
        self.n_output_features = n_output_features
        self.dilation_list = dilation_list
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_features = n_features
        self.n_nodes = n_nodes
        self.window_size = window_size

        self.FC = nn.Sequential(
            nn.Linear(n_nodes * n_data_features, fc1_dims),
            F.relu(),
            nn.Linear(fc1_dims, fc2_dims),
            F.relu(),
            nn.Linear(fc2_dims, n_features),
            F.relu()
        )

        self.W_forward = nn.Parameter(T.ones((n_features, n_output_features)))
        self.W_backward = nn.Parameter(T.ones((n_features, n_output_features)))
        self.b = nn.Parameter(T.ones((n_output_features)))

        self.optimizer = T.optim.Adam(self.parameters(), lr=1e-4)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def normalize_degree_matrix(self, space, time1, time2):
        graph = GraphConstructor(
            self.n_nodes, 
            self.n_features
        )
        adjacency_matrix = graph.create_adjacency_matrix(space, time1, time2)
        degree_matrix = adjacency_matrix.sum(-1)
        return T.mm(T.mm(T.pow(degree_matrix, -1/2), adjacency_matrix), T.pow(degree_matrix, 1/2))

    def conv(self, space, time):
        Z = T.zeros((self.n_nodes, self.n_output_features))
        for k in range(self.kernel_size):
            L1 = self.normalize_degree_matrix(space, time - k, time)
            L2 = self.normalize_degree_matrix(space, time - k, time)
            X = self.FC(space)
            x = T.mm(T.mm(L1, X), self.W_forward) + T.mm(T.mm(L2, X), self.W_backward) \
                + self.b
            Z += F.relu(x)
        return Z

    def conv_layer(self, space, time, dilation_factor):
        for t in range(time - self.window_size, time):
            Z = self.conv(space, t)

    def STJGN_layer(self, space, time):
        layer = nn.Sequential(
            self.conv_layer(space, time, self.dilation_list[0]),
            self.conv_layer(space, time, self.dilation_list[1]),
            self.conv_layer(space, time, self.dilation_list[2]),
            self.conv_layer(space, time, self.dilation_list[3])
        )


