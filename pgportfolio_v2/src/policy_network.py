import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(
            self,
            lr,
            input_dims,
            lookback_period,
            n_actions
            ):
        super(PolicyNetwork, self).__init__()
        self.input_dims = input_dims
        self.lr = lr

        assert lookback_period > 24, "Lookback period must be greater than 0"

        self.n_actions = n_actions

        # Define the layers of the network
        conv0 = nn.Conv1d(in_channels=3,   out_channels=64,  kernel_size=3)
        conv1 = nn.Conv1d(in_channels=64,  out_channels=128, kernel_size=8)
        conv2 = nn.Conv1d(in_channels=128, out_channels=1, kernel_size=13)

        self.conv_net = nn.Sequential(
            conv0,
            nn.GeLU(),
            nn.LayerNorm(32),
            nn.Dropout(0.2),
            conv1,
            nn.GeLU(),
            nn.LayerNorm(64),
            nn.Dropout(0.2),
            conv2,
            nn.GeLU(),
            nn.LayerNorm(128),
            nn.Dropout(0.2),
        )

        # Define the fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(lookback_period - 24, 1024),
            nn.GeLU(),
            nn.LayerNorm(1024),
            nn.Linear(1024, 1)
        )

        # Define the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        # Define the device
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        # Move the network to the device
        self.to(self.device)

    def forward(self, X):
        X = self.conv_net(X)
        X = self.fc(X)
        return X

