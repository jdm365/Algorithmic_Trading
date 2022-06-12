import torch as T
import torch.nn as nn
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, in_features, block_dims, kernel_size):
        super(ResidualBlock, self).__init__()
        padding = self.calculate_padding(kernel_size)

        if in_features != block_dims:
            self.proj = nn.Conv2d(in_features, block_dims, 1)

        self.residual_tower = nn.Sequential(
                nn.Conv2d(in_features, block_dims, kernel_size, padding=padding),
                nn.BatchNorm2d(block_dims),
                nn.ReLU(),
                nn.Dropout2d(p=0.2),
                nn.Conv2d(block_dims, block_dims, kernel_size, padding=padding),
                nn.BatchNorm2d(block_dims),
                nn.ReLU(),
                nn.Dropout2d(p=0.2),
                nn.Conv2d(block_dims, block_dims, kernel_size, padding=padding),
                nn.BatchNorm2d(block_dims),
                nn.ReLU()
                )

    def forward(self, X: T.tensor):
        if self.proj is not None:
            residual = self.proj(X)
            return self.residual_tower(X) + residual
        residual = X
        return self.residual_tower(X) + residual


    def calculate_padding(self, kernel_size):
        padding = (np.array(kernel_size) - 1) / 2
        return (padding[0], padding[1])



class ResnetMain(nn.Module):
    def __init__(self, in_features, block_dims, kernel_sizes, n_features=75): 
        super(ResnetMain, self).__init__()
        first_block = [ResidualBlock(in_features, block_dims, kernel_sizes[0])]
        other_blocks = [ResidualBlock(block_dims, block_dims, kernel_sizes[i])\
                        for i in range(1, len(kernel_sizes))]
        self.residual_tower = nn.ModuleList(first_block + other_blocks)
        self.output_module = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
            )


    def forward(self, X: T.tensor):
        if len(X.shape) != 4:
            X = X.unsqueeze(dim=0)
        out = self.residual_tower(X)
        out = T.mean(out, dim=1, keepdim=False)
        return self.output_module(out)
        

