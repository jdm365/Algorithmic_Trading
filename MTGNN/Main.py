import torch as T
import numpy as np
import os
from Model_Main import *
from Dataloader import dataloader
from tqdm import tqdm

# X(batch_size, node_dim, num_nodes, seq_length)
X = dataloader().load('Stock_Data')

model = MTGNN(
    gcn_true=True,
    build_adj=True,
    gcn_depth=2,
    num_nodes=X.size(2),
    kernel_set=[2, 3, 6, 7],
    kernel_size=2,
    dropout=.2,
    subgraph_size=X.size(2),
    node_dim=X.size(1),
    dilation_exponential=2,
    conv_channels=32,
    residual_channels=32,
    skip_channels=64,
    end_channels=128,
    seq_length=12,
    in_dim=3,
    out_dim=5,
    layers=3,
    propalpha=.05,
    tanhalpha=3,
    layer_norm_affine=True,
    xd=None
)
optimizer = T.optim.Adam(model.parameters(), lr=.01)

if __name__ == '__main__':
    num_epochs = 100
    for epoch in tqdm(range(num_epochs)):
        X_in = dataloader
        y_hat = model.forward(X_in)
        loss = T.mean((y_hat - y)**2)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()