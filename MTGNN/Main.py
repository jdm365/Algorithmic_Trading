import torch as T
import numpy as np
import os

from torch.utils.data.dataloader import DataLoader
from Model_Main import *
from Dataloader import dataloader
from tqdm import tqdm

seq_length = 168
out_dim = 5
# X(batch_size, node_dim, num_nodes, seq_length)
X = dataloader(window=seq_length, out_dim=5)

model = MTGNN(
    gcn_true=True,
    build_adj=True,
    gcn_depth=2,
    num_nodes=X.__num_nodes__(),
    kernel_set=[2, 3, 6, 7],
    kernel_size=2,
    dropout=.2,
    subgraph_size=X.__num_nodes__(),
    node_dim=X.__node_dim__(),
    dilation_exponential=2,
    conv_channels=32,
    residual_channels=32,
    skip_channels=64,
    end_channels=128,
    seq_length=seq_length,
    in_dim=X.__node_dim__(),
    out_dim=out_dim,
    layers=3,
    propalpha=.05,
    tanhalpha=3,
    layer_norm_affine=True,
    xd=None
)
optimizer = T.optim.Adam(model.parameters(), lr=.01)

if __name__ == '__main__':
    num_epochs = 100
    train_loader = DataLoader(X, batch_size=32, shuffle=False)
    for epoch in tqdm(range(num_epochs)):
        for X_in, y in train_loader:
            y_hat = model.forward(X_in)
            loss = T.mean((y_hat - y)**2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    T.save(model.state_dict())