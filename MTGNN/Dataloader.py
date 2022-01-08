import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from MTGNN.Get_Data import CreateData

class dataloader(Dataset):
    def __init__(self, window, out_dim, filename=None):
        self._data = CreateData(tickers=None, From=None, To=None).getTensorData()
        self._window = window
        self._out_dim = out_dim

    def __len__(self):
        return self._data.size(-1) - self._window
    
    def __getitem__(self, idx):
        if idx + self._out_dim > self._data.size(-1):
            out_dim = self._data.size(-1) - idx
        out_dim = self._out_dim
        X = self._data[:, :, idx:self._window + idx]
        y = self._data[:, :, idx + self._window:idx + self._window + out_dim]
        return X, y

    def __num_nodes__(self):
        return self._data.size(1)
    
    def __node_dim__(self):
        return self._data.size(0)