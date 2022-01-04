import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

class dataloader(Dataset):
    def __init__(self):
        