import numpy as np
import matplotlib.pyplot as plt
import torch as T
import torch.nn as nn

def plot_learning(scores, filename=None, x=None, window=100):   
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Profits')       
    plt.xlabel('Episode')                     
    plt.plot(x, running_avg)
    if filename:
      plt.savefig(filename)

class BuyAndHold(nn.Module):
    def __init__(self, data):
        super(BuyAndHold, self).__init__()
        self.inital_action = T.ones(data.shape[0]) / data.shape[0]

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def step(self, observation):
        price_change_vector = observation[:, 2, -1]
        profit = T.dot(self.inital_action.to(self.device), price_change_vector)
        return profit


