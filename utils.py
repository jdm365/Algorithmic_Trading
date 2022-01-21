import numpy as np
import matplotlib.pyplot as plt
import torch as T

def plot_learning(scores, filename=None, x=None, window=5):   
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

class BuyAndHold():
    def __init__(self, data):
        self.inital_action = T.ones(data.shape[0], device=self.device) / data.shape[0]

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def step(self, observation):
        price_change_vector = observation[:, 2, -1]
        profit = T.dot(self.inital_action, price_change_vector)
        return self.inital_action, profit


