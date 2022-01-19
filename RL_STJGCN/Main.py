import torch as T
import shutup
import numpy as np
from torch._C import device
from tqdm import tqdm
from RL_STJGCN.Long_Only import GetData, Agent
from RL_STJGCN.Long_Short import Agent as ShortAgent
from utils import plot_learning

class Trainer():
    def __init__(self, trade_frequency, minibatch_size=30, long_only=True, cuda=False):
        self.trade_frequency = trade_frequency
        self.minibatch_size = minibatch_size
        self.long_only = long_only
        self.margin = 1.5
        T.cuda.is_available = lambda: cuda
        self.device = 'cuda:0' if T.cuda.is_available() else 'cpu'
        self.n_time_features = 36 + 4
        if trade_frequency == 'Hourly':
            self.n_time_features = 36 + 6

    
    def train(self, n_epochs):
        shutup.please()
        X = GetData(self.trade_frequency).make_global_tensor_no_time()
        M = GetData(self.trade_frequency).make_global_temporal_tensor()
        if self.long_only:
            agent = Agent(
                kernel_size=2, 
                n_data_features=4, 
                dilation_list=[2, 2, 4, 2, 2], 
                fc1_dims=256, 
                fc2_dims=512, 
                n_features=64, 
                n_nodes=X.shape[0], 
                lookback_window=64,
                minibatch_size=self.minibatch_size,
                n_time_features=self.n_time_features
            )
        else:
            agent = ShortAgent(
                kernel_size=2, 
                n_data_features=4, 
                dilation_list=[2, 2, 4, 2, 2], 
                fc1_dims=256, 
                fc2_dims=512, 
                n_features=64, 
                n_nodes=X.shape[0], 
                lookback_window=64,
                minibatch_size=self.minibatch_size,
                margin=self.margin,
                n_time_features=self.n_time_features
            )
        Profit_History = []
        for epoch in tqdm(range(n_epochs)):
            done = False
            time_initial = np.random.randint(agent.network.lookback_window, \
                X.shape[-1] - agent.minibatch_size)
            Reward = 0
            cntr = 0
            capital = 10000
            last_action = (self.margin * ((T.rand(X.shape[0])).softmax(dim=0))).to(self.device)
            if self.long_only:
                last_action = (T.rand(X.shape[0])).softmax(dim=0).to(self.device)
            while done is False:
                observation = X[:, :, time_initial + cntr - agent.network.lookback_window:cntr + time_initial]
                time_features = M[time_initial + cntr - agent.network.lookback_window:cntr + time_initial, :]
                last_action, reward = agent.step(observation, time_features, last_action)
                Reward += reward
                capital *= T.exp(reward * agent.minibatch_size)
                cntr += 1
                if cntr % agent.minibatch_size == 0:
                    done = True
            Loss = -Reward.to(self.device)
            Loss.backward()

            agent.optimizer.step()
            agent.network.optimizer.step()
            agent.network.STJGCN.optimizer.step()
            agent.network.STJGCN.graph.optimizer.step()

            agent.optimizer.zero_grad()
            agent.network.optimizer.zero_grad()
            agent.network.STJGCN.optimizer.zero_grad()
            agent.network.STJGCN.graph.optimizer.zero_grad()

            Profits = capital - 10000
            Profit_History.append(Profits.detach().numpy())
            History = np.mean(Profit_History[-190:])

            if epoch % 109 == 0:
                print(f'Episode profits: {History}')
        plot_learning(Profit_History)
        T.save(agent.state_dict(), 'RL_STJGCN_model.pt')

if __name__ == '__main__':
    DataFrequency = ['Minute', 'Hourly']
    Train = Trainer(DataFrequency[1], cuda=False)
    Train.train(n_epochs=5000)
