import torch as T
import shutup
import numpy as np
from tqdm import tqdm
from RL_STJGCN.Long_Only import GetData, Agent, AttentionOutputModule, DilatedGraphConvolutionCell, GraphConstructor


if __name__ == '__main__':
    shutup.please()
    T.cuda.is_available = lambda: False
    n_epochs = 1000
    X = GetData().make_global_tensor_no_time()
    M = GetData().make_global_temporal_tensor()
    agent = Agent(
        kernel_size=2, 
        n_data_features=4, 
        dilation_list=[2, 4, 2, 2], 
        fc1_dims=256, 
        fc2_dims=512, 
        n_features=64, 
        n_nodes=X.shape[0], 
        lookback_window=64,
        minibatch_size=60
    )
    Profit_History = []
    for epoch in tqdm(range(n_epochs)):
        done = False
        time_initial = np.random.randint(agent.network.lookback_window, \
            X.shape[-1] - agent.minibatch_size)
        Reward = 0
        cntr = 0
        capital = 10000
        last_action = (T.rand(X.shape[0])).softmax(dim=0).to('cuda:0' if T.cuda.is_available() else 'cpu')
        while done is False:
            observation = X[:, :, time_initial + cntr - agent.network.lookback_window:cntr + time_initial]
            time_features = M[time_initial + cntr - agent.network.lookback_window:cntr + time_initial, :]
            last_action, reward = agent.step(observation, time_features, last_action)
            Reward += reward
            capital *= T.exp(reward * agent.minibatch_size)
            cntr += 1
            if cntr % agent.minibatch_size == 0:
                done = True
        Loss = -Reward
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
        History = np.mean(Profit_History[-40:])

        if epoch % 10 == 0:
            print(f'Episode profits: {History}')

    T.save(agent.state_dict(), 'RL_STJGCN_model.pt')
