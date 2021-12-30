from Agent_ddpg import Agent
import numpy as np
from Trading_Environment_SS import Correlation_Env
#from utils import plotLearning


env = Correlation_Env()

agent = Agent(input_dims=env.input_dims, tau=.001, env=env, 
    batch_size=64, n_actions=1)

PnL_history = []
for i in range(100):
    done = False
    capital = 10000
    observation = env.reset()
    while not done:
        action = agent.choose_action(observation)
        state_, reward, done = env.step(action)
        agent.remember(observation, action, reward, state_, done)
        agent.learn()
        capital *= reward
        observation = state_

    PnL_history.append(capital)
    print('Episode', i+1, 'profit: %.2f' % (10000-capital), 
        '10 game MA: %.2f' % np.mean(PnL_history[-10:]))
    if i % 5 == 0:
        agent.save_models()

filename = 'SingleStockRNNLearning.png'
#plotLearning(score_history, filename, window=5)