import torch as T
from Agent_FullyConvolutional import Agent
from Trading_Env import DataFeatures, TradingEnv
import numpy as np
from pgportfolio.DDPG.utils import *
import time
import random

DATE = '2021-12-19'
env = TradingEnv(DATE)


agent = Agent(env=env, alpha=3e-4, beta=3e-4, reward_scale=5)

PnL_History, PnL_HistoryBH = [], []
cntr = 0
for i in range(1000):
    done = False
    capital = 10000
    buy_and_hold = capital
    train_batch_window = 500 #env.TIME_STEPS
    time_initial = random.randint(0, env.TIME_STEPS-train_batch_window)
    observation, last_action = env.reset(time_initial)
    last_action = np.ones((12,1)) #Random initialization for exploration
    last_action = np.exp(last_action) / np.sum(np.exp(last_action), axis=0)
    buy_and_hold_action = last_action
    start = time.time()
    while not done:
        action = agent.choose_action(observation, last_action)
        new_state, last_action, reward, done = env.step(action, last_action, time_initial, train_batch_window)
        agent.memory.store_transition(observation, last_action, action, reward, new_state, done)
        agent.update(train_batch_window)
        capital *= np.exp(reward * train_batch_window)
        buy_and_hold *= env.buyAndHoldReturns(buy_and_hold_action)
        cntr += 1
        observation = new_state

        #if env.time_step % 500 == 0:
        #    print('Capital: %.2f' %capital)
        #    print('buy_and_hold: %.2f' %buy_and_hold, '\n Time: %d' %env.time_step, '\n')


    end = time.time()
    EpisodeTime = time.strftime('%H:%M:%S', time.gmtime(end-start))
    print(f'Episode run time: {EpisodeTime}')
    PnL_History.append(capital - 10000)
    PnL_HistoryBH.append(buy_and_hold - 10000)

    print('Episode', i+1, 'Profit: %.2f' % (capital - 10000), 
        '40 episode MA: %.2f' % np.mean(PnL_History[-40:]))
    print('Episode', i+1, 'Profit (Buy and Hold): %.2f' % (buy_and_hold - 10000), 
        '40 episode MA: %.2f' % np.mean(PnL_HistoryBH[-40:]), '\n')
    print(f'Time Steps: {cntr}')
    
    if i % 50 == 0:
        agent.save_models()
filename = 'PnL_First.png'
plotLearning(PnL_History, filename, window=5)