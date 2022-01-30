from cgitb import reset
from cmath import nan
from math import gamma
from random import randint
from unicodedata import decimal
from Get_Data import GetData
from PPO_Agent import Agent
from PPO_Agent import Preproccess
import numpy as np
from numpy import NaN, random
from utils import plot_learning
from tqdm import tqdm

def train(n_episodes=500, commission_rate=.0025):
    data = GetData()
    agent = Agent()
    gamma_comm = 1 - commission_rate

    figure_file = 'Profit_History.png'
    profit_history = []
    learn_iters = 0
    steps = 0

    for i in tqdm(range(n_episodes)):
        time_initial = random.randint(50, data.X_m.shape[0]-3072)
        minutely_data, daily_data, weekly_data = data.create_observation(time_initial)
        done = False
        cash = 8000
        equity = 2000
        capital = cash + equity
        cntr = 0
        closes = []
        while not done:
            steps += 1
            initial_cash = cash
            initial_equity = equity
            initial_capital = cash + equity

            last_close = data.X_m[time_initial + cntr, -2]
            action, prob, val, observation = agent.choose_action(minutely_data, daily_data, weekly_data)
            cntr += 1
            minutely_data, daily_data, weekly_data = data.create_observation(time_initial + cntr)
            close = data.X_m[time_initial + cntr, -2]

            delta_c = close - last_close
            closes.append(close)
            running_mean = np.mean(closes[-30:])

            if action < 0:
                cash = (initial_equity + delta_c) * -action * gamma_comm + initial_cash
                equity = (initial_equity + delta_c) * (1 + action)
            else:
                cash = initial_cash * (1 - action)
                equity = (initial_equity + delta_c) + initial_cash * action * gamma_comm
            capital = cash + equity
            delta_capital = (capital - initial_capital) / initial_capital

            #reward = ((action * (close - last_close)) / last_close) ##standard reward
            reward = (action * ((close - last_close) / last_close) ** 3) ## momentum reward
            #reward = (action * ((running_mean - last_close) / last_close)) ## mean reverting reward
            if cntr >= 1024:
                done = True
            agent.remember(observation, action, prob, val, reward, done)
            
            if steps % agent.N == 0 and steps > 2048:
                agent.learn()
                learn_iters += 1
            
        if learn_iters % 100 == 0:
            agent.save_models()

        profit_history.append(capital - 10000)
        print('Episode Profits: $', profit_history[-1][0].round(decimals=2), 'Profit History Average: $',\
            np.mean(profit_history[-100:]).round(decimals=2), 'n_steps:', steps, 'Learning Steps: ', learn_iters)

    plot_learning(profit_history, filename=figure_file)

def test(steps=4000, commission_rate=0.0025):
    data = GetData()
    agent = Agent()
    gamma_comm = 1 - commission_rate

    time_initial = random.randint(32, data.X_m.shape[0]-(steps+250))
    minutely_data, daily_data, weekly_data = data.create_observation(time_initial)
    done = False
    cash = 8000
    equity = 2000
    capital = cash + equity
    cntr = 0
    while not done:
        steps += 1
        initial_cash = cash
        initial_equity = equity

        last_close = data.X_m[time_initial + cntr, -2]
        action = agent.choose_action(minutely_data, daily_data, weekly_data)[0]
        cntr += 1
        minutely_data, daily_data, weekly_data = data.create_observation(time_initial + cntr)
        close = data.X_m[time_initial + cntr, -2]

        delta_c = close - last_close

        if action < 0:
            cash = (initial_equity + delta_c) * -action * gamma_comm + initial_cash
            equity = (initial_equity + delta_c) * (1 + action)
        else:
            cash = initial_cash * (1 - action)
            equity = (initial_equity + delta_c) + initial_cash * action * gamma_comm
        capital = cash + equity

        if cntr >= steps:
            done = True
    print('Total Profits: $', np.round((capital-10000)[0], decimals=2))

if __name__ == '__main__':
    train(n_episodes=1000)
    n_backtests = 5
    for _ in range(n_backtests):
        test()
