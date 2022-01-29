from cmath import nan
from math import gamma
from random import randint
from Get_Data import GetData
from PPO_Agent import Agent
from PPO_Agent import Preproccess
import numpy as np
from numpy import NaN, random
from utils import plot_learning
from tqdm import tqdm

if __name__ == '__main__':
    data = GetData()
    agent = Agent()
    n_episodes = 500
    commission_rate = .0025
    gamma_comm = 1 - commission_rate

    figure_file = 'Profit_History,png'
    profit_history = []
    learn_iters = 0
    steps = 0

    for i in tqdm(range(n_episodes)):
        time_initial = random.randint(32, data.X_m.shape[0]-3072)
        minutely_data, daily_data, weekly_data = data.create_observation(time_initial)
        done = False
        cash = 10000
        equity = 0
        capital = cash + equity
        cntr = 0
        while not done:
            cntr += 1
            steps += 1
            initial_capital = cash + equity
            initial_cash = cash
            initial_equity = equity

            previous_last_minutely_close = data.X_m[time_initial + cntr - 1, -2]
            last_minutely_close = data.X_m[time_initial + cntr, -2]

            action, prob, val, observation = agent.choose_action(minutely_data, daily_data, weekly_data)
            minutely_data, daily_data, weekly_data = data.create_observation(time_initial + cntr)
            delta_c = last_minutely_close - previous_last_minutely_close

            if action < 0:
                cash = (initial_equity + delta_c) * -action * gamma_comm + initial_cash
                equity = (initial_equity + delta_c) * (1 + action * gamma_comm)
            else:
                cash = initial_cash * (1 - action * gamma_comm)
                equity = (initial_equity + delta_c) + initial_cash * action * gamma_comm
            capital = cash + equity

            reward = (capital - initial_capital) / initial_capital
            if cntr >= 1024:
                done = True
                
            agent.remember(observation, action, prob, val, reward, done)
            
            if steps % agent.N == 0 and steps > 2048:
                agent.learn()
                learn_iters += 1

        profit_history.append(np.floor(capital) - 10000)
        print('Profit History Average: ', np.mean(profit_history[-100:]), 'n_steps: ', learn_iters)

    plot_learning(profit_history, filename=figure_file)