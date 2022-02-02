from GARCH_PPO.Get_Data import GetData
from PPO_Agent_RConvolutional import Agent
from PPO_Agent_RConvolutional import Preproccess
import numpy as np
from numpy import NaN, random
from utils import plot_learning
from tqdm import tqdm
import torch as T
import os

def train(n_episodes=500, commission_rate=.0025, reward_type='standard', ticker='.INX'):
    data = GetData(convolutional=True, ticker=ticker)
    agent = Agent()
    gamma_comm = 1#\ - commission_rate

    figure_file = 'Profit_History.png'
    profit_history = []
    BnH_profit_history = []
    learn_iters = 0
    steps = 0

    for i in tqdm(range(n_episodes)):
        time_initial = random.randint(50, data.X_m.shape[0]-3072)
        minutely_data, daily_data, weekly_data = data.create_observation(time_initial)
        done = False
        cash = 5000
        equity = 5000
        capital = cash + equity
        cntr = 0
        closes = []
        hx_M = T.zeros(2, minutely_data.shape[0], 64).to(agent.preprocess.device)
        hx_D = T.zeros(2, daily_data.shape[0], 64).to(agent.preprocess.device)
        hx_W = T.zeros(2, weekly_data.shape[0], 64).to(agent.preprocess.device)
        while not done:
            steps += 1
            initial_cash = cash
            initial_equity = equity
            initial_capital = cash + equity

            last_close = data.X_m[time_initial + cntr - 1, -2]
            action, prob, val, observation, hx_M, hx_D, hx_W = \
                agent.choose_action(minutely_data, daily_data, weekly_data, hx_M, hx_D, hx_W)
            cntr += 1
            minutely_data, daily_data, weekly_data = data.create_observation(time_initial + cntr)
            close = data.X_m[time_initial + cntr - 1, -2]

            delta_c = ((close - last_close) / last_close) * initial_equity
            closes.append(close)
            running_mean_long = np.mean(closes[-21:])
            running_mean_medium = np.mean(closes[-13:])
            running_mean_short = np.mean(closes[-8:])
            if action < 0:
                cash = (initial_equity + delta_c) * -action * gamma_comm + initial_cash
                equity = (initial_equity + delta_c) * (1 + action)
            else:
                cash = initial_cash * (1 - action)
                equity = (initial_equity + delta_c) + initial_cash * action * gamma_comm
            capital = cash + equity
            delta_capital = (capital - initial_capital) / initial_capital
            
            if reward_type == 'standard':
                reward = ((action * (close - last_close)) / last_close)
            elif reward_type == 'momentum':
                reward = (action * ((running_mean_medium - running_mean_long) / (running_mean_long))) + \
                    2*(action * ((running_mean_short - running_mean_medium) / (running_mean_medium))) + \
                    3*(action * ((last_close - running_mean_short) / (running_mean_short))) + \
                    4*(action * ((close - last_close) / (last_close)))
            elif reward_type == 'mean_reverting':
                reward = (action * ((running_mean_long - last_close) / last_close)) + \
                    ((action * (close - last_close)) / last_close)
            elif reward_type == 'traditional':
                reward = (capital - initial_capital)

            agent.remember(observation, action, prob, val, reward, done)
            
            if steps % agent.N == 0:
                agent.learn()
                learn_iters += 1
                done = True
            
        if learn_iters % 25 == 0:
            agent.save_models(reward_type)
        BnH_profits = ((closes[-1] / closes[2]) * 10000) - 10000

        BnH_profit_history.append(BnH_profits)
        profit_history.append(capital - 10000)
        print('Strategy:', reward_type, 'Episode Profits: $', profit_history[-1][0],\
            'Episode Relative Profits: $', (profit_history[-1][0] - BnH_profits).round(decimals=2),\
            'Relative Profit History Average: $', np.round(np.mean(profit_history[-100:])\
            - np.mean(BnH_profit_history[-100:]), decimals=2), 'n_steps:',\
            steps, 'Learning Steps: ', learn_iters)
        if i % 25 != 0:
            os.system('clear')

    plot_learning(profit_history, filename=figure_file)
    agent.save_models(reward_type)

def test(steps=20000, commission_rate=0.0025, ticker='.INX', strategies=['traditional', 'mean_reverting']):
    data = GetData(convolutional=True, ticker=ticker)
    agent_1 = Agent()
    agent_2 = Agent()
    agent_1.load_models(strategies[0])
    agent_2.load_models(strategies[1])
    gamma_comm = 1# - commission_rate

    time_initial = random.randint(32, data.X_m.shape[0]-(steps+250))
    minutely_data, daily_data, weekly_data = data.create_observation(time_initial)
    done = False

    cash_1 = 8000
    equity_1 = 2000
    capital_1 = cash_1 + equity_1

    cash_2 = 8000
    equity_2 = 2000
    capital_2 = cash_2 + equity_2

    max_drawdown_1 = 0
    capital_history_1 = []

    max_drawdown_2 = 0
    capital_history_2 = []

    closes = []
    cntr = 0
    while not done:
        initial_cash_1 = cash_1
        initial_equity_1 = equity_1
        initial_cash_2 = cash_2
        initial_equity_2 = equity_2

        last_close = data.X_m[time_initial + cntr - 1, -2]
        action_1 = agent_1.choose_action(minutely_data, daily_data, weekly_data)[0]
        action_2 = agent_2.choose_action(minutely_data, daily_data, weekly_data)[0]
        cntr += 1
        minutely_data, daily_data, weekly_data = data.create_observation(time_initial + cntr)
        close = data.X_m[time_initial + cntr - 1, -2]
        closes.append(close)

        delta_c_1 = ((close - last_close) / last_close) * initial_equity_1
        
        if action_1 < 0:
            cash_1 = (initial_equity_1 + delta_c_1) * -action_1 * gamma_comm + initial_cash_1
            equity_1= (initial_equity_1 + delta_c_1) * (1 + action_1)
        else:
            cash_1 = initial_cash_1 * (1 - action_1)
            equity_1 = (initial_equity_1 + delta_c_1) + initial_cash_1 * action_1 * gamma_comm
        capital_1 = cash_1 + equity_1

        capital_history_1.append(capital_1)
        if capital_1 == min(capital_history_1):
            max_drawdown_1 = capital_1[0] - 10000

        delta_c_2 = ((close - last_close) / last_close) * initial_equity_2

        if action_2 < 0:
            cash_2 = (initial_equity_2 + delta_c_2) * -action_2 * gamma_comm + initial_cash_2
            equity_2 = (initial_equity_2 + delta_c_2) * (1 + action_2)
        else:
            cash_2 = initial_cash_2 * (1 - action_2)
            equity_2 = (initial_equity_2 + delta_c_2) + initial_cash_2 * action_2 * gamma_comm
        capital_2 = cash_2 + equity_2

        capital_history_2.append(capital_2)
        if capital_2 == min(capital_history_2):
            max_drawdown_2 = capital_2[0] - 10000

        if cntr >= steps:
            done = True
    print(f'Total {strategy[0]} Profits: $', np.round((capital_1-10000)[0], decimals=2), 'Max Drawdown $', np.round(max_drawdown_1, decimals=2))
    print(f'Total {strategy[1]} Profits: $', np.round((capital_2-10000)[0], decimals=2), 'Max Drawdown $', np.round(max_drawdown_2, decimals=2))
    print('Total Buy and Hold Profits: $', np.round(10000 * (closes[-1] / closes[0]) - 10000, decimals=2))
    os.system('clear')

if __name__ == '__main__':
    strategies = ['standard', 'momentum', 'mean_reverting']
    for strategy in strategies:
        train(n_episodes=500, reward_type=strategy, ticker='.INX2')
    
    n_backtests = 5
    for _ in range(n_backtests):
        test(ticker='.INX2', strategies=strategies[1:3])
