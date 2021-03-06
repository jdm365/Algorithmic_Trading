from GARCH_PPO.Get_Data import GetData
from PPO_Agent_RConvolutional import Agent
from PPO_Agent_RConvolutional import Preproccess
import numpy as np
from numpy import NaN, random
from utils import plot_learning
from tqdm import tqdm
import torch as T
import os
import shutup

def train(n_episodes=500, commission_rate=.0025, reward_type='standard', ticker='.INX'):
    shutup.please()
    data = GetData(recurrent=True, ticker=ticker)
    agent = Agent()
    gamma_comm = 1#\ - commission_rate

    figure_file = 'Profit_History.png'
    return_history = []
    sharpe_history = []
    learn_iters = 0
    steps = 0
    starting_capital = 10000

    for i in tqdm(range(n_episodes), desc=f'{reward_type}'):
        device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        time_initial = random.randint(50, data.X_m.shape[0]-3072)
        minutely_data, daily_data, weekly_data = data.create_observation(time_initial)
        done = False
        cash = starting_capital / 2
        equity = starting_capital / 2
        capital = starting_capital
        cntr = 0
        closes = []
        return_history = []
        hx_M = T.zeros(2, 64)
        hx_D = T.zeros(2, 64)
        hx_W = T.zeros(2, 64)
        while not done:
            agent.preprocess.to('cpu')
            agent.actor.to('cpu')
            agent.critic.to('cpu')
            steps += 1
            initial_cash = cash
            initial_equity = equity
            initial_capital = cash + equity

            last_close = data.X_m[time_initial + cntr - 1, -2]
            action, prob, val, minutely_data, daily_data, weekly_data, hx_M, hx_D, hx_W = \
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
                reward = ((capital - initial_capital) / initial_capital) - ((equity - initial_equity) / initial_equity)
            agent.remember(minutely_data, daily_data, weekly_data, hx_M, hx_D, hx_W, action, prob, val, reward, done)

            return_history.append((1 + ((capital - initial_capital) / initial_capital)))
            
            if steps % agent.N == 0:
                agent.preprocess.to(device)
                agent.actor.to(device)
                agent.critic.to(device)
                agent.learn()
                learn_iters += 1
                done = True
            
        if learn_iters % 25 == 0:
            agent.save_models(reward_type)

        volatility = (np.std(return_history)) * np.sqrt(24)
        portfolio_expected_return = np.mean(return_history) ** 24
        market_rate = (np.mean((np.array(closes[1:]) - np.array(closes[:-1])) / np.array(closes[:-1])) + 1) ** 24
        risk_free_rate = 1.0001

        #sharpe = (portfolio_expected_return - market_rate) / volatility
        sharpe = (portfolio_expected_return - risk_free_rate) / volatility
        sharpe_history.append(sharpe)

        print('Strategy:', reward_type, 'Episode Returns:', 100 * (capital - starting_capital) / starting_capital,\
            '% Episode Sharpe Ratio: ', np.round(sharpe, decimals=4),\
            'Sharpe Ratio Average:', np.round(np.mean(sharpe_history[-100:]), decimals=4),\
            'n_steps:', steps, 'Learning Steps: ', learn_iters)

        if i % 5 == 0:
            os.system('clear')
    
    print('Strategy:', reward_type, 'Episode Returns:', 100 * (capital - starting_capital) / starting_capital,\
           'Episode Sharpe Ratio: ', np.round(sharpe, decimals=4),\
           'Sharpe Ratio Average:', np.round(np.mean(sharpe_history[-100:]), decimals=4),\
           'n_steps:', steps, 'Learning Steps: ', learn_iters)

    plot_learning(return_history, filename=figure_file)
    agent.save_models(reward_type)

def test(steps=20000, commission_rate=0.0025, ticker='.INX', strategies=['traditional', 'mean_reverting']):
    data = GetData(recurrent=True, ticker=ticker)
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
    strategies = ['traditional', 'momentum', 'mean_reverting']
    for strategy in strategies:
        train(n_episodes=1000, reward_type=strategy, ticker='.INX2')
    
    #n_backtests = 5
    #for _ in range(n_backtests):
    #    test(ticker='.INX2', strategies=strategies[1:3])
