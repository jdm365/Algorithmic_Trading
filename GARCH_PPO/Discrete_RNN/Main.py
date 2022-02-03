from GARCH_PPO.Get_Data import GetData
from PPO_RAgent import Agent
import numpy as np
from numpy import NaN, random
from utils import plot_learning
from tqdm import tqdm
import shutup
import torch as T
import os

def train(n_episodes=500, commission_rate=.0025, reward_type='standard', ticker='.INX'):
    shutup.please()
    data = GetData(convolutional=True, ticker=ticker)
    agent = Agent()
    gamma_comm = 1 - commission_rate

    figure_file = 'Profit_History.png'
    profit_history = []
    sharpe_history = []
    learn_iters = 0
    steps = 0
    starting_capital = 100000

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

            action -= 1
            if action == 1 and initial_cash < close:
                cash = initial_cash
                equity = (initial_equity + delta_c)
            elif action == -1 and (initial_equity + delta_c) < close:
                cash = initial_cash
                equity = (initial_equity + delta_c)
            else:
                cash = initial_cash - (action * close * gamma_comm)
                equity = (initial_equity + delta_c) + (action * close * gamma_comm)
            action += 1
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

        volatility = (np.std(return_history))
        portfolio_expected_return = np.mean(return_history)
        market_rate = np.mean((np.array(closes[1:]) - np.array(closes[:-1])) / np.array(closes[:-1])) + 1
        risk_free_rate = 1
2
        #sharpe = (portfolio_expected_return - market_rate) / volatility
        sharpe = (portfolio_expected_return - risk_free_rate) / volatility
        sharpe_history.append(sharpe)

        print('Strategy:', reward_type, 'Episode Returns:', 100 * (capital - starting_capital) / starting_capital,\
            '% Episode Sharpe Ratio: ', np.round(sharpe, decimals=4),\
            'Sharpe Ratio Average:', np.round(np.mean(sharpe_history[-100:]), decimals=4),\
            'n_steps:', steps, 'Learning Steps: ', learn_iters)

        if i % 2 == 0:
            os.system('clear')
    
    print('Strategy:', reward_type, 'Episode Returns:', 100 * (capital - starting_capital) / starting_capital,\
           '% Episode Sharpe Ratio: ', np.round(sharpe, decimals=4),\
           'Sharpe Ratio Average:', np.round(np.mean(sharpe_history[-100:]), decimals=4),\
           'n_steps:', steps, 'Learning Steps: ', learn_iters)

    plot_learning(profit_history, filename=figure_file)
    agent.save_models(reward_type)

def test(steps=20000, commission_rate=0.0025, ticker='.INX', strategies=['traditional', 'mean_reverting']):
    shutup.please()
    data = GetData(convolutional=True, ticker=ticker)
    agent_1 = Agent()
    agent_2 = Agent()
    agent_1.load_models(strategies[0])
    agent_2.load_models(strategies[1])
    
    agent_1.preprocess.to('cpu')
    agent_1.actor.to('cpu')
    agent_1.critic.to('cpu')
    
    agent_2.preprocess.to('cpu')
    agent_2.actor.to('cpu')
    agent_2.critic.to('cpu')

    gamma_comm = 1 - commission_rate

    time_initial = random.randint(32, data.X_m.shape[0]-(steps+250))
    minutely_data, daily_data, weekly_data = data.create_observation(time_initial)
    done = False

    cash_1 = 500000
    equity_1 = 500000
    capital_1 = cash_1 + equity_1

    cash_2 = 500000
    equity_2 = 500000
    capital_2 = cash_2 + equity_2

    max_drawdown_1 = 0
    capital_history_1 = []

    max_drawdown_2 = 0
    capital_history_2 = []

    closes = []
    cntr = 0
    
    hx_M = T.zeros(2, 64)
    hx_D = T.zeros(2, 64)
    hx_W = T.zeros(2, 64)
    while not done:
        with T.no_grad():
            initial_cash_1 = cash_1
            initial_equity_1 = equity_1
            initial_cash_2 = cash_2
            initial_equity_2 = equity_2

            last_close = data.X_m[time_initial + cntr - 1, -2]
            action_1,  _, _, _, _, _, hx_M, hx_D, hx_W = agent_1.choose_action(minutely_data, daily_data, weekly_data, hx_M, hx_D, hx_W)
            action_2,  _, _, _, _, _, hx_M, hx_D, hx_W = agent_2.choose_action(minutely_data, daily_data, weekly_data, hx_M, hx_D, hx_W)
            cntr += 1
            minutely_data, daily_data, weekly_data = data.create_observation(time_initial + cntr)
            close = data.X_m[time_initial + cntr - 1, -2]
            closes.append(close)

            delta_c_1 = ((close - last_close) / last_close) * initial_equity_1

            action_1 -= 1
            if action_1 == 1 and initial_cash_1 < close:
                cash_1 = initial_cash_1
                equity_1 = (initial_equity_1 + delta_c_1)
            elif action_1 == -1 and (initial_equity_1 + delta_c_1) < close:
                cash_1 = initial_cash_1
                equity_1 = (initial_equity_1 + delta_c_1)
            else:
                cash_1 = initial_cash_1 - (action_1 * close * gamma_comm)
                equity_1 = (initial_equity_1 + delta_c_1) + (action_1 * close * gamma_comm)
            action_1 += 1
            capital_1 = cash_1 + equity_1

            capital_history_1.append(capital_1)
            if capital_1 == min(capital_history_1):
                max_drawdown_1 = capital_1 - 1000000

            delta_c_2 = ((close - last_close) / last_close) * initial_equity_2

            action_2 -= 1
            if action_2 == 1 and initial_cash_2 < close:
                cash_2 = initial_cash_2
                equity_2 = (initial_equity_2 + delta_c_1)
            elif action_2 == -1 and (initial_equity_2 + delta_c_2) < close:
                cash_2 = initial_cash_2
                equity_2 = (initial_equity_2 + delta_c_2)
            else:
                cash_2 = initial_cash_2 - (action_2 * close * gamma_comm)
                equity_2 = (initial_equity_2 + delta_c_1) + (action_2 * close * gamma_comm)
            action_2 += 1
            capital_2 = cash_2 + equity_2
            capital_history_2.append(capital_2)
            if capital_2 == min(capital_history_2):
                max_drawdown_2 = capital_2 - 1000000
            
            if cntr >= steps:
                done = True
    print(f'Total {strategies[0]} Profits: $', np.round((capital_1-1000000), decimals=2), \
        'Max Drawdown $', np.round(max_drawdown_1, decimals=2))
    print(f'Total {strategies[1]} Profits: $', np.round((capital_2-1000000), decimals=2), \
        'Max Drawdown $', np.round(max_drawdown_2, decimals=2))
    print('Total Buy and Hold Profits: $', np.round(1000000 * ((closes[-1] - closes[0]) / closes[0])\
        , decimals=2))


if __name__ == '__main__':
    strategies = ['traditional', 'mean_reverting']
    for strategy in [strategies[0]]:
        train(n_episodes=000, reward_type=strategy, ticker='.INX2')
    
    n_backtests = 3
    for _ in range(n_backtests):
        test(ticker='.INX2', strategies=strategies)
