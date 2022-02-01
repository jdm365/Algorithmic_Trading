from GARCH_PPO.Get_Data import GetData
from PPO_Agent_RConvolutional import Agent
from PPO_Agent_RConvolutional import Preproccess
import numpy as np
from numpy import NaN, random
from utils import plot_learning
from tqdm import tqdm
import torch as T

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
        cash = 8000
        equity = 2000
        capital = cash + equity
        cntr = 0
        closes = []
        hx_M = T.zeros(1, 64) 
        hx_D = T.zeros(1, 64)
        hx_W = T.zeros(1, 64)
        while not done:
            steps += 1
            initial_cash = cash
            initial_equity = equity
            initial_capital = cash + equity

            last_close = data.X_m[time_initial + cntr, -2]
            action, prob, val, observation, hx_M, hx_D, hx_W = agent.choose_action(minutely_data, daily_data, weekly_data, hx_M, hx_D, hx_W)
            cntr += 1
            minutely_data, daily_data, weekly_data = data.create_observation(time_initial + cntr)
            close = data.X_m[time_initial + cntr, -2]

            delta_c = ((close - last_close) / last_close) * initial_equity
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
            
            if reward_type == 'standard':
                reward = ((action * (close - last_close)) / last_close) ## standard reward
            elif reward_type == 'momentum':
                reward = (action * ((close - last_close) ** 3 / (last_close))) ## momentum reward
            elif reward_type == 'mean_reverting':
                reward = (action * ((running_mean - last_close) / last_close)) + \
                    ((action * (close - last_close)) / last_close) ## mean reverting reward
            elif reward_type == 'traditional':
                reward = (capital - initial_capital)

            agent.remember(observation, action, prob, val, reward, done)
            
            if steps % agent.N == 0 and steps > 2048:
                agent.learn()
                learn_iters += 1
                done = True
            
        if learn_iters % 100 == 0:
            agent.save_models(reward_type)
        BnH_profits = ((closes[-1] / closes[2]) * 10000) - 10000

        BnH_profit_history.append(BnH_profits)
        profit_history.append(capital - 10000)
        print('Strategy:', reward_type, 'Episode Profits: $', profit_history[-1][0], 
            'Episode Relative Profits: $', (profit_history[-1][0] - BnH_profits).round(decimals=2),\
            'Relative Profit History Average: $', np.round(np.mean(profit_history[-100:])\
            - np.mean(BnH_profit_history[-100:]), decimals=2), 'n_steps:',\
            steps, 'Learning Steps: ', learn_iters)

    plot_learning(profit_history, filename=figure_file)
    agent.save_models(reward_type)

def test(steps=20000, commission_rate=0.0025, ticker='.INX'):
    data = GetData(convolutional=True, ticker=ticker)
    agent_MOM = Agent()
    agent_MR = Agent()
    agent_MOM.load_models('momentum')
    agent_MR.load_models('mean_reverting')
    gamma_comm = 1# - commission_rate

    time_initial = random.randint(32, data.X_m.shape[0]-(steps+250))
    minutely_data, daily_data, weekly_data = data.create_observation(time_initial)
    done = False
    cash_MOM = 8000
    equity_MOM = 2000
    capital_MOM = cash_MOM + equity_MOM
    cash_MR = 8000
    equity_MR = 2000
    capital_MR = cash_MR + equity_MR
    cntr = 0
    max_drawdown_MOM = 0
    capital_history_MOM = []
    max_drawdown_MR = 0
    capital_history_MR = []
    closes = []
    while not done:
        initial_cash_MOM = cash_MOM
        initial_equity_MOM = equity_MOM
        initial_cash_MR = cash_MR
        initial_equity_MR = equity_MR

        last_close = data.X_m[time_initial + cntr, -2]
        action_MOM = agent_MOM.choose_action(minutely_data, daily_data, weekly_data)[0]
        action_MR = agent_MR.choose_action(minutely_data, daily_data, weekly_data)[0]
        cntr += 1
        minutely_data, daily_data, weekly_data = data.create_observation(time_initial + cntr)
        close = data.X_m[time_initial + cntr, -2]
        closes.append(close)

        delta_c_MOM = ((close - last_close) / last_close) * initial_equity_MOM
        
        if action_MOM < 0:
            cash_MOM = (initial_equity_MOM + delta_c_MOM) * -action_MOM * gamma_comm + initial_cash_MOM
            equity_MOM = (initial_equity_MOM + delta_c_MOM) * (1 + action_MOM)
        else:
            cash_MOM = initial_cash_MOM * (1 - action_MOM)
            equity_MOM = (initial_equity_MOM + delta_c_MOM) + initial_cash_MOM * action_MOM * gamma_comm
        capital_MOM = cash_MOM + equity_MOM

        capital_history_MOM.append(capital_MOM)
        if capital_MOM == min(capital_history_MOM):
            max_drawdown_MOM = capital_MOM[0] - 10000

        delta_c_MR = ((close - last_close) / last_close) * initial_equity_MR

        if action_MR < 0:
            cash_MR = (initial_equity_MR + delta_c_MR) * -action_MR * gamma_comm + initial_cash_MR
            equity_MR = (initial_equity_MR + delta_c_MR) * (1 + action_MR)
        else:
            cash_MR = initial_cash_MR * (1 - action_MR)
            equity_MR = (initial_equity_MR + delta_c_MR) + initial_cash_MR * action_MR * gamma_comm
        capital_MR = cash_MR + equity_MR

        capital_history_MR.append(capital_MR)
        if capital_MR == min(capital_history_MR):
            max_drawdown_MR = capital_MR[0] - 10000

        if cntr >= steps:
            done = True
    print('Total Momentum Profits: $', np.round((capital_MOM-10000)[0], decimals=2), 'Max Drawdown $', np.round(max_drawdown_MOM, decimals=2))
    print('Total Mean Reversion Profits: $', np.round((capital_MR-10000)[0], decimals=2), 'Max Drawdown $', np.round(max_drawdown_MR, decimals=2))
    print('Total Buy and Hold Profits: $', np.round(10000 * (closes[-1] / closes[0]) - 10000, decimals=2))


if __name__ == '__main__':
    for strategy in ['mean_reverting', 'traditional']:
        train(n_episodes=500, reward_type=strategy, ticker='.INX')
    
    n_backtests = 5
    for _ in range(n_backtests):
        test(ticker='.INX')
