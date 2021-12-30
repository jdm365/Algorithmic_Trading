import requests
import pandas as pd
import numpy as np
import torch as T
import requests
import os


class CreateData:
    def __init__(self, tickers, From, To):
        self.tickers = tickers
        self.From = From
        self.To = To
        if not os.path.exists('Stocks_Data'):
            os.makedirs('Stocks_Data')
    
    def makeDF(self):
        self.returns_directory = os.getcwd()
        _DF = pd.DataFrame()

        for ticker in self.tickers:
            _df = pd.DataFrame()
            params = {
                'apikey': '9dc69f38005c441c8186a293754d682e',
                'symbol': ticker,
                'interval': '30min',
                'start_date': self.From,
                'end_date': self.To,
                'format': 'JSON',
                'outputsize': '5000',
                'timezone': 'exchange'
                }

            api_result = requests.get('https://api.twelvedata.com/time_series', params)
            api_response = api_result.json()['values']
            
            data = []
            for stock_data in api_response:
                data.append({
                'Close': stock_data['close']
                })

            _df = pd.DataFrame(data)
            _df = _df.iloc[::-1].reset_index(drop=True)
            
            _DF = pd.concat([_DF, _df], axis=1)
            print('...saving %s...' %ticker)
        _DF.to_csv('/Users/jakemehlman/Stocks_Data/stock_data.csv')

if __name__ == '__main__':
    TICKERS = ['AAPL', 'NFLX', 'GOOG', 'GSPC', 'NIO', 'AMD', 'F', 'MU', 'NVDA', 
        'TSLA', 'UBER', 'PTON', 'XOM']
    data = CreateData(tickers=TICKERS, From='2021-01-01', To='2021-12-01')
    data.makeDF()