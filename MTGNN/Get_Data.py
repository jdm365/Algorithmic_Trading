import requests
import pandas as pd
import numpy as np
import torch as T
import requests
import os
from MTGNN.config import API_KEY_TWELVE


class CreateData:
    def __init__(self, tickers, From, To):
        self.tickers = tickers
        self.From = From
        self.To = To
        try:
            os.chdir('/Users/jakemehlman/Algorithmic_Trading/')
            if not os.path.exists('Stocks_Data'):
                os.makedirs('Stocks_Data')
        except FileNotFoundError:
            print("Can't find file")

        self.filename = os.path.join('Stocks_Data', 'Stocks_Data_Test.csv')
    
    def makeDF(self):
        for idx, ticker in enumerate(self.tickers):
            params = {
                'apikey': API_KEY_TWELVE,
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
                'Close': stock_data['close'],
                'High': stock_data['high'],
                'Low': stock_data['low'],
                'Volume': stock_data['volume'],
                'Datetime': stock_data['datetime']
                })
            _df = pd.DataFrame(data).replace('0', np.NaN).fillna(method='ffill').fillna(method='bfill')

            if idx == 0:
                _DF = _df
            else:
                _DF = _DF.merge(_df, on='Datetime', how='left')
            print('...saving %s...' %ticker)

        _DF.set_index('Datetime', inplace=True)
        _DF = _DF[::-1]
        _DF.to_csv(self.filename)
        return self.filename

    def makeTensorData(self, filename=None):
        ### X_out - (feature_dims, stock_dims, time_steps)
        TensorFilename = 'Stocks_Data/TensorDataFile.pt'
        if filename is not None:
            data = pd.read_csv(filename, index_col=0)
            data.iloc[:, 1:] = data.iloc[:, 1:].astype(float)
        else:
            data = self.makeDF()
            data.iloc[:, 1:] = data.iloc[:, 1:].astype(float)

        Tensor = T.zeros(4, len(self.tickers), len(data.iloc[1:, 1]))
        for idx, col in enumerate(data.columns):
            Tensor[(idx+4)%4, (idx//4), :] = T.tensor(np.array(data.iloc[1:, idx]) / np.array(data.iloc[:-1, idx]))

        T.save(Tensor, TensorFilename)




if __name__ == '__main__':
    TICKERS = ['AAPL', 'INTC', 'NFLX', 'GOOG', 'GSPC', 'NIO', 'AMD', 'F', 'MU', 'NVDA', 
        'TSLA', 'UBER', 'PTON', 'XOM']
    data = CreateData(tickers=TICKERS, From='2021-01-01', To='2021-12-01')
    data.makeTensorData(data.filename)
