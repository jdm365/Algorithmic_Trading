import requests
import pandas as pd
import numpy as np
import torch as T
import requests
import os
import time
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

        self.filename = os.path.join('Stocks_Data', 'Stocks_Data_Test2.csv')
    
    def makeDF(self):
        for idx, ticker in enumerate(self.tickers):
            time.sleep(12)
            done = False
            API_response = []
            end = self.To
            while done is False:
                params = {
                    'apikey': API_KEY_TWELVE,
                    'symbol': ticker,
                    'interval': '30min',
                    'start_date': self.From,
                    'end_date': end,
                    'format': 'JSON',
                    'outputsize': '5000',
                    'timezone': 'exchange'
                    }

                api_result = requests.get('https://api.twelvedata.com/time_series', params)
                api_response = api_result.json()['values']
                API_response = API_response + api_response
                if len(api_response) != 5000:
                    print(len(api_response))
                    done = True
                end = api_response[-1]['datetime']
            data = []
            for stock_data in API_response:
                data.append({
                'Close': stock_data['close'],
                'High': stock_data['high'],
                'Low': stock_data['low'],
                'Volume': stock_data['volume'],
                'Datetime': stock_data['datetime']
                })
            _df = pd.DataFrame(data).drop_duplicates(subset=['Datetime']).replace('0', np.NaN)\
                .fillna(method='ffill').fillna(method='bfill').fillna(method='pad')

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
        self.TensorFilename = 'Stocks_Data/TensorDataFile.pt'
        if filename is not None:
            data = pd.read_csv(filename, index_col=0)
            data.iloc[:, 1:] = data.iloc[:, 1:].astype(float)
        else:
            data = self.makeDF()
            data.iloc[:, 1:] = data.iloc[:, 1:].astype(float)

        Tensor = T.zeros(4, len(self.tickers), len(data.iloc[1:, 1]))
        for idx, col in enumerate(data.columns):
            Tensor[(idx+4)%4, (idx//4), :] = T.tensor(np.array(data.iloc[1:, idx]) / np.array(data.iloc[:-1, idx]))

        T.save(Tensor, self.TensorFilename)

    def getTensorData(self, filename=None):
        self.TensorFilename = 'Stocks_Data/TensorDataFile.pt'
        if filename is None:
            filename = self.TensorFilename
        return T.load(self.TensorFilename)




if __name__ == '__main__':
    TICKERS = ['AAPL', 'INTC', 'NFLX', 'GOOG', 'PINS', 'NIO', 'AMD', 'F', 'MU', 'NVDA', 
        'TSLA', 'WFC', 'BAC', 'XOM']
    data = CreateData(tickers=TICKERS, From='2019-10-01', To='2021-12-01')
    data.makeDF()
    #data.makeTensorData(data.filename)
