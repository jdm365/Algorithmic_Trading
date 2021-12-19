from pgportfolio.Market_Data.config import *
import requests
import pandas as pd
import numpy as np
import torch as T
import requests
import datetime
import os
import csv
import time
from pgportfolio.DDPG.utils import *


class CreateTrainData:
    def __init__(self, tickers, From, To):
        self.tickers = tickers
        self.From = From
        self.To = To
        self.parent_directory = '/Users/jakemehlman/Algorithmic_Trading/pgportfolio/Market_Data/Train_Data/'
        try:
            os.mkdir(os.path.join(self.parent_directory, str(datetime.date.today())))
            self.train_directory = os.mkdir(os.path.join(self.parent_directory, str(datetime.date.today())))
        except OSError as error:
            self.train_directory = os.path.join(self.parent_directory, str(datetime.date.today()))
    
    def makeDF(self):
        DFs = []
        try:
            os.mkdir(os.path.join(self.train_directory, 'CoinReturns'))
            self.returns_directory = os.mkdir(os.path.join(self.train_directory, 'CoinReturns'))
        except OSError as error:
            self.returns_directory = os.path.join(self.train_directory, 'CoinReturns')

        for ticker in self.tickers:
            FROM_ = int(pd.Timestamp(self.From).timestamp())
            TO_ = int(pd.Timestamp(self.To).timestamp())
            _DF = pd.DataFrame()
            done=False
            while not done:
                params = {
                    'token': API_KEY_FINNHUB,
                    'symbol': ticker,
                    'resolution': '30',
                    'from': FROM_,
                    'to': TO_
                    }

                FROM_ += 500*HALF_HOUR
                if FROM_ > TO_:
                    done = True
                api_result = requests.get('https://finnhub.io/api/v1/crypto/candle/', params)

                api_response = api_result.json()

                _df = pd.DataFrame(api_response)
                _df['t'] = pd.to_datetime(_df['t'], unit='s')
                _DF = pd.concat([_DF, _df], ignore_index=True)

            print('...saving %s' % ticker)
            _DF['Ticker'] = ticker
            _DF.set_index('Ticker', inplace=True)
            _DF = _DF[['v', 'c', 'h', 'l', 't']]
            _DF.rename(columns={'v': 'Volume', "c": "Close", "h": "High", 'l': 'Low',
                            't': 'Time'}, inplace=True)
            _DF.to_csv('%s/%s.csv' % (self.returns_directory, ticker))
            DFs.append(_DF)
            time.sleep(60)
            
        self.DFs = DFs
        self.time_steps = len(self.DFs[0]['Close'])
        return self.DFs

    def calculatePriceChangeVector(self):
        self.DFs = self.makeDF()
        Y = T.ones((1, len(self.DFs), self.time_steps-1), dtype=T.float32)
        for i in range(len(self.DFs)):
            stock = self.DFs[i]
            X = np.array(stock['Close'])
            Y[0,i,:] = T.tensor(X[1:]/X[:-1], dtype=T.float32)
        Z = T.ones(size=(1, 1, len(Y[0,0,:])), dtype=T.float32)
        self.y = T.cat((Z, Y), dim=1)
        return self.y

    def constructGlobalPriceTensor(self):
        self.DFs = self.makeDF()
        P = T.zeros(4, len(self.DFs), self.time_steps, dtype=T.float32)
        for i in range(len(self.DFs)):
            stock = self.DFs[i]
            price_data = np.array(stock[['Volume', 'Close', 'High', 'Low']])
            # (features, coins, training window)
            P[:,i,:] = T.reshape(T.tensor(price_data, dtype=T.float32), (4, self.time_steps))
        Z = T.ones(size=(len(P[:,0,0]), 1, len(P[0,0,:])), dtype=T.float32)
        self.P = T.cat((Z, P), dim=1)
        return self.P

    def saveGlobalPriceTensor(self):
        print('...saving prices to', self.train_directory, '...')
        T.save(self.constructGlobalPriceTensor(), '%s/Train_Data.pt' % self.train_directory)
        T.save(self.calculatePriceChangeVector(), '%s/Train_priceChangeVectors.pt' % self.train_directory)
        with open( '%s/Train_Data_Coins.csv' % self.train_directory, 'w') as f:
            write = csv.writer(f)  
            write.writerow(self.tickers)

class GetTrainData:
    def __init__(self, date):
        self.date = date    
        self.parent_directory = '/Users/jakemehlman/Algorithmic_Trading/pgportfolio/Market_Data/Train_Data/'

    def loadTrainData(self):
        P = T.load('%s/%s/Train_Data.pt' % (self.parent_directory, self.date))
        y = T.load('%s/%s/Train_priceChangeVectors.pt' % (self.parent_directory, self.date))
        return P, y


class CreateTestData:
    def __init__(self, tickers, From, To):
        self.From = From
        self.To = To
        self.tickers = tickers
        self.parent_directory = '/Users/jakemehlman/Algorithmic_Trading/pgportfolio/Market_Data/Test_Data/'
        try:
            os.mkdir(os.path.join(self.parent_directory, str(datetime.date.today())))
            self.train_directory = os.mkdir(os.path.join(self.parent_directory, str(datetime.date.today())))
        except OSError as error:
            self.train_directory = os.path.join(self.parent_directory, str(datetime.date.today()))
    
    def makeDF(self):
        DFs = []
        try:
            os.mkdir(os.path.join(self.train_directory, 'CoinReturns'))
            self.returns_directory = os.mkdir(os.path.join(self.train_directory, 'CoinReturns'))
        except OSError as error:
            self.returns_directory = os.path.join(self.train_directory, 'CoinReturns')

        for ticker in self.tickers:
            FROM_ = int(pd.Timestamp(self.From).timestamp())
            TO_ = int(pd.Timestamp(self.To).timestamp())
            _DF = pd.DataFrame()
            done=False
            while not done:
                params = {
                    'token': API_KEY_FINNHUB,
                    'symbol': ticker,
                    'resolution': '30',
                    'from': FROM_,
                    'to': TO_
                    }

                FROM_ += 500*HALF_HOUR
                if FROM_ > TO_:
                    done = True
                api_result = requests.get('https://finnhub.io/api/v1/crypto/candle/', params)

                api_response = api_result.json()

                _df = pd.DataFrame(api_response)
                _df['t'] = pd.to_datetime(_df['t'], unit='s')
                _DF = pd.concat([_DF, _df], ignore_index=True)

            print('...saving %s' % ticker)
            _DF['Ticker'] = ticker
            _DF.set_index('Ticker', inplace=True)
            _DF = _DF[['v', 'c', 'h', 'l', 't']]
            _DF.rename(columns={'v': 'Volume', "c": "Close", "h": "High", 'l': 'Low',
                            't': 'Time'}, inplace=True)
            _DF.to_csv('%s/%s.csv' % (self.returns_directory, ticker))
            DFs.append(_DF)
            time.sleep(60)
            
        self.DFs = DFs
        self.time_steps = len(self.DFs[0]['Close'])
        return self.DFs

    def calculatePriceChangeVector(self):
        self.DFs = self.makeDF()
        Y = T.ones((1, len(self.DFs), self.time_steps-1), dtype=T.float32)
        for i in range(len(self.DFs)):
            stock = self.DFs[i]
            X = np.array(stock['Close'])
            Y[0,i,:] = T.tensor(X[1:]/X[:-1], dtype=T.float32)
        Z = T.ones(size=(1, 1, len(Y[0,0,:])), dtype=T.float32)
        self.y = T.cat((Z, Y), dim=1)
        return self.y

    def constructGlobalPriceTensor(self):
        self.DFs = self.makeDF()
        P = T.zeros(4, len(self.DFs), self.time_steps, dtype=T.float32)
        for i in range(len(self.DFs)):
            stock = self.DFs[i]
            price_data = np.array(stock[['Volume', 'Close', 'High', 'Low']])
            # (features, coins, training window)
            P[:,i,:] = T.reshape(T.tensor(price_data, dtype=T.float32), (4, self.time_steps))
        Z = T.ones(size=(len(P[:,0,0]), 1, len(P[0,0,:])), dtype=T.float32)
        self.P = T.cat((Z, P), dim=1)
        return self.P

    def saveGlobalPriceTensor(self):
        print('...saving prices to', self.train_directory, '...')
        T.save(self.constructGlobalPriceTensor(), '%s/Test_Data.pt' % self.train_directory)
        T.save(self.calculatePriceChangeVector(), '%s/Test_priceChangeVectors.pt' % self.train_directory)
        with open( '%s/Test_Data_Coins.csv' % self.train_directory, 'w') as f:
            write = csv.writer(f)  
            write.writerow(self.tickers)

class GetTestData:
    def __init__(self, date):
        self.date = date    
        self.parent_directory = '/Users/jakemehlman/Algorithmic_Trading/pgportfolio/Market_Data/Test_Data/'
    
    def loadTrainData(self):
        P = T.load('%s/%s/Test_Data.pt' % (self.parent_directory, self.date))
        y = T.load('%s/%s/Test_priceChangeVectors.pt' % (self.parent_directory, self.date))
        return P, y



if __name__ == '__main__':
    Tickers =  ['BINANCE:BTCUSDT', 'BINANCE:ETHUSDT', 'BINANCE:LTCUSDT', 'BINANCE:ADAUSDT', 
    'BINANCE:XRPUSDT', 'BINANCE:BCHUSDT', 'BINANCE:DOGEUSDT', 'BINANCE:BNBUSDT', 
    'BINANCE:MATICUSDT', 'BINANCE:XMRUSDT', 'BINANCE:SOLUSDT']

    train_data = CreateTrainData(tickers=Tickers, From='2021-02-01', To='2021-12-01')
    train_data.saveGlobalPriceTensor()

    #test_data = CreateTestData(tickers=Tickers, From='2020-08-01', To='2020-12-01')
    #test_data.saveGlobalPriceTensor()