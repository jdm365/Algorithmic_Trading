from Finnhub import CreateTrainData, CreateTestData
import pandas as pd
import os
import torch as T


Tickers =  ['BINANCE:BTCUSDT', 'BINANCE:ETHUSDT', 'BINANCE:LTCUSDT', 'BINANCE:ADAUSDT', 
    'BINANCE:XRPUSDT', 'BINANCE:BCHUSDT', 'BINANCE:DOGEUSDT', 'BINANCE:BNBUSDT', 
    'BINANCE:MATICUSDT', 'BINANCE:XMRUSDT', 'BINANCE:SOLUSDT']

#train_data = CreateTrainData(tickers=Tickers, From='2021-02-01', To='2021-12-01')
#train_data.saveGlobalPriceTensor()


test_data = CreateTestData(tickers=Tickers, From='2020-08-01', To='2020-12-01')
test_data.saveGlobalPriceTensor()