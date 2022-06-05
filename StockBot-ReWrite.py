import os
import configparser
# import quandl (premium api, requires payment)
import yfinance as yf
from datetime import date, timedelta
import numpy as np

# Machine Learning
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Linear Regression and Support Vector Regression (Weak Machine Learning Method's)
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# Long Term Short Memory
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, LSTM

CONFIG = "StockBot.cfg"
TITLE = "StockBot"
AUTHOR = "Ashwin Charathsandran"
VERSION = "1.0.2"
DATE_CURRENT = date.today().isoformat()

# config acess
def _settings(x):
	c = configparser.ConfigParser()
	c.read(CONFIG)
	y = c.get('settings', x)
	return y

# debug in console
def debug(x):
	if bool(_settings('debugMode')) == True:
		print(x)
	pass

# debug output to file
def log(x):
	if bool(_settings('debugMode')) == True:
		f = open('data.log', 'a', encoding='utf-8')
		f.write('\n' + DATE_CURRENT + ': ' + x + '\n')
		f.close()
	pass

def _shiftdate(x):
	return (date.today() - timedelta(days=int(x))).isoformat()

def _percentagechange(base, curr):
	return ((float(cur)-base) / abs(base)) * 100.00

# calculate exponential moving average
def _EMA(arr, prev_ema):

	N = 0
	for each in arr:
		N = N+1
	K = 2 / (N+1)
	last_close = float(arr[-1])
	prev_ema = float(prev_ema)

	EMA = float(((last_close - prev_ema) * K) + prev_ema)
	EMA = round(EMA, 4)
	return EMA

# calculate simple moving average (mean)
def _SMA(x):

	SUM = 0.00
	N = 0

	for i in x:
		N = N+1
		i = float(i)
		SUM = SUM + i
	
	SMA = round(SUM/N,4)
	return SMA

def _trend(x):

	closing_price = x[-1]
	long_term_sma = _SMA(x)
	long_term_ema = _EMA(x, x[-2])
	decision = None

	if closing_price < long_term_sma and closing_price < long_term_ema:
		decision = 'SELL'
	elif closing_price > long_term_sma and closing_price > long_term_ema:
		decision = 'BUY'
	else:
		decision = 'N/A (NO TREND CHANGES NOTICED)'
	return decision

def _quandl(id, start, end):
	quandl.ApiConfig.api_key = _settings('quandl')
	
	stock_data = quandl.get(id, start_date=start, end_date=end)
	stock_data = stock_data[['Adj. Close']]
	stock_data = stock_data.head()
	return stock_data

def _yahoo(mode, stockid, start, end):
	closing_prices = list()
	stock_name = yf.Ticker(stockid).info['longName']
	print('Downloading data for ' + stock_name + ' via Yahoo Finance.')
	raw_dataframe = yf.download(stockid, start=start, end=end)

	if 'closing' in mode:
		raw_dataframe = raw_dataframe['Adj Close']
		for price in raw_dataframe:
			price = round(price, 4)
			closing_prices.append(price)
		return closing_prices
	elif 'raw' in mode:
		#print(raw_dataframe)
		return raw_dataframe
	pass

def _checktrend(stockid):
	long_term = list()
	long_term = _yahoo('closing', stockid, _shiftdate(200), DATE_CURRENT)
	decision = _trend(long_term)
	stock_name = yf.Ticker(stockid).info['longName']

	if "N/A" in decision:
		print('\nNo trend changes have been observed for ' + stock_name  + '. (currently ' + str(long_term[-1]) + ')')
		log('\nNo trend changes have been observed for ' + stock_name  + '. (currently ' + str(long_term[-1]) + ')\n')
	else:
		print('\nTrend change to ' + decision + " " + stock_name + " has been observed. " + '(currently ' + str(long_term[-1]) + ')')
		log('\nTrend change to ' + decision + " " + stock_name + " has been observed. " + '(currently ' + str(long_term[-1]) + ')\n')

def _machinelearning(stockid):

	# Constants
	data_frame = _yahoo('raw', stockid, (date.today()-timedelta(days=200)).isoformat(), DATE_CURRENT)
	# print(data_frame)
	data_shift = 5

	# copy the closing prices and shift them 'data_shift' values up
	data_frame['Adj Close'] = round(data_frame['Adj Close'], 4)
	data_frame['Prediction'] = round(data_frame['Adj Close'].shift(-data_shift), 4)
	#print(data_frame)

	# create x parameter
	x = np.array(data_frame.drop(['Prediction'], 1))
	x = x[:-data_shift]

	# create y parameter
	y = np.array(data_frame['Prediction'])
	y = y[:-data_shift]

	# start feeding input
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

	# default kernel rbf
	svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
	svr_rbf.fit(x_train, y_train)
	x_forecast = np.array(data_frame.drop(['Prediction'], 1))[-data_shift:]
	svr_confidence = (svr_rbf.score(x_test, y_test))*100
	print('SVR RBF Confidence:' + str(svr_confidence) + '%')
	
	svr_rbf_prediction = svr_rbf.predict(x_forecast)
	#print(svr_rbf_prediction)

	
	lr = LinearRegression()
	lr.fit(x_train, y_train)
	lr_confidence = (lr.score(x_test, y_test))*100
	print('Linear Regression Confidence: ' + str(lr_confidence) + '%')
	x_forecast = np.array(data_frame.drop(['Prediction'], 1))[-data_shift:]

	# predict future values from array
	lr_prediction = lr.predict(x_forecast)
	#print(lr_prediction)

	return lr_prediction[0]

def _lraverages():
	lr_average = list()
	i = 0
	while i < 10:
		lr_average.append(_machinelearning('ACB.TO'))
		i = i+1
	for price in lr_average:
		total = 0.00
		total = total + price
		print(total)
	total = total/i
	print(total)

def func():
	_checktrend('ACB.TO')
	_lraverages()
	# _machinelearning('CGC')

if __name__ == "__main__":
	func()