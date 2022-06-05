# StockBot.py
# Ashwin Charathsandran

# To notify stock trends using Simple Moving Average:
# Calculate long-term SMA and short-term SMA, and compare
# Sell stock if the closing price goes below the long-term SMA.
# Buy stock if the closing price goes above the long-term SMA.
# Calculate SMA: SMA = (n - period_sum) / n
# Calcualte EMA: Current EMA = ((Price(current) - previous EMA) X multiplier) + previous EMA

# To notify stock trends using both Simple Moving Average and Exponential Moving Average:
# You should SELL stock, if the closing price goes below the long-term SMA;

# To predict stock trends:
# 1. Use linear regression and predict the stock values for the next five days.
# 2. Check trend notifications and see if they match the predicitions made through linear regression.
# 3.

# Comparing predictions with stock trends and see if they are correct.
	# 1. Use linear regression to check the stock values within the next 30 days.
	# 2. We check trend notifications and compare it to the values from linear regression.
	# 3. Comparing.
	# Comparison is done by seeing if the values are over the current closing price, and giving it an output of wheter to buy or sell.
	# For example:
	# LR Value: $34.23
	# Closing Price: $32.01
	# if trend is BUY and LR Value is greater than closing price, then SUCEED BUY.
	# if trend is SELL and LR Value is lower than closing price, then SUCEED SELL.
	# if trend is NONE and LR Value has a large margin from closing price, then invalid prediction.
	# to further support prediction, we will look at the confidence level of the trained sets and compare.
	# in addition, comparison between both long-term and short-term data sets will also be observed to see whether any major change is noted.
	# if both predictions by long-term and short-term have a short margin then the prediction is authorized.


import os
import configparser
import quandl
import yfinance as yf
from datetime import date, timedelta
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

DEBUG = True
CONFIG_URL = "StockBot.cfg"
TITLE = "StockBot.py"
AUTHOR = "Ashwin Charathsandran"
VERSION = "1.0 (Moving Average Trend Detection)"

def _setting(sub):
	C = configparser.ConfigParser()
	C.read(CONFIG_URL)
	secret = C.get('settings', sub)
	return secret

def debug(x):
	if DEBUG == True:
		print(str(x))
	pass

def log(file, x):
	f = open(file + ".log", 'a', encoding='utf-8')
	f.write("\n" + date.today().isoformat()+ ":" + x + "\n")
	f.close()

# Notifying Trends Changes

# Simple Moving Average
def _calculateSMA(arrayCP):

	PERIOD_SUM = 0.00
	SMA = 0.00
	w = 0

	for i in arrayCP:
		w = w+1
		i = float(i)
		PERIOD_SUM = PERIOD_SUM + i

	N = w
	SMA = (PERIOD_SUM) / N

	return round(SMA, 4)

# Exponential Moving Average
def _calculateEMA(arr, lastCP, prevEMA):

	N = 0
	for each in arr:
		N = N+1
	K = 2/(N + 1)
	lastCP = float(lastCP)
	prevEMA = float(prevEMA)

	EMA = float(((lastCP - prevEMA) * K) + prevEMA)

	return round(EMA, 4)

def _checkTrend(inpLong):

	LONG_TERM_SMA = _calculateSMA(inpLong)
	LONG_TERM_EMA = _calculateEMA(inpLong, inpLong[-1], inpLong[-2])
	CLOSING_PRICE = float(inpLong[-1])
	decision = None

	if CLOSING_PRICE < LONG_TERM_SMA and CLOSING_PRICE < LONG_TERM_EMA:
		decision = 'SELL'
	elif CLOSING_PRICE > LONG_TERM_SMA and CLOSING_PRICE > LONG_TERM_EMA:
		decision = 'BUY'
	else:
		decision = 'NONE'

	return decision

"""
Quandl is a premium stock data service,
and the free user is limited to only a certain amount of stocks.
"""
def _getQuandlData(stockName, dataName, start, end):

	# QUANDL is a paid api, only offers certain stocks
	quandl.ApiConfig.api_key = _getSecret('quandl')
	quandlStockData = quandl.get(stockName, start_date=start, end_date=end)
	if "closing" in dataName:
		quandlStockData = quandlStockData[['Adj. Close']]
	elif "volume" in dataName:
		quandlStockData = quandlStockData[['Adj. Volume']]
	else:
		pass

	quandlStockData = quandlStockData.head()
	return quandlStockData

def _getYahooData(stockName, start, end):

	closingPrices = list()
	os.system('cls')
	name = yf.Ticker(stockName)
	print("StockBot: Downloading data for " + name.info['longName'] + " through Yahoo Finance...\n")
	values = yf.download(stockName, start=start, end=end)
	values = values['Adj Close']

	for value in values:
			value = round(value, 4)
			closingPrices.append(value)
	return closingPrices

def _checkStock(stockName):

	today = date.today().isoformat()
	longTerm = list()
	start = (date.today()-timedelta(days=200)).isoformat()
	end = today
	longTerm = _getYahooData(stockName, start, end)
	decision = _checkTrend(longTerm)

	if "NONE" in decision:
		print("\nNo current notifications for stock, " + stockName + ".")
		log("StockBot Notifications", "No current notifications for stock, " + stockName + ". Last known CP: $" + str(longTerm[-1]))
	else:
		print("\nYou should " + decision + " '" + stockName + "' based on calculations using moving averages over the last 200 days. Last known CP: $" +  str(longTerm[-1]) + "\n")
		log("StockBot Notifications", "You should " + decision + " '" + stockName + "' based on calculations using moving averages over the last 200 days. (" + start + " till " + end + ")")

def _linearRegressionModel():

	# Constants
	END_DATE = date.today().isoformat()
	START_DATE = (date.today()-timedelta(days=200)).isoformat()
	DATA = yf.download('HD', start=START_DATE, end=END_DATE)
	DATA_SHIFT = 1

	# Shift values and create new coloum called 'Prediction'
	DATA['Adj Close'] = round(DATA['Adj Close'], 4)
	DATA['Prediction'] = round(DATA['Adj Close'].shift(-DATA_SHIFT), 4)

	# Create x data set to train
	x = np.array(DATA.drop(['Prediction'], 1))
	x = x[:-DATA_SHIFT]

	# Create y data set to train
	y = np.array(DATA['Prediction'])
	y = y[:-DATA_SHIFT]

	# Create data set based on 50% of predictions based on testing, other 50% based on training.
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)

	# SVM (Regressor)
	# Params: [Kernel: Radial Basis Function,
	svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
	svr_rbf.fit(x_train, y_train)
	#x_forecast = np.array(DATA.drop['Prediction'])
	#print(x_forecast)
	print(svr_rbf.score(x_test, y_test))

	# The best possible score is 1.0 (fuckin L)
	svm_confidence = svr_rbf.score(x_test, y_test)
	print("Confidence: " + str(svm_confidence))

	lr = LinearRegression()
	lr.fit(x_train, y_train)
	lr_confidence = lr.score(x_test, y_test)
	print("LR Confidence: " + str(lr_confidence))

	x_forecast = np.array(DATA.drop(['Prediction'], 1))[-DATA_SHIFT:]
	# print(x_forecast)

	lr_prediction = lr.predict(x_forecast)
	print("Prediction: " + str(lr_prediction))
	return lr_prediction

def _comparePredictions(stockName):

	STOCK_NAME = stockName

	longTerm = list()
	start = (date.today()-timedelta(days=200)).isoformat()
	end = today
	longTerm = _getYahooData(stockName, start, end)
	close = float(longTerm[-1])
	if _linearRegressionModel() > close:
		print('BUY')
	elif _linearRegressionModel() < close:
		print('SELL')
	# Comparing predictions with stock trends and see if they are correct.
	# 1. Use linear regression to check the stock values within the next 30 days.
	# 2. We check trend notifications and compare it to the values from linear regression.
	# 3. Comparing.
	# Comparison is done by seeing if the values are over the current closing price, and giving it an output of wheter to buy or sell.
	# For example:
	# LR Value: $34.23
	# Closing Price: $32.01
	# if trend is BUY and LR Value is greater than closing price, then SUCEED BUY.
	# if trend is SELL and LR Value is lower than closing price, then SUCEED SELL.
	# if trend is NONE and LR Value has a large margin from closing price, then invalid prediction.
	# to further support prediction, we will look at the confidence level of the trained sets and compare.
	# in addition, comparison between both long-term and short-term data sets will also be observed to see whether any major change is noted.
	# if both predictions by long-term and short-term have a short margin then the prediction is authorized.

def _stockBotInit():
	os.system('title ' + TITLE + " " + VERSION + " by " + AUTHOR + ' | color 04')
	print('\n[NOTE]: all prices are in USD\n')
	x = input("Enter a stock ID to detect any trend changes:\n>> ")
	_checkStock(x)
	os.system('pause')
if __name__ == '__main__':
	#_stockBotInit()
	#_getYahooData()
	_stockBotInit()
	_linearRegressionModel()
