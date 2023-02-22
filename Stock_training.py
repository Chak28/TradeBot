## Importing required Libraries
import pandas as pd
import yfinance as yf
import datetime
from pmdarima import pipeline, preprocessing as ppc, arima
from pmdarima.preprocessing import BoxCoxEndogTransformer 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set()
import joblib
import pickle
import sys
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import joblib

## Function Definitions
def days_between(d1, d2):

	''' 
		This function will take input of two dates and 
		returns the number of days in between the two dates
	'''
	d1 = datetime.datetime.strptime(d1, "%Y-%m-%d")
	d2 = datetime.datetime.strptime(d2, "%Y-%m-%d")
	
	return abs((d2 - d1).days)

def days_index(d1,n):

	''' 
		This function takes in start date and the number of days
		required to create a daily range of date values between start and end
	'''
	d1 = datetime.datetime.strptime(d1, "%Y-%m-%d")
	date_list = [str(datetime.datetime.strftime(d1 + datetime.timedelta(days=x),'%Y-%m-%d')) for x in range(n)]
	
	return date_list

def get_dates():

	'''
		This function returns 4 date values as strings, 1 for training start (3 years ago from today), 
		1 for training end (which is today), 
		1 for prediction start(tomorrow) 
		& the last 1 for prediction end (2 years from tomorrow)
	'''

	start_training = str(datetime.datetime.strftime(datetime.datetime.today() - datetime.timedelta(days=3*365),'%Y-%m-%d'))
	end_training = str(datetime.datetime.strftime(datetime.datetime.today(),'%Y-%m-%d'))
	start_testing = str(datetime.datetime.strftime(datetime.datetime.today() + datetime.timedelta(days=1),'%Y-%m-%d'))
	end_testing = str(datetime.datetime.strftime(datetime.datetime.today() + datetime.timedelta(days=2*365),'%Y-%m-%d'))

	return start_training, end_training, start_testing, end_testing

def get_yahoo_data(ticker,start,end):

	''' 
		This function hits Yahoo Finance API and pulls the historical data for a particular stock
		on specification of the start and end time period to pull the stock information
	'''
	df_training = yf.download(ticker, start=start, end=end, progress=False)
	print(f"Downloaded training data with {df_training.shape[0]} rows and {df_training.shape[1]} columns of {ticker} data")
	# breakpoint()

	return df_training

def AARIMA_training(df,ticker):

	''' 
		This function builds a pipeline that trains on the actuals
		and helps to train an auto arima model which in 
		turn takes care of stationarity check & seasonal Patterns
	'''

	## Training Pipeline Definition
	pipe = pipeline.Pipeline([## Setting max value of k , an integer which is less than m/2 => (7/2 , hence 3)
	    ("arima", arima.AutoARIMA(stepwise=True, trace=1, error_action="ignore",
	                              seasonal=True,  # because we use Fourier
	                              transparams=False,
	                               m = 7,# setting the seasonality to daily
	                              suppress_warnings=True))
	])

	## Saving the training Pipeline Object
	
	joblib.dump(pipe, f'./Models/{ticker}_pipeline.pkl')

	## Training the model using pipeline
	pipe.fit(df)
	print("Model fit:")
	print(pipe)

	## Making Predictions using the pipeline
	preds, conf_int = pipe.predict(n_periods=30, return_conf_int=True)
	print("\nForecasts:")
	print(preds)
	Predictions = pd.concat([pd.DataFrame(preds).reset_index(),pd.DataFrame(conf_int,columns = ['Lower_Limit', 'Upper_Limit'])],axis = 1)
	Predictions.columns = ['Date','Forecasts','Lower_Limit','Upper_Limit']
	return Predictions



def stationarity_check(ts,ticker):
        
    ''' 
    	This function aims to perform a stationarity check of the data
    	and ensure that the time series data is handled with appropriate transformation 
    	before training the model 
	'''

    # Calculate rolling statistics
    roll_mean = ts.rolling(window=8, center=False).mean()
    roll_std = ts.rolling(window=8, center=False).std()

    # Perform the Dickey Fuller test
    dftest = adfuller(ts) 
    
    # Plot rolling statistics:
    fig = plt.figure(figsize=(12,6))
    orig = plt.plot(ts, color='blue',label='Original')
    mean = plt.plot(roll_mean, color='red', label='Rolling Mean')
    std = plt.plot(roll_std, color='green', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.savefig(f'./Plots/{ticker}_ADFTest.png', format = 'png')
    plt.close()

    # Print Dickey-Fuller test results

    print('\nResults of Dickey-Fuller Test: \n')

    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', 
                                             '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)


def decomposition_plot(ts,ticker):

	''' 
		This function decomposes the time series data to understand
		seasonality, trends and noises in the data
	'''
	# Apply seasonal_decompose 
	decomposition = seasonal_decompose(np.log(ts))
    
	# Get trend, seasonality, and residuals
	trend = decomposition.trend
	seasonal = decomposition.seasonal
	residual = decomposition.resid

	# Plotting
	fig,ax = plt.subplots(nrows=4,figsize=(12,8))
	fig.plot(np.log(ts), label='Original', color='blue',ax = ax[0])
	fig.plot(trend, label='Trend', color='blue',ax = ax[1])
	fig.plot(seasonal,label='Seasonality', color='blue',ax=ax[2])
	fig.plot(residual, label='Residuals', color='blue',ax =ax[3])
	fig.savefig(f'./Plots/{ticker}DecomposedTS.png',format = 'png')

	

def plot_acf_pacf(ts,ticker, figsize=(10,8),lags=5):
    
    ''' 
    	This function helps to visualize the auto-correlation (Lags) in the data
    	and helps to select the p and q parameters in Auto ARIMA
	'''
    fig,ax = plt.subplots(nrows=3, figsize=figsize)
    
    # Plot ts
    ts.plot(ax=ax[0])
    
    # Plot acf, pavf
    plot_acf(ts, ax=ax[1], lags=lags)
    plot_pacf(ts, ax=ax[2], lags=lags) 
    fig.tight_layout()
    
    for a in ax[1:]:
        a.xaxis.set_major_locator(mpl.ticker.MaxNLocator(min_n_ticks=lags, integer=True))
        a.xaxis.grid()
    fig.savefig(f'./Plots/{ticker}_ACF_PACF.png', format = 'png')

#########################################  MAIN Function  #########################################

## Taking Stock Code as input
# print("Enter Ticker")
# ticker = str(input("Ticker"))

#ticker = 'MSFT'
print('Enter 2 input Stock Codes separated by comma')
tickers = list(input().split(',', 2)) ## Here we can use Database to pull the list of stocks to scale more than 100 stocks
print(f'The Entered Stock Codes are: {tickers[0]} & {tickers[1]}')
# Getting the Time Series relevant dates and ingest data from Yahoo finance

start_training, end_training, start_testing, end_testing = get_dates()

for i in range(len(list(tickers))):
	ticker = tickers[i]
	print(f'Training ARIMA Model for {ticker} Stock')
	df = get_yahoo_data(ticker,start_training,end_training)
	print(df.shape)
	## Selecting only the market close prices for everyday
	df = df['Close']

	## Fill all missing dates to account for Market closure on weekends
	df = df.resample('D').fillna(method = 'ffill')

	## Saving Evaluation of Time Series Data - Stationarity test, Time Series Decomposition and ACF, PACF
	stationarity_check(df,ticker)
	plot_acf_pacf(df,ticker)

	## Time Series Decomposition
	decomposition = seasonal_decompose(np.log(df))

	# Get trend, seasonality, and residuals
	trend = decomposition.trend
	seasonal = decomposition.seasonal
	residual = decomposition.resid

	# Saving the Decomposed Time Series Data
	plt.figure(figsize=(12,8))
	plt.subplot(411)
	plt.plot(np.log(df), label='Original', color='blue')
	plt.legend(loc='best')
	plt.subplot(412)
	plt.plot(trend, label='Trend', color='blue')
	plt.legend(loc='best')
	plt.subplot(413)
	plt.plot(seasonal,label='Seasonality', color='blue')
	plt.legend(loc='best')
	plt.subplot(414)
	plt.plot(residual, label='Residuals', color='blue')
	plt.legend(loc='best')
	plt.tight_layout()
	plt.savefig(f'./Plots/{ticker}_DecomposedTS.png',format = 'png')

	## Training ARIMA Model

	Predictions = AARIMA_training(df,ticker)
	df.to_csv(f'./Data/{ticker}_historical_actuals.csv')
	Predictions.to_csv(f'./Data/{ticker}_Predictions.csv')


