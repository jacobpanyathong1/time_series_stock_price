#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#Importing passcode for data
import os
import Acquire as a
import Prepare as p

#Importing libraries
import os
import pandas as pd
from datetime import timedelta, datetime
import numpy as np
import matplotlib.pyplot as plt
import json
import warnings
import seaborn as sns
import requests
import yfinance as yf
import functions as f

#Statsmodels
from statsmodels.tsa.stattools import adfuller
import scipy.stats as stats

# modeling
import statsmodels.api as sm
from statsmodels.tsa.api import Holt, ExponentialSmoothing
np.random.seed(0)

# evaluate
from sklearn.metrics import mean_squared_error
from math import sqrt 
warnings.filterwarnings("ignore")

# plotting
import seaborn as sns 
import plotly.express as px
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# settings
plt.style.use('seaborn')
plt.rcParams["figure.figsize"] = (16, 8)

# In[1]:


def plot1(df):
    '''This function returns monthly average of each day of buy/sell transaction price'''
    cols = ['month', 'day_of_week']
    for col in cols:
        plt.style.use('_mpl-gallery')

        # plot:
        plt.figure(figsize=(10,6))
        sns.histplot(df[col], bins=25)
        plt.title(f'Total Count Of {col.capitalize().strip("_")}'+'s')
        plt.ylabel("Total Count")
        plt.show()


# In[ ]:


def plot3(df):
    '''This function plots the average of each day of buy/sell transaction price'''
    plt.figure(figsize=(12,6))
    wm_df=df[['Open', 'High', 'Low', 'Close']]
    wm_df= wm_df.resample('D').mean()
    sns.lineplot(wm_df, palette='viridis', dashes =True)
    plt.ylabel("Average Price")
    plt.xlabel("Year")
    plt.title("Average of Each Day Of Buy/Sell Transaction Price")
    plt.plot()


# In[2]:


def plot4(df):
    '''This function plots the distribution of Close Price on Mondays and Tuesdays
    '''
    #Plotting the distribution of Close Price on Mondays and Tuesdays
    mt= df[df.day_of_week == 'Tuesday'].Close.mean()
    #Plotting figure
    plt.figure(figsize=(10,6))
    #Plotting histogram
    tues= df[df.day_of_week == 'Tuesday'].Close
    tues.hist(bins=25, ec='white', color='seagreen')
    #Plotting title, x and y labels
    plt.title("Total Average of Buy/Sell Transactions on Tuesdays")
    plt.xlabel("Distribution of Close Price")
    plt.ylabel("Count of Close Price")
    #Displaying plot
    plt.show()
    #Printing mean of Close Price on Tuesdays
    print(f'Mean Count of Buy/Sell Transactions on Tuesday: {mt:.2f}')

    #Plotting the distribution of Close Price on Mondays and Tuesdays
    mon= df[df.day_of_week == 'Monday'].Close
    #Plotting figure
    plt.figure(figsize=(10,6))
    #Plotting histogram
    mon.hist(bins=25, ec='white', color='rebeccapurple')
    #Plotting title, x and y labels
    plt.title("Total Average of Buy/Sell Transactions on Mondays")
    plt.xlabel("Distribution of Close Price")
    plt.ylabel("Count of Close Price")
    #Displaying plot
    plt.show()
    
    #Printing mean of Close Price on Mondays
    mm= df[df.day_of_week == 'Monday'].Close.mean()
    print(f'Mean Count of Buy/Sell Transactions on Monday: {mm:.2f}')


# In[ ]:


def plot5(df):
    '''This function plots the Open, Close, Low, and High prices of Apple Stock
    '''
    df[["Open", "Close", "Low", "High"]].plot(figsize=(14,8), 
                                          title="(AAPL) Apple Stock", 
                                          fontsize=12,
                                             linestyle='-.')
    plt.ylabel("Price (HUN)")
    plt.xlabel("Year")
    plt.show()

    df['Volume'].plot(figsize=(14,8), 
                                              title="(AAPL) Apple Stock", 
                                              fontsize=12,
                                             linestyle='-.', label="Volume",color = "seagreen")
    plt.ylabel("Average Unit Transactions (M)",fontsize=14)
    plt.xlabel("Year", fontsize=14)
    plt.legend(fontsize=12)
    plt.show()


# In[ ]:


def plot6(df):
    '''This function plots the Rate of Change of Apple Stock'''
    roc_df = df[['Close']]

    roc_df['Close_Shift']=roc_df.Close.shift(-1)

    roc_df['Variance']= roc_df.Close-roc_df.Close_Shift

    roc_df['roc']= roc_df.Variance/roc_df.Close_Shift*100
    
    plt.figure(figsize=(12,6))
    roc_df['Close'].plot(color='seagreen', fontsize=12)
    plt.title("Close Price: AAPL", fontsize= 14)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Close Price", fontsize=12)
    plt.legend(fontsize=12)
    plt.show()

    plt.figure(figsize=(12,6))
    plt.title("ROC Close: AAPL", fontsize= 14)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Percentage of Change", fontsize=12)
    roc_df['roc'].plot(color='rebeccapurple', label='ROC')
    plt.legend(fontsize=12)
    plt.show()
    print("Highest Increase of Change within 1 Period:", round(roc_df['roc'].max(), 3))
    print("Highest Decrease of Change within 1 Period:", round(roc_df['roc'].min(),3))


# In[3]:


def plot7(df):
    '''This function plots the RSI of Apple Stock'''
    #Finding the difference of each date from the preivous date
    change = df['Close'].diff()
    #Change drop any null values
    change.dropna(inplace=True)
    # Create two copies of the Closing price Series
    change_up = change.copy()
    change_down = change.copy()
    #Change series of only poisitive values
    change_up[change_up<0] = 0
    #Change Series of only negative values
    change_down[change_down>0] = 0
    
    # Calculate the rolling average of average up and average down
    avg_up = change_up.rolling(30).mean()
    avg_down = change_down.rolling(30).mean().abs()
    #RSI  Calculations
    rsi = 100 * avg_up / (avg_up + avg_down)
    # Set the theme of our chart
    plt.style.use('fivethirtyeight')

    # Make our resulting figure much bigger
    plt.rcParams['figure.figsize'] = (20, 20)
    # Create two charts on the same figure.
    ax1 = plt.subplot2grid((10,1), (0,0), rowspan = 4, colspan = 1)
    ax2 = plt.subplot2grid((10,1), (5,0), rowspan = 4, colspan = 1)

    # First chart:
    # Plot the closing price on the first chart
    ax1.plot(df['Close'], linewidth=2, color='seagreen')
    ax1.set_title('Apple: Close Price')

    # Second chart
    # Plot the RSI
    ax2.set_title('Relative Strength Index')
    ax2.plot(rsi, color='rebeccapurple', linewidth=1)
    # Add two horizontal lines, signalling the buy and sell ranges.
    # Oversold
    ax2.axhline(30, linestyle='--', linewidth=1.5, color='green')
    # Overbought
    ax2.axhline(70, linestyle='--', linewidth=1.5, color='red')
    plt.show()

def plot8(train):
    '''This function plots the distribution of the target variable'''
    plt.figure(figsize=(12,6))
    plt.hist(train, label=["Open", "Close"], ec='white')
    plt.xlabel('Open/Close')
    plt.ylabel('Count')
    plt.title('Distribution of Target')
    plt.legend()
    plt.show()
# In[ ]:


def ttest(df):
    '''This function performs a t-test to determine if there is a significant relationship between the avg stock prices closed on monday and tuesday'''
    mon= df[df.day_of_week == 'Monday'].Close
    tues= df[df.day_of_week == 'Tuesday'].Close
    α = .05
    t, p = stats.ttest_ind(tues, mon, equal_var=False)
    if p/2 < α:   
        print("Is t < 0?", t < 0)
        print("Is p < α:", p/2 <α)
        print("We are confident that there is a significant relationship between the AVG stock prices closed on monday and tuesday")
    else:
        print("Is t < 0?", t < 0)
        print("Is p < α:", p/2 <α)
        print('We fail to reject the null:')
        print('There is no significant relationship between the avg stock prices closed on monday and tuesday')
        
def adfuller_test(df):
    '''
    This function performs the Augmented Dickey-Fuller test to determine if the time series is stationary.
    '''
    series = df['Close'].values
    series
    result = adfuller(series, autolag='AIC')

    print('ADF Statistic: %f' % result[0])

    print('p-value: %f' % result[1])

    print('Critical Values:')

    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    if result[0] < result[4]["5%"]:
        print ("Reject Ho - Time Series is Stationary")
    else:
        print ("Failed to Reject Ho - Time Series is Non-Stationary")

def train__val_test(df):
    '''
    Creating train, validate, test split.
    '''
    #Resample DataFrame to weeks sum.
    df_resampled = df.resample('W')[['Open','Close']].sum()

    # set train size to be 50% of total 
    train_size = int(round(df_resampled.shape[0] * 0.5))
    train_size

    # set validate size to be 30% of total 
    validate_size = int(round(df_resampled.shape[0] * 0.3))
    validate_size

    # set test size to be number of rows remaining. 
    test_size = int(round(df_resampled.shape[0] * 0.2))
    test_size
    
    #Checking to see if resample is equal length of splitting data.
    len(df_resampled) == train_size + validate_size + test_size

    # validate index
    validate_end_index = train_size + validate_size
    validate_end_index
    #Creating Train sample
    train = df_resampled[:train_size]
    #Checking last dates of train
    train.tail()
    #Creating validate sample
    validate = df_resampled[train_size:validate_end_index]
    #Creating test sample
    test = df_resampled[validate_end_index:]
    #Shape of samples rows
    train.shape[0], validate.shape[0], test.shape[0]
    #Verifying the samples match the length.
    len(train) + len(validate) + len(test) == len(df_resampled)
    #print(df_resampled.head(1) == train.head(1))
    #pd.concat([test.tail(1), df_resampled.tail(1)])
    return train, validate, test

def plot_sample(train, validate, test):
    '''
    Plotting train, validate, test split.
    '''
    # Plot train, validate, test
    for col in train.columns:
        # Plot figure
        plt.figure(figsize=(14,8))
        #Plot Data
        plt.plot(train[col], color='#377eb8', label = 'Train')
        plt.plot(validate[col], color='#ff7f00', label = 'Validate')
        plt.plot(test[col], color='#4daf4a', label = 'Test')
        #Plot Labels
        plt.legend()
        plt.ylabel(col)
        plt.title(col)
        #Show Plot
        plt.show()

def evaluate(target_var):
    '''
    This function will take the actual values of the target_var from validate, 
    and the predicted values stored in yhat_df, 
    and compute the rmse, rounding to 0 decimal places. 
    it will return the rmse. 
    '''
    rmse = round(sqrt(mean_squared_error(validate[target_var], yhat_df[target_var])), 0)
    return rmse

def plot_and_eval(target_var):
    '''
    This function takes in the target var name (string), and returns a plot
    of the values of train for that variable, validate, and the predicted values from yhat_df. 
    it will als lable the rmse. 
    '''
    plt.figure(figsize = (12,4))
    plt.plot(train[target_var], label='Train', linewidth=1, color='#377eb8')
    plt.plot(validate[target_var], label='Validate', linewidth=1, color='#ff7f00')
    plt.plot(yhat_df[target_var], label='yhat', linewidth=2, color='#a65628')
    plt.legend()
    plt.title(target_var)
    rmse = evaluate(target_var)
    print(target_var, '-- RMSE: {:.0f}'.format(rmse))
    plt.show()

# function to store the rmse so that we can compare
def append_eval_df(model_type, target_var):
    '''
    this function takes in as arguments the type of model run, and the name of the target variable. 
    It returns the eval_df with the rmse appended to it for that model and target_var. 
    '''
    rmse = evaluate(target_var)
    d = {'model_type': [model_type], 'target_var': [target_var],
        'rmse': [rmse]}
    d = pd.DataFrame(d)
    return eval_df.append(d, ignore_index = True)

def lov_base(train):
    '''
    Last observation carried forward baseline model.
    '''
    train['Close'][-1:][0]

    # take the last item of close price and assign to variable
    last_close = train['Close'][-1:][0]

    # take the last open price and assign to variable
    last_open = train['Open'][-1:][0]
    print(f"Last Open value:{last_open}")
    print(f"Last Close value:{last_close}")
    
def yhat_df(train, validate):
    '''
    yhat_df function to create a dataframe with the last observation carried forward baseline model.
    '''
    train['Close'][-1:][0]

    # take the last item of close price and assign to variable
    last_close = train['Close'][-1:][0]

    # take the last open price and assign to variable
    last_open = train['Open'][-1:][0]
    
    yhat_df = pd.DataFrame(
        {'Open': [last_open],
         'Close': [last_close]},
        index=validate.index)
    return yhat_df

def plot_base(train):
    for col in train.columns:
        plot_and_eval(col)
# %%
def final_function(df):
    '''returns model'''
    def train__val_test(df):
        '''
        Creating train, validate, test split.
        '''
        #Resample DataFrame to weeks sum.
        df_resampled = df.resample('W')[['Open','Close']].sum()

        # set train size to be 50% of total 
        train_size = int(round(df_resampled.shape[0] * 0.5))
        train_size

        # set validate size to be 30% of total 
        validate_size = int(round(df_resampled.shape[0] * 0.3))
        validate_size

        # set test size to be number of rows remaining. 
        test_size = int(round(df_resampled.shape[0] * 0.2))
        test_size
        
        #Checking to see if resample is equal length of splitting data.
        len(df_resampled) == train_size + validate_size + test_size

        # validate index
        validate_end_index = train_size + validate_size
        validate_end_index
        #Creating Train sample
        train = df_resampled[:train_size]
        #Checking last dates of train
        train.tail()
        #Creating validate sample
        validate = df_resampled[train_size:validate_end_index]
        #Creating test sample
        test = df_resampled[validate_end_index:]
        #Shape of samples rows
        train.shape[0], validate.shape[0], test.shape[0]
        #Verifying the samples match the length.
        len(train) + len(validate) + len(test) == len(df_resampled)
        #print(df_resampled.head(1) == train.head(1))
        #pd.concat([test.tail(1), df_resampled.tail(1)])
        return train, validate, test

    df_resampled = df.resample('W')[['Open','Close']].sum()
    train_size = int(round(df_resampled.shape[0] * 0.5))


    # set validate size to be 30% of total 
    validate_size = int(round(df_resampled.shape[0] * 0.3))
    validate_size

        # set test size to be number of rows remaining. 
    test_size = int(round(df_resampled.shape[0] * 0.2))
    test_size

    # validate index
    validate_end_index = train_size + validate_size
    validate_end_index
    #Creating Train sample
    train = df_resampled[:train_size]
    #Checking last dates of train
    train.tail()
    #Creating validate sample
    validate = df_resampled[train_size:validate_end_index]
    #Creating test sample
        

    def plot_sample(train, validate, test):
        for col in train.columns:
            plt.figure(figsize=(14,8))
            plt.plot(train[col], color='#377eb8', label = 'Train')
            plt.plot(validate[col], color='#ff7f00', label = 'Validate')
            plt.plot(test[col], color='#4daf4a', label = 'Test')
            plt.legend()
            plt.ylabel(col)
            plt.title(col)
            plt.show()

    sns.boxplot(data = train, y=train['Open'])

    plt.plot(train.index, train['Open'])
    plt.plot(validate.index, validate['Open'])
    plt.plot(test.index, test['Open'])
    plt.show()

    def plot8(train):
        plt.figure(figsize=(12,6))
        plt.hist(train, label=["Open", "Close"], ec='white')
        plt.xlabel('Open/Close')
        plt.ylabel('Count')
        plt.title('Distribution of Target')
        plt.legend()
        plt.show()

    df.groupby('month')['Close'].mean().plot.bar()
    plt.xticks(rotation=20)
    plt.show()

    def evaluate(target_var):
        '''
        This function will take the actual values of the target_var from validate, 
        and the predicted values stored in yhat_df, 
        and compute the rmse, rounding to 0 decimal places. 
        it will return the rmse. 
        '''
        rmse = round(sqrt(mean_squared_error(validate[target_var], yhat_df[target_var])), 0)
        return rmse

    def plot_and_eval(target_var):
        '''
        This function takes in the target var name (string), and returns a plot
        of the values of train for that variable, validate, and the predicted values from yhat_df. 
        it will als lable the rmse. 
        '''
        plt.figure(figsize = (12,4))
        plt.plot(train[target_var], label='Train', linewidth=1, color='#377eb8')
        plt.plot(validate[target_var], label='Validate', linewidth=1, color='#ff7f00')
        plt.plot(yhat_df[target_var], label='yhat', linewidth=2, color='#a65628')
        plt.legend()
        plt.title(target_var)
        rmse = evaluate(target_var)
        print(target_var, '-- RMSE: {:.0f}'.format(rmse))
        plt.show()

# create an empty dataframe
    eval_df = pd.DataFrame(columns=['model_type', 'target_var', 'rmse'])
    eval_df

# function to store the rmse so that we can compare
    def append_eval_df(model_type, target_var):
        '''
        this function takes in as arguments the type of model run, and the name of the target variable. 
        It returns the eval_df with the rmse appended to it for that model and target_var. 
        '''
        rmse = evaluate(target_var)
        d = {'model_type': [model_type], 'target_var': [target_var],
            'rmse': [rmse]}
        d = pd.DataFrame(d)
        return eval_df.append(d, ignore_index = True)

    train['Close'][-1:][0]

    # take the last item of close price and assign to variable
    last_close = train['Close'][-1:][0]

    # take the last open price and assign to variable
    last_open = train['Open'][-1:][0]
    last_open 

    yhat_df = pd.DataFrame(
        {'Open': [last_open],
        'Close': [last_close]},
        index=validate.index)

    yhat_df.head()

    for col in train.columns:
        plot_and_eval(col)

    # compute simple average of sales_total (from train data)
    avg_close = round(train['Close'].mean(), 2)
    avg_close

    # compute simple average of quantity (from train data)
    avg_open = round(train['Open'].mean(), 2)
    avg_open

    def make_baseline_predictions(open_predictions=None, close_predictions=None):
        yhat_df = pd.DataFrame({'Open': [open_predictions],
                            'Close': [close_predictions]},
                            index=validate.index)
        return yhat_df

    yhat_df = make_baseline_predictions(avg_open, avg_close)

    for col in train.columns:
        plot_and_eval(col)



    for col in train.columns:
        eval_df = append_eval_df(model_type='simple_average', 
                                target_var = col)
    eval_df



    period=30
    train['Close'].rolling(period).mean()

    train['Close'].rolling(period).mean()[-1]

    # Saving the last 30 day moving average for each column
    rolling_open = round(train['Open'].rolling(period).mean()[-1], 2)
    rolling_close = round(train['Close'].rolling(period).mean()[-1], 2)
    print(rolling_open, rolling_close)

    yhat_df = make_baseline_predictions(rolling_open, rolling_close)
    yhat_df.head()

    for col in train.columns:
        plot_and_eval(col)

    for col in train.columns:
        eval_df = append_eval_df(model_type = '30d_moving_avg', 
                                target_var = col)

    eval_df

    periods = [4, 12, 26, 52, 104]

    for p in periods: 
        rolling_open = round(train['Open'].rolling(p).mean()[-1], 2)
        rolling_close = round(train['Close'].rolling(p).mean()[-1], 2)
        yhat_df = make_baseline_predictions(rolling_open, rolling_close)
        model_type = str(p) + '_day_moving_avg'
        for col in train.columns:
            eval_df = append_eval_df(model_type = model_type,
                                    target_var = col)

    eval_df

    best_open_rmse = eval_df[eval_df.target_var == 'Open']['rmse'].min()

    best_open_rmse

    eval_df[eval_df.rmse == best_open_rmse]

    best_close_total_rmse = eval_df[eval_df.target_var == 'Close']['rmse'].min()

    eval_df[eval_df.rmse == best_close_total_rmse]

    for col in train.columns:
        sm.tsa.seasonal_decompose(train[col].resample('W').mean()).plot()

    col = 'Close' 
    # create our Holt Object
    model = Holt(train[col], exponential=False, damped=True)

    # fit the Holt object
    model = model.fit(optimized=True)

    yhat_close_total = model.predict(start = validate.index[0],
                                end = validate.index[-1])

    yhat_close_total

    # doing this in a loop for each column
    for col in train.columns:
        model = Holt(train[col], exponential=False, damped=True)
        model = model.fit(optimized=True)
        yhat_values = model.predict(start = validate.index[0],
                                end = validate.index[-1])
        yhat_df[col] = round(yhat_values, 2)

    for col in train.columns:
        plot_and_eval(target_var = col)

    for col in train.columns:
        eval_df = append_eval_df(model_type = 'holts_optimized', 
                                target_var = col)

    eval_df.sort_values(by='rmse')

    # Models for quantity
    hst_open_fit1 = ExponentialSmoothing(train['Open'], seasonal_periods=52, trend='add', seasonal='add').fit()
    hst_open_fit2 = ExponentialSmoothing(train['Open'], seasonal_periods=52, trend='add', seasonal='mul').fit()
    hst_open_fit3 = ExponentialSmoothing(train['Open'], seasonal_periods=52, trend='add', seasonal='add', damped=True).fit()
    hst_open_fit4 = ExponentialSmoothing(train['Open'], seasonal_periods=52, trend='add', seasonal='mul', damped=True).fit()

    # Models for sales
    hst_close_fit1 = ExponentialSmoothing(train['Close'], seasonal_periods=52, trend='add', seasonal='add').fit()
    hst_close_fit2 = ExponentialSmoothing(train['Close'], seasonal_periods=52, trend='add', seasonal='mul').fit()
    hst_close_fit3 = ExponentialSmoothing(train['Close'], seasonal_periods=52, trend='add', seasonal='add', damped=True).fit()
    hst_close_fit4 = ExponentialSmoothing(train['Close'], seasonal_periods=52, trend='add', seasonal='mul', damped=True).fit()

    results_open=pd.DataFrame({'model':['hst_open_fit1', 'hst_open_fit2', 'hst_open_fit3', 'hst_open_fit4'],
                                'SSE':[hst_open_fit1.sse, hst_open_fit2.sse, hst_open_fit3.sse, hst_open_fit4.sse]})
    results_open

    results_open.sort_values(by='SSE')

    results_close=pd.DataFrame({'model':['hst_close_fit1', 'hst_close_fit2', 'hst_close_fit3', 'hst_close_fit4'],
                                'SSE':[hst_close_fit1.sse, hst_close_fit2.sse, hst_close_fit3.sse, hst_close_fit4.sse]})
    results_close

    results_close.sort_values(by='SSE')

    yhat_df = pd.DataFrame({'Open': hst_open_fit1.forecast(validate.shape[0]),
                            'Close': hst_close_fit3.forecast(validate.shape[0])},
                            index=validate.index)
    yhat_df

    for col in train.columns:
        plot_and_eval(col)

    eval_df

    for col in train.columns:
        eval_df = append_eval_df(model_type = 'holts_seasonal_add_add', 
                                target_var = col)

    eval_df.sort_values(by='rmse')

    train = df_resampled[:'2016']
    validate = df_resampled['2017']
    test = df_resampled['2018']

    print(train.shape)
    print(validate.shape)
    print(test.shape)

    train.head()
    train.tail()

    train.diff(365)

    yhat_df = train['2016'] + train.diff(365).mean()
    yhat_df

    train.loc['2016'].head()

    pd.concat([yhat_df.head(1), validate.head(1)])

    train.loc['2016'].head()

    yhat_df.shape

    yhat_df

    validate.shape

    validate = validate[validate.index != '2017-12-31']

    yhat_df.index = validate.index

    for col in train.columns:
        plot_and_eval(target_var = col)
        eval_df = append_eval_df(model_type = "previous_year", 
                                target_var = col)



    eval_df.sort_values(by='rmse')

    open_total_min_rmse = eval_df.groupby('target_var')['rmse'].min()[0]

    close_min_rmse = eval_df.groupby('target_var')['rmse'].min()[1]

    # find which model that is
    eval_df[((eval_df.rmse == open_total_min_rmse) | 
            (eval_df.rmse == close_min_rmse))]

    train = df_resampled[:train_size]
    validate = df_resampled[train_size:validate_end_index]
    test = df_resampled[validate_end_index:]



    train.shape, validate.shape, test.shape

    yhat_df = pd.DataFrame({'Open': hst_open_fit1.forecast(validate.shape[0] + test.shape[0]),
                            'Close': hst_close_fit3.forecast(validate.shape[0] + test.shape[0])})
    yhat_df

    validate.head(1)

    test.head(1)

    yhat_df = yhat_df['2018-07-01':]

    def final_plot(target_var):
        plt.figure(figsize=(12,4))
        plt.plot(train[target_var], color='#377eb8', label='train')
        plt.plot(validate[target_var], color='#ff7f00', label='validate')
        plt.plot(test[target_var], color='#4daf4a',label='test')
        plt.plot(yhat_df[target_var], color='#a65628', label='yhat')
        plt.legend()
        plt.title(target_var)
        plt.show()

    yhat_df

    rmse_open_total = sqrt(mean_squared_error(test['Open'], 
                                        yhat_df['Open']))

    rmse_close = sqrt(mean_squared_error(test['Close'], 
                                        yhat_df['Close']))

    print('FINAL PERFORMANCE OF MODEL ON TEST DATA')
    print('rmse-open total: ', rmse_open_total)
    print('rmse-close: ', rmse_close)
    for col in train.columns:
        final_plot(col)

    forecast = pd.DataFrame({'Open': hst_open_fit1.forecast(validate.shape[0] + test.shape[0] + 365),
                            'Close': hst_close_fit3.forecast(validate.shape[0] + test.shape[0] + 365)})
    forecast = forecast['2022':]
    forecast

    def final_plot(target_var):
        plt.figure(figsize=(12,4))
        plt.plot(train[target_var], color='#377eb8', label='Train')
        plt.plot(validate[target_var], color='#ff7f00', label='Validate')
        plt.plot(test[target_var], color='#4daf4a', label='Test')
        plt.plot(yhat_df[target_var], color='#a65628', label='yhat')
        plt.plot(forecast[target_var], color='#984ea3', label='Forecast')
        plt.title(target_var)
        plt.legend()
        plt.show()

    for col in train.columns:
        final_plot(col)