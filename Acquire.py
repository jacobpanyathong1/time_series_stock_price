#!/usr/bin/env python
# coding: utf-8

# In[5]:


#Importing passcode for data
from env import apipasscode
import pandas as pd

#Import requests library
import requests


# In[ ]:


#url for json data
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=AAPL&apikey={apipasscode}'
#Extracting JSON data
r = requests.get(url)
#Getting the JSON reponse data
data = r.json()


# In[ ]:


def get_apple_data():
    '''
    Acquire JSON data from URL and return DataFrame of AAPL data.
    '''
    #Complete JSON data of AAPL 
    data = data['Time Series (Daily)']
    #Values only of AAPL 
    data_values= data.values()
    #Creating DataFrame of data 
    df=pd.DataFrame(data_values, index = data.keys())
    #Establishing Datetime index for DataFrame
    df.index = pd.to_datetime(df.index)
    #Setting index for DataFrame
    df= df.set_index(df.index)
    #Returning DataFrame
    return df


# In[ ]:


def get_apple_data_info():
    '''
    Acquire JSON data from URL and return DataFrame of AAPL data.
    '''
    #Extracting JSON data
    r = requests.get(url)
    #Getting the JSON reponse data
    data = r.json()
    #DataFrame of information 
    df= pd.DataFrame([data['Meta Data']])
    #Returning DataFrame
    return df

