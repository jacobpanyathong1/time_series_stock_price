#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing passcode for data
from env import apipasscode
import pandas as pd

#Import Acquire
import Acquire as a
#Import requests library
import requests


# In[3]:


df = a.get_apple_data()


# In[4]:


def clean_aapl(df):
    df['month'] = df.index.month_name()

    df['month']= df.month.str.slice(stop=3)

    df['day_of_week']= df.index.day_name()

    df.drop(columns=['7. dividend amount', '8. split coefficient'], inplace=True)

    df.rename(columns={'1. open':'Open', 
                       '2. high': 'High',
                       '3. low':'Low', 
                       '4. close' : 'Close',
                       '5. adjusted close': 'Adjusted_Close',
                       '6. volume':'Volume'}, inplace=True)
    df['Open']=df.Open.astype('float')
    df['High']=df.High.astype('float')
    df['Low']=df.Low.astype('float')
    df['Close']=df.Close.astype('float')
    df['Adjusted_Close']=df.Adjusted_Close.astype('float')
    df['Volume']=df.Volume.astype('float')

    return df

