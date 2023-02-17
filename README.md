# Time Series: AAPL Stock

## Project Description
Data Scientist using the Apple (AAPL) stock price data set to develop a machine learning model that will help to predict the price of stocks. 

## Project Goals
- Explore and analyze Apple's stock for pricing trends and patterns.
- Create a machine learning model that predicts the future stock price of Apple stocks
- Gather findings, draw conclusions and recommended next steps for forecasting.

## Questions to answer
- Is there a relationship with stock trading on Monday and Tuesday?
- Is the stock price data stationary?

## Initial Thoughts and Hypothesis
Initially, the data records the past 20 years of stock prices and years of volatility and momentum. This large dataset will help draw accurate predictions for price predictions.


## Planning
- Perform Data acquisition and Preparation
- Analyze Data features and analyze market trends and patterns. 
- Establish baseline of model using baseline methods. 
  * Last Observed Value
  * Simple Average
  * Moving Average
- Create Machine Learning model to predict and forecast price.
  * Holt Winter Method
  * Holt's Linear Model
  * Holt Seasonal Trend
- Develop a model using ML to determine wine quality based on the top drivers
- Draw and record conclusions


## Data Dictionary

|Target Variable | Definition|
|-----------------|-----------|
| Close | The closing price for stocks on date/time. |

| Feature  | Definition |
|----------|------------|
| Open | Is the first price at which a security traded during the regular trading day. |
| High	| Min value for the period. |
| Low	| Min value for the period. |
| Close	| is the last price at which a security traded during the regular trading day. |
| Adjusted_Close | is the closing price after adjustments for all applicable splits and dividend distributions. |
| Volume | Volume is the amount of an asset or security that changes hands over some period of time, often over the course of a day. |
| month	Month | the stock transaction was placed. |
| day_of_week	| Day of the week the transaction was placed. |


## Conclustions and Recommendations
- Our model prredicted that the closing price for Apple stock will plateau and slowly increase into the next 20 years.
- I find that the stock for Apple is volitile and our prediction is strong but not confident in our future predictions. 
- I recommend deeper Machine Learning models for volitile markets, i.e. Tensor Flow.

## Next Steps
- The next steps will be to forecast on a stationary model to help predict further trends.
