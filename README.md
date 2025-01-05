# Decision-Theory-1
Stock Price Prediction using Historical Data and Regression Models

This repository contains the code and files related to a project to forecast stock closing prices, based on historical data and the use of regression algorithms. 
The goal is to analyze and predict the next day's closing price using features based on past closing prices.


Project Functions and Objectives:
Data collection:

Collect daily data (e.g., Open, High, Low, Close, Volume) for a stock via the Alpha Vantage API.
Registration is required to use a free API key.
Create input attributes:

Use lagged features, such as:
close_t-1: The previous day's closing price.
close_t-2: The closing value two days ago.
... up to close_t-N.
Possibility to use longer time intervals (e.g., average weekly/decade price).
Data preprocessing:

Data smoothing with Gaussian filters to reduce noise.
Separation into training and validation sets:
Training with data before 2024.
Validation with 2024 data.
Price prediction:

Prediction of the next closing price based on previous characteristics.
Regression models:

Linear regression model:
Consideration of appropriate number of parameters.
Presentation of error metrics for training and validation sets.
Multinomial regression model with normalization (L1, L2):
Hyperparameter selection.
Presentation of error metrics and model parameters.

Prerequisites:
Python 3.x
Libraries:
pandas, numpy, scikit-learn, matplotlib, seaborn
requests to use APIs

