Stock Price Predictor

Predicts a stock's next-day closing price from historical OHLCV
(Open/High/Low/Close/Volume) data.

What it does


Loads data — real CSV if you provide one, otherwise generates a
realistic synthetic price series so the demo runs immediately.
Engineers features — lagged prices, moving averages, daily
returns, rolling volatility, volume change.
Trains two models:

LinearRegression (the baseline requested in the brief)
RandomForestRegressor (the "more advanced technique")



Evaluates with RMSE, MAE, and R².
Plots actual vs. predicted prices, and feature importance.
Predicts tomorrow's close using the latest available row.


Run it

bashpip install pandas numpy scikit-learn matplotlib
python stock_predictor.py

Use your own data

Get a CSV with columns Date, Open, High, Low, Close, Volume — e.g.
export one from Yahoo Finance, or with the yfinance package:

pythonimport yfinance as yf
df = yf.download("AAPL", start="2020-01-01", end="2026-01-01").reset_index()
df.to_csv("aapl.csv", index=False)

Then in stock_predictor.py, change:

pythonraw_df = load_data(csv_path=None)

to:

pythonraw_df = load_data(csv_path="aapl.csv")

Notes / good talking points for your internship writeup


Time-based train/test split is used (not random shuffling) — for
time series, shuffling would leak future information into training.
Linear Regression usually looks deceptively strong on lagged price
features because "tomorrow ≈ today" is a decent naive baseline —
worth mentioning R² isn't the whole story here.
Ideas to extend: add technical indicators (RSI, MACD, Bollinger
Bands), try an LSTM/GRU for sequence modeling, or predict direction
(up/down) as a classification problem instead of exact price.
