 Project: 1.Stock Price Predictor

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

Project: 2.Chatbot for Customer Service

A customer service chatbot that answers common queries (orders, returns,
shipping, payments, account help, business hours, escalation to a human).
Implements both approaches mentioned in the brief so you can compare
them directly:


Rule-based: keyword/word-overlap matching against known patterns.
Simple NLP-based: TF-IDF vectorization + cosine similarity, which
generalizes to rephrased questions the rule-based version misses.


Files


intents.json — the knowledge base: each "intent" (order status, returns,
shipping, etc.) has example phrasings and one or more responses.
chatbot.py — both chatbot engines + a CLI to chat interactively or run
a side-by-side demo.


Run it

bashpip install scikit-learn

# Chat interactively (NLP mode by default)
python chatbot.py

# Chat interactively using the simpler rule-based engine
python chatbot.py --mode rule

# See both engines respond to the same sample questions side by side
python chatbot.py --demo

Type quit to exit an interactive session.

Why include both approaches?

This is the most instructive part of the project. Run --demo and you'll
see cases like:

User saysRule-basedNLP-based (TF-IDF)"where's my package"fails (no shared keywords with training patterns)correctly identifies order_status"what time do you guys open"failscorrectly identifies business_hours"what's the meaning of life"correctly falls backcorrectly falls back

The rule-based bot only recognizes phrasing close to what you explicitly
listed in intents.json. The NLP-based bot, using TF-IDF, can match a
rephrased question to the right intent because it scores word
importance, not just exact overlap — words that appear across many
patterns (like "the", "is", "my") are downweighted automatically, so
distinctive words (like "package", "refund", "declined") drive the match.
Both filter out common stopwords before scoring so generic words don't
cause false-positive matches.

Extending it


Add more intents/patterns to intents.json — no code changes needed.
Swap in a real dataset of past support tickets to auto-generate patterns.
Add slot-filling (e.g. actually capture an order number when the user
mentions order_status).
Upgrade the NLP engine further with sentence embeddings (e.g.
sentence-transformers) for even better semantic matching than TF-IDF.
Wire it up to a real chat widget (Flask/FastAPI backend + simple frontend).
