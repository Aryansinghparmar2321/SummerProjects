"""
Stock Price Predictor
======================
Predicts stock closing prices using historical data.

Two models are included:
  1. Linear Regression  (the baseline, as requested)
  2. Random Forest Regressor (an "advanced technique", as the brief allows)

HOW TO USE WITH YOUR OWN DATA
------------------------------
Replace `load_data()` with a call that reads a CSV of real historical
prices. The CSV just needs these columns: Date, Open, High, Low, Close, Volume
(this is the exact format Yahoo Finance / most brokers export).

    df = pd.read_csv("your_stock.csv", parse_dates=["Date"])

If you have internet access in your own environment, you can fetch real
data with the `yfinance` package:

    import yfinance as yf
    df = yf.download("AAPL", start="2020-01-01", end="2026-01-01").reset_index()

For this demo (sandboxed, no internet), we generate a realistic synthetic
price series so the whole pipeline runs end-to-end out of the box.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ----------------------------------------------------------------------
# 1. LOAD DATA
# ----------------------------------------------------------------------
def load_data(csv_path=None, n_days=1000, seed=42):
    """
    If csv_path is given, load real historical data from it.
    Otherwise, generate a synthetic-but-realistic price series
    (random walk + drift + seasonality + noise) so the demo runs
    without needing internet access.
    """
    if csv_path:
        df = pd.read_csv(csv_path, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start="2021-01-01", periods=n_days)  # business days

    # Random walk with slight upward drift
    daily_returns = rng.normal(loc=0.0004, scale=0.018, size=n_days)
    # add a mild seasonal wave so there's structure to learn
    seasonality = 0.01 * np.sin(np.arange(n_days) * (2 * np.pi / 60))
    log_price = np.cumsum(daily_returns + seasonality) 
    close = 100 * np.exp(log_price)  # start around $100

    high = close * (1 + rng.uniform(0.001, 0.02, n_days))
    low = close * (1 - rng.uniform(0.001, 0.02, n_days))
    open_ = low + (high - low) * rng.uniform(0, 1, n_days)
    volume = rng.integers(1_000_000, 10_000_000, n_days)

    df = pd.DataFrame({
        "Date": dates,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })
    return df


# ----------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ----------------------------------------------------------------------
def engineer_features(df, lags=(1, 2, 3, 5, 10), ma_windows=(5, 10, 20)):
    """
    Build predictive features from raw OHLCV data:
      - lagged closing prices (yesterday's close, 2 days ago, etc.)
      - moving averages (trend signal)
      - daily return
      - rolling volatility
    Target: next day's closing price.
    """
    data = df.copy()

    for lag in lags:
        data[f"close_lag_{lag}"] = data["Close"].shift(lag)

    for window in ma_windows:
        data[f"ma_{window}"] = data["Close"].rolling(window).mean()

    data["daily_return"] = data["Close"].pct_change()
    data["volatility_10"] = data["daily_return"].rolling(10).std()
    data["volume_change"] = data["Volume"].pct_change()

    # Target = next day's close (what we want to predict)
    data["target"] = data["Close"].shift(-1)

    data = data.dropna().reset_index(drop=True)
    return data


# ----------------------------------------------------------------------
# 3. TRAIN / EVALUATE MODELS
# ----------------------------------------------------------------------
def train_and_evaluate(data):
    feature_cols = [c for c in data.columns if c not in ("Date", "target")]
    X = data[feature_cols]
    y = data["target"]

    # Time-based split: train on the earlier portion, test on the later
    # portion (this avoids "lookahead" leakage that a random shuffle would cause)
    split_idx = int(len(data) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = data["Date"].iloc[split_idx:]

    results = {}

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=6, random_state=42
        ),
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results[name] = {
            "model": model,
            "preds": preds,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
        }

        print(f"\n{name}")
        print("-" * len(name))
        print(f"  RMSE: {rmse:.3f}")
        print(f"  MAE:  {mae:.3f}")
        print(f"  R^2:  {r2:.3f}")

    return results, dates_test, y_test, feature_cols


# ----------------------------------------------------------------------
# 4. PLOT RESULTS
# ----------------------------------------------------------------------
def plot_results(dates_test, y_test, results, out_path="prediction_results.png"):
    plt.figure(figsize=(12, 6))
    plt.plot(dates_test, y_test.values, label="Actual Close", color="black", linewidth=1.5)

    colors = {"Linear Regression": "tab:blue", "Random Forest": "tab:orange"}
    for name, res in results.items():
        plt.plot(dates_test, res["preds"], label=f"{name} (RMSE={res['rmse']:.2f})",
                  color=colors.get(name), linewidth=1.2, alpha=0.85)

    plt.title("Stock Price Prediction: Actual vs. Predicted")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


def plot_feature_importance(results, feature_cols, out_path="feature_importance.png"):
    rf_model = results["Random Forest"]["model"]
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=True)

    plt.figure(figsize=(8, 6))
    importances.plot(kind="barh", color="tab:green")
    plt.title("Random Forest — Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


# ----------------------------------------------------------------------
# 5. PREDICT THE NEXT DAY (using the most recent row of real data)
# ----------------------------------------------------------------------
def predict_next_day(model, data, feature_cols):
    latest_features = data[feature_cols].iloc[[-1]]
    prediction = model.predict(latest_features)[0]
    return prediction


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading data...")
    raw_df = load_data(csv_path=None)  # swap to csv_path="your_stock.csv" for real data
    print(f"Loaded {len(raw_df)} rows from {raw_df['Date'].min().date()} to {raw_df['Date'].max().date()}")

    print("\nEngineering features...")
    feature_df = engineer_features(raw_df)
    print(f"{len(feature_df)} rows remain after feature engineering (lags drop early rows)")

    print("\nTraining models...")
    results, dates_test, y_test, feature_cols = train_and_evaluate(feature_df)

    print("\nGenerating plots...")
    plot_results(dates_test, y_test, results)
    plot_feature_importance(results, feature_cols)

    best_model = results["Random Forest"]["model"]
    next_day_price = predict_next_day(best_model, feature_df, feature_cols)
    print(f"\nPredicted next closing price (Random Forest): ${next_day_price:.2f}")
    print(f"Last known closing price: ${feature_df['Close'].iloc[-1]:.2f}")