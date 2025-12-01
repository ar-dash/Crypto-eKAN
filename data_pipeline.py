import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def download_data(ticker='BTC-USD', start_date='2010-03-01', end_date='2025-03-01'):
    asset = yf.Ticker(ticker)
    data = asset.history(start=start_date, end=end_date)
    return data

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def engineer_features(data):
    scaler = MinMaxScaler()
    data[['Close']] = scaler.fit_transform(data[['Close']])

    # Lag features
    for i in range(1, 5):
        data[f'prev{i}'] = data['Close'].shift(i)
        data[f'prevh{i}'] = data['High'].shift(i)
        data[f'prevl{i}'] = data['Low'].shift(i)

    # Technical indicators
    data['SMA'] = data['prev1'].rolling(window=7).mean()
    data['EMA'] = data['prev1'].ewm(span=30, adjust=False).mean()
    data['RSI'] = calculate_rsi(data['prev1'], window=14)

    # Drop unused columns
    data.drop(columns=['Open', 'High', 'Low', 'Dividends', 'Stock Splits'], inplace=True)
    data.dropna(inplace=True)
    data['Date'] = data.index

    return data, scaler

def split_data(data, test_size=0.2, val_split=True):
    train_df, test_df = train_test_split(data, test_size=test_size, random_state=42, shuffle=False)

    if val_split:
        val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42, shuffle=False)
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), None

def create_xy(df, features=None, target='Close'):
    if features is None:
        features = [col for col in df.columns if col.startswith('prev') or col in ['SMA', 'EMA', 'RSI']]
    X = df[features]
    y = df[target]
    return X, y
