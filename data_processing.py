import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

def get_stock_data(stock_symbol):
    """
    Retrieves stock data from a CSV file.
    """
    file_path = 'prices.csv'
    df = pd.read_csv(file_path)
    df = df[df['symbol'] == stock_symbol].copy()
    
    # Convert date to datetime and set as index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    # Rename columns to uppercase
    df.rename(columns={
        'open': 'Open',
        'close': 'Close',
        'low': 'Low',
        'high': 'High',
        'volume': 'Volume'
    }, inplace=True)
    
    # Drop the symbol column as it's no longer needed
    df.drop(columns=['symbol'], inplace=True)
    
    return df

def preprocess_data(df, features_to_drop=()):
    """
    Cleans and preprocesses the stock data.
    """
    # Drop unnecessary columns
    df = df.drop(columns=features_to_drop, errors='ignore')
    # Handle missing values (forward fill, then backward fill)
    df = df.ffill().bfill()

    # Check for any remaining missing values
    if df.isnull().any().any():
        print("Warning: Missing values still present after handling.")
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
    return df

def create_features(df, lags=5):
    """
    Creates lagged features and technical indicators.
    """
    # Lagged features (previous days' prices)
    for lag in range(1, lags + 1):
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)

    # Rolling mean (Moving Average)
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()

    # Exponential Moving Average
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # MACD (Moving Average Convergence Divergence)
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['StdDev_20'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (df['StdDev_20'] * 2)
    df['Lower_Band'] = df['SMA_20'] - (df['StdDev_20'] * 2)

    # Drop rows with NaN values resulting from feature engineering
    df = df.dropna()
    return df

def scale_data(df):
    """
    Scales the data using MinMaxScaler.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    scaled_df = pd.DataFrame(scaled_data, columns=df.columns, index=df.index)
    return scaled_df, scaler

def create_sequences(data, seq_length, target_column_index):
    """
    Creates sequences for LSTM input.
    """
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length, target_column_index]  # Predicting based on index
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_test_split_ts(X, y, test_size=0.2):
    """
    Performs a time-series split.  Crucial for time-series data.
    """
    tscv = TimeSeriesSplit(n_splits=int(1/test_size))
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    return X_train, X_test, y_train, y_test