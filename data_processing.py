# data_processing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
import yfinance as yf
from datetime import datetime, timedelta
from typing import Tuple, Optional
import warnings

def get_stock_data(stock_symbol):
    """
    Retrieves stock data from Yahoo Finance.
    """
    try:
        # Get data for the last 5 years
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)
        
        # Download data from Yahoo Finance
        stock = yf.Ticker(stock_symbol)
        df = stock.history(start=start_date, end=end_date)
        
        if df.empty:
            print(f"No data found for symbol '{stock_symbol}'")
            return None
            
        # Rename columns to match our expected format
        df = df.rename(columns={
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })
        
        # Remove any other columns we don't need
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        print("----- get_stock_data -----")
        print("Shape of DataFrame:", df.shape)
        print("First 5 rows:\n", df.head())
        print("Data types:\n", df.dtypes)
        print("Missing values:\n", df.isnull().sum())
        
        return df
        
    except Exception as e:
        print(f"Error fetching data for {stock_symbol}: {str(e)}")
        return None

def preprocess_data(df, features_to_drop=()):
    """
    Cleans and preprocesses the data.
    """
    if df is None:  # Handle the case where get_stock_data returns None
        print("Error in preprocess_data: Input DataFrame is None.")
        return None
    # Drop unnecessary columns
    df = df.drop(columns=features_to_drop, errors='ignore')

    # Handle missing values (forward fill, then backward fill)
    df = df.ffill().bfill()

    # Check for any remaining missing values (and handle them)
    if df.isnull().any().any():
        print("Warning: Missing values still present after handling.")
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mean())

    # Debugging prints
    print("\n----- preprocess_data -----")
    print("Shape after preprocessing:", df.shape)
    print("First 5 rows after preprocessing:\n", df.head())
    print("Missing values after handling:\n", df.isnull().sum())

    return df

def scale_data(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[RobustScaler]]:
    """
    Scales the features using RobustScaler.
    """
    try:
        # Ensure data types
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)
        
        # Calculate percentage returns (target)
        df['Returns'] = df['Close'].pct_change(1) * 100  # Convert to percentage
        df['Target'] = df['Returns'].shift(-1)  # Tomorrow's return is our target
        
        # Calculate volatility for sample weights
        df['Volatility'] = df['Returns'].rolling(window=5).std()
        df['Sample_Weight'] = 1 / (df['Volatility'] + 1e-6)  # Avoid division by zero
        
        # Technical indicators (all in percentage terms)
        df['Returns_3d'] = df['Close'].pct_change(3) * 100
        df['Returns_5d'] = df['Close'].pct_change(5) * 100
        
        # Price momentum
        df['RSI'] = calculate_rsi(df['Close'], window=14)
        df['MACD'] = calculate_macd(df['Close'])
        
        # Volatility indicators
        df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['Close'] * 100
        
        # Volume features
        df['Volume_Change'] = df['Volume'].pct_change() * 100
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_MA5_Change'] = df['Volume_MA5'].pct_change() * 100
        
        # Drop rows with NaN values from feature calculations
        df = df.dropna()
        
        # Filter for normal market conditions
        normal_market = (
            (df['Volatility'] < df['Volatility'].quantile(0.8)) & 
            (df['Volume'] > df['Volume'].quantile(0.2))
        )
        df = df[normal_market]
        
        # Initialize scaler
        feature_scaler = RobustScaler()
        
        # Scale features
        feature_cols = [col for col in df.columns if col not in ['Target', 'Sample_Weight']]
        scaled_features = feature_scaler.fit_transform(df[feature_cols])
        scaled_df = pd.DataFrame(scaled_features, columns=feature_cols, index=df.index)
        
        # Add back target and sample weights
        scaled_df['Target'] = df['Target']
        scaled_df['Sample_Weight'] = df['Sample_Weight']
        
        return scaled_df, feature_scaler
        
    except Exception as e:
        print(f"Error in scale_data: {str(e)}")
        return None, None

def create_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Creates features from price data with proper percentage returns and volatility scaling.
    """
    try:
        # Ensure data types
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(np.float64)
        
        # Calculate percentage returns (target)
        df['Returns'] = df['Close'].pct_change(1) * 100  # Convert to percentage
        df['Target'] = df['Returns'].shift(-1)  # Tomorrow's return is our target
        
        # Calculate volatility for sample weights
        df['Volatility'] = df['Returns'].rolling(window=5).std()
        df['Sample_Weight'] = 1 / (df['Volatility'] + 1e-6)  # Avoid division by zero
        
        # Technical indicators (all in percentage terms)
        df['Returns_3d'] = df['Close'].pct_change(3) * 100
        df['Returns_5d'] = df['Close'].pct_change(5) * 100
        
        # Price momentum
        df['RSI'] = calculate_rsi(df['Close'], window=14)
        df['MACD'] = calculate_macd(df['Close'])
        
        # Volatility indicators
        df['BB_Upper'], df['BB_Lower'] = calculate_bollinger_bands(df['Close'])
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['Close'] * 100
        
        # Volume features
        df['Volume_Change'] = df['Volume'].pct_change() * 100
        df['Volume_MA5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_MA5_Change'] = df['Volume_MA5'].pct_change() * 100
        
        # Drop rows with NaN values from feature calculations
        df = df.dropna()
        
        # Filter for normal market conditions
        normal_market = (
            (df['Volatility'] < df['Volatility'].quantile(0.8)) & 
            (df['Volume'] > df['Volume'].quantile(0.2))
        )
        df = df[normal_market]
        
        return df
        
    except Exception as e:
        print(f"Error in create_features: {str(e)}")
        return None

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices):
    exp1 = prices.ewm(span=12, adjust=False).mean()
    exp2 = prices.ewm(span=26, adjust=False).mean()
    return exp1 - exp2

def calculate_bollinger_bands(prices, window=20, num_std=2):
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def prepare_sequences(df: pd.DataFrame, seq_length: int = 10) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Prepares sequences ensuring no future data leakage.
    """
    try:
        # Separate features from target and weights
        features = df.drop(['Target', 'Sample_Weight'], axis=1).values
        targets = df['Target'].values
        weights = df['Sample_Weight'].values
        
        X, y, w = [], [], []
        for i in range(len(df) - seq_length):
            X.append(features[i:(i + seq_length)])
            y.append(targets[i + seq_length])
            w.append(weights[i + seq_length])
        
        return np.array(X), np.array(y), np.array(w)
        
    except Exception as e:
        print(f"Error in prepare_sequences: {str(e)}")
        return None, None, None