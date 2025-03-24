# data_processing.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import yfinance as yf
from datetime import datetime, timedelta
from typing import Tuple, Optional

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

def scale_data(df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[MinMaxScaler], Optional[MinMaxScaler]]:
    """
    Scales the features using MinMaxScaler.
    """
    try:
        # Create scalers
        feature_scaler = MinMaxScaler(feature_range=(-1, 1))
        return_scaler = MinMaxScaler(feature_range=(-1, 1))
        
        # Convert Returns to numpy array
        returns = df['Returns'].to_numpy().reshape(-1, 1)
        
        # Scale returns
        scaled_returns = return_scaler.fit_transform(returns)
        
        # Scale other features
        other_features = df.drop('Returns', axis=1)
        other_features_array = other_features.to_numpy()
        scaled_features = feature_scaler.fit_transform(other_features_array)
        
        # Combine scaled data back into a DataFrame
        scaled_df = pd.DataFrame(
            np.hstack([scaled_returns, scaled_features]),
            columns=['Returns'] + list(other_features.columns),
            dtype=np.float64
        )
        
        return scaled_df, feature_scaler, return_scaler
        
    except Exception as e:
        print(f"Error in scale_data: {str(e)}")
        return None, None, None

def create_features(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Creates technical indicators and features from price data.
    """
    try:
        # Create a copy to avoid modifying original data
        df = df.copy()
        
        # Convert to float64 to ensure consistent calculations
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = df[col].astype(np.float64)
        
        # Basic price features
        df['Returns'] = df['Close'].pct_change()
        df['Range'] = (df['High'] - df['Low']) / df['Close']
        df['Volume_1d_Change'] = df['Volume'].pct_change()
        
        # Moving averages
        ma5 = df['Close'].rolling(window=5).mean()
        ma20 = df['Close'].rolling(window=20).mean()
        ma50 = df['Close'].rolling(window=50).mean()
        
        # Price distances from MAs (as percentages)
        df['Price_MA5_Dist'] = ((df['Close'] - ma5) / df['Close']).astype(np.float64)
        df['Price_MA20_Dist'] = ((df['Close'] - ma20) / df['Close']).astype(np.float64)
        df['Price_MA50_Dist'] = ((df['Close'] - ma50) / df['Close']).astype(np.float64)
        
        # Volatility
        df['Volatility'] = df['Returns'].rolling(window=20).std()
        
        # RSI
        delta = df['Close'].diff()
        gain = delta.copy()
        loss = delta.copy()
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = -loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = (exp1 - exp2) / df['Close']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
        
        # Bollinger Bands
        rolling_mean = df['Close'].rolling(window=20).mean()
        rolling_std = df['Close'].rolling(window=20).std()
        
        # Calculate Bollinger Band distances as percentages
        df['BB_Upper_Dist'] = ((rolling_mean + (2 * rolling_std) - df['Close']) / df['Close']).astype(np.float64)
        df['BB_Lower_Dist'] = ((rolling_mean - (2 * rolling_std) - df['Close']) / df['Close']).astype(np.float64)
        df['BB_Width'] = ((4 * rolling_std) / rolling_mean).astype(np.float64)
        
        # Momentum indicators
        df['ROC'] = df['Close'].pct_change(periods=10)
        df['MOM'] = df['Close'].pct_change(periods=10)
        
        # Drop any rows with NaN values
        df = df.dropna()
        
        # Keep only the features we want for prediction
        feature_columns = [
            'Returns', 'Range', 'Volume_1d_Change',
            'Price_MA5_Dist', 'Price_MA20_Dist', 'Price_MA50_Dist',
            'Volatility', 'RSI', 'MACD', 'MACD_Hist',
            'BB_Upper_Dist', 'BB_Lower_Dist', 'BB_Width',
            'ROC', 'MOM'
        ]
        
        # Ensure all columns are float64
        result_df = df[feature_columns].astype(np.float64)
        
        return result_df
        
    except Exception as e:
        print(f"Error in create_features: {str(e)}")
        return None

def prepare_sequences(df: pd.DataFrame, seq_length: int = 20) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Creates sequences for LSTM and splits into training and testing sets.
    """
    try:
        # Convert DataFrame to numpy array
        data = df.values
        
        # Create sequences
        X, y = [], []
        for i in range(len(data) - seq_length):
            # Sequence of features
            X.append(data[i:(i + seq_length)])
            # Target is the next return
            y.append(data[i + seq_length, 0])  # 0 is the Returns column
        
        X = np.array(X)
        y = np.array(y)
        
        # Split into train and test sets (80-20 split)
        split_idx = int(len(X) * 0.8)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"Error in prepare_sequences: {str(e)}")
        return None, None, None, None