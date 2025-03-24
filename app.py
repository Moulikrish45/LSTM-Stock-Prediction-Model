# app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import data_processing
import model
import evaluate
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Optional, Tuple

def prepare_data(df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Helper function to prepare data and handle None values"""
    try:
        # First create features
        featured_df = data_processing.create_features(df)
        if featured_df is None:
            return None, None, None, None, None
            
        # Scale the features
        scaled_df, feature_scaler, return_scaler = data_processing.scale_data(featured_df)
        if scaled_df is None or feature_scaler is None or return_scaler is None:
            return None, None, None, None, None
            
        # Prepare sequences for LSTM
        X_train, X_test, y_train, y_test = data_processing.prepare_sequences(scaled_df, seq_length=20)
        if any(x is None for x in [X_train, X_test, y_train, y_test]):
            return None, None, None, None, None
            
        # Get original prices for later use
        original_prices = df['Close'].to_numpy()
        if isinstance(y_test, np.ndarray) and y_test.size > 0:
            original_prices = original_prices[:-y_test.size]
            
        return X_train, X_test, y_train, y_test, original_prices
    except Exception as e:
        st.error(f"Error in data preparation: {str(e)}")
        return None, None, None, None, None

def main():
    st.title('Stock Price Prediction App')
    
    # Sidebar for user input
    st.sidebar.header('User Input Parameters')
    
    # Stock symbol input
    stock_symbol = st.sidebar.text_input('Enter Stock Symbol (e.g., AAPL):', 'AAPL')
    
    # Date range input
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1000)  # Using more historical data
    
    try:
        # Fetch data
        df = yf.download(stock_symbol, start=start_date, end=end_date)
        if df is None or df.empty:
            st.error('No data found for the specified stock symbol.')
            return
            
        st.write(f"### Stock Data for {stock_symbol}")
        st.write("Last 5 days of raw data:")
        st.write(df.tail())
        
        # Prepare all data
        X_train, X_test, y_train, y_test, original_prices = prepare_data(df)
        
        # Verify all arrays exist and have the correct shape
        if (not isinstance(X_train, np.ndarray) or not isinstance(X_test, np.ndarray) or
            not isinstance(y_train, np.ndarray) or not isinstance(y_test, np.ndarray) or
            not isinstance(original_prices, np.ndarray)):
            st.error('Error in data preparation: Invalid array types')
            return
            
        if X_train.size == 0 or X_test.size == 0 or y_train.size == 0 or y_test.size == 0:
            st.error('Error in data preparation: Empty arrays')
            return
            
        # Build and train model
        input_shape = (X_train.shape[1], X_train.shape[2])
        lstm_model = model.build_lstm_model(input_shape=input_shape)
        trained_model, history = model.train_model(lstm_model, X_train, y_train, X_test, y_test)
        
        # Make predictions
        y_pred = trained_model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Display metrics
        st.write("### Model Performance Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("R-squared", f"{r2:.4f}")
        col2.metric("RMSE", f"{rmse:.4f}")
        col3.metric("MAE", f"{mae:.4f}")
        
        # Convert predictions and actual values back to prices
        last_price = original_prices[-1] if original_prices.size > 0 else 0
        predicted_prices = np.zeros(y_pred.size)
        actual_prices = np.zeros(y_test.size)
        
        # First price is based on the last training price
        if y_pred.size > 0 and y_test.size > 0:
            predicted_prices[0] = last_price * (1 + y_pred[0])
            actual_prices[0] = last_price * (1 + y_test[0])
            
            # Calculate subsequent prices
            for i in range(1, y_pred.size):
                predicted_prices[i] = predicted_prices[i-1] * (1 + y_pred[i])
                actual_prices[i] = actual_prices[i-1] * (1 + y_test[i])
        
        # Plot results
        fig = go.Figure()
        
        # Add actual prices
        fig.add_trace(go.Scatter(
            y=actual_prices,
            name='Actual',
            line=dict(color='blue')
        ))
        
        # Add predicted prices
        fig.add_trace(go.Scatter(
            y=predicted_prices,
            name='Predicted',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title=f'{stock_symbol} Stock Price Prediction',
            yaxis_title='Price',
            xaxis_title='Time',
            hovermode='x'
        )
        
        st.plotly_chart(fig)
        
        # Future predictions
        st.write("### Future Price Predictions")
        days_to_predict = st.slider('Select number of days to predict:', 1, 30, 7)
        
        # Prepare last sequence for prediction
        if X_test.size > 0:
            last_sequence = X_test[-1:].copy()
            
            # Predict future values
            future_returns = []
            current_sequence = last_sequence.copy()
            
            for _ in range(days_to_predict):
                next_return = trained_model.predict(current_sequence)
                if isinstance(next_return, np.ndarray) and next_return.size > 0:
                    future_returns.append(next_return[0])
                    
                    # Update sequence
                    current_sequence = np.roll(current_sequence, -1, axis=1)
                    current_sequence[0, -1] = next_return[0]
            
            # Convert future returns to prices
            future_prices = np.zeros(days_to_predict)
            if predicted_prices.size > 0 and len(future_returns) > 0:
                future_prices[0] = predicted_prices[-1] * (1 + future_returns[0])
                
                for i in range(1, days_to_predict):
                    future_prices[i] = future_prices[i-1] * (1 + future_returns[i])
            
            # Plot future predictions
            future_fig = go.Figure()
            
            # Add historical prices
            historical_prices = actual_prices[-30:] if actual_prices.size >= 30 else actual_prices
            future_fig.add_trace(go.Scatter(
                y=historical_prices,
                name='Historical',
                line=dict(color='blue')
            ))
            
            # Add future predictions
            future_fig.add_trace(go.Scatter(
                y=future_prices,
                name='Future Prediction',
                line=dict(color='red', dash='dash')
            ))
            
            future_fig.update_layout(
                title=f'{stock_symbol} Future Price Prediction',
                yaxis_title='Price',
                xaxis_title='Days',
                hovermode='x'
            )
            
            st.plotly_chart(future_fig)
            
            # Display predicted values
            st.write("### Predicted Prices for Next {} Days".format(days_to_predict))
            future_dates = pd.date_range(start=end_date, periods=days_to_predict, freq='B')
            future_predictions_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Price': future_prices
            })
            st.write(future_predictions_df)
        else:
            st.error("Not enough data for future predictions.")
        
    except Exception as e:
        st.error(f'An error occurred: {str(e)}')

if __name__ == '__main__':
    main()