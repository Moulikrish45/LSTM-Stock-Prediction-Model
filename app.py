from data_processing import get_stock_data, preprocess_data, create_features, scale_data, create_sequences, train_test_split_ts
from model import build_lstm_model, train_model
from evaluate import evaluate_model, predict_future
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import streamlit as st

def main():
    st.title("Stock Price Prediction")

    # Get user inputs
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, GOOG):", "AAPL").upper()
    n_days = st.number_input("Number of Days to Predict:", min_value=1, max_value=365, value=30)
    seq_length = st.number_input("Sequence Length (LSTM Input):", min_value=5, max_value=60, value=20)

    if st.button("Predict"):
        with st.spinner("Fetching and processing data..."):
            # Use get_stock_data from data_processing.py
            df = get_stock_data(stock_symbol)  # Pass the symbol
            if df is None:
                st.error(f"Could not retrieve data for {stock_symbol}. Please check the symbol and file.")
                return

            df = preprocess_data(df)
            df = create_features(df, lags=seq_length)
            scaled_df, scaler = scale_data(df)

            # Get the index of the 'Close' column AFTER scaling
            target_column_name = 'Close'
            target_column_index = scaled_df.columns.get_loc(target_column_name)

            X, y = create_sequences(scaled_df.values, seq_length, target_column_index)
            X_train, X_test, y_train, y_test = train_test_split_ts(X, y)

        with st.spinner("Building and training model..."):
            X_train, X_val, y_train, y_val = train_test_split_ts(X_train, y_train, test_size=0.2)
            lstm_model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            trained_model, history = train_model(lstm_model, X_train, y_train, X_val, y_val)

        with st.spinner("Evaluating model..."):
            rmse, mae, r2, y_test_inv, y_pred_inv = evaluate_model(trained_model, X_test, y_test, scaler)
            st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
            st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
            st.write(f"R-squared: {r2:.3f}")

        st.subheader("Training History")
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(history.history['loss'], label='Training Loss')
        ax_loss.plot(history.history['val_loss'], label='Validation Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.legend()
        st.pyplot(fig_loss)

        st.subheader("Actual vs Predicted (Test Set)")
        fig_pred, ax_pred = plt.subplots()
        ax_pred.plot(y_test_inv, label='Actual')
        ax_pred.plot(y_pred_inv, label='Predicted')
        ax_pred.set_xlabel('Time')
        ax_pred.set_ylabel('Price')
        ax_pred.legend()
        st.pyplot(fig_pred)

    with st.spinner("Predicting future prices..."):
            last_sequence = X_test[-1]
            future_preds = predict_future(trained_model, last_sequence, scaler, n_days)
            st.subheader(f"Predicted Prices for the Next {n_days} Days:")
            # Create dates for the future predictions
            last_date = df.index[-1]
            future_dates = [last_date + datetime.timedelta(days=i) for i in range(1, n_days + 1)]
            future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_preds})
            future_df = future_df.set_index('Date')
            st.dataframe(future_df) #Display as dataframe

            st.subheader("Future Price Predictions")
            fig_future, ax_future = plt.subplots()
            ax_future.plot(df.index, df['Close'], label='Historical Close')
            ax_future.plot(future_df.index, future_df['Predicted Price'], label='Future Predictions')
            ax_future.set_xlabel('Date')
            ax_future.set_ylabel('Price')
            ax_future.legend()
            st.pyplot(fig_future)

if __name__ == "__main__":
    main()