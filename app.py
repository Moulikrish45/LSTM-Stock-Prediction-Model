# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import data_processing
import model
import evaluate

# Add custom CSS for styling
st.markdown("""
    <style>
    .stButton > button {
        background-color: #262730;
        color: white;
        border: none;
        padding: 10px;
        margin: 2px 0;
        width: 100%;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background-color: #FF4B4B;
        color: white;
    }
    .stButton > button:focus {
        background-color: #FF4B4B;
        color: white;
        border: none;
    }
    .stSlider > div > div > div:nth-child(2) > div { /*  Slider */
        background-color: #FF4B4B;
    }
    .stNumberInput{
        width: 20;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title('Stock Price Prediction App')
    
    # Project Description
    st.markdown("""
This app utilizes a sophisticated **LSTM (Long Short-Term Memory) neural network** to predict future stock prices.

**How it works:**

* Analyzes historical stock data to identify underlying patterns.
* Forecasts future price movements by predicting **percentage returns**.
* Converts these predicted percentage returns back into actionable price predictions.

**Important Considerations:**

* Stock market prediction is inherently challenging.
* Past performance is not indicative of future results.

---
    """)
    
    # Sidebar inputs
    st.sidebar.header('User Input Parameters')
    
    # Stock Symbol Selection (Dropdown)
    available_symbols = ['AAPL', 'AMZN', 'NVDA', 'WMT', 'GOOGL', 'META', 'JPM', 'MSFT', 'TSLA', 'V']
    stock_symbol = st.sidebar.selectbox(
        'Select Stock Symbol:', 
        available_symbols, 
        index=available_symbols.index('GOOGL')
    )
    
    # Number of Days to Predict (Slider)
    n_days = st.sidebar.slider(
        "Number of Days to Predict:",
        min_value=1,
        max_value=30,
        value=7,
        step=1,
        help="Select the number of days for future price prediction"
    )
    
    # Sequence Length (Slider)
    seq_length = st.sidebar.slider(
        "Sequence Length:",
        min_value=5,
        max_value=60,
        value=20,
        step=1,
        help="Number of time steps to use for prediction"
    )
    
    if st.button('Predict'):
        with st.spinner('Please be patient, even our AI needs some potions :potion:...(Abracadabra :magic_wand:)'):
            # Get and process data
            df = data_processing.get_stock_data(stock_symbol)
            if df is None:
                st.error(f'Could not retrieve data for {stock_symbol}')
                return
                
            df = data_processing.preprocess_data(df)
            if df is None:
                st.error('Error preprocessing data')
                return
                
            # Create features and scale data
            featured_df = data_processing.create_features(df)
            if featured_df is None:
                st.error('Error creating features')
                return
                
            scaled_df, feature_scaler = data_processing.scale_data(featured_df)
            if scaled_df is None or feature_scaler is None:
                st.error('Error scaling data')
                return
            # Prepare sequences for LSTM
            X, y, sample_weights = data_processing.prepare_sequences(scaled_df, seq_length)
            if X is None or y is None or sample_weights is None:
                st.error('Error preparing sequences')
                return
        with st.spinner('We are doing some magic to make it work...(Abracadabra :magic_wand:)'):
            # Split training data into train and validation sets
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            weights_train = sample_weights[:split_idx]  # Only split training weights
            
            # Build and train model
            lstm_model = model.build_lstm_model(input_shape=(X.shape[1], X.shape[2]))
            trained_model, history = model.train_model(
                lstm_model, X_train, y_train, 
                sample_weights=weights_train,
                X_val=X_test, 
                y_val=y_test
            )
            
            if trained_model is None or history is None:
                st.error('Error training model')
                return
            
            # # Display training history
            # st.subheader('Training History')
            # fig_history = go.Figure()
            # fig_history.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'))
            # if 'val_loss' in history.history:
            #     fig_history.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
            # fig_history.update_layout(title='Model Loss During Training',
            #                         xaxis_title='Epoch',
            #                         yaxis_title='Loss')
            # st.plotly_chart(fig_history)
        
        with st.spinner('We are cooking the results...(Let him cook :pizza:)'):
            # Evaluate model
            if df is not None and isinstance(df['Close'].values, np.ndarray):
                predictions, actuals, confidence_bounds, metrics = evaluate.evaluate_model(
                    trained_model, X_test, y_test, scaler=feature_scaler, original_prices=df['Close'].values
                )
                
                if predictions is not None and actuals is not None and metrics is not None:
                    # Display metrics
                    st.subheader('Model Performance Metrics:')
                    with st.expander("ℹ️ What kind of sorcery is this?"):
                        st.markdown("""
                            * **R-squared**: Measures prediction accuracy (0-1). Higher is better. 
                            * **RMSE**: Average prediction error in percentage. Lower is better.
                            * **Sharpe Ratio**: Risk-adjusted return measure. Higher than 1 is good.
                            * **Max Drawdown**: Largest price drop from peak to trough.
                            * **MAE**: Average absolute error in percentage. Lower is better.
                            * **Turnover Correlation**: How well the model captures price changes.
                        """)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric('R-squared', f"{metrics['r2']:.3f}")
                        st.metric('RMSE (%)', f"{metrics['rmse']:.2f}")
                    with col2:
                        st.metric('Sharpe Ratio', f"{metrics['sharpe']:.2f}")
                        st.metric('Max Drawdown (%)', f"{metrics['max_drawdown']*100:.1f}")
                    with col3:
                        st.metric('MAE (%)', f"{metrics['mae']:.2f}")
                        st.metric('Turnover Correlation', f"{metrics['turnover_corr']:.2f}")
                    
                    # Plot actual vs predicted
                    st.subheader('Actual vs Predicted Prices (Test Set)')
                    with st.expander("ℹ️ How am I supposed to read this?"):
                        st.markdown("""
                            * **Blue line**: Actual historical prices
                            * **Orange line**: Model's predictions
                            * Close alignment between lines indicates good prediction accuracy
                            * Large gaps suggest prediction errors
                        """)
                    fig_test = go.Figure()
                    test_dates = df.index[-len(actuals):]
                    fig_test.add_trace(go.Scatter(x=test_dates, y=actuals, name='Actual', line=dict(color='orange', width=2)))
                    fig_test.add_trace(go.Scatter(x=test_dates, y=predictions, name='Predicted', line=dict(color='blue', width=2)))
                    fig_test.update_layout(xaxis_title='Date',
                                         yaxis_title='Price')
                    st.plotly_chart(fig_test)
                    
                    # Plot predictions with confidence bands
                    st.subheader('Cumulative Returns:')
                    with st.expander("ℹ️ Understanding cumulative returns"):
                        st.markdown("""
                            * Shows total percentage return over time
                            * Steeper upward slope = better performance
                            * Helps visualize long-term prediction accuracy
                            * Useful for comparing with market benchmarks
                        """)
                    fig = evaluate.plot_predictions(actuals, predictions, confidence_bounds)
                    st.plotly_chart(fig, key='cumulative_returns_plot')
                    
                    # Plot return distribution
                    st.subheader('Return Distribution:')
                    with st.expander("ℹ️ What is this drawings?"):
                        st.markdown("""
                            * Shows the distribution of returns
                            * Helps understand the model's performance
                            * Helps diagnose overfitting or underfitting
                        """)    
                    fig_dist = evaluate.plot_return_distribution(actuals, predictions)
                    st.plotly_chart(fig_dist, key='return_distribution_plot')
                    
                    # Plot training history
                    st.subheader('Training History:')
                    with st.expander("ℹ️ Another drawing, eh?"):
                        st.markdown("""
                            * Shows the loss (error) during training
                            * Lower loss = better model
                            * Helps diagnose overfitting or underfitting(i.e. If you see a large gap between training and validation loss, it means the model is overfitting)
                        """)
                    fig_history = evaluate.plot_training_history(history)
                    st.plotly_chart(fig_history, key='training_history_plot')
                else:
                    st.error('Error evaluating model: Invalid predictions or metrics')
            else:
                st.error('Error evaluating model: Invalid input data')
        
        with st.spinner('Generating future predictions...'):
            # Predict future prices
            if isinstance(X_test, np.ndarray) and len(X_test) > 0:
                last_sequence = X_test[-1]
                last_price = df['Close'].iloc[-1]
                future_preds, future_bounds = evaluate.predict_future(
                    trained_model, 
                    last_sequence,
                    n_steps=30,
                    scaler=feature_scaler,
                    last_price=last_price
                )
                
                if future_preds is not None:
                    # Generate future dates
                    last_date = df.index[-1]
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                               periods=len(future_preds),
                                               freq='B')  # Business days
                    
                    # Display future predictions
                    st.subheader(f'Predicted Prices for Next {n_days} Days')
                    future_df = pd.DataFrame({
                        'Date': future_dates,
                        'Predicted Price': future_preds
                    }).set_index('Date')
                    st.dataframe(future_df)
                    
                    # Plot historical + future prices
                    st.subheader('Historical and Future Price Predictions')
                    with st.expander("ℹ️ Unveiling the future through mystic charts..."):
                        st.markdown("""
                            * **Blue line**: Historical price data
                            * **Green line**: Predicted future prices
                            * **Shaded area**: 95% confidence interval
                            * Wider confidence bands = more uncertainty
                            * Use with caution for investment decisions
                        """)
                    fig_future = go.Figure()
                    
                    # Add historical data
                    fig_future.add_trace(go.Scatter(
                        x=df.index[-30:],
                        y=df['Close'].iloc[-30:],
                        name='Historical',
                        line=dict(color='#17BECF', width=2)  # Blue color
                    ))
                    
                    if future_bounds is not None and len(future_bounds) == 2:
                        upper_bound, lower_bound = future_bounds
                        
                        # Add confidence interval
                        fig_future.add_trace(go.Scatter(
                            x=future_dates,
                            y=upper_bound,
                            name='Confidence Interval',
                            line=dict(color='rgba(0, 177, 106, 0)'),
                            showlegend=False
                        ))
                        
                        fig_future.add_trace(go.Scatter(
                            x=future_dates,
                            y=lower_bound,
                            fill='tonexty',
                            fillcolor='rgba(0, 177, 106, 0.2)',
                            line=dict(color='rgba(0, 177, 106, 0)'),
                            name='95% Confidence',
                            showlegend=True
                        ))
                        
                        # Add prediction line
                        fig_future.add_trace(go.Scatter(
                            x=future_dates,
                            y=future_preds,
                            name='Prediction',
                            line=dict(color='#00B16A', width=2)  # Green color
                        ))
                    else:
                        st.error('Error: Future bounds are invalid or not provided.')
                    
                    # Update layout with improved styling
                    fig_future.update_layout(
                        title='30-Day Price Forecast',
                        xaxis_title='Date',
                        yaxis_title='Price',
                        hovermode='x',
                        showlegend=True,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        ),
                        template="plotly_dark",  # Dark theme
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    st.plotly_chart(fig_future)
                else:
                    st.error('Error generating future predictions')
            else:
                st.error('Error: Invalid test data for future predictions')

if __name__ == '__main__':
    main()