# evaluate.py (MAKE SURE THIS IS EXACTLY WHAT YOU HAVE)
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter1d
import model as model_module
import tensorflow as tf

def evaluate_model(model, X_test, y_test, scaler, original_prices):
    """
    Evaluates the model and returns predictions with confidence intervals.
    """
    try:
        # Make predictions (returns mean and variance)
        predictions = model.predict(X_test)
        mean_pred = predictions[:, 0]
        var_pred = tf.math.softplus(predictions[:, 1]).numpy() + 1e-6
        
        # Calculate confidence intervals (2 standard deviations)
        std_pred = np.sqrt(var_pred)
        upper_bound = mean_pred + 2 * std_pred
        lower_bound = mean_pred - 2 * std_pred
        
        # Convert predictions from percentage returns to prices
        last_prices = original_prices[-len(mean_pred)-1:-1]
        predicted_prices = last_prices * (1 + mean_pred/100)
        upper_prices = last_prices * (1 + upper_bound/100)
        lower_prices = last_prices * (1 + lower_bound/100)
        actual_prices = original_prices[-len(mean_pred):]
        
        # Calculate metrics
        metrics = calculate_metrics(actual_prices, predicted_prices)
        
        return predicted_prices, actual_prices, (upper_prices, lower_prices), metrics
        
    except Exception as e:
        print(f"Error in evaluate_model: {str(e)}")
        return None, None, None, None

def predict_future(model, last_sequence, n_steps, scaler, last_price):
    """
    Predicts future values with confidence intervals.
    """
    try:
        future_predictions = []
        future_upper = []
        future_lower = []
        current_sequence = last_sequence.copy()
        current_price = last_price
        
        for _ in range(n_steps):
            # Make prediction (returns mean and variance)
            pred = model.predict(current_sequence.reshape(1, *current_sequence.shape))
            mean_pred = pred[0, 0]
            var_pred = tf.math.softplus(pred[0, 1]).numpy() + 1e-6
            
            # Calculate confidence intervals
            std_pred = np.sqrt(var_pred)
            upper_bound = mean_pred + 2 * std_pred
            lower_bound = mean_pred - 2 * std_pred
            
            # Convert percentage returns to prices
            next_price = current_price * (1 + mean_pred/100)
            upper_price = current_price * (1 + upper_bound/100)
            lower_price = current_price * (1 + lower_bound/100)
            
            future_predictions.append(next_price)
            future_upper.append(upper_price)
            future_lower.append(lower_price)
            
            # Update sequence for next prediction
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1, 0] = mean_pred
            
            # Update current price
            current_price = next_price
        
        return np.array(future_predictions), (np.array(future_upper), np.array(future_lower))
        
    except Exception as e:
        print(f"Error in predict_future: {str(e)}")
        return None, None

def calculate_metrics(y_true, y_pred):
    """
    Calculates regression metrics and trading metrics.
    """
    try:
        # Basic regression metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        # Calculate returns
        true_returns = np.diff(y_true) / y_true[:-1]
        pred_returns = np.diff(y_pred) / y_pred[:-1]
        
        # Trading metrics
        sharpe_ratio = np.mean(pred_returns) / np.std(pred_returns) * np.sqrt(252)  # Annualized
        max_drawdown = np.max(np.maximum.accumulate(y_pred) - y_pred) / np.max(y_pred)
        turnover_corr = np.corrcoef(np.abs(true_returns), np.abs(pred_returns))[0, 1]
        
        return {
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'sharpe': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'turnover_corr': turnover_corr
        }
    except Exception as e:
        print(f"Error in calculate_metrics: {str(e)}")
        return None

def plot_predictions(y_true, y_pred, confidence_bounds=None):
    """
    Creates a plotly figure with confidence bands and cumulative returns.
    """
    try:
        # Calculate cumulative returns
        cum_true = np.cumprod(1 + np.diff(y_true) / y_true[:-1])
        cum_pred = np.cumprod(1 + np.diff(y_pred) / y_pred[:-1])
        
        fig = go.Figure()
        
        # Add confidence bands if available
        if confidence_bounds is not None:
            upper, lower = confidence_bounds
            cum_upper = np.cumprod(1 + np.diff(upper) / upper[:-1])
            cum_lower = np.cumprod(1 + np.diff(lower) / lower[:-1])
            
            fig.add_trace(go.Scatter(
                y=cum_upper,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                name='Upper Bound'
            ))
            
            fig.add_trace(go.Scatter(
                y=cum_lower,
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(68, 68, 68, 0.3)',
                line=dict(width=0),
                showlegend=False,
                name='Lower Bound'
            ))
        
        # Add actual and predicted lines
        fig.add_trace(go.Scatter(y=cum_true, name='Actual Returns'))
        fig.add_trace(go.Scatter(y=cum_pred, name='Predicted Returns'))
        
        fig.update_layout(
            title='Cumulative Returns',
            xaxis_title='Time',
            yaxis_title='Cumulative Return',
            showlegend=True
        )
        
        return fig
    except Exception as e:
        print(f"Error in plot_predictions: {str(e)}")
        return None

def plot_return_distribution(y_true, y_pred):
    """
    Creates a histogram of daily returns.
    """
    try:
        # Calculate daily returns
        true_returns = np.diff(y_true) / y_true[:-1]
        pred_returns = np.diff(y_pred) / y_pred[:-1]
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=true_returns, name='Actual Returns', opacity=0.75))
        fig.add_trace(go.Histogram(x=pred_returns, name='Predicted Returns', opacity=0.75))
        
        fig.update_layout(
            title='Daily Returns Distribution',
            xaxis_title='Return (%)',
            yaxis_title='Count',
            barmode='overlay'
        )
        
        return fig
    except Exception as e:
        print(f"Error in plot_return_distribution: {str(e)}")
        return None

def plot_training_history(history):
    """
    Creates a plotly figure showing training history.
    """
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=history.history['loss'], name='Training Loss'))
        if 'val_loss' in history.history:
            fig.add_trace(go.Scatter(y=history.history['val_loss'], name='Validation Loss'))
        fig.update_layout(
            title='Model Loss During Training',
            xaxis_title='Epoch',
            yaxis_title='Loss'
        )
        return fig
    except Exception as e:
        print(f"Error in plot_training_history: {str(e)}")
        return None