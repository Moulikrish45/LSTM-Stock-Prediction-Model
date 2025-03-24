# evaluate.py (MAKE SURE THIS IS EXACTLY WHAT YOU HAVE)
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def evaluate_model(model, X_test, y_test, close_scaler, original_prices):
    """
    Evaluates the model and converts return predictions back to prices.
    """
    # Get predictions (returns)
    predictions = model.predict(X_test)
    
    # Inverse transform the scaled returns
    predictions = close_scaler.inverse_transform(predictions.reshape(-1, 1))
    actual_returns = close_scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Convert returns back to prices
    predicted_prices = np.zeros(len(predictions))
    actual_prices = np.zeros(len(actual_returns))
    
    # First price is the last price from training data
    predicted_prices[0] = original_prices[-1] * (1 + predictions[0][0])
    actual_prices[0] = original_prices[-1] * (1 + actual_returns[0][0])
    
    # Calculate subsequent prices using cumulative returns
    for i in range(1, len(predictions)):
        predicted_prices[i] = predicted_prices[i-1] * (1 + predictions[i][0])
        actual_prices[i] = actual_prices[i-1] * (1 + actual_returns[i][0])
    
    # Calculate metrics using the actual prices
    r2 = r2_score(actual_prices, predicted_prices)
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mae = mean_absolute_error(actual_prices, predicted_prices)
    
    print(f"\nModel Evaluation Metrics:")
    print(f"R-squared: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    return predicted_prices, actual_prices, (r2, rmse, mae)

def predict_future(model, last_sequence, n_future, close_scaler, last_price):
    """
    Predicts future stock returns and converts them to prices.
    """
    future_returns = []
    current_sequence = last_sequence.copy()
    
    for _ in range(n_future):
        # Predict next return
        current_return = model.predict(current_sequence.reshape(1, *current_sequence.shape[1:]))
        future_returns.append(current_return[0][0])
        
        # Update sequence for next prediction
        current_sequence = np.roll(current_sequence, -1, axis=1)
        current_sequence[0, -1, -1] = current_return[0][0]  # Update only the return value
    
    # Convert predicted returns to prices
    future_returns = np.array(future_returns).reshape(-1, 1)
    future_returns = close_scaler.inverse_transform(future_returns)
    
    future_prices = np.zeros(len(future_returns))
    future_prices[0] = last_price * (1 + future_returns[0][0])
    
    for i in range(1, len(future_returns)):
        future_prices[i] = future_prices[i-1] * (1 + future_returns[i][0])
    
    return future_prices