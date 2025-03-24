import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, X_test, y_test, scaler):
    """
    Evaluates the model using appropriate metrics.
    """
    y_pred = model.predict(X_test)
    # Inverse transform the predictions and actual values
    y_pred_inv = scaler.inverse_transform(np.concatenate((np.zeros((y_pred.shape[0], 3)), y_pred, np.zeros((y_pred.shape[0], X_test.shape[2]-4))), axis=1))[:, 3] #Inverse transform only the 3rd index
    y_test_inv = scaler.inverse_transform(np.concatenate((np.zeros((y_test.shape[0], 3)), y_test.reshape(-1,1), np.zeros((y_test.shape[0], X_test.shape[2]-4))), axis=1))[:, 3] #Inverse transform only the 3rd index

    rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv) #Added R-squared

    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")
    print(f"R-squared: {r2}") #Print R-squared

    return rmse, mae, r2, y_test_inv, y_pred_inv

def predict_future(model, last_sequence, scaler, n_days):
    """
    Predicts future stock prices for n_days.
    """
    future_predictions = []
    current_sequence = last_sequence.copy()

    for _ in range(n_days):
        # Reshape the sequence to match the model's input shape
        current_sequence_reshaped = current_sequence.reshape(1, current_sequence.shape[0], current_sequence.shape[1])
        # Make a prediction
        next_day_pred = model.predict(current_sequence_reshaped)[0, 0]

        # Inverse transform *only the predicted close price*
        dummy_array = np.zeros((1, last_sequence.shape[1]))  # Create a dummy array of the right shape
        dummy_array[0, 3] = next_day_pred  #  Put the prediction in the 'Close' column index
        next_day_pred_inv = scaler.inverse_transform(dummy_array)[0, 3]
        future_predictions.append(next_day_pred_inv)

        # Update the sequence:  Shift values and add the *scaled* prediction
        new_row = current_sequence[-1, 1:].tolist()  # Drop the oldest value
        new_row.append(next_day_pred) # Append the new *scaled* prediction
        current_sequence = np.vstack((current_sequence[1:], np.array(new_row).reshape(1, -1)))

    return future_predictions