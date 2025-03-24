import tensorflow as tf
from keras import models, layers, callbacks

def build_lstm_model(input_shape):
    """
    Builds an LSTM model.
    """
    model = models.Sequential()
    model.add(layers.LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(units=50, return_sequences=True))  # Add another LSTM layer
    model.add(layers.Dropout(0.2))
    model.add(layers.LSTM(units=50)) # And another one
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Trains the LSTM model with early stopping and learning rate reduction.
    """
    early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_val, y_val), callbacks=[early_stopping, reduce_lr],
                        verbose=1)  # Add verbose for progress output
    return model, history