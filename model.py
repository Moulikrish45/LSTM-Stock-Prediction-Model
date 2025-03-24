# model.py
import tensorflow as tf
from keras import models, layers, callbacks, regularizers, optimizers, losses

def build_lstm_model(input_shape):
    """
    Builds an improved LSTM model specifically designed for returns prediction.
    """
    model = models.Sequential([
        # Input layer with normalization
        layers.LayerNormalization(input_shape=input_shape),
        
        # First LSTM layer
        layers.LSTM(50, return_sequences=True,
                   kernel_regularizer=regularizers.l2(0.001),
                   recurrent_regularizer=regularizers.l2(0.001)),
        layers.LayerNormalization(),
        layers.Dropout(0.2),
        
        # Second LSTM layer
        layers.LSTM(30, return_sequences=False,
                   kernel_regularizer=regularizers.l2(0.001),
                   recurrent_regularizer=regularizers.l2(0.001)),
        layers.LayerNormalization(),
        layers.Dropout(0.2),
        
        # Dense layers
        layers.Dense(20, activation='relu',
                    kernel_regularizer=regularizers.l2(0.001)),
        layers.LayerNormalization(),
        layers.Dropout(0.1),
        
        # Output layer (no activation for regression)
        layers.Dense(1)
    ])
    
    # Use Huber loss for robustness to outliers
    optimizer = optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, 
                loss=losses.Huber(delta=1.0))
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """
    Trains the LSTM model with improved training parameters.
    """
    callbacks_list = [
        # Early stopping with more patience
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            mode='min'
        ),
        # Learning rate reduction
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=10,
            min_lr=1e-6,
            mode='min',
            verbose=1
        ),
        # Model checkpoint
        callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
    ]

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        callbacks=callbacks_list,
        verbose=1
    )
    
    return model, history