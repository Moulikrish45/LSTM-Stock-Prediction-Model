# model.py
import tensorflow as tf
from keras import layers, models, callbacks, regularizers, optimizers, metrics
from scipy.ndimage import gaussian_filter1d
import numpy as np

class DataNoise(layers.Layer):
    def __init__(self, noise_level=0.01, **kwargs):
        super().__init__(**kwargs)
        self.noise_level = noise_level

    def call(self, inputs, training=False):
        if training:
            return inputs + tf.random.normal(tf.shape(inputs), 0, self.noise_level)
        return inputs

class AttentionLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.attention_dense = layers.Dense(1, activation='tanh')

    def call(self, inputs):
        # Calculate attention scores
        attention_scores = self.attention_dense(inputs)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)

        # Apply attention weights
        attended_output = tf.reduce_sum(inputs * attention_weights, axis=1)
        return attended_output

class DetrendLayer(layers.Layer):
    def __init__(self, window_size=5, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size

    def call(self, inputs):
        # Handle both 2D and 3D inputs
        input_shape = tf.shape(inputs)
        if len(inputs.shape) == 2:
            inputs = tf.expand_dims(inputs, axis=-1)

        # Calculate moving average
        ma = tf.nn.avg_pool1d(inputs, self.window_size, 1, 'SAME')
        detrended = inputs - ma

        # Restore original shape if needed
        if len(inputs.shape) == 2:
            detrended = tf.squeeze(detrended, axis=-1)
        return detrended

class MeanAbsoluteErrorOnMean(metrics.Metric):
    def __init__(self, name='mae_mean', **kwargs):
        super().__init__(name=name, **kwargs)
        self.mae = metrics.MeanAbsoluteError()

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Extract mean predictions (first column)
        y_pred_mean = y_pred[:, 0]
        return self.mae.update_state(y_true, y_pred_mean, sample_weight)

    def result(self):
        return self.mae.result()

    def reset_state(self):
        self.mae.reset_state()

class MeanSquaredErrorOnMean(metrics.Metric):
    def __init__(self, name='mse_mean', **kwargs):
        super().__init__(name=name, **kwargs)
        self.mse = metrics.MeanSquaredError()

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Extract mean predictions (first column)
        y_pred_mean = y_pred[:, 0]
        return self.mse.update_state(y_true, y_pred_mean, sample_weight)

    def result(self):
        return self.mse.result()

    def reset_state(self):
        self.mse.reset_state()

def gaussian_likelihood_loss(y_true, y_pred):
    """
    Enhanced loss function with trading-specific penalties
    """
    mean = y_pred[:, 0]
    var = tf.math.softplus(y_pred[:, 1]) + 1e-6
    
    # Standard gaussian NLL
    nll = 0.5 * (tf.math.log(var) + tf.square(y_true - mean)/var)
    
    # Trading-specific penalties
    direction_penalty = 0.1 * tf.reduce_mean(tf.square(tf.sign(mean) - tf.sign(y_true)))
    momentum_penalty = 0.05 * tf.reduce_mean(tf.abs(mean[1:] - mean[:-1]))
    
    return nll + direction_penalty + momentum_penalty

class SharpeRatioMetric(metrics.Metric):
    def __init__(self, name='sharpe_ratio', **kwargs):
        super().__init__(name=name, **kwargs)
        self.returns_sum = self.add_weight(name='returns_sum', initializer='zeros')
        self.returns_squared_sum = self.add_weight(name='returns_squared_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        mean_pred = y_pred[:, 0]
        # Calculate returns based on directional accuracy
        returns = tf.where(tf.sign(mean_pred) == tf.sign(y_true),
                          tf.abs(mean_pred),
                          -tf.abs(mean_pred))
        
        self.returns_sum.assign_add(tf.reduce_sum(returns))
        self.returns_squared_sum.assign_add(tf.reduce_sum(tf.square(returns)))
        self.count.assign_add(tf.cast(tf.size(returns), tf.float32))
    
    def result(self):
        mean_return = self.returns_sum / (self.count + 1e-6)
        variance = (self.returns_squared_sum / (self.count + 1e-6)) - tf.square(mean_return)
        return mean_return / (tf.sqrt(variance) + 1e-6)

def build_lstm_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Enhanced feature extraction
    x = DataNoise(noise_level=0.008)(inputs)  # Reduced noise
    x = DetrendLayer(window_size=7)(x)  # Increased window size
    
    # Multi-scale convolution block
    conv_3 = layers.Conv1D(32, 3, padding='causal', activation='relu')(x)
    conv_5 = layers.Conv1D(32, 5, padding='causal', activation='relu')(x)
    conv_7 = layers.Conv1D(32, 7, padding='causal', activation='relu')(x)
    x = layers.Concatenate()([conv_3, conv_5, conv_7])
    
    # Bidirectional LSTM layers for better pattern recognition
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True,
                                       kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=2e-4)))(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True,
                                       kernel_regularizer=regularizers.l1_l2(l1=1e-4, l2=2e-4)))(x)
    
    # Multi-head attention
    attention_1 = AttentionLayer()(x)
    attention_2 = AttentionLayer()(x)
    x = layers.Concatenate()([attention_1, attention_2])
    
    # Deep dense layers with residual connections
    dense_1 = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.15)(dense_1)
    dense_2 = layers.Dense(128, activation='relu')(x)
    x = layers.Add()([dense_1, dense_2])  # Residual connection
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    # Separate branches for mean and variance
    mean_branch = layers.Dense(32, activation='relu')(x)
    var_branch = layers.Dense(32, activation='relu')(x)
    
    mean_output = layers.Dense(1)(mean_branch)
    var_output = layers.Dense(1)(var_branch)
    outputs = layers.Concatenate()([mean_output, var_output])
    
    model = models.Model(inputs, outputs)
    
    optimizer = optimizers.AdamW(
        learning_rate=0.0003,  # Further reduced learning rate
        weight_decay=0.0003,
        clipnorm=0.3
    )
    
    model.compile(
        optimizer=optimizer,
        loss=gaussian_likelihood_loss,
        metrics=[
            MeanAbsoluteErrorOnMean(),
            MeanSquaredErrorOnMean(),
            SharpeRatioMetric()
        ]
    )
    
    return model

def train_model(model, X_train, y_train, sample_weights, X_val=None, y_val=None, epochs=150):
    """
    Trains the model with improved callbacks and monitoring.
    """
    def cosine_decay(epoch):
        """Cosine decay learning rate schedule that returns a Python float"""
        initial_lr = 0.001
        decay = 0.1 + 0.9 * (1 + np.cos(epoch/epochs * np.pi))/2
        return float(initial_lr * decay)

    callbacks_list = [
        # Early stopping with conservative patience
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=25,
            restore_best_weights=True,
            mode='min'
        ),

        # Cosine decay learning rate
        callbacks.LearningRateScheduler(cosine_decay),

        # Model checkpoint
        callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_loss',
            save_best_only=True,
            mode='min'
        )
    ]

    # Use validation split if validation data not provided
    if X_val is None or y_val is None:
        validation_split = 0.2
        validation_data = None
    else:
        validation_split = 0.0
        validation_data = (X_val, y_val)

    # Train with sample weights
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=64,
        sample_weight=sample_weights,
        validation_split=validation_split,
        validation_data=validation_data,
        callbacks=callbacks_list,
        verbose=1
    )

    return model, history

def smooth_predictions(preds, sigma=2):
    """
    Applies Bayesian smoothing to predictions.
    """
    return gaussian_filter1d(preds, sigma=sigma)