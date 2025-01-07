class LSTMAutoencoder:
    def __init__(self, train_data, test_data, timesteps: int, features: int, latent_dim: int = 32, lstm_units: int = 64, window_size: int = 128, threshold_sigma=2.0):

        self.train_data = train_data
        self.test_data = test_data 
        self.train_data_window = create_windows(self.train_data, window_size, step_size=1)
        self.test_data_window = create_windows(self.test_data, window_size, step_size=1)
        self.timesteps = timesteps
        self.features = features
        self.latent_dim = latent_dim
        self.lstm_units = lstm_units
        self.model = None  # Model is not built yet.
        self.threshold = self.compute_threshold(threshold_sigma=threshold_sigma)

    def _build_model(self):

        inputs = tf.keras.Input(shape=(self.timesteps, self.features), name='input_layer')
        x = layers.LSTM(self.lstm_units, return_sequences=True, name='lstm_1')(inputs)
        x = layers.LSTM(self.latent_dim, return_sequences=False, name='lstm_2')(x)

        # Decoder
        x = layers.RepeatVector(self.timesteps, name='repeat_vector')(x)
        x = layers.LSTM(self.latent_dim, return_sequences=True, name='lstm_3')(x)
        x = layers.LSTM(self.lstm_units, return_sequences=True, name='lstm_4')(x)
        outputs = layers.TimeDistributed(layers.Dense(self.features, name='dense_output'))(x)

        self.model = models.Model(inputs, outputs, name='model')

        return self.model

    def compute_threshold(self, threshold_sigma=2.0):

        rec = self.model.predict(self.train_data, verbose=0)
        mse = np.mean(np.square(self.train_data - rec), axis=(1, 2))
        return np.mean(mse) + threshold_sigma * np.std(mse)

    def infer_anomalies(self, threshold, window_size):

        rec = self.model.predict(self.test_data_window, verbose=0)
        mse = np.mean(np.square(self.test_data_window - rec), axis=(1, 2))
        anomaly_preds = mse > threshold
        anomaly_scores = mse
        return anomaly_preds, anomaly_scores

    def train(self, batch_size=32, epochs=50, window_size=None, step_size=1, optimizer='adam', loss='mse'):
        # Ensure the model is built before training
        self.model = self._build_model()

        # Compile the model with the specified optimizer and loss function
        self.model.compile(optimizer=optimizer, loss=loss)

        # Train the model
        self.model.fit(
            self.train_data_window, self.train_data_window,  # Use self.train_data for both input and output
            batch_size=batch_size,
            validation_split=0.1,  # Split 10% of the training data for validation
            epochs=epochs,
            verbose=1
        )

    def evaluate(self, batch_size=32, window_size=None, step_size=1):

        model = self._build_model()  # Ensure model is built before evaluation.
        return model.evaluate(self.test_data_window, batch_size=batch_size)

    def save_model(self, path: str):

        model = self._build_model()  # Ensure model is built before saving.
        model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):

        self.model = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")
