import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import os
from Globals import MODELS_ROOT, PLOTS_ROOT

# Create directories if they don't exist
Path(MODELS_ROOT).mkdir(parents=True, exist_ok=True)
Path(PLOTS_ROOT).mkdir(parents=True, exist_ok=True)


class LSTMEnergyModel:
    """
    LSTM model for energy consumption prediction based on AAI-530 Module 4 template.
    
    This model uses sequential time series data to predict future energy consumption.
    """
    
    def __init__(self, sequence_length: int = 24, lstm_units: int = 50, 
                 num_layers: int = 2,
                 dropout_rate: float = 0.2, random_seed: int = 42):
        """
        Initialize LSTM model.
        
        Args:
            sequence_length: Number of time steps to look back
            lstm_units: Number of LSTM units in each layer
            num_layers: Number of LSTM layers
            dropout_rate: Dropout rate for regularization
            random_seed: Random seed for reproducibility
        """
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.random_seed = random_seed
        
        # Set random seeds for reproducibility
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
        
        self.model = None
        self.scaler = None
        self.feature_scaler = None
        self.history = None
        
    def create_sequences(self, data, target, sequence_length):
        """
        Create sequences for LSTM training.
        
        Args:
            data: Feature data (numpy array)
            target: Target data (numpy array)
            sequence_length: Number of time steps in each sequence
            
        Returns:
            X: Sequences of features (samples, sequence_length, features)
            y: Corresponding targets
        """
        X, y = [], []
        
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(target[i + sequence_length])
        
        return np.array(X), np.array(y)
    
    def prepare_data(self, df, feature_cols, target_col, train_split: float = 0.8, 
                     use_exogenous: bool = True):
        """
        Prepare data for LSTM training.
        
        Args:
            df: DataFrame containing the data
            feature_cols: List of feature column names (only used if use_exogenous=True)
            target_col: Target column name
            train_split: Fraction of data to use for training
            use_exogenous: If True, use exogenous features; if False, univariate (only target)
            
        Returns:
            X_train, X_test, y_train, y_test: Prepared sequences
        """
        # Extract target
        target = df[target_col].values.reshape(-1, 1)
        
        # Extract features based on mode
        if use_exogenous:
            # Multivariate: use exogenous features + target
            features = df[feature_cols + [target_col]].values
            print(f"\nMode: Multivariate (using {len(feature_cols)} exogenous features + target)")
        else:
            # Univariate: only use target (autoregressive)
            features = target
            print(f"\nMode: Univariate (autoregressive - only using target variable)")
        
        # Scale target (scaler fitted on target only — used for inverse_transform)
        self.scaler = MinMaxScaler()
        target_scaled = self.scaler.fit_transform(target)

        # Scale features
        if use_exogenous:
            self.feature_scaler = MinMaxScaler()
            features_scaled = self.feature_scaler.fit_transform(features)
        else:
            features_scaled = target_scaled
        
        # Create sequences
        X, y = self.create_sequences(features_scaled, target_scaled, self.sequence_length)
        
        # Split into train and test
        split_idx = int(len(X) * train_split)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        print(f"Training sequences: {X_train.shape}")
        print(f"Test sequences: {X_test.shape}")
        print(f"Sequence length: {self.sequence_length}")
        print(f"Number of features: {X_train.shape[2]}")
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, nb_features, nb_out=1):
        """
        Build LSTM model architecture.
        
        Args:
            nb_features: Number of input features
            nb_out: Number of output values (default 1 for regression)
        """
        self.model = Sequential()

        for i in range(self.num_layers):
            if i == 0:
            
                # First LSTM layer with return sequences
                self.model.add(LSTM(
                    units=self.lstm_units,
                    return_sequences=True,
                    input_shape=(self.sequence_length, nb_features)
                ))
            else:
                # Subsequent LSTM layers
                self.model.add(LSTM(units=self.lstm_units, return_sequences=(i < self.num_layers - 1)
                ))

            self.model.add(Dropout(self.dropout_rate))
        
        # Output layer (linear activation for regression)
        self.model.add(Dense(units=nb_out, activation='linear'))
        
        # Compile model (MSE for regression)
        self.model.compile(
            loss='mean_squared_error',
            optimizer='adam',
            metrics=['mae']
        )
        
        print("\nModel Architecture:")
        self.model.summary()
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs: int = 100, batch_size: int = 32, 
              validation_split: float = 0.1, model_path: str = None):
        """
        Train the LSTM model.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of training data to use for validation
            model_path: Path to save the best model
            
        Returns:
            Training history
        """
        if self.model is None:
            nb_features = X_train.shape[2]
            self.build_model(nb_features)
        
        # Set default model path if not provided
        if model_path is None:
            model_path = str(Path(MODELS_ROOT) / "lstm_model.h5")
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Prepare validation data
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = 0.0
        else:
            validation_data = None
        
        # Train model
        print("\nTraining LSTM model...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test sequences
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Inverse transform to get actual values
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred_actual = self.scaler.inverse_transform(y_pred)
        
        # Calculate metrics
        mse = mean_squared_error(y_test_actual, y_pred_actual)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_actual, y_pred_actual)
        r2 = r2_score(y_test_actual, y_pred_actual)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        print("\nLSTM Model Evaluation:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        
        return metrics, y_pred_actual, y_test_actual
    
    def plot_history(self, save_path: str = None):
        """
        Plot training history.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss (MSE)')
        axes[0].set_title('Model Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # MAE plot
        if 'mae' in self.history.history:
            axes[1].plot(self.history.history['mae'], label='Training MAE')
            if 'val_mae' in self.history.history:
                axes[1].plot(self.history.history['val_mae'], label='Validation MAE')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('MAE')
            axes[1].set_title('Mean Absolute Error')
            axes[1].legend()
            axes[1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_predictions(self, y_test, y_pred, save_path: str = None):
        """
        Plot predictions vs actual values.
        
        Args:
            y_test: Actual values
            y_pred: Predicted values
            save_path: Path to save the plot (optional)
        """
        plt.figure(figsize=(14, 6))
        
        # Plot subset for clarity
        plot_samples = min(200, len(y_test))
        
        plt.plot(y_test[:plot_samples], label='Actual', alpha=0.7)
        plt.plot(y_pred[:plot_samples], label='Predicted', alpha=0.7)
        plt.xlabel('Sample')
        plt.ylabel('Energy Consumption')
        plt.title('LSTM Predictions vs Actual Values')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_validation_predictions(self, X_val, y_val, save_path: str = None):
        """
        Plot predictions vs ground truth for validation set with detailed visualization.
        
        Args:
            X_val: Validation sequences
            y_val: Validation ground truth
            save_path: Path to save the plot (optional)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Make predictions
        y_pred = self.model.predict(X_val, verbose=0)
        
        # Inverse transform to get actual values
        y_val_actual = self.scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
        y_pred_actual = self.scaler.inverse_transform(y_pred).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_val_actual, y_pred_actual)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val_actual, y_pred_actual)
        r2 = r2_score(y_val_actual, y_pred_actual)
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Subplot 1: Time series comparison
        axes[0].plot(y_val_actual, label='Ground Truth', alpha=0.7, linewidth=2)
        axes[0].plot(y_pred_actual, label='Predictions', alpha=0.7, linewidth=2)
        axes[0].set_xlabel('Sample Index', fontsize=12)
        axes[0].set_ylabel('Energy Consumption (kWh)', fontsize=12)
        axes[0].set_title('Validation Set: Predictions vs Ground Truth', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        # Add metrics text box
        metrics_text = f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}'
        axes[0].text(0.02, 0.98, metrics_text, transform=axes[0].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', 
                    facecolor='wheat', alpha=0.5), fontsize=10)
        
        # Subplot 2: Scatter plot (predicted vs actual)
        axes[1].scatter(y_val_actual, y_pred_actual, alpha=0.5, s=30)
        
        # Add perfect prediction line
        min_val = min(y_val_actual.min(), y_pred_actual.min())
        max_val = max(y_val_actual.max(), y_pred_actual.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', 
                    linewidth=2, label='Perfect Prediction')
        
        axes[1].set_xlabel('Ground Truth (kWh)', fontsize=12)
        axes[1].set_ylabel('Predictions (kWh)', fontsize=12)
        axes[1].set_title('Prediction Accuracy Scatter Plot', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Validation plot saved to {save_path}")
        
        plt.show()
        
        return {
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred_actual,
            'ground_truth': y_val_actual
        }
    
    def save_model(self, path: str = None):
        """Save trained model to file."""
        if self.model is None:
            raise ValueError("No model to save")
        
        if path is None:
            path = str(Path(MODELS_ROOT) / 'lstm_energy_model.h5')
        
        self.model.save(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model from file."""
        from keras.models import load_model
        self.model = load_model(path)
        print(f"Model loaded from {path}")
