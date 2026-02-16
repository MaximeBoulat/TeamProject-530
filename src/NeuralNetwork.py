import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from Globals import MODELS_ROOT, PLOTS_ROOT

Path(MODELS_ROOT).mkdir(parents=True, exist_ok=True)
Path(PLOTS_ROOT).mkdir(parents=True, exist_ok=True)


class NeuralNetwork:

    def __init__(self, hidden_layer_sizes=(100, 50), dropout_rate=0.3, random_seed=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_rate = dropout_rate
        self.random_seed = random_seed

        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)

        self.model = None
        self.scaler = None
        self.le_acorn = None
        self.le_season = None
        self.history = None
        self.class_labels = ["Low", "Normal", "High"]

    def prepare_data(self, df, feature_cols, target_col, test_size=0.2):
        """
        Prepare preprocessed data for Neural Network training.
        Handles encoding, scaling, and train/test split.

        Args:
            df: Preprocessed DataFrame from DataPreprocessing
            feature_cols: List of feature column names
            target_col: Target column name
            test_size: Fraction of data for testing

        Returns:
            X_train, X_test, y_train, y_test: Scaled and split data
        """
        ml_model_data = df[feature_cols + [target_col]].copy()

        # Encode categoricals
        self.le_acorn = LabelEncoder()
        self.le_season = LabelEncoder()

        ml_model_data["Acorn_grouped"] = self.le_acorn.fit_transform(
            ml_model_data["Acorn_grouped"]
        )
        ml_model_data["season"] = self.le_season.fit_transform(
            ml_model_data["season"]
        )
        ml_model_data["is_holiday"] = ml_model_data["is_holiday"].astype(int)

        X = ml_model_data[feature_cols].values
        y = ml_model_data[target_col].values

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_seed
        )

        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"\nNeural Network Data Preparation:")
        print(f"  Training samples: {X_train_scaled.shape[0]}")
        print(f"  Test samples: {X_test_scaled.shape[0]}")
        print(f"  Features: {len(feature_cols)}")
        print(f"  Target: {target_col}")

        return X_train_scaled, X_test_scaled, y_train, y_test

    def build_model(self, nb_features, nb_out=1):
        """
        Build Neural Network model architecture.

        Args:
            nb_features: Number of input features
            nb_out: Number of output values
        """
        layer_list = [layers.InputLayer(input_shape=(nb_features,))]
        for units in self.hidden_layer_sizes:
            layer_list.append(layers.Dense(units, activation='relu'))
            layer_list.append(layers.Dropout(self.dropout_rate))
        layer_list.append(layers.Dense(nb_out, activation='softmax'))

        self.model = models.Sequential(layer_list)

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        print("\nModel Architecture:")
        self.model.summary()

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=50, batch_size=64,
              validation_split=0.1, model_path=None):
        """
        Train the Neural Network model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of training data for validation
            model_path: Path to save the best model

        Returns:
            Training history
        """
        if self.model is None:
            nb_features = X_train.shape[1]
            nb_classes = len(self.class_labels)
            self.build_model(nb_features, nb_out=nb_classes)

        if model_path is None:
            model_path = str(Path(MODELS_ROOT) / "nn_model.h5")

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                verbose=1,
                min_lr=1e-7
            )
        ]

        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            validation_split = 0.0
        else:
            validation_data = None

        print("\nTraining Neural Network...")
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
            X_test: Test features
            y_test: Test targets (integer class labels)

        Returns:
            Dictionary of metrics, predicted class labels
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)

        accuracy = accuracy_score(y_test, y_pred)

        metrics = {'accuracy': accuracy}

        print("\nNeural Network Evaluation:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(
            y_test, y_pred, target_names=self.class_labels
        ))

        return metrics, y_pred

    def plot_history(self, save_path=None):
        """Plot training history."""
        if self.history is None:
            print("No training history available")
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(self.history.history['loss'], label='Training Loss',
                     linewidth=2)
        axes[0].plot(self.history.history['val_loss'], label='Validation Loss',
                     linewidth=2)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Model Loss Over Epochs')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(self.history.history['accuracy'], label='Training Accuracy',
                     linewidth=2)
        axes[1].plot(self.history.history['val_accuracy'], label='Validation Accuracy',
                     linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Accuracy Over Epochs')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def plot_predictions(self, y_test, y_pred, save_path=None):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=self.class_labels,
               yticklabels=self.class_labels,
               xlabel='Predicted',
               ylabel='Actual',
               title='Confusion Matrix')

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        # Add text annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")

        plt.show()

    def save_model(self, path=None):
        """Save trained model to file."""
        if self.model is None:
            raise ValueError("No model to save")

        if path is None:
            path = str(Path(MODELS_ROOT) / 'nn_energy_model.h5')

        self.model.save(path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """Load trained model from file."""
        self.model = models.load_model(path)
        print(f"Model loaded from {path}")
