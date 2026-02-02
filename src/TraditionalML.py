import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from abc import ABC, abstractmethod
from Globals import RESULTS_ROOT


# Data schema version for tracking
DATA_SCHEMA_VERSION = "1.0"

# Create directories if they don't exist
Path(RESULTS_ROOT).mkdir(parents=True, exist_ok=True)


def log_model_results(model_type: str, metrics: dict, results_file: str = None):
    """
    Log model results to CSV file. Overwrites existing row if model_type and schema version match.
    
    Args:
        model_type: Name of the model type
        metrics: Dictionary containing metrics (mse, rmse, mae, r2)
        results_file: Path to results CSV file (defaults to build/results/model_results.csv)
    """
    if results_file is None:
        results_file = Path(RESULTS_ROOT) / "model_results.csv"
    else:
        results_file = Path(results_file)
    
    try:
        # Check if results file exists
        if os.path.exists(results_file):
            df = pd.read_csv(results_file)
        else:
            df = pd.DataFrame(columns=['model_type', 'data_schema_version', 'mse', 'rmse', 'mae', 'r2'])
        
        # Remove any existing rows with the same model_type and schema version
        df['data_schema_version'] = df['data_schema_version'].astype(str)
        mask = (df['model_type'] == model_type) & (df['data_schema_version'] == DATA_SCHEMA_VERSION)
        df = df[~mask]
        
        # Add new row
        new_row = pd.DataFrame({
            'model_type': [model_type],
            'data_schema_version': [DATA_SCHEMA_VERSION],
            'mse': [round(metrics['mse'], 4)],
            'rmse': [round(metrics['rmse'], 4)],
            'mae': [round(metrics['mae'], 4)],
            'r2': [round(metrics['r2'], 4)]
        })
        
        df = pd.concat([df, new_row], ignore_index=True)
        
        # Save to CSV
        df.to_csv(results_file, index=False)
        print(f"Results logged for {model_type}")
        
    except Exception as e:
        print(f"Error logging results for {model_type}: {e}")
        import traceback
        traceback.print_exc()


class BaseRegressor(ABC):
    """Abstract base class for regression models."""
    
    def __init__(self):
        """Initialize regressor."""
        self.model = None
        self.scaler = None
        
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate regression metrics."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def train(self, X_train, X_test, y_train, y_test, model_name: str, scale_features: bool = True):
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training targets
            y_test: Test targets
            model_name: Name of the model for logging
            scale_features: Whether to scale features using StandardScaler
        """
        try:
            print(f"\nTraining {model_name}...")
            print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
            
            # Scale features if requested
            if scale_features:
                self.scaler = StandardScaler()
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
            else:
                X_train_scaled = X_train
                X_test_scaled = X_test
            
            # Train the model
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_test, y_pred)
            
            print(f"{model_name} Results:")
            print(f"  RMSE: {metrics['rmse']:.4f}")
            print(f"  MAE:  {metrics['mae']:.4f}")
            print(f"  R²:   {metrics['r2']:.4f}")
            
            # Log results
            log_model_results(model_name, metrics)
            
            return metrics
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict(self, X):
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)


class RandomForestModel(BaseRegressor):
    """Random Forest regressor."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = None, random_state: int = 42):
        super().__init__()
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )


class LinearRegressionModel(BaseRegressor):
    """Linear Regression model."""
    
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()


class RidgeModel(BaseRegressor):
    """Ridge Regression model."""
    
    def __init__(self, alpha: float = 1.0, random_state: int = 42):
        super().__init__()
        self.model = Ridge(alpha=alpha, random_state=random_state)


class LassoModel(BaseRegressor):
    """Lasso Regression model."""
    
    def __init__(self, alpha: float = 1.0, random_state: int = 42):
        super().__init__()
        self.model = Lasso(alpha=alpha, random_state=random_state)


class SVRModel(BaseRegressor):
    """Support Vector Regressor."""
    
    def __init__(self, kernel: str = 'rbf', C: float = 1.0):
        super().__init__()
        self.model = SVR(kernel=kernel, C=C)


class KNNModel(BaseRegressor):
    """K-Nearest Neighbors regressor."""
    
    def __init__(self, n_neighbors: int = 5):
        super().__init__()
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, n_jobs=-1)


class GradientBoostingModel(BaseRegressor):
    """Gradient Boosting regressor."""
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, 
                 max_depth: int = 3, random_state: int = 42):
        super().__init__()
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )


class XGBoostModel(BaseRegressor):
    """XGBoost regressor."""
    
    def __init__(self, n_estimators: int = 100, learning_rate: float = 0.1, 
                 max_depth: int = 3, random_state: int = 42):
        super().__init__()
        try:
            import xgboost as xgb
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=learning_rate,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        except ImportError:
            print("XGBoost is not available. Install with: pip install xgboost")
            self.model = None


class NeuralNetworkModel(BaseRegressor):
    """Neural Network (MLP) regressor."""
    
    def __init__(self, hidden_layer_sizes: tuple = (100, 50), 
                 random_state: int = 42, max_iter: int = 500):
        super().__init__()
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            random_state=random_state,
            max_iter=max_iter,
            early_stopping=True
        )


def train_all_models(X_train, X_test, y_train, y_test, scale_features: bool = True):
    """
    Train all available regression models and log their results.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training targets
        y_test: Test targets
        scale_features: Whether to scale features
        
    Returns:
        Dictionary of trained models
    """
    models = {
        'LinearRegression': LinearRegressionModel(),
        'Ridge': RidgeModel(),
        'Lasso': LassoModel(),
        'RandomForest': RandomForestModel(n_estimators=100, max_depth=20),
        'GradientBoosting': GradientBoostingModel(),
        'KNN': KNNModel(n_neighbors=5),
        'NeuralNetwork': NeuralNetworkModel(),
        'XGBoost': XGBoostModel()
    }
    
    results = {}
    
    for name, model in models.items():
        if model.model is not None:
            metrics = model.train(X_train, X_test, y_train, y_test, name, scale_features)
            if metrics is not None:
                results[name] = {
                    'model': model,
                    'metrics': metrics
                }
    
    return results
