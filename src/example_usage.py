"""
Example usage of TraditionalML and LSTMModel classes.
"""

from pathlib import Path
from Globals import DATASETS_ROOT, MODELS_ROOT, PLOTS_ROOT
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from DataPreprocessing import DataPreprocessing
from TraditionalML import train_all_models, RandomForestModel, XGBoostModel
from LSTMModel import LSTMEnergyModel


def example_traditional_ml(df):
    """Example: Train traditional ML models."""
    
    print("="*80)
    print("TRADITIONAL ML MODELS")
    print("="*80)
    
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Select features for modeling
    feature_cols = [
        'Acorn_grouped',
        'stdorToU', 
        'is_holiday',
        'temperatureHigh',
        'season'
    ]
    
    target_col = 'avg_kwh_per_household_per_day'
    
    # Check which columns exist
    available_features = [col for col in feature_cols if col in df.columns]
    print(f"\nAvailable features: {available_features}")
    
    # Prepare modeling dataframe
    df_model = df[available_features + [target_col]].copy()
    df_model = df_model.dropna()
    
    # Encode categorical columns
    le_dict = {}
    if 'season' in df_model.columns:
        le = LabelEncoder()
        df_model['season'] = le.fit_transform(df_model['season'])
        le_dict['season'] = le
    
    # Convert boolean to int
    if 'is_holiday' in df_model.columns:
        df_model['is_holiday'] = df_model['is_holiday'].astype(int)
    
    # Prepare features and target
    X = df_model[available_features].values
    y = df_model[target_col].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train all models
    results = train_all_models(X_train, X_test, y_train, y_test, scale_features=True)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"\n{model_name}:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  R²:   {metrics['r2']:.4f}")
    
    print(f"\nResults saved to model_results.csv")


def example_lstm(df):
    """Example: Train LSTM model on normalized day-level data."""
    
    print("\n" + "="*80)
    print("LSTM MODEL")
    print("="*80)
    
    # Sort by day for time series (normalized day-level data)
    df = df.sort_values('day').reset_index(drop=True)
    
    print(f"\nTime series data: {len(df)} days")
    
    # Select features
    feature_cols = [
        'month',
        'day_of_week',
        'temperatureHigh',
        'temperatureLow',
        'humidity',
        'windSpeed',
        'is_holiday',
        'num_active_households'
    ]
    
    # Check available columns
    available_features = [col for col in feature_cols if col in df.columns]
    print(f"Available features: {available_features}")
    
    target_col = 'avg_kwh_per_household_per_day'
    
    # Drop missing values
    df_clean = df[available_features + [target_col]].dropna()
    
    # Convert boolean to int
    if 'is_holiday' in df_clean.columns:
        df_clean['is_holiday'] = df_clean['is_holiday'].astype(int)
    
    print(f"Data shape after cleaning: {df_clean.shape}")
    
    # Initialize LSTM model
    lstm_model = LSTMEnergyModel(
        sequence_length=48,  # Use 48 days of history
        lstm_units=75,
        num_layers=3,
        dropout_rate=0.2,
        random_seed=42
    )
    
    # Prepare data
    # Set use_exogenous=False for univariate (only past energy consumption)
    # Set use_exogenous=True for multivariate (include weather, holidays, etc.)
    USE_EXOGENOUS = False  # Toggle this to switch between univariate and multivariate
    
    X_train, X_test, y_train, y_test = lstm_model.prepare_data(
        df_clean,
        feature_cols=available_features,
        target_col=target_col,
        train_split=0.7,
        use_exogenous=USE_EXOGENOUS
    )
    
    # Train model
    lstm_model.train(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        epochs=50,
        batch_size=32,
        model_path=str(Path(MODELS_ROOT) / 'lstm_energy_model.h5')
    )
    
    # Evaluate
    metrics, y_pred, y_test_actual = lstm_model.evaluate(X_test, y_test)
    
    # Plot results
    lstm_model.plot_history(save_path=str(Path(PLOTS_ROOT) / 'lstm_training_history.png'))
    lstm_model.plot_predictions(y_test_actual, y_pred, save_path=str(Path(PLOTS_ROOT) / 'lstm_predictions.png'))
    
    # Plot validation predictions vs ground truth
    lstm_model.plot_validation_predictions(X_test, y_test, 
                                          save_path=str(Path(PLOTS_ROOT) / 'lstm_validation_comparison.png'))
    
    print("\n" + "="*80)
    print("LSTM Model saved successfully")
    print("="*80)


def main():
    """Run examples."""
    
    # Load preprocessed data once
    print("Loading preprocessed data...")
    preprocessor = DataPreprocessing()
    df = preprocessor.preprocess_data()
    
    # Example 1: Traditional ML models
    try:
        example_traditional_ml()
    except Exception as e:
        print(f"\nError in Traditional ML example: {e}")
        import traceback
        traceback.print_exc()
    
    # Example 2: LSTM model
    try:
        example_lstm(df)
    except Exception as e:
        print(f"\nError in LSTM example: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
