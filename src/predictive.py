"""
Predictive modeling module for Whisky & Spirits Forecasting Dashboard
Includes model training, forecasting, and evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Optional, Dict, Any


FORECAST_COVARIATES = [
    'Inflation',
    'compensasion_employee',
    'current_balance',
    'age_dependency_ratio',
    'GPD_Capita',
    'total_labor_force'
]


def prepare_forecast_data(
    df: pd.DataFrame,
    target_var: str,
    max_lags: int = 2
) -> Optional[pd.DataFrame]:
    """
    Prepare data for forecasting with lag features.
    Creates autoregressive features (lagged target variable).
    
    Args:
        df: Input dataframe
        target_var: Target variable column name
        max_lags: Number of lag features to create (1, 2, etc.)
        
    Returns:
        Prepared dataframe with lag features or None if error
    """
    if target_var not in df.columns:
        return None
    
    df = df.copy()
    df = df.sort_values(['Country', 'Year']).reset_index(drop=True)
    
    # Create lag features
    for lag_num in range(1, max_lags + 1):
        lag_col_name = f'lag_{lag_num}'
        df[lag_col_name] = df.groupby('Country')[target_var].shift(lag_num)
    
    # Drop rows with NaN from lags (initial observations)
    df = df.dropna()
    
    return df if len(df) > 0 else None


def train_random_forest(
    df: pd.DataFrame,
    target_var: str,
    test_size: float = 0.2,
    n_estimators: int = 100,
    max_depth: int = 10
) -> Optional[Dict[str, Any]]:
    """
    Train Random Forest model for forecasting.
    
    Args:
        df: Input dataframe with lag features
        target_var: Target variable column name
        test_size: Proportion of data for testing
        n_estimators: Number of trees in forest
        max_depth: Maximum depth of trees
        
    Returns:
        Dictionary with model, metrics, and forecasts or None if error
    """
    try:
        # Select features
        feature_cols = ['lag_1', 'lag_2', 'Year']
        feature_cols.extend([col for col in FORECAST_COVARIATES if col in df.columns])
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[feature_cols].copy()
        y = df[target_var].copy()
        
        # Handle any remaining NaN
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 10:
            return None
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train Random Forest
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_mae = mean_absolute_error(y_test, test_pred)
        
        # Feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Generate forecasts
        forecasts = generate_forecasts(df, model, target_var, feature_cols)
        
        return {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'feature_importance': importance_df,
            'forecasts': forecasts,
            'feature_cols': feature_cols,
            'model_type': 'Random Forest'
        }
    
    except Exception as e:
        print(f"Random Forest training error: {e}")
        return None


def train_xgboost(
    df: pd.DataFrame,
    target_var: str,
    test_size: float = 0.2,
    n_estimators: int = 100,
    learning_rate: float = 0.05
) -> Optional[Dict[str, Any]]:
    """
    Train XGBoost model for forecasting.
    
    Args:
        df: Input dataframe with lag features
        target_var: Target variable column name
        test_size: Proportion of data for testing
        n_estimators: Number of boosting rounds
        learning_rate: Learning rate for boosting
        
    Returns:
        Dictionary with model, metrics, and forecasts or None if error
    """
    try:
        import xgboost as xgb
    except ImportError:
        print("XGBoost not installed")
        return None
    
    try:
        # Select features
        feature_cols = ['lag_1', 'lag_2', 'Year']
        feature_cols.extend([col for col in FORECAST_COVARIATES if col in df.columns])
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        X = df[feature_cols].copy()
        y = df[target_var].copy()
        
        # Handle NaN
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < 10:
            return None
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Train XGBoost
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_mae = mean_absolute_error(y_test, test_pred)
        
        # Feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Generate forecasts
        forecasts = generate_forecasts(df, model, target_var, feature_cols)
        
        return {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'feature_importance': importance_df,
            'forecasts': forecasts,
            'feature_cols': feature_cols,
            'model_type': 'XGBoost'
        }
    
    except Exception as e:
        print(f"XGBoost training error: {e}")
        return None


def generate_forecasts(
    df: pd.DataFrame,
    model,
    target_var: str,
    feature_cols: list,
    horizon: int = 10
) -> pd.DataFrame:
    """
    Generate multi-step forecasts for each country.
    
    Args:
        df: Input dataframe with historical data
        model: Trained model object
        target_var: Target variable column name
        feature_cols: List of feature column names
        horizon: Number of years to forecast
        
    Returns:
        DataFrame with forecasts
    """
    forecasts = []
    
    for country in df['Country'].unique():
        country_data = df[df['Country'] == country].sort_values('Year')
        
        if len(country_data) < 2:
            continue
        
        last_year = country_data['Year'].max()
        last_lag1 = country_data[target_var].iloc[-1]
        last_lag2 = country_data[target_var].iloc[-2] if len(country_data) >= 2 else last_lag1
        
        # Project forward
        for step in range(1, horizon + 1):
            future_year = last_year + step
            
            # Create feature row
            feature_row = {
                'lag_1': last_lag1,
                'lag_2': last_lag2,
                'Year': future_year
            }
            
            # Add covariates (use last known value)
            for col in FORECAST_COVARIATES:
                if col in country_data.columns:
                    feature_row[col] = country_data[col].iloc[-1]
            
            # Ensure all features present
            for col in feature_cols:
                if col not in feature_row:
                    feature_row[col] = 0
            
            # Predict
            feature_array = np.array([[feature_row.get(col, 0) for col in feature_cols]])
            prediction = model.predict(feature_array)[0]
            
            forecasts.append({
                'Country': country,
                'Year': int(future_year),
                target_var: max(prediction, 0),  # Ensure non-negative
                'Type': 'Forecast'
            })
            
            # Update lag for next iteration
            last_lag2 = last_lag1
            last_lag1 = prediction
    
    return pd.DataFrame(forecasts)


def evaluate_model(y_true, y_pred) -> Dict[str, float]:
    """
    Evaluate model performance with multiple metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with performance metrics
    """
    return {
        'r2': r2_score(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
    }


def cross_validate_model(
    df: pd.DataFrame,
    target_var: str,
    model_class,
    n_splits: int = 5
) -> Dict[str, Any]:
    """
    Perform k-fold cross-validation on model.
    
    Args:
        df: Input dataframe
        target_var: Target variable
        model_class: Model class to use
        n_splits: Number of folds
        
    Returns:
        Dictionary with cross-validation results
    """
    from sklearn.model_selection import cross_val_score
    
    feature_cols = ['lag_1', 'lag_2', 'Year']
    feature_cols.extend([col for col in FORECAST_COVARIATES if col in df.columns])
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    X = df[feature_cols].copy()
    y = df[target_var].copy()
    
    valid_idx = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_idx]
    y = y[valid_idx]
    
    if len(X) < n_splits:
        return {"Error": "Not enough data for cross-validation"}
    
    model = model_class()
    
    cv_scores = cross_val_score(model, X, y, cv=n_splits, scoring='r2')
    
    return {
        'Mean CV R²': cv_scores.mean(),
        'Std CV R²': cv_scores.std(),
        'CV Scores': cv_scores,
        'N Splits': n_splits
    }
