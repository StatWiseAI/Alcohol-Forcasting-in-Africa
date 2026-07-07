"""
Data preprocessing module for Whisky & Spirits Forecasting Dashboard
Handles missing values, outliers, and data transformations.
"""

import pandas as pd
import numpy as np
from typing import Optional


def preprocess_data(
    df: pd.DataFrame,
    missing_strategy: str = "mean",
    remove_outliers: bool = False
) -> pd.DataFrame:
    """
    Apply full preprocessing pipeline to dataset.
    
    Args:
        df: Input dataframe
        missing_strategy: Strategy for handling missing values
                         Options: 'mean', 'median', 'forward_fill', 'drop_rows'
        remove_outliers: Whether to remove statistical outliers
        
    Returns:
        Preprocessed dataframe
    """
    df = df.copy()
    
    # Handle missing values
    df = handle_missing_values(df, strategy=missing_strategy)
    
    # Remove outliers if requested
    if remove_outliers:
        df = remove_statistical_outliers(df)
    
    return df


def handle_missing_values(df: pd.DataFrame, strategy: str = "mean") -> pd.DataFrame:
    """
    Handle missing values in numeric columns using specified strategy.
    
    Args:
        df: Input dataframe
        strategy: Imputation strategy
                 - 'mean': Fill with column mean
                 - 'median': Fill with column median
                 - 'forward_fill': Forward fill then backward fill
                 - 'drop_rows': Remove rows with missing values
        
    Returns:
        Dataframe with missing values handled
    """
    df = df.copy()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            if strategy == "mean":
                fill_value = df[col].mean()
                df[col].fillna(fill_value, inplace=True)
                
            elif strategy == "median":
                fill_value = df[col].median()
                df[col].fillna(fill_value, inplace=True)
                
            elif strategy == "forward_fill":
                df[col].fillna(method='ffill', inplace=True)
                df[col].fillna(method='bfill', inplace=True)
                
            elif strategy == "drop_rows":
                df = df.dropna(subset=[col])
    
    # For categorical columns, fill with mode or 'Unknown'
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_value = df[col].mode()
            if len(mode_value) > 0:
                df[col].fillna(mode_value[0], inplace=True)
            else:
                df[col].fillna('Unknown', inplace=True)
    
    return df


def remove_statistical_outliers(
    df: pd.DataFrame,
    iqr_multiplier: float = 1.5
) -> pd.DataFrame:
    """
    Remove rows with statistical outliers using Interquartile Range (IQR) method.
    
    Args:
        df: Input dataframe
        iqr_multiplier: Multiplier for IQR (standard is 1.5)
                       Higher values are more lenient
        
    Returns:
        Dataframe with outlier rows removed
    """
    df = df.copy()
    initial_rows = len(df)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR
        
        # Remove rows outside bounds
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    rows_removed = initial_rows - len(df)
    if rows_removed > 0:
        print(f"Removed {rows_removed} rows with outliers")
    
    return df


def standardize_features(
    df: pd.DataFrame,
    numeric_cols: Optional[list] = None
) -> tuple:
    """
    Standardize (z-score normalize) numeric features.
    Useful for machine learning models.
    
    Args:
        df: Input dataframe
        numeric_cols: List of numeric columns to standardize.
                     If None, standardizes all numeric columns.
        
    Returns:
        Tuple of (standardized_dataframe, scaler_dict)
        where scaler_dict contains means and stds for inverse transform
    """
    df = df.copy()
    
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    scaler_dict = {}
    
    for col in numeric_cols:
        if col in df.columns:
            mean_val = df[col].mean()
            std_val = df[col].std()
            
            if std_val > 0:
                df[col] = (df[col] - mean_val) / std_val
                scaler_dict[col] = {'mean': mean_val, 'std': std_val}
    
    return df, scaler_dict


def scale_to_range(
    df: pd.DataFrame,
    min_val: float = 0,
    max_val: float = 1,
    numeric_cols: Optional[list] = None
) -> tuple:
    """
    Scale numeric features to a specified range (e.g., 0-1).
    
    Args:
        df: Input dataframe
        min_val: Minimum value of range
        max_val: Maximum value of range
        numeric_cols: List of columns to scale. If None, scales all numeric columns.
        
    Returns:
        Tuple of (scaled_dataframe, scaler_dict)
    """
    df = df.copy()
    
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    scaler_dict = {}
    
    for col in numeric_cols:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            col_range = col_max - col_min
            
            if col_range > 0:
                df[col] = (df[col] - col_min) / col_range * (max_val - min_val) + min_val
                scaler_dict[col] = {'min': col_min, 'max': col_max}
    
    return df, scaler_dict


def get_preprocessing_summary(original_df: pd.DataFrame, processed_df: pd.DataFrame) -> dict:
    """
    Generate summary of preprocessing changes.
    
    Args:
        original_df: Original dataframe before preprocessing
        processed_df: Processed dataframe after preprocessing
        
    Returns:
        Dictionary with preprocessing statistics
    """
    return {
        'rows_removed': len(original_df) - len(processed_df),
        'rows_retained': len(processed_df),
        'original_missing': original_df.isnull().sum().sum(),
        'processed_missing': processed_df.isnull().sum().sum(),
        'missing_pct_original': (original_df.isnull().sum().sum() / 
                                (original_df.shape[0] * original_df.shape[1]) * 100),
        'missing_pct_processed': (processed_df.isnull().sum().sum() / 
                                 (processed_df.shape[0] * processed_df.shape[1]) * 100),
    }
