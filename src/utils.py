"""
Utility functions for Whisky & Spirits Forecasting Dashboard
Helper functions for data manipulation, formatting, and downloads.
"""

import pandas as pd
import numpy as np
from typing import Any


def download_dataframe(df: pd.DataFrame, name: str) -> bytes:
    """
    Convert dataframe to CSV bytes for download.
    
    Args:
        df: DataFrame to convert
        name: Name for the file (used in logging)
        
    Returns:
        Bytes object containing CSV data
    """
    return df.to_csv(index=False).encode('utf-8')


def safe_divide(numerator: float, denominator: float, default: float = 0) -> float:
    """
    Safely divide two numbers without raising ZeroDivisionError.
    
    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Value to return if denominator is 0 or NaN
        
    Returns:
        Result of division or default value
    """
    if denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a decimal value as a percentage string.
    
    Args:
        value: Decimal value (0-1 range)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to snake_case.
    Converts to lowercase, removes spaces and special characters.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dataframe with standardized column names
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    df.columns = df.columns.str.replace(' ', '_').str.replace('-', '_')
    return df


def round_dataframe(df: pd.DataFrame, decimals: int = 2) -> pd.DataFrame:
    """
    Round all numeric columns in a dataframe.
    
    Args:
        df: Input dataframe
        decimals: Number of decimal places
        
    Returns:
        Dataframe with rounded numeric columns
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(decimals)
    return df


def get_data_types_summary(df: pd.DataFrame) -> dict:
    """
    Get a summary of data types in a dataframe.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dictionary with data type counts
    """
    return {
        'numeric': len(df.select_dtypes(include=[np.number]).columns),
        'categorical': len(df.select_dtypes(include=['object']).columns),
        'datetime': len(df.select_dtypes(include=['datetime64']).columns),
        'boolean': len(df.select_dtypes(include=['bool']).columns),
    }
