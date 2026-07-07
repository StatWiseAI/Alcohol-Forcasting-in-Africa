"""
Data loader module for Whisky & Spirits Forecasting Dashboard
Handles CSV/Excel upload, sample data generation, and data import.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from src.utils import standardize_column_names


def load_uploaded_file(uploaded_file) -> Optional[pd.DataFrame]:
    """
    Load CSV or Excel file uploaded through Streamlit file uploader.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Loaded and standardized DataFrame or None if error
        
    Raises:
        ValueError: If file format is not CSV or Excel
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError("File must be CSV or Excel (.xls, .xlsx)")
        
        # Standardize column names
        df = standardize_column_names(df)
        
        return df
    
    except Exception as e:
        raise ValueError(f"Error reading file: {str(e)}")


def load_sample_data() -> pd.DataFrame:
    """
    Load sample data for demonstration purposes.
    First attempts to load from CSV file, falls back to generating synthetic data.
    
    Returns:
        Sample DataFrame with whisky and spirits sales data
    """
    # Try to load from file
    sample_file = Path(__file__).parent.parent / "sample_data" / "sample_whisky_spirits.csv"
    
    if sample_file.exists():
        try:
            df = pd.read_csv(sample_file)
        except Exception as e:
            print(f"Could not load sample file: {e}. Generating synthetic data.")
            df = generate_sample_data()
    else:
        # Generate synthetic data if file not found
        df = generate_sample_data()
    
    # Standardize columns
    df = standardize_column_names(df)
    
    return df


def generate_sample_data() -> pd.DataFrame:
    """
    Generate synthetic whisky and spirits sales data for demonstration.
    Creates 60 rows across 6 African countries over 10 years (2015-2024).
    
    Returns:
        Synthetic DataFrame with realistic patterns
    """
    np.random.seed(42)
    
    countries = ["Kenya", "Nigeria", "South Africa", "Morocco", "Ethiopia", "Egypt"]
    years = list(range(2015, 2025))
    
    data = []
    
    for country in countries:
        # Set country-specific baseline values
        base_whisky = np.random.uniform(100, 500)
        base_spirits = np.random.uniform(150, 600)
        base_inflation = np.random.uniform(5, 25)
        base_gdp = np.random.uniform(1000, 10000)
        
        for year in years:
            # Create realistic trends
            year_factor = 1 + (year - 2015) * np.random.uniform(0.02, 0.08)
            
            whisky_value = base_whisky * year_factor + np.random.normal(0, 20)
            spirits_value = base_spirits * year_factor + np.random.normal(0, 30)
            
            data.append({
                'Country': country,
                'Year': year,
                'dependent_Premium_Whisky': max(whisky_value, 50),  # Ensure positive
                'dependent_Premium_Spirits': max(spirits_value, 50),
                'Inflation': base_inflation + np.random.normal(0, 5),
                'compensasion_employee': np.random.uniform(1000, 5000),
                'current_balance': np.random.uniform(-20, 20),
                'age_dependency_ratio': np.random.uniform(40, 90),
                'GPD_Capita': base_gdp + np.random.normal(0, 2000),
                'total_labor_force': np.random.uniform(1e7, 1e8),
            })
    
    return pd.DataFrame(data)


def validate_file_format(filename: str) -> bool:
    """
    Check if filename has supported file format.
    
    Args:
        filename: Name of the file to check
        
    Returns:
        True if file format is supported, False otherwise
    """
    supported_formats = ('.csv', '.xls', '.xlsx')
    return any(filename.lower().endswith(fmt) for fmt in supported_formats)
