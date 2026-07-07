"""
Data validation module for Whisky & Spirits Forecasting Dashboard
Validates dataset structure, types, and quality.
"""

import pandas as pd
from typing import Dict, List, Any


REQUIRED_COLUMNS = [
    'Country',
    'Year',
]

EXPECTED_COLUMNS = [
    'dependent_Premium_Whisky',
    'dependent_Premium_Spirits',
    'Inflation',
    'compensasion_employee',
    'current_balance',
    'age_dependency_ratio',
    'GPD_Capita',
    'total_labor_force',
]

MIN_ROWS = 10


def validate_dataset(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate dataset structure and content comprehensively.
    Checks for required columns, data types, and data quality.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Dictionary with validation result, issues, and metadata
    """
    issues = []
    
    # Check required columns exist
    missing_required = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_required:
        issues.append(f"Missing required columns: {', '.join(missing_required)}")
    
    # Check for at least one dependent variable
    dep_vars = [col for col in df.columns if col in EXPECTED_COLUMNS[:2]]
    if not dep_vars:
        issues.append("Must have at least 'dependent_Premium_Whisky' or 'dependent_Premium_Spirits'")
    
    # Check Year column is numeric
    if 'Year' in df.columns:
        try:
            pd.to_numeric(df['Year'])
        except (ValueError, TypeError):
            issues.append("Year column must be numeric (integer)")
    
    # Check Country column is categorical
    if 'Country' in df.columns:
        if df['Country'].dtype == 'object':
            pass  # OK
        else:
            issues.append("Country column should be text/categorical")
    
    # Check minimum data volume
    if len(df) < MIN_ROWS:
        issues.append(f"Dataset has fewer than {MIN_ROWS} rows (minimum recommended)")
    
    # Check for duplicate rows
    duplicates = df.duplicated(subset=['Country', 'Year']).sum()
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate country-year combinations")
    
    # Get metadata
    validation_result = {
        'is_valid': len(issues) == 0,
        'issues': issues,
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'date_range': f"{int(df['Year'].min())} - {int(df['Year'].max())}" if 'Year' in df.columns else "N/A",
        'countries': df['Country'].nunique() if 'Country' in df.columns else 0,
        'missing_pct': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
    }
    
    return validation_result


def get_validation_report(df: pd.DataFrame) -> str:
    """
    Generate a human-readable validation report.
    
    Args:
        df: DataFrame to validate
        
    Returns:
        Formatted validation report string
    """
    report = validate_dataset(df)
    
    text = "=== DATA VALIDATION REPORT ===\n"
    text += f"Status: {'✅ PASSED' if report['is_valid'] else '❌ FAILED'}\n"
    text += f"Rows: {report['n_rows']}\n"
    text += f"Columns: {report['n_cols']}\n"
    text += f"Date Range: {report['date_range']}\n"
    text += f"Countries: {report['countries']}\n"
    text += f"Missing Data: {report['missing_pct']:.2f}%\n"
    
    if report['issues']:
        text += "\nIssues Found:\n"
        for issue in report['issues']:
            text += f"  • {issue}\n"
    else:
        text += "\nNo validation issues found!\n"
    
    return text


def check_column_exists(df: pd.DataFrame, column_name: str) -> bool:
    """
    Check if a column exists in the dataframe.
    
    Args:
        df: DataFrame to check
        column_name: Name of column to look for
        
    Returns:
        True if column exists, False otherwise
    """
    return column_name in df.columns


def check_columns_exist(df: pd.DataFrame, column_names: List[str]) -> List[str]:
    """
    Check which columns from a list exist in the dataframe.
    
    Args:
        df: DataFrame to check
        column_names: List of column names to check
        
    Returns:
        List of columns that exist
    """
    return [col for col in column_names if col in df.columns]


def get_missing_columns(df: pd.DataFrame, required_cols: List[str]) -> List[str]:
    """
    Get list of required columns that are missing from dataframe.
    
    Args:
        df: DataFrame to check
        required_cols: List of required column names
        
    Returns:
        List of missing column names
    """
    return [col for col in required_cols if col not in df.columns]
