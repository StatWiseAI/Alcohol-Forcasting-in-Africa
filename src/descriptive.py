"""
Descriptive analytics module for Whisky & Spirits Forecasting Dashboard
Includes KPI computation, summary statistics, distributions, and clustering.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional


def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute key performance indicators from dataset.
    
    Args:
        df: Input dataframe
        
    Returns:
        Dictionary with KPI values
    """
    kpis = {}
    
    # Count unique countries
    if 'Country' in df.columns:
        kpis['n_countries'] = df['Country'].nunique()
    
    # Average sales metrics
    if 'dependent_Premium_Whisky' in df.columns:
        kpis['avg_premium_whisky'] = df['dependent_Premium_Whisky'].mean()
        kpis['total_premium_whisky'] = df['dependent_Premium_Whisky'].sum()
    
    if 'dependent_Premium_Spirits' in df.columns:
        kpis['avg_premium_spirits'] = df['dependent_Premium_Spirits'].mean()
        kpis['total_premium_spirits'] = df['dependent_Premium_Spirits'].sum()
    
    # Economic indicators
    if 'GPD_Capita' in df.columns:
        kpis['avg_gdp_capita'] = df['GPD_Capita'].mean()
    
    if 'Inflation' in df.columns:
        kpis['avg_inflation'] = df['Inflation'].mean()
    
    if 'compensasion_employee' in df.columns:
        kpis['avg_compensation'] = df['compensasion_employee'].mean()
    
    # Time metrics
    if 'Year' in df.columns:
        kpis['years_covered'] = df['Year'].max() - df['Year'].min() + 1
        kpis['start_year'] = df['Year'].min()
        kpis['end_year'] = df['Year'].max()
    
    return kpis


def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate comprehensive summary statistics for all numeric columns.
    
    Args:
        df: Input dataframe
        
    Returns:
        DataFrame with summary statistics
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    summary = numeric_df.describe().T
    summary['cv'] = (summary['std'] / summary['mean']).replace([np.inf, -np.inf], 0)
    summary = summary.round(2)
    
    return summary[['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'cv']]


def analyze_distributions(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """
    Analyze distribution characteristics of a numeric column.
    
    Args:
        df: Input dataframe
        column: Column name to analyze
        
    Returns:
        Dictionary with distribution statistics
    """
    data = df[column].dropna()
    
    return {
        'count': len(data),
        'mean': data.mean(),
        'median': data.median(),
        'mode': data.mode().values[0] if len(data.mode()) > 0 else np.nan,
        'std': data.std(),
        'variance': data.var(),
        'skewness': data.skew(),
        'kurtosis': data.kurtosis(),
        'min': data.min(),
        'max': data.max(),
        'iqr': data.quantile(0.75) - data.quantile(0.25),
        'q1': data.quantile(0.25),
        'q3': data.quantile(0.75),
    }


def analyze_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate correlation matrix for all numeric columns.
    
    Args:
        df: Input dataframe
        
    Returns:
        Correlation matrix DataFrame
    """
    numeric_df = df.select_dtypes(include=[np.number])
    return numeric_df.corr()


def find_top_correlations(
    df: pd.DataFrame,
    column: str,
    n_top: int = 5
) -> pd.DataFrame:
    """
    Find top N correlations for a specific column.
    
    Args:
        df: Input dataframe
        column: Column to find correlations for
        n_top: Number of top correlations to return
        
    Returns:
        DataFrame with top correlations sorted
    """
    corr_matrix = analyze_correlations(df)
    
    if column not in corr_matrix.columns:
        return pd.DataFrame()
    
    correlations = corr_matrix[column].sort_values(ascending=False)
    # Exclude self-correlation
    correlations = correlations[correlations.index != column]
    
    return correlations.head(n_top).reset_index().rename(
        columns={'index': 'Variable', column: 'Correlation'}
    )


def perform_clustering(df: pd.DataFrame, n_clusters: int = 3) -> Optional[pd.DataFrame]:
    """
    Perform K-means clustering on aggregated country data.
    Groups countries by economic indicators.
    
    Args:
        df: Input dataframe
        n_clusters: Number of clusters to create
        
    Returns:
        DataFrame with cluster assignments or None if clustering fails
    """
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        
        # Check for Country column
        if 'Country' not in df.columns:
            return None
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            return None
        
        # Aggregate by country
        cluster_data = df.groupby('Country')[numeric_cols].mean().reset_index()
        
        if len(cluster_data) < n_clusters:
            n_clusters = max(2, len(cluster_data) - 1)
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(cluster_data[numeric_cols])
        
        # Apply K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_data['Cluster'] = kmeans.fit_predict(scaled_features)
        cluster_data['Cluster'] = 'Cluster ' + cluster_data['Cluster'].astype(str)
        
        return cluster_data
    
    except Exception as e:
        print(f"Clustering error: {e}")
        return None


def get_country_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics aggregated by country.
    
    Args:
        df: Input dataframe
        
    Returns:
        DataFrame with country-level statistics
    """
    if 'Country' not in df.columns:
        return pd.DataFrame()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    country_stats = df.groupby('Country')[numeric_cols].agg(['mean', 'std', 'min', 'max']).round(2)
    
    return country_stats


def get_time_series_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get summary statistics aggregated by year.
    
    Args:
        df: Input dataframe
        
    Returns:
        DataFrame with time-aggregated statistics
    """
    if 'Year' not in df.columns:
        return pd.DataFrame()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != 'Year']
    
    time_stats = df.groupby('Year')[numeric_cols].mean().round(2)
    
    return time_stats
