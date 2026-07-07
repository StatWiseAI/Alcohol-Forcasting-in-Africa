"""
Visualizations module for Whisky & Spirits Forecasting Dashboard
Creates interactive Plotly charts for data exploration and reporting.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


def plot_correlation_matrix(corr_matrix: pd.DataFrame):
    """
    Create interactive correlation heatmap.
    
    Args:
        corr_matrix: Correlation matrix DataFrame
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Correlation")
    ))
    
    fig.update_layout(
        title="Correlation Matrix Heatmap",
        xaxis_title="",
        yaxis_title="",
        width=700,
        height=700,
        template="plotly_white",
        hovermode="closest"
    )
    
    return fig


def plot_time_series(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None):
    """
    Create time series line plot with markers.
    
    Args:
        df: Input dataframe
        x_col: X-axis column (typically 'Year')
        y_col: Y-axis column (metric to plot)
        color_col: Optional column for color grouping
        
    Returns:
        Plotly figure object
    """
    if color_col and color_col in df.columns:
        fig = px.line(df, x=x_col, y=y_col, color=color_col, 
                     title=f"{y_col} Over Time", markers=True)
    else:
        fig = px.line(df, x=x_col, y=y_col, title=f"{y_col} Over Time", markers=True)
    
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        yaxis_title=y_col,
        xaxis_title=x_col
    )
    return fig


def plot_3d_scatter(df: pd.DataFrame, x_var: str, y_var: str, z_var: str = None):
    """
    Create interactive 3D scatter plot.
    
    Args:
        df: Input dataframe
        x_var: X-axis variable
        y_var: Y-axis variable
        z_var: Z-axis variable (optional)
        
    Returns:
        Plotly figure object
    """
    if z_var is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if col not in [x_var, y_var]:
                z_var = col
                break
    
    fig = px.scatter_3d(
        df,
        x=x_var,
        y=y_var,
        z=z_var,
        color='Country' if 'Country' in df.columns else None,
        title=f"3D: {x_var} vs {y_var} vs {z_var}",
        hover_data={'Year': True} if 'Year' in df.columns else {}
    )
    
    fig.update_layout(template="plotly_white")
    return fig


def plot_distribution_ridges(df: pd.DataFrame, column: str):
    """
    Create distribution histogram, optionally split by year.
    
    Args:
        df: Input dataframe
        column: Column to plot distribution for
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    if 'Year' in df.columns and df['Year'].nunique() > 1:
        for year in sorted(df['Year'].unique()):
            year_data = df[df['Year'] == year][column].dropna()
            fig.add_trace(go.Histogram(
                x=year_data,
                name=f"Year {int(year)}",
                opacity=0.7
            ))
    else:
        fig.add_trace(go.Histogram(
            x=df[column].dropna(),
            nbinsx=30,
            name=column
        ))
    
    fig.update_layout(
        title=f"Distribution of {column}",
        xaxis_title=column,
        yaxis_title="Frequency",
        template="plotly_white",
        barmode='overlay',
        hovermode="x unified"
    )
    
    return fig


def plot_pairwise(df: pd.DataFrame, var1: str, var2: str):
    """
    Create scatter plot with trendline for pairwise analysis.
    
    Args:
        df: Input dataframe
        var1: First variable
        var2: Second variable
        
    Returns:
        Plotly figure object
    """
    fig = px.scatter(
        df,
        x=var1,
        y=var2,
        color='Country' if 'Country' in df.columns else None,
        title=f"{var1} vs {var2}",
        trendline='ols'
    )
    
    fig.update_layout(template="plotly_white", hovermode="closest")
    return fig


def plot_heatmap(df: pd.DataFrame, x_col: str, y_col: str, value_col: str):
    """
    Create heatmap visualization.
    
    Args:
        df: Input dataframe
        x_col: Column for x-axis
        y_col: Column for y-axis
        value_col: Column for heatmap values
        
    Returns:
        Plotly figure object
    """
    pivot_data = df.pivot_table(values=value_col, index=y_col, columns=x_col)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title=f"{value_col} Heatmap",
        xaxis_title=x_col,
        yaxis_title=y_col,
        template="plotly_white"
    )
    
    return fig


def plot_boxplot(df: pd.DataFrame, x_col: str, y_col: str):
    """
    Create box plot for distribution comparison.
    
    Args:
        df: Input dataframe
        x_col: Grouping column
        y_col: Value column
        
    Returns:
        Plotly figure object
    """
    fig = px.box(df, x=x_col, y=y_col, title=f"{y_col} by {x_col}")
    fig.update_layout(template="plotly_white")
    return fig


def plot_clustering(cluster_df: pd.DataFrame, x_var: str, y_var: str):
    """
    Create clustering scatter plot with labels.
    
    Args:
        cluster_df: DataFrame with cluster assignments
        x_var: X-axis variable
        y_var: Y-axis variable
        
    Returns:
        Plotly figure object
    """
    fig = px.scatter(
        cluster_df,
        x=x_var,
        y=y_var,
        color='Cluster' if 'Cluster' in cluster_df.columns else None,
        text='Country' if 'Country' in cluster_df.columns else None,
        title="Country Market Segmentation",
        size_max=60
    )
    
    fig.update_traces(textposition='top center')
    fig.update_layout(template="plotly_white", hovermode="closest")
    return fig


def plot_forecast(forecast_df: pd.DataFrame, target_var: str):
    """
    Create forecast visualization with multiple countries.
    
    Args:
        forecast_df: DataFrame with forecasts
        target_var: Target variable column name
        
    Returns:
        Plotly figure object
    """
    fig = px.line(
        forecast_df,
        x='Year',
        y=target_var,
        color='Country' if 'Country' in forecast_df.columns else None,
        title=f"10-Year {target_var} Forecast",
        markers=True
    )
    
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        yaxis_title=target_var,
        xaxis_title="Year"
    )
    return fig


def plot_variable_importance(importance_df: pd.DataFrame):
    """
    Create feature importance bar plot.
    
    Args:
        importance_df: DataFrame with Feature and Importance columns
        
    Returns:
        Plotly figure object
    """
    fig = px.bar(
        importance_df.sort_values('Importance'),
        y='Feature',
        x='Importance',
        orientation='h',
        title="Feature Importance in Model",
        color='Importance',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'}
    )
    return fig


def plot_residuals(residuals: np.ndarray, title: str = "Residual Plot"):
    """
    Create residual plot for model diagnostics.
    
    Args:
        residuals: Array of residuals
        title: Title for plot
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=residuals,
        name='Residuals',
        nbinsx=30
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Residual Value",
        yaxis_title="Frequency",
        template="plotly_white"
    )
    
    return fig


def plot_actual_vs_predicted(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Create actual vs predicted scatter plot.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Plotly figure object
    """
    df_plot = pd.DataFrame({
        'Actual': y_true,
        'Predicted': y_pred
    })
    
    fig = px.scatter(
        df_plot,
        x='Actual',
        y='Predicted',
        title="Actual vs Predicted Values",
        trendline='ols'
    )
    
    # Add diagonal line (perfect prediction)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(dash='dash', color='red')
    ))
    
    fig.update_layout(template="plotly_white")
    return fig


def plot_country_comparison(df: pd.DataFrame, metric_col: str):
    """
    Create bar plot comparing metric across countries.
    
    Args:
        df: Input dataframe
        metric_col: Column with metric values
        
    Returns:
        Plotly figure object
    """
    if 'Country' not in df.columns:
        return None
    
    country_avg = df.groupby('Country')[metric_col].mean().sort_values()
    
    fig = px.bar(
        x=country_avg.values,
        y=country_avg.index,
        orientation='h',
        title=f"Average {metric_col} by Country",
        labels={'x': metric_col, 'y': 'Country'}
    )
    
    fig.update_layout(template="plotly_white")
    return fig
