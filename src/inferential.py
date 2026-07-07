"""
Inferential statistics module for Whisky & Spirits Forecasting Dashboard
Includes hypothesis testing, statistical tests, and confidence intervals.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, Tuple


def correlation_test(df: pd.DataFrame, var1: str, var2: str) -> Dict[str, Any]:
    """
    Perform Pearson correlation test between two variables.
    Tests for linear relationship and statistical significance.
    
    Args:
        df: Input dataframe
        var1: First variable name
        var2: Second variable name
        
    Returns:
        Dictionary with test results and interpretation
    """
    # Extract and clean data
    data1 = df[var1].dropna()
    data2 = df[var2].dropna()
    
    # Match indices for paired comparison
    common_idx = pd.Index(data1.index).intersection(pd.Index(data2.index))
    data1 = data1[common_idx]
    data2 = data2[common_idx]
    
    if len(data1) < 3:
        return {"Error": "Need at least 3 paired observations"}
    
    # Perform Pearson correlation test
    corr, p_value = stats.pearsonr(data1, data2)
    
    # Interpret correlation strength
    abs_corr = abs(corr)
    if abs_corr > 0.7:
        strength = "Strong"
    elif abs_corr > 0.4:
        strength = "Moderate"
    elif abs_corr > 0.2:
        strength = "Weak"
    else:
        strength = "Very weak/None"
    
    # Determine significance
    significant = "Yes" if p_value < 0.05 else "No"
    
    # Create interpretation
    direction = "positive" if corr > 0 else "negative"
    interpretation = f"{strength} {direction} correlation"
    
    if p_value >= 0.05:
        interpretation += " (not statistically significant)"
    
    return {
        'Variable 1': var1,
        'Variable 2': var2,
        'Pearson r': f"{corr:.4f}",
        'P-Value': f"{p_value:.6f}",
        'Significant (α=0.05)': significant,
        'Strength': strength,
        'Interpretation': interpretation,
        'Sample Size': len(data1)
    }


def test_normality(data: pd.Series, method: str = "shapiro") -> Dict[str, Any]:
    """
    Test if data follows a normal distribution.
    
    Args:
        data: Series to test
        method: Test method - 'shapiro' or 'normaltest'
        
    Returns:
        Dictionary with test results and interpretation
    """
    data = data.dropna()
    
    if len(data) < 3:
        return {"Error": "Need at least 3 observations to test normality"}
    
    try:
        if method == "shapiro" or len(data) <= 5000:
            statistic, p_value = stats.shapiro(data)
            test_name = "Shapiro-Wilk Test"
        else:
            statistic, p_value = stats.normaltest(data)
            test_name = "D'Agostino-Pearson Test"
        
        # Interpretation
        is_normal = p_value > 0.05
        interpretation = "Data appears normally distributed" if is_normal else "Data deviates from normal distribution"
        
        return {
            'Test': test_name,
            'Statistic': f"{statistic:.4f}",
            'P-Value': f"{p_value:.6f}",
            'Normally Distributed (α=0.05)': "Yes" if is_normal else "No",
            'Interpretation': interpretation,
            'Sample Size': len(data),
            'Recommendation': "Can use parametric tests" if is_normal else "Consider non-parametric tests"
        }
    
    except Exception as e:
        return {"Error": f"Normality test failed: {str(e)}"}


def group_comparison_test(
    df: pd.DataFrame,
    metric: str,
    group_col: str
) -> Dict[str, Any]:
    """
    Perform ANOVA (Analysis of Variance) to compare metric across groups.
    Tests if group means are significantly different.
    
    Args:
        df: Input dataframe
        metric: Metric column to compare
        group_col: Column defining groups
        
    Returns:
        Dictionary with ANOVA results and interpretation
    """
    # Extract groups
    groups = [group_data[metric].dropna().values 
              for name, group_data in df.groupby(group_col)]
    
    if len(groups) < 2:
        return {"Error": "Need at least 2 groups for comparison"}
    
    # Check group sizes
    valid_groups = [g for g in groups if len(g) > 0]
    if len(valid_groups) < 2:
        return {"Error": "Insufficient data in groups"}
    
    try:
        # Perform one-way ANOVA
        f_statistic, p_value = stats.f_oneway(*valid_groups)
        
        # Interpretation
        significant = p_value < 0.05
        interpretation = "Groups have significantly different means" if significant else "No significant difference between groups"
        
        # Calculate effect size (eta-squared)
        # This is a simplified calculation
        grand_mean = np.concatenate(valid_groups).mean()
        ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in valid_groups)
        ss_total = sum((x - grand_mean)**2 for g in valid_groups for x in g)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        return {
            'Metric': metric,
            'Groups': group_col,
            'Number of Groups': len(valid_groups),
            'F-Statistic': f"{f_statistic:.4f}",
            'P-Value': f"{p_value:.6f}",
            'Significant (α=0.05)': "Yes" if significant else "No",
            'Effect Size (Eta²)': f"{eta_squared:.4f}",
            'Interpretation': interpretation,
        }
    
    except Exception as e:
        return {"Error": f"ANOVA test failed: {str(e)}"}


def t_test_independent(
    group1: pd.Series,
    group2: pd.Series
) -> Dict[str, Any]:
    """
    Perform independent samples t-test between two groups.
    
    Args:
        group1: First group data
        group2: Second group data
        
    Returns:
        Dictionary with test results
    """
    group1 = group1.dropna()
    group2 = group2.dropna()
    
    if len(group1) < 2 or len(group2) < 2:
        return {"Error": "Each group needs at least 2 observations"}
    
    t_statistic, p_value = stats.ttest_ind(group1, group2)
    
    return {
        'T-Statistic': f"{t_statistic:.4f}",
        'P-Value': f"{p_value:.6f}",
        'Significant (α=0.05)': "Yes" if p_value < 0.05 else "No",
        'Mean Difference': f"{group1.mean() - group2.mean():.4f}",
        'Group 1 Mean': f"{group1.mean():.4f}",
        'Group 2 Mean': f"{group2.mean():.4f}",
    }


def confidence_interval(
    data: pd.Series,
    confidence: float = 0.95
) -> Dict[str, Any]:
    """
    Calculate confidence interval for mean of a distribution.
    
    Args:
        data: Data series
        confidence: Confidence level (0.95 for 95% CI)
        
    Returns:
        Dictionary with CI bounds and statistics
    """
    data = data.dropna()
    
    if len(data) < 2:
        return {"Error": "Need at least 2 observations"}
    
    mean = data.mean()
    std_err = stats.sem(data)
    margin_error = std_err * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    
    ci_lower = mean - margin_error
    ci_upper = mean + margin_error
    
    return {
        'Mean': f"{mean:.4f}",
        'Standard Error': f"{std_err:.4f}",
        f'{int(confidence*100)}% CI Lower': f"{ci_lower:.4f}",
        f'{int(confidence*100)}% CI Upper': f"{ci_upper:.4f}",
        'Margin of Error': f"{margin_error:.4f}",
        'Sample Size': len(data),
    }


def chi_square_test(
    df: pd.DataFrame,
    col1: str,
    col2: str
) -> Dict[str, Any]:
    """
    Perform chi-square test for independence between two categorical variables.
    
    Args:
        df: Input dataframe
        col1: First categorical column
        col2: Second categorical column
        
    Returns:
        Dictionary with test results
    """
    try:
        # Create contingency table
        contingency_table = pd.crosstab(df[col1], df[col2])
        
        # Perform chi-square test
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        
        return {
            'Chi-Square Statistic': f"{chi2:.4f}",
            'P-Value': f"{p_value:.6f}",
            'Degrees of Freedom': dof,
            'Significant (α=0.05)': "Yes" if p_value < 0.05 else "No",
            'Interpretation': "Variables are associated" if p_value < 0.05 else "No association found",
        }
    
    except Exception as e:
        return {"Error": f"Chi-square test failed: {str(e)}"}
