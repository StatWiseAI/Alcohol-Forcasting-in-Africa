"""
Unit tests for Whisky & Spirits Forecasting Dashboard
Tests cover data validation, preprocessing, descriptive analytics, and inferential statistics.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation import validate_dataset, REQUIRED_COLUMNS
from src.preprocess import handle_missing_values, remove_statistical_outliers
from src.descriptive import compute_kpis, analyze_correlations
from src.inferential import correlation_test, test_normality


class TestDataValidation:
    """Test data validation functions"""
    
    @pytest.fixture
    def valid_df(self):
        """Create a valid test dataframe"""
        return pd.DataFrame({
            'Country': ['Kenya', 'Nigeria', 'Kenya', 'Nigeria'],
            'Year': [2020, 2020, 2021, 2021],
            'dependent_Premium_Whisky': [150.0, 280.0, 155.0, 285.0],
            'Inflation': [5.0, 10.0, 5.5, 11.0],
        })
    
    @pytest.fixture
    def invalid_df_missing_country(self):
        """DataFrame missing Country column"""
        return pd.DataFrame({
            'Year': [2020, 2021],
            'dependent_Premium_Whisky': [150.0, 155.0],
        })
    
    @pytest.fixture
    def invalid_df_missing_year(self):
        """DataFrame missing Year column"""
        return pd.DataFrame({
            'Country': ['Kenya', 'Nigeria'],
            'dependent_Premium_Whisky': [150.0, 280.0],
        })
    
    def test_valid_dataset(self, valid_df):
        """Test validation of valid dataset"""
        result = validate_dataset(valid_df)
        assert result['is_valid'] == True
        assert len(result['issues']) == 0
    
    def test_missing_country_column(self, invalid_df_missing_country):
        """Test validation fails when Country column missing"""
        result = validate_dataset(invalid_df_missing_country)
        assert result['is_valid'] == False
        assert any('Country' in issue for issue in result['issues'])
    
    def test_missing_year_column(self, invalid_df_missing_year):
        """Test validation fails when Year column missing"""
        result = validate_dataset(invalid_df_missing_year)
        assert result['is_valid'] == False
        assert any('Year' in issue for issue in result['issues'])
    
    def test_small_dataset_warning(self):
        """Test warning for small datasets"""
        small_df = pd.DataFrame({
            'Country': ['Kenya'],
            'Year': [2020],
            'dependent_Premium_Whisky': [150.0],
        })
        result = validate_dataset(small_df)
        assert len(result['issues']) > 0
    
    def test_metadata_extraction(self, valid_df):
        """Test metadata is correctly extracted"""
        result = validate_dataset(valid_df)
        assert result['n_rows'] == 4
        assert result['n_cols'] == 4
        assert '2020' in result['date_range']


class TestPreprocessing:
    """Test data preprocessing functions"""
    
    @pytest.fixture
    def df_with_missing(self):
        """DataFrame with missing values"""
        return pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [10, np.nan, 30, 40, 50],
            'C': ['x', 'y', 'z', 'x', 'y']
        })
    
    @pytest.fixture
    def df_with_outliers(self):
        """DataFrame with outliers"""
        return pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 100]  # 100 is outlier
        })
    
    def test_handle_missing_mean(self, df_with_missing):
        """Test mean imputation"""
        result = handle_missing_values(df_with_missing, strategy='mean')
        assert result['A'].isnull().sum() == 0
        assert result['B'].isnull().sum() == 0
    
    def test_handle_missing_median(self, df_with_missing):
        """Test median imputation"""
        result = handle_missing_values(df_with_missing, strategy='median')
        assert result['A'].isnull().sum() == 0
        assert result['B'].isnull().sum() == 0
    
    def test_handle_missing_drop(self, df_with_missing):
        """Test drop missing values"""
        result = handle_missing_values(df_with_missing, strategy='drop_rows')
        assert len(result) < len(df_with_missing)
    
    def test_remove_outliers(self, df_with_outliers):
        """Test outlier removal"""
        result = remove_statistical_outliers(df_with_outliers)
        assert 100 not in result['value'].values
        assert len(result) < len(df_with_outliers)


class TestDescriptiveAnalytics:
    """Test descriptive analytics functions"""
    
    @pytest.fixture
    def sample_df(self):
        """Sample data for testing"""
        return pd.DataFrame({
            'Country': ['Kenya', 'Nigeria', 'Kenya', 'Nigeria'],
            'Year': [2020, 2020, 2021, 2021],
            'dependent_Premium_Whisky': [150.0, 280.0, 155.0, 285.0],
            'dependent_Premium_Spirits': [200.0, 320.0, 205.0, 325.0],
            'Inflation': [5.0, 10.0, 5.5, 11.0],
            'GPD_Capita': [1500.0, 2100.0, 1600.0, 2200.0],
        })
    
    def test_compute_kpis(self, sample_df):
        """Test KPI computation"""
        kpis = compute_kpis(sample_df)
        assert 'n_countries' in kpis
        assert kpis['n_countries'] == 2
        assert 'avg_premium_whisky' in kpis
        assert kpis['avg_premium_whisky'] > 0
    
    def test_correlation_matrix(self, sample_df):
        """Test correlation matrix computation"""
        corr = analyze_correlations(sample_df)
        assert isinstance(corr, pd.DataFrame)
        assert (corr.index == corr.columns).all()
        # Correlation with self should be 1
        assert abs(corr.loc['dependent_Premium_Whisky', 'dependent_Premium_Whisky'] - 1.0) < 0.01
    
    def test_correlation_symmetry(self, sample_df):
        """Test correlation matrix is symmetric"""
        corr = analyze_correlations(sample_df)
        pd.testing.assert_frame_equal(corr, corr.T)


class TestInferentialStatistics:
    """Test inferential statistics functions"""
    
    @pytest.fixture
    def sample_df(self):
        """Sample data for testing"""
        np.random.seed(42)
        return pd.DataFrame({
            'var1': np.random.normal(100, 20, 100),
            'var2': np.random.normal(50, 15, 100),
            'var3': np.random.uniform(0, 100, 100)
        })
    
    def test_correlation_test(self, sample_df):
        """Test correlation hypothesis test"""
        result = correlation_test(sample_df, 'var1', 'var2')
        assert 'Pearson r' in result
        assert 'P-Value' in result
        assert 'Interpretation' in result
    
    def test_normality_test(self, sample_df):
        """Test normality test"""
        result = test_normality(sample_df['var1'])
        assert 'Test' in result
        assert 'P-Value' in result
        assert 'Sample Size' in result
    
    def test_normality_insufficient_data(self):
        """Test normality with insufficient data"""
        small_series = pd.Series([1, 2])
        result = test_normality(small_series)
        assert 'Error' in result


class TestIntegration:
    """Integration tests for full pipeline"""
    
    @pytest.fixture
    def full_df(self):
        """Create a complete test dataset"""
        np.random.seed(42)
        countries = ['Kenya', 'Nigeria', 'South Africa']
        years = list(range(2015, 2025))
        
        data = []
        for country in countries:
            for year in years:
                data.append({
                    'Country': country,
                    'Year': year,
                    'dependent_Premium_Whisky': np.random.uniform(100, 400),
                    'dependent_Premium_Spirits': np.random.uniform(150, 500),
                    'Inflation': np.random.uniform(2, 15),
                    'compensasion_employee': np.random.uniform(1000, 5000),
                    'current_balance': np.random.uniform(-20, 20),
                    'age_dependency_ratio': np.random.uniform(40, 90),
                    'GPD_Capita': np.random.uniform(1000, 10000),
                    'total_labor_force': np.random.uniform(1e7, 1e8),
                })
        
        return pd.DataFrame(data)
    
    def test_full_pipeline(self, full_df):
        """Test complete analysis pipeline"""
        # Validate
        validation = validate_dataset(full_df)
        assert validation['is_valid']
        
        # Preprocess
        processed = handle_missing_values(full_df, strategy='mean')
        assert processed.isnull().sum().sum() == 0
        
        # Descriptive
        kpis = compute_kpis(processed)
        assert len(kpis) > 0
        
        corr = analyze_correlations(processed)
        assert corr.shape[0] > 0
        
        # Inferential
        numeric_cols = processed.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            corr_test_result = correlation_test(processed, numeric_cols[0], numeric_cols[1])
            assert isinstance(corr_test_result, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
