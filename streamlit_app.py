"""
Whisky & Spirits Forecasting Dashboard
A professional Streamlit application for exploratory and predictive analytics on whisky/spirits sales data.

Author: Data Analytics Team
Date: 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
from io import StringIO

warnings.filterwarnings('ignore')

# Import custom modules
from src.data_loader import load_uploaded_file, load_sample_data
from src.validation import validate_dataset, get_validation_report
from src.preprocess import preprocess_data, handle_missing_values
from src.descriptive import (
    compute_kpis,
    get_summary_statistics,
    analyze_distributions,
    analyze_correlations,
    perform_clustering,
)
from src.inferential import (
    correlation_test,
    group_comparison_test,
    test_normality,
)
from src.predictive import (
    prepare_forecast_data,
    train_random_forest,
    train_xgboost,
    generate_forecasts,
    evaluate_model,
)
from src.visualizations import (
    plot_correlation_matrix,
    plot_time_series,
    plot_3d_scatter,
    plot_distribution_ridges,
    plot_pairwise,
    plot_heatmap,
    plot_boxplot,
    plot_clustering,
    plot_forecast,
    plot_variable_importance,
)
from src.utils import download_dataframe

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Whisky & Spirits Forecaster",
    page_icon="🥃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E4C6E;
        margin-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1E4C6E;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E46726;
        padding-bottom: 0.5rem;
    }
    .kpi-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #E46726;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-left: 4px solid #1E4C6E;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3e0;
        padding: 1rem;
        border-left: 4px solid #ff9800;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

if "data" not in st.session_state:
    st.session_state.data = None
if "validation_passed" not in st.session_state:
    st.session_state.validation_passed = False
if "preprocessed_data" not in st.session_state:
    st.session_state.preprocessed_data = None
if "forecast_results" not in st.session_state:
    st.session_state.forecast_results = None

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Title and intro
    st.markdown('<p class="main-header">🥃 Whisky & Spirits Sales Forecasting</p>', 
                unsafe_allow_html=True)
    st.markdown("""
    **Professional Analytics & Forecasting Dashboard**
    
    Upload your whisky or spirits sales data to receive instant insights including:
    - Exploratory data analysis and correlations
    - Statistical hypothesis testing
    - Advanced predictive modeling with 10-year forecasts
    - Professional visualizations and downloadable reports
    """)

    # ========================================================================
    # SECTION 1: DATA UPLOAD & VALIDATION
    # ========================================================================
    
    st.markdown('<p class="section-header">📥 1. Data Upload & Validation</p>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your data (CSV or Excel)",
            type=["csv", "xlsx", "xls"],
            help="Expected columns: Country, Year, dependent_Premium_Whisky or dependent_Premium_Spirits, and economic indicators"
        )
    
    with col2:
        use_sample = st.checkbox("Or use sample data", value=False)
    
    if use_sample:
        st.session_state.data = load_sample_data()
        st.success("✅ Sample data loaded successfully!")
    elif uploaded_file is not None:
        try:
            st.session_state.data = load_uploaded_file(uploaded_file)
            st.success("✅ File loaded successfully!")
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.session_state.data = None
    
    # Validate data
    if st.session_state.data is not None:
        validation_report = validate_dataset(st.session_state.data)
        
        if validation_report["is_valid"]:
            st.markdown('<div class="info-box">✅ <b>Data Validation Passed</b></div>', 
                       unsafe_allow_html=True)
            st.session_state.validation_passed = True
            
            # Display data preview and stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", validation_report["n_rows"])
            with col2:
                st.metric("Columns", validation_report["n_cols"])
            with col3:
                st.metric("Date Range", f"{validation_report['date_range']}")
            
            st.subheader("Data Preview")
            st.dataframe(st.session_state.data.head(10), use_container_width=True)
            
        else:
            st.markdown('<div class="warning-box">⚠️ <b>Data Validation Issues</b></div>', 
                       unsafe_allow_html=True)
            for issue in validation_report["issues"]:
                st.warning(f"• {issue}")
            st.session_state.validation_passed = False
    
    # ========================================================================
    # SECTION 2: DATA PREPROCESSING
    # ========================================================================
    
    if st.session_state.validation_passed and st.session_state.data is not None:
        st.markdown('<p class="section-header">🔧 2. Data Preprocessing</p>', 
                    unsafe_allow_html=True)
        
        with st.expander("Show Preprocessing Options", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                handle_missing = st.selectbox(
                    "Handle Missing Values",
                    ["mean", "median", "forward_fill", "drop_rows"],
                    help="Strategy for imputing missing numeric values"
                )
            
            with col2:
                remove_outliers = st.checkbox(
                    "Remove Statistical Outliers (IQR method)",
                    value=False,
                    help="Remove extreme values beyond 1.5×IQR"
                )
            
            preprocess_button = st.button("Apply Preprocessing", key="preprocess_btn")
        
        if preprocess_button or st.session_state.preprocessed_data is not None:
            try:
                st.session_state.preprocessed_data = preprocess_data(
                    st.session_state.data.copy(),
                    missing_strategy=handle_missing,
                    remove_outliers=remove_outliers
                )
                st.success("✅ Preprocessing complete!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rows After Processing", len(st.session_state.preprocessed_data))
                with col2:
                    missing_pct = (st.session_state.preprocessed_data.isnull().sum().sum() / 
                                   (st.session_state.preprocessed_data.shape[0] * 
                                    st.session_state.preprocessed_data.shape[1]) * 100)
                    st.metric("Missing Data %", f"{missing_pct:.1f}%")
                
            except Exception as e:
                st.error(f"❌ Preprocessing error: {str(e)}")
        
        # ====================================================================
        # SECTION 3: DESCRIPTIVE ANALYTICS
        # ====================================================================
        
        if st.session_state.preprocessed_data is not None:
            data_for_analysis = st.session_state.preprocessed_data
            
            st.markdown('<p class="section-header">📊 3. Descriptive Analytics</p>', 
                        unsafe_allow_html=True)
            
            # KPIs
            st.subheader("Key Performance Indicators")
            kpis = compute_kpis(data_for_analysis)
            
            col1, col2, col3, col4 = st.columns(4)
            metrics_order = [
                ("Avg Whisky Sales", "avg_premium_whisky"),
                ("Avg Spirits Sales", "avg_premium_spirits"),
                ("Avg GDP/Capita", "avg_gdp_capita"),
                ("Countries", "n_countries"),
            ]
            
            cols = [col1, col2, col3, col4]
            for idx, (label, key) in enumerate(metrics_order):
                if key in kpis:
                    with cols[idx]:
                        value = kpis[key]
                        if isinstance(value, (int, float)):
                            st.metric(label, f"{value:,.0f}")
                        else:
                            st.metric(label, value)
            
            # Summary Statistics
            st.subheader("Summary Statistics")
            summary_stats = get_summary_statistics(data_for_analysis)
            st.dataframe(summary_stats, use_container_width=True)
            
            # Distributions
            st.subheader("Distribution Analysis")
            tab1, tab2, tab3 = st.tabs(["Whisky Distribution", "Spirits Distribution", "Other Variables"])
            
            with tab1:
                if "dependent_Premium_Whisky" in data_for_analysis.columns:
                    fig = st.plotly_chart(
                        plot_distribution_ridges(
                            data_for_analysis, 
                            "dependent_Premium_Whisky"
                        ),
                        use_container_width=True
                    )
            
            with tab2:
                if "dependent_Premium_Spirits" in data_for_analysis.columns:
                    fig = st.plotly_chart(
                        plot_distribution_ridges(
                            data_for_analysis, 
                            "dependent_Premium_Spirits"
                        ),
                        use_container_width=True
                    )
            
            with tab3:
                numeric_cols = data_for_analysis.select_dtypes(include=[np.number]).columns.tolist()
                selected_col = st.selectbox(
                    "Select variable",
                    numeric_cols,
                    key="dist_select"
                )
                if selected_col:
                    fig = st.plotly_chart(
                        plot_distribution_ridges(
                            data_for_analysis, 
                            selected_col
                        ),
                        use_container_width=True
                    )
            
            # ================================================================
            # SECTION 4: CORRELATION & RELATIONSHIP ANALYSIS
            # ================================================================
            
            st.markdown('<p class="section-header">🔗 4. Correlation & Relationship Analysis</p>', 
                        unsafe_allow_html=True)
            
            st.subheader("Correlation Matrix")
            cor_matrix = analyze_correlations(data_for_analysis)
            fig = st.plotly_chart(plot_correlation_matrix(cor_matrix), use_container_width=True)
            
            st.markdown("""
            **Interpretation:**
            - Values close to +1 indicate strong positive relationships
            - Values close to -1 indicate strong negative relationships
            - Values near 0 indicate weak or no linear relationship
            """)
            
            st.subheader("Pairwise Relationships")
            numeric_cols = data_for_analysis.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    x_var = st.selectbox("X Variable", numeric_cols, key="x_pair")
                with col2:
                    y_var = st.selectbox("Y Variable", numeric_cols, key="y_pair", 
                                        index=1 if len(numeric_cols) > 1 else 0)
                
                if x_var and y_var:
                    fig = st.plotly_chart(
                        plot_3d_scatter(data_for_analysis, x_var, y_var),
                        use_container_width=True
                    )
            
            # ================================================================
            # SECTION 5: INFERENTIAL STATISTICS
            # ================================================================
            
            st.markdown('<p class="section-header">📈 5. Inferential Statistics</p>', 
                        unsafe_allow_html=True)
            
            st.subheader("Hypothesis Tests & Statistical Tests")
            
            inference_tab1, inference_tab2, inference_tab3 = st.tabs(
                ["Correlation Tests", "Normality Tests", "Group Comparisons"]
            )
            
            with inference_tab1:
                numeric_cols = data_for_analysis.select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_cols) >= 2:
                    col1, col2 = st.columns(2)
                    with col1:
                        test_var1 = st.selectbox("Variable 1", numeric_cols, key="test_var1")
                    with col2:
                        test_var2 = st.selectbox("Variable 2", numeric_cols, 
                                                key="test_var2", index=1 if len(numeric_cols) > 1 else 0)
                    
                    if test_var1 and test_var2 and st.button("Run Correlation Test"):
                        result = correlation_test(data_for_analysis, test_var1, test_var2)
                        st.write(result)
            
            with inference_tab2:
                numeric_cols = data_for_analysis.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    test_var = st.selectbox("Select Variable", numeric_cols, key="norm_test_var")
                    if st.button("Test Normality (Shapiro-Wilk)"):
                        result = test_normality(data_for_analysis[test_var].dropna())
                        st.write(result)
            
            with inference_tab3:
                if "Country" in data_for_analysis.columns:
                    numeric_cols = data_for_analysis.select_dtypes(include=[np.number]).columns.tolist()
                    if numeric_cols:
                        metric_var = st.selectbox("Select Metric", numeric_cols, key="group_metric")
                        if st.button("Compare Countries (ANOVA)"):
                            result = group_comparison_test(data_for_analysis, metric_var, "Country")
                            st.write(result)
                else:
                    st.info("Country column not found for group comparisons")
            
            # ================================================================
            # SECTION 6: CLUSTERING ANALYSIS
            # ================================================================
            
            st.markdown('<p class="section-header">🎯 6. Country Clustering</p>', 
                        unsafe_allow_html=True)
            
            if "Country" in data_for_analysis.columns:
                st.subheader("Market Segmentation by Economic Indicators")
                
                try:
                    cluster_result = perform_clustering(data_for_analysis)
                    
                    if cluster_result is not None and len(cluster_result) > 0:
                        col1, col2 = st.columns(2)
                        
                        numeric_cols = cluster_result.select_dtypes(include=[np.number]).columns.tolist()
                        numeric_cols = [c for c in numeric_cols if c != "Cluster"]
                        
                        with col1:
                            x_cluster = st.selectbox("X Axis", numeric_cols, key="cluster_x")
                        with col2:
                            y_cluster = st.selectbox("Y Axis", numeric_cols, key="cluster_y",
                                                    index=1 if len(numeric_cols) > 1 else 0)
                        
                        if x_cluster and y_cluster:
                            fig = st.plotly_chart(
                                plot_clustering(cluster_result, x_cluster, y_cluster),
                                use_container_width=True
                            )
                        
                        st.subheader("Cluster Assignments")
                        st.dataframe(cluster_result, use_container_width=True)
                
                except Exception as e:
                    st.warning(f"Could not perform clustering: {str(e)}")
            
            # ================================================================
            # SECTION 7: PREDICTIVE MODELING & FORECASTING
            # ================================================================
            
            st.markdown('<p class="section-header">🔮 7. Predictive Modeling & Forecasting</p>', 
                        unsafe_allow_html=True)
            
            with st.expander("Model Configuration", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    target_var = st.selectbox(
                        "Select Target Variable",
                        ["dependent_Premium_Whisky", "dependent_Premium_Spirits"],
                        key="target_select"
                    )
                
                with col2:
                    test_size = st.slider("Test Size (%)", 10, 50, 20, key="test_size")
                
                with col3:
                    forecast_horizon = st.slider("Forecast Horizon (years)", 5, 20, 10, key="horizon")
                
                model_type = st.selectbox(
                    "Model Type",
                    ["Random Forest", "XGBoost"],
                    key="model_select"
                )
            
            if st.button("Train Model & Generate Forecasts", key="train_model_btn"):
                with st.spinner("Training models and generating forecasts..."):
                    try:
                        # Prepare data
                        model_data = prepare_forecast_data(
                            data_for_analysis,
                            target_var=target_var
                        )
                        
                        if model_data is not None and len(model_data) > 0:
                            # Train models
                            if model_type == "Random Forest":
                                results = train_random_forest(model_data, target_var, test_size/100)
                            else:
                                results = train_xgboost(model_data, target_var, test_size/100)
                            
                            if results:
                                st.session_state.forecast_results = results
                                st.success("✅ Model trained successfully!")
                            else:
                                st.error("Could not train model")
                        else:
                            st.error("Insufficient data for modeling")
                    
                    except Exception as e:
                        st.error(f"❌ Error during model training: {str(e)}")
            
            # Display results
            if st.session_state.forecast_results is not None:
                results = st.session_state.forecast_results
                
                col1, col2, col3, col4 = st.columns(4)
                
                if "train_r2" in results:
                    with col1:
                        st.metric("Train R²", f"{results['train_r2']:.3f}")
                if "test_r2" in results:
                    with col2:
                        st.metric("Test R²", f"{results['test_r2']:.3f}")
                if "test_rmse" in results:
                    with col3:
                        st.metric("Test RMSE", f"{results['test_rmse']:.2f}")
                if "test_mae" in results:
                    with col4:
                        st.metric("Test MAE", f"{results['test_mae']:.2f}")
                
                # Feature Importance
                if "feature_importance" in results and results["feature_importance"] is not None:
                    st.subheader("Feature Importance")
                    fig = st.plotly_chart(
                        plot_variable_importance(results["feature_importance"]),
                        use_container_width=True
                    )
                
                # Forecasts
                if "forecasts" in results:
                    st.subheader("10-Year Forecast")
                    
                    forecasts_df = results["forecasts"]
                    
                    # Plot by country
                    countries = forecasts_df["Country"].unique()
                    selected_countries = st.multiselect(
                        "Select countries to display",
                        countries,
                        default=list(countries)[:3] if len(countries) > 3 else list(countries)
                    )
                    
                    if selected_countries:
                        filtered_forecasts = forecasts_df[forecasts_df["Country"].isin(selected_countries)]
                        
                        fig = st.plotly_chart(
                            plot_forecast(filtered_forecasts, target_var),
                            use_container_width=True
                        )
                    
                    # Download forecasts
                    st.subheader("Download Forecasts")
                    csv_download = download_dataframe(forecasts_df, "forecasts")
                    st.download_button(
                        label="📥 Download Forecast Data (CSV)",
                        data=csv_download,
                        file_name="forecasts.csv",
                        mime="text/csv",
                        key="download_forecasts"
                    )
            
            # ================================================================
            # SECTION 8: DOWNLOADS
            # ================================================================
            
            st.markdown('<p class="section-header">💾 8. Download Results</p>', 
                        unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                cleaned_csv = download_dataframe(data_for_analysis, "cleaned_data")
                st.download_button(
                    label="📥 Cleaned Dataset",
                    data=cleaned_csv,
                    file_name="cleaned_data.csv",
                    mime="text/csv",
                    key="download_cleaned"
                )
            
            with col2:
                summary_csv = download_dataframe(summary_stats, "summary_stats")
                st.download_button(
                    label="📥 Summary Statistics",
                    data=summary_csv,
                    file_name="summary_statistics.csv",
                    mime="text/csv",
                    key="download_summary"
                )
            
            with col3:
                if st.session_state.forecast_results and "forecasts" in st.session_state.forecast_results:
                    forecasts_df = st.session_state.forecast_results["forecasts"]
                    forecast_csv = download_dataframe(forecasts_df, "forecast_results")
                    st.download_button(
                        label="📥 Forecast Results",
                        data=forecast_csv,
                        file_name="forecast_results.csv",
                        mime="text/csv",
                        key="download_forecast"
                    )

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()
