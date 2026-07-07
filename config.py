"""
Configuration for Whisky & Spirits Forecasting Dashboard
All application settings in one centralized location.
"""

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

REQUIRED_COLUMNS = [
    'Country',
    'Year',
]

DEPENDENT_VARIABLES = [
    'dependent_Premium_Whisky',
    'dependent_Premium_Spirits',
]

COVARIATE_COLUMNS = [
    'Inflation',
    'compensasion_employee',
    'current_balance',
    'age_dependency_ratio',
    'GPD_Capita',
    'total_labor_force',
]

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

DEFAULT_TEST_SIZE = 0.2
DEFAULT_FORECAST_HORIZON = 10
DEFAULT_N_CLUSTERS = 3
RANDOM_STATE = 42

# Model hyperparameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
XGB_N_ESTIMATORS = 100
XGB_LEARNING_RATE = 0.05
XGB_MAX_DEPTH = 5

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

PLOTLY_TEMPLATE = "plotly_white"
COLOR_SCALE_DIVERGING = "RdBu"
COLOR_SCALE_SEQUENTIAL = "Viridis"
HOVER_MODE = "x unified"

# ============================================================================
# STATISTICAL CONFIGURATION
# ============================================================================

ALPHA_SIGNIFICANCE = 0.05
IQR_MULTIPLIER = 1.5
MIN_DATA_POINTS = 10

# ============================================================================
# STREAMLIT CONFIGURATION
# ============================================================================

PAGE_LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"
THEME = "light"

# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================

MISSING_STRATEGIES = ["mean", "median", "forward_fill", "drop_rows"]
DEFAULT_MISSING_STRATEGY = "mean"
DEFAULT_REMOVE_OUTLIERS = False

# ============================================================================
# UI CONFIGURATION
# ============================================================================

APP_TITLE = "🥃 Whisky & Spirits Sales Forecasting"
APP_SUBTITLE = "Professional Analytics & Forecasting Dashboard"

SECTION_TITLES = {
    1: "📥 1. Data Upload & Validation",
    2: "🔧 2. Data Preprocessing",
    3: "📊 3. Descriptive Analytics",
    4: "🔗 4. Correlation & Relationship Analysis",
    5: "📈 5. Inferential Statistics",
    6: "🎯 6. Country Clustering",
    7: "🔮 7. Predictive Modeling & Forecasting",
    8: "💾 8. Download Results",
}
