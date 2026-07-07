# 🥃 Whisky & Spirits Sales Forecasting Dashboard

A professional Streamlit application for exploratory data analysis, statistical inference, and machine learning-based forecasting of whisky and spirits sales across African countries.

## 🎯 Features

### 1. **Data Upload & Validation**
- Upload CSV or Excel files
- Automatic schema validation
- Data preview and summary statistics
- Comprehensive error messaging

### 2. **Data Preprocessing**
- 4 missing value imputation strategies (mean, median, forward-fill, drop)
- Outlier detection and removal (IQR method)
- Data quality assessment
- Detailed preprocessing reports

### 3. **Descriptive Analytics**
- Key performance indicators (KPIs)
- Summary statistics (mean, median, std, quartiles, CV)
- Distribution analysis with visualizations
- Correlation matrices with hierarchical clustering
- Country clustering and market segmentation
- Time series trends

### 4. **Inferential Statistics**
- Pearson correlation testing with p-values
- Shapiro-Wilk and D'Agostino-Pearson normality tests
- ANOVA for group comparisons
- Confidence intervals
- Effect sizes and statistical interpretation

### 5. **Predictive Modeling**
- Random Forest and XGBoost regression models
- Automatic lag feature engineering
- Train/test evaluation (R², RMSE, MAE)
- 10-year rolling forecasts per country
- Feature importance visualization

### 6. **Professional Visualizations**
- Correlation heatmaps
- Time series plots
- 3D scatter plots
- Distribution plots
- Country clustering charts
- Forecast comparison visualizations
- Feature importance bars

### 7. **Downloadable Outputs**
- Cleaned dataset (CSV)
- Summary statistics table
- Model predictions and forecasts
- Variable importance rankings

---

## 📊 Expected Input Data Structure

### Required Columns
| Column | Type | Description |
|--------|------|-------------|
| **Country** | Text | Country name (e.g., "Kenya", "Nigeria") |
| **Year** | Integer | Calendar year (e.g., 2015, 2024) |

### Dependent Variables (at least one required)
| Column | Type | Description |
|--------|------|-------------|
| **dependent_Premium_Whisky** | Numeric | Sales volume or revenue for premium whisky |
| **dependent_Premium_Spirits** | Numeric | Sales volume or revenue for premium spirits |

### Economic & Demographic Covariates (optional but recommended)
| Column | Type | Description |
|--------|------|-------------|
| **Inflation** | Numeric | Inflation rate (%) |
| **compensasion_employee** | Numeric | Employee compensation (USD millions) |
| **current_balance** | Numeric | Current account balance (% YoY) |
| **age_dependency_ratio** | Numeric | Dependency ratio (old-age + youth) |
| **GPD_Capita** | Numeric | GDP per capita (current USD) |
| **total_labor_force** | Numeric | Total labor force size |

### Example Data Structure

```csv
Country,Year,dependent_Premium_Whisky,dependent_Premium_Spirits,Inflation,compensasion_employee,current_balance,age_dependency_ratio,GPD_Capita,total_labor_force
Kenya,2015,150.5,200.3,6.2,2500,2.1,78.5,1400,18500000
Kenya,2016,155.2,205.1,6.3,2600,2.3,78.3,1500,19000000
Nigeria,2015,280.1,320.5,9.1,3200,-5.2,85.2,2100,95000000
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/whisky-spirits-forecaster.git
cd whisky-spirits-forecaster
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Locally
```bash
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## ☁️ Deployment on Streamlit Cloud

### Step 1: Push to GitHub
1. Create a GitHub repository (public)
2. Push your code:
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

### Step 2: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository, branch (`main`), and file (`streamlit_app.py`)
5. Click "Deploy"

Your app will be live within seconds!

---

## 📖 How to Use the Dashboard

### 1. Upload Data
- Click "Upload your data" and select a CSV/Excel file
- Or check "Use sample data" for demo purposes
- Review validation report

### 2. Preprocess (Optional)
- Choose missing value strategy (mean/median/forward-fill/drop)
- Toggle outlier removal if needed
- Click "Apply Preprocessing"

### 3. Explore Descriptive Analytics
- Review KPIs and summary statistics
- Examine distributions and correlations
- Analyze country clustering patterns

### 4. Perform Inferential Tests
- Test correlations between variables
- Check normality of distributions
- Compare group means with ANOVA

### 5. Build Predictive Model
- Select target variable (Whisky or Spirits)
- Choose model type (Random Forest or XGBoost)
- Configure test size and forecast horizon
- Click "Train Model & Generate Forecasts"

### 6. Review Results
- Inspect model performance metrics
- View feature importance
- Examine 10-year forecasts by country

### 7. Download Outputs
- Download cleaned dataset
- Export summary statistics
- Save forecast predictions

---

## ⚙️ Configuration & Customization

### Modify Model Parameters
Edit `config.py` to change:
- `DEFAULT_TEST_SIZE` - Train/test split ratio
- `DEFAULT_FORECAST_HORIZON` - Forecast years ahead
- `DEFAULT_N_CLUSTERS` - Number of clusters for segmentation
- `ALPHA_SIGNIFICANCE` - P-value threshold for tests

### Add New Covariates
1. Update `COVARIATE_COLUMNS` in `config.py`
2. Ensure your data includes those columns
3. Models automatically include them

### Adjust Visualization Themes
Change `PLOTLY_TEMPLATE`, `COLOR_SCALE_DIVERGING`, and `COLOR_SCALE_SEQUENTIAL` in `config.py`.

---

## 🔍 Assumptions & Limitations

### Assumptions
1. **Data Independence**: Observations are independent
2. **Linearity**: Correlation tests assume linear relationships
3. **Normality**: Some tests assume normally distributed residuals
4. **Covariate Stationarity**: Future covariates remain stable
5. **No External Shocks**: Forecasts assume normal conditions continue

### Limitations
1. **Minimum Data**: Requires at least 10 rows
2. **Multicollinearity**: Not explicitly handled
3. **Temporal Dependencies**: Basic lag features only
4. **Forecast Uncertainty**: 10+ years ahead becomes increasingly uncertain
5. **Causality**: Correlations do not imply causation

---

## 🐛 Troubleshooting

### **Error: "Missing required columns"**
- Check your CSV has `Country` and `Year` columns
- Ensure column names match exactly (case-sensitive)

### **Error: "No data after preprocessing"**
- Your data has too many missing values
- Try a different imputation strategy
- Disable outlier removal

### **Model won't train**
- Ensure you have at least 10 rows of data
- Check that target variable has sufficient variance
- Try preprocessing with "median" strategy

### **Forecasts look unrealistic**
- Check feature importance; lagged values may dominate
- Verify covariate projections are reasonable
- Consider adjusting preprocessing parameters

### **Streamlit app won't load**
- Verify all files are in correct directory structure
- Check `requirements.txt` has all dependencies
- Run `pip install -r requirements.txt` again
- Check for typos in import statements

---

## 📁 Project Structure

```
whisky-spirits-forecaster/
├── streamlit_app.py           # Main dashboard (765 lines)
├── config.py                   # Configuration parameters
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore rules
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # CSV/Excel upload & sample data
│   ├── validation.py          # Data schema validation
│   ├── preprocess.py          # Missing value & outlier handling
│   ├── descriptive.py         # EDA functions
│   ├── inferential.py         # Statistical tests
│   ├── predictive.py          # ML models & forecasting
│   ├── visualizations.py      # Plotly charts
│   └── utils.py               # Helper functions
├── sample_data/
│   └── sample_whisky_spirits.csv  # Demo dataset
└── tests/
    ├── __init__.py
    └── test_validation.py     # Unit tests
```

---

## 🧪 Testing

### Run Unit Tests
```bash
pip install pytest
pytest tests/ -v
```

### Test Manually
1. Upload sample data
2. Run preprocessing
3. View KPIs and statistics
4. Run correlation test
5. Train model and generate forecasts
6. Download files

---

## 🛠️ Technologies Used

- **Streamlit** - Web framework
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning
- **XGBoost** - Gradient boosting
- **Plotly** - Interactive visualizations
- **SciPy** - Statistical functions

---

## 📈 Key Metrics & Interpretations

### Correlation Coefficient (Pearson r)
- **+1**: Perfect positive relationship
- **0**: No linear relationship
- **-1**: Perfect negative relationship
- **p-value < 0.05**: Statistically significant

### Model Performance
- **R²**: Proportion of variance explained (0-1, higher is better)
- **RMSE**: Root mean squared error (lower is better)
- **MAE**: Mean absolute error (lower is better)

### Feature Importance
- Higher values = stronger predictive power
- Can guide variable selection for future models

---

## 📞 Support & Maintenance

### For Users
- Check README for common issues
- Review data structure requirements
- Use sample data to test features

### For Developers
- Code is modular and well-documented
- Type hints provided for clarity
- Unit tests cover main functionality

### Future Enhancements
- ARIMA/Prophet time series models
- Confidence intervals on forecasts
- Feature engineering UI
- Model ensemble capabilities
- PDF report generation

---

## 📄 License

MIT License - Feel free to use and modify for your projects.

---

## 👥 Contributing

To contribute:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Created by**: Data Analytics Team  
**Last Updated**: 2025  
**Version**: 1.0.0

For questions or issues, please open a GitHub issue or contact the development team.

Happy forecasting! 🚀📊
