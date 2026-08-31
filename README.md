# Customer Churn Prediction

A comprehensive machine learning solution for predicting customer churn using classification algorithms. This project demonstrates the complete ML pipeline from data generation through model evaluation and production predictions.

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active-brightgreen.svg)

## 📋 Project Overview

This repository contains a production-ready machine learning pipeline for customer churn prediction. The model analyzes customer demographics, service subscriptions, and billing information to identify churn risk and enable proactive retention strategies.

### Key Highlights

- **5,000 synthetic customer records** with realistic patterns and distributions
- **Comprehensive EDA** with 6 visualization plots
- **3 classification models**: Logistic Regression, Random Forest, Gradient Boosting
- **Cross-validation & evaluation** using ROC-AUC, confusion matrices, and classification reports
- **Feature importance analysis** identifying key churn drivers
- **Production-ready predictions** for new customer scoring

## 🎯 Business Value

- **Identify high-risk customers** before they churn
- **Optimize retention budgets** by targeting high-value customers
- **Understand churn drivers** to improve products/services
- **Measure campaign effectiveness** with before/after churn rates
- **Real-time risk scoring** for new customer acquisition

## 📊 Dataset

### Features (13 total)

| Feature | Type | Description |
|---------|------|-------------|
| Gender | Categorical | Male / Female |
| SeniorCitizen | Binary | 0 = No, 1 = Yes |
| Partner | Binary | Whether customer has partner |
| Dependents | Binary | Whether customer has dependents |
| Tenure | Numeric | Months as customer (1-72) |
| PhoneService | Binary | Phone service subscription |
| InternetService | Categorical | DSL / Fiber optic / No |
| Contract | Categorical | Month-to-month / 1-year / 2-year |
| PaperlessBilling | Binary | Paperless billing enabled |
| PaymentMethod | Categorical | 4 payment types |
| MonthlyCharges | Numeric | Monthly bill amount ($) |
| TotalCharges | Numeric | Total lifetime charges ($) |
| **Churn** | **Target** | **0 = No churn, 1 = Churn** |

### Churn Distribution

- **No Churn**: 73% (3,650 customers)
- **Churn**: 27% (1,350 customers)

## 🏗️ Project Structure

```
customer-churn-prediction/
├── churn_prediction.py          # Main script with complete pipeline
├── requirements.txt              # Python dependencies
├── README.md                      # This file
├── LICENSE                        # MIT License
├── .gitignore                     # Git ignore patterns
└── docs/
    └── Customer_Churn_Prediction_Documentation.docx  # Full documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- ~2GB disk space for visualizations

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

```bash
# Execute complete pipeline
python churn_prediction.py
```

This will:
1. Generate 5,000 synthetic customer records
2. Display 6 EDA visualization plots
3. Preprocess and engineer 30+ features
4. Train 3 classification models
5. Display cross-validation scores
6. Generate ROC curves and confusion matrices
7. Display feature importance chart
8. Score 2 sample customers and print predictions

## 📈 Model Performance

### Cross-Validation Results (5-Fold ROC-AUC)

| Model | Mean AUC | Std Dev |
|-------|----------|---------|
| Logistic Regression | ~0.82 | ±0.03 |
| Random Forest | ~0.87 | ±0.02 |
| Gradient Boosting | ~0.85 | ±0.03 |

**Best Model**: Random Forest (selected for production)

### Evaluation Metrics

```
Precision:   Proportion of predicted churners who actually churn
Recall:      Proportion of actual churners correctly identified
F1-Score:    Harmonic mean of precision and recall
ROC-AUC:     Area under ROC curve (0.5 = random, 1.0 = perfect)
```

## 🔍 Key Findings

### Top Churn Drivers (Feature Importance)

1. **Tenure** (strongest negative indicator)
   - Longer tenure → lower churn risk
   - Impact: Each month increases retention

2. **Contract Type**
   - Month-to-month: +43% churn rate
   - 2-year: +8% churn rate
   - Impact: Contract length is critical

3. **Monthly Charges**
   - Higher charges → higher churn
   - Impact: Price sensitivity exists

4. **Internet Service**
   - Fiber optic: +35% churn rate
   - DSL: +20% churn rate
   - Impact: Service quality affects retention

5. **Payment Method**
   - Electronic check: +32% churn rate
   - Other methods: +15-18% churn rate
   - Impact: Payment friction indicates risk

### Business Recommendations

✅ **High Priority**
- Convert month-to-month to longer contracts with incentives
- Improve fiber optic service quality and support
- Implement early intervention for electronic check payers
- Increase engagement in first 3-6 months

✅ **Medium Priority**
- Monitor customers with charges >$75/month
- Create loyalty programs for long-tenure customers
- Optimize billing processes to reduce friction

✅ **Low Priority**
- Review demographics in marketing
- Test pricing strategies for high-charge segments

## 📊 Visualizations Generated

1. **Churn Distribution Pie Chart**
   - Overall churn rate overview

2. **Tenure Histograms**
   - Tenure patterns by churn status

3. **Monthly Charges Boxplot**
   - Charges distribution by churn

4. **Contract Type Bar Chart**
   - Churn rates by contract length

5. **Internet Service Bar Chart**
   - Churn rates by service type

6. **Correlation Heatmap**
   - Feature correlation analysis

7. **ROC Curves (3 models)**
   - Model discrimination ability comparison

8. **Confusion Matrices (3 models)**
   - Classification performance details

9. **Feature Importance Bar Chart**
   - Top 15 influential features

## 🔮 Making Predictions

### Scoring New Customers

```python
from churn_prediction import preprocess_new
import pandas as pd

# Prepare new customer data
new_customer = pd.DataFrame({
    "Gender": ["Male"],
    "SeniorCitizen": [0],
    "Partner": ["Yes"],
    "Dependents": ["No"],
    "Tenure": [12],
    "PhoneService": ["Yes"],
    "InternetService": ["Fiber optic"],
    "Contract": ["Month-to-month"],
    "PaperlessBilling": ["Yes"],
    "PaymentMethod": ["Electronic check"],
    "MonthlyCharges": [99.5],
    "TotalCharges": [1194.0],
})

# Preprocess and predict
X_new = preprocess_new(new_customer, trained_features)
churn_probability = model.predict_proba(X_new)[0, 1]
prediction = model.predict(X_new)[0]

print(f"Churn Probability: {churn_probability:.2%}")
print(f"Prediction: {'CHURN' if prediction else 'STAY'}")
```

### Probability Interpretation

| Probability Range | Risk Level | Action |
|-------------------|-----------|--------|
| 0.0 - 0.3 | Low | Monitor normally |
| 0.3 - 0.7 | Medium | Targeted retention campaigns |
| 0.7 - 1.0 | High | Urgent intervention needed |

## 📚 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | ≥1.21.0 | Numerical computing |
| pandas | ≥1.3.0 | Data manipulation |
| scikit-learn | ≥1.0.0 | ML algorithms |
| matplotlib | ≥3.4.0 | Static visualizations |
| seaborn | ≥0.11.0 | Statistical graphics |

### Optional Dependencies

```bash
# For Jupyter notebook development
pip install jupyter ipython

# Install all including optional
pip install -r requirements.txt
```

## 🔧 Configuration

### Model Hyperparameters

**Logistic Regression**
```python
{
    "max_iter": 1000,
    "random_state": 42,
    "scaler": StandardScaler()
}
```

**Random Forest**
```python
{
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_leaf": 20,
    "class_weight": "balanced",
    "random_state": 42
}
```

**Gradient Boosting**
```python
{
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "random_state": 42
}
```

## 🧪 Testing & Validation

### Cross-Validation Strategy

- **Method**: Stratified K-Fold (5 splits)
- **Stratification**: Maintains churn class distribution
- **Metric**: ROC-AUC score
- **Purpose**: Robust performance estimation

### Performance Verification

```bash
python churn_prediction.py

# Expected output:
# - 6 visualization plots
# - Cross-validation scores for 3 models
# - Test set metrics and confusion matrices
# - Feature importance rankings
# - Predictions for 2 sample customers
```

## 📁 Output Files

After running the script, you'll get:

- **Console output**: Model metrics and predictions
- **Matplotlib figures**:
  - `EDA.png` - Exploratory data analysis plots
  - `ROC_Curves.png` - Model ROC curves comparison
  - `Confusion_Matrices.png` - Classification results
  - `Feature_Importance.png` - Top 15 features chart

## 🤝 Contributing

Contributions welcome! Areas for enhancement:

- [ ] Add hyperparameter tuning (GridSearchCV)
- [ ] Implement feature selection techniques
- [ ] Add cross-validation strategy comparison
- [ ] Create API wrapper for model serving
- [ ] Add explainability features (SHAP values)
- [ ] Implement data drift monitoring
- [ ] Add unit tests and CI/CD pipeline
- [ ] Create interactive dashboard

## 📋 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## ✍️ Author

BUDAGALA SRI SAI
August 2025

## 🙏 Acknowledgments

- Inspired by real-world telco churn prediction challenges
- Built with scikit-learn, pandas, and matplotlib
- Synthetic data generation for demonstration purposes

## 📞 Support

For issues, questions, or suggestions:
1. Check existing [Issues](https://github.com/yourusername/customer-churn-prediction/issues)
2. Create a new issue with detailed description
3. Include error messages and environment details

## 📖 Additional Resources

- [Complete Documentation](docs/Customer_Churn_Prediction_Documentation.docx)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Matplotlib Guide](https://matplotlib.org/)

---

⭐ If you found this project useful, please consider giving it a star!

Made with ❤️ for the data science community
