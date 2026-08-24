"""
Configuration and Settings for Customer Churn Prediction
=========================================================
This module contains all configurable parameters for the project.
Modify settings here rather than hardcoding values in scripts.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS & DIRECTORIES
# ============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, OUTPUTS_DIR, REPORTS_DIR]:
    directory.mkdir(exist_ok=True)

# ============================================================================
# DATA GENERATION SETTINGS
# ============================================================================

# Dataset size
DATASET_SIZE = 5000  # Number of customer records to generate

# Random seed for reproducibility
RANDOM_SEED = 42

# Tenure range (months)
TENURE_MIN = 1
TENURE_MAX = 72

# Monthly charges range (dollars)
MONTHLY_CHARGES_MIN = 18.0
MONTHLY_CHARGES_MAX = 118.0

# Customer demographics distribution
SENIOR_CITIZEN_RATIO = 0.16  # 16% senior citizens
PARTNER_RATIO = 0.50  # 50% have partners
DEPENDENTS_RATIO = 0.30  # 30% have dependents
PHONE_SERVICE_RATIO = 0.90  # 90% have phone service

# Service distributions
INTERNET_SERVICE_DISTRIBUTION = {
    "DSL": 0.35,
    "Fiber optic": 0.44,
    "No": 0.21
}

CONTRACT_DISTRIBUTION = {
    "Month-to-month": 0.55,
    "One year": 0.24,
    "Two year": 0.21
}

PAYMENT_METHOD_OPTIONS = [
    "Electronic check",
    "Mailed check",
    "Bank transfer",
    "Credit card"
]

# Churn probability factors
CHURN_BASE_PROBABILITY = 0.05
CHURN_FACTORS = {
    "month_to_month_increase": 0.25,
    "two_year_decrease": -0.12,
    "fiber_optic_increase": 0.15,
    "electronic_check_increase": 0.10,
    "tenure_above_36_decrease": -0.05,
    "high_charges_increase": 0.08,  # For charges > $75
    "senior_citizen_increase": 0.04
}

CHURN_PROBABILITY_BOUNDS = (0.02, 0.92)  # Min and max possible churn probability

# ============================================================================
# PREPROCESSING SETTINGS
# ============================================================================

# Test/Train split
TEST_SIZE = 0.2  # 20% test, 80% train
STRATIFY_SPLIT = True  # Maintain class distribution

# Features to encode as binary
BINARY_FEATURES = [
    "Partner",
    "Dependents", 
    "PhoneService",
    "PaperlessBilling"
]

# Features to one-hot encode
CATEGORICAL_FEATURES = [
    "InternetService",
    "Contract",
    "PaymentMethod"
]

# Feature engineering settings
CHARGES_PER_MONTH_DENOMINATOR_ADJUSTMENT = 1  # Add to tenure to avoid division by zero
TENURE_BINS = [0, 12, 24, 48, 72]
TENURE_GROUP_LABELS = [1, 2, 3, 4]

# ============================================================================
# MODEL SETTINGS
# ============================================================================

# Cross-validation
CV_FOLDS = 5
CV_STRATIFIED = True
CV_SHUFFLE = True
CV_SHUFFLE_SEED = 42

# Logistic Regression
LOGISTIC_REGRESSION_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
    "solver": "lbfgs"
}

# Random Forest
RANDOM_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_leaf": 20,
    "min_samples_split": 10,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1  # Use all processors
}

# Gradient Boosting
GRADIENT_BOOSTING_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_samples_leaf": 10,
    "min_samples_split": 10,
    "subsample": 0.8,
    "random_state": RANDOM_SEED
}

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================

# Primary evaluation metric
PRIMARY_METRIC = "roc_auc"

# Probability threshold for churn classification
CHURN_PROBABILITY_THRESHOLD = 0.5

# Risk levels for predictions
RISK_LEVELS = {
    "low": (0.0, 0.3),
    "medium": (0.3, 0.7),
    "high": (0.7, 1.0)
}

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Figure sizes (width, height)
FIGURE_SIZE_SMALL = (8, 6)
FIGURE_SIZE_MEDIUM = (12, 8)
FIGURE_SIZE_LARGE = (16, 10)
FIGURE_SIZE_WIDE = (18, 5)

# Colors
COLORS = {
    "no_churn": "#4CAF50",  # Green
    "churn": "#F44336",      # Red
    "logistic": "#3498DB",   # Blue
    "random_forest": "#2ECC71",  # Green
    "gradient_boost": "#E74C3C"   # Red
}

# Plot styling
PLOT_STYLE = "seaborn-v0_8-darkgrid"
DPI = 100
FONT_SIZE_TITLE = 16
FONT_SIZE_HEADING = 12
FONT_SIZE_LABEL = 10

# ============================================================================
# LOGGING SETTINGS
# ============================================================================

# Log level
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Log file
LOG_FILE = OUTPUTS_DIR / "churn_prediction.log"

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ============================================================================
# PREDICTION SETTINGS
# ============================================================================

# Sample customers for demonstration
SAMPLE_CUSTOMERS = [
    {
        "Gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "Tenure": 3,
        "PhoneService": "Yes",
        "InternetService": "Fiber optic",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.5,
        "TotalCharges": 286.5,
    },
    {
        "Gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "Tenure": 48,
        "PhoneService": "Yes",
        "InternetService": "DSL",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer",
        "MonthlyCharges": 45.0,
        "TotalCharges": 2160.0,
    }
]

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# Feature selection
FEATURE_SELECTION_METHOD = None  # Options: None, "SelectKBest", "RFE"
NUM_FEATURES_TO_SELECT = None  # If using feature selection

# Feature importance
NUM_TOP_FEATURES_DISPLAY = 15  # Number of features to show in importance plot

# ============================================================================
# PERFORMANCE & OPTIMIZATION
# ============================================================================

# Batch prediction size
BATCH_PREDICTION_SIZE = 1000

# Cache preprocessed data
USE_CACHE = False
CACHE_DIR = PROJECT_ROOT / ".cache"

# Parallel processing
N_JOBS = -1  # -1 = all processors

# ============================================================================
# EXPERIMENTAL SETTINGS
# ============================================================================

# Enable SHAP explainability (requires shap package)
USE_SHAP = False

# Enable hyperparameter tuning
ENABLE_HYPERPARAMETER_TUNING = False

# GridSearch CV parameters for tuning
TUNING_CV_FOLDS = 3
TUNING_SCORING_METRIC = "roc_auc"

# ============================================================================
# ERROR HANDLING
# ============================================================================

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1

# Validation settings
VALIDATE_DATA = True
HANDLE_MISSING_VALUES = True
REMOVE_OUTLIERS = False

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_config_summary():
    """Return a summary of current configuration."""
    return {
        "dataset_size": DATASET_SIZE,
        "test_size": TEST_SIZE,
        "cv_folds": CV_FOLDS,
        "models": {
            "logistic_regression": LOGISTIC_REGRESSION_PARAMS,
            "random_forest": RANDOM_FOREST_PARAMS,
            "gradient_boosting": GRADIENT_BOOSTING_PARAMS
        },
        "random_seed": RANDOM_SEED,
        "primary_metric": PRIMARY_METRIC
    }

def validate_configuration():
    """Validate configuration settings."""
    errors = []
    
    if DATASET_SIZE <= 0:
        errors.append("DATASET_SIZE must be positive")
    
    if not 0 < TEST_SIZE < 1:
        errors.append("TEST_SIZE must be between 0 and 1")
    
    if CV_FOLDS <= 1:
        errors.append("CV_FOLDS must be greater than 1")
    
    if not 0 < CHURN_PROBABILITY_THRESHOLD < 1:
        errors.append("CHURN_PROBABILITY_THRESHOLD must be between 0 and 1")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    
    return True

# Validate on import
try:
    validate_configuration()
except ValueError as e:
    print(f"Warning: {e}")

if __name__ == "__main__":
    # Print configuration summary
    import json
    print("Current Configuration:")
    print(json.dumps(get_config_summary(), indent=2, default=str))
