"""
Customer Churn Prediction - Complete ML Pipeline
================================================
This module implements a comprehensive machine learning solution for predicting 
customer churn using classification algorithms.

Author: Data Science Team
Date: August 2026
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline

# Set random seed for reproducibility
np.random.seed(42)
N = 5000


def generate_churn_dataset(n=N):
    """
    Generate synthetic customer dataset with realistic patterns.
    
    Parameters:
    -----------
    n : int
        Number of customer records to generate (default: 5000)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with customer data and churn labels
    """
    tenure          = np.random.randint(1, 72, n)
    monthly_charges = np.round(np.random.uniform(18, 118, n), 2)
    total_charges   = np.round(tenure * monthly_charges * np.random.uniform(0.9, 1.1, n), 2)
    gender          = np.random.choice(["Male", "Female"], n)
    senior_citizen  = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner         = np.random.choice(["Yes", "No"], n)
    dependents      = np.random.choice(["Yes", "No"], n, p=[0.7, 0.3])
    phone_service   = np.random.choice(["Yes", "No"], n, p=[0.9, 0.1])
    internet_service= np.random.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.44, 0.21])
    contract        = np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.24, 0.21])
    payment_method  = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n
    )
    paperless_billing = np.random.choice(["Yes", "No"], n, p=[0.59, 0.41])

    # Churn probability based on customer characteristics
    churn_prob = (
        0.05
        + 0.25 * (contract == "Month-to-month")
        - 0.12 * (contract == "Two year")
        + 0.15 * (internet_service == "Fiber optic")
        + 0.10 * (payment_method == "Electronic check")
        - 0.05 * (tenure > 36)
        + 0.08 * (monthly_charges > 75)
        + 0.04 * senior_citizen
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.92)
    churn = (np.random.rand(n) < churn_prob).astype(int)

    return pd.DataFrame({
        "CustomerID"       : [f"CUST-{i:05d}" for i in range(n)],
        "Gender"           : gender,
        "SeniorCitizen"    : senior_citizen,
        "Partner"          : partner,
        "Dependents"       : dependents,
        "Tenure"           : tenure,
        "PhoneService"     : phone_service,
        "InternetService"  : internet_service,
        "Contract"         : contract,
        "PaperlessBilling" : paperless_billing,
        "PaymentMethod"    : payment_method,
        "MonthlyCharges"   : monthly_charges,
        "TotalCharges"     : total_charges,
        "Churn"            : churn,
    })


def perform_eda(df):
    """
    Perform Exploratory Data Analysis with visualizations.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Customer Churn – EDA", fontsize=16, fontweight="bold")

    # Churn distribution
    churn_counts = df["Churn"].value_counts().reindex([0, 1], fill_value=0)
    axes[0, 0].pie(churn_counts, labels=["No Churn", "Churn"],
                   autopct="%1.1f%%", colors=["#4CAF50", "#F44336"],
                   startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[0, 0].set_title("Overall Churn Rate")

    # Tenure distribution
    df.groupby("Churn")["Tenure"].plot(kind="hist", bins=30, alpha=0.6,
                                        ax=axes[0, 1], color=["#4CAF50", "#F44336"])
    axes[0, 1].set_title("Tenure by Churn Status")
    axes[0, 1].legend(["No Churn", "Churn"])

    # Monthly charges
    df.boxplot(column="MonthlyCharges", by="Churn", ax=axes[0, 2], patch_artist=True)
    axes[0, 2].set_title("Monthly Charges by Churn")
    plt.sca(axes[0, 2]); plt.suptitle("")

    # Contract type
    df.groupby("Contract")["Churn"].mean().sort_values(ascending=False).plot(
        kind="bar", ax=axes[1, 0], color=["#FF7043", "#FFA726", "#66BB6A"], edgecolor="white", rot=15)
    axes[1, 0].set_title("Churn Rate by Contract Type")

    # Internet service
    df.groupby("InternetService")["Churn"].mean().sort_values(ascending=False).plot(
        kind="bar", ax=axes[1, 1], color=["#EF5350", "#AB47BC", "#42A5F5"], edgecolor="white", rot=10)
    axes[1, 1].set_title("Churn Rate by Internet Service")

    # Correlation heatmap
    sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, fmt=".2f",
                cmap="RdYlGn", ax=axes[1, 2], linewidths=0.5)
    axes[1, 2].set_title("Correlation Heatmap")

    plt.tight_layout()
    plt.show()


def preprocess_data(df):
    """
    Preprocess and engineer features for model training.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw input dataset
    
    Returns:
    --------
    tuple
        (X, y) feature matrix and target vector
    """
    df_model = df.drop(columns=["CustomerID"]).copy()

    # Binary encoding
    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        df_model[col] = (df_model[col] == "Yes").astype(int)
    df_model["Gender"] = (df_model["Gender"] == "Male").astype(int)
    
    # One-hot encoding
    df_model = pd.get_dummies(df_model,
                               columns=["InternetService", "Contract", "PaymentMethod"],
                               drop_first=True)
    
    # Feature engineering
    df_model["ChargesPerMonth"] = df_model["TotalCharges"] / (df_model["Tenure"] + 1)
    df_model["TenureGroup"]     = pd.cut(df_model["Tenure"],
                                          bins=[0, 12, 24, 48, 72],
                                          labels=[1, 2, 3, 4]).astype(int)

    X = df_model.drop(columns=["Churn"])
    y = df_model["Churn"]
    
    return X, y


def train_models(X_train, X_test, y_train, y_test):
    """
    Train and evaluate multiple classification models.
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Training and test features
    y_train, y_test : pd.Series
        Training and test labels
    
    Returns:
    --------
    dict
        Trained models dictionary
    """
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4,
            subsample=0.8, random_state=42
        ),
    }

    print("\n" + "="*50)
    print("Cross-Validation ROC-AUC (5-fold)")
    print("="*50)
    
    for name, model in models.items():
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        model.fit(X_train, y_train)
        print(f"  {name:25s}  AUC = {scores.mean():.4f} ± {scores.std():.4f}")

    return models


def evaluate_models(models, X_test, y_test):
    """
    Evaluate models with ROC curves and classification reports.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Model Evaluation – ROC Curves", fontsize=15, fontweight="bold")

    for ax, (name, model), color in zip(axes, models.items(), ["#3498DB", "#2ECC71", "#E74C3C"]):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, color=color, lw=2.5, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.fill_between(fpr, tpr, alpha=0.08, color=color)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(loc="lower right")
        
        print(f"\n{'─'*50}")
        print(f"{name}")
        print(f"{'─'*50}")
        print(classification_report(y_test, model.predict(X_test), target_names=["No Churn", "Churn"]))

    plt.tight_layout()
    plt.show()


def plot_confusion_matrices(models, X_test, y_test):
    """
    Plot confusion matrices for all models.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test labels
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    
    for ax, (name, model) in zip(axes, models.items()):
        ConfusionMatrixDisplay(confusion_matrix(y_test, model.predict(X_test)),
                               display_labels=["No Churn", "Churn"]).plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    plt.show()


def plot_feature_importance(models, X):
    """
    Plot feature importance from Random Forest model.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X : pd.DataFrame
        Feature matrix
    """
    feat_imp = pd.Series(models["Random Forest"].feature_importances_, index=X.columns)
    feat_imp.sort_values(ascending=True).tail(15).plot(
        kind="barh", color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, 15)), edgecolor="white",
        figsize=(9, 6), title="Top 15 Feature Importances – Random Forest")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()


def predict_new_customers(models, X_columns):
    """
    Predict churn for new customers.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_columns : list
        List of feature column names for reference
    """
    new_raw = pd.DataFrame({
        "Gender": ["Male", "Female"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "No"],
        "Tenure": [3, 48],
        "PhoneService": ["Yes", "Yes"],
        "InternetService": ["Fiber optic", "DSL"],
        "Contract": ["Month-to-month", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Bank transfer"],
        "MonthlyCharges": [95.5, 45.0],
        "TotalCharges": [286.5, 2160.0],
    })

    X_new = preprocess_new(new_raw, X_columns)
    proba = models["Random Forest"].predict_proba(X_new)[:, 1]
    pred  = models["Random Forest"].predict(X_new)

    print("\n" + "="*50)
    print("New Customer Predictions")
    print("="*50)
    for i, (p, c) in enumerate(zip(proba, pred)):
        status = "CHURN" if c else "STAY"
        print(f"  Customer {i + 1}: {p:.2%} -> {status}")


def preprocess_new(df_new, ref_cols):
    """
    Preprocess new customer data using same transformations as training.
    
    Parameters:
    -----------
    df_new : pd.DataFrame
        Raw new customer data
    ref_cols : list
        Reference column names from training data
    
    Returns:
    --------
    pd.DataFrame
        Preprocessed features aligned with training data
    """
    df_new = df_new.copy()
    
    for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        df_new[col] = (df_new[col] == "Yes").astype(int)
    df_new["Gender"] = (df_new["Gender"] == "Male").astype(int)
    df_new = pd.get_dummies(df_new, columns=["InternetService", "Contract", "PaymentMethod"], drop_first=True)
    df_new["ChargesPerMonth"] = df_new["TotalCharges"] / (df_new["Tenure"] + 1)
    df_new["TenureGroup"] = pd.cut(df_new["Tenure"], bins=[0,12,24,48,72], labels=[1,2,3,4]).astype(int)
    
    for col in ref_cols:
        if col not in df_new.columns:
            df_new[col] = 0
    
    return df_new[ref_cols]


def main():
    """
    Execute the complete machine learning pipeline.
    """
    print("\n" + "="*60)
    print("CUSTOMER CHURN PREDICTION - COMPLETE ML PIPELINE")
    print("="*60)
    
    # Generate dataset
    print("\n[1/7] Generating synthetic dataset...")
    df = generate_churn_dataset()
    print(f"      Generated {len(df)} customer records")
    
    # EDA
    print("\n[2/7] Performing exploratory data analysis...")
    perform_eda(df)
    
    # Preprocessing
    print("\n[3/7] Preprocessing and engineering features...")
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"      Training set: {len(X_train)} samples")
    print(f"      Test set: {len(X_test)} samples")
    
    # Model training
    print("\n[4/7] Training classification models...")
    models = train_models(X_train, X_test, y_train, y_test)
    
    # Model evaluation
    print("\n[5/7] Evaluating models...")
    evaluate_models(models, X_test, y_test)
    
    # Confusion matrices
    print("\n[6/7] Plotting confusion matrices...")
    plot_confusion_matrices(models, X_test, y_test)
    
    # Feature importance
    print("\n[7/7] Analyzing feature importance...")
    plot_feature_importance(models, X)
    
    # Predictions
    print("\nMaking predictions on new customers...")
    predict_new_customers(models, X.columns.tolist())
    
    print("\n" + "="*60)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
