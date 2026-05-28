"""
Customer Churn Analysis - Complete Data Science Project
========================================================
This script covers:
1. Data Generation (synthetic telecom dataset)
2. Exploratory Data Analysis (EDA)
3. Data Preprocessing & Feature Engineering
4. Model Training (Logistic Regression, Random Forest, XGBoost)
5. Model Evaluation (ROC-AUC, Confusion Matrix, Classification Report)
6. Feature Importance
7. Churn Prediction on new data
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

np.random.seed(42)
N = 5000

def generate_churn_dataset(n=N):
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

df = generate_churn_dataset()

# ── EDA ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Customer Churn – EDA", fontsize=16, fontweight="bold")

churn_counts = df["Churn"].value_counts()
axes[0, 0].pie(churn_counts, labels=["No Churn", "Churn"],
               autopct="%1.1f%%", colors=["#4CAF50", "#F44336"],
               startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
axes[0, 0].set_title("Overall Churn Rate")

df.groupby("Churn")["Tenure"].plot(kind="hist", bins=30, alpha=0.6,
                                    ax=axes[0, 1], color=["#4CAF50", "#F44336"])
axes[0, 1].set_title("Tenure by Churn Status")
axes[0, 1].legend(["No Churn", "Churn"])

df.boxplot(column="MonthlyCharges", by="Churn", ax=axes[0, 2], patch_artist=True)
axes[0, 2].set_title("Monthly Charges by Churn")
plt.sca(axes[0, 2]); plt.suptitle("")

df.groupby("Contract")["Churn"].mean().sort_values(ascending=False).plot(
    kind="bar", ax=axes[1, 0], color=["#FF7043", "#FFA726", "#66BB6A"], edgecolor="white", rot=15)
axes[1, 0].set_title("Churn Rate by Contract Type")

df.groupby("InternetService")["Churn"].mean().sort_values(ascending=False).plot(
    kind="bar", ax=axes[1, 1], color=["#EF5350", "#AB47BC", "#42A5F5"], edgecolor="white", rot=10)
axes[1, 1].set_title("Churn Rate by Internet Service")

sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, fmt=".2f",
            cmap="RdYlGn", ax=axes[1, 2], linewidths=0.5)
axes[1, 2].set_title("Correlation Heatmap")

plt.tight_layout()
plt.show()

# ── PREPROCESSING ─────────────────────────────────────────────────
df_model = df.drop(columns=["CustomerID"]).copy()

for col in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
    df_model[col] = (df_model[col] == "Yes").astype(int)
df_model["Gender"] = (df_model["Gender"] == "Male").astype(int)
df_model = pd.get_dummies(df_model,
                           columns=["InternetService", "Contract", "PaymentMethod"],
                           drop_first=True)
df_model["ChargesPerMonth"] = df_model["TotalCharges"] / (df_model["Tenure"] + 1)
df_model["TenureGroup"]     = pd.cut(df_model["Tenure"],
                                      bins=[0, 12, 24, 48, 72],
                                      labels=[1, 2, 3, 4]).astype(int)

X = df_model.drop(columns=["Churn"])
y = df_model["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ── MODEL TRAINING ────────────────────────────────────────────────
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

print("\n==== Cross-Validation ROC-AUC (5-fold) ====")
for name, model in models.items():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    model.fit(X_train, y_train)
    print(f"  {name:25s}  AUC = {scores.mean():.4f} ± {scores.std():.4f}")

# ── ROC CURVES ───────────────────────────────────────────────────
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
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    print(f"\n── {name} ──")
    print(classification_report(y_test, model.predict(X_test), target_names=["No Churn", "Churn"]))

plt.tight_layout(); plt.show()

# ── CONFUSION MATRICES ────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
for ax, (name, model) in zip(axes, models.items()):
    ConfusionMatrixDisplay(confusion_matrix(y_test, model.predict(X_test)),
                           display_labels=["No Churn", "Churn"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=10, fontweight="bold")
plt.tight_layout(); plt.show()

# ── FEATURE IMPORTANCE ────────────────────────────────────────────
feat_imp = pd.Series(models["Random Forest"].feature_importances_, index=X.columns)
feat_imp.sort_values(ascending=True).tail(15).plot(
    kind="barh", color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, 15)), edgecolor="white",
    figsize=(9, 6), title="Top 15 Feature Importances – Random Forest")
plt.xlabel("Importance Score"); plt.tight_layout(); plt.show()

# ── PREDICT NEW CUSTOMERS ─────────────────────────────────────────
new_raw = pd.DataFrame({
    "Gender": ["Male", "Female"], "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"], "Dependents": ["No", "No"],
    "Tenure": [3, 48], "PhoneService": ["Yes", "Yes"],
    "InternetService": ["Fiber optic", "DSL"],
    "Contract": ["Month-to-month", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": ["Electronic check", "Bank transfer"],
    "MonthlyCharges": [95.5, 45.0], "TotalCharges": [286.5, 2160.0],
})

def preprocess_new(df_new, ref_cols):
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

X_new = preprocess_new(new_raw, X.columns.tolist())
proba = models["Random Forest"].predict_proba(X_new)[:, 1]
pred  = models["Random Forest"].predict(X_new)

print("\n── New Customer Predictions ──")
for i, (p, c) in enumerate(zip(proba, pred)):
    print(f"  Customer {i+1}: {p:.2%} → {'⚠️ CHURN' if c else '✅ STAY'}")