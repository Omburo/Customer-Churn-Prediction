# Customer Churn Prediction using Machine Learning
# Models compared: KNN, Decision Tree, SVM, ANN (MLPClassifier)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
)

RANDOM_STATE = 42
sns.set(style="whitegrid")

# ---------------------------------------------------------------
# STEP 1: LOAD DATA
# ---------------------------------------------------------------
df = pd.read_csv("telco.csv")
print(df.head())
print(df.info())
print("Initial shape:", df.shape)

# ---------------------------------------------------------------
# STEP 2: DATA PREPROCESSING
# ---------------------------------------------------------------
df = df.drop(columns=["customerID"])  # identifier, not predictive

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("Missing values in TotalCharges:", df["TotalCharges"].isnull().sum())
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
print("Missing values in TotalCharges after filling:", df["TotalCharges"].isnull().sum())

# Target: Yes/No -> 1/0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Simple Yes/No columns -> 1/0
binary_columns = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
for col in binary_columns:
    df[col] = df[col].map({"Yes": 1, "No": 0})

df["gender"] = df["gender"].map({"Male": 1, "Female": 0})

# One-hot encode remaining multi-category columns
multi_category_columns = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod"
]
df = pd.get_dummies(df, columns=multi_category_columns, drop_first=True)
print("Final shape after preprocessing:", df.shape)
print(df.head())

# ---------------------------------------------------------------
# STEP 3: TRAIN/TEST SPLIT + SCALING
# ---------------------------------------------------------------
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print("Train size:", X_train.shape)
print("Test size:", X_test.shape)
print("Churn rate - train:", y_train.mean(), "| test:", y_test.mean())

# Scaling is required for KNN, SVM, and ANN (all distance/gradient based).
# Decision Tree does not need it, since it only compares raw threshold values.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# STEP 4a: TRAIN KNN
# ---------------------------------------------------------------
knn_param_grid = {"n_neighbors": list(range(1, 31, 2))}

knn_grid = GridSearchCV(
    KNeighborsClassifier(), knn_param_grid, cv=2, scoring="f1", n_jobs=-1
)
knn_grid.fit(X_train_scaled, y_train)

best_knn = knn_grid.best_estimator_
print("\nBest k:", knn_grid.best_params_["n_neighbors"])
print("Best KNN CV F1 score:", knn_grid.best_score_)

# ---------------------------------------------------------------
# STEP 4b: TRAIN DECISION TREE
# ---------------------------------------------------------------
dt_param_grid = {
    "max_depth": list(range(2, 15)),
    "min_samples_leaf": [1, 5, 10, 20],
}

dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE), dt_param_grid,
    cv=2, scoring="f1", n_jobs=-1
)
dt_grid.fit(X_train, y_train)  # unscaled — trees don't need scaling

best_dt = dt_grid.best_estimator_
print("\nBest Decision Tree Params:", dt_grid.best_params_)
print("Best Decision Tree CV F1 score:", dt_grid.best_score_)

# ---------------------------------------------------------------
# STEP 4c: TRAIN SVM
# ---------------------------------------------------------------
# probability=True is required to get predict_proba() for ROC AUC.
# Kept the grid small since probability calibration makes SVC slower to fit.
svm_param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["rbf", "linear"],
    "gamma": ["scale"],
}

svm_grid = GridSearchCV(
    SVC(probability=True, random_state=RANDOM_STATE), svm_param_grid,
    cv=2, scoring="f1", n_jobs=-1
)
svm_grid.fit(X_train_scaled, y_train)  # scaled — SVM is distance-based

best_svm = svm_grid.best_estimator_
print("\nBest SVM Params:", svm_grid.best_params_)
print("Best SVM CV F1 score:", svm_grid.best_score_)

# ---------------------------------------------------------------
# STEP 4d: TRAIN ANN (MLPClassifier)
# ---------------------------------------------------------------
# A small feed-forward neural network. max_iter is raised so training
# actually converges instead of stopping early with a warning.
mlp_param_grid = {
    "hidden_layer_sizes": [(50,), (100,), (50, 25)],
    "alpha": [0.0001, 0.001, 0.01],  # L2 regularization strength
}

mlp_grid = GridSearchCV(
    MLPClassifier(max_iter=500, random_state=RANDOM_STATE, early_stopping=True),
    mlp_param_grid, cv=2, scoring="f1", n_jobs=-1
)
mlp_grid.fit(X_train_scaled, y_train)  # scaled — neural nets train much better on scaled input

best_mlp = mlp_grid.best_estimator_
print("\nBest ANN Params:", mlp_grid.best_params_)
print("Best ANN CV F1 score:", mlp_grid.best_score_)

# ---------------------------------------------------------------
# STEP 5: EVALUATE ALL MODELS ON THE TEST SET
# ---------------------------------------------------------------
def evaluate_model(name, model, X_te, y_te):
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1]
    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_te, y_pred),
        "Precision": precision_score(y_te, y_pred),
        "Recall": recall_score(y_te, y_pred),
        "F1": f1_score(y_te, y_pred),
        "ROC AUC": roc_auc_score(y_te, y_proba),
    }
    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        if k != "Model":
            print(f"{k}: {v:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_te, y_pred))
    print("Classification Report:\n",
          classification_report(y_te, y_pred, target_names=["Stay", "Churn"]))
    return metrics, y_pred, y_proba

results = []
predictions = {}

m, pred, proba = evaluate_model("KNN", best_knn, X_test_scaled, y_test)
results.append(m); predictions["KNN"] = (pred, proba)

m, pred, proba = evaluate_model("Decision Tree", best_dt, X_test, y_test)
results.append(m); predictions["Decision Tree"] = (pred, proba)

m, pred, proba = evaluate_model("SVM", best_svm, X_test_scaled, y_test)
results.append(m); predictions["SVM"] = (pred, proba)

m, pred, proba = evaluate_model("ANN (MLP)", best_mlp, X_test_scaled, y_test)
results.append(m); predictions["ANN (MLP)"] = (pred, proba)

results_df = pd.DataFrame(results).set_index("Model")
print("\n=== Model Comparison ===")
print(results_df.round(4).to_string())
results_df.round(4).to_csv("model_comparison.csv")

# ---------------------------------------------------------------
# STEP 6: VISUALIZATIONS
# ---------------------------------------------------------------

# Confusion matrices — 2x2 grid, one per model
fig, axes = plt.subplots(2, 2, figsize=(11, 10))
for ax, name in zip(axes.flatten(), predictions.keys()):
    pred, _ = predictions[name]
    cm = confusion_matrix(y_test, pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Stay", "Churn"], yticklabels=["Stay", "Churn"])
    ax.set_title(f"{name} — Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.close()

# ROC curves — all four models on one plot
plt.figure(figsize=(8, 6))
for name in predictions.keys():
    _, proba = predictions[name]
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random Classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — KNN vs Decision Tree vs SVM vs ANN")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curves.png", dpi=150)
plt.close()

# Feature importance — Decision Tree only (the only model here with a
# direct, built-in feature_importances_ attribute)
importances = pd.Series(best_dt.feature_importances_, index=X_train.columns)
top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
top_features.sort_values().plot(kind="barh", color="skyblue")
plt.title("Top 10 Feature Importances — Decision Tree")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importances.png", dpi=150)
plt.close()

# Model comparison bar chart (F1 score, easy at-a-glance ranking)
plt.figure(figsize=(8, 5))
results_df["F1"].sort_values().plot(kind="barh", color="teal")
plt.title("Model Comparison — F1 Score (Churn class)")
plt.xlabel("F1 Score")
plt.tight_layout()
plt.savefig("model_comparison_f1.png", dpi=150)
plt.close()

print("\nSaved: confusion_matrices.png, roc_curves.png, feature_importances.png, "
      "model_comparison_f1.png, model_comparison.csv")
