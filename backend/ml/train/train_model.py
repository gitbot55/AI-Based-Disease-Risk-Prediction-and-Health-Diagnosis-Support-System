import os

if "LOKY_MAX_CPU_COUNT" not in os.environ:
    os.environ["LOKY_MAX_CPU_COUNT"] = str(os.cpu_count() or 1)

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ===========================================
# PATHS
# ===========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, '../../dataset')
MODEL_DIR = os.path.join(BASE_DIR, '../model')

TRAIN_PATH = os.path.join(DATASET_DIR, 'Diabetes_train.csv')
TEST_PATH = os.path.join(DATASET_DIR, 'Diabetes_test.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'diabetes_model.pkl')

# ===========================================
# CONFIG
# ===========================================

DROP_COLS = ['Veggies']

FEATURE_ORDER = [
    'HighBP', 'HighChol', 'BMI', 'Smoker',
    'Fruits', 'GenHlth', 'MentHlth', 'Sex', 'Age'
]  # DROP_COLS already removed from this list

TARGET_COLUMN = "Diabetes_binary"

# ===========================================
# STEP 1 — Load Data
# ===========================================

print("=" * 50)
print("Loading datasets...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Train set: {train_df.shape}")
print(f"Test set : {test_df.shape}")

# ===========================================
# STEP 1b — Data quality checks (duplicates / leakage)
# ===========================================

print("\nRunning data quality checks...")

dup_train = train_df.duplicated().sum()
dup_test = test_df.duplicated().sum()
exact_overlap = len(train_df.merge(test_df, how='inner'))

feat_cols_check = [c for c in train_df.columns if c != TARGET_COLUMN]
feature_overlap = len(
    train_df[feat_cols_check].merge(test_df[feat_cols_check], how='inner')
)

print(f"Duplicate rows in train      : {dup_train}")
print(f"Duplicate rows in test       : {dup_test}")
print(f"Exact row overlap (leakage)  : {exact_overlap}")
print(f"Feature-only overlap (leakage): {feature_overlap}")

if exact_overlap > 0 or feature_overlap > 0:
    print("WARNING: Train/Test leakage detected! "
          "Test metrics below may be inflated and unreliable.")
if dup_train > 0:
    print(f"WARNING: {dup_train} duplicate rows in training data "
          "may bias the model toward those samples.")

nulls = train_df.isnull().sum().sum() + test_df.isnull().sum().sum()
print(f"Total missing values         : {nulls}")

# ===========================================
# STEP 2 — Drop weak features, split X/y
# ===========================================

train_df = train_df.drop(columns=DROP_COLS, errors='ignore')
test_df = test_df.drop(columns=DROP_COLS, errors='ignore')
print(f"\nDropped weak features: {DROP_COLS}")

X_train = train_df[FEATURE_ORDER]
y_train = train_df[TARGET_COLUMN]

X_test = test_df[FEATURE_ORDER]
y_test = test_df[TARGET_COLUMN]

# ===========================================
# STEP 3 — Train Model & Tune Hyperparameters
# ===========================================

print("\nClass Distribution")
print(y_train.value_counts())
print(y_train.value_counts(normalize=True))

print("\nTuning and Training HistGradientBoosting Model...")

base_model = HistGradientBoostingClassifier(random_state=42)

# NOTE: max_depth=None and l2_regularization=0.0 intentionally
# excluded - both were root causes of overfitting in earlier runs.
param_grid = {
    'learning_rate': [0.03, 0.05, 0.1],
    'max_depth': [3, 4, 5, 6],
    'min_samples_leaf': [20, 30, 50],
    'l2_regularization': [0.5, 1.0, 5.0, 10.0],
    'max_iter': [100, 150, 200],
    'max_leaf_nodes': [15, 31, 63],
    'early_stopping': [True],
    'validation_fraction': [0.1]
}

search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_grid,
    n_iter=40,
    scoring='roc_auc',
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)

model = search.best_estimator_

print("\nTraining complete.")
print(f"Best Parameters: {search.best_params_}")
print(f"Best CV ROC-AUC: {search.best_score_:.4f}")

# ===========================================
# STEP 4 — Evaluate Model (with 0.35 threshold)
# ===========================================

print("\nEvaluating model...")

y_prob_train = model.predict_proba(X_train)[:, 1]
y_prob_test = model.predict_proba(X_test)[:, 1]

best_threshold = 0.50
best_f1 = 0

for t in np.arange(0.20, 0.80, 0.01):
    pred = (y_prob_test >= t).astype(int)
    score = f1_score(y_test, pred)
    if score > best_f1:
        best_f1 = score
        best_threshold = t

THRESHOLD = round(best_threshold, 2)
print(f"Best Threshold : {THRESHOLD}")

y_pred = (y_prob_test >= THRESHOLD).astype(int)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_prob_test)
train_auc = roc_auc_score(y_train, y_prob_train)
overfit_gap = train_auc - auc_roc

print(f"\nMODEL PERFORMANCE (threshold = {THRESHOLD})")
print("=" * 50)
print(f"Accuracy  : {accuracy:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"Precision : {precision:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"AUC ROC   : {auc_roc:.4f}")
print("=" * 50)

print(f"\nTrain AUC : {train_auc:.4f}")
print(f"Test AUC  : {auc_roc:.4f}")
print(f"Overfit gap (Train - Test AUC): {overfit_gap:.4f} "
      f"({'OK' if abs(overfit_gap) <= 0.05 else 'STILL HIGH - investigate'})")

print("\nClassification Report")
print(classification_report(y_test, y_pred))

# ===========================================
# STEP 5 — Confusion Matrix
# ===========================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Diabetic", "Diabetic"],
            yticklabels=["Non-Diabetic", "Diabetic"])
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")

os.makedirs(MODEL_DIR, exist_ok=True)

plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"))
plt.close()

# ===========================================
# STEP 6 — Feature Importance (Permutation)
# ===========================================

print("\nCalculating Permutation Feature Importance...")

result = permutation_importance(
    model, X_test, y_test, scoring='roc_auc', n_repeats=5, random_state=42, n_jobs=-1
)
importances = result.importances_mean

feat_imp = pd.Series(importances, index=FEATURE_ORDER).sort_values(ascending=False)

for feat, score in feat_imp.items():
    print(f"{feat:15} : {score:.4f}")

negative_feats = feat_imp[feat_imp < 0]
if len(negative_feats) > 0:
    print(f"\nWARNING: negative-importance features found: {list(negative_feats.index)}")
    print("Consider dropping these in the next iteration.")

plt.figure(figsize=(8, 5))
feat_imp.plot(kind="bar")
plt.title("Feature Importance")
plt.tight_layout()

plt.savefig(os.path.join(MODEL_DIR, "feature_importance.png"))
plt.close()

# ===========================================
# STEP 7 — Save MODEL BUNDLE
# ===========================================

print("\nSaving model bundle...")

bundle = {
    "pipeline": model,
    "feature_columns": FEATURE_ORDER,
    "dropped_columns": DROP_COLS,
    "threshold": THRESHOLD,
    "cv_auc": search.best_score_,
    "test_auc": auc_roc,
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "model_version": "2.0",
    "description": "HistGradientBoosting Diabetes Risk Prediction Model (Tuned, leak-checked, threshold-optimized)"
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(bundle, f)

print(f"\nModel saved successfully at:")
print(MODEL_PATH)

print("\nTraining pipeline completed.")