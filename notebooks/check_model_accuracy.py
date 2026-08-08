"""
Prints model accuracy metrics for the ISCDF disruption model, evaluated on
the held-out test predictions in outputs/predictions.csv (y_true vs y_prob).

Run: python notebooks/check_model_accuracy.py
"""
import os

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pred_path = os.path.join(ROOT, "outputs", "predictions.csv")

df = pd.read_csv(pred_path)
y_true = df["y_true"]
y_prob = df["y_prob"]
y_pred = (y_prob >= 0.5).astype(int)

print(f"Evaluated on {len(df)} held-out shipments ({pred_path})")
print("-" * 50)
print(f"ROC-AUC   : {roc_auc_score(y_true, y_prob):.3f}")
print(f"PR-AUC    : {average_precision_score(y_true, y_prob):.3f}")
print(f"Accuracy  : {accuracy_score(y_true, y_pred):.3f}")
print(f"Precision : {precision_score(y_true, y_pred):.3f}")
print(f"Recall    : {recall_score(y_true, y_pred):.3f}")
print(f"F1        : {f1_score(y_true, y_pred):.3f}")
print("-" * 50)
print("Confusion matrix [rows=actual, cols=predicted]:")
print(confusion_matrix(y_true, y_pred))
print("-" * 50)
print(classification_report(y_true, y_pred, target_names=["No Disruption", "Disruption"]))
print("-" * 50)
print("Mean predicted risk by actual outcome:")
print(df.groupby("y_true")["y_prob"].mean().rename({0: "No Disruption", 1: "Disruption"}))
if "risk_band" in df.columns:
    print()
    print("Actual disruption rate by risk band (should increase Low -> Medium -> High):")
    print(df.groupby("risk_band")["y_true"].mean().sort_values())
