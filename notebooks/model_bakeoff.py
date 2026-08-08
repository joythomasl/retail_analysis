"""
Trains the 5 candidate models originally proposed in
notebooks/02_ml_pipeline.ipynb's "5-Model Bakeoff" cell -- Logistic
Regression, Decision Tree, Random Forest, AdaBoost, HistGradientBoosting --
on the real dataset (data/raw/train.parquet), using the identical features,
chronological train/test split, and categorical encoding as
notebooks/rebuild_pipeline.py, so this is an apples-to-apples comparison
against the model actually deployed to outputs/.

Run: python notebooks/model_bakeoff.py
"""
import os
import time

import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(ROOT, "data", "raw", "train.parquet")

if not os.path.exists(RAW_PATH):
    raise SystemExit(
        f"Real dataset not found at {RAW_PATH}.\n"
        "Run `python notebooks/rebuild_pipeline.py` first -- it needs the same file."
    )

df = pd.read_parquet(RAW_PATH).sort_values("order_date").reset_index(drop=True)

NUMERIC = [
    "customs_clearance_hours", "demand_forecast_units", "distance_km", "fuel_price_index",
    "geopolitical_risk_index", "historical_disruption_count", "inventory_level_percent",
    "num_alternate_suppliers", "order_quantity", "order_value_usd", "payment_terms_days",
    "planned_lead_time_days", "supplier_financial_health_score", "supplier_reliability_score",
    "unit_cost_usd", "weather_risk_index",
]
CAT_COLS = [
    "carrier_name", "contract_type", "destination_city", "destination_country",
    "port_congestion_level", "product_category", "product_name", "supplier_country",
    "supplier_id", "transportation_mode", "warehouse_id", "is_peak_season",
]
FEATURES = NUMERIC + CAT_COLS
TARGET = "disruption_occurred"
df[TARGET] = df[TARGET].astype(int)

split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
encoder.fit(train_df[CAT_COLS].astype(str))

X_train = train_df[FEATURES].copy()
X_test = test_df[FEATURES].copy()
X_train[CAT_COLS] = encoder.transform(train_df[CAT_COLS].astype(str))
X_test[CAT_COLS] = encoder.transform(test_df[CAT_COLS].astype(str))
y_train = train_df[TARGET]
y_test = test_df[TARGET]

print(f"Train: {len(X_train):,} rows   Test: {len(X_test):,} rows   Features: {len(FEATURES)}")
print("-" * 70)

MODELS = {
    "Logistic Regression (Baseline)": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "HistGradientBoosting (Champion)": HistGradientBoostingClassifier(random_state=42),
}

results = []
for name, model in MODELS.items():
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    results.append(
        {
            "Model": name,
            "ROC-AUC": round(roc_auc_score(y_test, prob), 3),
            "PR-AUC": round(average_precision_score(y_test, prob), 3),
            "Accuracy": round(accuracy_score(y_test, pred), 3),
            "Train Time (s)": round(train_time, 2),
        }
    )
    print(f"  {name}: done in {train_time:.1f}s")

leaderboard = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
print()
print("MODEL BAKEOFF RESULTS (sorted by ROC-AUC)")
print("-" * 70)
print(leaderboard.to_string(index=False))
