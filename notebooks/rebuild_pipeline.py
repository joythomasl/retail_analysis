"""
Rebuilds the ISCDF ML pipeline end-to-end.

Prefers the real dataset at data/raw/train.parquet (158,551 real shipments,
2022-2024, 43.3% disruption rate) when present. That file is gitignored
(data/ is excluded) so it won't exist on a fresh clone -- in that case this
falls back to a synthetic generator that keeps the same property the real
data has (a genuine, noisy, feature-driven relationship to the target),
instead of the original notebook's "ghost data" bug where the target was
sampled fully independently of every feature (ROC-AUC ~0.46-0.50, worse
than a coin flip).

Columns intentionally EXCLUDED as features when training on real data,
because they leak the outcome (only knowable after the shipment resolves):
  - risk_score            (post-hoc: averages ~19.5 when no disruption vs
                            ~74.7 when disrupted -- clearly derived FROM the
                            outcome, not a predictor of it)
  - disruption_type       (only non-"None" when a disruption occurred)
  - delay_days            (the outcome itself, in days)
  - actual_delivery_date  (only known after delivery)
  - is_degraded_record / n_missing_fields (dataset-construction metadata,
                            not a real-world business signal)

Run: python notebooks/rebuild_pipeline.py
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
import shap
from datetime import datetime, timedelta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.preprocessing import OrdinalEncoder

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "processed")
OUT_DIR = os.path.join(ROOT, "outputs")
RAW_PATH = os.path.join(ROOT, "data", "raw", "train.parquet")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

USE_REAL = os.path.exists(RAW_PATH)


# ═══════════════════════════════════════════════════════════════════════════
# Dataset construction
# ═══════════════════════════════════════════════════════════════════════════
def load_real_dataset():
    df = pd.read_parquet(RAW_PATH).sort_values("order_date").reset_index(drop=True)

    numeric = [
        "customs_clearance_hours", "demand_forecast_units", "distance_km", "fuel_price_index",
        "geopolitical_risk_index", "historical_disruption_count", "inventory_level_percent",
        "num_alternate_suppliers", "order_quantity", "order_value_usd", "payment_terms_days",
        "planned_lead_time_days", "supplier_financial_health_score", "supplier_reliability_score",
        "unit_cost_usd", "weather_risk_index",
    ]
    cat_cols = [
        "carrier_name", "contract_type", "destination_city", "destination_country",
        "port_congestion_level", "product_category", "product_name", "supplier_country",
        "supplier_id", "transportation_mode", "warehouse_id", "is_peak_season",
    ]
    features = numeric + cat_cols
    target = "disruption_occurred"
    df[target] = df[target].astype(int)
    return df, features, cat_cols, target


def build_synthetic_dataset():
    """Fallback used only when data/raw/train.parquet isn't present."""
    rng = np.random.default_rng(42)
    n = 2000
    dates = [datetime(2026, 1, 1) + timedelta(hours=12 * i) for i in range(n)]

    supplier_id = rng.choice(["S1", "S2", "S3"], n)
    supplier_name = pd.Series(supplier_id).map(
        {"S1": "TechCorp", "S2": "GlobalSupply", "S3": "AeroParts"}
    ).to_numpy()
    destination_city = rng.choice(["Los Angeles", "New York", "Frankfurt"], n)
    transportation_mode = rng.choice(["Sea", "Air", "Rail"], n)
    warehouse_id = rng.choice(["WH_A", "WH_B"], n)
    port_congestion_level_e = rng.choice(["Low", "Medium", "High"], n, p=[0.5, 0.3, 0.2])
    sea_x_congestion = rng.choice(["Low", "High"], n, p=[0.7, 0.3])
    sourcing_fragility = rng.choice(["Low", "High"], n, p=[0.7, 0.3])
    customs_clearance_hours = rng.choice(["Low", "High"], n, p=[0.7, 0.3])
    ext_risk = rng.choice(["Low", "High"], n, p=[0.7, 0.3])
    supplier_id_hist = rng.beta(2, 5, n)
    order_value_usd = rng.uniform(5000, 85000, n).round(2)

    logit = (
        -4.0
        + np.where(port_congestion_level_e == "Medium", 1.0, 0.0)
        + np.where(port_congestion_level_e == "High", 2.0, 0.0)
        + np.where(sourcing_fragility == "High", 1.8, 0.0)
        + np.where(ext_risk == "High", 1.6, 0.0)
        + np.where(sea_x_congestion == "High", 1.0, 0.0)
        + np.where(customs_clearance_hours == "High", 1.0, 0.0)
        + 2.2 * supplier_id_hist
        + np.where(transportation_mode == "Sea", 0.3, np.where(transportation_mode == "Air", -0.3, 0.0))
        + rng.normal(0, 1.0, n)
    )
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    delay_days = np.where(y == 1, rng.integers(1, 15, n), 0)
    disruption_type = np.where(
        y == 1, rng.choice(["Weather", "Customs", "Port Congestion", "Supplier Failure"], n), "None"
    )
    planned_delivery_date = [d + timedelta(days=14) for d in dates]
    actual_delivery_date = [
        pdd + timedelta(days=int(dd)) for pdd, dd in zip(planned_delivery_date, delay_days)
    ]

    df = (
        pd.DataFrame(
            {
                "shipment_id": [f"SC-{1000+i}" for i in range(n)],
                "order_date": dates,
                "supplier_name": supplier_name,
                "supplier_id": supplier_id,
                "planned_delivery_date": planned_delivery_date,
                "actual_delivery_date": actual_delivery_date,
                "destination_city": destination_city,
                "transportation_mode": transportation_mode,
                "order_value_usd": order_value_usd,
                "warehouse_id": warehouse_id,
                "port_congestion_level_e": port_congestion_level_e,
                "sea_x_congestion": sea_x_congestion,
                "sourcing_fragility": sourcing_fragility,
                "customs_clearance_hours": customs_clearance_hours,
                "ext_risk": ext_risk,
                "supplier_id_hist": supplier_id_hist,
                "y": y,
                "delay_days": delay_days,
                "disruption_type": disruption_type,
                "notes": "synthetic (rebuild_pipeline.py fallback -- data/raw/train.parquet not found)",
            }
        )
        .sort_values("order_date")
        .reset_index(drop=True)
    )
    df.to_parquet(os.path.join(DATA_DIR, "base_table.parquet"), index=False)

    features = [
        "supplier_id", "destination_city", "transportation_mode", "order_value_usd",
        "warehouse_id", "port_congestion_level_e", "sea_x_congestion", "sourcing_fragility",
        "customs_clearance_hours", "ext_risk", "supplier_id_hist",
    ]
    cat_cols = [
        "supplier_id", "destination_city", "transportation_mode", "warehouse_id",
        "port_congestion_level_e", "sea_x_congestion", "sourcing_fragility",
        "customs_clearance_hours", "ext_risk",
    ]
    return df, features, cat_cols, "y"


if USE_REAL:
    print(f"Found real dataset at {RAW_PATH} -- training on real data.")
    df, FEATURES, CAT_COLS, TARGET = load_real_dataset()
else:
    print(f"No real dataset at {RAW_PATH} -- falling back to synthetic data.")
    df, FEATURES, CAT_COLS, TARGET = build_synthetic_dataset()

print(f"{len(df)} rows. Positive rate: {df[TARGET].mean():.1%}")

# ── Chronological train/test split + categorical encoding ──────────────────
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
encoder.fit(train_df[CAT_COLS].astype(str))

# Per-feature stats so any UI (e.g. app/app.py's What-If Simulator) can build
# its inputs generically from whatever FEATURES/CAT_COLS this run produced,
# instead of hardcoding column names that break the moment the schema changes
# (this is exactly what happened when the real dataset replaced the synthetic
# one: the simulator's hardcoded synthetic-schema fields silently mapped to
# nothing, and defaulting missing fields to 0 crashed the encoder on a mix of
# int and str in the same column).
feature_stats = {}
for col in FEATURES:
    if col in CAT_COLS:
        vals = train_df[col].astype(str)
        feature_stats[col] = {
            "kind": "categorical",
            "options": sorted(vals.unique().tolist()),
            "default": vals.mode().iloc[0],
        }
    else:
        vals = train_df[col].astype(float)
        feature_stats[col] = {
            "kind": "numeric",
            "min": float(vals.min()),
            "max": float(vals.max()),
            "default": float(vals.median()),
        }
joblib.dump(feature_stats, os.path.join(OUT_DIR, "feature_stats.pkl"))

X_train = train_df[FEATURES].copy()
X_test = test_df[FEATURES].copy()
X_train[CAT_COLS] = encoder.transform(train_df[CAT_COLS].astype(str))
X_test[CAT_COLS] = encoder.transform(test_df[CAT_COLS].astype(str))
y_train = train_df[TARGET].astype(int)
y_test = test_df[TARGET].astype(int)

# ── Train baseline + tuned models ───────────────────────────────────────────
baseline = HistGradientBoostingClassifier(random_state=42)
baseline.fit(X_train, y_train)

param_distributions = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "max_iter": [100, 200, 300],
    "max_depth": [3, 5, 7, None],
    "min_samples_leaf": [10, 20, 30],
}
search = RandomizedSearchCV(
    estimator=HistGradientBoostingClassifier(random_state=42),
    param_distributions=param_distributions,
    n_iter=15,
    scoring="roc_auc",
    cv=TimeSeriesSplit(n_splits=3),
    n_jobs=-1,
    random_state=42,
)
search.fit(X_train, y_train)
tuned = search.best_estimator_

for name, model in [("baseline", baseline), ("tuned", tuned)]:
    prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    ap = average_precision_score(y_test, prob)
    print(f"{name}: ROC-AUC={auc:.3f}  PR-AUC={ap:.3f}")

print("Best params:", json.dumps(search.best_params_, indent=2))

joblib.dump(baseline, os.path.join(OUT_DIR, "model.pkl"))
joblib.dump(tuned, os.path.join(OUT_DIR, "model_tuned.pkl"))
joblib.dump(encoder, os.path.join(OUT_DIR, "encoder.pkl"))
joblib.dump(FEATURES, os.path.join(OUT_DIR, "features.pkl"))

# ── SHAP top-driver + rules-based recommended action -> predictions.csv ────
# shap.Explainer auto-selects the fast, exact TreeExplainer for this model
# type (HistGradientBoostingClassifier), not the slow permutation fallback.
background = X_train.sample(min(2000, len(X_train)), random_state=42)
explainer = shap.Explainer(tuned, background)
shap_values = explainer(X_test, check_additivity=False)
shap_abs = np.abs(shap_values.values)
top_driver_idx = np.argmax(shap_abs, axis=1)
feature_names = np.array(FEATURES)
driver_1 = feature_names[top_driver_idx]

y_prob = tuned.predict_proba(X_test)[:, 1]
risk_score = (y_prob * 100).round().astype(int)
risk_band = pd.cut(risk_score, bins=[-1, 33, 66, 100], labels=["Low", "Medium", "High"]).astype(str)

if USE_REAL:
    # "High"/"Low" cutoffs for continuous drivers use the 75th/25th percentile
    # of the real training data as the "elevated" threshold.
    RULES = {
        ("port_congestion_level", "High"): "Pre-position stock; consider air freight",
        ("weather_risk_index", "High"): "Increase safety stock buffer; monitor weather corridor",
        ("geopolitical_risk_index", "High"): "Monitor geopolitical corridor; evaluate reroute options",
        ("customs_clearance_hours", "High"): "Pre-clear documentation; engage customs broker",
        ("historical_disruption_count", "High"): "Escalate to procurement; request delivery confirmation",
        ("supplier_reliability_score", "Low"): "Dual-source this lane; qualify a backup supplier",
        ("num_alternate_suppliers", "Low"): "Qualify additional backup suppliers for this lane",
        ("transportation_mode", "Sea"): "Evaluate switching to Air or Rail for this shipment",
    }
    THRESH = {
        "weather_risk_index": train_df["weather_risk_index"].quantile(0.75),
        "geopolitical_risk_index": train_df["geopolitical_risk_index"].quantile(0.75),
        "customs_clearance_hours": train_df["customs_clearance_hours"].quantile(0.75),
        "historical_disruption_count": train_df["historical_disruption_count"].quantile(0.75),
        "supplier_reliability_score": train_df["supplier_reliability_score"].quantile(0.25),
        "num_alternate_suppliers": train_df["num_alternate_suppliers"].quantile(0.25),
    }

    def driver_level(d1, row):
        if d1 == "port_congestion_level":
            return row[d1]
        if d1 == "transportation_mode":
            return row[d1]
        if d1 == "supplier_reliability_score":
            return "Low" if row[d1] <= THRESH[d1] else "High"
        if d1 == "num_alternate_suppliers":
            return "Low" if row[d1] <= THRESH[d1] else "High"
        if d1 in THRESH:
            return "High" if row[d1] >= THRESH[d1] else "Low"
        return None
else:
    RULES = {
        ("port_congestion_level_e", "High"): "Pre-position stock; consider air freight",
        ("sea_x_congestion", "High"): "Switch transport mode to Air or Rail",
        ("sourcing_fragility", "High"): "Dual-source this lane; qualify a backup supplier",
        ("ext_risk", "High"): "Increase safety stock buffer; monitor weather corridor",
        ("supplier_id_hist", "High"): "Escalate to procurement; request delivery confirmation",
        ("customs_clearance_hours", "High"): "Pre-clear documentation; engage customs broker",
    }

    def driver_level(d1, row):
        if d1 in CAT_COLS:
            return row[d1]
        if d1 == "supplier_id_hist":
            return "High" if row[d1] >= 0.5 else "Low"
        return None

DEFAULT = {
    "High": "Escalate to procurement and confirm next two deliveries",
    "Medium": "Increase check-in frequency to twice weekly",
    "Low": "Standard monitoring — no action required",
}

recommended_action = []
for i, d1 in enumerate(driver_1):
    row = test_df.iloc[i]
    band = risk_band[i]
    level = driver_level(d1, row)
    recommended_action.append(RULES.get((d1, level), DEFAULT[band]))

predictions = pd.DataFrame(
    {
        "shipment_id": test_df["shipment_id"].values,
        "supplier_id": test_df["supplier_id"].values,
        "destination_city": test_df["destination_city"].values,
        "transportation_mode": test_df["transportation_mode"].values,
        "order_value_usd": test_df["order_value_usd"].values,
        "y_true": y_test.values,
        "y_prob": y_prob,
        "risk_score": risk_score,
        "risk_band": risk_band,
        "driver_1": driver_1,
        "recommended_action": recommended_action,
    }
)
predictions.to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)
print(f"Wrote {len(predictions)} rows to outputs/predictions.csv")
print(predictions["risk_band"].value_counts())
print(predictions.groupby("risk_band")["y_true"].mean())
