import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix
)

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import joblib

PROJECT = Path(__file__).resolve().parents[1]

DATA = PROJECT / "data" / "processed" / "dataset_learning_v2.csv"

RESULTS = PROJECT / "results" / "tables"

MODELS_DIR = PROJECT / "models_v2"

RESULTS.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

df_raw = pd.read_csv(DATA)

df_raw["timestamp"] = pd.to_datetime(
    df_raw["timestamp"], format="mixed", errors="coerce"
)

print("Shape :", df_raw.shape)

TARGET = "label"

FEATURES = [
    "return_1d",
    "return_3d",
    "return_7d",
    "return_14d",
    "volatility_7d",
    "volume_ratio",
    "volume_ratio20",
    "price_ma_ratio",
    "body_ratio",
    "close_position",
    "upper_shadow_norm",
    "lower_shadow_norm",
    "momentum_7d",
    "momentum_14d",
    "ema20",
    "ema50",
    "rsi14",
    "atr14",
    "bb_width",
    "dist_max20",
    "dist_min20",
]

missing = [f for f in FEATURES if f not in df_raw.columns]

if missing:
    raise ValueError(f"Colonnes manquantes dans dataset_learning_v2.csv : {missing}")

print("Nombre de features :", len(FEATURES))


def build_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        ),
    }


def compute_all_metrics(y_test, pred_default, proba):

    roc = roc_auc_score(y_test, proba)
    pr = average_precision_score(y_test, proba)

    precision, recall, thresholds = precision_recall_curve(y_test, proba)

    f1_curve = 2 * precision * recall / (precision + recall + 1e-12)

    best_idx = np.argmax(f1_curve)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
    best_f1 = f1_curve[best_idx]

    return {
        "Accuracy": accuracy_score(y_test, pred_default),
        "Precision_at_0.5": precision_score(y_test, pred_default, zero_division=0),
        "Recall_at_0.5": recall_score(y_test, pred_default, zero_division=0),
        "F1_at_0.5": f1_score(y_test, pred_default, zero_division=0),
        "ROC_AUC": roc,
        "PR_AUC": pr,
        "BestThreshold": best_threshold,
        "BestF1": best_f1,
    }


def run_one(protocol_name, model_name, model, X_train, y_train, X_test, y_test):

    weights = compute_sample_weight(class_weight="balanced", y=y_train)

    model.fit(X_train, y_train, sample_weight=weights)

    joblib.dump(model, MODELS_DIR / f"{protocol_name}_{model_name}_v2.pkl")

    proba = model.predict_proba(X_test)[:, 1]

    pred_default = (proba >= 0.50).astype(int)

    metrics = compute_all_metrics(y_test, pred_default, proba)

    metrics["Protocol"] = protocol_name
    metrics["Model"] = model_name
    metrics["TrainSize"] = len(X_train)
    metrics["TestSize"] = len(X_test)

    print(
        model_name,
        "| ROC-AUC:", round(metrics["ROC_AUC"], 4),
        "| PR-AUC:", round(metrics["PR_AUC"], 4),
        "| F1@0.5:", round(metrics["F1_at_0.5"], 4),
        "| BestF1:", round(metrics["BestF1"], 4)
    )

    return metrics


print("PROTOCOL 1 - BIASED (random 80/20)")

X = df_raw[FEATURES]
y = df_raw[TARGET]

X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print("Train :", len(X_train_b), "| Test :", len(X_test_b))

all_results = []

models_biased = build_models()

for name, model in models_biased.items():
    res = run_one("Biased", name, model, X_train_b, y_train_b, X_test_b, y_test_b)
    all_results.append(res)


print("PROTOCOL 2 - STANDARD (chronological quantile 80/20)")

df_sorted = df_raw.sort_values("timestamp").reset_index(drop=True)

split_date = df_sorted["timestamp"].quantile(0.80)

train_s = df_sorted[df_sorted.timestamp < split_date]
test_s = df_sorted[df_sorted.timestamp >= split_date]

print("Split date :", split_date)
print("Train :", train_s.shape, "| Test :", test_s.shape)

X_train_s = train_s[FEATURES]
y_train_s = train_s[TARGET]
X_test_s = test_s[FEATURES]
y_test_s = test_s[TARGET]

models_standard = build_models()

for name, model in models_standard.items():
    res = run_one("Standard", name, model, X_train_s, y_train_s, X_test_s, y_test_s)
    all_results.append(res)


print("PROTOCOL 3 - ULTRASTRICT (walk-forward)")

df_wf = df_raw.copy()
df_wf["year"] = df_wf["timestamp"].dt.year

years = sorted(df_wf["year"].unique())

print("Years :", years)

walkforward_rows = []

for test_year in years[1:]:

    train_y = df_wf[df_wf.year < test_year]
    test_y = df_wf[df_wf.year == test_year]

    if len(train_y) == 0 or len(test_y) == 0:
        continue

    if test_y[TARGET].nunique() < 2:
        continue

    print("TEST YEAR :", test_year, "| Train :", len(train_y), "| Test :", len(test_y))

    X_train_y = train_y[FEATURES]
    y_train_y = train_y[TARGET]
    X_test_y = test_y[FEATURES]
    y_test_y = test_y[TARGET]

    models_year = build_models()

    for name, model in models_year.items():

        weights = compute_sample_weight(class_weight="balanced", y=y_train_y)

        model.fit(X_train_y, y_train_y, sample_weight=weights)

        proba = model.predict_proba(X_test_y)[:, 1]

        pred_default = (proba >= 0.50).astype(int)

        metrics = compute_all_metrics(y_test_y, pred_default, proba)

        metrics["Year"] = test_year
        metrics["Model"] = name
        metrics["TrainSize"] = len(X_train_y)
        metrics["TestSize"] = len(X_test_y)

        print(
            name,
            "| ROC-AUC:", round(metrics["ROC_AUC"], 4),
            "| PR-AUC:", round(metrics["PR_AUC"], 4),
            "| BestF1:", round(metrics["BestF1"], 4)
        )

        walkforward_rows.append(metrics)

walkforward_results = pd.DataFrame(walkforward_rows)

walkforward_results.to_csv(
    RESULTS / "protocol_ultrastrict_by_year_v2.csv", index=False
)

walkforward_summary = (
    walkforward_results
    .groupby("Model")[
        ["Accuracy", "Precision_at_0.5", "Recall_at_0.5",
         "F1_at_0.5", "ROC_AUC", "PR_AUC", "BestThreshold",
         "BestF1", "TrainSize", "TestSize"]
    ]
    .mean()
    .reset_index()
)

walkforward_summary["Protocol"] = "UltraStrict"

for row in walkforward_summary.to_dict("records"):
    all_results.append(row)


results = pd.DataFrame(all_results)

column_order = [
    "Protocol", "Model", "TrainSize", "TestSize",
    "Accuracy", "Precision_at_0.5", "Recall_at_0.5", "F1_at_0.5",
    "ROC_AUC", "PR_AUC", "BestThreshold", "BestF1"
]

results = results[[c for c in column_order if c in results.columns]]

print(results)

results.to_csv(RESULTS / "multiexchange_protocol_summary_v2.csv", index=False)


v1_path = RESULTS / "multiexchange_protocol_summary.csv"

if v1_path.exists():

    v1 = pd.read_csv(v1_path)

    comparison = results[["Protocol", "Model", "PR_AUC", "ROC_AUC"]].merge(
        v1[["Protocol", "Model", "PR_AUC", "ROC_AUC"]],
        on=["Protocol", "Model"],
        suffixes=("_v2_corrected", "_v1_original")
    )

    comparison["Delta_PR_AUC"] = (
        comparison["PR_AUC_v2_corrected"] - comparison["PR_AUC_v1_original"]
    )

    print(comparison)

    comparison.to_csv(RESULTS / "v1_vs_v2_comparison.csv", index=False)

else:
    print("multiexchange_protocol_summary.csv v1 introuvable, comparaison ignoree")

print("FINISHED")
