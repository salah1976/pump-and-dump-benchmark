import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef
)

from sklearn.utils.class_weight import compute_sample_weight

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

PROJECT = Path(__file__).resolve().parents[1]

TABLES = PROJECT / "results" / "tables"
FIGURES = PROJECT / "results" / "figures"

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PROJECT / "data" / "processed" / "dataset_learning_v2.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")

df = df.sort_values("timestamp").reset_index(drop=True)

print(df.shape)

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

missing = [f for f in FEATURES if f not in df.columns]

if missing:
    raise ValueError(f"Colonnes manquantes dans dataset_learning_v2.csv : {missing}")

TARGET = "label"

split_date = df.timestamp.quantile(0.80)

print("Split date (identique a Protocol Standard) :", split_date)


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


best_thresholds_path = TABLES / "best_thresholds_v2.csv"

if not best_thresholds_path.exists():
    raise FileNotFoundError(
        "best_thresholds_v2.csv introuvable. Executez d'abord le script "
        "d'interpretabilite / threshold optimization (v2)."
    )

best_thresholds_df = pd.read_csv(best_thresholds_path)

THRESHOLDS = dict(
    zip(best_thresholds_df["Model"], best_thresholds_df["Threshold"])
)

print("Seuils optimaux charges (v2) :", THRESHOLDS)


def evaluate_model(model_name, model, Xtrain, ytrain, Xtest, ytest):

    weights = compute_sample_weight(class_weight="balanced", y=ytrain)

    model.fit(Xtrain, ytrain, sample_weight=weights)

    proba = model.predict_proba(Xtest)[:, 1]

    threshold = THRESHOLDS[model_name]

    pred = (proba >= threshold).astype(int)

    return {
        "Model": model_name,
        "Threshold": threshold,
        "Accuracy": accuracy_score(ytest, pred),
        "Precision": precision_score(ytest, pred, zero_division=0),
        "Recall": recall_score(ytest, pred, zero_division=0),
        "F1": f1_score(ytest, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(ytest, proba),
        "PR_AUC": average_precision_score(ytest, proba),
        "MCC": matthews_corrcoef(ytest, pred),
    }


print("CROSS EXCHANGE GENERALIZATION (temporally constrained, v2)")

results_list = []

exchanges = sorted(df["exchange"].unique())

print("Exchanges:", exchanges)

for test_exchange in exchanges:

    print("TEST EXCHANGE:", test_exchange)

    train = df[
        (df.exchange != test_exchange) &
        (df.timestamp < split_date)
    ]

    test = df[
        (df.exchange == test_exchange) &
        (df.timestamp >= split_date)
    ]

    print("Train (autres exchanges, avant split_date):", train.shape)
    print("Test  (", test_exchange, ", apres split_date):", test.shape)

    if len(test) == 0:
        print("Aucune ligne de test pour cet exchange apres split_date -> skip")
        continue

    if test[TARGET].nunique() < 2:
        print("Un seul label present dans le test -> skip")
        continue

    Xtrain = train[FEATURES]
    ytrain = train[TARGET]

    Xtest = test[FEATURES]
    ytest = test[TARGET]

    models = build_models()

    for model_name, model in models.items():

        print(model_name)

        result = evaluate_model(model_name, model, Xtrain, ytrain, Xtest, ytest)

        result["TestExchange"] = test_exchange
        result["TrainSize"] = len(Xtrain)
        result["TestSize"] = len(Xtest)

        results_list.append(result)

results = pd.DataFrame(results_list)

print("CROSS EXCHANGE RESULTS (temporally constrained, v2)")
print(results)

average_results = (
    results
    .groupby("Model")
    .mean(numeric_only=True)
    .reset_index()
)

print("AVERAGE RESULTS")
print(average_results)

standard_path = TABLES / "final_threshold_results_v2.csv"

if standard_path.exists():

    standard = pd.read_csv(standard_path)

    comparison = average_results[["Model", "PR_AUC", "ROC_AUC"]].merge(
        standard[["Model", "PR_AUC", "ROC_AUC"]],
        on="Model",
        suffixes=("_CrossExchange", "_Standard")
    )

    print("SANITY CHECK - CrossExchange vs Standard (v2)")
    print(comparison)

    comparison.to_csv(
        TABLES / "cross_exchange_vs_standard_sanity_check_v2.csv",
        index=False
    )

else:
    print("final_threshold_results_v2.csv introuvable, sanity check ignore")

results.to_csv(TABLES / "cross_exchange_results_v2_temporal.csv", index=False)

average_results.to_csv(TABLES / "cross_exchange_average_v2_temporal.csv", index=False)

print("FINISHED (v2, temporally constrained)")
