import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import train_test_split

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)

import joblib

PROJECT = Path(__file__).resolve().parents[1]

DATA = PROJECT / "data" / "processed"
MODELS_DIR = PROJECT / "models"
RESULTS = PROJECT / "results"
TABLES = RESULTS / "tables"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA / "YahooFinance_dataset_learning_v2.csv")

df["date"] = pd.to_datetime(df["date"])

print(df.shape)

TARGET = "label"

FEATURES_BIASED = [
    "return_1d", "return_3d", "return_7d", "return_14d", "return_30d",
    "vol_ma7", "vol_ma30",
    "vol_ratio7", "vol_ratio30",
    "volatility30",
    "ma7", "ma30",
    "price_ma_ratio",
    "price_vs_max30", "price_vs_min30",
    "rsi14",
    "momentum",
    "candle_range", "upper_shadow", "lower_shadow",
]

FEATURES_STANDARD = [
    "return_1d", "return_3d", "return_7d",
    "vol_ma7",
    "vol_ratio7",
    "volatility30",
    "ma7",
    "price_ma_ratio",
    "rsi14",
    "momentum",
    "candle_range", "upper_shadow", "lower_shadow",
]

FEATURES_ULTRA = [
    "return_1d",
    "vol_ma7",
    "ma7",
    "rsi14",
    "momentum",
    "candle_range", "upper_shadow", "lower_shadow",
]

PROTOCOLS = {
    "Biased": FEATURES_BIASED,
    "Standard": FEATURES_STANDARD,
    "Ultra": FEATURES_ULTRA,
}


def split_chronological(data, test_size=0.20):

    data_sorted = data.sort_values("date").reset_index(drop=True)

    cut = int(len(data_sorted) * (1 - test_size))

    train = data_sorted.iloc[:cut].copy()
    test = data_sorted.iloc[cut:].copy()

    return train, test


def split_random(data, test_size=0.20):

    train, test = train_test_split(
        data,
        test_size=test_size,
        random_state=42,
        stratify=data[TARGET]
    )

    return train.copy(), test.copy()


SPLITS = {
    "Chronological_80_20": split_chronological,
    "Random_80_20": split_random,
}


def build_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.80,
            colsample_bytree=0.80,
            eval_metric="logloss",
            random_state=42
        ),
    }


all_results = []

for split_name, split_fn in SPLITS.items():

    print("SPLIT :", split_name)

    train, test = split_fn(df)

    print("Train :", train.shape)
    print("Test  :", test.shape)
    print("Test ratio :", round(len(test) / len(df), 4))

    for protocol_name, feature_set in PROTOCOLS.items():

        print(split_name, "-", protocol_name, "(", len(feature_set), "features )")

        Xtrain = train[feature_set]
        ytrain = train[TARGET]

        Xtest = test[feature_set]
        ytest = test[TARGET]

        weights = compute_sample_weight(class_weight="balanced", y=ytrain)

        print("Train :", Xtrain.shape, ytrain.shape)
        print("Test  :", Xtest.shape, ytest.shape)

        MODELS = build_models()

        for name, model in MODELS.items():

            print(split_name, "-", protocol_name, "-", name)

            model.fit(Xtrain, ytrain, sample_weight=weights)

            joblib.dump(
                model,
                MODELS_DIR / f"{split_name}_{protocol_name}_{name}_Yahoo.pkl"
            )

            proba = model.predict_proba(Xtest)[:, 1]

            roc = roc_auc_score(ytest, proba)
            pr = average_precision_score(ytest, proba)

            precision, recall, thresholds = precision_recall_curve(ytest, proba)

            f1 = 2 * precision * recall / (precision + recall + 1e-12)

            best = np.argmax(f1)

            best_threshold = thresholds[best] if best < len(thresholds) else 1.0
            best_f1 = f1[best]

            print("ROC-AUC :", round(roc, 4))
            print("PR-AUC  :", round(pr, 4))
            print("Best Threshold :", round(best_threshold, 4))
            print("Best F1 :", round(best_f1, 4))

            all_results.append({
                "SplitType": split_name,
                "Protocol": protocol_name,
                "NumFeatures": len(feature_set),
                "Model": name,
                "TrainSize": len(Xtrain),
                "TestSize": len(Xtest),
                "ROC_AUC": roc,
                "PR_AUC": pr,
                "BestThreshold": best_threshold,
                "BestF1": best_f1,
            })

results = pd.DataFrame(all_results)

print("SUMMARY")
print(results)

results.to_csv(TABLES / "Yahoo_split_x_protocol_summary_v2.csv", index=False)

print("TRAINING FINISHED")
