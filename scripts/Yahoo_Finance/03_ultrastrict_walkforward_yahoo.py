import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.utils.class_weight import compute_sample_weight

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)

PROJECT = Path(__file__).resolve().parents[1]

DATA = PROJECT / "data" / "processed"
TABLES = PROJECT / "results" / "tables"

TABLES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA / "YahooFinance_dataset_learning_v2.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

df["year"] = df["date"].dt.year

print("Shape :", df.shape)

TARGET = "label"

FEATURES = [
    "return_1d", "return_3d", "return_7d", "return_14d", "return_30d",
    "vol_ma7", "vol_ma30", "vol_ratio7", "vol_ratio30",
    "volatility30",
    "ma7", "ma30", "price_ma_ratio",
    "price_vs_max30", "price_vs_min30",
    "rsi14",
    "momentum",
    "candle_range", "upper_shadow", "lower_shadow",
]

assert len(FEATURES) == 20

years = sorted(df["year"].unique())

print("Years :", years)


def build_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=500, random_state=42, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500, learning_rate=0.05, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", random_state=42
        ),
    }


rows = []

for test_year in years[1:]:

    train = df[df.year < test_year]
    test = df[df.year == test_year]

    if len(train) == 0 or len(test) == 0:
        continue

    if test[TARGET].nunique() < 2:
        print("TEST YEAR :", test_year, "-> skip (une seule classe presente)")
        continue

    print("TEST YEAR :", test_year, "| Train :", len(train), "| Test :", len(test))
    print("Positifs train :", train[TARGET].sum(), "| Positifs test :", test[TARGET].sum())

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    weights = compute_sample_weight(class_weight="balanced", y=y_train)

    models = build_models()

    for name, model in models.items():

        model.fit(X_train, y_train, sample_weight=weights)

        proba = model.predict_proba(X_test)[:, 1]

        roc = roc_auc_score(y_test, proba)
        pr = average_precision_score(y_test, proba)

        precision, recall, thresholds = precision_recall_curve(y_test, proba)

        f1_curve = 2 * precision * recall / (precision + recall + 1e-12)

        best_idx = np.argmax(f1_curve)

        best_f1 = f1_curve[best_idx]
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 1.0

        print(
            name,
            "| ROC-AUC:", round(roc, 4),
            "| PR-AUC:", round(pr, 4),
            "| BestF1:", round(best_f1, 4)
        )

        rows.append({
            "Year": test_year,
            "Model": name,
            "TrainSize": len(X_train),
            "TestSize": len(X_test),
            "PositivesTrain": int(y_train.sum()),
            "PositivesTest": int(y_test.sum()),
            "ROC_AUC": roc,
            "PR_AUC": pr,
            "BestThreshold": best_threshold,
            "BestF1": best_f1,
        })

results = pd.DataFrame(rows)

print("ULTRASTRICT RESULTS BY YEAR")
print(results)

results.to_csv(TABLES / "Yahoo_UltraStrict_by_year_v2.csv", index=False)

summary = (
    results
    .groupby("Model")[["ROC_AUC", "PR_AUC", "BestF1"]]
    .mean()
    .reset_index()
)

print("ULTRASTRICT SUMMARY (moyenne sur les annees)")
print(summary)

summary.to_csv(TABLES / "Yahoo_UltraStrict_summary_v2.csv", index=False)

split_protocol_path = TABLES / "Yahoo_split_x_protocol_summary_v2.csv"

if split_protocol_path.exists():

    prior = pd.read_csv(split_protocol_path)

    biased_chrono = prior[
        (prior.SplitType == "Chronological_80_20") & (prior.Protocol == "Biased")
    ][["Model", "PR_AUC", "ROC_AUC"]].rename(
        columns={"PR_AUC": "PR_AUC_Biased_Chrono", "ROC_AUC": "ROC_AUC_Biased_Chrono"}
    )

    comparison = summary.merge(biased_chrono, on="Model", how="left")

    comparison = comparison.rename(
        columns={"PR_AUC": "PR_AUC_UltraStrict", "ROC_AUC": "ROC_AUC_UltraStrict"}
    )

    print("COMPARAISON UltraStrict vs Biased-Chronological (meme dataset v2)")
    print(comparison)

    comparison.to_csv(TABLES / "Yahoo_UltraStrict_vs_Biased_v2.csv", index=False)

else:
    print("Yahoo_split_x_protocol_summary_v2.csv introuvable, comparaison ignoree")

print("FINISHED")
