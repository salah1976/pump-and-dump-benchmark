import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils import resample

from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "tables"

TABLES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA / "YahooFinance_dataset_learning_v2.csv")

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

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

split_date = df["date"].quantile(0.80)

train = df[df.date < split_date]
test = df[df.date >= split_date]

print("Split date :", split_date)
print("Train :", train.shape, "| Test :", test.shape)
print("Positifs train :", train[TARGET].sum(), "| Positifs test :", test[TARGET].sum())

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]

weights = compute_sample_weight(class_weight="balanced", y=y_train)

models = {
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

N = 1000

rows = []

for name, model in models.items():

    print("Fitting", name)

    model.fit(X_train, y_train, sample_weight=weights)

    proba = model.predict_proba(X_test)[:, 1]

    point_roc = roc_auc_score(y_test, proba)
    point_pr = average_precision_score(y_test, proba)

    roc_list, pr_list = [], []

    for i in range(N):

        idx = resample(np.arange(len(y_test)), replace=True, random_state=i)

        yb = y_test.iloc[idx]
        pb = proba[idx]

        if yb.nunique() < 2:
            continue

        roc_list.append(roc_auc_score(yb, pb))
        pr_list.append(average_precision_score(yb, pb))

    rows.append({
        "Model": name, "Metric": "ROC_AUC",
        "PointEstimate": point_roc,
        "BootstrapMean": np.mean(roc_list),
        "CI_low": np.percentile(roc_list, 2.5),
        "CI_high": np.percentile(roc_list, 97.5),
        "N_valid_resamples": len(roc_list),
    })

    rows.append({
        "Model": name, "Metric": "PR_AUC",
        "PointEstimate": point_pr,
        "BootstrapMean": np.mean(pr_list),
        "CI_low": np.percentile(pr_list, 2.5),
        "CI_high": np.percentile(pr_list, 97.5),
        "N_valid_resamples": len(pr_list),
    })

    print(name, "| ROC-AUC :", round(point_roc, 4), "| PR-AUC :", round(point_pr, 4))

results = pd.DataFrame(rows)

print("BOOTSTRAP CI - YAHOO FINANCE STANDARD (20 features, corrected)")
print(results)

results.to_csv(TABLES / "Yahoo_Standard_Bootstrap_CI_20features_v2.csv", index=False)

print("Saved :", TABLES / "Yahoo_Standard_Bootstrap_CI_20features_v2.csv")

prior_path = TABLES / "Yahoo_split_x_protocol_summary_v2.csv"

if prior_path.exists():

    prior = pd.read_csv(prior_path)

    prior_biased_chrono = prior[
        (prior.SplitType == "Chronological_80_20") & (prior.Protocol == "Biased")
    ][["Model", "PR_AUC", "ROC_AUC"]]

    print("SANITY CHECK vs Biased-Chronological deja obtenu (meme 20 features)")
    print(prior_biased_chrono)

    print(
        "Les point estimates PR_AUC/ROC_AUC ci-dessus doivent etre "
        "quasi identiques a la colonne PointEstimate du bootstrap "
        "(meme split, memes features, memes hyperparametres)."
    )
