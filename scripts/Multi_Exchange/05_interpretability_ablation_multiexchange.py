import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from sklearn.utils import resample
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

PROJECT = Path(__file__).resolve().parents[1]

DATA = PROJECT / "data" / "processed" / "dataset_learning_v2.csv"

TABLES = PROJECT / "results" / "tables"
FIGURES = PROJECT / "results" / "figures"

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA)

df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")

df = df.sort_values("timestamp").reset_index(drop=True)

print("Shape :", df.shape)

TARGET = "label"

CANONICAL_FEATURES = [
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

missing = [f for f in CANONICAL_FEATURES if f not in df.columns]

if missing:
    raise ValueError(f"Colonnes manquantes dans dataset_learning_v2.csv : {missing}")

print("Nombre de features :", len(CANONICAL_FEATURES))

FEATURE_GROUPS = {
    "Returns": ["return_1d", "return_3d", "return_7d", "return_14d"],
    "Volatility": ["volatility_7d", "atr14", "bb_width"],
    "Volume": ["volume_ratio", "volume_ratio20"],
    "CandleShape": ["body_ratio", "close_position", "upper_shadow_norm", "lower_shadow_norm"],
    "Momentum": ["momentum_7d", "momentum_14d"],
    "Trend_EMA": ["ema20", "ema50", "price_ma_ratio"],
    "Oscillator_RSI": ["rsi14"],
    "RangePosition": ["dist_max20", "dist_min20"],
}


def features_minus_group(group_features):
    excluded = set(group_features)
    return [f for f in CANONICAL_FEATURES if f not in excluded]


def features_only_group(group_features):
    included = set(group_features)
    return [f for f in CANONICAL_FEATURES if f in included]


split_date = df.timestamp.quantile(0.80)

train = df[df.timestamp < split_date]
test = df[df.timestamp >= split_date]

print("Split :", split_date)
print("Train :", train.shape)
print("Test  :", test.shape)

y_train = train[TARGET]
y_test = test[TARGET]

weights = compute_sample_weight(class_weight="balanced", y=y_train)


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


print("BASELINE FIT (all features)")

baseline_models = build_models()

baseline_proba = {}

X_train_base = train[CANONICAL_FEATURES]
X_test_base = test[CANONICAL_FEATURES]

for name, model in baseline_models.items():
    print("Fitting", name)
    model.fit(X_train_base, y_train, sample_weight=weights)
    baseline_proba[name] = model.predict_proba(X_test_base)[:, 1]


print("THRESHOLD OPTIMIZATION")

thresholds_grid = np.arange(0.01, 0.51, 0.01)

all_threshold_rows = []
best_models_info = {}

for name, proba in baseline_proba.items():

    roc = roc_auc_score(y_test, proba)
    pr = average_precision_score(y_test, proba)

    best_f1 = -1
    best_threshold = None
    best_prediction = None

    for t in thresholds_grid:

        pred = (proba >= t).astype(int)

        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        mcc = matthews_corrcoef(y_test, pred)

        all_threshold_rows.append({
            "Model": name,
            "Threshold": t,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "MCC": mcc,
            "ROC_AUC": roc,
            "PR_AUC": pr,
        })

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t
            best_prediction = pred

    best_models_info[name] = {
        "proba": proba,
        "prediction": best_prediction,
        "threshold": best_threshold,
    }

    print(name, "| Best threshold :", best_threshold, "| Best F1 :", round(best_f1, 4))

threshold_results = pd.DataFrame(all_threshold_rows)

threshold_results.to_csv(TABLES / "threshold_results_v2.csv", index=False)

best_thresholds = (
    threshold_results
    .sort_values("F1", ascending=False)
    .groupby("Model")
    .head(1)
    .reset_index(drop=True)
)

print("BEST THRESHOLDS v2 :")
print(best_thresholds)

best_thresholds.to_csv(TABLES / "best_thresholds_v2.csv", index=False)

for metric in ["Precision", "Recall", "F1", "MCC"]:

    plt.figure(figsize=(7, 5))

    for model_name in threshold_results.Model.unique():
        tmp = threshold_results[threshold_results.Model == model_name]
        plt.plot(tmp["Threshold"], tmp[metric], label=model_name, linewidth=2)

    plt.xlabel("Threshold")
    plt.ylabel(metric)
    plt.title(metric + " vs Threshold (v2 corrected)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(FIGURES / f"{metric.lower()}_threshold_v2.png", dpi=300)

    plt.close()

plt.figure(figsize=(7, 6))

for name, obj in best_models_info.items():
    precision, recall, _ = precision_recall_curve(y_test, obj["proba"])
    plt.plot(recall, precision, linewidth=2, label=name)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (v2 corrected)")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(FIGURES / "precision_recall_curve_v2.png", dpi=300)

plt.close()

for name, obj in best_models_info.items():

    cm = confusion_matrix(y_test, obj["prediction"])

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Pump"])

    fig, ax = plt.subplots(figsize=(5, 5))

    disp.plot(ax=ax)

    plt.title(f"{name}\nThreshold={obj['threshold']:.2f} (v2)")

    plt.tight_layout()

    plt.savefig(FIGURES / f"confusion_{name}_v2.png", dpi=300)

    plt.close()

final_threshold_table = best_thresholds[[
    "Model", "Threshold", "Accuracy", "Precision", "Recall", "F1", "MCC", "ROC_AUC", "PR_AUC"
]].sort_values("PR_AUC", ascending=False)

final_threshold_table.to_csv(TABLES / "final_threshold_results_v2.csv", index=False)

final_threshold_table.to_latex(
    TABLES / "final_threshold_results_v2.tex",
    index=False,
    float_format="%.4f"
)


print("SHAP + FEATURE IMPORTANCE + BOOTSTRAP")

rf_model = baseline_models["RandomForest"]
lgb_model = baseline_models["LightGBM"]

importance = pd.DataFrame({
    "feature": CANONICAL_FEATURES,
    "importance": rf_model.feature_importances_
}).sort_values("importance", ascending=False)

importance.to_csv(TABLES / "feature_importance_RF_v2.csv", index=False)

plt.figure(figsize=(8, 7))
plt.barh(importance.feature, importance.importance)
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(FIGURES / "feature_importance_RF_v2.png", dpi=300)
plt.close()

print("RF importance OK")

explainer = shap.TreeExplainer(lgb_model)

sample = X_test_base.sample(min(5000, len(X_test_base)), random_state=42)

shap_values = explainer.shap_values(sample)

if isinstance(shap_values, list):
    shap_values = shap_values[1]

mean_abs = np.abs(shap_values).mean(axis=0)

shap_df = pd.DataFrame({
    "feature": CANONICAL_FEATURES,
    "mean_abs_shap": mean_abs
}).sort_values("mean_abs_shap", ascending=False)

shap_df.to_csv(TABLES / "SHAP_importance_v2.csv", index=False)

plt.figure()
shap.summary_plot(shap_values, sample, show=False)
plt.tight_layout()
plt.savefig(FIGURES / "SHAP_summary_v2.png", dpi=300)
plt.close()

print("SHAP OK")

print("Bootstrap (1000 resamples x 3 models)...")

N = 1000

bootstrap_rows = []

for name, proba in baseline_proba.items():

    roc_list = []
    pr_list = []

    for i in range(N):

        idx = resample(np.arange(len(y_test)), replace=True, random_state=i)

        yb = y_test.iloc[idx]
        pb = proba[idx]

        if yb.nunique() < 2:
            continue

        roc_list.append(roc_auc_score(yb, pb))
        pr_list.append(average_precision_score(yb, pb))

    bootstrap_rows.append({
        "Model": name, "Metric": "ROC_AUC",
        "Mean": np.mean(roc_list),
        "CI_low": np.percentile(roc_list, 2.5),
        "CI_high": np.percentile(roc_list, 97.5),
    })

    bootstrap_rows.append({
        "Model": name, "Metric": "PR_AUC",
        "Mean": np.mean(pr_list),
        "CI_low": np.percentile(pr_list, 2.5),
        "CI_high": np.percentile(pr_list, 97.5),
    })

    print(name, "bootstrap done")

bootstrap = pd.DataFrame(bootstrap_rows)

bootstrap.to_csv(TABLES / "Bootstrap_CI_v2.csv", index=False)

print(bootstrap)


print("ABLATION STUDY")


def train_eval(feature_list, run_name, model_name, model):

    Xtr = train[feature_list]
    Xte = test[feature_list]

    model.fit(Xtr, y_train, sample_weight=weights)

    proba = model.predict_proba(Xte)[:, 1]

    roc = roc_auc_score(y_test, proba)
    pr = average_precision_score(y_test, proba)

    precision, recall, thr = precision_recall_curve(y_test, proba)

    f1_curve = 2 * precision * recall / (precision + recall + 1e-12)

    best_idx = np.argmax(f1_curve)
    best_f1 = f1_curve[best_idx]

    print(
        run_name, "-", model_name,
        "| NumFeatures:", len(feature_list),
        "| ROC-AUC:", round(roc, 4),
        "| PR-AUC:", round(pr, 4),
        "| BestF1:", round(best_f1, 4)
    )

    return {
        "Run": run_name,
        "Model": model_name,
        "NumFeatures": len(feature_list),
        "ROC_AUC": roc,
        "PR_AUC": pr,
        "BestF1": best_f1,
    }, proba


ablation_results = []
ablation_proba_store = {}

for name, proba in baseline_proba.items():

    roc = roc_auc_score(y_test, proba)
    pr = average_precision_score(y_test, proba)

    precision, recall, thr = precision_recall_curve(y_test, proba)

    f1_curve = 2 * precision * recall / (precision + recall + 1e-12)

    best_f1 = f1_curve[np.argmax(f1_curve)]

    ablation_results.append({
        "Run": "Baseline_AllFeatures",
        "Model": name,
        "NumFeatures": len(CANONICAL_FEATURES),
        "ROC_AUC": roc,
        "PR_AUC": pr,
        "BestF1": best_f1,
    })

    ablation_proba_store[("Baseline_AllFeatures", name)] = proba

baseline_pr = {
    r["Model"]: r["PR_AUC"]
    for r in ablation_results
    if r["Run"] == "Baseline_AllFeatures"
}

print("-- LEAVE-ONE-GROUP-OUT --")

for group_name, group_features in FEATURE_GROUPS.items():

    remaining = features_minus_group(group_features)

    models_logo = build_models()

    for name, model in models_logo.items():

        res, proba = train_eval(remaining, f"Remove_{group_name}", name, model)

        res["RemovedGroup"] = group_name
        res["DeltaPR_vs_Baseline"] = res["PR_AUC"] - baseline_pr[name]

        ablation_results.append(res)

        ablation_proba_store[(f"Remove_{group_name}", name)] = proba

print("-- GROUP-ONLY --")

for group_name, group_features in FEATURE_GROUPS.items():

    only_features = features_only_group(group_features)

    models_only = build_models()

    for name, model in models_only.items():

        res, proba = train_eval(only_features, f"Only_{group_name}", name, model)

        res["OnlyGroup"] = group_name

        ablation_results.append(res)

        ablation_proba_store[(f"Only_{group_name}", name)] = proba

ablation_df = pd.DataFrame(ablation_results)

ablation_df.to_csv(TABLES / "ablation_study_results_v2_corrected.csv", index=False)

print(ablation_df)

logo = ablation_df[ablation_df["Run"].str.startswith("Remove_")].copy()

logo_summary = logo.groupby("RemovedGroup")["DeltaPR_vs_Baseline"].mean().sort_values()

plt.figure(figsize=(8, 6))

colors = ["firebrick" if v < 0 else "steelblue" for v in logo_summary.values]

plt.barh(logo_summary.index, logo_summary.values, color=colors)
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Delta PR-AUC vs baseline (average of 3 models)")
plt.title("Leave-One-Group-Out (v2 corrected)")
plt.tight_layout()

plt.savefig(FIGURES / "ablation_leave_one_group_out_v2_corrected.png", dpi=300)

plt.close()

only = ablation_df[ablation_df["Run"].str.startswith("Only_")].copy()

only_summary = only.groupby("OnlyGroup")["PR_AUC"].mean().sort_values(ascending=False)

plt.figure(figsize=(8, 6))

plt.barh(only_summary.index, only_summary.values, color="seagreen")

plt.axvline(
    np.mean(list(baseline_pr.values())),
    color="black", linestyle="--", linewidth=1, label="Baseline (all features)"
)

plt.xlabel("PR-AUC (moyenne des 3 modeles)")
plt.title("Group-Only (v2 corrected)")
plt.legend()
plt.tight_layout()

plt.savefig(FIGURES / "ablation_group_only_v2_corrected.png", dpi=300)

plt.close()

print("-- BOOTSTRAP CIBLE : Baseline vs Only_CandleShape vs Only_RangePosition --")

RUNS_TO_TEST = ["Baseline_AllFeatures", "Only_CandleShape", "Only_RangePosition"]

ablation_bootstrap_rows = []

for run_name in RUNS_TO_TEST:

    for model_name in ["RandomForest", "LightGBM", "XGBoost"]:

        proba = ablation_proba_store[(run_name, model_name)]

        roc_list = []
        pr_list = []

        for i in range(N):

            idx = resample(np.arange(len(y_test)), replace=True, random_state=i)

            yb = y_test.iloc[idx]
            pb = proba[idx]

            if yb.nunique() < 2:
                continue

            roc_list.append(roc_auc_score(yb, pb))
            pr_list.append(average_precision_score(yb, pb))

        ablation_bootstrap_rows.append({
            "Run": run_name, "Model": model_name, "Metric": "ROC_AUC",
            "Mean": np.mean(roc_list),
            "CI_low": np.percentile(roc_list, 2.5),
            "CI_high": np.percentile(roc_list, 97.5),
        })

        ablation_bootstrap_rows.append({
            "Run": run_name, "Model": model_name, "Metric": "PR_AUC",
            "Mean": np.mean(pr_list),
            "CI_low": np.percentile(pr_list, 2.5),
            "CI_high": np.percentile(pr_list, 97.5),
        })

        print(run_name, "-", model_name, "bootstrap done")

ablation_bootstrap = pd.DataFrame(ablation_bootstrap_rows)

ablation_bootstrap.to_csv(TABLES / "ablation_bootstrap_CI_v2_corrected.csv", index=False)

print(ablation_bootstrap)

print("ALL RESULTS SAVED (v2 corrected)")
print("Tables :", TABLES)
print("Figures:", FIGURES)
