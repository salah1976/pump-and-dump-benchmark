import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

DATA = PROJECT / "data" / "processed"

df = pd.read_csv(DATA / "dataset_features.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")

df = df.sort_values(["coin", "exchange", "timestamp"]).reset_index(drop=True)

print("Shape :", df.shape)

df["upper_shadow_norm"] = df["upper_shadow"] / df["close"]

df["lower_shadow_norm"] = df["lower_shadow"] / df["close"]

selected_features_v2 = [
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

assert len(selected_features_v2) == 21, "La liste corrigee doit rester a 21 features."

HORIZON = 7
RETURN_THRESHOLD = 0.30
VOLUME_THRESHOLD = 3.0

future_return = (
    df.groupby(["coin", "exchange"])["close"]
    .shift(-HORIZON)
    .div(df["close"])
    .sub(1)
)

future_volume = (
    df.groupby(["coin", "exchange"])["volume"]
    .shift(-HORIZON)
    / df["volume"]
)

df["label"] = (
    (future_return >= RETURN_THRESHOLD)
    & (future_volume >= VOLUME_THRESHOLD)
).astype(int)

n_before = len(df)

df = df[~(future_return.isna() | future_volume.isna())].copy()

n_after = len(df)

print("Lignes de bord retirees (future_return/future_volume NaN) :", n_before - n_after)

df = df.dropna(subset=selected_features_v2 + ["label"]).copy()

print("Shape apres dropna features+label :", df.shape)

final_columns = ["timestamp", "coin", "exchange"] + selected_features_v2 + ["label"]

dataset_learning_v2 = df[final_columns].copy()

output_path = DATA / "dataset_learning_v2.csv"

dataset_learning_v2.to_csv(output_path, index=False)

print("Shape :", dataset_learning_v2.shape)
print(dataset_learning_v2["label"].value_counts())
print(dataset_learning_v2["label"].value_counts(normalize=True) * 100)
print(output_path)
