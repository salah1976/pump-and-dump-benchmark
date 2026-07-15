from pathlib import Path
import pandas as pd
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]

df = pd.read_csv(PROJECT / "data" / "processed" / "dataset_features_clean.csv")

df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", errors="coerce")

HORIZON = 7

PUMP_THRESHOLD = 0.15
MAX_DRAWDOWN = 0.08

future_max = (
    df.groupby(["coin", "exchange"])["close"]
    .transform(
        lambda s: s.shift(-1).rolling(HORIZON, min_periods=1).max()
    )
)

future_min = (
    df.groupby(["coin", "exchange"])["close"]
    .transform(
        lambda s: s.shift(-1).rolling(HORIZON, min_periods=1).min()
    )
)

df["future_return"] = (future_max - df["close"]) / df["close"]

df["future_drawdown"] = (df["close"] - future_min) / df["close"]

condition1 = df["future_return"] >= PUMP_THRESHOLD
condition2 = df["future_drawdown"] <= MAX_DRAWDOWN
condition3 = df["volatility_7d"] < df["volatility_7d"].median()
condition4 = df["volume_ratio20"] > 1

df["label"] = (
    condition1 & condition2 & condition3 & condition4
).astype(int)

df = df.dropna(
    subset=["future_return", "future_drawdown"]
).reset_index(drop=True)

summary = (
    df["label"]
    .value_counts()
    .rename_axis("label")
    .reset_index(name="count")
)

summary["percent"] = 100 * summary["count"] / len(df)

print(summary)
print("Nombre total :", len(df))
print("Pompes :", df["label"].sum())
print("Ratio :", round(df["label"].mean() * 100, 2), "%")

save_path = PROJECT / "data" / "processed" / "dataset_labeled_multiexchange.csv"

df.to_csv(save_path, index=False)

print(save_path)
