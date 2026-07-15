from pathlib import Path
import pandas as pd
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]

RAW = PROJECT / "data" / "raw"
PROCESSED = PROJECT / "data" / "processed"

RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(RAW / "ALL_coins_daily_multi_exchange_v5.csv")

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="mixed",
    errors="coerce"
)

df = df.sort_values(
    ["coin", "exchange", "timestamp"]
).reset_index(drop=True)

df["return_1d"] = (
    df.groupby(["coin", "exchange"])["close"]
    .pct_change()
)

df["return_3d"] = (
    df.groupby(["coin", "exchange"])["close"]
    .pct_change(3)
)

df["return_7d"] = (
    df.groupby(["coin", "exchange"])["close"]
    .pct_change(7)
)

df["return_14d"] = (
    df.groupby(["coin", "exchange"])["close"]
    .pct_change(14)
)

df["volatility_7d"] = (
    df.groupby(["coin", "exchange"])["return_1d"]
    .rolling(7)
    .std()
    .reset_index(level=[0, 1], drop=True)
)

df["volume_ma7"] = (
    df.groupby(["coin", "exchange"])["volume"]
    .rolling(7)
    .mean()
    .reset_index(level=[0, 1], drop=True)
)

df["volume_ratio"] = df["volume"] / df["volume_ma7"]

df["ma7"] = (
    df.groupby(["coin", "exchange"])["close"]
    .rolling(7)
    .mean()
    .reset_index(level=[0, 1], drop=True)
)

df["price_ma_ratio"] = df["close"] / df["ma7"]

df["body"] = abs(df["close"] - df["open"])

df["range"] = (df["high"] - df["low"]) / df["close"]

df["upper_shadow"] = df["high"] - df[["open", "close"]].max(axis=1)
df["lower_shadow"] = df[["open", "close"]].min(axis=1) - df["low"]

df["body_ratio"] = df["body"] / (df["high"] - df["low"] + 1e-9)

df["close_position"] = (
    (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-9)
)

df["momentum_7d"] = (
    df.groupby(["coin", "exchange"])["close"].diff(7)
)

df["momentum_14d"] = (
    df.groupby(["coin", "exchange"])["close"].diff(14)
)

df["ema20"] = (
    df.groupby(["coin", "exchange"])["close"]
    .transform(lambda x: x.ewm(span=20, adjust=False).mean())
)

df["ema50"] = (
    df.groupby(["coin", "exchange"])["close"]
    .transform(lambda x: x.ewm(span=50, adjust=False).mean())
)

delta = df.groupby(["coin", "exchange"])["close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = (
    gain.groupby([df.coin, df.exchange])
    .rolling(14)
    .mean()
    .reset_index(level=[0, 1], drop=True)
)

avg_loss = (
    loss.groupby([df.coin, df.exchange])
    .rolling(14)
    .mean()
    .reset_index(level=[0, 1], drop=True)
)

rs = avg_gain / (avg_loss + 1e-9)

df["rsi14"] = 100 - (100 / (1 + rs))

tr1 = df["high"] - df["low"]

tr2 = (
    df["high"] - df.groupby(["coin", "exchange"])["close"].shift()
).abs()

tr3 = (
    df["low"] - df.groupby(["coin", "exchange"])["close"].shift()
).abs()

tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

df["atr14"] = (
    tr.groupby([df.coin, df.exchange])
    .rolling(14)
    .mean()
    .reset_index(level=[0, 1], drop=True)
)

ma20 = (
    df.groupby(["coin", "exchange"])["close"]
    .rolling(20)
    .mean()
    .reset_index(level=[0, 1], drop=True)
)

std20 = (
    df.groupby(["coin", "exchange"])["close"]
    .rolling(20)
    .std()
    .reset_index(level=[0, 1], drop=True)
)

upper = ma20 + 2 * std20
lower = ma20 - 2 * std20

df["bb_width"] = (upper - lower) / ma20

df["volume_ma20"] = (
    df.groupby(["coin", "exchange"])["volume"]
    .rolling(20)
    .mean()
    .reset_index(level=[0, 1], drop=True)
)

df["volume_ratio20"] = df["volume"] / df["volume_ma20"]

df["rolling_max20"] = (
    df.groupby(["coin", "exchange"])["close"]
    .rolling(20)
    .max()
    .reset_index(level=[0, 1], drop=True)
)

df["dist_max20"] = df["close"] / df["rolling_max20"]

df["rolling_min20"] = (
    df.groupby(["coin", "exchange"])["close"]
    .rolling(20)
    .min()
    .reset_index(level=[0, 1], drop=True)
)

df["dist_min20"] = df["close"] / df["rolling_min20"]

df["year"] = pd.to_datetime(df["timestamp"]).dt.year

OUTPUT = PROCESSED / "dataset_features.csv"

df.to_csv(OUTPUT, index=False)

print(OUTPUT)
print(df.shape)
