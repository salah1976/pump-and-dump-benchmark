import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

RAW = PROJECT / "data" / "raw"
PROCESSED = PROJECT / "data" / "processed"

PROCESSED.mkdir(parents=True, exist_ok=True)

PATH = RAW / "YahooFinance_25coins_2017_2025.csv"

df = pd.read_csv(PATH)

print(df.shape)

df.columns = df.columns.str.lower()

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(["ticker", "date"])

df = df.drop_duplicates()

df = df.reset_index(drop=True)

print(df.shape)
print(df.isna().sum())

df = df.dropna()

df = df[df["close"] > 0]
df = df[df["volume"] > 0]

print(df.shape)
print(df.date.min())
print(df.date.max())

for n in [1, 3, 7, 14, 30]:
    df[f"return_{n}d"] = (
        df.groupby("ticker")["close"].pct_change(n) * 100
    )

df["vol_ma7"] = (
    df.groupby("ticker")["volume"].transform(lambda x: x.rolling(7).mean())
)

df["vol_ma30"] = (
    df.groupby("ticker")["volume"].transform(lambda x: x.rolling(30).mean())
)

df["vol_ratio7"] = df["volume"] / df["vol_ma7"]

df["vol_ratio30"] = df["volume"] / df["vol_ma30"]

daily_return = df.groupby("ticker")["close"].pct_change()

df["volatility30"] = (
    daily_return.groupby(df["ticker"]).transform(lambda x: x.rolling(30).std())
)

df["ma7"] = (
    df.groupby("ticker")["close"].transform(lambda x: x.rolling(7).mean())
)

df["ma30"] = (
    df.groupby("ticker")["close"].transform(lambda x: x.rolling(30).mean())
)

df["price_ma_ratio"] = df["ma7"] / df["ma30"]

rolling_max = (
    df.groupby("ticker")["close"].transform(lambda x: x.rolling(30).max())
)

rolling_min = (
    df.groupby("ticker")["close"].transform(lambda x: x.rolling(30).min())
)

df["price_vs_max30"] = df["close"] / rolling_max

df["price_vs_min30"] = df["close"] / rolling_min


def RSI(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / (avg_loss + 1e-9)

    return 100 - (100 / (1 + rs))


df["rsi14"] = df.groupby("ticker")["close"].transform(RSI)

ma5 = df.groupby("ticker")["close"].transform(lambda x: x.rolling(5).mean())

ma20 = df.groupby("ticker")["close"].transform(lambda x: x.rolling(20).mean())

df["momentum"] = ma5 / ma20

df["candle_range"] = (df["high"] - df["low"]) / df["close"]

upper = np.maximum(df["open"], df["close"])
lower = np.minimum(df["open"], df["close"])

df["upper_shadow"] = (df["high"] - upper) / df["close"]

df["lower_shadow"] = (lower - df["low"]) / df["close"]

print("Features calculees :", df.shape)

HORIZON = 7

RETURN_THRESHOLD = 30.0
VOLUME_THRESHOLD = 3.0

future_close = df.groupby("ticker")["close"].shift(-HORIZON)

future_return_pct = (future_close / df["close"] - 1) * 100

future_volume = df.groupby("ticker")["volume"].shift(-HORIZON)

future_volume_ratio = future_volume / df["volume"]

df["future_return_7d"] = future_return_pct

df["future_volume_ratio_7d"] = future_volume_ratio

df["label"] = (
    (future_return_pct >= RETURN_THRESHOLD)
    & (future_volume_ratio >= VOLUME_THRESHOLD)
).astype(int)

before = len(df)

df = df.dropna(subset=[
    "return_30d", "vol_ratio30", "volatility30", "ma30",
    "price_vs_max30", "price_vs_min30", "rsi14", "momentum",
    "future_return_7d", "future_volume_ratio_7d"
])

after = len(df)

print("Lignes retirees (NaN features/label window) :", before - after)
print(df.shape)
print(df["label"].value_counts())
print("Pump ratio : %.4f%%" % (df["label"].mean() * 100))

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

assert len(FEATURES) == 20, "La liste de features doit rester a 20."

final_columns = ["date", "ticker"] + FEATURES + ["label"]

dataset_learning = df[final_columns].copy()

OUTPUT = PROCESSED / "YahooFinance_dataset_learning_v2.csv"

dataset_learning.to_csv(OUTPUT, index=False)

print("SAVED")
print(OUTPUT)

print("Observations :", len(dataset_learning))
print("Pump events :", dataset_learning["label"].sum())
print("Pump ratio : %.4f%%" % (100 * dataset_learning["label"].mean()))
print("Tickers :", dataset_learning["ticker"].nunique())
print("Period :", dataset_learning["date"].min(), "-", dataset_learning["date"].max())
