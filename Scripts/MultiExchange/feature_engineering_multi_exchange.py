from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]

RAW = PROJECT / "datasets" / "raw"
PROCESSED = PROJECT / "datasets" / "processed"

INPUT_FILE = RAW / "ALL_coins_daily_multi_exchange_v5.csv"
OUTPUT_FILE = PROCESSED / "dataset_features.csv"


def main():

    PROCESSED.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df = (
        df.sort_values(["coin", "exchange", "timestamp"])
          .reset_index(drop=True)
    )

    g = df.groupby(["coin", "exchange"])

    # Returns

    df["return_1d"] = g["close"].pct_change()
    df["return_3d"] = g["close"].pct_change(3)
    df["return_7d"] = g["close"].pct_change(7)
    df["return_14d"] = g["close"].pct_change(14)

    # Volatility

    df["volatility_7d"] = (
        g["return_1d"]
        .rolling(7)
        .std()
        .reset_index(level=[0, 1], drop=True)
    )

    # Volume

    df["volume_ma7"] = (
        g["volume"]
        .rolling(7)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    df["volume_ratio"] = df["volume"] / df["volume_ma7"]

    df["volume_ma20"] = (
        g["volume"]
        .rolling(20)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    df["volume_ratio20"] = (
        df["volume"] /
        df["volume_ma20"]
    )

    # Moving averages

    df["ma7"] = (
        g["close"]
        .rolling(7)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    df["price_ma_ratio"] = (
        df["close"] /
        df["ma7"]
    )

    df["ema20"] = (
        g["close"]
        .transform(lambda x: x.ewm(span=20, adjust=False).mean())
    )

    df["ema50"] = (
        g["close"]
        .transform(lambda x: x.ewm(span=50, adjust=False).mean())
    )

    # Candlesticks

    df["body"] = (df["close"] - df["open"]).abs()

    df["range"] = (
        df["high"] - df["low"]
    ) / df["close"]

    df["upper_shadow"] = (
        df["high"] -
        df[["open", "close"]].max(axis=1)
    )

    df["lower_shadow"] = (
        df[["open", "close"]].min(axis=1)
        - df["low"]
    )

    df["body_ratio"] = (
        df["body"] /
        (df["high"] - df["low"] + 1e-9)
    )

    df["close_position"] = (
        (df["close"] - df["low"]) /
        (df["high"] - df["low"] + 1e-9)
    )

    # Momentum

    df["momentum_7d"] = g["close"].diff(7)
    df["momentum_14d"] = g["close"].diff(14)

    # RSI

    delta = g["close"].diff()

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

    # ATR

    prev_close = g["close"].shift()

    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)

    df["atr14"] = (
        tr.groupby([df.coin, df.exchange])
        .rolling(14)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    # Bollinger Bands

    ma20 = (
        g["close"]
        .rolling(20)
        .mean()
        .reset_index(level=[0, 1], drop=True)
    )

    std20 = (
        g["close"]
        .rolling(20)
        .std()
        .reset_index(level=[0, 1], drop=True)
    )

    upper = ma20 + 2 * std20
    lower = ma20 - 2 * std20

    df["bb_width"] = (
        upper - lower
    ) / ma20

    # Distance to extrema

    df["rolling_max20"] = (
        g["close"]
        .rolling(20)
        .max()
        .reset_index(level=[0, 1], drop=True)
    )

    df["rolling_min20"] = (
        g["close"]
        .rolling(20)
        .min()
        .reset_index(level=[0, 1], drop=True)
    )

    df["dist_max20"] = (
        df["close"] /
        df["rolling_max20"]
    )

    df["dist_min20"] = (
        df["close"] /
        df["rolling_min20"]
    )

    df["year"] = df["timestamp"].dt.year

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"Saved: {OUTPUT_FILE}")
    print(df.shape)


if __name__ == "__main__":
    main()