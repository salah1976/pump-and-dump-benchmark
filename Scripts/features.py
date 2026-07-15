"""
Feature engineering for cryptocurrency market datasets.
"""

import numpy as np
import pandas as pd


# =============================================================================
# Dataset cleaning
# =============================================================================

def clean_market_data(df, group_cols, date_col):
    """
    Basic cleaning shared by all datasets.
    """

    df = df.copy()

    df.columns = df.columns.str.lower()

    df[date_col] = pd.to_datetime(df[date_col])

    df = (
        df.sort_values(group_cols + [date_col])
          .drop_duplicates()
          .reset_index(drop=True)
    )

    if "close" in df.columns:
        df = df[df["close"] > 0]

    if "volume" in df.columns:
        df = df[df["volume"] > 0]

    df = df.dropna()

    return df


# =============================================================================
# Returns
# =============================================================================

def add_returns(df, group_cols, periods=(1, 3, 7, 14)):

    g = df.groupby(group_cols)

    for p in periods:
        df[f"return_{p}d"] = g["close"].pct_change(p)

    return df


# =============================================================================
# Volume features
# =============================================================================

def add_volume_features(df, group_cols, windows=(7, 20, 30)):

    g = df.groupby(group_cols)

    for w in windows:

        ma = (
            g["volume"]
            .transform(lambda x: x.rolling(w).mean())
        )

        df[f"volume_ma{w}"] = ma

        df[f"volume_ratio{w}"] = df["volume"] / ma

    return df


# =============================================================================
# Moving averages
# =============================================================================

def add_moving_averages(df, group_cols, windows=(7, 20, 30)):

    g = df.groupby(group_cols)

    for w in windows:

        df[f"ma{w}"] = (
            g["close"]
            .transform(lambda x: x.rolling(w).mean())
        )

    return df


# =============================================================================
# RSI
# =============================================================================

def add_rsi(df, group_cols, period=14):

    def rsi(series):

        delta = series.diff()

        gain = delta.clip(lower=0)

        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(period).mean()

        avg_loss = loss.rolling(period).mean()

        rs = avg_gain / (avg_loss + 1e-9)

        return 100 - (100 / (1 + rs))

    df["rsi14"] = (
        df.groupby(group_cols)["close"]
          .transform(rsi)
    )

    return df


# =============================================================================
# Candlestick features
# =============================================================================

def add_candlestick_features(df):

    body = (df["close"] - df["open"]).abs()

    upper = np.maximum(df["open"], df["close"])

    lower = np.minimum(df["open"], df["close"])

    df["body"] = body

    df["range"] = (
        df["high"] - df["low"]
    ) / df["close"]

    df["upper_shadow"] = (
        df["high"] - upper
    ) / df["close"]

    df["lower_shadow"] = (
        lower - df["low"]
    ) / df["close"]

    df["body_ratio"] = (
        body /
        (df["high"] - df["low"] + 1e-9)
    )

    df["close_position"] = (
        (df["close"] - df["low"])
        /
        (df["high"] - df["low"] + 1e-9)
    )

    return df