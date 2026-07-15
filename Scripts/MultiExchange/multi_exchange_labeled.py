from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]

PROCESSED = PROJECT / "datasets" / "processed"

INPUT_FILE = PROCESSED / "dataset_features_clean.csv"
OUTPUT_FILE = PROCESSED / "dataset_labeled.csv"


HORIZON = 7
PUMP_THRESHOLD = 0.15
MAX_DRAWDOWN = 0.08


def main():

    df = pd.read_csv(INPUT_FILE)

    g = df.groupby(["coin", "exchange"])

    future_max = g["close"].transform(
        lambda s: s.shift(-1).rolling(HORIZON, min_periods=1).max()
    )

    future_min = g["close"].transform(
        lambda s: s.shift(-1).rolling(HORIZON, min_periods=1).min()
    )

    df["future_return"] = (
        future_max - df["close"]
    ) / df["close"]

    df["future_drawdown"] = (
        df["close"] - future_min
    ) / df["close"]

    df["label"] = (
        (df["future_return"] >= PUMP_THRESHOLD)
        & (df["future_drawdown"] <= MAX_DRAWDOWN)
        & (df["volatility_7d"] < df["volatility_7d"].median())
        & (df["volume_ratio20"] > 1)
    ).astype(int)

    df = (
        df.dropna(
            subset=[
                "future_return",
                "future_drawdown",
            ]
        )
        .reset_index(drop=True)
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    summary = (
        df["label"]
        .value_counts()
        .sort_index()
        .rename_axis("label")
        .reset_index(name="count")
    )

    summary["percent"] = (
        summary["count"] / len(df) * 100
    ).round(2)

    print(summary)
    print(f"\nSamples : {len(df):,}")
    print(f"Positive labels : {df['label'].sum():,}")
    print(f"Positive ratio : {df['label'].mean()*100:.2f}%")
    print(f"Saved to : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()