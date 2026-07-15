from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]

PROCESSED = PROJECT / "datasets" / "processed"

INPUT_FILE = PROCESSED / "dataset_features.csv"
OUTPUT_FILE = PROCESSED / "dataset_features_clean.csv"


FEATURE_COLUMNS = [
    "return_1d",
    "return_3d",
    "return_7d",
    "return_14d",
    "volatility_7d",
    "volume_ratio",
    "price_ma_ratio",
    "body",
    "body_ratio",
    "range",
    "upper_shadow",
    "lower_shadow",
    "close_position",
    "momentum_7d",
    "momentum_14d",
    "ema20",
    "ema50",
    "rsi14",
    "atr14",
    "bb_width",
    "volume_ratio20",
    "dist_max20",
    "dist_min20",
]


def main():

    df = pd.read_csv(INPUT_FILE)

    print(f"Initial shape : {df.shape}")

    df = df.dropna(subset=FEATURE_COLUMNS)

    df = df.drop_duplicates()

    df = (
        df.sort_values(["coin", "exchange", "timestamp"])
          .reset_index(drop=True)
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"Final shape   : {df.shape}")
    print(f"Saved to      : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()