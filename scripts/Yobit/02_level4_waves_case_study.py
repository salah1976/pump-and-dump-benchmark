import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

DATA = PROJECT / "data" / "processed"
RESULTS = PROJECT / "results" / "tables"
FIGURES = PROJECT / "results" / "figures"

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

enriched = pd.read_csv(RESULTS / "yobit_trades_enriched_v3.csv")
enriched["time"] = pd.to_datetime(enriched["time"])
enriched["date"] = pd.to_datetime(enriched["date"])

print("Trades enrichis (v3) :", enriched.shape)

assert "near_pump_7d" in enriched.columns, (
    "near_pump_7d absent -> relancer d'abord le script Level 4 (windowed analysis)."
)

features_raw = pd.read_csv(
    DATA / "dataset_features.csv",
    usecols=["timestamp", "coin", "exchange", "close"]
)
features_raw["timestamp"] = pd.to_datetime(
    features_raw["timestamp"], format="mixed", errors="coerce"
)

learning_v2 = pd.read_csv(
    DATA / "dataset_learning_v2.csv",
    usecols=["timestamp", "coin", "exchange", "label"]
)
learning_v2["timestamp"] = pd.to_datetime(
    learning_v2["timestamp"], format="mixed", errors="coerce"
)

market = features_raw.merge(
    learning_v2, on=["timestamp", "coin", "exchange"], how="inner"
)
market["date"] = market["timestamp"].dt.normalize()
market["coin"] = market["coin"].str.upper()

market_daily = (
    market
    .groupby(["coin", "date"])
    .agg(market_close=("close", "median"), label_pump=("label", "max"))
    .reset_index()
    .sort_values(["coin", "date"])
)

pump_days = market_daily[market_daily["label_pump"] == 1][["coin", "date"]].copy()
pump_days = pump_days.rename(columns={"date": "pump_date"})

print("Coin-jours labellises pump (Formule 1, couverture marche) :", len(pump_days))

market_dates_by_coin = (
    market_daily.groupby("coin")["date"].apply(set).to_dict()
)


def is_evaluable(row):
    coin_dates = market_dates_by_coin.get(row["coin"])
    if not coin_dates:
        return False
    return row["date"] in coin_dates


enriched["evaluable"] = enriched.apply(is_evaluable, axis=1)


def nearest_pump_distance(row):
    coin_pumps = pump_days[pump_days["coin"] == row["coin"]]
    if coin_pumps.empty:
        return np.nan
    delta_days = (coin_pumps["pump_date"] - row["date"]).dt.days
    idx = delta_days.abs().idxmin()
    return int(delta_days.loc[idx])


enriched["days_to_nearest_pump"] = enriched.apply(nearest_pump_distance, axis=1)

print("COUVERTURE GLOBALE DES ALERTES (BUY evaluables, Formule 1, +/-7j)")

buys_eval = enriched[
    (enriched["type"] == "BUY") & (enriched["evaluable"])
].copy()

n_buys_eval = len(buys_eval)
n_buys_flagged = int(buys_eval["near_pump_7d"].sum())

print(f"BUY evaluables (couverture marche disponible) : {n_buys_eval}")
print(f"BUY precedes/entoures d'un pump labellise (+/-7j) : {n_buys_flagged} "
      f"({100 * n_buys_flagged / max(n_buys_eval, 1):.1f}%)")

coverage_by_coin = (
    buys_eval[buys_eval["near_pump_7d"] == 1]
    .groupby("coin")
    .size()
    .sort_values(ascending=False)
)

print(coverage_by_coin.to_string())

coverage_summary = pd.DataFrame({
    "n_buys_evaluable": [n_buys_eval],
    "n_buys_flagged_near_pump_7d": [n_buys_flagged],
    "pct_flagged": [round(100 * n_buys_flagged / max(n_buys_eval, 1), 2)],
})

coverage_summary.to_csv(RESULTS / "level4_alert_coverage_summary.csv", index=False)
coverage_by_coin.to_csv(RESULTS / "level4_alert_coverage_by_coin.csv")

print("CHRONOLOGIE WAVES/USD (avril-mai 2022), Formule 1 canonique")

waves_buys = enriched[
    (enriched["coin"] == "WAVES") &
    (enriched["type"] == "BUY") &
    (enriched["date"] >= "2022-03-01") &
    (enriched["date"] <= "2022-06-30")
].copy().sort_values("date")

cols_out = [
    "date", "price", "amount", "total",
    "near_pump_7d", "days_to_nearest_pump", "evaluable"
]
cols_out = [c for c in cols_out if c in waves_buys.columns]

print(waves_buys[cols_out].to_string(index=False))

waves_buys[cols_out].to_csv(RESULTS / "level4_waves_chronology_v2.csv", index=False)

n_waves_flagged = int(waves_buys["near_pump_7d"].sum())

print(f"Lots WAVES post-pic (mars-juin 2022) evalues : {len(waves_buys)}")
print(f"Lots signales par near_pump_7d=1 : {n_waves_flagged}")

print("RESUME POUR REDACTION 7.4 / 7.6")
print(f"- BUY evaluables (couverture marche) : {n_buys_eval}")
print(f"- BUY signales (near_pump_7d=1)      : {n_buys_flagged} "
      f"({100 * n_buys_flagged / max(n_buys_eval, 1):.1f}%)")
print("- Coins concernes par au moins 1 alerte :")
print(coverage_by_coin.to_string())
print(f"- Lots WAVES post-pic evalues : {len(waves_buys)}, signales : {n_waves_flagged}")

print("Fichiers generes :")
print(" -", RESULTS / "level4_alert_coverage_summary.csv")
print(" -", RESULTS / "level4_alert_coverage_by_coin.csv")
print(" -", RESULTS / "level4_waves_chronology_v2.csv")
