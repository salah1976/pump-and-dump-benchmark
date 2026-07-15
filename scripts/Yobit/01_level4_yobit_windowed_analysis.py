import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from scipy.stats import mannwhitneyu, spearmanr

PROJECT = Path(__file__).resolve().parents[1]

DATA = PROJECT / "data" / "processed"

RESULTS = PROJECT / "results" / "tables"
FIGURES = PROJECT / "results" / "figures"

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

enriched = pd.read_csv(RESULTS / "yobit_trades_enriched_v2.csv")

enriched["time"] = pd.to_datetime(enriched["time"])
enriched["date"] = pd.to_datetime(enriched["date"])

print("Trades enrichis (v2) :", enriched.shape)

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

print("Marche journalier agrege :", market_daily.shape)

pump_days = market_daily[market_daily["label_pump"] == 1][["coin", "date"]].copy()

pump_days = pump_days.rename(columns={"date": "pump_date"})

print("Nombre total de (coin, jour) labellises pump dans la couverture :", len(pump_days))

enriched = enriched.sort_values("date").reset_index(drop=True)

near_pump_flags = []

for coin, group in enriched.groupby("coin"):

    coin_pumps = pump_days[pump_days["coin"] == coin]

    if coin_pumps.empty:
        near_pump_flags.append(pd.Series(0, index=group.index))
        continue

    flags = group["date"].apply(
        lambda d: int(((coin_pumps["pump_date"] - d).dt.days.abs() <= 7).any())
    )

    near_pump_flags.append(flags)

enriched["near_pump_7d"] = pd.concat(near_pump_flags).sort_index()

print("Trades near_pump_7d = 1 :", enriched["near_pump_7d"].sum())
print("Trades near_pump_7d = 0 :", (enriched["near_pump_7d"] == 0).sum())

buys = enriched[
    (enriched["type"] == "BUY") & enriched["premium_pct"].notna()
].copy()

near_buys = buys[buys["near_pump_7d"] == 1]["premium_pct"]

far_buys = buys[buys["near_pump_7d"] == 0]["premium_pct"]

print("Achats pres d'un pump (+/-7j) : n =", len(near_buys))
print("Achats loin d'un pump : n =", len(far_buys))

window_test = {}

if len(near_buys) >= 5 and len(far_buys) >= 5:

    u_stat, p_mw = mannwhitneyu(near_buys, far_buys, alternative="greater")

    print("Mann-Whitney (near > far) : U =", round(u_stat, 1), "| p =", round(p_mw, 4))

    window_test = {
        "n_near": len(near_buys), "n_far": len(far_buys),
        "median_near": float(near_buys.median()), "median_far": float(far_buys.median()),
        "U": float(u_stat), "p_value": float(p_mw),
    }

else:
    print("Pas assez d'observations meme avec fenetre elargie.")

pd.DataFrame([window_test]).to_csv(RESULTS / "yobit_windowed_premium_test.csv", index=False)

market_daily["forward_return_7d"] = (
    market_daily
    .groupby("coin")["market_close"]
    .transform(lambda s: s.shift(-7) / s - 1) * 100
)

fwd_ref = market_daily[["coin", "date", "forward_return_7d"]]

enriched = enriched.merge(fwd_ref, on=["coin", "date"], how="left")

corr_data = enriched[
    (enriched["type"] == "BUY") &
    enriched["premium_pct"].notna() &
    enriched["forward_return_7d"].notna()
]

print("Trades BUY avec premium ET rendement futur disponibles :", len(corr_data))

correlation_result = {}

if len(corr_data) >= 10:

    rho, p_corr = spearmanr(corr_data["premium_pct"], corr_data["forward_return_7d"])

    print("Spearman rho (premium vs rendement J+7) :", round(rho, 4), "| p =", round(p_corr, 4))

    correlation_result = {
        "n": len(corr_data), "spearman_rho": float(rho), "p_value": float(p_corr),
    }

else:
    print("Pas assez d'observations pour la correlation (n < 10).")

pd.DataFrame([correlation_result]).to_csv(
    RESULTS / "yobit_premium_forward_return_correlation.csv", index=False
)

enriched["year"] = enriched["time"].dt.year

enriched["signed_total"] = np.where(
    enriched["type"] == "SELL", enriched["total"], -enriched["total"]
)

yearly = (
    enriched
    .groupby("year")
    .agg(
        n_trades=("time", "size"),
        n_buy=("type", lambda s: (s == "BUY").sum()),
        n_sell=("type", lambda s: (s == "SELL").sum()),
        volume_usd=("total", "sum"),
        net_cashflow_usd=("signed_total", "sum"),
        distinct_coins=("coin", "nunique"),
    )
)

print(yearly)

yearly.to_csv(RESULTS / "yobit_yearly_breakdown.csv")

enriched.to_csv(RESULTS / "yobit_trades_enriched_v3.csv", index=False)

plt.figure(figsize=(10, 5))

plt.bar(yearly.index.astype(str), yearly["n_trades"], color="steelblue")

plt.xlabel("Annee")
plt.ylabel("Nombre de trades")
plt.title("Activite de trading par annee (2016-2025)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(FIGURES / "yobit_yearly_activity.png", dpi=300)

plt.close()

if len(corr_data) >= 10:

    plt.figure(figsize=(8, 6))

    plt.scatter(corr_data["premium_pct"], corr_data["forward_return_7d"], alpha=0.5, color="darkorange")

    plt.axhline(0, color="black", linewidth=0.8)
    plt.axvline(0, color="black", linewidth=0.8)

    plt.xlabel("Premium paye a l'achat (%)")
    plt.ylabel("Rendement du coin a J+7 (%)")
    plt.title(f"Premium paye vs rendement futur (Spearman rho={rho:.3f}, p={p_corr:.4f})")
    plt.tight_layout()

    plt.savefig(FIGURES / "yobit_premium_vs_forward_return.png", dpi=300)

    plt.close()

if len(near_buys) > 0 or len(far_buys) > 0:

    plt.figure(figsize=(8, 5))

    if len(far_buys) > 0:
        plt.hist(far_buys.dropna(), bins=30, alpha=0.6, label="Loin d'un pump (>7j)", color="steelblue")

    if len(near_buys) > 0:
        plt.hist(near_buys.dropna(), bins=30, alpha=0.6, label="Pres d'un pump (+/-7j)", color="firebrick")

    plt.axvline(0, color="black", linewidth=0.8)

    plt.xlabel("Premium vs prix de marche (%)")
    plt.ylabel("Nombre de trades")
    plt.title("Premium paye, fenetre elargie +/-7 jours autour d'un pump")
    plt.legend()
    plt.tight_layout()

    plt.savefig(FIGURES / "yobit_premium_windowed_distribution.png", dpi=300)

    plt.close()

print("FINISHED")
print("Tables :", RESULTS)
print("Figures:", FIGURES)
