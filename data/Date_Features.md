# data/FEATURES.md

Feature dictionary for the pump-and-dump detection pipelines. Feature
names differ between the multi-exchange and Yahoo Finance datasets even
when the underlying definition is the same — this file maps them.

## Label (shared definition, Formula 1)

Applied identically across the multi-exchange and Yahoo Finance datasets:

```
label = 1  if  future_return_7d  >= 30%
           AND future_volume_ratio_7d >= 3.0x
```

- Horizon: 7 days
- `future_return_7d = (close[t+7] / close[t] - 1) * 100`
- `future_volume_ratio_7d = volume[t+7] / volume[t]`

## Feature equivalence table

| Concept | Multi-exchange column | Yahoo Finance column | Definition |
|---|---|---|---|
| 1-day return | return_1d | return_1d | `pct_change(1) * 100` |
| 3-day return | return_3d | return_3d | `pct_change(3) * 100` |
| 7-day return | return_7d | return_7d | `pct_change(7) * 100` |
| 14-day return | return_14d | return_14d | `pct_change(14) * 100` |
| 30-day return | — | return_30d | `pct_change(30) * 100` |
| Volatility | volatility_7d | volatility30 | rolling std of daily returns (window differs: 7d vs 30d) |
| Volume moving average | volume_ma7 (intermediate) | vol_ma7, vol_ma30 | rolling mean of volume |
| Volume ratio | volume_ratio, volume_ratio20 | vol_ratio7, vol_ratio30 | `volume / volume_ma{window}` |
| Price moving average | ma7 (intermediate) | ma7, ma30 | rolling mean of close |
| Price / MA ratio | price_ma_ratio | price_ma_ratio | multi-exchange: `close / ma7`; Yahoo: `ma7 / ma30` — **not the same formula, see note below** |
| Position vs 30d high | dist_max20 | price_vs_max30 | multi-exchange: `close / rolling_max20`; Yahoo: `close / rolling_max30` (window differs: 20d vs 30d) |
| Position vs 30d low | dist_min20 | price_vs_min30 | same pattern, window differs |
| RSI | rsi14 | rsi14 | standard 14-day RSI |
| Momentum | momentum_7d, momentum_14d | momentum | multi-exchange: `close.diff(7)`, `close.diff(14)` (absolute); Yahoo: `ma5 / ma20` (ratio) — **different definitions, not directly comparable** |
| Candle range | (not in canonical list) | candle_range | `(high - low) / close` |
| Upper shadow | upper_shadow_norm | upper_shadow | `(high - max(open, close)) / close` |
| Lower shadow | lower_shadow_norm | lower_shadow | `(min(open, close) - low) / close` |
| Body ratio | body_ratio | — | `body / (high - low + 1e-9)`, where `body = abs(close - open)` |
| Close position in range | close_position | — | `(close - low) / (high - low + 1e-9)` |
| EMA 20 / 50 | ema20, ema50 | — | exponential moving average, span=20 / span=50 |
| ATR 14 | atr14 | — | 14-day average true range |
| Bollinger band width | bb_width | — | `(upper_band - lower_band) / ma20`, bands = `ma20 +/- 2*std20` |

## Important note on shadow normalization (multi-exchange)

`upper_shadow` and `lower_shadow` were originally computed in absolute
price units (dollars), which created a coin-identity leakage artifact:
BTC/ETH (thousands of dollars) dominated the signal over small altcoins
(fractions of a cent), independent of any real pump dynamic. This was
confirmed via Mann-Whitney U / Cliff's delta (effect size collapsed by
more than 60% after normalization). The corrected, published features
are `upper_shadow_norm` and `lower_shadow_norm`, divided by closing
price — same formula Yahoo Finance used from the start.

## Important note on `price_ma_ratio` and `momentum`

These two features are **not defined identically** across the two
pipelines despite sharing a column name. Do not merge or directly
compare their raw values across datasets without accounting for this.

## Full canonical feature lists

**Multi-exchange (21 features)**, used in `dataset_learning_v2.csv`:
```
return_1d, return_3d, return_7d, return_14d, volatility_7d,
volume_ratio, volume_ratio20, price_ma_ratio, body_ratio,
close_position, upper_shadow_norm, lower_shadow_norm,
momentum_7d, momentum_14d, ema20, ema50, rsi14, atr14, bb_width,
dist_max20, dist_min20
```

**Yahoo Finance (20 features)**, used in `YahooFinance_dataset_learning_v2.csv`:
```
return_1d, return_3d, return_7d, return_14d, return_30d,
vol_ma7, vol_ma30, vol_ratio7, vol_ratio30, volatility30,
ma7, ma30, price_ma_ratio, price_vs_max30, price_vs_min30,
rsi14, momentum, candle_range, upper_shadow, lower_shadow
```

Yahoo Finance evaluation additionally uses two reduced subsets:
- **Standard (13 features)**: return_1d, return_3d, return_7d, vol_ma7,
  vol_ratio7, volatility30, ma7, price_ma_ratio, rsi14, momentum,
  candle_range, upper_shadow, lower_shadow
- **Ultra (8 features)**: return_1d, vol_ma7, ma7, rsi14, momentum,
  candle_range, upper_shadow, lower_shadow
