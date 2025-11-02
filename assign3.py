"""
PA3: Automated Trading Strategy - NVDA Momentum with Mean Reversion
Implements a two-sleeve approach with regime awareness
"""

# --- Imports ---
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Create figures directory
import os
os.makedirs('figures', exist_ok=True)
os.makedirs('data', exist_ok=True)

# --- Parameters ---
TICKER = "NVDA"
BENCH = "SPY"
START = "2015-01-01"
END = None  # Today
RF_ANNUAL = 0.04  # 4% risk-free rate

# Strategy parameters
MA_SHORT = 50
MA_LONG = 200
MOM_LOOKBACK = 20  # Breakout lookback
Z_WINDOW = 20  # Mean reversion window
Z_ENTRY = -2.0  # Z-score entry threshold
VIX_THRESHOLD = 30  # Regime filter

# Risk parameters
ATR_PERIOD = 14
TARGET_VOL = 0.15  # 15% annualized
TC_BPS = 15  # 15 bps transaction cost per trade
MGMT_FEE = 0.01  # 1% annual management fee

# Position weights
TREND_WEIGHT = 0.7
MR_WEIGHT = 0.3

print("=" * 80)
print("PA3: NVDA Automated Trading Strategy")
print("=" * 80)
print(f"\nParameters:")
print(f"  Ticker: {TICKER}")
print(f"  Period: {START} to present")
print(f"  Trend weight: {TREND_WEIGHT:.0%}, Mean-reversion weight: {MR_WEIGHT:.0%}")
print(f"  Transaction costs: {TC_BPS} bps, Management fee: {MGMT_FEE:.1%}")

# --- Data Download ---
print("\n" + "-" * 80)
print("Downloading data...")

tickers = [TICKER, BENCH, "^VIX"]
data = yf.download(tickers, start=START, end=END, progress=False)

# Extract close prices
px = data["Close"].copy()
px = px.dropna()

# Save data
px.to_csv('data/price_data.csv')
print(f"  Downloaded {len(px)} days of data")
print(f"  Period: {px.index[0].date()} to {px.index[-1].date()}")

# --- Calculate Returns ---
ret = np.log(px / px.shift(1)).dropna()
rf_daily = (1 + RF_ANNUAL) ** (1/252) - 1

# Extract individual series
p = px[TICKER].copy()
vix = px["^VIX"].copy()
nv_ret = ret[TICKER].copy()
spy_ret = ret[BENCH].copy()

# --- Feature Engineering ---
print("\n" + "-" * 80)
print("Engineering features...")

# Moving averages
ma_short = p.rolling(MA_SHORT).mean()
ma_long = p.rolling(MA_LONG).mean()
ma_slope = ma_long.diff(20) > 0

# ATR for volatility
high = data["High"][TICKER]
low = data["Low"][TICKER]
tr1 = high - low
tr2 = abs(high - p.shift(1))
tr3 = abs(low - p.shift(1))
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
atr = tr.rolling(ATR_PERIOD).mean()

# Momentum features
mom_breakout = p == p.rolling(MOM_LOOKBACK).max()

# Mean reversion features
rolling_mean = p.rolling(Z_WINDOW).mean()
rolling_std = p.rolling(Z_WINDOW).std()
z_score = (p - rolling_mean) / rolling_std

print(f"  MA Short: {MA_SHORT}, MA Long: {MA_LONG}")
print(f"  ATR Period: {ATR_PERIOD}")
print(f"  Z-score window: {Z_WINDOW}")

# --- Signal Generation ---
print("\n" + "-" * 80)
print("Generating signals...")

# TREND SLEEVE: Momentum/Breakout
trend_cond = (p > ma_long) & (ma_short > ma_long) & mom_breakout
signal_trend = trend_cond.astype(float)

# MEAN REVERSION SLEEVE: Z-score
mr_entry = z_score < Z_ENTRY
mr_exit = z_score > -0.5
signal_mr = pd.Series(0.0, index=p.index)
position = False
for i in range(len(signal_mr)):
    if mr_entry.iloc[i] and not position:
        position = True
    elif mr_exit.iloc[i] and position:
        position = False
    signal_mr.iloc[i] = 1.0 if position else 0.0

# REGIME FILTER: VIX and trend quality
regime_ok = (vix < VIX_THRESHOLD) & (p > ma_long)

# Combine sleeves
signal_combined = (TREND_WEIGHT * signal_trend + MR_WEIGHT * signal_mr)
signal_combined = signal_combined * regime_ok.astype(float)

# VOLATILITY TARGETING: Scale by ATR
realized_vol = nv_ret.rolling(20).std() * np.sqrt(252)
vol_scalar = (TARGET_VOL / realized_vol).clip(0.5, 2.0)
signal = (signal_combined * vol_scalar).clip(0, 1)

print(f"  Trend signals: {signal_trend.sum():.0f} days")
print(f"  Mean-reversion signals: {signal_mr.sum():.0f} days")
print(f"  Combined signals (after regime filter): {(signal > 0).sum():.0f} days")

# --- Trading Costs ---
position_change = signal.diff().fillna(signal).abs()
trade_cost = position_change * (TC_BPS / 10000.0)
mgmt_cost = (MGMT_FEE / 252) * signal.shift(1).fillna(0)

# --- Strategy Returns ---
strat_gross = signal.shift(1).fillna(0) * nv_ret + (1 - signal.shift(1).fillna(0)) * rf_daily
strat_net = strat_gross - trade_cost - mgmt_cost

# Benchmark returns
nvda_bh = nv_ret
spy_bh = spy_ret

# --- Performance Metrics ---
print("\n" + "-" * 80)
print("Calculating performance metrics...")

def equity_curve(r):
    return (1 + r).cumprod()

def max_drawdown(curve):
    roll_max = curve.cummax()
    dd = curve / roll_max - 1.0
    return dd.min(), dd

def annualize_metrics(r, rf_annual=RF_ANNUAL):
    if len(r) == 0:
        return 0, 0, 0
    cagr = (1 + r).prod() ** (252 / len(r)) - 1
    vol = r.std() * np.sqrt(252)
    sharpe = (cagr - rf_annual) / (vol + 1e-12)
    return cagr, vol, sharpe

def sortino_ratio(r, rf_annual=RF_ANNUAL):
    if len(r) == 0:
        return 0
    rf_daily = (1 + rf_annual) ** (1/252) - 1
    excess = r - rf_daily
    downside = excess[excess < 0].std() * np.sqrt(252)
    if downside == 0:
        return 0
    return (excess.mean() * 252) / downside

def calmar_ratio(r, mdd):
    cagr = (1 + r).prod() ** (252 / len(r)) - 1
    return cagr / abs(mdd) if mdd != 0 else 0

def hit_rate(r):
    if len(r) == 0:
        return 0
    return (r > 0).mean()

# Calculate equity curves
ec_strat = equity_curve(strat_net)
ec_nvda = equity_curve(nvda_bh)
ec_spy = equity_curve(spy_bh)

# Calculate metrics for all strategies
cagr_s, vol_s, shp_s = annualize_metrics(strat_net)
cagr_n, vol_n, shp_n = annualize_metrics(nvda_bh)
cagr_p, vol_p, shp_p = annualize_metrics(spy_bh)

mdd_s, dd_s = max_drawdown(ec_strat)
mdd_n, dd_n = max_drawdown(ec_nvda)
mdd_p, dd_p = max_drawdown(ec_spy)

sortino_s = sortino_ratio(strat_net)
sortino_n = sortino_ratio(nvda_bh)
sortino_p = sortino_ratio(spy_bh)

calmar_s = calmar_ratio(strat_net, mdd_s)
calmar_n = calmar_ratio(nvda_bh, mdd_n)
calmar_p = calmar_ratio(spy_bh, mdd_p)

hit_s = hit_rate(strat_net[signal.shift(1) > 0])
hit_n = hit_rate(nvda_bh)
hit_p = hit_rate(spy_bh)

# Turnover
turnover = position_change.sum() / len(signal) * 252

# Alpha/Beta vs SPY
excess_strat = strat_net - rf_daily
excess_spy = spy_bh - rf_daily
valid_idx = excess_strat.notna() & excess_spy.notna()
if valid_idx.sum() > 0:
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        excess_spy[valid_idx],
        excess_strat[valid_idx]
    )
    alpha_annual = intercept * 252
    beta = slope
else:
    alpha_annual, beta = 0, 0

# Summary table
summary = pd.DataFrame({
    "Strategy": ["NVDA Two-Sleeve", "NVDA Buy&Hold", "SPY Buy&Hold"],
    "CAGR": [cagr_s, cagr_n, cagr_p],
    "Vol": [vol_s, vol_n, vol_p],
    "Sharpe": [shp_s, shp_n, shp_p],
    "Sortino": [sortino_s, sortino_n, sortino_p],
    "Calmar": [calmar_s, calmar_n, calmar_p],
    "MaxDD": [mdd_s, mdd_n, mdd_p],
    "HitRate": [hit_s, hit_n, hit_p],
}).set_index("Strategy")

print("\n" + "=" * 80)
print("PERFORMANCE SUMMARY")
print("=" * 80)
print(summary.round(3).to_string())

print("\n" + "-" * 80)
print("Additional Metrics:")
print(f"  Alpha (vs SPY): {alpha_annual:.2%}")
print(f"  Beta (vs SPY): {beta:.2f}")
print(f"  Annual Turnover: {turnover:.1f}x")
print(f"  Total Trades: {int(position_change[position_change > 0].count())}")
print(f"  Avg Position Size: {signal[signal > 0].mean():.1%}")

# --- VIX Regime Analysis ---
print("\n" + "-" * 80)
print("Performance by VIX Regime:")

vix_low = strat_net[vix < 15]
vix_med = strat_net[(vix >= 15) & (vix < 25)]
vix_high = strat_net[vix >= 25]

for name, subset in [("Low (<15)", vix_low), ("Med (15-25)", vix_med), ("High (>25)", vix_high)]:
    if len(subset) > 20:
        c, v, s = annualize_metrics(subset)
        print(f"  VIX {name:12s}: CAGR={c:6.2%}, Vol={v:6.2%}, Sharpe={s:5.2f}, Days={len(subset):4d}")

# --- Visualizations ---
print("\n" + "-" * 80)
print("Creating visualizations...")

# 1. Equity Curves
plt.figure(figsize=(12, 6))
plt.plot(ec_strat.index, ec_strat.values, label="Two-Sleeve Strategy", linewidth=2)
plt.plot(ec_nvda.index, ec_nvda.values, label="NVDA Buy & Hold", alpha=0.7)
plt.plot(ec_spy.index, ec_spy.values, label="SPY Buy & Hold", alpha=0.7)
plt.title("Growth of $1 Investment", fontsize=14, fontweight='bold')
plt.xlabel("Date")
plt.ylabel("Portfolio Value ($)")
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/equity_curves.png", dpi=160)
print("  Saved: figures/equity_curves.png")

# 2. Drawdown
plt.figure(figsize=(12, 4))
plt.fill_between(dd_s.index, 0, dd_s.values * 100, alpha=0.3, color='red')
plt.plot(dd_s.index, dd_s.values * 100, color='darkred', linewidth=1)
plt.title("Strategy Drawdown", fontsize=14, fontweight='bold')
plt.xlabel("Date")
plt.ylabel("Drawdown (%)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/strategy_drawdown.png", dpi=160)
print("  Saved: figures/strategy_drawdown.png")

# 3. Signal Overlay
plt.figure(figsize=(12, 6))
plt.plot(p.index, p.values, label="NVDA Price", linewidth=1.5, color='black')
plt.plot(ma_long.index, ma_long.values, label=f"{MA_LONG}D MA", alpha=0.7, color='blue')
plt.fill_between(p.index, p.min(), p.max(), 
                 where=signal > 0.5, alpha=0.15, color='green', label='Long Signal')
plt.title("Trading Signals and Price", fontsize=14, fontweight='bold')
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/signals.png", dpi=160)
print("  Saved: figures/signals.png")

# 4. Rolling Sharpe
rolling_sharpe = strat_net.rolling(252).apply(
    lambda x: (x.mean() * 252 - RF_ANNUAL) / (x.std() * np.sqrt(252) + 1e-12)
)
plt.figure(figsize=(12, 4))
plt.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=1.5)
plt.axhline(0, color='red', linestyle='--', alpha=0.5)
plt.axhline(shp_s, color='green', linestyle='--', alpha=0.5, label=f'Average: {shp_s:.2f}')
plt.title("Rolling 1-Year Sharpe Ratio", fontsize=14, fontweight='bold')
plt.xlabel("Date")
plt.ylabel("Sharpe Ratio")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/rolling_sharpe.png", dpi=160)
print("  Saved: figures/rolling_sharpe.png")

# 5. Monthly Returns Heatmap
monthly_ret = strat_net.resample('M').apply(lambda x: (1 + x).prod() - 1)
monthly_ret_pct = monthly_ret * 100
returns_pivot = monthly_ret_pct.to_frame('ret')
returns_pivot['year'] = returns_pivot.index.year
returns_pivot['month'] = returns_pivot.index.month
heatmap_data = returns_pivot.pivot(index='year', columns='month', values='ret')

plt.figure(figsize=(12, 8))
plt.imshow(heatmap_data.values, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
plt.colorbar(label='Return (%)')
plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
plt.title("Monthly Returns Heatmap (%)", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("figures/monthly_returns.png", dpi=160)
print("  Saved: figures/monthly_returns.png")

# 6. Return Distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(strat_net.dropna() * 100, bins=50, alpha=0.7, edgecolor='black')
plt.axvline(strat_net.mean() * 100, color='red', linestyle='--', label=f'Mean: {strat_net.mean()*100:.3f}%')
plt.title("Daily Return Distribution", fontsize=12, fontweight='bold')
plt.xlabel("Daily Return (%)")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
stats.probplot(strat_net.dropna(), dist="norm", plot=plt)
plt.title("Q-Q Plot (Normality Check)", fontsize=12, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/return_distribution.png", dpi=160)
print("  Saved: figures/return_distribution.png")

# --- Save Results ---
summary.to_csv('data/performance_summary.csv')
print("\n" + "-" * 80)
print("Results saved to data/performance_summary.csv")

print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)