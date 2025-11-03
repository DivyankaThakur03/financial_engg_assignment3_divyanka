# PA3: Automated NVDA Trading Strategy

## Overview

This repository contains an automated trading strategy for NVIDIA (NVDA) that combines momentum and mean-reversion approaches with regime-aware filtering. The strategy achieved a 6.3% CAGR with a 7.4% maximum drawdown compared to NVDA buy-and-hold's 71.9% drawdown over 2015-2025.

## Repository Structure
```
assignment3/
├── assign3.py              # Main strategy implementation
├── report.pdf              # Comprehensive research report
├── README.md               # This file
├── data/
│   ├── price_data.csv      # Historical price data
│   └── performance_summary.csv
└── figures/
    ├── equity_curves.png
    ├── strategy_drawdown.png
    ├── signals.png
    ├── rolling_sharpe.png
    ├── monthly_returns.png
    └── return_distribution.png
```

## Installation

### Requirements
- Python 3.8+
- pandas
- numpy
- yfinance
- matplotlib
- scipy

### Setup
```bash
# Clone repository
git clone "repo name"
cd assignment3

# Install dependencies
pip install pandas numpy yfinance matplotlib scipy

# Run strategy
python3 assign3.py
```

## Strategy Description

**Two-Sleeve Approach:**
- **Trend Sleeve (70%):** Momentum using 50/200 MA filter + 20-day breakouts
- **Mean-Reversion Sleeve (30%):** Z-score < -2 entries, exit at z > -0.5

**Risk Management:**
- VIX < 30 regime filter
- ATR-based volatility targeting (15% annualized)
- Transaction costs: 15 bps per trade
- Management fee: 1% annual

## Key Results

| Metric | Strategy | NVDA B&H | SPY B&H |
|--------|----------|----------|---------|
| CAGR | 6.3% | 55.4% | 11.8% |
| Sharpe | 0.37 | 1.06 | 0.44 |
| Max DD | -7.4% | -71.9% | -35.7% |

**Performance by VIX Regime:**
- Low VIX (<15): 13.8% CAGR, Sharpe 1.54
- Medium VIX (15-25): 4.3% CAGR, Sharpe 0.06
- High VIX (>25): -6.0% CAGR, Sharpe -1.67

## Usage

### Run Full Analysis
```bash
python3 assign3.py
```

This will:
1. Download NVDA, SPY, and VIX data from 2015-present
2. Generate trading signals
3. Backtest strategy with costs
4. Create performance metrics and visualizations
5. Save results to `data/` and `figures/`

### Output Files
- `data/price_data.csv` - Raw price data
- `data/performance_summary.csv` - Performance metrics table
- `figures/*.png` - Six visualization charts

## Key Parameters
```python
TICKER = "NVDA"           # Primary asset
START = "2015-01-01"      # Backtest start date
MA_SHORT = 50             # Short moving average
MA_LONG = 200             # Long moving average
VIX_THRESHOLD = 30        # Regime filter cutoff
Z_ENTRY = -2.0            # Mean reversion entry
TREND_WEIGHT = 0.7        # Trend sleeve allocation
MR_WEIGHT = 0.3           # Mean reversion allocation
TC_BPS = 15               # Transaction cost (bps)
```

## Use of AI Tools

AI assistance (ChatGPT/GPT-4) was primarily used for refining grammar in the report. All core model choices, parameter definitions, mathematical implementation, and result interpretations were executed and validated by the author.
