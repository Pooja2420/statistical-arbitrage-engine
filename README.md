# Statistical Arbitrage Engine

A production-grade pairs trading system that identifies cointegrated stock pairs, generates trading signals using Kalman filters and mean-reversion strategies, and runs realistic backtests with transaction costs and slippage modelling.

## Methodology

### Why Pairs Trading?

Two stocks in the same sector often share common risk factors (interest rates, commodity prices, macro cycles). Even though each stock's price follows a random walk on its own, a linear combination of the two prices may be *stationary* — it reverts to a long-run mean. This property is called **cointegration**, and it is the statistical foundation of pairs trading.

When the spread between the two prices diverges from its historical mean, we expect it to revert. We go long the cheap leg and short the expensive leg, profiting when the spread collapses.

### Step 1 — Cointegration Testing

We test every pair in our universe using two complementary methods:

**Engle-Granger (1987)**
Run OLS regression of log(A) on log(B) to estimate the hedge ratio β, then run an Augmented Dickey-Fuller test on the residuals. A low p-value (< 5 %) indicates the residuals are stationary, confirming cointegration.

**Johansen (1988)**
Likelihood-ratio test on a Vector Error Correction Model. More powerful than Engle-Granger, and directly estimates the cointegrating rank. We require the trace statistic to exceed the 95 % critical value.

**Half-life filter**
We model the spread as an Ornstein-Uhlenbeck process and estimate its mean-reversion half-life via OLS. Pairs with a half-life shorter than 5 days (too noisy, high turnover) or longer than 60 days (too slow to exploit) are rejected.

A pair must pass all three filters to be included in the portfolio.

### Step 2 — Kalman Filter (Dynamic Hedge Ratio)

OLS gives a *static* hedge ratio fitted to the entire history. In practice, the relationship between two stocks drifts over time. We use a **Kalman Filter** to estimate a time-varying hedge ratio.

State-space model:
```
Observation:  y_t = β_t · x_t + ε_t     ε ~ N(0, R)
Transition:   β_t = β_{t-1} + η_t        η ~ N(0, Q)
```

Where `y_t = log(A_t)`, `x_t = log(B_t)`, and `β_t` is the hedge ratio. The filter updates the hedge ratio estimate every day as new prices arrive.

### Step 3 — Signal Generation

We z-score the Kalman spread using a rolling window:

```
z_t = (spread_t - μ_rolling) / σ_rolling
```

Trading rules:
| z-score | Action |
|---------|--------|
| z < -2.0 | Open long spread (buy A, sell B) |
| z > +2.0 | Open short spread (sell A, buy B) |
| \|z\| < 0.5 | Close position |
| \|z\| > 3.5 | Emergency stop-loss |

### Step 4 — Backtesting

The backtesting engine enforces:
- **Next-day execution** — signals generated at close are executed at the next day's open (no look-ahead bias)
- **Transaction costs** — configurable basis points deducted on each trade (default 5 bps one-way)
- **Slippage** — additional market-impact cost (default 2 bps)
- **Position stop-loss** — close position if unrealised loss exceeds 5 % of notional
- **Maximum holding period** — force-close after 30 days

## Target Metrics

| Metric | Target |
|--------|--------|
| Sharpe Ratio | > 1.5 (target 1.8-2.2) |
| Maximum Drawdown | < 15 % |
| Win Rate | > 55 % |
| Cointegrated pairs | 15-25 (p-value < 0.05) |

## Project Structure

```
statistical-arbitrage-engine/
├── app.py                        # Streamlit dashboard (4 tabs)
├── requirements.txt
├── src/
│   ├── data/
│   │   └── data_loader.py        # yfinance downloader + caching
│   ├── signals/
│   │   ├── cointegration.py      # EG + Johansen tests, half-life, PairScanner
│   │   └── kalman_filter.py      # Kalman Filter + MeanReversionSignal
│   ├── backtesting/
│   │   └── engine.py             # Realistic backtest engine
│   ├── risk/
│   │   └── risk_manager.py       # VaR, Sharpe, Sortino, Calmar, Kelly sizing
│   └── utils/
│       └── plotting.py           # Plotly chart functions
├── notebooks/
│   └── 01_data_exploration.ipynb
├── tests/
│   ├── test_cointegration.py     # 11 tests
│   ├── test_kalman_filter.py     # 10 tests
│   └── test_backtest.py          # 16 tests
├── data/
│   ├── raw/                      # Downloaded parquet files
│   └── processed/
├── results/
│   ├── backtests/
│   └── plots/
└── docs/
```

## Quick Start

### 1. Set up the environment

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run the Streamlit dashboard

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

**Tab 1 — Pair Scanner:** Select tickers from the S&P 500 universe and run cointegration tests. The scanner downloads prices from Yahoo Finance (cached to disk), tests all pairs, and shows the ones that pass both EG and Johansen tests.

**Tab 2 — Signal Monitor:** For any cointegrated pair, visualise the Kalman-filtered z-score, entry/exit signal markers, and the time-varying hedge ratio.

**Tab 3 — Backtest:** Run a full realistic backtest on any selected pair. View the equity curve, drawdown, return distribution, trade log, and key performance metrics.

**Tab 4 — Risk Report:** Sharpe, Sortino, Calmar, VaR, CVaR, rolling Sharpe chart.

### 3. Run the tests

```bash
python -m pytest tests/ -v
```

All 37 tests should pass in under 2 seconds.

### 4. Use the API directly

```python
from src.data.data_loader import DataLoader
from src.signals.cointegration import PairScanner
from src.signals.kalman_filter import KalmanFilterHedge, MeanReversionSignal
from src.backtesting.engine import PairsBacktestEngine, BacktestConfig
from src.risk.risk_manager import RiskAnalytics

# 1. Download data
loader = DataLoader(start_date="2020-01-01", end_date="2024-12-31")
prices = loader.fetch_sp500_universe()

# 2. Find cointegrated pairs
scanner = PairScanner(prices, eg_pvalue_threshold=0.05)
report = scanner.scan()
print(f"Found {len(report.cointegrated_pairs)} pairs")

# 3. Fit Kalman filter on a pair
pair = report.cointegrated_pairs[0]
pa, pb = prices[pair.ticker_a], prices[pair.ticker_b]
kf = KalmanFilterHedge(delta=1e-4, zscore_window=30)
kf_output = kf.fit(pa, pb)

# 4. Generate signals
signals = MeanReversionSignal(entry_z=2.0, exit_z=0.5).generate(kf_output)

# 5. Backtest
cfg = BacktestConfig(initial_capital=1_000_000, transaction_cost_bps=5, slippage_bps=2)
metrics = PairsBacktestEngine(cfg).run(pa, pb, signals, kf_output["beta"], pair.pair_name)
print(metrics.to_dict())

# 6. Risk analytics
ra = RiskAnalytics(metrics.daily_returns)
print(ra.summary())
```

## Key Concepts (Beginner-Friendly)

**Cointegration** — Two non-stationary series whose difference (or linear combination) is stationary. Think of a drunk person and their dog on a long leash: each wanders randomly, but they stay close together.

**Kalman Filter** — An algorithm that estimates a hidden state (the hedge ratio) from noisy observations (prices). Unlike a moving average, it weights recent observations optimally based on a noise model.

**Z-score** — How many standard deviations the current spread is from its rolling mean. Z = 0 means the spread is at its average; Z = 2 means it is unusually wide.

**Sharpe Ratio** — Annualised return divided by annualised volatility. A Sharpe > 1.5 is considered good for a systematic strategy.

**Maximum Drawdown** — The largest peak-to-trough loss in the equity curve. Measures worst-case pain a strategy inflicts on its holder.

**Half-life** — How quickly the spread reverts to its mean after a shock. Shorter half-life = more trading opportunities but also higher transaction costs.

## Data Sources

- **Yahoo Finance** via `yfinance` (free, no API key required)
- Universe: 50 liquid S&P 500 stocks across 8 sectors

## References

1. Engle, R. F., & Granger, C. W. J. (1987). Co-integration and error correction: Representation, estimation, and testing. *Econometrica*, 55(2), 251-276.
2. Johansen, S. (1988). Statistical analysis of cointegration vectors. *Journal of Economic Dynamics and Control*, 12(2-3), 231-254.
3. Chan, E. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*. Wiley.
4. Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *Journal of Basic Engineering*, 82(1), 35-45.
