"""Unit tests for the backtesting engine and risk analytics."""

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import BacktestConfig, PairsBacktestEngine, PerformanceMetrics
from src.risk.risk_manager import RiskAnalytics, position_sizing_kelly

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 500
RNG = np.random.default_rng(7)


@pytest.fixture
def simple_pair() -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Minimal synthetic pair for backtest testing."""
    dates = pd.bdate_range("2022-01-03", periods=N)
    common = np.cumsum(RNG.normal(0, 0.01, N))
    beta_true = 1.1
    log_b = 5.0 + common + RNG.normal(0, 0.005, N)
    log_a = 4.9 + beta_true * common + RNG.normal(0, 0.005, N)
    pa = pd.Series(np.exp(log_a), index=dates, name="A")
    pb = pd.Series(np.exp(log_b), index=dates, name="B")
    beta_series = pd.Series(np.full(N, beta_true), index=dates, name="beta")
    # Alternate signals: long for 20 days, flat 5, short 20, flat 5, ...
    raw = []
    day = 0
    while day < N:
        raw.extend([1] * min(20, N - day))
        day += 20
        raw.extend([0] * min(5, N - day))
        day += 5
        raw.extend([-1] * min(20, N - day))
        day += 20
        raw.extend([0] * min(5, N - day))
        day += 5
    signals = pd.Series(raw[:N], index=dates, name="signal")
    return pa, pb, signals, beta_series


# ---------------------------------------------------------------------------
# PairsBacktestEngine
# ---------------------------------------------------------------------------


def test_backtest_returns_metrics(simple_pair):
    pa, pb, signals, beta = simple_pair
    engine = PairsBacktestEngine()
    metrics = engine.run(pa, pb, signals, beta, pair_name="A/B")
    assert isinstance(metrics, PerformanceMetrics)


def test_equity_curve_length(simple_pair):
    pa, pb, signals, beta = simple_pair
    engine = PairsBacktestEngine()
    metrics = engine.run(pa, pb, signals, beta)
    # Equity curve has N-1 entries (starts at day 1)
    assert len(metrics.equity_curve) == N - 1


def test_equity_starts_near_capital(simple_pair):
    pa, pb, signals, beta = simple_pair
    cfg = BacktestConfig(initial_capital=500_000)
    engine = PairsBacktestEngine(config=cfg)
    metrics = engine.run(pa, pb, signals, beta)
    # First equity value should be close to initial capital
    assert abs(metrics.equity_curve.iloc[0] - 500_000) < 50_000


def test_zero_signal_no_trades(simple_pair):
    pa, pb, _, beta = simple_pair
    flat_signals = pd.Series(0, index=pa.index, name="signal")
    engine = PairsBacktestEngine()
    metrics = engine.run(pa, pb, flat_signals, beta)
    assert metrics.n_trades == 0


def test_sharpe_is_finite(simple_pair):
    pa, pb, signals, beta = simple_pair
    engine = PairsBacktestEngine()
    metrics = engine.run(pa, pb, signals, beta)
    assert np.isfinite(metrics.sharpe_ratio)


def test_max_drawdown_negative_or_zero(simple_pair):
    pa, pb, signals, beta = simple_pair
    engine = PairsBacktestEngine()
    metrics = engine.run(pa, pb, signals, beta)
    assert metrics.max_drawdown <= 0.0, "Max drawdown should be ≤ 0."


def test_win_rate_in_range(simple_pair):
    pa, pb, signals, beta = simple_pair
    engine = PairsBacktestEngine()
    metrics = engine.run(pa, pb, signals, beta)
    assert 0.0 <= metrics.win_rate <= 1.0


def test_transaction_costs_reduce_pnl(simple_pair):
    """A backtest with high costs should have lower returns than one with zero costs."""
    pa, pb, signals, beta = simple_pair
    cfg_cheap = BacktestConfig(transaction_cost_bps=0, slippage_bps=0)
    cfg_expensive = BacktestConfig(transaction_cost_bps=20, slippage_bps=10)
    m_cheap = PairsBacktestEngine(cfg_cheap).run(pa, pb, signals, beta)
    m_expensive = PairsBacktestEngine(cfg_expensive).run(pa, pb, signals, beta)
    assert m_cheap.total_return >= m_expensive.total_return


# ---------------------------------------------------------------------------
# RiskAnalytics
# ---------------------------------------------------------------------------


@pytest.fixture
def daily_returns() -> pd.Series:
    dates = pd.bdate_range("2022-01-03", periods=252)
    rets = RNG.normal(0.0005, 0.01, 252)
    return pd.Series(rets, index=dates)


def test_sharpe_positive(daily_returns):
    ra = RiskAnalytics(daily_returns)
    # With positive mean returns the Sharpe should be positive
    assert ra.sharpe_ratio() > 0


def test_max_drawdown_negative(daily_returns):
    ra = RiskAnalytics(daily_returns)
    assert ra.max_drawdown() <= 0.0


def test_var_positive(daily_returns):
    ra = RiskAnalytics(daily_returns)
    assert ra.historical_var(0.95) >= 0


def test_cvar_gte_var(daily_returns):
    ra = RiskAnalytics(daily_returns)
    assert ra.expected_shortfall(0.95) >= ra.historical_var(0.95)


def test_summary_has_required_keys(daily_returns):
    ra = RiskAnalytics(daily_returns)
    summary = ra.summary()
    required = [
        "sharpe_ratio", "sortino_ratio", "max_drawdown_pct",
        "annualised_return_pct", "var_95_pct",
    ]
    for key in required:
        assert key in summary, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Kelly criterion
# ---------------------------------------------------------------------------


def test_kelly_positive_edge():
    """With edge (p·b > q), Kelly should return positive size."""
    size = position_sizing_kelly(win_rate=0.6, avg_win=100, avg_loss=80)
    assert size > 0


def test_kelly_no_edge():
    """With no edge (p·b ≈ q), Kelly should return ~0."""
    size = position_sizing_kelly(win_rate=0.4, avg_win=80, avg_loss=100)
    assert size == 0.0


def test_kelly_capped():
    """Kelly fraction should never exceed 0.5 (hard cap)."""
    size = position_sizing_kelly(win_rate=0.9, avg_win=1000, avg_loss=10, fraction=1.0)
    assert size <= 0.5
