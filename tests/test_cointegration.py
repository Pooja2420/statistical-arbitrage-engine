"""Unit tests for the cointegration module."""

import numpy as np
import pandas as pd
import pytest

from src.signals.cointegration import (
    PairResult,
    PairScanner,
    compute_spread,
    estimate_half_life,
    estimate_hedge_ratio,
    run_pair_test,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 500  # trading days (~2 years)
RNG = np.random.default_rng(42)


@pytest.fixture
def cointegrated_pair() -> tuple[pd.Series, pd.Series]:
    """Generate a pair of truly cointegrated price series."""
    dates = pd.bdate_range("2022-01-03", periods=N)
    common = np.cumsum(RNG.normal(0, 0.01, N))   # shared random walk
    noise_a = RNG.normal(0, 0.005, N)
    noise_b = RNG.normal(0, 0.005, N)
    beta = 1.3
    log_b = 5.0 + common + noise_b
    log_a = 4.6 + beta * common + noise_a         # cointegrated with B
    price_a = pd.Series(np.exp(log_a), index=dates, name="A")
    price_b = pd.Series(np.exp(log_b), index=dates, name="B")
    return price_a, price_b


@pytest.fixture
def non_cointegrated_pair() -> tuple[pd.Series, pd.Series]:
    """Two independent random walks — should NOT be cointegrated."""
    dates = pd.bdate_range("2022-01-03", periods=N)
    log_a = np.cumsum(RNG.normal(0.0003, 0.015, N))
    log_b = np.cumsum(RNG.normal(0.0003, 0.015, N))
    price_a = pd.Series(np.exp(log_a) * 100, index=dates, name="C")
    price_b = pd.Series(np.exp(log_b) * 100, index=dates, name="D")
    return price_a, price_b


# ---------------------------------------------------------------------------
# estimate_hedge_ratio
# ---------------------------------------------------------------------------


def test_hedge_ratio_positive(cointegrated_pair):
    pa, pb = cointegrated_pair
    beta = estimate_hedge_ratio(pa, pb)
    assert beta > 0, "Hedge ratio should be positive for correlated series."


def test_hedge_ratio_close_to_true(cointegrated_pair):
    """The estimated β should be close to the true value of 1.3."""
    pa, pb = cointegrated_pair
    beta = estimate_hedge_ratio(pa, pb)
    assert abs(beta - 1.3) < 0.3, f"Expected β ≈ 1.3, got {beta:.3f}"


# ---------------------------------------------------------------------------
# compute_spread
# ---------------------------------------------------------------------------


def test_spread_is_stationary(cointegrated_pair):
    """The spread of a cointegrated pair should be stationary (small variance)."""
    pa, pb = cointegrated_pair
    beta = estimate_hedge_ratio(pa, pb)
    spread = compute_spread(pa, pb, beta)
    assert spread.std() < 0.5, "Spread std should be small for cointegrated pair."


def test_spread_has_correct_length(cointegrated_pair):
    pa, pb = cointegrated_pair
    beta = estimate_hedge_ratio(pa, pb)
    spread = compute_spread(pa, pb, beta)
    assert len(spread) == N


# ---------------------------------------------------------------------------
# estimate_half_life
# ---------------------------------------------------------------------------


def test_half_life_positive(cointegrated_pair):
    pa, pb = cointegrated_pair
    beta = estimate_hedge_ratio(pa, pb)
    spread = compute_spread(pa, pb, beta)
    hl = estimate_half_life(spread)
    assert hl > 0, "Half-life should be positive for a cointegrated pair."


def test_half_life_reasonable(cointegrated_pair):
    pa, pb = cointegrated_pair
    beta = estimate_hedge_ratio(pa, pb)
    spread = compute_spread(pa, pb, beta)
    hl = estimate_half_life(spread)
    assert hl < 200, "Half-life should be finite for a mean-reverting spread."


# ---------------------------------------------------------------------------
# test_pair
# ---------------------------------------------------------------------------


def test_cointegrated_pair_detected(cointegrated_pair):
    pa, pb = cointegrated_pair
    result = run_pair_test("A", "B", pa, pb)
    assert isinstance(result, PairResult)
    assert result.eg_pvalue < 0.10, (
        f"Expected low EG p-value for cointegrated pair, got {result.eg_pvalue:.4f}"
    )


def test_non_cointegrated_pair_rejected(non_cointegrated_pair):
    pa, pb = non_cointegrated_pair
    result = run_pair_test("C", "D", pa, pb)
    # Independent random walks should mostly fail cointegration
    # We allow a small false-positive rate at 5% threshold
    assert result.eg_pvalue > 0.01 or not result.is_cointegrated


def test_pair_result_fields(cointegrated_pair):
    pa, pb = cointegrated_pair
    result = run_pair_test("A", "B", pa, pb)
    assert result.ticker_a == "A"
    assert result.ticker_b == "B"
    assert result.pair_name == "A/B"
    assert np.isfinite(result.hedge_ratio)
    assert np.isfinite(result.eg_pvalue)


# ---------------------------------------------------------------------------
# PairScanner
# ---------------------------------------------------------------------------


def test_scanner_returns_dataframe(cointegrated_pair, non_cointegrated_pair):
    pa, pb = cointegrated_pair
    pc, pd_ = non_cointegrated_pair
    prices = pd.concat([pa, pb, pc, pd_], axis=1)
    prices.columns = ["A", "B", "C", "D"]

    scanner = PairScanner(prices, eg_pvalue_threshold=0.10)
    report = scanner.scan()
    df = report.to_dataframe()

    assert isinstance(df, pd.DataFrame)
    assert "pair" in df.columns
    assert "eg_pvalue" in df.columns
    assert "is_cointegrated" in df.columns


def test_scanner_finds_cointegrated_pair(cointegrated_pair, non_cointegrated_pair):
    pa, pb = cointegrated_pair
    pc, pd_ = non_cointegrated_pair
    prices = pd.concat([pa, pb, pc, pd_], axis=1)
    prices.columns = ["A", "B", "C", "D"]

    # Use relaxed thresholds so synthetic data passes both the EG and half-life filters
    scanner = PairScanner(
        prices,
        eg_pvalue_threshold=0.10,
        min_half_life=0.5,
        max_half_life=500,
    )
    report = scanner.scan()

    # The A/B pair should at least have the lowest p-value among all tested pairs
    df = report.to_dataframe()
    lowest_pvalue_pair = df.sort_values("eg_pvalue").iloc[0]["pair"]
    assert lowest_pvalue_pair == "A/B", (
        f"Expected A/B to have lowest p-value, got {lowest_pvalue_pair}"
    )
