"""Unit tests for the Kalman filter and signal generation modules."""

import numpy as np
import pandas as pd
import pytest

from src.signals.kalman_filter import KalmanFilterHedge, MeanReversionSignal

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N = 400
RNG = np.random.default_rng(99)


@pytest.fixture
def price_pair() -> tuple[pd.Series, pd.Series]:
    """Synthetic cointegrated pair for filter testing."""
    dates = pd.bdate_range("2022-01-03", periods=N)
    common = np.cumsum(RNG.normal(0, 0.01, N))
    beta_true = 1.2
    log_b = 5.0 + common + RNG.normal(0, 0.005, N)
    log_a = 4.8 + beta_true * common + RNG.normal(0, 0.005, N)
    pa = pd.Series(np.exp(log_a), index=dates, name="A")
    pb = pd.Series(np.exp(log_b), index=dates, name="B")
    return pa, pb


# ---------------------------------------------------------------------------
# KalmanFilterHedge
# ---------------------------------------------------------------------------


def test_kalman_output_shape(price_pair):
    pa, pb = price_pair
    kf = KalmanFilterHedge()
    out = kf.fit(pa, pb)
    assert out.shape[0] == N
    assert set(["beta", "variance", "spread", "spread_mean", "spread_std", "zscore"]).issubset(out.columns)


def test_kalman_beta_positive(price_pair):
    """Hedge ratio should stay positive for positively correlated stocks."""
    pa, pb = price_pair
    kf = KalmanFilterHedge()
    out = kf.fit(pa, pb)
    assert (out["beta"] > 0).all(), "All beta values should be positive."


def test_kalman_variance_positive(price_pair):
    pa, pb = price_pair
    kf = KalmanFilterHedge()
    out = kf.fit(pa, pb)
    assert (out["variance"] > 0).all(), "Posterior variance must always be positive."


def test_kalman_beta_near_true(price_pair):
    """After burn-in, the estimated beta should converge close to true value (1.2)."""
    pa, pb = price_pair
    kf = KalmanFilterHedge(delta=1e-4)
    out = kf.fit(pa, pb)
    late_beta = out["beta"].iloc[200:]
    assert abs(late_beta.mean() - 1.2) < 0.4, (
        f"Late-stage beta mean {late_beta.mean():.3f} is too far from true 1.2"
    )


def test_kalman_zscore_has_nan_at_start(price_pair):
    """Z-score should be NaN for the first few observations (insufficient window)."""
    pa, pb = price_pair
    kf = KalmanFilterHedge(zscore_window=30)
    out = kf.fit(pa, pb)
    assert out["zscore"].iloc[:4].isna().all(), "Early z-scores should be NaN."


def test_online_update_reduces_variance():
    """Kalman update should reduce posterior variance vs prior."""
    kf = KalmanFilterHedge(delta=1e-4)
    beta, P = 1.0, 1.0
    x, y, R = 5.0, 6.0, 0.01
    beta_new, P_new = kf.online_update(beta, P, x, y, R)
    assert P_new < P, "Posterior variance should decrease after an update."
    assert np.isfinite(beta_new)


# ---------------------------------------------------------------------------
# MeanReversionSignal
# ---------------------------------------------------------------------------


def test_signal_values_are_valid(price_pair):
    pa, pb = price_pair
    kf = KalmanFilterHedge()
    out = kf.fit(pa, pb)
    sig_gen = MeanReversionSignal(entry_z=2.0, exit_z=0.5)
    signals = sig_gen.generate(out)
    assert set(signals.unique()).issubset({-1, 0, 1}), "Signals must be in {-1, 0, 1}."


def test_signal_no_position_at_start(price_pair):
    """Should start with no position (NaN z-scores initially)."""
    pa, pb = price_pair
    kf = KalmanFilterHedge(zscore_window=30)
    out = kf.fit(pa, pb)
    sig_gen = MeanReversionSignal()
    signals = sig_gen.generate(out)
    # First 4 should be 0 (NaN z-score → no trade)
    assert (signals.iloc[:4] == 0).all()


def test_signal_length(price_pair):
    pa, pb = price_pair
    kf = KalmanFilterHedge()
    out = kf.fit(pa, pb)
    sig_gen = MeanReversionSignal()
    signals = sig_gen.generate(out)
    assert len(signals) == N


def test_continuous_signal_range(price_pair):
    pa, pb = price_pair
    kf = KalmanFilterHedge()
    out = kf.fit(pa, pb)
    sig_gen = MeanReversionSignal(entry_z=2.0)
    cont = sig_gen.generate_continuous(out)
    valid = cont.dropna()
    assert (valid >= -1.0).all() and (valid <= 1.0).all(), (
        "Continuous signal must be in [-1, 1]."
    )
