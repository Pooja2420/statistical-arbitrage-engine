"""
Kalman Filter for dynamic hedge-ratio estimation.

Why Kalman instead of fixed OLS?
---------------------------------
OLS fits a *static* hedge ratio over the entire history.  In practice the
relationship between two stocks drifts over time — sector rotations,
earnings, macro shocks all shift the spread dynamics.  The Kalman Filter
treats the hedge ratio as a *latent state* that evolves according to a
random walk, and updates our estimate every day as new prices arrive.

State-space model
-----------------
Observation equation:
    y_t = x_t · β_t + ε_t,    ε_t ~ N(0, R)

State transition equation (random walk hedge ratio):
    β_t = β_{t-1} + η_t,      η_t ~ N(0, Q)

where:
    y_t  = log price of stock A at time t
    x_t  = log price of stock B at time t
    β_t  = time-varying hedge ratio
    R    = observation noise variance (measurement error)
    Q    = state noise variance (how fast β is allowed to drift)

The spread at time t is:
    spread_t = y_t − β_t · x_t

We then z-score the spread using a rolling window of the Kalman-smoothed
spread to generate entry/exit signals.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class KalmanState:
    """Snapshot of the Kalman filter state at a single time step."""

    beta: float           # Current hedge-ratio estimate
    variance: float       # Posterior variance of beta
    spread: float         # Observed spread at this step
    spread_mean: float    # Rolling mean of the spread
    spread_std: float     # Rolling std of the spread
    zscore: float         # Normalised spread signal


class KalmanFilterHedge:
    """
    Online (sequential) Kalman Filter for time-varying hedge ratio.

    Parameters
    ----------
    delta : float
        Controls how fast the hedge ratio can drift.  Higher ⟹ faster
        adaptation but noisier estimates.  Typical range: 1e-5 – 1e-3.
    observation_cov : float
        Observation noise R.  Usually calibrated from OLS residual variance.
        If None, it is estimated from the first 60 days of data.
    zscore_window : int
        Rolling window (in trading days) used to compute the spread mean
        and std for z-scoring.  Should match your expected half-life.
    """

    def __init__(
        self,
        delta: float = 1e-4,
        observation_cov: Optional[float] = None,
        zscore_window: int = 30,
    ) -> None:
        self.delta = delta
        self.observation_cov = observation_cov
        self.zscore_window = zscore_window

        # State covariance Q = delta / (1 - delta)
        self._Q = delta / (1.0 - delta)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
    ) -> pd.DataFrame:
        """
        Run the Kalman Filter over the full price history and return
        a DataFrame with the hedge ratio, spread, and z-score at every date.

        Parameters
        ----------
        price_a, price_b : pd.Series
            Aligned daily adjusted close prices (same index).

        Returns
        -------
        pd.DataFrame with columns:
            beta        — time-varying hedge ratio
            variance    — posterior variance of beta
            spread      — raw spread (log A − beta · log B)
            spread_mean — rolling mean of spread (zscore_window days)
            spread_std  — rolling std of spread (zscore_window days)
            zscore      — (spread − mean) / std  ← the trading signal
        """
        log_a = np.log(price_a.values)
        log_b = np.log(price_b.values)
        n = len(log_a)

        # --- initialise state ---
        # Prior for beta: OLS estimate on first 30 days (or all if < 30)
        init_window = min(30, n)
        beta_init = float(np.polyfit(log_b[:init_window], log_a[:init_window], 1)[0])
        P = 1.0           # prior variance — deliberately vague

        # Observation covariance: residual variance of initial OLS
        if self.observation_cov is None:
            resid = log_a[:init_window] - beta_init * log_b[:init_window]
            R = float(np.var(resid)) if np.var(resid) > 0 else 1e-4
        else:
            R = self.observation_cov

        # Storage
        betas = np.empty(n)
        variances = np.empty(n)
        spreads = np.empty(n)

        beta = beta_init
        for t in range(n):
            x = log_b[t]
            y = log_a[t]

            # --- Predict ---
            P_pred = P + self._Q

            # --- Update (scalar Kalman gain) ---
            S = x * P_pred * x + R          # innovation variance
            K = P_pred * x / S              # Kalman gain
            innovation = y - x * beta       # measurement residual
            beta = beta + K * innovation    # posterior mean
            P = (1 - K * x) * P_pred       # posterior variance

            betas[t] = beta
            variances[t] = P
            spreads[t] = y - beta * x

        index = price_a.index
        spread_series = pd.Series(spreads, index=index, name="spread")

        # Rolling z-score
        roll_mean = spread_series.rolling(self.zscore_window, min_periods=5).mean()
        roll_std = spread_series.rolling(self.zscore_window, min_periods=5).std()
        zscore = (spread_series - roll_mean) / roll_std.replace(0, np.nan)

        return pd.DataFrame(
            {
                "beta": betas,
                "variance": variances,
                "spread": spreads,
                "spread_mean": roll_mean,
                "spread_std": roll_std,
                "zscore": zscore,
            },
            index=index,
        )

    def online_update(
        self,
        beta: float,
        P: float,
        x: float,
        y: float,
        R: float,
    ) -> Tuple[float, float]:
        """
        Single-step Kalman update.  Useful for live streaming inference.

        Parameters
        ----------
        beta : float  Current hedge ratio estimate.
        P : float     Current posterior variance.
        x : float     log price of stock B.
        y : float     log price of stock A.
        R : float     Observation noise variance.

        Returns
        -------
        beta_new : float  Updated hedge ratio.
        P_new : float     Updated posterior variance.
        """
        P_pred = P + self._Q
        S = x * P_pred * x + R
        K = P_pred * x / S
        beta_new = beta + K * (y - x * beta)
        P_new = (1 - K * x) * P_pred
        return beta_new, P_new


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------


class MeanReversionSignal:
    """
    Translates Kalman Filter z-scores into discrete trading signals.

    Signals
    -------
    +1  Long the spread (A is cheap relative to B): enter when zscore < -entry_z
    -1  Short the spread (A is expensive relative to B): enter when zscore > +entry_z
     0  No position / exit

    Parameters
    ----------
    entry_z : float
        Z-score threshold to open a new position (default 2.0).
    exit_z : float
        Z-score threshold to close a position (default 0.5).
    stop_z : float
        Emergency stop-loss z-score (default 3.5).  If the spread moves
        this far against us, we close regardless.
    """

    def __init__(
        self,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
        stop_z: float = 3.5,
    ) -> None:
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_z = stop_z

    def generate(self, kalman_output: pd.DataFrame) -> pd.Series:
        """
        Generate position signals from the Kalman Filter output DataFrame.

        Returns
        -------
        pd.Series of int  {-1, 0, +1}, indexed same as kalman_output.
            +1 = long spread  (buy A, sell B)
            -1 = short spread (sell A, buy B)
             0 = flat
        """
        zscore = kalman_output["zscore"].copy()
        signals = pd.Series(0, index=zscore.index, dtype=int)
        position = 0

        for i, (date, z) in enumerate(zscore.items()):
            if np.isnan(z):
                signals[date] = 0
                continue

            if position == 0:
                # No open position — check for entry
                if z < -self.entry_z:
                    position = 1          # spread is low → long
                elif z > self.entry_z:
                    position = -1         # spread is high → short

            elif position == 1:
                # Long the spread — check for exit or stop
                if z > -self.exit_z or z > self.stop_z:
                    position = 0

            elif position == -1:
                # Short the spread — check for exit or stop
                if z < self.exit_z or z < -self.stop_z:
                    position = 0

            signals[date] = position

        return signals

    def generate_continuous(self, kalman_output: pd.DataFrame) -> pd.Series:
        """
        Alternative: return raw z-score clamped to [-1, +1].

        Useful for position-sizing strategies that scale exposure
        proportionally to signal strength.
        """
        z = kalman_output["zscore"]
        return (-z / self.entry_z).clip(-1.0, 1.0)
