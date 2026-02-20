"""
Risk management for the statistical arbitrage portfolio.

Three layers of risk control
-----------------------------
1. **Position-level** — per-pair stop-loss and maximum holding period.
   These are enforced inside the backtest engine and configurable via
   BacktestConfig.

2. **Portfolio-level** — limits on total gross exposure, maximum number
   of concurrent open positions, and daily P&L drawdown.

3. **Reporting** — portfolio heat map, drawdown analysis, VaR estimation,
   and correlation of position returns (are our "independent" pairs
   actually independent?).
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Portfolio-level risk controls
# ---------------------------------------------------------------------------


class PortfolioRiskManager:
    """
    Enforces portfolio-level risk limits and provides risk reporting.

    Parameters
    ----------
    max_pairs : int
        Maximum number of simultaneously open pairs.
    max_gross_exposure : float
        Maximum total notional as a fraction of NAV (e.g. 2.0 = 200 % gross).
    daily_stop_loss : float
        If portfolio daily P&L drops below this fraction of NAV, flatten all
        positions (circuit breaker).
    var_confidence : float
        Confidence level for historical VaR (e.g. 0.95 = 95 % VaR).
    """

    def __init__(
        self,
        max_pairs: int = 10,
        max_gross_exposure: float = 2.0,
        daily_stop_loss: float = -0.02,
        var_confidence: float = 0.95,
    ) -> None:
        self.max_pairs = max_pairs
        self.max_gross_exposure = max_gross_exposure
        self.daily_stop_loss = daily_stop_loss
        self.var_confidence = var_confidence

    def check_new_position(
        self,
        n_open: int,
        current_gross_exposure: float,
        position_notional: float,
        nav: float,
    ) -> bool:
        """
        Return True if a new position is allowed given current portfolio state.

        Parameters
        ----------
        n_open : int
            Current number of open pair positions.
        current_gross_exposure : float
            Sum of absolute notional values of all open positions.
        position_notional : float
            Notional value of the proposed new position (one leg).
        nav : float
            Current net asset value.
        """
        if n_open >= self.max_pairs:
            logger.debug("Risk check failed: max_pairs limit (%d) reached.", self.max_pairs)
            return False

        new_gross = current_gross_exposure + 2 * position_notional  # long + short legs
        if new_gross / nav > self.max_gross_exposure:
            logger.debug(
                "Risk check failed: gross exposure %.1f%% would exceed limit %.1f%%.",
                new_gross / nav * 100,
                self.max_gross_exposure * 100,
            )
            return False

        return True

    def circuit_breaker_triggered(self, daily_pnl: float, nav: float) -> bool:
        """Return True if the daily stop-loss circuit breaker should fire."""
        pnl_pct = daily_pnl / nav
        if pnl_pct < self.daily_stop_loss:
            logger.warning(
                "Circuit breaker! Daily P&L = %.2f%% < limit %.2f%%.",
                pnl_pct * 100,
                self.daily_stop_loss * 100,
            )
            return True
        return False


# ---------------------------------------------------------------------------
# Risk analytics
# ---------------------------------------------------------------------------


class RiskAnalytics:
    """
    Compute risk metrics from a portfolio's daily return stream.

    Parameters
    ----------
    daily_returns : pd.Series
        Daily portfolio returns (fractions, not percentages).
    risk_free_rate : float
        Annualised risk-free rate for Sharpe calculation (default 0.05 = 5 %).
    """

    TRADING_DAYS = 252

    def __init__(
        self,
        daily_returns: pd.Series,
        risk_free_rate: float = 0.05,
    ) -> None:
        self.returns = daily_returns.dropna()
        self.rf_daily = (1 + risk_free_rate) ** (1 / self.TRADING_DAYS) - 1

    # ------------------------------------------------------------------
    # Core metrics
    # ------------------------------------------------------------------

    def sharpe_ratio(self) -> float:
        """Annualised Sharpe ratio."""
        excess = self.returns - self.rf_daily
        if excess.std() == 0:
            return 0.0
        return float(excess.mean() / excess.std() * np.sqrt(self.TRADING_DAYS))

    def sortino_ratio(self) -> float:
        """Annualised Sortino ratio (penalises only downside volatility)."""
        excess = self.returns - self.rf_daily
        downside = excess[excess < 0]
        if len(downside) < 2:
            return np.inf
        return float(excess.mean() / downside.std() * np.sqrt(self.TRADING_DAYS))

    def max_drawdown(self) -> float:
        """Maximum peak-to-trough drawdown (as a negative fraction)."""
        cumulative = (1 + self.returns).cumprod()
        rolling_peak = cumulative.cummax()
        drawdown = (cumulative - rolling_peak) / rolling_peak
        return float(drawdown.min())

    def drawdown_series(self) -> pd.Series:
        """Full drawdown time series."""
        cumulative = (1 + self.returns).cumprod()
        rolling_peak = cumulative.cummax()
        return (cumulative - rolling_peak) / rolling_peak

    def calmar_ratio(self) -> float:
        """Annualised return / |Max drawdown|."""
        ann_ret = self.annualised_return()
        mdd = abs(self.max_drawdown())
        return ann_ret / mdd if mdd > 0 else np.inf

    def annualised_return(self) -> float:
        """Compound annualised growth rate."""
        total = (1 + self.returns).prod()
        n_years = len(self.returns) / self.TRADING_DAYS
        return float(total ** (1 / max(n_years, 1e-9)) - 1)

    def annualised_volatility(self) -> float:
        """Annualised standard deviation of daily returns."""
        return float(self.returns.std() * np.sqrt(self.TRADING_DAYS))

    # ------------------------------------------------------------------
    # Value at Risk
    # ------------------------------------------------------------------

    def historical_var(self, confidence: float = 0.95) -> float:
        """
        Historical (empirical) Value at Risk.

        Returns the loss (positive number) not exceeded with probability
        ``confidence`` over one trading day.
        """
        return float(-np.percentile(self.returns, (1 - confidence) * 100))

    def parametric_var(self, confidence: float = 0.95) -> float:
        """
        Parametric (Gaussian) VaR.  Assumes returns are normally distributed.
        """
        z = stats.norm.ppf(1 - confidence)
        return float(-(self.returns.mean() + z * self.returns.std()))

    def expected_shortfall(self, confidence: float = 0.95) -> float:
        """
        Expected Shortfall (CVaR) — average loss in the worst (1-confidence)
        fraction of days.  More informative than VaR for fat-tailed returns.
        """
        threshold = np.percentile(self.returns, (1 - confidence) * 100)
        tail = self.returns[self.returns <= threshold]
        return float(-tail.mean()) if len(tail) > 0 else 0.0

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return all metrics as a dictionary."""
        return {
            "annualised_return_pct": round(self.annualised_return() * 100, 2),
            "annualised_volatility_pct": round(self.annualised_volatility() * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio(), 3),
            "sortino_ratio": round(self.sortino_ratio(), 3),
            "calmar_ratio": round(self.calmar_ratio(), 3),
            "max_drawdown_pct": round(self.max_drawdown() * 100, 2),
            "var_95_pct": round(self.historical_var(0.95) * 100, 2),
            "cvar_95_pct": round(self.expected_shortfall(0.95) * 100, 2),
            "n_trading_days": len(self.returns),
        }


# ---------------------------------------------------------------------------
# Portfolio heat map
# ---------------------------------------------------------------------------


def compute_portfolio_heatmap(
    pair_returns: Dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Build a correlation matrix of per-pair daily returns.

    Ideally, pairs in our portfolio should be uncorrelated with each other.
    High cross-pair correlations indicate hidden common-factor exposure and
    mean our portfolio is less diversified than we think.

    Parameters
    ----------
    pair_returns : dict {pair_name: daily_return_series}

    Returns
    -------
    pd.DataFrame
        Correlation matrix, pairs × pairs.
    """
    df = pd.DataFrame(pair_returns).dropna()
    return df.corr()


def rolling_sharpe(returns: pd.Series, window: int = 63) -> pd.Series:
    """
    Rolling Sharpe ratio with a given lookback window (default 63 = 1 quarter).

    Useful for detecting strategy decay over time.
    """
    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    return (roll_mean / roll_std * np.sqrt(252)).rename("rolling_sharpe")


def position_sizing_kelly(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    fraction: float = 0.25,
) -> float:
    """
    Kelly criterion position size.

    Full Kelly = (p/|loss| − q/win) where p = win_rate, q = 1 − win_rate.
    We return *fractional Kelly* (default 25 %) to account for estimation
    error and to reduce volatility.

    Parameters
    ----------
    win_rate : float    Fraction of trades that are profitable (0–1).
    avg_win : float     Average P&L of winning trades (positive).
    avg_loss : float    Average P&L of losing trades (positive magnitude).
    fraction : float    Kelly fraction (default 0.25 = quarter-Kelly).

    Returns
    -------
    float
        Suggested position size as a fraction of capital (0–1).
    """
    if avg_loss == 0 or avg_win == 0:
        return 0.0
    b = avg_win / avg_loss          # win/loss ratio
    p = win_rate
    q = 1 - win_rate
    kelly = (p * b - q) / b
    return max(0.0, min(fraction * kelly, 0.5))   # cap at 50 % of capital
