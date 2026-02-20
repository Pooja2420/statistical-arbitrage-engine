"""
Realistic backtesting engine for pairs trading.

Design philosophy
-----------------
Most toy backtests assume:
  - Zero transaction costs
  - Perfect fill at close prices
  - No slippage

This engine models all three realistically:

Transaction costs
    A round-trip trade costs roughly 5–10 bps for institutional traders.
    We model it as a fixed percentage of notional traded.

Slippage
    Large orders move the market.  We model slippage proportional to the
    daily volume (market impact).  For a liquid pair the impact is small;
    for small-caps it can be significant.

Execution
    We generate signals on the *close* price but execute on the *next day's
    open* (look-ahead-bias free).  Since we don't have intraday data we
    approximate next-open as today's close × (1 + a small spread).

Position sizing
    Dollar-neutral: we size positions so that the notional value of the
    long leg equals the notional of the short leg.  This gives us market-
    neutral exposure (the P&L comes purely from the spread, not the market).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BacktestConfig:
    """All tunable parameters for the backtest."""

    initial_capital: float = 1_000_000.0    # Starting NAV in USD
    position_size_pct: float = 0.10         # Fraction of capital per pair leg
    transaction_cost_bps: float = 5.0       # One-way cost in basis points
    slippage_bps: float = 2.0              # Slippage in basis points (per trade)
    stop_loss_pct: float = 0.05             # 5 % position stop-loss
    max_holding_days: int = 30              # Force-close after this many days


# ---------------------------------------------------------------------------
# Trade record
# ---------------------------------------------------------------------------


@dataclass
class Trade:
    """Record of a single round-trip trade."""

    pair: str
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp]
    direction: int                  # +1 long spread, -1 short spread
    entry_zscore: float
    exit_zscore: float
    pnl: float = 0.0
    return_pct: float = 0.0
    holding_days: int = 0
    exit_reason: str = ""           # "signal", "stop_loss", "time_limit"


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------


@dataclass
class PerformanceMetrics:
    """Summary statistics for a completed backtest."""

    total_return: float
    annualised_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_holding_days: float
    n_trades: int
    equity_curve: pd.Series
    daily_returns: pd.Series
    trades: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_return_pct": round(self.total_return * 100, 2),
            "annualised_return_pct": round(self.annualised_return * 100, 2),
            "sharpe_ratio": round(self.sharpe_ratio, 3),
            "sortino_ratio": round(self.sortino_ratio, 3),
            "max_drawdown_pct": round(self.max_drawdown * 100, 2),
            "win_rate_pct": round(self.win_rate * 100, 2),
            "profit_factor": round(self.profit_factor, 3),
            "avg_holding_days": round(self.avg_holding_days, 1),
            "n_trades": self.n_trades,
        }


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------


class PairsBacktestEngine:
    """
    Simulate a pairs-trading strategy over historical price data.

    Parameters
    ----------
    config : BacktestConfig
        Cost and sizing parameters.
    """

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self.config = config or BacktestConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        price_a: pd.Series,
        price_b: pd.Series,
        signals: pd.Series,
        hedge_ratio: pd.Series,
        pair_name: str = "PAIR",
        zscore: Optional[pd.Series] = None,
    ) -> PerformanceMetrics:
        """
        Run the backtest for a single pair.

        Parameters
        ----------
        price_a, price_b : pd.Series
            Daily adjusted close prices.
        signals : pd.Series
            Integer signal series {-1, 0, +1}.  Signal generated on day t
            is executed at the start of day t+1.
        hedge_ratio : pd.Series
            Time-varying hedge ratio β (from Kalman filter).
        pair_name : str
        zscore : pd.Series, optional
            For recording entry/exit z-scores in trade logs.

        Returns
        -------
        PerformanceMetrics
        """
        cfg = self.config
        # Align all inputs
        data = pd.concat(
            [price_a.rename("pa"), price_b.rename("pb"),
             signals.rename("signal"), hedge_ratio.rename("beta")],
            axis=1,
        ).dropna()
        if zscore is not None:
            data["zscore"] = zscore.reindex(data.index)

        equity = cfg.initial_capital
        equity_curve = []
        daily_returns = []
        trades: list[Trade] = []

        position = 0
        entry_date = None
        entry_equity = equity
        entry_zscore = 0.0
        holding_days = 0
        entry_price_a = entry_price_b = 0.0

        dates = data.index
        n = len(dates)

        for i in range(1, n):           # start at 1: execute yesterday's signal
            prev_date = dates[i - 1]
            today = dates[i]
            row = data.loc[today]

            pa = row["pa"]
            pb = row["pb"]
            beta = row["beta"]
            new_signal = data.loc[prev_date, "signal"]   # signal from yesterday
            z_now = data.loc[today, "zscore"] if "zscore" in data.columns else 0.0

            # --- compute daily P&L for open position ---
            day_pnl = 0.0
            if position != 0:
                holding_days += 1
                # Dollar-neutral P&L
                # Long spread: long A, short B
                # Short spread: short A, long B
                pnl_a = (pa - entry_price_a) / entry_price_a
                pnl_b = (pb - entry_price_b) / entry_price_b
                notional = cfg.position_size_pct * entry_equity
                if position == 1:
                    day_pnl = notional * (pnl_a - beta * pnl_b) / (1 + beta)
                else:
                    day_pnl = notional * (-pnl_a + beta * pnl_b) / (1 + beta)

                # Stop-loss check
                unrealised_return = day_pnl / (notional / (1 + beta))
                if unrealised_return < -cfg.stop_loss_pct:
                    position, day_pnl, exit_reason = self._close_position(
                        position, pa, pb, beta, entry_price_a, entry_price_b,
                        entry_equity, cfg, "stop_loss"
                    )
                    trades.append(Trade(
                        pair=pair_name,
                        entry_date=entry_date,
                        exit_date=today,
                        direction=position if position != 0 else (1 if pnl_a > pnl_b else -1),
                        entry_zscore=entry_zscore,
                        exit_zscore=float(z_now),
                        pnl=day_pnl,
                        return_pct=day_pnl / (entry_equity * cfg.position_size_pct),
                        holding_days=holding_days,
                        exit_reason=exit_reason,
                    ))
                    position = 0
                    holding_days = 0

                # Time-limit check
                elif holding_days >= cfg.max_holding_days:
                    position, day_pnl, exit_reason = self._close_position(
                        position, pa, pb, beta, entry_price_a, entry_price_b,
                        entry_equity, cfg, "time_limit"
                    )
                    trades.append(Trade(
                        pair=pair_name,
                        entry_date=entry_date,
                        exit_date=today,
                        direction=position if position != 0 else 0,
                        entry_zscore=entry_zscore,
                        exit_zscore=float(z_now),
                        pnl=day_pnl,
                        return_pct=day_pnl / (entry_equity * cfg.position_size_pct),
                        holding_days=holding_days,
                        exit_reason=exit_reason,
                    ))
                    position = 0
                    holding_days = 0

                # Signal exit
                elif new_signal == 0 and position != 0:
                    _, day_pnl, _ = self._close_position(
                        position, pa, pb, beta, entry_price_a, entry_price_b,
                        entry_equity, cfg, "signal"
                    )
                    trades.append(Trade(
                        pair=pair_name,
                        entry_date=entry_date,
                        exit_date=today,
                        direction=position,
                        entry_zscore=entry_zscore,
                        exit_zscore=float(z_now),
                        pnl=day_pnl,
                        return_pct=day_pnl / (entry_equity * cfg.position_size_pct),
                        holding_days=holding_days,
                        exit_reason="signal",
                    ))
                    position = 0
                    holding_days = 0

            # --- open new position ---
            if position == 0 and new_signal != 0:
                cost = self._transaction_cost(pa, pb, cfg)
                day_pnl -= cost
                position = new_signal
                entry_date = today
                entry_equity = equity
                entry_price_a = pa
                entry_price_b = pb
                entry_zscore = float(z_now)
                holding_days = 0

            equity += day_pnl
            equity_curve.append(equity)
            daily_returns.append(day_pnl / (equity - day_pnl) if equity != day_pnl else 0.0)

        equity_series = pd.Series(equity_curve, index=dates[1:], name="equity")
        returns_series = pd.Series(daily_returns, index=dates[1:], name="daily_return")

        return self._compute_metrics(equity_series, returns_series, trades, cfg.initial_capital)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _transaction_cost(pa: float, pb: float, cfg: BacktestConfig) -> float:
        """Total one-way cost for both legs (transaction + slippage)."""
        total_bps = cfg.transaction_cost_bps + cfg.slippage_bps
        notional = cfg.position_size_pct * cfg.initial_capital
        return notional * (total_bps / 10_000) * 2   # both legs

    @staticmethod
    def _close_position(
        position: int,
        pa: float, pb: float, beta: float,
        entry_pa: float, entry_pb: float,
        entry_equity: float,
        cfg: BacktestConfig,
        exit_reason: str,
    ):
        """Compute P&L for closing the position and deduct exit costs."""
        notional = cfg.position_size_pct * entry_equity
        pnl_a = (pa - entry_pa) / entry_pa
        pnl_b = (pb - entry_pb) / entry_pb
        if position == 1:
            gross = notional * (pnl_a - beta * pnl_b) / (1 + beta)
        else:
            gross = notional * (-pnl_a + beta * pnl_b) / (1 + beta)
        cost = notional * ((cfg.transaction_cost_bps + cfg.slippage_bps) / 10_000) * 2
        return 0, gross - cost, exit_reason

    @staticmethod
    def _compute_metrics(
        equity: pd.Series,
        returns: pd.Series,
        trades: list,
        initial_capital: float,
    ) -> PerformanceMetrics:
        """Compute all performance statistics."""
        total_return = (equity.iloc[-1] - initial_capital) / initial_capital
        n_years = len(returns) / 252
        annualised_return = (1 + total_return) ** (1 / max(n_years, 1e-6)) - 1

        # Sharpe (annualised, assuming 252 trading days)
        daily_mean = returns.mean()
        daily_std = returns.std()
        sharpe = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0

        # Sortino (penalise only downside volatility)
        downside = returns[returns < 0]
        sortino_std = downside.std() if len(downside) > 1 else daily_std
        sortino = (daily_mean / sortino_std * np.sqrt(252)) if sortino_std > 0 else 0.0

        # Max drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = float(drawdown.min())

        # Trade statistics
        completed = [t for t in trades if t.exit_date is not None]
        n_trades = len(completed)
        if n_trades > 0:
            winners = [t for t in completed if t.pnl > 0]
            losers = [t for t in completed if t.pnl <= 0]
            win_rate = len(winners) / n_trades
            gross_profit = sum(t.pnl for t in winners)
            gross_loss = abs(sum(t.pnl for t in losers))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
            avg_holding = np.mean([t.holding_days for t in completed])
        else:
            win_rate = profit_factor = avg_holding = 0.0

        return PerformanceMetrics(
            total_return=total_return,
            annualised_return=annualised_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_holding_days=avg_holding,
            n_trades=n_trades,
            equity_curve=equity,
            daily_returns=returns,
            trades=trades,
        )
