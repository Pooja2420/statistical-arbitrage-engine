"""
Cointegration testing for pairs selection.

Two price series are *cointegrated* if a linear combination of them is
stationary (mean-reverting), even though each series individually is
non-stationary (a random walk).  This is the statistical foundation of
pairs trading: when the spread deviates from its long-run mean we expect
it to revert, giving us a trading signal.

We implement two complementary tests:

1. **Engle-Granger (1987)** — two-step OLS regression then ADF test on
   residuals.  Simple and widely used, but assumes a known cointegrating
   vector (fixed hedge ratio).

2. **Johansen (1988)** — likelihood-ratio test that simultaneously
   estimates the cointegrating rank and vectors without assuming a fixed
   hedge ratio.  More powerful for pairs but more sensitive to lag choice.
"""

import itertools
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class PairResult:
    """Stores all statistics for a single cointegrated pair candidate."""

    ticker_a: str
    ticker_b: str
    eg_pvalue: float          # Engle-Granger p-value
    eg_tstat: float           # ADF t-statistic on residuals
    johansen_trace: float     # Johansen trace statistic (rank ≥ 1)
    johansen_cv_95: float     # 95 % critical value for trace test
    hedge_ratio: float        # OLS hedge ratio (β): spread = A − β·B
    half_life: float          # Mean-reversion half-life in trading days
    is_cointegrated: bool     # True if both tests pass at 5 % level
    spread_mean: float = 0.0
    spread_std: float = 1.0

    @property
    def pair_name(self) -> str:
        return f"{self.ticker_a}/{self.ticker_b}"


@dataclass
class CointegrationReport:
    """Aggregated results from scanning all pairs in a universe."""

    all_results: List[PairResult] = field(default_factory=list)

    @property
    def cointegrated_pairs(self) -> List[PairResult]:
        """Only pairs that passed both tests."""
        return [r for r in self.all_results if r.is_cointegrated]

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self.all_results:
            rows.append({
                "pair": r.pair_name,
                "ticker_a": r.ticker_a,
                "ticker_b": r.ticker_b,
                "eg_pvalue": r.eg_pvalue,
                "johansen_trace": r.johansen_trace,
                "johansen_cv_95": r.johansen_cv_95,
                "hedge_ratio": r.hedge_ratio,
                "half_life_days": r.half_life,
                "is_cointegrated": r.is_cointegrated,
            })
        return pd.DataFrame(rows).sort_values("eg_pvalue")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def estimate_hedge_ratio(price_a: pd.Series, price_b: pd.Series) -> float:
    """
    OLS regression of price_a on price_b to estimate the hedge ratio β.

    The spread is defined as:  spread_t = ln(A_t) − β · ln(B_t)

    Using log prices makes the spread scale-invariant.
    """
    log_a = np.log(price_a.values)
    log_b = np.log(price_b.values)
    X = add_constant(log_b)
    result = OLS(log_a, X).fit()
    return float(result.params[1])


def compute_spread(
    price_a: pd.Series,
    price_b: pd.Series,
    hedge_ratio: float,
) -> pd.Series:
    """
    Construct the log-price spread.

    spread_t = ln(A_t) − β · ln(B_t)
    """
    spread = np.log(price_a) - hedge_ratio * np.log(price_b)
    spread.name = "spread"
    return spread


def estimate_half_life(spread: pd.Series) -> float:
    """
    Ornstein-Uhlenbeck mean-reversion half-life (in trading days).

    Fit:  Δspread_t = θ · (μ − spread_{t-1}) + ε_t
    via OLS.  Half-life = ln(2) / θ.

    A well-behaved pair should have a half-life of 5–60 trading days.
    Shorter: too noisy / too many transaction costs.
    Longer: not mean-reverting fast enough to exploit.
    """
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()

    aligned = pd.concat([spread_diff, spread_lag], axis=1).dropna()
    X = add_constant(aligned.iloc[:, 1])
    result = OLS(aligned.iloc[:, 0], X).fit()

    theta = -result.params.iloc[1]          # mean-reversion speed
    if theta <= 0:
        return np.inf                        # not mean-reverting
    return float(np.log(2) / theta)


# ---------------------------------------------------------------------------
# Individual pair tests
# ---------------------------------------------------------------------------


def engle_granger_test(
    price_a: pd.Series,
    price_b: pd.Series,
) -> Tuple[float, float, float]:
    """
    Engle-Granger cointegration test.

    Returns
    -------
    eg_tstat : float
        ADF t-statistic on OLS residuals.
    eg_pvalue : float
        Approximate p-value (< 0.05 ⟹ cointegrated at 5 % level).
    hedge_ratio : float
        OLS hedge ratio β.
    """
    log_a = np.log(price_a.values)
    log_b = np.log(price_b.values)
    eg_tstat, eg_pvalue, _ = coint(log_a, log_b)
    hedge_ratio = estimate_hedge_ratio(price_a, price_b)
    return eg_tstat, eg_pvalue, hedge_ratio


def johansen_test(
    price_a: pd.Series,
    price_b: pd.Series,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> Tuple[float, float]:
    """
    Johansen trace test for cointegration rank ≥ 1.

    Parameters
    ----------
    det_order : int
        -1 = no deterministic terms, 0 = constant, 1 = linear trend.
    k_ar_diff : int
        Number of lagged differences in the VAR.

    Returns
    -------
    trace_stat : float
        Trace statistic for the hypothesis H0: rank = 0.
    cv_95 : float
        95 % critical value.  If trace_stat > cv_95 ⟹ cointegrated.
    """
    log_prices = pd.concat(
        [np.log(price_a), np.log(price_b)], axis=1
    ).dropna().values

    result = coint_johansen(log_prices, det_order=det_order, k_ar_diff=k_ar_diff)
    # trace_stat[0] tests H0: rank=0 vs H1: rank≥1
    trace_stat = float(result.lr1[0])
    cv_95 = float(result.cvt[0, 1])      # column 1 = 95% CV
    return trace_stat, cv_95


def run_pair_test(
    ticker_a: str,
    ticker_b: str,
    price_a: pd.Series,
    price_b: pd.Series,
    eg_pvalue_threshold: float = 0.05,
    max_half_life: float = 60.0,
    min_half_life: float = 5.0,
) -> PairResult:
    """
    Run both cointegration tests on a single pair and return a PairResult.

    A pair is marked ``is_cointegrated=True`` only if:
    - Engle-Granger p-value < threshold (default 5 %)
    - Johansen trace statistic exceeds the 95 % critical value
    - Half-life is between min_half_life and max_half_life trading days
    """
    price_a, price_b = _align(price_a, price_b)

    try:
        eg_tstat, eg_pvalue, hedge_ratio = engle_granger_test(price_a, price_b)
        trace_stat, cv_95 = johansen_test(price_a, price_b)

        spread = compute_spread(price_a, price_b, hedge_ratio)
        half_life = estimate_half_life(spread)

        eg_pass = eg_pvalue < eg_pvalue_threshold
        jo_pass = trace_stat > cv_95
        hl_pass = min_half_life <= half_life <= max_half_life
        is_cointegrated = eg_pass and jo_pass and hl_pass

        return PairResult(
            ticker_a=ticker_a,
            ticker_b=ticker_b,
            eg_pvalue=eg_pvalue,
            eg_tstat=eg_tstat,
            johansen_trace=trace_stat,
            johansen_cv_95=cv_95,
            hedge_ratio=hedge_ratio,
            half_life=half_life,
            is_cointegrated=is_cointegrated,
            spread_mean=float(spread.mean()),
            spread_std=float(spread.std()),
        )

    except Exception as exc:  # noqa: BLE001
        logger.debug("Failed to test %s/%s: %s", ticker_a, ticker_b, exc)
        return PairResult(
            ticker_a=ticker_a,
            ticker_b=ticker_b,
            eg_pvalue=1.0,
            eg_tstat=0.0,
            johansen_trace=0.0,
            johansen_cv_95=np.inf,
            hedge_ratio=1.0,
            half_life=np.inf,
            is_cointegrated=False,
        )


# ---------------------------------------------------------------------------
# Universe scanner
# ---------------------------------------------------------------------------


class PairScanner:
    """
    Scan all pairs in a price universe for cointegration.

    Parameters
    ----------
    prices : pd.DataFrame
        Date-indexed adjusted close prices, one column per ticker.
    eg_pvalue_threshold : float
        Engle-Granger p-value cutoff (default 0.05).
    max_half_life : float
        Reject pairs with half-life > this many trading days (default 60).
    min_half_life : float
        Reject pairs with half-life < this many trading days (default 5).
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        eg_pvalue_threshold: float = 0.05,
        max_half_life: float = 60.0,
        min_half_life: float = 5.0,
    ) -> None:
        self.prices = prices
        self.eg_pvalue_threshold = eg_pvalue_threshold
        self.max_half_life = max_half_life
        self.min_half_life = min_half_life

    def scan(self, same_sector_only: bool = False) -> CointegrationReport:
        """
        Test all unique pairs and return a CointegrationReport.

        Parameters
        ----------
        same_sector_only : bool
            If True, only test pairs within the same GICS sector (requires
            a ``sector`` column in prices or a sector mapping dict).
            Default: False — test all combinations.
        """
        tickers = list(self.prices.columns)
        pairs = list(itertools.combinations(tickers, 2))
        logger.info("Testing %d pairs for cointegration…", len(pairs))

        report = CointegrationReport()
        for i, (a, b) in enumerate(pairs):
            result = run_pair_test(
                ticker_a=a,
                ticker_b=b,
                price_a=self.prices[a],
                price_b=self.prices[b],
                eg_pvalue_threshold=self.eg_pvalue_threshold,
                max_half_life=self.max_half_life,
                min_half_life=self.min_half_life,
            )
            report.all_results.append(result)

            if (i + 1) % 100 == 0:
                n_coint = len(report.cointegrated_pairs)
                logger.info("  … %d/%d pairs tested, %d cointegrated so far", i + 1, len(pairs), n_coint)

        n_coint = len(report.cointegrated_pairs)
        logger.info("Scan complete: %d/%d pairs cointegrated.", n_coint, len(pairs))
        return report


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _align(a: pd.Series, b: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Inner-join two series on their date index and drop NaNs."""
    combined = pd.concat([a, b], axis=1).dropna()
    return combined.iloc[:, 0], combined.iloc[:, 1]
