"""
Data loader for stock price data.

Downloads historical OHLCV data from Yahoo Finance and provides
clean adjusted close prices for the statistical arbitrage pipeline.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# A representative universe of liquid S&P 500 stocks across sectors
SP500_UNIVERSE = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "META", "NVDA", "ADBE", "CRM", "CSCO", "INTC", "TXN", "QCOM", "IBM",
    # Financials
    "JPM", "BAC", "V", "MA", "GS", "MS", "AXP", "BLK",
    # Consumer Discretionary
    "AMZN", "TSLA", "HD", "NKE", "MCD", "SBUX", "LOW", "TGT",
    # Consumer Staples
    "WMT", "PG", "KO", "PEP", "COST", "PM",
    # Healthcare
    "JNJ", "PFE", "MRK", "ABT", "TMO", "ABBV", "DHR", "MDT", "BMY",
    # Communication
    "DIS", "NFLX", "CMCSA", "VZ", "T",
    # Industrials
    "UNP", "HON", "UPS", "RTX", "ACN",
    # Energy
    "CVX", "XOM",
    # Utilities / Materials
    "NEE", "LIN",
]


class DataLoader:
    """
    Downloads and caches adjusted close prices from Yahoo Finance.

    Parameters
    ----------
    start_date : str
        Start of the historical window, e.g. "2020-01-01".
    end_date : str
        End of the historical window, e.g. "2024-12-31".
    cache_dir : str | Path
        Directory to cache downloaded parquet files.
    """

    def __init__(
        self,
        start_date: str = "2020-01-01",
        end_date: str = "2024-12-31",
        cache_dir: str = "data/raw",
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("DataLoader initialised: %s → %s", start_date, end_date)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_sp500_universe(self, force_download: bool = False) -> pd.DataFrame:
        """
        Fetch adjusted close prices for the built-in S&P 500 universe.

        Results are cached on disk. Pass ``force_download=True`` to bypass
        the cache and re-download from Yahoo Finance.

        Returns
        -------
        pd.DataFrame
            Date-indexed DataFrame of adjusted close prices, one column per ticker.
        """
        cache_path = self.cache_dir / "sp500_universe.parquet"
        if cache_path.exists() and not force_download:
            logger.info("Loading cached universe from %s", cache_path)
            return pd.read_parquet(cache_path)

        prices = self.fetch_multiple_tickers(SP500_UNIVERSE)
        prices.to_parquet(cache_path)
        logger.info("Cached universe to %s  shape=%s", cache_path, prices.shape)
        return prices

    def fetch_multiple_tickers(self, tickers: List[str]) -> pd.DataFrame:
        """
        Download adjusted close prices for a list of tickers.

        Tickers that fail to download (delistings, bad symbols) are silently
        dropped. Only tickers with at least 95 % of expected trading days are
        kept.

        Parameters
        ----------
        tickers : list of str

        Returns
        -------
        pd.DataFrame
            Date-indexed DataFrame, one column per valid ticker.
        """
        logger.info("Fetching %d tickers from Yahoo Finance…", len(tickers))

        raw = yf.download(
            tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,          # gives us adjusted prices directly
            progress=False,
        )

        # yfinance returns a multi-level column frame when >1 ticker requested
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"]
        else:
            # Single ticker — raw itself has OHLCV columns
            prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

        # Drop columns with excessive missing data (< 95 % coverage)
        min_obs = int(len(prices) * 0.95)
        prices = prices.dropna(axis=1, thresh=min_obs)

        n_dropped = len(tickers) - prices.shape[1]
        if n_dropped:
            logger.warning("Dropped %d tickers due to insufficient data.", n_dropped)

        logger.info("Fetched price matrix: %s", prices.shape)
        return prices

    def load_from_parquet(self, path: str) -> pd.DataFrame:
        """Load a previously saved parquet file."""
        return pd.read_parquet(path)

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_returns(prices: pd.DataFrame, log: bool = True) -> pd.DataFrame:
        """
        Compute daily returns.

        Parameters
        ----------
        prices : pd.DataFrame
            Adjusted close prices.
        log : bool
            If True (default) compute log-returns; otherwise simple returns.

        Returns
        -------
        pd.DataFrame
        """
        if log:
            return np.log(prices / prices.shift(1)).dropna()
        return prices.pct_change().dropna()

    @staticmethod
    def align_prices(price_a: pd.Series, price_b: pd.Series) -> tuple[pd.Series, pd.Series]:
        """
        Inner-join two price series on their date index.

        Returns a pair of aligned, NaN-free series.
        """
        combined = pd.concat([price_a, price_b], axis=1).dropna()
        return combined.iloc[:, 0], combined.iloc[:, 1]

    @staticmethod
    def winsorise(returns: pd.DataFrame, lower: float = 0.005, upper: float = 0.995) -> pd.DataFrame:
        """
        Clip extreme return outliers at the given quantile bounds.

        Helps prevent outlier days (halts, earnings gaps) from distorting
        cointegration tests and Kalman filter estimates.
        """
        return returns.clip(
            lower=returns.quantile(lower),
            upper=returns.quantile(upper),
            axis=1,
        )
