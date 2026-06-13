"""
var/data.py — Data fetching and preprocessing layer.
Handles all Yahoo Finance interaction and return computation.
"""

import pandas as pd
import yfinance as yf


class MarketDataLoader:
    """
    Fetches and preprocesses OHLCV data from Yahoo Finance.

    Attributes
    ----------
    tickers    : list[str]     — e.g. ['AAPL', 'MSFT']
    start_date : str           — 'YYYY-MM-DD'
    end_date   : str           — 'YYYY-MM-DD'
    prices     : pd.DataFrame  — adjusted closing prices
    returns    : pd.DataFrame  — daily log returns
    """

    def __init__(self, tickers: list, start_date: str, end_date: str):
        self.tickers    = tickers
        self.start_date = start_date
        self.end_date   = end_date
        self.prices     = self._fetch()
        self.returns    = self._compute_returns()

    def _fetch(self) -> pd.DataFrame:
        print(f"[DataLoader] Fetching: {', '.join(self.tickers)}")
        raw = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True,
            progress=False,
        )
        prices = raw["Close"]
        if isinstance(prices, pd.Series):          # single ticker edge case
            prices = prices.to_frame(name=self.tickers[0])
        prices.columns = self.tickers
        prices.dropna(how="all", inplace=True)
        print(f"[DataLoader] {len(prices)} rows | "
              f"{prices.index[0].date()} → {prices.index[-1].date()}")
        return prices

    def _compute_returns(self) -> pd.DataFrame:
        """Log returns: ln(P_t / P_{t-1}). More statistically sound than simple returns."""
        import numpy as np
        return np.log(self.prices / self.prices.shift(1)).dropna()

    def simple_returns(self) -> pd.DataFrame:
        """Simple percentage returns: (P_t - P_{t-1}) / P_{t-1}."""
        return self.prices.pct_change().dropna()

    @property
    def n_assets(self) -> int:
        return len(self.tickers)

    @property
    def n_observations(self) -> int:
        return len(self.returns)
