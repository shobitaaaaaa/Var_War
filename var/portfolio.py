"""
var/portfolio.py — Portfolio construction layer.
Handles weights, weighted returns, and correlation/covariance analysis.
"""

import numpy as np
import pandas as pd


class Portfolio:
    """
    Constructs a weighted portfolio from individual asset returns.

    Parameters
    ----------
    returns : pd.DataFrame — per-asset daily returns (from MarketDataLoader)
    weights : list/None    — portfolio weights; equal-weight if None
    """

    def __init__(self, returns: pd.DataFrame, weights: list = None):
        self.returns  = returns
        self.tickers  = list(returns.columns)
        self.weights  = self._validate_weights(weights)
        self.port_returns = self._weighted_returns()

    def _validate_weights(self, weights) -> np.ndarray:
        if weights is None:
            n = len(self.tickers)
            return np.ones(n) / n
        w = np.array(weights, dtype=float)
        if len(w) != len(self.tickers):
            raise ValueError(
                f"Expected {len(self.tickers)} weights, got {len(w)}"
            )
        return w / w.sum()      # normalise to sum-to-1

    def _weighted_returns(self) -> pd.Series:
        return self.returns.dot(self.weights).rename("portfolio")

    @property
    def mean_return(self) -> float:
        return float(self.port_returns.mean())

    @property
    def volatility(self) -> float:
        """Annualised portfolio volatility (σ × √252)."""
        return float(self.port_returns.std() * np.sqrt(252))

    @property
    def sharpe_ratio(self) -> float:
        """Annualised Sharpe ratio assuming 0% risk-free rate."""
        daily_sharpe = self.mean_return / (self.port_returns.std() + 1e-10)
        return float(daily_sharpe * np.sqrt(252))

    def correlation_matrix(self) -> pd.DataFrame:
        return self.returns.corr()

    def covariance_matrix(self) -> pd.DataFrame:
        return self.returns.cov()

    def asset_weights_summary(self) -> pd.DataFrame:
        return pd.DataFrame({
            "Ticker": self.tickers,
            "Weight": [f"{w:.1%}" for w in self.weights],
        }).set_index("Ticker")

    def stats_summary(self) -> dict:
        return {
            "mean_daily_return"   : round(self.mean_return, 6),
            "annualised_vol"      : round(self.volatility, 4),
            "sharpe_ratio"        : round(self.sharpe_ratio, 4),
            "total_observations"  : len(self.port_returns),
        }
