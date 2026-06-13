"""
var/models.py — VaR model implementations.

Each method is a separate class inheriting from BaseVaRModel,
making it easy to extend with new models (e.g. GARCH-VaR, EVT-VaR).

Classes
-------
BaseVaRModel        — abstract base with shared interface
HistoricalVaR       — empirical distribution, no assumptions
ParametricVaR       — normal distribution (variance-covariance)
MonteCarloVaR       — simulation-based
ConditionalVaR      — Expected Shortfall (CVaR / ES)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from abc import ABC, abstractmethod


class BaseVaRModel(ABC):
    """
    Abstract base class for all VaR models.
    Every subclass must implement compute() and return a float (dollar VaR).
    """

    def __init__(
        self,
        port_returns: pd.Series,
        confidence_level: float,
        portfolio_val: float,
    ):
        self.port_returns     = port_returns
        self.confidence_level = confidence_level
        self.portfolio_val    = portfolio_val
        self.alpha            = 1 - confidence_level   # tail probability

    @abstractmethod
    def compute(self) -> float:
        """Return VaR in dollars (positive number = potential loss)."""
        ...

    def _to_dollars(self, var_pct: float) -> float:
        return abs(var_pct) * self.portfolio_val


class HistoricalVaR(BaseVaRModel):
    """
    Historical Simulation VaR.

    Uses the empirical distribution of actual past returns.
    No distributional assumption — captures fat tails and skewness.

    VaR = percentile(returns, α) × portfolio_value
    """

    def compute(self) -> float:
        var_pct = np.percentile(self.port_returns, self.alpha * 100)
        return self._to_dollars(var_pct)


class ParametricVaR(BaseVaRModel):
    """
    Parametric (Variance-Covariance) VaR.

    Assumes returns ~ Normal(μ, σ).
    Uses the analytical formula:  VaR = -(μ + Z_α × σ)

    where Z_α is the α-quantile of the standard normal distribution.
    Fast but underestimates tail risk for fat-tailed assets.
    """

    def compute(self) -> float:
        mu    = self.port_returns.mean()
        sigma = self.port_returns.std()
        z     = norm.ppf(self.alpha)        # e.g. -1.645 at α=0.05
        var_pct = mu + z * sigma
        return self._to_dollars(var_pct)

    @property
    def mu(self) -> float:
        return float(self.port_returns.mean())

    @property
    def sigma(self) -> float:
        return float(self.port_returns.std())

    @property
    def z_score(self) -> float:
        return float(norm.ppf(self.alpha))


class MonteCarloVaR(BaseVaRModel):
    """
    Monte Carlo Simulation VaR.

    Fits Normal(μ, σ) to historical returns, generates N random paths,
    then takes the empirical α-percentile of simulated P&Ls.

    Equivalent to Parametric at this level but extensible to:
      - Correlated multi-asset simulations (Cholesky decomposition)
      - Fat-tailed distributions (Student-t)
      - Time-varying volatility (GARCH)

    Parameters
    ----------
    simulations : int  — number of random paths (default 10,000)
    seed        : int  — random seed for reproducibility
    """

    def __init__(self, *args, simulations: int = 10_000, seed: int = 42, **kwargs):
        super().__init__(*args, **kwargs)
        self.simulations = simulations
        self.seed        = seed

    def compute(self) -> float:
        mu    = self.port_returns.mean()
        sigma = self.port_returns.std()
        np.random.seed(self.seed)
        simulated = np.random.normal(mu, sigma, self.simulations)
        var_pct   = np.percentile(simulated, self.alpha * 100)
        return self._to_dollars(var_pct)

    def simulated_returns(self) -> np.ndarray:
        """Expose simulated return array for plotting."""
        mu    = self.port_returns.mean()
        sigma = self.port_returns.std()
        np.random.seed(self.seed)
        return np.random.normal(mu, sigma, self.simulations)


class ConditionalVaR(BaseVaRModel):
    """
    Conditional VaR (CVaR) — also called Expected Shortfall (ES).

    CVaR = E[ Loss | Loss > VaR ]
    i.e. the average loss on the days that breach VaR.

    Always >= VaR. Required under Basel III / FRTB (2016) which replaced
    VaR with ES as the primary market risk metric for trading books.
    """

    def compute(self) -> float:
        cutoff    = np.percentile(self.port_returns, self.alpha * 100)
        tail      = self.port_returns[self.port_returns <= cutoff]
        cvar_pct  = tail.mean()
        return self._to_dollars(cvar_pct)


class RollingVaR:
    """
    Computes rolling Historical VaR over a sliding window.
    Not a BaseVaRModel subclass — returns a Series, not a scalar.

    Parameters
    ----------
    port_returns     : pd.Series
    confidence_level : float
    portfolio_val    : float
    window           : int      — rolling window in trading days
    """

    def __init__(
        self,
        port_returns: pd.Series,
        confidence_level: float,
        portfolio_val: float,
        window: int = 252,
    ):
        self.port_returns     = port_returns
        self.confidence_level = confidence_level
        self.portfolio_val    = portfolio_val
        self.window           = window
        self.alpha            = 1 - confidence_level

    def compute(self) -> pd.Series:
        """Returns time-series of rolling VaR in dollars."""
        return (
            self.port_returns
            .rolling(self.window)
            .quantile(self.alpha)
            .dropna()
            .abs()
            .mul(self.portfolio_val)
            .rename("rolling_var")
        )
