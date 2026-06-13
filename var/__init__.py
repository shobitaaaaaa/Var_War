"""
var — Value-at-Risk calculation package.

Public API
----------
from var import VaREngine          # main orchestrator
from var.models import (
    HistoricalVaR, ParametricVaR,
    MonteCarloVaR, ConditionalVaR, RollingVaR
)
from var.portfolio import Portfolio
from var.data import MarketDataLoader
"""

from .data      import MarketDataLoader
from .portfolio import Portfolio
from .models    import (
    HistoricalVaR,
    ParametricVaR,
    MonteCarloVaR,
    ConditionalVaR,
    RollingVaR,
)
from .visualizer import VaRVisualizer

__all__ = [
    "MarketDataLoader",
    "Portfolio",
    "HistoricalVaR",
    "ParametricVaR",
    "MonteCarloVaR",
    "ConditionalVaR",
    "RollingVaR",
    "VaRVisualizer",
]
