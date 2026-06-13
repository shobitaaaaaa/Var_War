"""
config.py — Central configuration for VaR Calculator.
Edit this file to change tickers, dates, risk parameters.
"""

# ── Portfolio ─────────────────────────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN"]
WEIGHTS = [0.30, 0.30, 0.20, 0.20]          # must sum to 1 (auto-normalised)

# ── Date Range ────────────────────────────────────────────────────────────────
START_DATE = "2023-01-01"
END_DATE   = "2025-01-01"

# ── Risk Parameters ───────────────────────────────────────────────────────────
PORTFOLIO_VALUE  = 1_000_000   # USD
CONFIDENCE_LEVEL = 0.95        # 95% — use 0.99 for Basel III
ROLLING_WINDOW   = 252         # trading days in 1 year
SIMULATIONS      = 10_000      # Monte Carlo paths

# ── Output ────────────────────────────────────────────────────────────────────
SAVE_PLOTS  = False            # True → saves PNGs instead of showing
PLOT_DIR    = "plots/"
