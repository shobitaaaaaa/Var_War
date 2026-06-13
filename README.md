# VaR Calculator

A modular **Value-at-Risk** engine for equity portfolios, implementing three risk methodologies and Expected Shortfall.

## Methods

| Method | Assumption | Use Case |
|---|---|---|
| Historical Simulation | None — uses empirical returns | Fat-tailed assets |
| Parametric (Var-Cov) | Returns ~ Normal(μ, σ) | Quick estimates |
| Monte Carlo | Simulated paths from fitted distribution | Extensible to complex models |
| CVaR / Expected Shortfall | Average loss beyond VaR | Basel III compliance |

## Project Structure

```
VaRCalculator/
├── var/
│   ├── data.py        # MarketDataLoader — Yahoo Finance ingestion
│   ├── portfolio.py   # Portfolio — weights, returns, correlation
│   ├── models.py      # VaR model classes (all inherit BaseVaRModel)
│   └── visualizer.py  # VaRVisualizer — all matplotlib plots
├── main.py            # Entry point
├── config.py          # All user settings
└── requirements.txt
```

## Quickstart

```bash
pip install -r requirements.txt
python main.py
```

## Configuration

Edit `config.py`:

```python
TICKERS          = ["AAPL", "MSFT", "GOOGL", "AMZN"]
WEIGHTS          = [0.30, 0.30, 0.20, 0.20]
START_DATE       = "2023-01-01"
END_DATE         = "2025-01-01"
PORTFOLIO_VALUE  = 1_000_000
CONFIDENCE_LEVEL = 0.95       # use 0.99 for Basel III
SIMULATIONS      = 10_000
```

## Output

```
── Portfolio Stats ───────────────────────────────────
  mean_daily_return         : 0.000812
  annualised_vol            : 0.1843
  sharpe_ratio              : 0.6991
  total_observations        : 502

══════════════════════════════════════════════════════
           VALUE-AT-RISK SUMMARY
══════════════════════════════════════════════════════
  Portfolio Value   :       $1,000,000.00
  Confidence Level  : 95%
  Period            : 2023-01-01 → 2025-01-01
────────────────────────────────────────────────────
  Historical VaR    :          $18,432.00
  Parametric VaR    :          $19,107.00
  Monte Carlo VaR   :          $18,891.00
  CVaR (ES)         :          $26,344.00
══════════════════════════════════════════════════════
```

## Extending

Add a new VaR model by subclassing `BaseVaRModel` in `var/models.py`:

```python
class GARCHVaR(BaseVaRModel):
    def compute(self) -> float:
        # fit GARCH(1,1), forecast σ, then apply parametric formula
        ...
```

## Reference


