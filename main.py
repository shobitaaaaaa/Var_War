"""
main.py — Entry point for the VaR Calculator.

Usage:
    python main.py

All settings are in config.py.
"""

import config
from var import (
    MarketDataLoader,
    Portfolio,
    HistoricalVaR,
    ParametricVaR,
    MonteCarloVaR,
    ConditionalVaR,
    RollingVaR,
    VaRVisualizer,
)


def run():
    # ── 1. Load market data ───────────────────────────────────────────────────
    loader = MarketDataLoader(
        tickers    = config.TICKERS,
        start_date = config.START_DATE,
        end_date   = config.END_DATE,
    )

    # ── 2. Build portfolio ────────────────────────────────────────────────────
    portfolio = Portfolio(
        returns = loader.returns,
        weights = config.WEIGHTS,
    )

    print("\n── Portfolio Stats ──────────────────────────────────")
    stats = portfolio.stats_summary()
    for k, v in stats.items():
        print(f"  {k:<26}: {v}")
    print(f"\n{portfolio.asset_weights_summary()}\n")

    # ── 3. Run VaR models ─────────────────────────────────────────────────────
    shared = dict(
        port_returns     = portfolio.port_returns,
        confidence_level = config.CONFIDENCE_LEVEL,
        portfolio_val    = config.PORTFOLIO_VALUE,
    )

    h_var  = HistoricalVaR(**shared).compute()
    p_var  = ParametricVaR(**shared).compute()
    mc     = MonteCarloVaR(**shared, simulations=config.SIMULATIONS)
    mc_var = mc.compute()
    cvar   = ConditionalVaR(**shared).compute()

    rolling = RollingVaR(
        **shared,
        window = config.ROLLING_WINDOW,
    ).compute()

    results = {
        "historical_var"  : h_var,
        "parametric_var"  : p_var,
        "monte_carlo_var" : mc_var,
        "cvar"            : cvar,
    }

    # ── 4. Print summary ──────────────────────────────────────────────────────
    print("=" * 52)
    print("           VALUE-AT-RISK SUMMARY")
    print("=" * 52)
    print(f"  Portfolio Value   : ${config.PORTFOLIO_VALUE:>15,.2f}")
    print(f"  Confidence Level  : {config.CONFIDENCE_LEVEL:.0%}")
    print(f"  Period            : {config.START_DATE} → {config.END_DATE}")
    print("-" * 52)
    print(f"  Historical VaR    : ${h_var:>15,.2f}")
    print(f"  Parametric VaR    : ${p_var:>15,.2f}")
    print(f"  Monte Carlo VaR   : ${mc_var:>15,.2f}")
    print(f"  CVaR (ES)         : ${cvar:>15,.2f}")
    print("=" * 52)

    # ── 5. Plots ──────────────────────────────────────────────────────────────
    viz = VaRVisualizer(
        port_returns = portfolio.port_returns,
        results      = results,
        config       = config,
        save         = config.SAVE_PLOTS,
        plot_dir     = config.PLOT_DIR,
    )

    viz.plot_return_distribution()
    viz.plot_rolling_var(rolling)
    viz.plot_monte_carlo(mc.simulated_returns())
    viz.plot_correlation_heatmap(portfolio.correlation_matrix())
    viz.plot_var_comparison()


if __name__ == "__main__":
    run()
