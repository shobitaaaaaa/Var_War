"""
var/visualizer.py — All plotting logic for the VaR Calculator.
Isolated from models so you can swap matplotlib for plotly easily.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sns.set_theme(style="darkgrid", palette="muted")
COLORS = {"historical": "#E05C5C", "parametric": "#F5A623",
          "monte_carlo": "#4CAF50", "cvar": "#9B59B6", "portfolio": "#4C72B0"}


class VaRVisualizer:
    """
    Generates all plots for VaR analysis.

    Parameters
    ----------
    port_returns : pd.Series   — weighted portfolio daily returns
    results      : dict        — output from VaREngine.run()
    config       : object      — config module (for labels/settings)
    save         : bool        — save to disk instead of showing
    plot_dir     : str         — directory to save plots
    """

    def __init__(
        self,
        port_returns: pd.Series,
        results: dict,
        config,
        save: bool = False,
        plot_dir: str = "plots/",
    ):
        self.port_returns = port_returns
        self.results      = results
        self.config       = config
        self.save         = save
        self.plot_dir     = plot_dir
        if save:
            os.makedirs(plot_dir, exist_ok=True)

    def _save_or_show(self, name: str):
        if self.save:
            path = os.path.join(self.plot_dir, f"{name}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            print(f"[Plot] Saved → {path}")
            plt.close()
        else:
            plt.show()

    # ── 1. Return Distribution ────────────────────────────────────────────────

    def plot_return_distribution(self):
        alpha     = 1 - self.config.CONFIDENCE_LEVEL
        pv        = self.config.PORTFOLIO_VALUE
        h_line    = -(self.results["historical_var"]  / pv)
        p_line    = -(self.results["parametric_var"]  / pv)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.hist(self.port_returns, bins=60, color=COLORS["portfolio"],
                alpha=0.75, edgecolor="white", label="Daily Portfolio Returns")
        ax.axvline(h_line, color=COLORS["historical"], lw=2, ls="--",
                   label=f"Historical VaR ({self.config.CONFIDENCE_LEVEL:.0%}): {h_line:.4f}")
        ax.axvline(p_line, color=COLORS["parametric"], lw=2, ls="-.",
                   label=f"Parametric VaR ({self.config.CONFIDENCE_LEVEL:.0%}): {p_line:.4f}")
        ax.fill_betweenx(
            [0, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 100],
            self.port_returns.min(), min(h_line, p_line),
            alpha=0.08, color=COLORS["historical"], label="Loss Tail"
        )
        ax.set_title("Portfolio Daily Return Distribution", fontsize=14, fontweight="bold")
        ax.set_xlabel("Daily Log Return")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=9)
        plt.tight_layout()
        self._save_or_show("return_distribution")

    # ── 2. Rolling VaR ────────────────────────────────────────────────────────

    def plot_rolling_var(self, rolling_series: pd.Series):
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(rolling_series.index, rolling_series.values,
                color=COLORS["historical"], lw=1.5,
                label=f"{self.config.ROLLING_WINDOW}-Day Rolling Historical VaR")
        ax.fill_between(rolling_series.index, rolling_series.values,
                        alpha=0.15, color=COLORS["historical"])
        ax.set_title(
            f"Rolling {self.config.ROLLING_WINDOW}-Day Historical VaR "
            f"(${self.config.PORTFOLIO_VALUE:,.0f} portfolio)",
            fontsize=13, fontweight="bold"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("VaR ($)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.legend()
        plt.tight_layout()
        self._save_or_show("rolling_var")

    # ── 3. Monte Carlo Distribution ──────────────────────────────────────────

    def plot_monte_carlo(self, simulated_returns: np.ndarray):
        alpha   = 1 - self.config.CONFIDENCE_LEVEL
        mc_line = np.percentile(simulated_returns, alpha * 100)

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.hist(simulated_returns, bins=80, color=COLORS["monte_carlo"],
                alpha=0.7, edgecolor="white", label="Simulated Returns")
        ax.axvline(mc_line, color=COLORS["historical"], lw=2, ls="--",
                   label=f"MC VaR ({self.config.CONFIDENCE_LEVEL:.0%}): {mc_line:.4f}")
        ax.set_title(
            f"Monte Carlo Simulated Return Distribution (n={self.config.SIMULATIONS:,})",
            fontsize=13, fontweight="bold"
        )
        ax.set_xlabel("Simulated Daily Return")
        ax.set_ylabel("Frequency")
        ax.legend()
        plt.tight_layout()
        self._save_or_show("monte_carlo")

    # ── 4. Correlation Heatmap ───────────────────────────────────────────────

    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame):
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax,
            annot_kws={"size": 11}
        )
        ax.set_title("Asset Return Correlation Matrix", fontsize=13, fontweight="bold")
        plt.tight_layout()
        self._save_or_show("correlation_heatmap")

    # ── 5. VaR Comparison Bar Chart ──────────────────────────────────────────

    def plot_var_comparison(self):
        labels = ["Historical", "Parametric", "Monte Carlo", "CVaR (ES)"]
        values = [
            self.results["historical_var"],
            self.results["parametric_var"],
            self.results["monte_carlo_var"],
            self.results["cvar"],
        ]
        colors = [COLORS["historical"], COLORS["parametric"],
                  COLORS["monte_carlo"], COLORS["cvar"]]

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white")
        ax.bar_label(bars, labels=[f"${v:,.0f}" for v in values],
                     padding=6, fontsize=10, fontweight="bold")
        ax.set_title(
            f"VaR Method Comparison — {self.config.CONFIDENCE_LEVEL:.0%} Confidence",
            fontsize=13, fontweight="bold"
        )
        ax.set_ylabel("VaR ($)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.set_ylim(0, max(values) * 1.2)
        plt.tight_layout()
        self._save_or_show("var_comparison")
