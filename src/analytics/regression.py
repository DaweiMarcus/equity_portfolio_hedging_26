import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


def _build_portfolio_value(prices: pd.DataFrame, quantities: dict[str, float]) -> pd.Series:
    """
    Compute daily portfolio value as sum(price_i * quantity_i) for each day.

    Args:
        prices:     DataFrame of daily adjusted close prices.
        quantities: Dict mapping ticker symbol -> share quantity.

    Returns:
        Series of daily portfolio values indexed by date.
    """
    portfolio_value = pd.Series(0.0, index=prices.index, name="Portfolio")
    for ticker, qty in quantities.items():
        if ticker not in prices.columns:
            print(f"  Warning: {ticker} not found in price data — skipped.")
            continue
        portfolio_value += prices[ticker] * qty
    return portfolio_value


def _run_ols_and_plot(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    """
    Run OLS regression on x and y, print results, and display scatter + fit plot.

    Args:
        x:      Independent variable array.
        y:      Dependent variable array.
        title:  Plot and section title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

    print()
    print("=" * 52)
    print(f"  {title}")
    print("=" * 52)
    print(f"  Coefficient (slope) : {slope:.4f}")
    print(f"  Intercept           : {intercept:.6f}")
    print(f"  R²                  : {r_value**2:.4f}")
    print(f"  P-value             : {p_value:.4e}")
    print(f"  Std error           : {std_err:.6f}")
    print(f"  Observations        : {len(x)}")
    print("=" * 52)

    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = slope * x_line + intercept

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, alpha=0.45, s=18, color="steelblue", label="Observed")
    ax.plot(
        x_line,
        y_line,
        color="tomato",
        linewidth=2,
        label=f"OLS fit:  y = {slope:.3f}x + {intercept:.4f}  (R²={r_value**2:.3f})",
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def compute_portfolio_regression_return(
    prices: pd.DataFrame,
    quantities: dict[str, float],
) -> None:
    """
    Regress daily portfolio returns against daily SPX returns via OLS.

    Holds portfolio positions constant at the given quantities. Portfolio value
    on each day = sum(price_i * quantity_i). Daily changes are expressed as
    percentage returns (pct_change). Prints slope, intercept, R², p-value,
    and displays a scatter + regression line plot.

    Args:
        prices:     DataFrame of daily adjusted close prices; must contain a
                    '^GSPC' column plus one column per portfolio ticker.
        quantities: Dict mapping ticker symbol -> share quantity.
    """
    portfolio_value = _build_portfolio_value(prices, quantities)
    spx = prices["^GSPC"]

    df = pd.DataFrame({
        "x": spx.pct_change(),
        "y": portfolio_value.pct_change(),
    }).dropna()

    _run_ols_and_plot(
        x=df["x"].values,
        y=df["y"].values,
        title="Portfolio vs SPX — Daily Return Regression",
        xlabel="SPX Daily Return",
        ylabel="Portfolio Daily Return",
    )


def compute_portfolio_regression_PnL(
    prices: pd.DataFrame,
    quantities: dict[str, float],
) -> None:
    """
    Regress daily portfolio PnL against daily SPX point changes via OLS.

    Holds portfolio positions constant at the given quantities. Portfolio value
    on each day = sum(price_i * quantity_i). Daily changes are expressed as
    absolute differences (diff): portfolio PnL in dollars, SPX in index points.
    Prints slope, intercept, R², p-value, and displays a scatter + regression
    line plot.

    Args:
        prices:     DataFrame of daily adjusted close prices; must contain a
                    '^GSPC' column plus one column per portfolio ticker.
        quantities: Dict mapping ticker symbol -> share quantity.
    """
    portfolio_value = _build_portfolio_value(prices, quantities)
    spx = prices["^GSPC"]

    df = pd.DataFrame({
        "x": spx.diff(),
        "y": portfolio_value.diff(),
    }).dropna()

    _run_ols_and_plot(
        x=df["x"].values,
        y=df["y"].values,
        title="Portfolio vs SPX — Daily PnL Regression",
        xlabel="SPX Daily Change (index points)",
        ylabel="Portfolio Daily PnL ($)",
    )
