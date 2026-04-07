import numpy as np
import pandas as pd
from scipy import stats

# Leverage multipliers for known leveraged/inverse ETFs
LEVERAGE_MAP: dict[str, float] = {
    # 3x long
    "TQQQ": 3.0, "UPRO": 3.0, "SPXL": 3.0, "SOXL": 3.0,
    "UDOW": 3.0, "TNA": 3.0, "LABU": 3.0,
    # 2x long
    "SSO": 2.0, "QLD": 2.0, "USD": 2.0, "UWM": 2.0,
    # 1x inverse
    "SH": -1.0, "PSQ": -1.0, "DOG": -1.0,
    # 2x inverse
    "SDS": -2.0, "QID": -2.0,
    # 3x inverse
    "SPXU": -3.0, "SQQQ": -3.0, "SDOW": -3.0,
}


def compute_rolling_beta(returns: pd.DataFrame, window: int = 52) -> pd.DataFrame:
    """
    Compute 52-week rolling OLS beta for each ticker vs SPY.

    Args:
        returns: DataFrame with weekly returns; must contain a 'SPY' column.
        window:  Rolling window in weeks (default 52).

    Returns:
        DataFrame of rolling betas indexed by date.
    """
    spy = returns["SPY"]
    tickers = [c for c in returns.columns if c != "SPY"]

    beta_series: dict[str, pd.Series] = {}
    for ticker in tickers:
        aligned = pd.concat([returns[ticker], spy], axis=1).dropna()
        aligned.columns = [ticker, "SPY"]

        if len(aligned) < window:
            beta_series[ticker] = pd.Series(dtype=float)
            continue

        betas = []
        for i in range(window, len(aligned) + 1):
            w = aligned.iloc[i - window : i]
            slope, *_ = stats.linregress(w["SPY"], w[ticker])
            betas.append(slope)

        beta_series[ticker] = pd.Series(betas, index=aligned.index[window - 1 :])

    return pd.DataFrame(beta_series)


def compute_portfolio_beta(
    portfolio: pd.DataFrame,
    rolling_betas: pd.DataFrame,
    latest_prices: pd.Series,
) -> tuple[float, pd.DataFrame]:
    """
    Compute current notional-weighted portfolio beta.

    Applies leverage multipliers for leveraged/inverse ETFs.

    Args:
        portfolio:     DataFrame with columns Symbol, Description, Quantity.
        rolling_betas: Output of compute_rolling_beta.
        latest_prices: Most recent price per ticker (from fetcher).

    Returns:
        (portfolio_beta, detail_df)
    """
    latest_betas = rolling_betas.iloc[-1] if not rolling_betas.empty else pd.Series(dtype=float)

    rows = []
    for _, row in portfolio.iterrows():
        sym = row["Symbol"]
        qty = float(row["Quantity"])
        price = float(latest_prices.get(sym, np.nan))
        leverage = LEVERAGE_MAP.get(sym, 1.0)
        raw_beta = float(latest_betas.get(sym, np.nan))
        effective_beta = raw_beta * leverage if not np.isnan(raw_beta) else np.nan
        notional = qty * price if not np.isnan(price) else np.nan
        notional_x_beta = (
            notional * effective_beta
            if not (np.isnan(notional) or np.isnan(effective_beta))
            else np.nan
        )
        rows.append(
            {
                "Symbol": sym,
                "Description": row["Description"],
                "Quantity": qty,
                "Price": round(price, 4) if not np.isnan(price) else np.nan,
                "Notional": round(notional, 2) if not np.isnan(notional) else np.nan,
                "Leverage": leverage,
                "Rolling_Beta": round(raw_beta, 6) if not np.isnan(raw_beta) else np.nan,
                "Effective_Beta": round(effective_beta, 6) if not np.isnan(effective_beta) else np.nan,
                "Notional_x_Beta": round(notional_x_beta, 2) if not np.isnan(notional_x_beta) else np.nan,
            }
        )

    detail_df = pd.DataFrame(rows)
    total_notional = detail_df["Notional"].sum()
    portfolio_beta = (
        detail_df["Notional_x_Beta"].sum() / total_notional
        if total_notional and total_notional != 0
        else np.nan
    )
    detail_df["Weight"] = (detail_df["Notional"] / total_notional).round(6)

    return float(portfolio_beta), detail_df
