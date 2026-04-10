from datetime import datetime

import pandas as pd
import yfinance as yf


def fetch_weekly_returns(
    tickers: list[str], start_date: datetime, end_date: datetime
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Fetch weekly adjusted close prices for tickers + SPY over the given date range.

    Args:
        tickers:    List of ticker symbols to fetch.
        start_date: History start date (inclusive).
        end_date:   History end date (inclusive); typically the last trading day.

    Returns:
        returns:       DataFrame of weekly pct_change returns (columns = tickers + SPY)
        latest_prices: Series of most recent closing price per ticker
    """
    all_tickers = sorted(set(tickers) | {"SPY"})
    raw = yf.download(
        all_tickers,
        start=start_date,
        end=end_date,
        interval="1wk",
        auto_adjust=True,
        progress=False,
    )

    # yfinance returns MultiIndex columns when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # Single ticker — columns are flat
        prices = raw[["Close"]].rename(columns={"Close": all_tickers[0]})

    latest_prices = prices.ffill().iloc[-1]
    returns = prices.pct_change().dropna()
    return returns, latest_prices
