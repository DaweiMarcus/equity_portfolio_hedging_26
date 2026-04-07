from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf


def fetch_weekly_returns(
    tickers: list[str], years: int = 2
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Fetch 2 years of weekly adjusted close prices for tickers + SPY.

    Returns:
        returns:       DataFrame of weekly pct_change returns (columns = tickers + SPY)
        latest_prices: Series of most recent closing price per ticker
    """
    end = datetime.today()
    start = end - timedelta(days=years * 365)

    all_tickers = sorted(set(tickers) | {"SPY"})
    raw = yf.download(
        all_tickers,
        start=start,
        end=end,
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
