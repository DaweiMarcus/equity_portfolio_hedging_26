"""CLI entry point: portfolio-hedge [filename]"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

from src.analytics.beta import compute_portfolio_beta, compute_rolling_beta
from src.data.fetcher import fetch_weekly_returns
from src.data.loader import load_portfolio

# Portfolio CSVs are always read from this directory
DATA_DIR = Path(__file__).parent / "data"


def get_last_trading_day() -> datetime:
    """Return the most recent weekday (Mon–Fri) as of today, at midnight."""
    today = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
    # weekday(): Mon=0 … Fri=4, Sat=5, Sun=6
    days_back = max(0, today.weekday() - 4)
    return today - timedelta(days=days_back)


def run_analysis(filename: str, start_date: datetime, end_date: datetime) -> None:
    """
    Run the full beta analysis pipeline and save results to analysis_results/.

    Resolves the portfolio CSV from src/data/<filename>, fetches weekly returns
    over [start_date, end_date], computes 52-week rolling OLS beta per ticker,
    and prints the notional-weighted portfolio beta.

    Args:
        filename:   CSV filename (no directory needed) located in src/data/.
        start_date: History start date (inclusive).
        end_date:   History end date (inclusive).
    """
    csv_path = DATA_DIR / filename
    if not csv_path.exists():
        raise FileNotFoundError(f"Portfolio CSV not found: {csv_path}")

    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)

    print(f"Run date : {datetime.today().strftime('%Y-%m-%d')}")
    print(f"Period   : {start_date.strftime('%Y-%m-%d')} -> {end_date.strftime('%Y-%m-%d')}")
    print()

    print(f"Loading portfolio from {csv_path} ...")
    portfolio = load_portfolio(csv_path)
    tickers = portfolio["Symbol"].tolist()
    print(f"  Tickers: {', '.join(tickers)}")

    print("Fetching weekly returns (+ SPY) ...")
    returns, latest_prices = fetch_weekly_returns(tickers, start_date, end_date)

    print(f"Computing 52-week rolling OLS beta ({len(returns)} weekly observations) ...")
    rolling_betas = compute_rolling_beta(returns)

    print("Computing notional-weighted portfolio beta ...")
    portfolio_beta, detail_df = compute_portfolio_beta(portfolio, rolling_betas, latest_prices)

    # Save results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rolling_path = output_dir / f"rolling_beta_{ts}.csv"
    detail_path = output_dir / f"portfolio_beta_{ts}.csv"
    rolling_betas.to_csv(rolling_path)
    detail_df.to_csv(detail_path, index=False)

    # Print summary table
    print()
    print(detail_df[["Symbol", "Notional", "Leverage", "Rolling_Beta", "Effective_Beta", "Weight"]].to_string(index=False))
    print()
    print("=" * 52)
    print(f"  Notional-Weighted Portfolio Beta:  {portfolio_beta:.4f}")
    print("=" * 52)
    print(f"\nSaved: {rolling_path}")
    print(f"Saved: {detail_path}")


def main() -> None:
    """
    Parse positional CLI arguments and delegate to run_analysis.

    Usage:
        python main.py <filename> <start_date> [end_date]

    Args:
        filename:   CSV filename inside src/data/.
        start_date: History start date in yyyy-mm-dd format.
        end_date:   (optional) History end date in yyyy-mm-dd format.
                    Defaults to the last trading day (most recent weekday).

    Examples:
        python main.py portfolio.csv 2023-01-01
        python main.py portfolio.csv 2023-01-01 2024-12-31
    """
    argv = sys.argv[1:]

    if len(argv) < 2:
        print("Usage: python main.py <filename> <start_date> [end_date]")
        print("  start_date / end_date format: yyyy-mm-dd")
        sys.exit(1)

    filename = argv[0]
    start_date_str = argv[1]
    end_date_str = argv[2] if len(argv) >= 3 else None

    # Parse start_date
    try:
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    except ValueError:
        print(f"Invalid start_date '{start_date_str}'. Expected yyyy-mm-dd.")
        sys.exit(1)

    # Parse end_date or default to last trading day
    if end_date_str:
        try:
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        except ValueError:
            print(f"Invalid end_date '{end_date_str}'. Expected yyyy-mm-dd.")
            sys.exit(1)
    else:
        end_date = get_last_trading_day()

    if end_date < start_date:
        print(f"end_date ({end_date.date()}) must not be before start_date ({start_date.date()}).")
        sys.exit(1)

    run_analysis(filename, start_date, end_date)


if __name__ == "__main__":
    main()
