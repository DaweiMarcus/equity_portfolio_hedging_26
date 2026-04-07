"""CLI entry point: portfolio-hedge [filename]"""

import argparse
from datetime import datetime
from pathlib import Path

from src.analytics.beta import compute_portfolio_beta, compute_rolling_beta
from src.data.fetcher import fetch_weekly_returns
from src.data.loader import load_portfolio

# Portfolio CSVs are always read from this directory
DATA_DIR = Path(__file__).parent / "data"


def run_analysis(filename: str) -> None:
    """
    Run the full beta analysis pipeline and save results to analysis_results/.

    Resolves the portfolio CSV from src/data/<filename>, then fetches 2 years
    of weekly returns, computes 52-week rolling OLS beta per ticker, and
    prints and saves the notional-weighted portfolio beta.

    Args:
        filename: CSV filename (no directory needed) located in src/data/.
    """
    csv_path = DATA_DIR / filename
    if not csv_path.exists():
        raise FileNotFoundError(f"Portfolio CSV not found: {csv_path}")

    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)

    print(f"Loading portfolio from {csv_path} ...")
    portfolio = load_portfolio(csv_path)
    tickers = portfolio["Symbol"].tolist()
    print(f"  Tickers: {', '.join(tickers)}")

    print("Fetching 2 years of weekly returns (+ SPY) ...")
    returns, latest_prices = fetch_weekly_returns(tickers)

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
    """Parse CLI arguments and delegate to run_analysis."""
    parser = argparse.ArgumentParser(
        prog="portfolio-hedge",
        description="Compute rolling beta and notional-weighted portfolio beta.",
    )
    parser.add_argument(
        "filename",
        nargs="?",
        default="portfolio.csv",
        help="CSV filename inside src/data/ (default: portfolio.csv)",
    )
    args = parser.parse_args()
    run_analysis(args.filename)


if __name__ == "__main__":
    main()
