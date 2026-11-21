# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Analyze characteristics and risks of an equity portfolio, and test potential hedging strategies. Personal use only.

## Project Structure

```
equity_portfolio_hedging_26/
├── main.py                         # CLI convenience wrapper
├── app.py                          # Desktop GUI convenience wrapper
├── pyproject.toml                  # Package config and dependencies
├── src/
│   ├── cli.py                      # CLI entry point (portfolio-hedge)
│   ├── gui.py                      # Desktop GUI entry point (portfolio-hedge-gui, tkinter)
│   ├── config/
│   │   └── leverage_map.yaml       # Leverage multipliers for leveraged/inverse ETFs
│   ├── data/
│   │   ├── loader.py               # Portfolio CSV loader (Symbol, Description, Quantity)
│   │   └── fetcher.py              # yfinance market data fetcher (weekly + daily)
│   └── analytics/
│       ├── beta.py                 # Rolling OLS beta and notional-weighted portfolio beta
│       └── regression.py           # Portfolio vs SPX OLS regression (returns and PnL)
├── analysis_results/               # Output CSVs (rolling_beta_*, portfolio_beta_*)
```

## Running the Program

Activate the virtual environment first:
```bash
source /Users/marcus/coding_python/venv_quant_finance_26/bin/activate
```

Place your portfolio CSV at `src/data/portfolio.csv`, then:
```bash
# CLI — start_date required, end_date optional (defaults to last trading day)
python main.py <filename> <start_date> [end_date]

# Examples
python main.py portfolio.csv 2023-01-01
python main.py portfolio.csv 2023-01-01 2024-12-31
python main.py my_holdings.csv 2022-06-01

# Desktop GUI
python app.py
```

## Portfolio CSV Format

File must be placed in `src/data/`. Required columns (case-insensitive, BOM-safe):
```
Symbol, Description, Quantity
```

## Dependencies

Managed via `pyproject.toml`. Installed in venv at `/Users/marcus/coding_python/venv_quant_finance_26`.

```
yfinance, numpy, pandas, scipy, matplotlib, plotly, pyyaml
```

To reinstall:
```bash
pip install -e .
```

## Architecture Notes

- **Data flow:** `loader.py` → `fetcher.py` → `beta.py` + `regression.py` → saved to `analysis_results/`
- **Date range:** CLI accepts `start_date` (required) and `end_date` (optional, defaults to last weekday) in `yyyy-mm-dd` format
- **Beta computation:** 52-week rolling OLS regression of weekly returns vs SPY using `scipy.stats.linregress`
- **Portfolio beta:** notional-weighted average of each position's effective beta
- **Effective beta:** `raw_beta × leverage_multiplier` (leverage multipliers in `src/config/leverage_map.yaml`)
- **Regression analyses:** portfolio positions held constant; portfolio value = Σ(price × quantity) per day; regressed against SPX (`^GSPC`)
  - `compute_portfolio_regression_return`: x = SPX daily % return, y = portfolio daily % return
  - `compute_portfolio_regression_PnL`: x = SPX daily point change, y = portfolio daily PnL ($)
- **Fetcher functions:**
  - `fetch_weekly_returns`: weekly interval, includes SPY, returns pct_change DataFrame + latest prices
  - `fetch_daily_prices`: daily interval, includes `^GSPC`, returns adjusted close price DataFrame
- **Output:** two timestamped CSVs per run saved to `analysis_results/` — `rolling_beta_*.csv` and `portfolio_beta_*.csv`; regression results printed to terminal only with matplotlib plots

## Conventions

- Every function must have a docstring
- Portfolio CSVs always read from `src/data/`
- Leverage config lives in `src/config/leverage_map.yaml` — add new ETFs there, no code changes needed

## Project Status

Data layer, beta computation, and portfolio regression analyses implemented. Currently on branch `add-optimizer-for-hedging-instruments`.

Next: hedging instruments optimizer (`src/hedging/`), reporting (`src/reporting/`), and backtesting.
