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
│   │   └── fetcher.py              # yfinance market data fetcher
│   └── analytics/
│       └── beta.py                 # Rolling OLS beta and notional-weighted portfolio beta
├── analysis_results/               # Output CSVs (rolling_beta_*, portfolio_beta_*)
```

## Running the Program

Activate the virtual environment first:
```bash
source /Users/marcus/coding_python/venv_quant_finance_26/bin/activate
```

Place your portfolio CSV at `src/data/portfolio.csv`, then:
```bash
# CLI (defaults to src/data/portfolio.csv)
python main.py

# CLI with a different file in src/data/
python main.py my_holdings.csv

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

- **Data flow:** `loader.py` → `fetcher.py` → `beta.py` → saved to `analysis_results/`
- **Beta computation:** 52-week rolling OLS regression of weekly returns vs SPY using `scipy.stats.linregress`
- **Portfolio beta:** notional-weighted average of each position's effective beta
- **Effective beta:** `raw_beta × leverage_multiplier` (leverage multipliers in `src/config/leverage_map.yaml`)
- **Prices:** fetched live from yfinance at runtime — results vary while market is open due to the current incomplete weekly bar
- **Output:** two timestamped CSVs per run saved to `analysis_results/` — `rolling_beta_*.csv` and `portfolio_beta_*.csv`

## Conventions

- Every function must have a docstring
- Portfolio CSVs always read from `src/data/`
- Leverage config lives in `src/config/leverage_map.yaml` — add new ETFs there, no code changes needed

## Project Status

Data layer and beta computation implemented and tested. Currently on branch `add-optimizer-for-hedging-instruments`.

Next: hedging instruments optimizer (`src/hedging/`), reporting (`src/reporting/`), and backtesting.
