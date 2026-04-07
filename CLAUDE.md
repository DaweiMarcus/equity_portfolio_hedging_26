# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

Analyze characteristics and risks of an equity portfolio, and test potential hedging strategies. Personal use only.

Build a deployable system called equity-portfolio-hedge for computing portfolio beta, sizing derivative hedges, and backtesting. Here's the structure I want:
* src/data/ — market data fetching (yfinance) and portfolio CSV loader. 
    * in this csv loader, it has headers of Symbol, Description, Quantity
* src/analytics/ — rolling beta vs SPY, correlation matrix, notional-adjusted exposure, VaR.
* fetch 2 years of weekly returns for each ticker and SPY, compute 52-week rolling OLS beta, and output a notional-weighted portfolio beta that accounts for leveraged ETFs.
* analysis_results - save the analysis of "52-week rolling OLS beta, and output a notional-weighted portfolio beta" into this folder in csv format
* also print the current notional-weighted portfolio beta to terminal.


Use pyproject.toml with dependencies: yfinance, numpy, pandas, scipy, matplotlib, plotly. Start by implementing the data layer and beta computation — I want to load a portfolio CSV (Symbol, Description, Quantity), 

make it with either desktop executable and CLI executable.

## Project Status

Early development — no source code exists yet. When adding code, build out the structure as needed and update this file with commands, architecture notes, and conventions discovered along the way.
