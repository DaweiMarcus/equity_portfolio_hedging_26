import pandas as pd
from pathlib import Path


def load_portfolio(csv_path: str | Path) -> pd.DataFrame:
    """Load portfolio CSV with required columns: Symbol, Description, Quantity."""
    df = pd.read_csv(csv_path)
    required = {"Symbol", "Description", "Quantity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    df["Symbol"] = df["Symbol"].str.upper().str.strip()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="raise")
    return df[["Symbol", "Description", "Quantity"]].copy()
