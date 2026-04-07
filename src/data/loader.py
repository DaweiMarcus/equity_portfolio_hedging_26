import pandas as pd
from pathlib import Path


def load_portfolio(csv_path: str | Path) -> pd.DataFrame:
    """
    Load portfolio CSV with required columns: Symbol, Description, Quantity.

    Handles common CSV formatting issues: UTF-8 BOM, leading/trailing
    whitespace in headers, and mixed-case column names.
    """
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    # Normalize column names: strip whitespace and title-case for matching
    df.columns = [c.strip() for c in df.columns]
    col_map = {c.lower(): c for c in df.columns}
    rename = {col_map[r.lower()]: r for r in ("Symbol", "Description", "Quantity") if r.lower() in col_map}
    df = df.rename(columns=rename)

    required = {"Symbol", "Description", "Quantity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )
    df["Symbol"] = df["Symbol"].str.upper().str.strip()
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="raise")
    return df[["Symbol", "Description", "Quantity"]].copy()
