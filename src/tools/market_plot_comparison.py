"""
Standalone tool: compare 1-16 market variables on one interactive, z-scored chart.

Each variable is a ticker (equity, index, ETF, ...) with a transform:
    level   the adjusted close itself (e.g. VIX level)
    change  absolute difference vs. n ago  (level_t - level_{t-n}),  e.g. +3 pts
    return  percentage change vs. n ago    (level_t / level_{t-n} - 1)

Windows (for change / return):
    1d, 5d, 10d   trading rows (the previous 1, 5 or 10 observations)
    1m, 3m        calendar offsets (date minus 1 / 3 calendar months, using the
                  last observation on or before that date)

Normalization (why and how)
---------------------------
Levels (~20 for VIX), percentage returns (~0.02) and yield changes (~0.1) live on
different scales, so they cannot share one y-axis. Every series is therefore
z-scored over the plotted window:

    z_t = (x_t - mean(x)) / std(x)        (sample std, ddof=1)

Each line is centred at 0 and measured in standard deviations of its own history,
so shapes are directly comparable. The raw value is shown on hover, so nothing is
lost. A constant series (std = 0) is drawn flat at 0 and reported in the notes.
Rebasing to 100 was rejected (meaningless for returns/changes that cross zero) and
min-max scaling was rejected (one outlier distorts the whole line).

Trending variables (e.g. the S&P 500 level has no long-run mean, so a z-score is
meaningless) can opt out with `normalize: false` in the YAML. They are drawn from
their raw values on a secondary right-hand y-axis, next to the z-scored lines on
the left axis. If every variable opts out, they all share the single left axis.

One variable may set `main_curve: true` to be highlighted with a bold line, drawn
on top of the others. Setting it on more than one variable is an error.

Data rules
----------
* Adjusted close prices from yfinance (daily).
* Inner join on dates: only dates on which every ticker has a price are kept.
* Extra history is fetched before the start date so the series begins on the start
  date; it begins later only if a ticker's history (or a window) is too short, and
  the chart footer says so.
* A partial bar (today, before the US close) is dropped; the footer says so.

Usage:
    python market_plot_comparison.py <start_date> [end_date] [--config path.yaml]
"""

import argparse
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yaml
from plotly.subplots import make_subplots
import yfinance as yf

CONFIG_PATH = Path(__file__).parent.parent / "config" / "market_plot_comparison.yaml"

MAX_VARIABLES = 16
MAIN_CURVE_WIDTH = 4.5  # plotly line width in px; ordinary lines use the default (2)
VALID_TYPES = ("level", "change", "return")

# window -> (mode, n): "rows" = trading rows, "months" = calendar months
WINDOWS = {
    "1d": ("rows", 1),
    "5d": ("rows", 5),
    "10d": ("rows", 10),
    "1m": ("months", 1),
    "3m": ("months", 3),
}

# Longest lookback is 3 calendar months (~92 days); the buffer adds slack for
# weekends/holidays so the first plotted day has a full window.
FETCH_BUFFER_DAYS = 110

# Daily bars are treated as complete after this time (ET) on a weekday.
# 16:15 rather than 16:00 so that ^VIX (closes 16:15) is also complete.
MARKET_TZ = ZoneInfo("America/New_York")
MARKET_CLOSE_HOUR, MARKET_CLOSE_MINUTE = 16, 15


@dataclass(frozen=True)
class Variable:
    """One plotted variable: a ticker plus how to transform its price series."""

    ticker: str
    kind: str  # "level" | "change" | "return"
    window: str | None  # None for level
    normalize: bool = True  # False -> raw values on the right-hand axis
    main_curve: bool = False  # True -> drawn bold and on top (at most one)

    @property
    def label(self) -> str:
        """Human-readable name used in the legend, e.g. 'NVDA 1M return' or '^GSPC level (raw)'."""
        base = f"{self.ticker} level" if self.kind == "level" else f"{self.ticker} {self.window.upper()} {self.kind}"
        return base if self.normalize else f"{base} (raw)"


def load_config(path: Path) -> list[Variable]:
    """
    Load and validate the variable definitions from a YAML file.

    Args:
        path: YAML file with a top-level 'variables' list; each entry has
              'ticker', optional 'type' (default 'level'), 'window'
              (required for change/return, not allowed for level) and
              optional 'normalize' (default true; false = raw values on the
              right-hand axis) and optional 'main_curve' (default false;
              true = bold line, allowed on at most one variable).

    Returns:
        List of Variable, in file order.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: on any invalid entry or when the count is outside 1..16.
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    entries = data.get("variables")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: 'variables' must be a non-empty list.")
    if len(entries) > MAX_VARIABLES:
        raise ValueError(f"{path}: at most {MAX_VARIABLES} variables allowed, got {len(entries)}.")

    variables: list[Variable] = []
    for i, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict) or not entry.get("ticker"):
            raise ValueError(f"{path}: variable #{i} needs a 'ticker'.")
        ticker = str(entry["ticker"]).strip().upper()
        kind = str(entry.get("type", "level")).strip().lower()
        window = entry.get("window")
        window = str(window).strip().lower() if window is not None else None
        normalize = entry.get("normalize", True)
        if not isinstance(normalize, bool):
            raise ValueError(f"{ticker}: 'normalize' must be true or false, got {normalize!r}.")
        main_curve = entry.get("main_curve", False)
        if not isinstance(main_curve, bool):
            raise ValueError(f"{ticker}: 'main_curve' must be true or false, got {main_curve!r}.")

        if kind not in VALID_TYPES:
            raise ValueError(f"{ticker}: type '{kind}' invalid; choose from {VALID_TYPES}.")
        if kind == "level" and window is not None:
            raise ValueError(f"{ticker}: 'window' is not allowed for type 'level'.")
        if kind != "level" and window not in WINDOWS:
            raise ValueError(f"{ticker}: type '{kind}' needs a window from {tuple(WINDOWS)}.")
        variables.append(Variable(ticker, kind, window, normalize, main_curve))

    mains = [v.label for v in variables if v.main_curve]
    if len(mains) > 1:
        raise ValueError(f"Only one variable may set main_curve: true, found {len(mains)}: {', '.join(mains)}")

    labels = [v.label for v in variables]
    duplicates = sorted({lab for lab in labels if labels.count(lab) > 1})
    if duplicates:
        raise ValueError(f"Duplicate variables in config: {', '.join(duplicates)}")
    return variables


def fetch_close_prices(tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """
    Fetch daily adjusted close prices from yfinance, without forward-filling.

    Missing values are deliberately left as NaN so the later inner join reflects
    real trading calendars.

    Args:
        tickers: Unique ticker symbols.
        start:   First date to download (inclusive).
        end:     Last date to download (inclusive).

    Returns:
        DataFrame indexed by date (tz-naive), one column per ticker, in the
        order given.

    Raises:
        ValueError: if the download is empty or any ticker returned no data.
    """
    raw = yf.download(
        tickers,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),  # yfinance end is exclusive
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise ValueError("yfinance returned no data. Check tickers and dates.")

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices.index = prices.index.normalize()

    missing = [t for t in tickers if t not in prices.columns or prices[t].dropna().empty]
    if missing:
        raise ValueError(f"No data returned for: {', '.join(missing)}")
    return prices[tickers].sort_index()


def drop_partial_bar(prices: pd.DataFrame, now: datetime) -> tuple[pd.DataFrame, str | None]:
    """
    Drop today's bar if the US market has not closed yet.

    A daily bar dated today is incomplete on a weekend or before 16:15 ET on a
    weekday, and 'last realized trading day' must exclude it.

    Args:
        prices: Daily price table.
        now:    Current time, timezone-aware in US/Eastern.

    Returns:
        (prices without the partial bar, notice text or None if nothing dropped).
    """
    market_closed = now.weekday() < 5 and (now.hour, now.minute) >= (
        MARKET_CLOSE_HOUR,
        MARKET_CLOSE_MINUTE,
    )
    today = pd.Timestamp(now.date())
    if prices.index[-1] == today and not market_closed:
        return prices.iloc[:-1], (
            f"Excluded: partial bar for {today:%Y-%m-%d} "
            f"(US market not yet closed, {now:%H:%M} ET)."
        )
    return prices, None


def build_variable_series(prices: pd.DataFrame, var: Variable) -> pd.Series:
    """
    Compute the raw (un-normalized) series for one variable.

    Args:
        prices: Inner-joined price table containing var.ticker.
        var:    Variable definition (ticker, type, window).

    Returns:
        Series named var.label. Leading values are NaN where the window is not
        yet filled.
    """
    px = prices[var.ticker]
    if var.kind == "level":
        series = px
    else:
        mode, n = WINDOWS[var.window]
        if mode == "rows":
            base = px.shift(n)
        else:
            # value n calendar months earlier = last observation on or before it
            refs = px.index - pd.DateOffset(months=n)
            base = pd.Series(px.reindex(refs, method="ffill").to_numpy(), index=px.index)
        series = px - base if var.kind == "change" else px / base - 1.0
    return series.rename(var.label)


def zscore(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Z-score each column over the whole frame: (x - mean) / std, sample std.

    Normalization rationale: see the module docstring. Constant columns
    (std = 0 or undefined) become 0 and are reported.

    Args:
        frame: Raw series, one column per variable.

    Returns:
        (z-scored frame, names of constant columns drawn flat at 0).
    """
    std = frame.std(ddof=1)
    flat = [c for c in frame.columns if not np.isfinite(std[c]) or std[c] == 0]
    z = (frame - frame.mean()) / std.replace(0, np.nan)
    z[flat] = 0.0
    return z, flat


def compute_window(
    prices: pd.DataFrame, variables: list[Variable], start: date, end: date
) -> tuple[pd.DataFrame, list[str]]:
    """
    Inner-join prices, compute every variable, and trim to [start, end].

    Args:
        prices:    Daily prices incl. lookback history before start (partial bar
                   already removed).
        variables: Variables to compute.
        start:     First date to plot (inclusive).
        end:       Last date to plot (inclusive).

    Returns:
        (raw values for the plotted window, notes describing excluded data).

    Raises:
        ValueError: if no common dates remain.
    """
    notes: list[str] = []
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)

    joined = prices.dropna(how="any")
    if joined.empty:
        raise ValueError("No dates on which all tickers have prices.")

    in_range = lambda idx: (idx >= start_ts) & (idx <= end_ts)  # noqa: E731
    any_price = prices.dropna(how="all").index
    n_dropped = int(in_range(any_price).sum() - in_range(joined.index).sum())
    if n_dropped > 0:
        notes.append(f"Excluded: {n_dropped} date(s) where not every ticker traded (inner join).")

    frame = pd.concat([build_variable_series(joined, v) for v in variables], axis=1)
    first_valid = frame.apply(pd.Series.first_valid_index)
    frame = frame.dropna(how="any")
    frame = frame.loc[in_range(frame.index)]
    if frame.empty:
        raise ValueError("No data left in the requested range after the inner join and windows.")

    expected_start = joined.index[joined.index >= start_ts]
    if len(expected_start) and frame.index[0] > expected_start[0]:
        limiter = first_valid.idxmax()
        notes.append(
            f"Series starts {frame.index[0]:%Y-%m-%d}, not {expected_start[0]:%Y-%m-%d}: "
            f"limited by {limiter} (shortest history or lookback window not yet filled)."
        )
    return frame, notes


def format_raw(values: pd.Series, kind: str) -> list[str]:
    """
    Format raw values for hover text.

    Args:
        values: Raw series.
        kind:   'level', 'change' or 'return'.

    Returns:
        List of strings: returns as '1.23%', changes as '+1.23', levels as '12.34'.
    """
    if kind == "return":
        return [f"{v * 100:.2f}%" for v in values]
    if kind == "change":
        return [f"{v:+.2f}" for v in values]
    return [f"{v:.2f}" for v in values]


def build_figure(
    z: pd.DataFrame, raw: pd.DataFrame, variables: list[Variable], notes: list[str]
) -> go.Figure:
    """
    Build the interactive plotly chart with title, legend, axis labels and notes.

    Normalized variables are drawn from z on the left axis. Variables with
    normalize=False are drawn from raw on a secondary right-hand axis; if no
    variable is normalized they use the single left axis instead.

    Args:
        z:         Z-scored series (columns for normalized variables only).
        raw:       Raw series for all variables (shown on hover; plotted for
                   normalize=False variables).
        variables: Variables in plotting order.
        notes:     Lines describing excluded data, printed under the chart.

    Returns:
        A plotly Figure.
    """
    raw_vars = [v for v in variables if not v.normalize]
    use_right = 0 < len(raw_vars) < len(variables)
    fig = make_subplots(specs=[[{"secondary_y": use_right}]])
    # draw the main curve last so it sits on top; legendrank keeps the config order
    for var in sorted(variables, key=lambda v: v.main_curve):
        label = var.label
        if var.normalize:
            y, hover = z[label], "z = %{y:.2f}<br>raw = %{customdata}"
        else:
            y, hover = raw[label], "raw = %{customdata}"
        fig.add_trace(
            go.Scatter(
                x=raw.index,
                y=y,
                mode="lines",
                name=label,
                line={"width": MAIN_CURVE_WIDTH} if var.main_curve else {},
                legendrank=variables.index(var) + 1,
                customdata=format_raw(raw[label], var.kind),
                hovertemplate=hover + "<extra>" + label + "</extra>",
            ),
            **({"secondary_y": not var.normalize} if use_right else {}),
        )

    footer = notes if notes else ["Excluded: nothing beyond the inner join (no partial bar, no gaps)."]
    fig.update_layout(
        title=(
            "Market comparison (z-scored)"
            f"<br><sup>{z.index[0]:%Y-%m-%d} to {z.index[-1]:%Y-%m-%d} | "
            f"{len(z)} common trading days</sup>"
        ),
        xaxis_title="Date",
        legend_title_text="Variable",
        hovermode="x unified",
        margin={"b": 100 + 18 * len(footer)},
        annotations=[
            {
                "text": "<br>".join(footer),
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.14,
                "xanchor": "left",
                "yanchor": "top",
                "showarrow": False,
                "align": "left",
            }
        ],
    )

    z_title = "Z-score (std devs from each series' mean over the plotted window)"
    raw_title = "Raw value (not normalized)"
    # raw returns are fractions (0.02); show them as percentages on the axis
    raw_axis = {"tickformat": ".1%"} if raw_vars and all(v.kind == "return" for v in raw_vars) else {}
    if use_right:
        fig.update_yaxes(title_text=z_title, secondary_y=False)
        fig.update_yaxes(title_text=raw_title, secondary_y=True, showgrid=False, **raw_axis)
    elif raw_vars:  # every variable opted out of normalization
        fig.update_yaxes(title_text=raw_title, **raw_axis)
    else:
        fig.update_yaxes(title_text=z_title)
    return fig


def run(config_path: Path, start: date, end: date | None) -> None:
    """
    Run the full pipeline and open the interactive chart in the browser.

    Args:
        config_path: YAML file listing the variables.
        start:       First plotted date.
        end:         Last plotted date, or None for the last realized trading day.
    """
    variables = load_config(config_path)
    tickers = list(dict.fromkeys(v.ticker for v in variables))

    now = datetime.now(MARKET_TZ)
    end_bound = end or now.date()
    if end_bound < start:
        raise ValueError(f"end_date ({end_bound}) must not be before start_date ({start}).")

    print(f"Tickers : {', '.join(tickers)}")
    print(f"Period  : {start} -> {end_bound}{' (default: last realized trading day)' if end is None else ''}")
    print("Fetching daily adjusted close ...")
    prices = fetch_close_prices(tickers, start - timedelta(days=FETCH_BUFFER_DAYS), end_bound)

    prices, partial_note = drop_partial_bar(prices, now)
    raw, notes = compute_window(prices, variables, start, end_bound)
    if partial_note:
        notes.insert(0, partial_note)

    norm_labels = [v.label for v in variables if v.normalize]
    z, flat = zscore(raw[norm_labels]) if norm_labels else (raw[[]], [])
    if flat:
        notes.append(f"Constant series drawn flat at 0: {', '.join(flat)}.")

    for line in notes:
        print(line)
    build_figure(z, raw, variables, notes).show()


def parse_date(text: str, name: str) -> date:
    """
    Parse a yyyy-mm-dd string.

    Args:
        text: The date string.
        name: Argument name, used in the error message.

    Returns:
        The parsed date.

    Raises:
        ValueError: if the string is not a valid yyyy-mm-dd date.
    """
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError:
        raise ValueError(f"Invalid {name} '{text}'. Expected yyyy-mm-dd.") from None


def main(argv: list[str] | None = None) -> None:
    """
    Parse command-line arguments and run the tool.

    Usage:
        python market_plot_comparison.py <start_date> [end_date] [--config path.yaml]

    Args:
        argv: Argument list (defaults to sys.argv[1:]).
    """
    parser = argparse.ArgumentParser(description="Compare market variables on one z-scored chart.")
    parser.add_argument("start_date", help="first plotted date, yyyy-mm-dd")
    parser.add_argument("end_date", nargs="?", help="last plotted date, yyyy-mm-dd (default: last realized trading day)")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help=f"variables YAML (default: {CONFIG_PATH})")
    args = parser.parse_args(argv)

    try:
        start = parse_date(args.start_date, "start_date")
        end = parse_date(args.end_date, "end_date") if args.end_date else None
        run(args.config, start, end)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
