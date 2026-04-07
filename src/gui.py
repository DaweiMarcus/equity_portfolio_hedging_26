"""Desktop GUI entry point: portfolio-hedge-gui"""

import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

# Portfolio CSVs are always read from this directory
DATA_DIR = Path(__file__).parent / "data"

from src.analytics.beta import compute_portfolio_beta, compute_rolling_beta
from src.data.fetcher import fetch_weekly_returns
from src.data.loader import load_portfolio


class App(tk.Tk):
    def __init__(self) -> None:
        """Initialize the main application window and build the UI."""
        super().__init__()
        self.title("Equity Portfolio Hedge Analyzer")
        self.geometry("760x520")
        self.resizable(True, True)
        self._build_ui()

    def _build_ui(self) -> None:
        """Create and lay out all widgets: file picker, run button, and log area."""
        # --- File picker row ---
        top = ttk.Frame(self, padding=10)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Portfolio CSV:").pack(side=tk.LEFT)
        self.csv_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.csv_var, width=52).pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Browse…", command=self._browse).pack(side=tk.LEFT)

        # --- Action row ---
        action = ttk.Frame(self, padding=(10, 0, 10, 6))
        action.pack(fill=tk.X)
        self.run_btn = ttk.Button(action, text="Run Analysis", command=self._run)
        self.run_btn.pack(side=tk.LEFT)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(action, textvariable=self.status_var, foreground="gray").pack(side=tk.LEFT, padx=10)

        # --- Log area ---
        log_frame = ttk.LabelFrame(self, text="Output", padding=8)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.log = scrolledtext.ScrolledText(
            log_frame, state=tk.DISABLED, font=("Courier", 11), wrap=tk.NONE
        )
        self.log.pack(fill=tk.BOTH, expand=True)

    def _browse(self) -> None:
        """Open a file dialog rooted at src/data/ and populate the CSV path entry."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        path = filedialog.askopenfilename(
            initialdir=str(DATA_DIR),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if path:
            self.csv_var.set(path)

    def _log(self, msg: str) -> None:
        """Append a line to the output log widget (thread-safe via after())."""
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, msg + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _run(self) -> None:
        """Validate input, clear the log, and launch the analysis worker thread."""
        csv_path = self.csv_var.get().strip()
        if not csv_path:
            messagebox.showwarning("No file selected", "Please select a portfolio CSV file.")
            return
        self.run_btn.configure(state=tk.DISABLED)
        self.status_var.set("Running…")
        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.configure(state=tk.DISABLED)
        threading.Thread(target=self._worker, args=(csv_path,), daemon=True).start()

    def _worker(self, csv_path: str) -> None:
        """
        Background thread: run the full analysis pipeline and stream output to the log.

        Args:
            csv_path: Path to the portfolio CSV selected by the user.
        """
        try:
            output_dir = Path("analysis_results")
            output_dir.mkdir(exist_ok=True)

            self.after(0, self._log, f"Loading portfolio: {csv_path}")
            portfolio = load_portfolio(csv_path)
            tickers = portfolio["Symbol"].tolist()
            self.after(0, self._log, f"  Tickers: {', '.join(tickers)}")

            self.after(0, self._log, "Fetching 2 years of weekly returns (+ SPY)…")
            returns, latest_prices = fetch_weekly_returns(tickers)
            self.after(0, self._log, f"  {len(returns)} weekly observations loaded.")

            self.after(0, self._log, "Computing 52-week rolling OLS beta…")
            rolling_betas = compute_rolling_beta(returns)

            self.after(0, self._log, "Computing notional-weighted portfolio beta…")
            portfolio_beta, detail_df = compute_portfolio_beta(
                portfolio, rolling_betas, latest_prices
            )

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            rolling_path = output_dir / f"rolling_beta_{ts}.csv"
            detail_path = output_dir / f"portfolio_beta_{ts}.csv"
            rolling_betas.to_csv(rolling_path)
            detail_df.to_csv(detail_path, index=False)

            table = detail_df[
                ["Symbol", "Notional", "Leverage", "Rolling_Beta", "Effective_Beta", "Weight"]
            ].to_string(index=False)
            self.after(0, self._log, "\n" + table)
            self.after(0, self._log, "\n" + "=" * 52)
            self.after(0, self._log, f"  Notional-Weighted Portfolio Beta:  {portfolio_beta:.4f}")
            self.after(0, self._log, "=" * 52)
            self.after(0, self._log, f"\nSaved: {rolling_path}")
            self.after(0, self._log, f"Saved: {detail_path}")
            self.after(0, self.status_var.set, "Done")
        except Exception as exc:
            self.after(0, self._log, f"\nERROR: {exc}")
            self.after(0, self.status_var.set, "Error")
        finally:
            self.after(0, self.run_btn.configure, {"state": tk.NORMAL})


def main() -> None:
    """Instantiate and run the desktop GUI application."""
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
