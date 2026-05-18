#!/usr/bin/env python3
"""
module2_optimiser.py

Reads prices.xlsx and returns_stats.xlsx produced by module1_data.py,
then runs mean-variance optimisation to find three portfolios:
  1. Maximum Sharpe Ratio
  2. Minimum Volatility
  3. Maximum Return

Constraints (retail investor):
  - No short selling  (weights >= 0)
  - Full investment   (sum of weights == 1)
  - Max 40% per stock (weights <= 0.40)

Exports results to optimised_portfolios.xlsx.
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
TRADING_DAYS = 252
MAX_WEIGHT   = 0.40     # 40 % single-stock cap
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))


def _load_risk_free_rate(fallback=0.043):
    """Load blended risk-free rate from risk_free_rates.json, or fall back to default."""
    json_path = os.path.join(SCRIPT_DIR, "risk_free_rates.json")
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        rate       = float(data["blended_rate"])
        fetched_at = data.get("fetched_at", "unknown")
        print(f"  Risk-free rate : {rate*100:.4f}%  "
              f"(blended, risk_free_rates.json — fetched {fetched_at})")
        return rate
    except FileNotFoundError:
        print(f"  WARNING: risk_free_rates.json not found — using fallback {fallback*100:.1f}%.")
        print("           Run module0_riskfree.py first for fresh rates.")
        return fallback
    except Exception as exc:
        print(f"  WARNING: Could not read risk_free_rates.json ({exc}). "
              f"Using fallback {fallback*100:.1f}%.")
        return fallback


RISK_FREE_RATE = _load_risk_free_rate()

# ── Helper: portfolio statistics ───────────────────────────────────────────────

def portfolio_stats(weights, mu, cov):
    """Return (annualised return, annualised volatility, Sharpe ratio)."""
    ret = float(weights @ mu)
    vol = float(np.sqrt(weights @ cov @ weights))
    sharpe = (ret - RISK_FREE_RATE) / vol if vol > 0 else float("nan")
    return ret, vol, sharpe


# ── Objective functions (scipy minimises, so we negate to maximise) ────────────

def neg_sharpe(weights, mu, cov):
    _, vol, sharpe = portfolio_stats(weights, mu, cov)
    return -sharpe if vol > 0 else 1e9


def portfolio_vol(weights, mu, cov):
    return float(np.sqrt(weights @ cov @ weights))


def neg_return(weights, mu, cov):
    return -float(weights @ mu)


# ── Core optimiser ─────────────────────────────────────────────────────────────

def optimise(objective, mu, cov, label):
    """
    Run SLSQP minimisation.
    Returns (weights_array, success_bool, message_str).
    """
    n = len(mu)
    bounds      = [(0.0, MAX_WEIGHT)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Try several random starting points; keep best result
    best_result = None
    rng = np.random.default_rng(seed=42)

    for attempt in range(50):
        # Random Dirichlet start, then clip to MAX_WEIGHT and renormalise
        w0 = rng.dirichlet(np.ones(n))
        w0 = np.clip(w0, 0, MAX_WEIGHT)
        w0 /= w0.sum()

        res = minimize(
            objective,
            w0,
            args=(mu, cov),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 2000},
        )

        if res.success or res.status == 0:
            if best_result is None or res.fun < best_result.fun:
                best_result = res

    if best_result is None or not (best_result.success or best_result.status == 0):
        return None, False, f"Optimisation did not converge for '{label}'"

    weights = np.clip(best_result.x, 0, None)
    weights /= weights.sum()   # re-normalise after clipping tiny negatives
    return weights, True, "OK"


# ── Display & export helpers ───────────────────────────────────────────────────

def build_result_df(tickers, weights, mu, cov):
    """Return a DataFrame summarising portfolio weights and aggregate stats."""
    ret, vol, sharpe = portfolio_stats(weights, mu, cov)

    rows = []
    for ticker, w in zip(tickers, weights):
        rows.append({"Stock": ticker, "Weight (%)": round(w * 100, 2)})

    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"Stock": "", "Weight (%)": ""}           # blank spacer
    df.loc[len(df)] = {"Stock": "Portfolio Return (%)",  "Weight (%)": round(ret * 100, 2)}
    df.loc[len(df)] = {"Stock": "Portfolio Volatility (%)", "Weight (%)": round(vol * 100, 2)}
    df.loc[len(df)] = {"Stock": "Portfolio Sharpe Ratio",   "Weight (%)": round(sharpe, 4)}
    return df, ret, vol, sharpe


def print_portfolio(label, tickers, weights, mu, cov):
    ret, vol, sharpe = portfolio_stats(weights, mu, cov)
    width = 62
    print(f"\n{'='*width}")
    print(f"  {label}")
    print(f"{'='*width}")
    print(f"  {'Stock':<20}  {'Weight':>10}")
    print(f"  {'-'*20}  {'-'*10}")
    for ticker, w in zip(tickers, weights):
        if w >= 0.0001:   # hide dust-level allocations
            print(f"  {ticker:<20}  {w*100:>9.2f}%")
    print(f"  {'-'*20}  {'-'*10}")
    print(f"\n  Expected Annual Return : {ret*100:>8.2f}%")
    print(f"  Annual Volatility      : {vol*100:>8.2f}%")
    print(f"  Sharpe Ratio           : {sharpe:>8.4f}")
    print(f"{'='*width}")


def export_to_excel(portfolios, out_path):
    """
    portfolios: list of (sheet_name, result_df) tuples.
    Writes each to its own sheet in out_path.
    """
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for sheet_name, df in portfolios:
            # Keep only weight rows (exclude spacer and summary rows) for
            # the stock block, then append a stats block below
            df.to_excel(writer, sheet_name=sheet_name, index=False)

            ws = writer.sheets[sheet_name]
            # Bold the header row
            from openpyxl.styles import Font, PatternFill, Alignment
            header_fill = PatternFill("solid", fgColor="1F4E79")
            for cell in ws[1]:
                cell.font      = Font(bold=True, color="FFFFFF")
                cell.fill      = header_fill
                cell.alignment = Alignment(horizontal="center")

            # Right-align the Weight column; highlight summary rows
            summary_fill = PatternFill("solid", fgColor="D9E1F2")
            for row in ws.iter_rows(min_row=2):
                label_cell  = row[0]
                weight_cell = row[1]
                weight_cell.alignment = Alignment(horizontal="right")
                if label_cell.value in (
                    "Portfolio Return (%)",
                    "Portfolio Volatility (%)",
                    "Portfolio Sharpe Ratio",
                ):
                    for cell in row:
                        cell.font = Font(bold=True)
                        cell.fill = summary_fill

            ws.column_dimensions["A"].width = 28
            ws.column_dimensions["B"].width = 16


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*62}")
    print("  Module 2 — Mean-Variance Portfolio Optimiser")
    print(f"{'='*62}\n")

    # ── 1. Load pre-computed mu and covariance from module1 ───────────────────
    returns_path = os.path.join(SCRIPT_DIR, "returns_stats.xlsx")
    if not os.path.exists(returns_path):
        print(f"  ERROR: '{returns_path}' not found.")
        print("  Run module1_data.py first to generate the required files.")
        sys.exit(1)

    mu  = None
    cov = None
    tickers = None

    # Primary path: read pre-computed split-estimation mu and cov
    try:
        mu_df  = pd.read_excel(returns_path, sheet_name="Annualised Mu")
        cov_df = pd.read_excel(returns_path, sheet_name="Annualised Cov", index_col=0)
        tickers = list(mu_df["Ticker"])
        mu      = mu_df["Annualised_Expected_Return"].values.astype(float)
        cov     = cov_df.loc[tickers, tickers].values.astype(float)
        print(f"  Loaded split-estimation parameters for {len(tickers)} stocks")
        print( "  Return source : momentum 12M-1M  (Annualised Mu sheet)")
        print( "  Risk source   : 10-year monthly  (Annualised Cov sheet)")
    except Exception:
        pass

    # Fallback: recompute from daily returns if new sheets are absent
    if mu is None:
        print("  WARNING: 'Annualised Mu'/'Annualised Cov' sheets not found.")
        print("           Re-run module1_data.py to generate split-estimation parameters.")
        print("           Falling back to daily-returns-based estimation.\n")
        try:
            returns_df = pd.read_excel(
                returns_path, sheet_name="Daily Returns", index_col=0, parse_dates=True
            )
        except Exception as exc:
            print(f"  ERROR reading returns_stats.xlsx: {exc}")
            sys.exit(1)
        returns_df = returns_df.dropna(how="all").dropna(axis=1, how="all")
        if returns_df.empty:
            print("  ERROR: 'Daily Returns' sheet is empty.")
            sys.exit(1)
        tickers = list(returns_df.columns)
        clean   = returns_df.dropna()
        mu      = clean.mean().values * TRADING_DAYS
        cov     = clean.cov().values  * TRADING_DAYS

    n = len(tickers)
    print(f"\n  Annualised expected returns (%):")
    for t, r in zip(tickers, mu):
        print(f"    {t:<22} {r*100:>7.2f}%")

    # ── 3. Feasibility check ───────────────────────────────────────────────────
    # With MAX_WEIGHT = 0.40, we need at least ceil(1/0.40) = 3 stocks.
    min_stocks = int(np.ceil(1.0 / MAX_WEIGHT))
    if n < min_stocks:
        print(f"\n  ERROR: Need at least {min_stocks} stocks to satisfy the "
              f"{MAX_WEIGHT*100:.0f}% cap constraint. Only {n} available.")
        sys.exit(1)

    # ── 4. Run optimisations ───────────────────────────────────────────────────
    configs = [
        ("1 — Maximum Sharpe Ratio", neg_sharpe),
        ("2 — Minimum Volatility",   portfolio_vol),
        ("3 — Maximum Return",       neg_return),
    ]

    sheet_names = [
        "Max Sharpe Ratio",
        "Min Volatility",
        "Max Return",
    ]

    portfolios = []   # (sheet_name, result_df)

    for (label, obj_fn), sheet in zip(configs, sheet_names):
        print(f"\n  Optimising: {label} ...", end=" ", flush=True)
        weights, success, msg = optimise(obj_fn, mu, cov, label)

        if not success:
            print(f"\n  WARNING: {msg}")
            continue

        print("done.")
        print_portfolio(label, tickers, weights, mu, cov)

        result_df, _, _, _ = build_result_df(tickers, weights, mu, cov)
        portfolios.append((sheet, result_df))

    if not portfolios:
        print("\n  All optimisations failed. No output file written.")
        sys.exit(1)

    # ── 5. Export to Excel ─────────────────────────────────────────────────────
    out_path = os.path.join(SCRIPT_DIR, "optimised_portfolios.xlsx")
    try:
        export_to_excel(portfolios, out_path)
        print(f"\n  Exported -> {out_path}")
    except Exception as exc:
        print(f"\n  ERROR writing optimised_portfolios.xlsx: {exc}")
        sys.exit(1)

    print(f"\n{'='*62}\n")


if __name__ == "__main__":
    main()
