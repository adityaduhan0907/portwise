#!/usr/bin/env python3
"""
stress_test.py  —  Historical crisis stress test (post-Layer-5 reporting)

Replays the CHOSEN (resampled) portfolio through named historical crisis windows
and records how it WOULD have performed, using the ACTUAL historical monthly
returns of the held stocks over each window. This is a backward-looking, real-data
replay -- NOT the forward Monte-Carlo drawdown in risk_evaluation.py.

Crisis windows (inclusive, month granularity):
    2008 GFC      Oct 2007 - Mar 2009
    COVID crash   Feb 2020 - Mar 2020
    2022 drawdown Jan 2022 - Oct 2022
    dot-com       Mar 2000 - Oct 2002

For each window, per held stock we fetch monthly adjusted closes over the window
(reusing fetch_util -- the same fetcher the pipeline uses) and compute monthly
returns. Then:
    cumulative return = prod(1 + r_month) - 1   over the covered months
    max drawdown      = worst peak-to-trough of the compounded wealth curve

COVERAGE IS HANDLED HONESTLY (the whole point of this report):
  * A stock with NO data for a window (listed after it, or the provider has no
    history that far back) is NEVER silently zeroed or back-filled. It is listed
    as MISSING for that window, with the reason.
  * If some-but-not-all holdings are covered, the portfolio is computed on the
    AVAILABLE SUBSET (weights renormalised across the covered names) and the result
    carries a clear "based on N of M holdings (covered weight X%)" note.
  * If NO holding is covered (or no common months exist), the window is marked
    "insufficient_data" -- no number is fabricated.

Output: stress_test.json keyed by crisis window name. Non-interactive, file-based.
run_all.py wires this in after Layer 5; Module 5 / the UI consume it next pass.
"""

import json
import os

import numpy as np
import pandas as pd

import fetch_util as fu

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
RESAMPLED_PATH = os.path.join(SCRIPT_DIR, "resampled_portfolios.xlsx")
OPTIMISED_PATH = os.path.join(SCRIPT_DIR, "optimised_portfolios.xlsx")
CONFIG_PATH    = os.path.join(SCRIPT_DIR, "run_config.json")
OUTPUT_PATH    = os.path.join(SCRIPT_DIR, "stress_test.json")

GOAL_SHEETS = ["Minimum Variance", "Max Risk-Adjusted", "Tail-Risk CVaR"]

# Crisis windows: (label, start 'YYYY-MM-DD', end 'YYYY-MM-DD'). Months inclusive.
CRISIS_WINDOWS = [
    ("2008 GFC",      "2007-10-01", "2009-03-31"),
    ("COVID crash",   "2020-02-01", "2020-03-31"),
    ("2022 drawdown", "2022-01-01", "2022-10-31"),
    ("dot-com",       "2000-03-01", "2002-10-31"),
]

# A stock "covers" a window only if its first monthly observation lands within this
# many days of the window start. Later first-data => listed after the window opened
# => we will NOT pretend to know its crisis return.
COVERAGE_TOLERANCE_DAYS = 50


# ── Portfolio loading ────────────────────────────────────────────────────────────

def _load_portfolio_choice(path=CONFIG_PATH):
    """Chosen goal SHEET name from run_config.json, if present."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("portfolio_choice")
    except Exception:
        return None


def load_chosen_weights(sheet=None, resampled_path=RESAMPLED_PATH,
                        optimised_path=OPTIMISED_PATH):
    """
    Return (sheet_name, {ticker: weight}) for the chosen (resampled) portfolio.

    Prefers the chosen sheet in resampled_portfolios.xlsx (canonical Layer-7 output).
    Falls back: chosen sheet in optimised_portfolios.xlsx, then the first available
    goal sheet. Reads only Stock/Weight(%) rows (skips spacer/summary rows).
    """
    sheet = sheet or _load_portfolio_choice()

    def _read(path, sh):
        if not os.path.exists(path):
            return None
        try:
            xl = pd.ExcelFile(path)
        except Exception:
            return None
        if sh not in xl.sheet_names:
            return None
        df = pd.read_excel(path, sheet_name=sh)
        wmap = {}
        for _, row in df.iterrows():
            stock = row.get("Stock")
            wt    = row.get("Weight (%)")
            if isinstance(stock, str) and stock and pd.notna(wt):
                if "(%)" in stock or stock.strip() == "":
                    continue            # summary / spacer row
                try:
                    wmap[stock] = float(wt) / 100.0
                except (TypeError, ValueError):
                    continue
        return wmap or None

    candidates = []
    if sheet:
        candidates += [(resampled_path, sheet), (optimised_path, sheet)]
    for sh in GOAL_SHEETS:
        candidates += [(resampled_path, sh), (optimised_path, sh)]

    for path, sh in candidates:
        wmap = _read(path, sh)
        if wmap:
            total = sum(wmap.values())
            if total > 0:
                wmap = {t: w / total for t, w in wmap.items()}
            return sh, wmap
    return None, {}


# ── Per-stock historical monthly returns over a window ───────────────────────────

def _monthly_returns(ticker, start, end):
    """
    Fetch monthly adjusted closes for one already-resolved ticker over [start, end]
    (a small lead-in buffer is added so the first in-window month gets a return) and
    return (returns_series, status, detail).

    status is one of:
      "ok"            -> returns_series is a non-empty monthly % return series
      "missing"       -> genuinely no data (provider has none / unknown symbol)
      "listed_after"  -> data exists but starts after the window opened
      "transient"     -> provider kept failing transiently (network) -- not a gap
    """
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)
    buf      = (start_ts - pd.DateOffset(months=2)).strftime("%Y-%m-%d")

    try:
        symbol, series, reason = fu.resolve_and_fetch(
            ticker, start=buf, end=end_ts.strftime("%Y-%m-%d"),
            interval="1mo", prefer=ticker, what=f"{ticker} monthly history",
        )
    except fu.TransientFetchError as exc:
        return None, "transient", str(getattr(exc, "detail", exc))

    if series is None or series.empty:
        return None, "missing", reason or "no data returned"

    series = series.sort_index()
    monthly = series.resample("ME").last().dropna()
    # Returns, then restrict to the window itself.
    rets = monthly.pct_change().dropna()
    rets = rets[(rets.index >= start_ts) & (rets.index <= end_ts)]
    if rets.empty:
        return None, "missing", "no monthly returns inside the window"

    first_obs = monthly.index[0]
    if first_obs > start_ts + pd.Timedelta(days=COVERAGE_TOLERANCE_DAYS):
        return None, "listed_after", f"first data {first_obs.date()}"

    return rets, "ok", f"{len(rets)} months, first {monthly.index[0].date()}"


# ── One window ───────────────────────────────────────────────────────────────────

def _portfolio_path_metrics(returns_df, weights):
    """
    Portfolio monthly returns over the COMMON months of `returns_df` (cols = covered
    tickers) under renormalised `weights`; return (cumulative_return, max_drawdown,
    n_months). Drawdown is a positive magnitude (0.30 == -30%).
    """
    aligned = returns_df.dropna(how="any")          # common months across covered names
    if aligned.empty:
        return None, None, 0
    w = np.array([weights[t] for t in aligned.columns], dtype=float)
    w = w / w.sum()
    port = aligned.values @ w                        # monthly portfolio returns
    wealth = np.cumprod(1.0 + port)
    cum_return = float(wealth[-1] - 1.0)
    peak = np.maximum.accumulate(wealth)
    max_dd = float((wealth / peak - 1.0).min())      # <= 0
    return cum_return, -max_dd, int(aligned.shape[0])


def stress_one_window(label, start, end, weights, verbose=True):
    """Replay one crisis window; return a result dict (never fabricates a return)."""
    holdings = [t for t, w in weights.items() if w > 1e-9]
    n_total  = len(holdings)

    covered, missing, transient = {}, [], []
    ret_cols = {}
    for t in holdings:
        rets, status, detail = _monthly_returns(t, start, end)
        if status == "ok":
            covered[t] = weights[t]
            ret_cols[t] = rets
        elif status == "transient":
            transient.append({"ticker": t, "detail": detail})
        else:  # missing / listed_after
            missing.append({"ticker": t, "reason": status, "detail": detail})
        if verbose:
            print(f"      {t:<14} {status:<13} {detail}")

    covered_weight = float(sum(covered.values()))
    n_cov = len(covered)

    base = {
        "window":          f"{start} .. {end}",
        "n_holdings":      n_total,
        "n_covered":       n_cov,
        "covered_weight":  round(covered_weight, 6),
        "missing":         missing,
        "transient":       transient,
    }

    if n_cov == 0:
        base.update({
            "status":          "insufficient_data",
            "cumulative_return": None,
            "max_drawdown":      None,
            "note": (f"INSUFFICIENT DATA: none of the {n_total} holdings have history "
                     f"in this window (all listed after it / no provider data). "
                     f"No crisis return computed."),
        })
        return base

    returns_df = pd.DataFrame(ret_cols)
    cum, dd, n_months = _portfolio_path_metrics(returns_df, covered)
    if cum is None:
        base.update({
            "status":          "insufficient_data",
            "cumulative_return": None,
            "max_drawdown":      None,
            "note": (f"INSUFFICIENT DATA: the {n_cov} covered holdings share no common "
                     f"month in this window. No crisis return computed."),
        })
        return base

    full = (n_cov == n_total)
    note = (f"Based on {n_cov} of {n_total} holdings "
            f"(covered weight {covered_weight*100:.1f}% of portfolio, "
            f"renormalised across covered names; {n_months} common months).")
    if not full:
        miss_names = ", ".join(m["ticker"] for m in missing)
        note += f" MISSING: {miss_names} -- not back-filled."
    base.update({
        "status":            "ok" if full else "partial_coverage",
        "cumulative_return": round(cum, 6),
        "max_drawdown":      round(dd, 6),
        "months_used":       n_months,
        "note":              note,
    })
    return base


# ── Public entry point ───────────────────────────────────────────────────────────

def run_stress_test(weights=None, sheet=None, windows=CRISIS_WINDOWS,
                    output_path=OUTPUT_PATH, verbose=True):
    """
    Replay the chosen portfolio through all crisis windows and write stress_test.json.

    weights : {ticker: weight} dict. If None, loaded from the chosen (resampled)
              portfolio via load_chosen_weights(sheet).
    Returns the full result dict (also written to `output_path`).
    """
    if weights is None:
        sheet, weights = load_chosen_weights(sheet)
    if not weights:
        payload = {"status": "no_portfolio",
                   "message": "Could not load a chosen portfolio to stress-test.",
                   "windows": {}}
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        if verbose:
            print("  Stress test: no portfolio weights available -- nothing to do.")
        return payload

    holdings = {t: w for t, w in weights.items() if w > 1e-9}
    if verbose:
        W = 74
        print(f"\n{'='*W}")
        print("  HISTORICAL CRISIS STRESS TEST")
        print(f"  Portfolio: {sheet or 'chosen'}   |   Holdings: "
              f"{', '.join(f'{t} {w*100:.1f}%' for t, w in holdings.items())}")
        print(f"{'='*W}")

    windows_out = {}
    for label, start, end in windows:
        if verbose:
            print(f"\n  [{label}]  {start} .. {end}")
        windows_out[label] = stress_one_window(label, start, end, holdings, verbose=verbose)
        r = windows_out[label]
        if verbose:
            if r["cumulative_return"] is None:
                print(f"    -> {r['status'].upper()}: {r['note']}")
            else:
                print(f"    -> cumulative {r['cumulative_return']*100:+.2f}%   "
                      f"max drawdown -{r['max_drawdown']*100:.2f}%   [{r['status']}]")
                print(f"       {r['note']}")

    payload = {
        "status":           "ok",
        "portfolio":        sheet or "chosen",
        "holdings":         {t: round(w, 6) for t, w in holdings.items()},
        "coverage_rule":    (f"a holding covers a window only if its first monthly "
                             f"observation is within {COVERAGE_TOLERANCE_DAYS} days of "
                             f"the window start; otherwise reported missing, never "
                             f"back-filled"),
        "windows":          windows_out,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    if verbose:
        print(f"\n  Wrote -> {output_path}")
    return payload


def main():
    run_stress_test()


if __name__ == "__main__":
    main()
