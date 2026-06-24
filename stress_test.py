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

COVERAGE IS A PER-STOCK, PER-WINDOW DECISION (the whole point of this report):
  1. ACTUAL    -- the stock has real return history covering the window: use its
                  real historical monthly returns over the window (unchanged).
  2. MODELED   -- the stock has NO history for the window (listed later, or the
                  provider has none that far back): RECONSTRUCT its monthly returns
                  from its factor exposures, using the SAME factors and category
                  betas as the Method-B simulation / expected-return model
                  (US 3-factor Mkt-RF/SMB/HML, India 4-factor MF/SMB/HML/WML):
                      r_modeled(t) = RF(t) + sum_f beta_f * factor_f(t)
                  i.e. the systematic component (the residual is idiosyncratic and
                  unobservable for a window the stock didn't trade, so it is dropped).
                  A reconstructed return is an ESTIMATE, not an observation.
  3. INSUFFICIENT -- even the factor data does not reach the window (it predates the
                  factor history): no number is fabricated; the holding is flagged.

The portfolio cumulative return / drawdown for a window is computed from the PER-STOCK
MIX of actual and modeled monthly returns, weighted by portfolio weight (renormalised
across the holdings that have a figure). stress_test.json records, per window, the
portfolio figure AND a per-holding breakdown tagging each holding "actual" / "modeled"
/ "insufficient", so the report can disclose -- honestly and per window -- that a
blended figure mixes measured history with factor-estimated returns, and which
holdings fell in each bucket. A blended number is never presented as fully observed.

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

# Factor handoffs (same files the simulation / expected-return model use):
#   factor_history.json   -- per-market factor returns + monthly RF, dated by month.
#   factor_residuals.json -- per-asset market + category betas (aligned by factor name).
FACTOR_HISTORY_PATH   = os.path.join(SCRIPT_DIR, "factor_history.json")
FACTOR_RESIDUALS_PATH = os.path.join(SCRIPT_DIR, "factor_residuals.json")

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


# ── Factor reconstruction (for windows a stock predates) ─────────────────────────

def load_factor_panels(path=FACTOR_HISTORY_PATH):
    """
    Per-market monthly factor panel from factor_history.json, indexed by month
    Period -- the SAME source the simulation / expected-return model read.

    Returns {market: {"factors": DataFrame (cols = factor names, no RF),
                      "rf": Series (monthly RF), "names": [...]}}.
    Returns {} if the file is missing/unreadable (caller degrades gracefully).
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    out = {}
    for market, md in data.get("markets", {}).items():
        try:
            idx = pd.PeriodIndex(pd.to_datetime(md["dates"]), freq="M")
            names = [c for c in md["columns"] if c != "RF"]      # factor names, by name
            factors = pd.DataFrame({c: md["factor_returns"][c] for c in names}, index=idx)
            rf = pd.Series(md.get("monthly_rf", []), index=idx, name="RF")
            out[market] = {"factors": factors, "rf": rf, "names": names}
        except Exception:
            continue
    return out


def load_betas_map(path=FACTOR_RESIDUALS_PATH):
    """
    {ticker: {"market": str, "betas": {factor_name: beta}}} from factor_residuals.json
    -- the SAME fixed category betas the Method-B simulation uses (aligned by name).
    Returns {} on failure.
    """
    try:
        with open(path, encoding="utf-8") as f:
            assets = json.load(f).get("assets", {})
    except Exception:
        return {}
    out = {}
    for t, a in assets.items():
        betas = a.get("betas")
        if isinstance(betas, dict):
            out[t] = {"market": a.get("market"), "betas": betas}
    return out


def _modeled_window_returns(ticker, start, end, factor_panels, betas_map):
    """
    Reconstruct one stock's MONTHLY returns over [start, end] from its factor
    exposures, for a window it has no real history in:

        r_modeled(t) = RF(t) + sum_f beta_f * factor_f(t)

    (the systematic component used by Method B / the expected-return model; the
    idiosyncratic residual is unobservable for a window the stock didn't trade, so
    it is dropped -- a reconstructed return is an ESTIMATE, not an observation).

    Returns (returns_series_indexed_by_Period, status, detail) where status is
    "modeled" on success or "insufficient" when the factor data does not reach the
    window (or the stock has no betas / market). Never fabricates.
    """
    info = betas_map.get(ticker)
    if not info or not info.get("betas"):
        return None, "insufficient", "no factor betas available for this holding"
    market = info.get("market")
    fh = factor_panels.get(market)
    if fh is None:
        return None, "insufficient", f"no factor history for market '{market}'"

    factors, rf = fh["factors"], fh["rf"]
    start_p = pd.Period(pd.Timestamp(start), freq="M")
    end_p   = pd.Period(pd.Timestamp(end),   freq="M")
    idx = factors.index[(factors.index >= start_p) & (factors.index <= end_p)]
    if len(idx) == 0:
        return None, "insufficient", ("factor history does not reach this window "
                                      f"(starts {factors.index.min()})")

    contrib = pd.Series(0.0, index=idx)
    for fname, beta in info["betas"].items():
        if fname in factors.columns:
            contrib = contrib + float(beta) * factors[fname].reindex(idx)
    modeled = (rf.reindex(idx) + contrib).dropna()
    if modeled.empty:
        return None, "insufficient", "no usable factor months inside this window"
    return modeled, "modeled", (f"{len(modeled)} months reconstructed from factor "
                                f"exposures ({market} {'/'.join(fh['names'])})")


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
    # Anchor the wealth curve at 1.0 (initial capital) BEFORE any month's return, so the
    # FIRST month's move is included in the drawdown. Using cumprod alone makes the first
    # post-return value the initial peak and silently drops a first-month decline -- which
    # let a window end down more than its reported "worst dip" (e.g. COVID: end -14% but
    # dip only -11%, since the trough was both months but the peak started after month 1).
    # With the 1.0 anchor the trough is measured from true starting capital, so the
    # invariant max_drawdown >= |min(0, cumulative_return)| always holds.
    wealth = np.concatenate(([1.0], np.cumprod(1.0 + port)))
    cum_return = float(wealth[-1] - 1.0)
    peak = np.maximum.accumulate(wealth)
    max_dd = float((wealth / peak - 1.0).min())      # <= 0
    return cum_return, -max_dd, int(aligned.shape[0])


def _as_period_index(series):
    """Re-index a month-end-Timestamp return series by month Period (for alignment
    with the factor-reconstructed series, which is Period-indexed)."""
    s = series.copy()
    s.index = pd.PeriodIndex(s.index, freq="M")
    return s[~s.index.duplicated(keep="last")]


def stress_one_window(label, start, end, weights, factor_panels=None, betas_map=None,
                      verbose=True):
    """
    Replay one crisis window with a PER-STOCK, PER-WINDOW basis decision:
      * ACTUAL       -- real history covers the window: use real monthly returns.
      * MODELED      -- no history: reconstruct from factor exposures (an estimate).
      * INSUFFICIENT -- not even the factor data reaches the window: no number.

    The portfolio figure blends the actual + modeled monthly returns weighted by
    portfolio weight (renormalised over holdings that have a figure). Returns a result
    dict including a per-holding breakdown so the report can disclose the mix. Never
    fabricates a return.
    """
    factor_panels = factor_panels if factor_panels is not None else load_factor_panels()
    betas_map     = betas_map     if betas_map     is not None else load_betas_map()

    holdings = [t for t, w in weights.items() if w > 1e-9]
    n_total  = len(holdings)

    used_w, ret_cols = {}, {}     # holdings with a figure (actual or modeled)
    per_holding = []              # ordered breakdown for the report
    actual_names, modeled_names, insufficient = [], [], []

    for t in holdings:
        rets, status, detail = _monthly_returns(t, start, end)
        if status == "ok":
            basis = "actual"
            used_w[t]   = weights[t]
            ret_cols[t] = _as_period_index(rets)
            actual_names.append(t)
        else:
            # No usable real history (missing / listed_after / transient) -> try to
            # reconstruct from factor exposures. The original reason is preserved so
            # the report can say WHY it had to be modeled ("listed after this period").
            real_reason = {"listed_after": "no price history before this window "
                                           "(listed later)",
                           "missing":      "no provider price history this far back",
                           "transient":    "price data unavailable this run"}.get(
                               status, detail)
            m_rets, m_status, m_detail = _modeled_window_returns(
                t, start, end, factor_panels, betas_map)
            if m_status == "modeled":
                basis = "modeled"
                used_w[t]   = weights[t]
                ret_cols[t] = m_rets
                modeled_names.append(t)
                detail = f"{m_detail}; real history unavailable ({real_reason})"
            else:
                basis = "insufficient"
                insufficient.append(t)
                detail = f"{m_detail}; real history also unavailable ({real_reason})"
        per_holding.append({
            "ticker": t,
            "weight": round(float(weights[t]), 6),
            "basis":  basis,
            "detail": detail,
        })
        if verbose:
            print(f"      {t:<14} {basis:<12} {detail}")

    used_weight = float(sum(used_w.values()))
    n_actual, n_modeled = len(actual_names), len(modeled_names)
    n_cov = n_actual + n_modeled       # holdings with a figure (back-compat: "covered")

    # Backward-compatible "missing" list (holdings with NO figure at all).
    missing = [{"ticker": t, "reason": "insufficient", "detail": "no real history and "
                "factor data does not reach this window"} for t in insufficient]

    base = {
        "window":           f"{start} .. {end}",
        "n_holdings":       n_total,
        "n_actual":         n_actual,
        "n_modeled":        n_modeled,
        "n_insufficient":   len(insufficient),
        "n_covered":        n_cov,
        "covered_weight":   round(used_weight, 6),
        "holdings":         per_holding,
        "actual_holdings":      actual_names,
        "modeled_holdings":     modeled_names,
        "insufficient_holdings": insufficient,
        "missing":          missing,
    }

    if n_cov == 0:
        base.update({
            "basis":             "insufficient",
            "status":            "insufficient_data",
            "cumulative_return": None,
            "max_drawdown":      None,
            "note": (f"INSUFFICIENT DATA: none of the {n_total} holdings have real "
                     "history OR factor data reaching this window. No crisis return "
                     "computed -- nothing fabricated."),
        })
        return base

    returns_df = pd.DataFrame(ret_cols)
    cum, dd, n_months = _portfolio_path_metrics(returns_df, used_w)
    if cum is None:
        base.update({
            "basis":             "insufficient",
            "status":            "insufficient_data",
            "cumulative_return": None,
            "max_drawdown":      None,
            "note": (f"INSUFFICIENT DATA: the {n_cov} usable holdings share no common "
                     "month in this window. No crisis return computed."),
        })
        return base

    # basis: "measured" (all actual), "blended" (any modeled), "partial" (actual
    # subset, rest insufficient, no modeling possible).
    if n_modeled > 0:
        basis = "blended"
    elif len(insufficient) == 0:
        basis = "measured"
    else:
        basis = "partial"
    status = {"measured": "ok", "blended": "blended",
              "partial": "partial_coverage"}[basis]

    # Plain-language note that NEVER reads a blended estimate as fully observed.
    bits = []
    if actual_names:
        bits.append(f"{', '.join(actual_names)} from actual history")
    if modeled_names:
        bits.append(f"{', '.join(modeled_names)} modeled from factor exposures")
    if insufficient:
        bits.append(f"{', '.join(insufficient)} excluded (insufficient data)")
    mix = "; ".join(bits)
    verb = "Estimated" if modeled_names else "Based on real history,"
    note = (f"{verb} {cum*100:+.0f}% over the window ({mix}; "
            f"covered weight {used_weight*100:.1f}% of portfolio, renormalised across "
            f"included names; {n_months} common months).")
    if modeled_names:
        note += (" This figure BLENDS measured history with factor-estimated returns -- "
                 "the modeled portion is an estimate, not an observation.")
    base.update({
        "basis":             basis,
        "status":            status,
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

    # Load the factor handoffs ONCE (shared across windows). If they are missing the
    # reconstruction simply can't run and stocks without history fall to "insufficient"
    # -- the old, honest behaviour -- rather than crashing.
    factor_panels = load_factor_panels()
    betas_map     = load_betas_map()

    if verbose:
        W = 74
        print(f"\n{'='*W}")
        print("  HISTORICAL CRISIS STRESS TEST")
        print(f"  Portfolio: {sheet or 'chosen'}   |   Holdings: "
              f"{', '.join(f'{t} {w*100:.1f}%' for t, w in holdings.items())}")
        if factor_panels and betas_map:
            print("  Short-history holdings are reconstructed from factor exposures "
                  "(US 3-factor / India 4-factor); flagged 'modeled' (estimate).")
        else:
            print("  NOTE: factor handoffs unavailable -- short-history holdings will "
                  "be flagged 'insufficient' (no reconstruction).")
        print(f"{'='*W}")

    windows_out = {}
    for label, start, end in windows:
        if verbose:
            print(f"\n  [{label}]  {start} .. {end}")
        windows_out[label] = stress_one_window(
            label, start, end, holdings,
            factor_panels=factor_panels, betas_map=betas_map, verbose=verbose)
        r = windows_out[label]
        if verbose:
            if r["cumulative_return"] is None:
                print(f"    -> {r['status'].upper()}: {r['note']}")
            else:
                print(f"    -> cumulative {r['cumulative_return']*100:+.2f}%   "
                      f"max drawdown -{r['max_drawdown']*100:.2f}%   [{r['basis']}]")
                print(f"       {r['note']}")

    payload = {
        "status":           "ok",
        "portfolio":        sheet or "chosen",
        "holdings":         {t: round(w, 6) for t, w in holdings.items()},
        "coverage_rule":    (f"per-stock, per-window: a holding uses REAL history if its "
                             f"first monthly observation is within {COVERAGE_TOLERANCE_DAYS} "
                             f"days of the window start ('actual'); otherwise its return is "
                             f"RECONSTRUCTED from factor exposures (RF + sum beta*factor, "
                             f"same factors/betas as the simulation) and flagged 'modeled' "
                             f"(an estimate); if the factor data does not reach the window "
                             f"it is 'insufficient' and excluded -- never back-filled or "
                             f"fabricated"),
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
