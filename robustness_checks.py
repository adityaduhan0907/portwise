#!/usr/bin/env python3
"""
robustness_checks.py  --  Layer 6 robustness annotations

Runs AFTER the portfolio has been chosen and stabilised (Layer 7 resampling +
Layer 5 risk evaluation). Its job is to surface plain-language warnings when the
chosen portfolio rests on shaky ground. It is NON-BLOCKING and NON-INTERACTIVE:
the checks annotate the output, they never halt the pipeline and never prompt.

Three checks, all evaluated against the chosen goal's RESAMPLED weights (the sheet
in resampled_portfolios.xlsx whose name matches inputs["portfolio_choice"]):

  1. Negative momentum   (MOMENTUM_MIN = 0.0)
       Flag any held stock whose 11-month momentum (returns_stats.xlsx ->
       'Annualised Mu') is below 0, reporting its weight so a heavily-weighted
       falling stock stands out.

  2. High pairwise corr  (CORR_MAX = 0.85)
       Flag every pair where BOTH stocks carry meaningful weight (> 1%) in the
       chosen portfolio and their correlation exceeds 0.85. The correlation
       matrix is read from returns_stats.xlsx if module 1 persisted one,
       otherwise derived from the annualised covariance
       (corr_ij = cov_ij / sqrt(cov_ii * cov_jj)).

  3. Sector concentration (SECTOR_MAX = 0.50)
       Sum the chosen portfolio's weights by sector (sectors.json from module 1),
       report the largest sector's share and the effective number of sectors
       (1 / sum(sector_weight^2)), and flag if the largest sector exceeds 50%.

Output
------
  robustness_warnings.json   keyed by check; each carries {triggered, ...details,
                             message}. The downstream Module 5 report reads this.
  Console summary            plain language; a clear check says so explicitly.

Usage
-----
  python robustness_checks.py [PORTFOLIO_NAME]

  PORTFOLIO_NAME (optional) overrides the chosen sheet. Otherwise the choice is
  read from run_config.json ("portfolio_choice"); if that is unavailable the
  first sheet in resampled_portfolios.xlsx is used (and the fallback is noted).
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────────
SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
RESAMPLED_PATH    = os.path.join(SCRIPT_DIR, "resampled_portfolios.xlsx")
RETURNS_PATH      = os.path.join(SCRIPT_DIR, "returns_stats.xlsx")
SECTORS_PATH      = os.path.join(SCRIPT_DIR, "sectors.json")
CONFIG_PATH       = os.path.join(SCRIPT_DIR, "run_config.json")
WARNINGS_PATH     = os.path.join(SCRIPT_DIR, "robustness_warnings.json")

# ── Thresholds (named constants) ────────────────────────────────────────────────
MOMENTUM_MIN      = 0.0     # 11-month momentum below this is flagged
CORR_MAX          = 0.85    # pairwise correlation above this is flagged
SECTOR_MAX        = 0.50    # largest sector share above this is flagged
MIN_PAIR_WEIGHT   = 0.01    # both stocks in a pair must exceed 1% to flag

W = 68

# Rows in resampled_portfolios.xlsx that are not ticker holdings.
_SKIP_LABELS = {
    "Portfolio Return (%)", "Portfolio Volatility (%)",
    "Portfolio Sharpe Ratio", "", "nan", "none",
}


# ── Console helpers ─────────────────────────────────────────────────────────────

def _bar(char="="):
    return char * W

def _pct(x):
    return f"{x * 100:.1f}%"


# ── Input loading ───────────────────────────────────────────────────────────────

def resolve_portfolio_choice():
    """
    Decide which resampled sheet to evaluate. Priority:
      1. CLI argument
      2. run_config.json -> "portfolio_choice"
      3. first sheet of resampled_portfolios.xlsx (fallback, noted)
    Returns (choice_or_None, available_sheets, note).
    """
    try:
        sheets = pd.ExcelFile(RESAMPLED_PATH).sheet_names
    except Exception as exc:
        return None, [], f"could not open resampled_portfolios.xlsx ({exc})"

    # 1. CLI arg
    if len(sys.argv) > 1 and sys.argv[1].strip():
        arg = sys.argv[1].strip()
        if arg in sheets:
            return arg, sheets, "chosen via command-line argument"
        return arg, sheets, f"command-line portfolio '{arg}' not among sheets {sheets}"

    # 2. run_config.json
    choice = None
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            choice = json.load(f).get("portfolio_choice")
    except Exception:
        choice = None
    if choice and choice in sheets:
        return choice, sheets, "chosen via run_config.json"

    # 3. fallback to first sheet
    if sheets:
        note = (f"portfolio_choice unavailable -- defaulting to first sheet "
                f"'{sheets[0]}'")
        return sheets[0], sheets, note
    return None, sheets, "resampled_portfolios.xlsx has no sheets"


def load_chosen_weights(sheet):
    """
    Parse the chosen resampled sheet into {ticker: weight_fraction}.
    Weights in the file are percentages; they are returned as fractions that
    sum to ~1.0 over the held stocks. Stat rows (return/vol/Sharpe) are skipped.
    """
    df = pd.read_excel(RESAMPLED_PATH, sheet_name=sheet)
    weights = {}
    for _, row in df.iterrows():
        stock = str(row.get("Stock", "")).strip()
        # Skip the stats block: explicit labels plus any "Portfolio ..." summary
        # row (Return / Volatility / Sharpe / CVaR / future additions).
        if (stock.lower() in {s.lower() for s in _SKIP_LABELS}
                or stock.lower().startswith("portfolio ")):
            continue
        try:
            wt = float(row.get("Weight (%)", None))
        except (ValueError, TypeError):
            continue
        if wt > 0:
            weights[stock] = wt / 100.0
    return weights


def load_momentum():
    """
    Read the per-stock 11-month momentum figure module 1 computes
    ('Annualised Mu' sheet, column 'Annualised_Expected_Return', a decimal).
    Returns {ticker: momentum_decimal}.
    """
    mu = pd.read_excel(RETURNS_PATH, sheet_name="Annualised Mu")
    out = {}
    for _, row in mu.iterrows():
        ticker = str(row.get("Ticker", "")).strip()
        try:
            out[ticker] = float(row.get("Annualised_Expected_Return"))
        except (ValueError, TypeError):
            continue
    return out


def _read_persisted_correlation():
    """
    Try to read the correlation matrix module 1 persists in the
    'Stats & Correlation' sheet (block under the 'Correlation Matrix' label,
    built from long-run monthly returns). Returns a DataFrame or None.
    """
    try:
        raw = pd.read_excel(RETURNS_PATH, sheet_name="Stats & Correlation",
                            header=None)
    except Exception:
        return None

    # Locate the 'Correlation Matrix' label row.
    label_row = None
    for i in range(len(raw)):
        cell = str(raw.iat[i, 0])
        if cell.strip().lower().startswith("correlation matrix"):
            label_row = i
            break
    if label_row is None:
        return None

    header_row = label_row + 1                      # ticker names across columns
    tickers = [str(x).strip() for x in raw.iloc[header_row, 1:].tolist()
               if str(x).strip() and str(x).strip().lower() != "nan"]
    if not tickers:
        return None

    data = {}
    for r in range(header_row + 1, len(raw)):
        name = str(raw.iat[r, 0]).strip()
        if not name or name.lower() == "nan":
            break
        vals = []
        for c in range(1, 1 + len(tickers)):
            try:
                vals.append(float(raw.iat[r, c]))
            except (ValueError, TypeError):
                vals.append(np.nan)
        data[name] = vals
    if not data:
        return None
    return pd.DataFrame(data, index=tickers).T[tickers]


def _correlation_from_cov():
    """
    Derive correlation from the annualised covariance:
        corr_ij = cov_ij / sqrt(cov_ii * cov_jj)
    Returns a DataFrame or None.
    """
    try:
        cov = pd.read_excel(RETURNS_PATH, sheet_name="Annualised Cov",
                            index_col=0)
    except Exception:
        return None
    d = np.sqrt(np.diag(cov.values))
    denom = np.outer(d, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov.values / denom
    return pd.DataFrame(corr, index=cov.index, columns=cov.columns)


def load_correlation():
    """
    Get the correlation matrix: the persisted one if module 1 wrote it,
    otherwise derived from the annualised covariance.
    Returns (DataFrame_or_None, source_label).
    """
    corr = _read_persisted_correlation()
    if corr is not None and not corr.empty:
        return corr, "returns_stats.xlsx 'Stats & Correlation' (long-run monthly returns)"
    corr = _correlation_from_cov()
    if corr is not None and not corr.empty:
        return corr, "derived from 'Annualised Cov' (corr_ij = cov_ij / sqrt(cov_ii*cov_jj))"
    return None, "unavailable"


def load_sectors():
    """Read {ticker: sector} from sectors.json; missing file -> empty dict."""
    try:
        with open(SECTORS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


# ── Check 1: negative momentum ──────────────────────────────────────────────────

def check_momentum(weights, momentum):
    flagged = []
    missing = []
    for ticker, wt in sorted(weights.items(), key=lambda kv: -kv[1]):
        if ticker not in momentum:
            missing.append(ticker)
            continue
        mom = momentum[ticker]
        if mom < MOMENTUM_MIN:
            flagged.append({"ticker": ticker, "momentum": mom, "weight": wt})

    if flagged:
        parts = []
        for f in flagged:
            parts.append(
                f"{f['ticker']} has fallen over the past ~11 months "
                f"(momentum {f['momentum'] * 100:+.1f}%) and makes up "
                f"{_pct(f['weight'])} of this portfolio."
            )
        message = " ".join(parts)
    else:
        message = "No momentum concerns flagged: every holding has non-negative 11-month momentum."

    if missing:
        message += (f" (Note: no momentum figure for {', '.join(missing)}; "
                    f"not assessed.)")

    return {
        "triggered":  bool(flagged),
        "threshold":  MOMENTUM_MIN,
        "flagged":    flagged,
        "unassessed": missing,
        "message":    message,
    }


# ── Check 2: high pairwise correlation ──────────────────────────────────────────

def check_correlation(weights, corr, source):
    flagged = []
    if corr is not None:
        # Only stocks with meaningful weight (> 1%) that are in the matrix.
        held = [t for t, w in weights.items()
                if w > MIN_PAIR_WEIGHT and t in corr.index and t in corr.columns]
        for i in range(len(held)):
            for j in range(i + 1, len(held)):
                a, b = held[i], held[j]
                try:
                    c = float(corr.loc[a, b])
                except Exception:
                    continue
                if np.isnan(c):
                    continue
                if c > CORR_MAX:
                    flagged.append({
                        "stock_a": a, "stock_b": b, "correlation": c,
                        "weight_a": weights[a], "weight_b": weights[b],
                    })

    if flagged:
        parts = []
        for f in flagged:
            parts.append(
                f"{f['stock_a']} and {f['stock_b']} move almost together "
                f"(correlation {f['correlation']:.2f}) -- holding both adds less "
                f"diversification than it looks."
            )
        message = " ".join(parts)
    elif corr is None:
        message = "No correlation matrix available -- correlation check could not run."
    else:
        message = (f"No high-correlation concerns flagged: no pair of meaningfully "
                   f"weighted holdings exceeds {CORR_MAX:.2f}.")

    return {
        "triggered":     bool(flagged),
        "threshold":     CORR_MAX,
        "min_weight":    MIN_PAIR_WEIGHT,
        "source":        source,
        "flagged_pairs": flagged,
        "message":       message,
    }


# ── Check 3: sector concentration ───────────────────────────────────────────────

def check_sector(weights, sectors):
    total = sum(weights.values())
    if total <= 0:
        return {
            "triggered": False, "threshold": SECTOR_MAX, "sector_weights": {},
            "largest_sector": None, "largest_share": 0.0,
            "effective_sectors": 0.0, "unclassified": [],
            "message": "No weights available -- sector check could not run.",
        }

    sector_w = {}
    unclassified = []
    for ticker, wt in weights.items():
        sec = sectors.get(ticker) or "Unknown"
        if sec == "Unknown":
            unclassified.append(ticker)
        sector_w[sec] = sector_w.get(sec, 0.0) + wt

    # Normalise to fractions of the held book (weights already ~sum to 1).
    sector_frac = {s: w / total for s, w in sector_w.items()}
    largest_sector = max(sector_frac, key=sector_frac.get)
    largest_share  = sector_frac[largest_sector]
    eff_sectors    = 1.0 / sum(v ** 2 for v in sector_frac.values())

    triggered = largest_share > SECTOR_MAX

    if triggered:
        message = (f"This portfolio is {_pct(largest_share)} {largest_sector} -- "
                   f"concentrated in one sector. Effective spread: about "
                   f"{eff_sectors:.1f} sectors.")
    else:
        message = (f"No sector-concentration concerns flagged: the largest sector "
                   f"({largest_sector}) is {_pct(largest_share)}, under the "
                   f"{SECTOR_MAX * 100:.0f}% limit. Effective spread: about "
                   f"{eff_sectors:.1f} sectors.")

    if unclassified:
        message += (f" (Note: sector unknown for {', '.join(unclassified)}; "
                    f"grouped as 'Unknown'.)")

    return {
        "triggered":         triggered,
        "threshold":         SECTOR_MAX,
        "sector_weights":    {s: round(v, 6) for s, v in sector_frac.items()},
        "largest_sector":    largest_sector,
        "largest_share":     largest_share,
        "effective_sectors": eff_sectors,
        "unclassified":      unclassified,
        "message":           message,
    }


# ── Orchestration ───────────────────────────────────────────────────────────────

def main():
    print(f"\n{_bar('=')}")
    print(f"{'LAYER 6  --  ROBUSTNESS CHECKS (non-blocking annotations)':^{W}}")
    print(_bar("="))

    choice, sheets, note = resolve_portfolio_choice()
    if choice is None or not sheets:
        print(f"  [SKIP] {note}")
        # Still write a clear, non-triggered file so the report has a state.
        result = {k: {"triggered": False, "message": f"Check skipped: {note}"}
                  for k in ("negative_momentum", "high_correlation",
                            "sector_concentration")}
        with open(WARNINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return

    print(f"  Chosen portfolio : {choice}")
    print(f"  Selection        : {note}")

    weights = load_chosen_weights(choice)
    momentum = load_momentum()
    corr, corr_source = load_correlation()
    sectors = load_sectors()

    print(f"  Holdings         : "
          + ", ".join(f"{t} {_pct(w)}" for t, w in
                      sorted(weights.items(), key=lambda kv: -kv[1])))

    mom_res    = check_momentum(weights, momentum)
    corr_res   = check_correlation(weights, corr, corr_source)
    sector_res = check_sector(weights, sectors)

    result = {
        "negative_momentum":    {**mom_res,    "portfolio_choice": choice},
        "high_correlation":     {**corr_res,   "portfolio_choice": choice},
        "sector_concentration": {**sector_res, "portfolio_choice": choice},
    }

    with open(WARNINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # ── Plain-language console summary ─────────────────────────────────────────
    print(f"\n{_bar('-')}")
    print("  ROBUSTNESS SUMMARY")
    print(_bar("-"))

    n_fired = sum(1 for c in (mom_res, corr_res, sector_res) if c["triggered"])

    for title, res in (
        ("1. Momentum",            mom_res),
        ("2. Correlation",         corr_res),
        ("3. Sector concentration", sector_res),
    ):
        tag = "[FLAG]" if res["triggered"] else "[ OK ]"
        print(f"\n  {tag} {title}")
        print(f"         {res['message']}")

    print(f"\n  {n_fired} of 3 checks flagged a concern.")
    print(f"  Written -> {os.path.basename(WARNINGS_PATH)}")
    print(f"{_bar('=')}\n")


if __name__ == "__main__":
    main()
