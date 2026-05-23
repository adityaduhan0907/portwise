#!/usr/bin/env python3
"""
module0_riskfree.py

Run this FIRST before any other module.

Fetches live risk-free rates:
  - US  : 13-week T-bill yield via ^IRX  (yfinance)
  - India: 10-year G-Sec yield via IN10Y-IN.BO  (yfinance)
            Falls back to 6.5 % if the ticker is unavailable.

Calculates a blended rate weighted by the proportion of US vs Indian
stocks in the portfolio, then writes all three rates to risk_free_rates.json
so module1, module2, and module3 can read from one shared source of truth.
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime

import yfinance as yf

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH  = os.path.join(SCRIPT_DIR, "risk_free_rates.json")

# Default fallback rates (decimal form)
US_FALLBACK_RATE    = 0.043   # 4.3 %  — approximate 2024/25 T-bill rate
INDIA_FALLBACK_RATE = 0.065   # 6.5 %  — approximate 10-year G-Sec rate

# yfinance ticker symbols
US_TICKER     = "^IRX"          # 13-week Treasury bill (annualised %)
INDIA_TICKER  = "IN10Y-IN.BO"   # India 10-year G-Sec proxy

def _load_portfolio_tickers():
    """
    Return the ticker list for US/India classification.
    When called from run_all.py, reads the user's tickers from run_config.json.
    Falls back to the hardcoded default when run standalone.
    """
    try:
        cfg_path = os.path.join(SCRIPT_DIR, "run_config.json")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        tickers = cfg.get("tickers")
        if tickers:
            return tickers
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"  WARNING: Could not read run_config.json for tickers ({exc}).")
    # Default standalone ticker list (edit here for standalone use)
    return [
        "AAPL", "MSFT", "GOOGL",
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.BO",
    ]

PORTFOLIO_TICKERS = _load_portfolio_tickers()

# ── Fetch helpers ──────────────────────────────────────────────────────────────

def fetch_latest_close(symbol, lookback="10d", max_retries=3, delay=2):
    """
    Return the most recent closing value for a yfinance ticker, or None on
    any failure (no data, HTTP error, empty frame, etc.).  Retries up to
    max_retries times before giving up.
    """
    for attempt in range(max_retries):
        try:
            hist = yf.Ticker(symbol).history(period=lookback, auto_adjust=True)
            if isinstance(hist.columns, object) and "Close" in hist.columns:
                close = hist["Close"].dropna()
                if not close.empty:
                    return float(close.iloc[-1])
            return None
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(delay)
    return None


def fetch_us_rate():
    """
    Fetch the annualised 13-week T-bill yield via ^IRX.
    ^IRX is quoted as a percentage (e.g. 3.59 means 3.59 %), so divide by 100.
    Returns (rate_decimal, source_description).
    """
    raw = fetch_latest_close(US_TICKER)
    if raw is not None and raw > 0:
        return raw / 100.0, f"{US_TICKER} (13-week T-bill, last close {raw:.3f}%)"

    print(f"  WARNING: Could not fetch {US_TICKER}. Using fallback {US_FALLBACK_RATE*100:.2f}%.")
    return US_FALLBACK_RATE, f"fallback ({US_TICKER} unavailable)"


def fetch_india_rate():
    """
    Fetch the India 10-year G-Sec yield via IN10Y-IN.BO.
    Quoted as a percentage on Yahoo Finance; divide by 100.
    Falls back to INDIA_FALLBACK_RATE if the ticker is unavailable.
    Returns (rate_decimal, source_description).
    """
    raw = fetch_latest_close(INDIA_TICKER)
    if raw is not None and raw > 0:
        return raw / 100.0, f"{INDIA_TICKER} (India 10-year G-Sec, last close {raw:.3f}%)"

    print(f"  WARNING: Could not fetch {INDIA_TICKER}.")
    print(f"  WARNING: Using fallback Indian risk-free rate of {INDIA_FALLBACK_RATE*100:.2f}%.")
    return INDIA_FALLBACK_RATE, f"fallback ({INDIA_TICKER} unavailable, default {INDIA_FALLBACK_RATE*100:.1f}%)"


# ── Portfolio classification ───────────────────────────────────────────────────

def classify_tickers(tickers):
    """
    Split tickers into Indian (.NS or .BO suffix) and US (everything else).
    Returns (us_list, india_list).
    """
    india = [t for t in tickers if t.upper().endswith(".NS") or t.upper().endswith(".BO")]
    us    = [t for t in tickers if t not in india]
    return us, india


def blended_rate(us_rate, india_rate, us_tickers, india_tickers):
    """
    Weighted average of US and Indian rates by stock count.
    Returns (blended_decimal, us_weight, india_weight).
    """
    total = len(us_tickers) + len(india_tickers)
    if total == 0:
        return us_rate, 1.0, 0.0

    w_us    = len(us_tickers)    / total
    w_india = len(india_tickers) / total
    blended = us_rate * w_us + india_rate * w_india
    return blended, w_us, w_india


# ── Display helpers ────────────────────────────────────────────────────────────

def print_section(title, width=62):
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def print_rate_line(label, rate_decimal, source):
    print(f"  {label:<30} {rate_decimal*100:>6.4f}%  ({source})")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*62}")
    print("  Module 0 -- Risk-Free Rate Fetcher")
    print(f"  Rates fetched at : {fetched_at}")
    print(f"{'='*62}")

    # ── 1. Fetch live rates ────────────────────────────────────────────────────
    print_section("FETCHING RATES")

    us_rate,    us_source    = fetch_us_rate()
    india_rate, india_source = fetch_india_rate()

    print_rate_line("US risk-free rate    ", us_rate,    us_source)
    print_rate_line("India risk-free rate ", india_rate, india_source)

    # ── 2. Classify portfolio tickers ─────────────────────────────────────────
    print_section("PORTFOLIO CLASSIFICATION")

    us_tickers, india_tickers = classify_tickers(PORTFOLIO_TICKERS)
    total = len(us_tickers) + len(india_tickers)

    print(f"  Total tickers    : {total}")
    print(f"  US stocks  ({len(us_tickers):>2}) : {', '.join(us_tickers) if us_tickers else 'none'}")
    print(f"  Indian stocks ({len(india_tickers):>2}): {', '.join(india_tickers) if india_tickers else 'none'}")

    # ── 3. Blended rate ────────────────────────────────────────────────────────
    print_section("BLENDED RATE CALCULATION")

    blended, w_us, w_india = blended_rate(us_rate, india_rate, us_tickers, india_tickers)

    print(f"  US weight        : {w_us*100:.1f}%  ({len(us_tickers)} of {total} stocks)")
    print(f"  India weight     : {w_india*100:.1f}%  ({len(india_tickers)} of {total} stocks)")
    print()
    print(f"  Blended = ({us_rate*100:.4f}% x {w_us*100:.1f}%) + ({india_rate*100:.4f}% x {w_india*100:.1f}%)")
    print(f"          = {us_rate*100*w_us:.4f}% + {india_rate*100*w_india:.4f}%")
    print(f"          = {blended*100:.4f}%")

    # ── 4. Summary ─────────────────────────────────────────────────────────────
    print_section("RATE SUMMARY")
    width_label = 30
    print(f"  {'US Risk-Free Rate':<{width_label}} {us_rate*100:>8.4f}%")
    print(f"  {'Indian Risk-Free Rate':<{width_label}} {india_rate*100:>8.4f}%")
    print(f"  {'Blended Risk-Free Rate':<{width_label}} {blended*100:>8.4f}%")
    print(f"\n  Use BLENDED rate in module1/2/3 for this mixed portfolio.")

    # ── 5. Write JSON ──────────────────────────────────────────────────────────
    payload = {
        "fetched_at":        fetched_at,
        "us_rate":           round(us_rate,    8),
        "india_rate":        round(india_rate, 8),
        "blended_rate":      round(blended,    8),
        "us_rate_pct":       round(us_rate    * 100, 4),
        "india_rate_pct":    round(india_rate * 100, 4),
        "blended_rate_pct":  round(blended    * 100, 4),
        "us_rate_source":    us_source,
        "india_rate_source": india_source,
        "portfolio": {
            "total_stocks":      total,
            "us_stocks":         us_tickers,
            "india_stocks":      india_tickers,
            "us_proportion":     round(w_us,    6),
            "india_proportion":  round(w_india, 6),
        },
    }

    try:
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n  Saved -> {JSON_PATH}")
    except Exception as exc:
        print(f"\n  ERROR: Could not write {JSON_PATH}: {exc}")
        sys.exit(1)

    print(f"\n{'='*62}\n")


if __name__ == "__main__":
    main()
