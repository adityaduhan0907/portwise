#!/usr/bin/env python3
"""
module1_data.py  —  Split Estimation (Momentum Returns + Long-Run Risk)

METHODOLOGY
  Expected returns  : 12M-1M momentum window.
                      Monthly prices fetched ~16 months back; the most recent
                      month is skipped to avoid short-term reversal, leaving an
                      11-month holding-period return annualised to 12 months.
  Covariance matrix : 10 years of monthly data for stable long-run risk.
  Display data      : 3 years of daily adjusted closing prices (prices.xlsx).

OUTPUTS
  prices.xlsx         — 3-year daily adjusted closing prices
  returns_stats.xlsx  — 6 sheets:
      "Daily Returns"           daily % returns (3y, display only)
      "Momentum Returns"        monthly % returns in the 11-month window
      "LongRun Monthly Returns" 10-year monthly % returns
      "Stats & Correlation"     per-stock stats + correlation (labeled)
      "Annualised Mu"           pre-computed expected-return vector
      "Annualised Cov"          pre-computed covariance matrix
  (module2 and module3 read "Annualised Mu" and "Annualised Cov" directly)
"""

import json
import os
import warnings
from datetime import date

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
TRADING_DAYS      = 252
TRADING_MONTHS    = 12
DAILY_YEARS       = 3
LONGRUN_YEARS     = 10
SKIP_RECENT       = 1      # months to skip at near end (short-term reversal)
MOM_WINDOW        = 11     # holding-period months = 12 − 1
MIN_DATA_FRAC     = 0.70
SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))


def _load_risk_free_rate(fallback=0.043):
    """Load blended risk-free rate from risk_free_rates.json, or fall back to default."""
    json_path = os.path.join(SCRIPT_DIR, "risk_free_rates.json")
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        rate       = float(data["blended_rate"])
        fetched_at = data.get("fetched_at", "unknown")
        print(f"  Risk-free rate : {rate*100:.4f}%  "
              f"(blended, risk_free_rates.json -- fetched {fetched_at})")
        return rate
    except FileNotFoundError:
        print(f"  WARNING: risk_free_rates.json not found -- using fallback {fallback*100:.1f}%.")
        print("           Run module0_riskfree.py first for fresh rates.")
        return fallback
    except Exception as exc:
        print(f"  WARNING: Could not read risk_free_rates.json ({exc}). "
              f"Using fallback {fallback*100:.1f}%.")
        return fallback


RISK_FREE_RATE = _load_risk_free_rate()

def _load_tickers():
    """
    Return the ticker list to process.
    When called from run_all.py, reads the user's tickers from run_config.json.
    Falls back to the hardcoded default when run standalone.
    """
    try:
        cfg_path = os.path.join(SCRIPT_DIR, "run_config.json")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        tickers = cfg.get("tickers")
        if tickers:
            print(f"  Tickers from run_config.json: {', '.join(tickers)}")
            return tickers
    except FileNotFoundError:
        pass
    except Exception as exc:
        print(f"  WARNING: Could not read run_config.json for tickers ({exc}).")
    # Default standalone ticker list (edit here for standalone use)
    return [
        "AAPL",
        "MSFT",
        "GOOGL",
        "RELIANCE",    # NSE -- auto-resolved to RELIANCE.NS
        "TCS",         # NSE -- auto-resolved to TCS.NS
        "INFY.NS",     # already has suffix
        "HDFCBANK",    # NSE -- auto-resolved to HDFCBANK.NS
        "WIPRO.BO",    # BSE -- already has suffix
    ]

# ─────────────────────────────────────────────────────────────────────────────
TICKERS = _load_tickers()
# ─────────────────────────────────────────────────────────────────────────────


# ── Download helpers ───────────────────────────────────────────────────────────

def _fetch_close(symbol, start, end, interval="1d"):
    """
    Download adjusted closing prices for one symbol at the given interval.
    Returns a pd.Series (DatetimeIndex -> float) or None on any failure.
    """
    try:
        raw = yf.download(
            symbol, start=start, end=end,
            interval=interval,
            auto_adjust=True, progress=False, threads=False,
        )
        if raw.empty:
            return None
        close = raw["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        return close if not close.empty else None
    except Exception:
        return None


def resolve_and_fetch(raw_ticker, start, end, interval="1d"):
    """
    Try the ticker as supplied; if no data and no dot suffix, retry with
    .NS then .BO.  Returns (resolved_symbol, series, error_message).
    """
    candidates = (
        [raw_ticker] if "." in raw_ticker
        else [raw_ticker, f"{raw_ticker}.NS", f"{raw_ticker}.BO"]
    )
    for symbol in candidates:
        series = _fetch_close(symbol, start, end, interval)
        if series is not None:
            if symbol != raw_ticker:
                print(f"    Auto-resolved '{raw_ticker}' -> '{symbol}'")
            return symbol, series, None
    tried = ", ".join(candidates)
    return None, None, f"No data returned (tried: {tried})"


# ── Momentum return (12M-1M) ───────────────────────────────────────────────────

def compute_momentum_return(monthly_prices):
    """
    Standard 12M-1M momentum:
      Skip the SKIP_RECENT most-recent monthly close(s) to avoid reversal.
      Compute the MOM_WINDOW-month holding return, annualise to 12 months.

    Requires >= (SKIP_RECENT + MOM_WINDOW + 2) = 14 monthly price points.
    Returns annualised return (decimal), or float('nan') if insufficient data.

    With SKIP_RECENT=1, MOM_WINDOW=11 and prices sorted ascending:
      end_price   = prices[-3]   (2 months ago)
      start_price = prices[-14]  (13 months ago)
      holding     = 11 months
    """
    prices = monthly_prices.sort_index().dropna()
    needed = SKIP_RECENT + MOM_WINDOW + 2
    if len(prices) < needed:
        return float("nan")

    end_price   = float(prices.iloc[-(SKIP_RECENT + 2)])
    start_price = float(prices.iloc[-(SKIP_RECENT + MOM_WINDOW + 2)])

    if start_price <= 0:
        return float("nan")

    total_return = end_price / start_price - 1
    ann_return   = (1.0 + total_return) ** (12.0 / MOM_WINDOW) - 1
    return ann_return


# ── Per-stock statistics (split estimation) ────────────────────────────────────

def annualised_stats_split(mom_return, longrun_monthly_returns):
    """
    Expected return from momentum window; volatility from long-run monthly data.
    """
    clean   = longrun_monthly_returns.dropna()
    ann_vol = clean.std() * np.sqrt(TRADING_MONTHS)

    if np.isnan(mom_return) or ann_vol == 0:
        sharpe = float("nan")
    else:
        sharpe = (mom_return - RISK_FREE_RATE) / ann_vol

    return {
        "Ann. Return % (Momentum 12M-1M)": (
            round(mom_return * 100, 2) if not np.isnan(mom_return) else float("nan")
        ),
        "Ann. Volatility % (10Y Monthly)": round(ann_vol * 100, 2),
        "Sharpe Ratio":                    (
            round(sharpe, 4) if not np.isnan(sharpe) else float("nan")
        ),
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    today = date.today()

    # Date windows
    daily_start = (pd.Timestamp(today)
                   - pd.DateOffset(years=DAILY_YEARS, days=10)).date()
    mom_start   = (pd.Timestamp(today)
                   - pd.DateOffset(months=SKIP_RECENT + MOM_WINDOW + 4)).date()
    cov_start   = (pd.Timestamp(today)
                   - pd.DateOffset(years=LONGRUN_YEARS, months=3)).date()

    min_obs_daily = int(TRADING_DAYS * DAILY_YEARS * MIN_DATA_FRAC)

    W = 70
    print(f"\n{'='*W}")
    print("  Module 1 -- Split Estimation: Momentum Returns + Long-Run Risk")
    print(f"  Daily data   : {daily_start} -> {today}  ({DAILY_YEARS}y)")
    print(f"  Momentum     : {SKIP_RECENT + MOM_WINDOW + 1}M lookback, skip {SKIP_RECENT}M,")
    print(f"                 {MOM_WINDOW}-month holding period, annualised (monthly data)")
    print(f"  Covariance   : {LONGRUN_YEARS}-year monthly data")
    print(f"  Risk-free r  : {RISK_FREE_RATE*100:.4f}%")
    print(f"{'='*W}\n")

    print("  NOTE: Expected returns use the momentum anomaly (12M-1M window).")
    print("        The most recent month is excluded to avoid short-term reversal.")
    print("  NOTE: Risk estimates use 10 years of monthly data for stability.\n")

    # ── 1. Fetch all data ──────────────────────────────────────────────────────
    daily_prices  = {}   # symbol -> pd.Series (daily)
    mom_prices    = {}   # symbol -> pd.Series (monthly closes, momentum window)
    cov_prices    = {}   # symbol -> pd.Series (monthly closes, long-run)
    failed        = {}   # original ticker -> error message

    for raw in TICKERS:
        ticker = raw.strip().upper()
        print(f"  [{ticker}]")

        # Daily (3 years) -- determines the resolved symbol
        symbol, daily_s, err = resolve_and_fetch(ticker, daily_start, today, "1d")
        if err:
            print(f"    SKIP: {err}")
            failed[ticker] = err
            continue
        if len(daily_s) < min_obs_daily:
            err = (f"only {len(daily_s)} daily obs, need >= {min_obs_daily} "
                   f"({MIN_DATA_FRAC:.0%} of {DAILY_YEARS}y)")
            print(f"    SKIP: {err}")
            failed[ticker] = err
            continue

        daily_prices[symbol] = daily_s
        print(f"    Daily      : {len(daily_s):>4} obs  OK")

        # Momentum monthly (reuse resolved symbol)
        _, mom_s, err_m = resolve_and_fetch(symbol, mom_start, today, "1mo")
        if err_m or mom_s is None:
            print(f"    Momentum   : WARNING -- {err_m or 'no data'}")
            mom_prices[symbol] = None
        else:
            needed = SKIP_RECENT + MOM_WINDOW + 2
            if len(mom_s) < needed:
                print(f"    Momentum   : WARNING -- only {len(mom_s)} monthly prices "
                      f"(need {needed}); momentum return will use fallback")
            else:
                print(f"    Momentum   : {len(mom_s):>4} monthly prices  OK")
            mom_prices[symbol] = mom_s

        # Long-run monthly (reuse resolved symbol)
        _, cov_s, err_c = resolve_and_fetch(symbol, cov_start, today, "1mo")
        if err_c or cov_s is None:
            print(f"    Long-run   : WARNING -- {err_c or 'no data'}; will use available data")
            cov_prices[symbol] = None
        else:
            min_cov = int(LONGRUN_YEARS * TRADING_MONTHS * MIN_DATA_FRAC)
            if len(cov_s) < min_cov:
                print(f"    Long-run   : WARNING -- only {len(cov_s)} monthly prices "
                      f"(need ~{min_cov} for full {LONGRUN_YEARS}y); using all available")
            else:
                print(f"    Long-run   : {len(cov_s):>4} monthly prices  OK")
            cov_prices[symbol] = cov_s

    if not daily_prices:
        print("\n  No tickers processed successfully. Exiting.\n")
        return

    symbols = list(daily_prices.keys())

    # ── 2. Build DataFrames ────────────────────────────────────────────────────
    prices_df  = pd.DataFrame(daily_prices).sort_index()
    returns_df = prices_df.pct_change().iloc[1:]

    # Momentum monthly returns (only the window rows, for the Excel sheet)
    mom_dfs = {s: v for s, v in mom_prices.items() if v is not None}
    if mom_dfs:
        mom_prices_df  = pd.DataFrame(mom_dfs).sort_index()
        mom_returns_df = mom_prices_df.pct_change().iloc[1:]
        # Keep only the rows within the momentum window (last MOM_WINDOW + SKIP_RECENT + 1)
        keep = MOM_WINDOW + SKIP_RECENT + 2
        mom_window_df  = mom_returns_df.tail(keep)
    else:
        mom_prices_df  = pd.DataFrame()
        mom_window_df  = pd.DataFrame()

    # Long-run monthly returns
    cov_dfs = {s: v for s, v in cov_prices.items() if v is not None}
    if cov_dfs:
        cov_prices_df = pd.DataFrame(cov_dfs).sort_index()
        cov_rets_df   = cov_prices_df.pct_change().iloc[1:]
    else:
        print("  WARNING: No long-run monthly data -- falling back to daily returns for covariance.")
        cov_rets_df = returns_df

    # ── 3. Compute mu (momentum) ───────────────────────────────────────────────
    mu_dict = {}
    for symbol in symbols:
        m_prices = mom_prices.get(symbol)
        if m_prices is not None:
            r = compute_momentum_return(m_prices)
        else:
            r = float("nan")

        if np.isnan(r):
            # Fallback: annualised mean of available daily returns
            if symbol in returns_df.columns:
                r = returns_df[symbol].dropna().mean() * TRADING_DAYS
                print(f"  [{symbol}] Momentum fallback: using daily-based annualised return.")
            else:
                r = 0.0
        mu_dict[symbol] = r

    # ── 4. Compute annualised covariance (long-run monthly) ────────────────────
    cov_cols    = [s for s in symbols if s in cov_rets_df.columns]
    cov_clean   = cov_rets_df[cov_cols].dropna()

    if not cov_clean.empty:
        cov_matrix = cov_clean.cov() * TRADING_MONTHS
    else:
        cov_matrix = returns_df[cov_cols].dropna().cov() * TRADING_DAYS
        print("  WARNING: Using daily returns for covariance -- long-run monthly unavailable.")

    # ── 5. Per-stock stats and correlation ────────────────────────────────────
    common = [s for s in symbols if s in cov_matrix.columns]

    stats_rows = {}
    for s in common:
        lr_rets = cov_clean[s] if s in cov_clean.columns else pd.Series(dtype=float)
        stats_rows[s] = annualised_stats_split(mu_dict[s], lr_rets)

    stats_df = pd.DataFrame(stats_rows).T
    stats_df.index.name = "Ticker"

    corr_df = cov_clean.corr() if not cov_clean.empty else returns_df[cov_cols].corr()

    # ── 6. Prepare Annualised Mu and Cov for export ───────────────────────────
    mu_series  = pd.Series({s: mu_dict[s] for s in common}, name="Annualised_Expected_Return")
    mu_series.index.name = "Ticker"
    cov_export = cov_matrix.loc[common, common]

    # ── 7. Console output ──────────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print("  INDIVIDUAL STOCK STATISTICS")
    print("  Return source : momentum window (12M-1M, monthly data)")
    print("  Risk source   : long-run monthly standard deviation (10y)")
    print(f"{'='*W}")
    print(stats_df.to_string())

    print(f"\n{'='*W}")
    print("  ANNUALISED EXPECTED RETURNS  (momentum 12M-1M)")
    print(f"{'='*W}")
    for s, r in mu_dict.items():
        print(f"    {s:<22}  {r*100:>8.2f}%")

    print(f"\n{'='*W}")
    print("  CORRELATION MATRIX  (source: long-run monthly returns)")
    print(f"{'='*W}")
    print(corr_df.round(4).to_string())

    print(f"\n{'='*W}")
    print("  *** MOMENTUM RISK WARNING ***")
    print("  Momentum-based expected returns carry REVERSAL RISK during market")
    print("  regime changes.  High recent performers may underperform sharply")
    print("  when trends reverse.  RERUN THIS MODEL AT LEAST MONTHLY to stay")
    print("  current.  Do not use stale estimates for live portfolio decisions.")
    print(f"{'='*W}\n")

    # ── 8. Excel export ────────────────────────────────────────────────────────
    prices_path  = os.path.join(SCRIPT_DIR, "prices.xlsx")
    returns_path = os.path.join(SCRIPT_DIR, "returns_stats.xlsx")

    prices_df.to_excel(prices_path, index_label="Date")

    note_rows = [
        "METHODOLOGY NOTES",
        f"Expected Return : Momentum window (12M-1M, monthly data; most recent "
        f"{SKIP_RECENT} month skipped for reversal)",
        f"Risk (Volatility): {LONGRUN_YEARS}-year monthly data, annualised (x sqrt(12))",
        f"Risk-free rate   : {RISK_FREE_RATE*100:.4f}% (from risk_free_rates.json)",
        "Module2 / Module3: read 'Annualised Mu' and 'Annualised Cov' sheets directly.",
    ]
    note_df = pd.DataFrame({"Note": note_rows})

    n_stats = len(stats_df)
    gap     = 2
    with pd.ExcelWriter(returns_path, engine="openpyxl") as writer:

        # Sheet 1: Daily Returns
        returns_df.to_excel(writer, sheet_name="Daily Returns", index_label="Date")

        # Sheet 2: Momentum Returns (window rows only)
        if not mom_window_df.empty:
            mom_window_df.to_excel(
                writer, sheet_name="Momentum Returns", index_label="Date"
            )

        # Sheet 3: LongRun Monthly Returns
        if not cov_clean.empty:
            cov_clean.to_excel(
                writer, sheet_name="LongRun Monthly Returns", index_label="Date"
            )

        # Sheet 4: Stats & Correlation  (with methodology header)
        sname = "Stats & Correlation"
        note_df.to_excel(writer, sheet_name=sname, startrow=0, index=False)

        stats_start   = len(note_df) + 2
        corr_lbl_row  = stats_start + n_stats + gap       # 0-based
        corr_data_row = corr_lbl_row + 1                  # 0-based

        stats_df.to_excel(writer, sheet_name=sname, startrow=stats_start)
        ws = writer.sheets[sname]
        ws.cell(row=corr_lbl_row + 1, column=1,
                value="Correlation Matrix  (long-run monthly returns)")
        corr_df.round(4).to_excel(
            writer, sheet_name=sname, startrow=corr_data_row
        )

        # Sheet 5: Annualised Mu  (module2 / module3 source)
        mu_export = mu_series.reset_index()
        mu_export.columns = ["Ticker", "Annualised_Expected_Return"]
        mu_export["Estimation"] = f"Momentum {SKIP_RECENT + MOM_WINDOW + 1}M-{SKIP_RECENT}M (monthly)"
        mu_export.to_excel(writer, sheet_name="Annualised Mu", index=False)

        # Sheet 6: Annualised Cov  (module2 / module3 source)
        cov_export.to_excel(writer, sheet_name="Annualised Cov")

    print(f"  Exported -> {prices_path}")
    print(f"  Exported -> {returns_path}")
    print( "    Sheets : Daily Returns | Momentum Returns | LongRun Monthly Returns")
    print( "             Stats & Correlation | Annualised Mu | Annualised Cov")

    # ── 9. Summary ─────────────────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print("  SUMMARY")
    print(f"{'='*W}")
    print(f"  Successfully processed : {len(daily_prices)}")
    print(f"  Failed / skipped       : {len(failed)}")
    if failed:
        print()
        for t, reason in failed.items():
            print(f"    {t}: {reason}")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    main()
