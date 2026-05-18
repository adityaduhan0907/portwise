#!/usr/bin/env python3
"""
run_all.py

Master pipeline runner for the portfolio model.
Collects ALL user inputs in one interactive block at the start, then
executes each module in sequence without further interruption.

Pipeline
--------
  module0_riskfree    -> fetch live risk-free rates
  module1_data        -> download prices, momentum returns, covariance
  module2_optimiser   -> mean-variance optimisation (3 portfolios)
  module3_frontier    -> efficient frontier chart
  module4_rebalance   -> portfolio rebalancing plan  (non-interactive)
  module5_report      -> HTML report (skipped if file not found)

Robustness checks (automatic, with user prompts where applicable)
  - Data freshness   : warn if prices.xlsx > 1 day old
  - Momentum         : warn on negative 11M-1M return; offer ticker removal
  - Correlation      : warn if any pair corr > 0.85
  - Concentration    : warn if any country > 30% in any optimal portfolio

Usage
-----
  python run_all.py
"""

import json
import os
import subprocess
import sys
import warnings
from datetime import datetime

import pandas as pd

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────
SCRIPT_DIR       = os.path.dirname(os.path.abspath(__file__))
W                = 68          # terminal column width
CONFIG_PATH      = os.path.join(SCRIPT_DIR, "run_config.json")
CORR_THRESHOLD   = 0.85        # correlation pair warning threshold
CONC_THRESHOLD   = 30.0        # single-country weight warning threshold (%)
INDIA_SFX        = (".NS", ".BO")

# Labels in optimised_portfolios.xlsx that are not ticker rows
_SKIP_LABELS = {
    "Portfolio Return (%)", "Portfolio Volatility (%)",
    "Portfolio Sharpe Ratio", "", "nan",
}


# ── Terminal helpers ───────────────────────────────────────────────────────────

def _bar(char="="):
    return char * W

def module_header(label, description=""):
    """Print a clearly delimited section header before each module runs."""
    print(f"\n{_bar('-')}")
    line = f"  RUNNING {label}"
    if description:
        line += f"  --  {description}"
    print(line)
    print(_bar("-"))


def _ok(label):
    print(f"\n  [OK] {label} complete")

def _warn(msg):
    print(f"\n  [WARN] {msg}")

def _fail(label, reason=""):
    print(f"\n  [FAIL] {label}")
    if reason:
        print(f"         {reason}")


# ── Portfolio options ──────────────────────────────────────────────────────────

def _load_portfolio_opts(xlsx_path):
    """
    Parse optimised_portfolios.xlsx.
    Returns {sheet_name: {ticker: weight_pct}} for non-zero weights only.
    """
    opts = {}
    try:
        xf = pd.ExcelFile(xlsx_path)
        for sheet in xf.sheet_names:
            df = pd.read_excel(xlsx_path, sheet_name=sheet)
            w = {}
            for _, row in df.iterrows():
                s  = str(row.get("Stock", "")).strip()
                wt = row.get("Weight (%)", None)
                if s in _SKIP_LABELS:
                    continue
                try:
                    val = float(wt)
                    if val > 0:
                        w[s] = val
                except (ValueError, TypeError):
                    pass
            if w:
                opts[sheet] = w
    except Exception:
        pass
    return opts


def _ask_portfolio_choice(opts):
    """Display available portfolios and return the user's chosen name."""
    names = list(opts.keys())
    print()
    print("  Available portfolios:")
    for i, name in enumerate(names, 1):
        tickers_str = ", ".join(opts[name].keys())
        print(f"    {i}.  {name}")
        print(f"         Holdings: {tickers_str}")
    while True:
        try:
            c = int(input("\n  Enter portfolio number: ").strip())
            if 1 <= c <= len(names):
                chosen = names[c - 1]
                print(f"  Selected: {chosen}")
                return chosen
            print(f"  Please enter a number between 1 and {len(names)}.")
        except ValueError:
            print("  Please enter a valid number.")


# ── User input collection ──────────────────────────────────────────────────────

def collect_inputs():
    """
    Gather all pipeline inputs in one clean interactive block.
    Returns a dict with keys: tickers, benchmark, currency, holdings,
    portfolio_choice (str or None if portfolios don't yet exist).
    """
    print(f"\n{_bar('=')}")
    print("  PIPELINE INPUT COLLECTION")
    print("  Answer each prompt below. You will be asked to confirm")
    print("  all inputs before the pipeline begins.")
    print(_bar("="))

    inputs = {}

    # ── Step 1: Tickers ────────────────────────────────────────────────────────
    print("\n  Step 1 of 5  --  Stock Tickers")
    print("  Comma-separated list. Indian stocks: append .NS or .BO suffix.")
    print("  Example: AAPL, GOOGL, RELIANCE.NS, TCS.NS")
    while True:
        raw = input("  Tickers: ").strip()
        tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
        if len(tickers) >= 3:
            inputs["tickers"] = tickers
            break
        print("  Please enter at least 3 tickers.")

    # ── Step 2: Benchmark ──────────────────────────────────────────────────────
    print("\n  Step 2 of 5  --  Benchmark Index Ticker")
    print("  Used for context in the report (e.g. ^GSPC, ^NSEI, ^FTSE).")
    bm = input("  Benchmark [^GSPC]: ").strip().upper()
    inputs["benchmark"] = bm if bm else "^GSPC"

    # ── Step 3: Currency ───────────────────────────────────────────────────────
    print("\n  Step 3 of 5  --  Holdings Display Currency")
    while True:
        c = input("  Display currency [USD/INR]: ").strip().upper()
        if c in ("USD", "INR"):
            inputs["currency"] = c
            break
        print("  Please type USD or INR.")

    # ── Step 4: Current holdings ───────────────────────────────────────────────
    print("\n  Step 4 of 5  --  Current Holdings")
    print("  Enter each position as: ticker + number of shares held.")
    print("  Type 'done' when all positions have been entered.")
    print()
    holdings = []
    while True:
        t = input("  Ticker (or 'done'): ").strip().upper()
        if t.lower() == "done":
            if not holdings:
                print("  Please enter at least one holding before typing 'done'.")
                continue
            break
        if not t:
            continue
        while True:
            try:
                s = float(
                    input(f"  Shares of {t}: ").strip().replace(",", "")
                )
                if s < 0:
                    print("  Shares cannot be negative.")
                    continue
                break
            except ValueError:
                print("  Please enter a valid number (e.g. 10 or 10.5).")
        holdings.append({"ticker": t, "shares": s})
        print(f"  Added: {t}  x  {s:,.4f} shares")
    inputs["holdings"] = holdings

    # ── Step 5: Portfolio choice (if portfolios already exist) ─────────────────
    print("\n  Step 5 of 5  --  Target Portfolio for Rebalancing")
    portfolios_path = os.path.join(SCRIPT_DIR, "optimised_portfolios.xlsx")
    if os.path.exists(portfolios_path):
        print("  Found optimised_portfolios.xlsx from a previous run.")
        opts = _load_portfolio_opts(portfolios_path)
        if opts:
            inputs["portfolio_choice"] = _ask_portfolio_choice(opts)
        else:
            inputs["portfolio_choice"] = None
            print("  Could not parse portfolios -- choice deferred to after optimisation.")
    else:
        inputs["portfolio_choice"] = None
        print("  optimised_portfolios.xlsx not yet available.")
        print("  Portfolio choice will be collected after module3 completes.")

    return inputs


def confirm_inputs(inputs):
    """
    Show a full summary of all collected inputs and ask the user to confirm
    before the pipeline starts -- giving them a chance to catch any mistakes.
    """
    print(f"\n{_bar('=')}")
    print("  PLEASE CONFIRM YOUR INPUTS")
    print(_bar("="))
    print(f"  Tickers      : {', '.join(inputs['tickers'])}")
    print(f"  Benchmark    : {inputs['benchmark']}")
    print(f"  Currency     : {inputs['currency']}")
    print(f"  Holdings     : {len(inputs['holdings'])} position(s)")
    for h in inputs["holdings"]:
        print(f"                 {h['ticker']:<22} {h['shares']:>12,.4f} shares")
    if inputs.get("portfolio_choice"):
        print(f"  Portfolio    : {inputs['portfolio_choice']}")
    else:
        print("  Portfolio    : Will be selected after optimisation completes")
    print()
    try:
        input("  Press Enter to begin the pipeline  (Ctrl+C to cancel) ... ")
    except KeyboardInterrupt:
        print("\n\n  Cancelled.")
        sys.exit(0)


# ── Run config file ────────────────────────────────────────────────────────────

def write_run_config(inputs):
    """
    Write run_config.json so that module0 and module1 read the user's tickers
    instead of their hardcoded default lists. All other inputs are also stored
    for reference and for the end summary.
    """
    cfg = {
        "tickers":          inputs["tickers"],
        "benchmark":        inputs.get("benchmark", "^GSPC"),
        "currency":         inputs.get("currency", "USD"),
        "holdings":         inputs.get("holdings", []),
        "portfolio_choice": inputs.get("portfolio_choice"),
        "created_at":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# ── Subprocess runner ──────────────────────────────────────────────────────────

def run_subprocess(label, script_name, description=""):
    """
    Execute a module script as a subprocess in the project directory.

    Returns
    -------
    True   -- script exited with code 0 (success)
    False  -- script exited with non-zero code (failure)
    None   -- script file not found (skipped)
    """
    module_header(label, description)
    script_path = os.path.join(SCRIPT_DIR, script_name)

    if not os.path.exists(script_path):
        print(f"  [SKIP] {script_name} not found in {SCRIPT_DIR}")
        return None

    try:
        subprocess.run(
            [sys.executable, script_path],
            cwd=SCRIPT_DIR,
            check=True,
        )
        _ok(label)
        return True

    except subprocess.CalledProcessError as exc:
        _fail(label, f"script exited with code {exc.returncode}")
        return False
    except Exception as exc:
        _fail(label, f"unexpected error: {exc}")
        return False


# ── Robustness checks ──────────────────────────────────────────────────────────

def check_momentum(inputs):
    """
    Read 'Annualised Mu' from returns_stats.xlsx.
    For each ticker with a negative momentum return, warn the user and
    optionally remove it from the ticker list.

    Returns a list of removed tickers (empty if none removed).
    """
    returns_path = os.path.join(SCRIPT_DIR, "returns_stats.xlsx")
    removed = []

    try:
        mu_df = pd.read_excel(returns_path, sheet_name="Annualised Mu")
    except Exception:
        return removed

    for _, row in mu_df.iterrows():
        ticker = str(row.get("Ticker", "")).strip()
        ret    = row.get("Annualised_Expected_Return", 0.0)
        try:
            ret = float(ret)
        except (ValueError, TypeError):
            continue

        if ret < 0:
            print(f"\n  [WARN] {ticker} has negative momentum "
                  f"({ret * 100:.2f}%) over the 11-month window.")
            resp = input(
                f"  Include {ticker} in the optimisation anyway? [yes/no]: "
            ).strip().lower()

            if resp not in ("yes", "y"):
                # Remove the resolved ticker AND any un-suffixed version the
                # user may have typed (e.g. remove both "RELIANCE.NS" and "RELIANCE")
                base = ticker
                for sfx in INDIA_SFX:
                    if ticker.upper().endswith(sfx):
                        base = ticker[: -len(sfx)]
                        break
                to_drop = {ticker.upper(), base.upper()}

                inputs["tickers"] = [
                    t for t in inputs["tickers"] if t.upper() not in to_drop
                ]
                inputs["holdings"] = [
                    h for h in inputs["holdings"]
                    if h["ticker"].upper() not in to_drop
                ]
                removed.append(ticker)
                print(f"  Removed {ticker} from the pipeline.")

    return removed


def check_correlation(inputs):
    """
    Read 'LongRun Monthly Returns' and warn about any stock pair with
    pairwise correlation above CORR_THRESHOLD.  No user action required.
    """
    returns_path = os.path.join(SCRIPT_DIR, "returns_stats.xlsx")
    try:
        lr_df = pd.read_excel(
            returns_path, sheet_name="LongRun Monthly Returns", index_col=0
        )
        corr  = lr_df.corr()
    except Exception:
        return

    cols    = list(corr.columns)
    warned  = set()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            t1, t2 = cols[i], cols[j]
            c = float(corr.loc[t1, t2])
            pair = tuple(sorted([t1, t2]))
            if c > CORR_THRESHOLD and pair not in warned:
                print(f"\n  [WARN] {t1} and {t2} have a correlation "
                      f"of {c:.4f}.")
                print("         They may not provide true diversification.")
                warned.add(pair)


def check_concentration(inputs):
    """
    Check whether any single country (US or India) holds more than
    CONC_THRESHOLD % of any optimal portfolio.  Prints a warning but
    does NOT stop the pipeline.
    """
    portfolios_path = os.path.join(SCRIPT_DIR, "optimised_portfolios.xlsx")
    try:
        xf = pd.ExcelFile(portfolios_path)
    except Exception:
        return

    for sheet in xf.sheet_names:
        try:
            df = pd.read_excel(portfolios_path, sheet_name=sheet)
        except Exception:
            continue

        us_w, in_w = 0.0, 0.0
        for _, row in df.iterrows():
            s  = str(row.get("Stock", "")).strip()
            wt = row.get("Weight (%)", None)
            if s in _SKIP_LABELS:
                continue
            try:
                w = float(wt)
                if s.upper().endswith(INDIA_SFX):
                    in_w += w
                else:
                    us_w += w
            except (ValueError, TypeError):
                pass

        for country, weight in [("US", us_w), ("India", in_w)]:
            if weight > CONC_THRESHOLD:
                print(
                    f"\n  [WARN] {country} represents {weight:.1f}% of the "
                    f"'{sheet}' portfolio."
                )
                print(
                    f"         This exceeds the {CONC_THRESHOLD:.0f}% "
                    "concentration limit. Continuing anyway."
                )


# ── Module 4 — non-interactive execution ──────────────────────────────────────

def run_module4(inputs, today_str):
    """
    Run the rebalancing analysis without interactive prompts by calling
    module4_rebalance's helper functions directly with the pre-collected inputs.

    Returns a dict of rebalancing stats for the end summary, or None on failure.
    """
    module_header("Module 4", "Portfolio Rebalancing  (non-interactive)")

    try:
        import module4_rebalance as m4
    except ImportError as exc:
        _fail("Module 4", f"cannot import module4_rebalance: {exc}")
        return None

    prices_path     = os.path.join(SCRIPT_DIR, "prices.xlsx")
    portfolios_path = os.path.join(SCRIPT_DIR, "optimised_portfolios.xlsx")

    # Load latest prices from prices.xlsx
    try:
        prices = m4.load_latest_prices(prices_path)
    except Exception as exc:
        _fail("Module 4", f"could not read prices.xlsx: {exc}")
        return None

    # Fetch live USD/INR exchange rate (needed even for USD display because
    # Indian stock prices in prices.xlsx are quoted in INR)
    print()
    usd_inr = m4.fetch_usd_inr_rate()

    # Map user holdings to resolved tickers in prices.xlsx
    raw_holdings = []
    for h in inputs.get("holdings", []):
        resolved, price = m4.lookup_price(h["ticker"], prices)
        if resolved is None:
            _warn(
                f"'{h['ticker']}' not found in prices.xlsx -- "
                "skipping this holding"
            )
            continue
        raw_holdings.append(
            (resolved, h["shares"], price, m4.is_indian(resolved))
        )

    if not raw_holdings:
        _warn("No valid holdings could be resolved. Skipping Module 4.")
        return None

    # Ensure a portfolio choice is available
    portfolio_opts = m4.load_portfolio_options(portfolios_path)
    choice         = inputs.get("portfolio_choice")

    if not choice or choice not in portfolio_opts:
        if choice:
            _warn(
                f"Previous portfolio choice '{choice}' is not available "
                "in the current optimisation. Please choose again."
            )
        print(f"\n{_bar()}")
        print("  SELECT TARGET PORTFOLIO FOR REBALANCING")
        print(_bar())
        choice = _ask_portfolio_choice(portfolio_opts)
        inputs["portfolio_choice"] = choice

    target_weights = portfolio_opts[choice]

    # Calculate current positions and generate instructions
    positions, total_usd = m4.build_current_portfolio(raw_holdings, usd_inr)
    instructions = m4.generate_instructions(
        positions, target_weights, total_usd, usd_inr
    )

    # Print the rebalancing plan to the terminal
    m4.print_summary(
        positions, total_usd, choice, target_weights,
        instructions, inputs["currency"], usd_inr, today_str,
    )

    # Export to Excel
    date_stamp = datetime.now().strftime("%Y%m%d")
    out_path   = os.path.join(SCRIPT_DIR, f"rebalancing_plan_{date_stamp}.xlsx")
    try:
        m4.export_to_excel(
            positions, total_usd, choice, target_weights,
            instructions, inputs["currency"], usd_inr, today_str, out_path,
        )
    except Exception as exc:
        _fail("Module 4 export", str(exc))
        return None

    _ok("Module 4")

    return {
        "total_usd": total_usd,
        "currency":  inputs["currency"],
        "usd_inr":   usd_inr,
        "actioned":  sum(1 for i in instructions if "Actioned" in i["status"]),
        "skipped":   sum(1 for i in instructions if "Skipped"  in i["status"]),
        "out_path":  out_path,
    }


# ── End summary ────────────────────────────────────────────────────────────────

def print_end_summary(inputs, rebalance_stats, today_str):
    """
    Print the final pipeline summary after all modules have completed.
    Reads portfolio weights and the risk-free rate from their output files.
    """
    portfolios_path = os.path.join(SCRIPT_DIR, "optimised_portfolios.xlsx")
    risk_json_path  = os.path.join(SCRIPT_DIR, "risk_free_rates.json")
    date_stamp      = datetime.now().strftime("%Y%m%d")

    # Load risk-free rate
    rfr_pct = 0.0
    try:
        with open(risk_json_path, encoding="utf-8") as f:
            rfr_pct = json.load(f).get("blended_rate", 0.0) * 100
    except Exception:
        pass

    # Load portfolio weight summaries
    portfolio_opts = (
        _load_portfolio_opts(portfolios_path)
        if os.path.exists(portfolios_path)
        else {}
    )

    border = "=" * W
    print(f"\n\n{border}")
    print(f"{'PORTFOLIO MODEL  --  RUN COMPLETE':^{W}}")
    print(border)
    print(f"  Run date         : {today_str}")
    print(f"  Tickers          : {', '.join(inputs['tickers'])}")
    print(f"  Benchmark        : {inputs['benchmark']}")
    print(f"  Risk-free rate   : {rfr_pct:.4f}%")

    # Optimal portfolio summaries
    if portfolio_opts:
        print()
        print("  OPTIMAL PORTFOLIOS")
        for name, weights in portfolio_opts.items():
            w_str = "  ".join(f"{t} {v:.1f}%" for t, v in weights.items())
            print(f"    {name:<24}: {w_str}")

    # Rebalancing summary
    if rebalance_stats:
        rs = rebalance_stats
        print()
        print("  REBALANCING SUMMARY")
        print(f"    Total portfolio value  : ${rs['total_usd']:>14,.2f} USD")
        if rs.get("currency") == "INR" and rs.get("usd_inr"):
            inr_total = rs["total_usd"] * rs["usd_inr"]
            print(f"                           INR {inr_total:>12,.2f}")
        print(f"    Trades to action       : {rs['actioned']}")
        print(f"    Trades skipped (< 1%)  : {rs['skipped']}")

    # Files created
    print()
    print("  FILES CREATED")
    expected_files = [
        "prices.xlsx",
        "returns_stats.xlsx",
        "optimised_portfolios.xlsx",
        "efficient_frontier.png",
        f"rebalancing_plan_{date_stamp}.xlsx",
        f"portfolio_report_{date_stamp}.html",
    ]
    for fname in expected_files:
        fpath  = os.path.join(SCRIPT_DIR, fname)
        marker = "[OK]  " if os.path.exists(fpath) else "[NOTE]"
        print(f"    {marker} {fname}")

    print(f"\n{border}\n")


# ── Main pipeline ──────────────────────────────────────────────────────────────

def main():
    today_str = datetime.now().strftime("%Y-%m-%d")

    print(f"\n{_bar('=')}")
    print(f"{'PORTFOLIO MODEL  --  MASTER PIPELINE RUNNER':^{W}}")
    print(f"  {today_str}")
    print(_bar("="))
    print("  Pipeline: module0 -> module1 -> module2 -> module3 ->"
          "  module4 -> module5")

    # ── Data freshness check (before collecting inputs) ────────────────────────
    prices_path = os.path.join(SCRIPT_DIR, "prices.xlsx")
    if os.path.exists(prices_path):
        age_days = (
            datetime.now()
            - datetime.fromtimestamp(os.path.getmtime(prices_path))
        ).total_seconds() / 86400

        if age_days > 1:
            print(f"\n  {_bar('*')}")
            print(f"  [WARN] prices.xlsx is {age_days:.1f} day(s) old.")
            print("  Recommend refreshing price data before optimising.")
            print(f"  {_bar('*')}")
            resp = input(
                "\n  Refresh price data now (run full pipeline fresh)? "
                "[yes/no]: "
            ).strip().lower()
            if resp not in ("yes", "y"):
                print("  Continuing with existing price data.")

    # ── Collect all inputs upfront ─────────────────────────────────────────────
    try:
        inputs = collect_inputs()
    except KeyboardInterrupt:
        print("\n\n  Pipeline cancelled.")
        sys.exit(0)

    # ── Confirm before starting ────────────────────────────────────────────────
    confirm_inputs(inputs)

    # ── Write run_config.json (modules 0 and 1 will read it) ──────────────────
    write_run_config(inputs)
    print(f"\n  Run configuration saved to run_config.json")

    rebalance_stats = None    # populated after module4 runs

    # ── MODULE 0 -- Risk-free rates ────────────────────────────────────────────
    if not run_subprocess(
        "Module 0", "module0_riskfree.py", "Fetching live risk-free rates"
    ):
        _fail("Pipeline", "Module 0 failed -- cannot continue without risk-free rates")
        return

    # ── MODULE 1 -- Prices & returns  (with momentum retry loop) ──────────────
    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        # Refresh config in case tickers were trimmed in a previous iteration
        write_run_config(inputs)

        label = "Module 1" if attempt == 0 else f"Module 1 (retry {attempt})"
        if not run_subprocess(
            label, "module1_data.py",
            "Downloading prices, momentum returns, long-run covariance",
        ):
            _fail("Pipeline", "Module 1 failed -- cannot continue")
            return

        # -- Momentum robustness check --
        print(f"\n{_bar()}")
        print("  ROBUSTNESS CHECK 1 of 3  --  Momentum")
        print(_bar())
        removed = check_momentum(inputs)

        # -- Correlation robustness check (only once per module1 run) --
        print(f"\n{_bar()}")
        print("  ROBUSTNESS CHECK 2 of 3  --  Correlation")
        print(_bar())
        check_correlation(inputs)

        if not removed:
            break    # no tickers dropped; proceed to module2

        remaining = len(inputs["tickers"])
        if remaining < 3:
            _fail(
                "Pipeline",
                f"Only {remaining} ticker(s) remain after removal -- "
                "need at least 3. Please restart with a broader list.",
            )
            return

        print(
            f"\n  Re-running Module 1 with updated tickers: "
            f"{', '.join(inputs['tickers'])}"
        )
    else:
        _warn(
            f"Maximum retries ({MAX_RETRIES}) reached for ticker removal. "
            "Proceeding with remaining tickers."
        )

    # ── MODULE 2 -- Optimisation ───────────────────────────────────────────────
    if not run_subprocess(
        "Module 2", "module2_optimiser.py", "Mean-variance optimisation"
    ):
        _fail("Pipeline", "Module 2 failed -- cannot continue")
        return

    # ── MODULE 3 -- Efficient frontier ────────────────────────────────────────
    if not run_subprocess(
        "Module 3", "module3_frontier.py", "Efficient frontier chart"
    ):
        _fail("Pipeline", "Module 3 failed -- cannot continue")
        return

    # -- Concentration robustness check (portfolios now finalised) --
    print(f"\n{_bar()}")
    print("  ROBUSTNESS CHECK 3 of 3  --  Country Concentration")
    print(_bar())
    check_concentration(inputs)

    # -- Portfolio choice (if not collected upfront) --
    if not inputs.get("portfolio_choice"):
        portfolios_path = os.path.join(SCRIPT_DIR, "optimised_portfolios.xlsx")
        opts = _load_portfolio_opts(portfolios_path)
        if opts:
            print(f"\n{_bar()}")
            print("  SELECT TARGET PORTFOLIO FOR REBALANCING")
            print(_bar())
            inputs["portfolio_choice"] = _ask_portfolio_choice(opts)
        else:
            _warn(
                "Could not load portfolio options -- skipping Module 4."
            )

    # ── MODULE 4 -- Rebalancing (non-interactive) ─────────────────────────────
    if inputs.get("portfolio_choice"):
        rebalance_stats = run_module4(inputs, today_str)
    else:
        _warn("No portfolio choice available -- Module 4 skipped.")

    # ── MODULE 5 -- HTML report (optional) ────────────────────────────────────
    result5 = run_subprocess(
        "Module 5", "module5_report.py", "HTML portfolio report"
    )
    if result5 is None:
        print(
            "  [NOTE] module5_report.py was not found -- HTML report skipped."
        )
        print(
            "         Build module5_report.py to enable this step."
        )

    # ── End summary ───────────────────────────────────────────────────────────
    print_end_summary(inputs, rebalance_stats, today_str)

    # Clean up the run config file now that the pipeline is complete
    try:
        os.remove(CONFIG_PATH)
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  Pipeline interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        print(f"\n  [FAIL] Unexpected error: {exc}")
        print(
            "  If this persists, run each module individually to diagnose."
        )
        sys.exit(1)
