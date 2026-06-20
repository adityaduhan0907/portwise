#!/usr/bin/env python3
"""
run_all.py

Master pipeline runner for the portfolio model.
Collects ALL user inputs in one interactive block at the start, then
executes each module in sequence without further interruption.

Pipeline
--------
  module0_riskfree    -> fetch live risk-free rates
  module1_data        -> parameter estimation (returns, covariance, factor residuals)
  simulation_engine   -> Monte Carlo scenarios (Layer 4) -> simulated_returns.npz
  module2_optimiser   -> Layer 3 goals incl. Goal 3 CVaR (single-shot record)
  resampling_wrapper  -> Layer 7 Michaud resampling (B=500) -> resampled_portfolios.xlsx [CANONICAL]
  risk_evaluation     -> Layer 5 risk metrics (3 resampled goals + current holdings)
                         -> risk_evaluation_summary.json
  module3_frontier    -> efficient frontier chart
  module4_rebalance   -> rebalancing plan (targets = resampled weights; single-shot fallback)
  robustness_checks   -> Layer 6 robustness annotations (momentum / correlation /
                         sector) over the CHOSEN resampled weights -> robustness_warnings.json
  module5_report      -> HTML report (skipped gracefully if file not found)

NOTES
  - simulation_engine (Layer 4) must run after module1 and before every CVaR
    consumer (module2 Goal 3, Layer 7's CVaR path, Layer 5). The simulation_is_fresh()
    guard enforces this before module2.
  - resampled_portfolios.xlsx is CANONICAL for rebalancing; optimised_portfolios.xlsx
    is retained as the single-shot record. Layer 7's base_seed is read from
    simulated_returns.npz so the whole stochastic pipeline is reproducible from one seed.

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


def simulation_is_fresh():
    """
    Cheap guard before the first CVaR consumer: the Layer 4 scenarios must exist
    and be at least as new as Module 1's returns_stats.xlsx (i.e. built from the
    current Module 1 outputs, not a stale prior run).
    Returns (ok: bool, reason: str).
    """
    sim_path = os.path.join(SCRIPT_DIR, "simulated_returns.npz")
    m1_path  = os.path.join(SCRIPT_DIR, "returns_stats.xlsx")
    if not os.path.exists(sim_path):
        return False, "simulated_returns.npz not found"
    if os.path.exists(m1_path) and os.path.getmtime(sim_path) < os.path.getmtime(m1_path):
        return False, "simulated_returns.npz is older than returns_stats.xlsx (stale)"
    return True, "ok"


def simulation_seed(default=None):
    """
    Read the random_seed recorded in simulated_returns.npz. Layer 7 reuses it as its
    base_seed so the entire stochastic pipeline (simulation + resampling) is
    reproducible from one seed. Returns int, or `default` if unavailable/unset.
    """
    import numpy as np
    sim_path = os.path.join(SCRIPT_DIR, "simulated_returns.npz")
    try:
        data = np.load(sim_path, allow_pickle=True)
        s = int(data["random_seed"])
        return s if s >= 0 else default
    except Exception:
        return default


# ── Layer 7 -- resampling (reuses resampling_wrapper; logic unchanged) ──────────

def run_layer7(base_seed):
    """
    Michaud-resample all three goals (B=500) into resampled_portfolios.xlsx, the
    CANONICAL weights file. Drives resampling_wrapper's public functions in-process
    so the module's logic is untouched.
    """
    module_header("Layer 7", "Resampling (Michaud, B=500) -> resampled_portfolios.xlsx")
    try:
        import resampling_wrapper as rw
    except Exception as exc:
        _fail("Layer 7", f"cannot import resampling_wrapper: {exc}")
        return False

    if base_seed is None:
        base_seed = rw.DEFAULT_SEED
        print(f"  simulated_returns.npz had no usable seed -- using resampling "
              f"default base_seed={base_seed}")
    else:
        print(f"  Base seed (from simulated_returns.npz): {base_seed}")

    try:
        tickers, capm_mu = rw._load_universe()
        results = [rw.resample_goal(g, B=rw.DEFAULT_B, base_seed=base_seed)
                   for g in rw.GOALS]
        rw.write_resampled_xlsx(results, capm_mu, tickers)
        rw.save_per_iter(results, tickers)
        _ok("Layer 7")
        return True
    except Exception as exc:
        _fail("Layer 7", f"resampling failed: {exc}")
        return False


# ── Layer 5 -- risk evaluation (reuses risk_evaluation; logic unchanged) ───────

def _current_weights(inputs, tickers):
    """
    Convert the user's holdings (share counts) into portfolio weights over the
    scenario tickers, value-weighted using prices.xlsx. Indian holdings are
    converted INR->USD (via module4's FX helper) so a mixed book weights correctly;
    for a single-currency book the conversion cancels in the normalisation.
    Returns an ndarray aligned to `tickers`, or None if it cannot be derived.
    """
    import numpy as np
    holdings = inputs.get("holdings", [])
    if not holdings:
        return None
    try:
        import module4_rebalance as m4
        prices = m4.load_latest_prices(os.path.join(SCRIPT_DIR, "prices.xlsx"))
    except Exception:
        return None

    values  = {t: 0.0 for t in tickers}
    usd_inr = None
    found   = False
    for h in holdings:
        try:
            resolved, price = m4.lookup_price(h["ticker"], prices)
        except Exception:
            resolved, price = None, None
        if resolved is None or resolved not in values or price is None:
            continue
        val = float(h["shares"]) * float(price)
        if m4.is_indian(resolved):
            if usd_inr is None:
                usd_inr = m4.fetch_usd_inr_rate()
            if usd_inr:
                val /= usd_inr
        values[resolved] += val
        found = True

    if not found:
        return None
    w = np.array([values[t] for t in tickers], dtype=float)
    if w.sum() <= 0:
        return None
    return w / w.sum()


def run_layer5(inputs, base_seed):
    """
    Evaluate risk metrics over the Layer 4 scenarios for each of the three CANONICAL
    resampled goal portfolios PLUS the user's current portfolio, and write
    risk_evaluation_summary.json keyed by portfolio. Drives risk_evaluation's public
    functions in-process; its metric logic is untouched.
    """
    module_header("Layer 5", "Risk evaluation over scenarios -> risk_evaluation_summary.json")
    try:
        import risk_evaluation as rev
    except Exception as exc:
        _fail("Layer 5", f"cannot import risk_evaluation: {exc}")
        return False

    try:
        returns, tickers = rev.load_scenarios(rev.SIM_PATH)
    except Exception as exc:
        _fail("Layer 5", f"could not load Layer 4 scenarios: {exc}")
        return False

    resampled_path = os.path.join(SCRIPT_DIR, "resampled_portfolios.xlsx")
    if not os.path.exists(resampled_path):
        _fail("Layer 5", "resampled_portfolios.xlsx not found -- Layer 7 must run first")
        return False

    seed = base_seed if base_seed is not None else 0
    portfolios = {}
    for sheet in ("Minimum Variance", "Max Risk-Adjusted", "Tail-Risk CVaR"):
        try:
            w = rev.load_weights_from_xlsx(sheet, tickers, path=resampled_path)
            portfolios[sheet] = rev.evaluate_risk(
                returns, w, tickers, random_seed=seed, label=sheet
            )
        except Exception as exc:
            _warn(f"Layer 5: could not evaluate '{sheet}' ({exc})")

    w_cur = _current_weights(inputs, tickers)
    if w_cur is not None:
        portfolios["Current Portfolio"] = rev.evaluate_risk(
            returns, w_cur, tickers, random_seed=seed, label="Current Portfolio"
        )
    else:
        _warn("Layer 5: could not derive current-portfolio weights from holdings; "
              "summary will omit the 'Current Portfolio' entry.")

    summary = {
        "scenarios_seed":  base_seed,
        "n_scenarios":     int(returns.shape[0]),
        "tickers":         list(tickers),
        "weights_source":  "resampled_portfolios.xlsx (canonical)",
        "portfolios":      portfolios,        # keyed by portfolio name
    }
    try:
        with open(rev.SUMMARY_PATH, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"  Evaluated: {', '.join(portfolios.keys())}")
        _ok("Layer 5")
        return True
    except Exception as exc:
        _fail("Layer 5", f"could not write risk_evaluation_summary.json: {exc}")
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

    prices_path = os.path.join(SCRIPT_DIR, "prices.xlsx")

    # Rebalancing targets come from the CANONICAL resampled weights; the single-shot
    # optimised file is the defensive fallback (kept as a record, never deleted).
    resampled_path = os.path.join(SCRIPT_DIR, "resampled_portfolios.xlsx")
    single_path    = os.path.join(SCRIPT_DIR, "optimised_portfolios.xlsx")
    if os.path.exists(resampled_path):
        portfolios_path = resampled_path
        weights_source  = "resampled_portfolios.xlsx (canonical, Layer 7)"
    else:
        portfolios_path = single_path
        weights_source  = "optimised_portfolios.xlsx (single-shot FALLBACK)"
        _warn("resampled_portfolios.xlsx not found -- falling back to single-shot "
              "optimised_portfolios.xlsx for rebalancing targets.")
    print(f"  Rebalancing target weights source: {weights_source}")

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

    # Ensure a portfolio choice is available (defensive: if the canonical resampled
    # file is unreadable, fall back to the single-shot record rather than crashing).
    try:
        portfolio_opts = m4.load_portfolio_options(portfolios_path)
        if not portfolio_opts:
            raise ValueError("no portfolios parsed")
    except Exception as exc:
        if portfolios_path != single_path and os.path.exists(single_path):
            _warn(f"Could not read {os.path.basename(portfolios_path)} ({exc}); "
                  "falling back to optimised_portfolios.xlsx (single-shot).")
            portfolios_path = single_path
            portfolio_opts  = m4.load_portfolio_options(single_path)
        else:
            _fail("Module 4", f"could not read portfolio weights: {exc}")
            return None
    choice = inputs.get("portfolio_choice")

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
        "simulated_returns.npz",
        "optimised_portfolios.xlsx",
        "resampled_portfolios.xlsx",
        "risk_evaluation_summary.json",
        "efficient_frontier.png",
        f"rebalancing_plan_{date_stamp}.xlsx",
        "robustness_warnings.json",
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
    print("  Pipeline: module0 -> module1 -> simulation -> module2 -> resample(L7)"
          " -> riskeval(L5) -> module3 -> module4 -> robustness(L6) -> module5")

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

    # ── LAYER 4 -- Monte Carlo simulation ─────────────────────────────────────
    #   Hard dependency: produces simulated_returns.npz, which every CVaR consumer
    #   needs. Runs AFTER Module 1 (uses its betas / factor covariance / residual
    #   pools and the risk-free rate) and BEFORE the first CVaR consumer (Module 2
    #   Goal 3). Placed after the Module 1 retry loop so it reflects the FINAL
    #   ticker set.
    sim_result = run_subprocess(
        "Layer 4", "simulation_engine.py",
        "Monte Carlo scenario engine -> simulated_returns.npz",
    )
    if sim_result is False:
        _fail("Pipeline", "Layer 4 simulation failed -- CVaR consumers (Module 2 "
              "Goal 3) cannot run without simulated_returns.npz")
        return

    # Guard: scenarios must exist and be newer than Module 1's outputs before any
    # CVaR consumer runs (mirrors module2's own "run simulation first" guard).
    fresh, why = simulation_is_fresh()
    if not fresh:
        _fail("Pipeline",
              f"Cannot proceed to CVaR consumers: {why}. "
              "Run simulation_engine.py (Layer 4) after Module 1 and before Module 2.")
        return

    # ── MODULE 2 -- Optimisation (Layer 3 goals + Goal 3 CVaR over Layer 4) ────
    if not run_subprocess(
        "Module 2", "module2_optimiser.py", "Mean-variance optimisation"
    ):
        _fail("Pipeline", "Module 2 failed -- cannot continue")
        return

    # ── LAYER 7 -- Resampling (CANONICAL weights) ─────────────────────────────
    #   Reuses the simulation's recorded seed so the whole stochastic pipeline is
    #   reproducible from one seed. Adds ~3-4 min (B=500 x three goals).
    base_seed = simulation_seed()
    if not run_layer7(base_seed):
        _fail("Pipeline", "Layer 7 resampling failed -- cannot produce canonical weights")
        return

    # ── LAYER 5 -- Risk evaluation (resampled goals + current holdings) ───────
    if not run_layer5(inputs, base_seed):
        _warn("Layer 5 risk evaluation failed -- continuing (rebalancing unaffected).")

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

    # ── LAYER 6 -- Robustness checks (non-blocking annotations) ───────────────
    #   Runs after the portfolio is chosen and stabilised (Layer 7 weights +
    #   Layer 5 risk). It reads the CHOSEN resampled weights, so the portfolio
    #   choice must be persisted to run_config.json first (it may have been
    #   selected after Module 3, leaving the on-disk config stale). Produces
    #   robustness_warnings.json, which Module 5 reads. Never halts the pipeline.
    write_run_config(inputs)
    run_subprocess(
        "Layer 6", "robustness_checks.py",
        "Robustness checks (momentum / correlation / sector) -> robustness_warnings.json",
    )

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
