#!/usr/bin/env python3
"""
risk_evaluation.py  —  Layer 5 (Risk Evaluation)

Runs NO new simulation. It takes a weight vector and the Layer 4 scenario matrix
(simulated_returns.npz, produced by simulation_engine.py) and READS OFF risk
metrics for the resulting portfolio. Does not modify any existing module.

INPUTS:
  - Layer 4 scenarios: 10,000 x n_assets matrix of per-asset simulated MONTHLY
    returns (from simulated_returns.npz, or pass the in-memory DataFrame).
  - A weight vector per ticker (dict {ticker: weight} or array aligned to the
    scenario columns). Verification reads BOTH the Min Variance and the Max
    Risk-Adjusted weights from optimised_portfolios.xlsx.

PIPELINE:
  Step 1  portfolio_return[s] = sum_i( w_i * asset_return[s, i] )   (10,000-vector)
  Step 2  read off, each labelled with its horizon:
    - Expected return (simulated) : monthly mean, ANNUALISED (mean * 12)
    - Volatility                  : monthly std,  ANNUALISED (std * sqrt(12))
    - VaR(95%)                    : 5th-percentile MONTHLY outcome (a loss)
    - CVaR(95%) / tail loss       : mean of outcomes <= 5th pct (MONTHLY loss)
    - Chance of a large loss      : P(monthly return < threshold), default -0.10
    - Worst-case drop (drawdown)  : sample n_paths sequences of `horizon` monthly
                                    returns from the scenario pool, take each
                                    path's max peak-to-trough drawdown; report
                                    MEDIAN and 95th-pct (worst). Defaults
                                    1,000 paths x 12 months. (Independent monthly
                                    draws strung into sequences -- this is NOT the
                                    historical stress test, which is separate.)

OUTPUT: a labelled metrics dict (in-memory); optional compact JSON summary.
"""

import json
import os

import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
SCRIPT_DIR         = os.path.dirname(os.path.abspath(__file__))
SIM_PATH           = os.path.join(SCRIPT_DIR, "simulated_returns.npz")
OPTIMISED_PATH     = os.path.join(SCRIPT_DIR, "optimised_portfolios.xlsx")
SUMMARY_PATH       = os.path.join(SCRIPT_DIR, "risk_evaluation_summary.json")

TRADING_MONTHS     = 12
LARGE_LOSS_DEFAULT = -0.10     # "large monthly loss" threshold
DD_PATHS_DEFAULT   = 1_000
DD_HORIZON_DEFAULT = 12


# ── Input loading ───────────────────────────────────────────────────────────────

def load_scenarios(source=SIM_PATH):
    """
    Load the Layer 4 scenario matrix. `source` may be a path to the .npz, or an
    in-memory DataFrame (rows = scenarios, cols = tickers).
    Returns (returns ndarray [S x n], tickers list).
    """
    if isinstance(source, pd.DataFrame):
        return source.values.astype(float), list(source.columns)
    data = np.load(source, allow_pickle=True)
    return data["returns"].astype(float), list(data["tickers"])


def align_weights(weights, tickers):
    """
    Align a weight spec to the scenario column order.
    `weights` may be a {ticker: weight} dict or a sequence aligned to `tickers`.
    Missing tickers default to 0. Re-normalises to sum 1 if the total is positive.
    """
    if isinstance(weights, dict):
        w = np.array([float(weights.get(t, 0.0)) for t in tickers])
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != len(tickers):
            raise ValueError(f"weight length {w.shape[0]} != n_tickers {len(tickers)}")
    total = w.sum()
    if total > 0:
        w = w / total
    return w


def load_weights_from_xlsx(sheet_name, tickers, path=OPTIMISED_PATH):
    """
    Read a goal's weights from optimised_portfolios.xlsx. Picks the rows whose
    'Stock' is one of `tickers` (ignores the spacer / summary rows), converts the
    'Weight (%)' column to decimals, and aligns to the scenario column order.
    """
    df = pd.read_excel(path, sheet_name=sheet_name)
    wmap = {}
    for _, row in df.iterrows():
        stock = row.get("Stock")
        if stock in tickers:
            wmap[stock] = float(row["Weight (%)"]) / 100.0
    return align_weights(wmap, tickers)


# ── Metrics ─────────────────────────────────────────────────────────────────────

def _sample_paths(pool, n_paths, horizon, rng):
    """
    String independent monthly draws from `pool` into an (n_paths, horizon) matrix
    of monthly returns. Shared helper for the drawdown metric and the annual
    VaR/CVaR distribution so both use the SAME path construction.
    """
    return pool[rng.integers(0, pool.shape[0], size=(n_paths, horizon))]


def _max_drawdown_from_paths(draws):
    """
    (median, p95) of per-path max peak-to-trough drawdown for a matrix of monthly
    returns (positive fractions, e.g. 0.25 = -25%).
    """
    wealth = np.cumprod(1.0 + draws, axis=1)            # (n_paths, horizon)
    running_peak = np.maximum.accumulate(wealth, axis=1)
    drawdowns = wealth / running_peak - 1.0             # <= 0
    worst_per_path = -drawdowns.min(axis=1)             # positive magnitude
    median_dd = float(np.median(worst_per_path))
    p95_dd    = float(np.percentile(worst_per_path, 95))
    return median_dd, p95_dd


def _annual_var_cvar_from_paths(draws, alpha=0.95):
    """
    Annual VaR / CVaR from a matrix of monthly returns. Compound each path's
    `horizon` monthly returns into ONE annual return, then read off the
    (1-alpha) tail of that annual distribution:
        VaR(alpha)  = (1-alpha)*100-th percentile annual return (a loss)
        CVaR(alpha) = mean of the annual returns at/below that percentile
    Returns (var_return, cvar_return) as SIGNED annual returns (losses are < 0).
    """
    annual = np.prod(1.0 + draws, axis=1) - 1.0         # (n_paths,) annual returns
    var_q  = float(np.percentile(annual, (1.0 - alpha) * 100.0))
    tail   = annual[annual <= var_q]
    cvar_q = float(tail.mean()) if tail.size else var_q
    return var_q, cvar_q


def evaluate_risk(scenarios, weights, tickers=None, *,
                  large_loss_threshold=LARGE_LOSS_DEFAULT,
                  dd_paths=DD_PATHS_DEFAULT, dd_horizon=DD_HORIZON_DEFAULT,
                  random_seed=None, label="portfolio"):
    """
    Evaluate Layer 5 risk metrics for one weight vector against the scenario pool.

    scenarios : ndarray [S x n] or a DataFrame (cols = tickers).
    weights   : {ticker: weight} dict or sequence aligned to the scenario columns.
    Returns a labelled metrics dict (values + their horizon labels).
    """
    if isinstance(scenarios, pd.DataFrame):
        returns, tickers = scenarios.values.astype(float), list(scenarios.columns)
    else:
        returns = np.asarray(scenarios, dtype=float)
        if tickers is None:
            tickers = [f"asset_{i}" for i in range(returns.shape[1])]

    w = align_weights(weights, tickers)

    # Step 1 -- portfolio monthly return per scenario.
    port = returns @ w                                  # (S,)
    S = port.shape[0]

    # Step 2 -- read off metrics.
    mean_m = float(port.mean())
    std_m  = float(port.std(ddof=1))

    var_q  = float(np.percentile(port, 5))              # 5th-pct monthly outcome
    tail   = port[port <= var_q]
    cvar_q = float(tail.mean()) if tail.size else var_q

    p_large = float(np.mean(port < large_loss_threshold))

    # Build the path matrix ONCE (same construction the drawdown metric uses):
    # n_paths sequences of `dd_horizon` independent monthly draws. The annual
    # VaR/CVaR compounds each path into an annual return; drawdown reads the
    # intra-path peak-to-trough off the same matrix.
    rng   = np.random.default_rng(random_seed)
    draws = _sample_paths(port, dd_paths, dd_horizon, rng)
    dd_median, dd_p95 = _max_drawdown_from_paths(draws)
    var_a, cvar_a     = _annual_var_cvar_from_paths(draws, alpha=0.95)

    return {
        "label":            label,
        "weights":          {t: round(float(wi), 6) for t, wi in zip(tickers, w)},
        "n_scenarios":      S,
        "expected_return": {
            "monthly_mean":      round(mean_m, 6),
            "annualized":        round(mean_m * TRADING_MONTHS, 6),
            "horizon":           "monthly mean; annualised = mean x 12",
        },
        "volatility": {
            "monthly_std":       round(std_m, 6),
            "annualized":        round(std_m * np.sqrt(TRADING_MONTHS), 6),
            "horizon":           "monthly std; annualised = std x sqrt(12)",
        },
        # VaR/CVaR are now reported on the ANNUAL return distribution (12 monthly
        # returns compounded across many paths), NOT monthly x 12. The monthly
        # figures are retained for sanity-checking the conversion.
        "var_95": {
            "annual_return":     round(var_a, 6),        # signed annual (a loss)
            "annual_loss":       round(-var_a, 6),       # positive loss magnitude
            "monthly_return":    round(var_q, 6),        # signed monthly (legacy)
            "monthly_loss":      round(-var_q, 6),
            "horizon":           ("annual; 5th-pct of compounded 12-month returns "
                                  f"({dd_paths} paths). monthly_* = legacy 5th-pct "
                                  "monthly outcome."),
        },
        "cvar_95": {
            "annual_return":     round(cvar_a, 6),       # signed annual (< VaR)
            "annual_loss":       round(-cvar_a, 6),
            "monthly_return":    round(cvar_q, 6),       # signed monthly (legacy)
            "monthly_loss":      round(-cvar_q, 6),
            "horizon":           ("annual; mean of worst 5% compounded 12-month "
                                  f"returns ({dd_paths} paths). monthly_* = legacy."),
        },
        "chance_large_loss": {
            "threshold":         large_loss_threshold,
            "probability":       round(p_large, 6),
            "horizon":           f"monthly; P(return < {large_loss_threshold})",
        },
        "max_drawdown": {
            "median":            round(dd_median, 6),
            "p95_worst":         round(dd_p95, 6),
            "paths":             dd_paths,
            "horizon":           f"{dd_horizon}-month paths ({dd_paths} sampled sequences)",
        },
    }


# ── Display / persistence ───────────────────────────────────────────────────────

def print_metrics(m):
    W = 64
    print(f"\n{'='*W}")
    print(f"  RISK EVALUATION -- {m['label']}")
    print(f"{'='*W}")
    nz = {t: w for t, w in m["weights"].items() if abs(w) > 1e-6}
    print("  Weights : " + ", ".join(f"{t} {w*100:.2f}%" for t, w in nz.items()))
    print(f"  Scenarios: {m['n_scenarios']}")
    er, vol = m["expected_return"], m["volatility"]
    print(f"\n  Expected return (annualised) : {er['annualized']*100:>8.2f}%   "
          f"[{er['monthly_mean']*100:.2f}%/mo]")
    print(f"  Volatility      (annualised) : {vol['annualized']*100:>8.2f}%   "
          f"[{vol['monthly_std']*100:.2f}%/mo]")
    print(f"  VaR  95% (annual loss)       : {m['var_95']['annual_loss']*100:>8.2f}%   "
          f"[monthly {m['var_95']['monthly_loss']*100:.2f}%]")
    print(f"  CVaR 95% (annual loss)       : {m['cvar_95']['annual_loss']*100:>8.2f}%   "
          f"[monthly {m['cvar_95']['monthly_loss']*100:.2f}%]")
    cl = m["chance_large_loss"]
    print(f"  P(loss worse than {cl['threshold']*100:.0f}%/mo)    : "
          f"{cl['probability']*100:>8.2f}%")
    dd = m["max_drawdown"]
    print(f"  Max drawdown ({dd['horizon']}):")
    print(f"      median                   : {dd['median']*100:>8.2f}%")
    print(f"      95th-pct (worst)         : {dd['p95_worst']*100:>8.2f}%")
    print(f"{'='*W}")


def persist_summary(metrics_list, path=SUMMARY_PATH):
    """Write a compact JSON summary for the evaluated portfolios."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"portfolios": metrics_list}, f, indent=2)
    return path


# ── Verification ────────────────────────────────────────────────────────────────

def _verify(seed=12345):
    returns, tickers = load_scenarios(SIM_PATH)
    print(f"\n  Layer 4 scenarios : {returns.shape}  cols={tickers}")

    goals = [
        ("Min Variance",      "Minimum Variance"),
        ("Max Risk-Adjusted", "Max Risk-Adjusted"),
    ]

    results = []
    for label, sheet in goals:
        w = load_weights_from_xlsx(sheet, tickers)
        m = evaluate_risk(returns, w, tickers, random_seed=seed, label=label)
        print_metrics(m)
        results.append(m)

    # Sanity checks comparing the two portfolios.
    mv, ra = results          # Min Variance, Max Risk-Adjusted
    print(f"\n{'='*64}")
    print("  SANITY CHECKS")
    print(f"{'='*64}")
    c1 = mv["volatility"]["annualized"] < ra["volatility"]["annualized"]
    print(f"  Min Variance vol < Max Risk-Adjusted vol : {c1}  "
          f"({mv['volatility']['annualized']*100:.2f}% vs {ra['volatility']['annualized']*100:.2f}%)")
    for m in results:
        var_lt_mean  = m["var_95"]["monthly_return"]  < m["expected_return"]["monthly_mean"]
        cvar_lt_var  = m["cvar_95"]["monthly_return"] < m["var_95"]["monthly_return"]
        print(f"  [{m['label']:<17}] VaR < mean : {var_lt_mean}   |   "
              f"CVaR < VaR : {cvar_lt_var}")
    print(f"{'='*64}")

    persist_summary(results)
    print(f"\n  Saved summary -> {SUMMARY_PATH}\n")
    return results


def main():
    if not os.path.exists(SIM_PATH):
        print(f"  ERROR: {SIM_PATH} not found. Run simulation_engine.py first.")
        return
    _verify()


if __name__ == "__main__":
    main()
