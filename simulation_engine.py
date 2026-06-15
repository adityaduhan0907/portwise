#!/usr/bin/env python3
"""
simulation_engine.py  —  Layer 4 (Monte Carlo Scenario Engine)

Builds the scenario table that Layers 5/6 (e.g. CVaR / tail-risk) consume.
It does NOT re-fetch any market data: every input comes from existing handoffs.

INPUTS (read-only handoffs):
  - factor_history.json   (Module 0): per-market cleaned monthly factor returns
        plus the monthly RF column  (US: Mkt-RF/SMB/HML/RF; India: MF/SMB/HML/WML/RF).
  - factor_residuals.json (Module 1 / Part A): per-asset category betas (aligned to
        factors BY NAME), the residual SHOCK pool, residual variance, and the
        per-market factor covariance F.  Betas are already in this handoff, so
        Layer 4 needs no re-fetch and no change to Part A.

MODEL (per scenario s):
  - Draw ONE factor month k at random (with replacement) from the portfolio's
    market factor history. The SAME k is shared by every asset in that scenario
    -- that shared draw is what generates factor-driven co-movement. It supplies
    both the factor vector factor_f[k] and that month's RF[k].
  - For each asset i, INDEPENDENTLY draw one residual shock from asset i's own
    pool (with replacement) -- independent across assets and of the factor draw.
  - total_return[s, i] = RF[k] + sum_f( beta[i, f] * factor_f[k] ) + residual[s, i]

  US assets use the 3 US factors from US history; India assets the 4 India factors
  from India history.  A MIXED US+India portfolio is the deferred cross-market
  case: as a stopgap we draw factor months INDEPENDENTLY per market (no date
  alignment) and emit a clear NOTE.

OUTPUT:
  - In-memory: a (n_scenarios x n_assets) DataFrame of simulated MONTHLY total
    returns, columns labelled by ticker (returned for Layers 5/6).
  - On disk : simulated_returns.npz  (compact: returns matrix + ticker labels).

Run standalone for the verification (shape, per-asset stats, and a consistency
check of the simulated covariance against the factor model B F Bᵀ + D).
"""

import json
import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
SCRIPT_DIR            = os.path.dirname(os.path.abspath(__file__))
FACTOR_HISTORY_PATH   = os.path.join(SCRIPT_DIR, "factor_history.json")
FACTOR_RESIDUALS_PATH = os.path.join(SCRIPT_DIR, "factor_residuals.json")
SIM_OUTPUT_PATH       = os.path.join(SCRIPT_DIR, "simulated_returns.npz")
N_SCENARIOS           = 10_000


# ── Handoff loading ─────────────────────────────────────────────────────────────

def _load_handoffs(history_path=FACTOR_HISTORY_PATH, residuals_path=FACTOR_RESIDUALS_PATH):
    """Load the Module 0 and Part A handoffs (raises if either is missing)."""
    with open(history_path, encoding="utf-8") as f:
        history = json.load(f)
    with open(residuals_path, encoding="utf-8") as f:
        residuals = json.load(f)
    return history, residuals


def _market_factor_arrays(history, market):
    """
    Return (factor_names, factor_matrix (M x nf), rf_vector (M,)) for a market,
    using the factor-name order from the factor history. Factors and RF are
    already decimals.
    """
    md = history["markets"][market]
    names = [c for c in md["columns"] if c != "RF"]          # factor names, by name
    factor_matrix = np.column_stack([np.asarray(md["factor_returns"][c], dtype=float)
                                     for c in names])
    rf_vector = np.asarray(md["monthly_rf"], dtype=float)
    return names, factor_matrix, rf_vector


# ── Simulation engine ───────────────────────────────────────────────────────────

def simulate_scenarios(tickers=None, n_scenarios=N_SCENARIOS, random_seed=None,
                       history=None, residuals=None, persist=True, verbose=True):
    """
    Build the (n_scenarios x n_assets) table of simulated monthly total returns.

    tickers      : assets to simulate; defaults to every asset in the Part A
                   handoff (the portfolio Module 1 processed). Must be handoff keys
                   (resolved symbols, e.g. 'RELIANCE.NS').
    random_seed  : optional int for reproducibility (default None -> fresh entropy).
    history/residuals : pre-loaded handoffs (loaded from disk if None).
    persist      : also write simulated_returns.npz.

    Returns a DataFrame (index = scenario, columns = tickers).
    """
    if history is None or residuals is None:
        history, residuals = _load_handoffs()

    assets = residuals["assets"]
    if tickers is None:
        tickers = list(assets.keys())
    missing = [t for t in tickers if t not in assets]
    if missing:
        raise KeyError(f"Tickers not in Part A handoff (re-run module1): {missing}")

    rng = np.random.default_rng(random_seed)
    S   = int(n_scenarios)
    n   = len(tickers)

    # Group assets by market; each market shares one factor-month draw per scenario.
    markets = {}
    for t in tickers:
        markets.setdefault(assets[t]["market"], []).append(t)

    if len(markets) > 1 and verbose:
        print(f"  NOTE: mixed-market portfolio {sorted(markets)} -- using the deferred "
              "stopgap:\n        independent per-market factor-month draws (no "
              "cross-market date alignment).")

    total = np.empty((S, n), dtype=float)
    col_of = {t: i for i, t in enumerate(tickers)}

    for market, mkt_tickers in markets.items():
        names, factor_matrix, rf_vector = _market_factor_arrays(history, market)
        M = factor_matrix.shape[0]

        # ONE shared factor-month draw for every asset in this market.
        k = rng.integers(0, M, size=S)                       # (S,)
        drawn_factors = factor_matrix[k]                     # (S, nf)
        drawn_rf      = rf_vector[k]                          # (S,)

        # Betas aligned to the factor names BY NAME (same order as F / Part A).
        B = np.array([[float(assets[t]["betas"].get(f, 0.0)) for f in names]
                      for t in mkt_tickers])                 # (n_mkt, nf)

        systematic = drawn_factors @ B.T                     # (S, n_mkt)

        for j, t in enumerate(mkt_tickers):
            pool = np.asarray(assets[t]["residual_shocks"], dtype=float)
            if pool.size == 0:                               # degenerate guard
                resid = np.zeros(S)
            else:
                resid = pool[rng.integers(0, pool.size, size=S)]   # independent draw
            total[:, col_of[t]] = drawn_rf + systematic[:, j] + resid

    df = pd.DataFrame(total, columns=tickers)
    df.index.name = "scenario"

    if persist:
        np.savez_compressed(
            SIM_OUTPUT_PATH,
            returns=total,
            tickers=np.array(tickers, dtype=object),
            n_scenarios=S,
            random_seed=(-1 if random_seed is None else random_seed),
        )
        if verbose:
            print(f"  Persisted -> {SIM_OUTPUT_PATH}  ({S} x {n})")

    return df


# ── Consistency model: factor covariance B F Bᵀ + D ─────────────────────────────

def factor_model_covariance(tickers, residuals=None):
    """
    Build the factor-model MONTHLY covariance B F Bᵀ + D from the SAME inputs the
    simulation uses (Part A's F, category betas, residual variances). This is the
    correct yardstick for the engine -- NOT the optimizer's Ledoit-Wolf matrix --
    because the simulation is factor-based by construction.

    Same-market block i,j : b_iᵀ F_market b_j.  Cross-market pairs are 0 (the
    stopgap draws factor months independently per market).  Diagonal adds the
    asset's residual variance.  Returns a labelled DataFrame.
    """
    if residuals is None:
        _, residuals = _load_handoffs()
    assets = residuals["assets"]
    fcov   = residuals["factor_covariance"]

    # Per-market F and factor order.
    F_by_market = {m: (v["factors"], np.array(v["matrix"], dtype=float))
                   for m, v in fcov.items()}

    n = len(tickers)
    sigma = np.zeros((n, n))
    for i, ti in enumerate(tickers):
        mi = assets[ti]["market"]
        names_i, Fi = F_by_market[mi]
        bi = np.array([float(assets[ti]["betas"].get(f, 0.0)) for f in names_i])
        for j, tj in enumerate(tickers):
            if assets[tj]["market"] != mi:
                continue                                      # cross-market -> 0
            bj = np.array([float(assets[tj]["betas"].get(f, 0.0)) for f in names_i])
            sigma[i, j] = bi @ Fi @ bj
        rv = assets[ti]["residual_variance"]
        sigma[i, i] += float(rv) if rv is not None else 0.0

    return pd.DataFrame(sigma, index=tickers, columns=tickers)


# ── Verification ────────────────────────────────────────────────────────────────

def _verify(tickers=None, seed=12345):
    W = 78
    history, residuals = _load_handoffs()
    if tickers is None:
        tickers = list(residuals["assets"].keys())

    markets = sorted({residuals["assets"][t]["market"] for t in tickers})
    print(f"\n{'='*W}")
    print("  Layer 4 -- Monte Carlo Scenario Engine  (VERIFICATION)")
    print(f"  Assets   : {', '.join(tickers)}")
    print(f"  Market(s): {', '.join(markets)}   |   Scenarios: {N_SCENARIOS}   |   seed={seed}")
    print(f"{'='*W}")

    sim = simulate_scenarios(tickers, n_scenarios=N_SCENARIOS, random_seed=seed,
                             history=history, residuals=residuals)

    print(f"\n  Matrix shape : {sim.shape}   (expected {N_SCENARIOS} x {len(tickers)})")

    # Per-asset distribution summary.
    print(f"\n  Per-asset simulated MONTHLY return distribution:")
    print(f"    {'Ticker':<14}{'mean':>10}{'std':>10}{'p5':>10}{'p50':>10}{'p95':>10}")
    print(f"    {'-'*14}{'-'*10}{'-'*10}{'-'*10}{'-'*10}{'-'*10}")
    for t in tickers:
        col = sim[t].values
        p5, p50, p95 = np.percentile(col, [5, 50, 95])
        print(f"    {t:<14}{col.mean():>10.4f}{col.std(ddof=1):>10.4f}"
              f"{p5:>10.4f}{p50:>10.4f}{p95:>10.4f}")

    # Consistency vs the factor model B F Bᵀ + D (the by-construction yardstick).
    model = factor_model_covariance(tickers, residuals)
    sim_cov = sim.cov()

    print(f"\n  CONSISTENCY (a) -- per-asset VARIANCE: simulated vs model (B F B^T + D) diag")
    print(f"    {'Ticker':<14}{'sim var':>14}{'model var':>14}{'rel err %':>12}")
    print(f"    {'-'*14}{'-'*14}{'-'*14}{'-'*12}")
    for t in tickers:
        sv, mv = sim_cov.loc[t, t], model.loc[t, t]
        rel = 100.0 * (sv - mv) / mv if mv else float("nan")
        print(f"    {t:<14}{sv:>14.6f}{mv:>14.6f}{rel:>12.2f}")

    print(f"\n  CONSISTENCY (b) -- COVARIANCE matrix: simulated vs model (side by side)")
    with pd.option_context("display.float_format", lambda x: f"{x:9.6f}"):
        print("\n    Simulated covariance:")
        print("    " + sim_cov.round(6).to_string().replace("\n", "\n    "))
        print("\n    Model covariance  B F B^T + D:")
        print("    " + model.round(6).to_string().replace("\n", "\n    "))

    max_abs = float((sim_cov.values - model.values).__abs__().max())
    denom = np.maximum(np.abs(model.values), 1e-12)
    max_rel = float((np.abs(sim_cov.values - model.values) / denom).max())
    print(f"\n  Max |sim - model| entry : {max_abs:.6e}")
    print(f"  Max relative difference : {max_rel*100:.2f}%   "
          "(expect small -- sampling error over 10,000 draws)")
    print(f"{'='*W}\n")


def main():
    _verify()


if __name__ == "__main__":
    main()
