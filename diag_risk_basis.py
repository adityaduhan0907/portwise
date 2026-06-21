#!/usr/bin/env python3
"""
diag_risk_basis.py  --  READ-ONLY diagnostic (throwaway, NOT a pipeline step)

Prints risk and return figures on BOTH the optimizer's basis (Ledoit-Wolf
covariance + factor-model expected returns) and the simulation's basis (the
Monte-Carlo scenarios), side by side, so the divergence is plain to see.

It changes NO pipeline logic and writes nothing. It only reads the artifacts the
last run (India 3-set) already produced:

  resampled_portfolios.xlsx     three goal sheets -> per-goal weights
  risk_evaluation_summary.json  Layer 5's "Current Portfolio" weights (same vector
                                Layer 5 uses; falls back to reconstructing from
                                run_config.json holdings via run_all._current_weights)
  returns_stats.xlsx
      "Annualised Cov"          optimizer covariance Sigma_LW (already annualized)
      "CAPM Returns"            factor-model expected returns capm_mu
  simulated_returns.npz         scenarios, key 'returns' (S x n monthly total returns)

What to look for
  * Minimum Variance should have the LOWEST Ledoit-Wolf vol among the goals.
  * How far simulation vol diverges from the Ledoit-Wolf vol.
  * The gap between the sane factor-model E[r] and the inflated simulation E[r].
  * Table B: which stock's residual drives the inflation (sim vol/return wildly
    above its factor-model figure).

Usage:  python diag_risk_basis.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
RESAMPLED_PATH = os.path.join(SCRIPT_DIR, "resampled_portfolios.xlsx")
RETURNS_PATH   = os.path.join(SCRIPT_DIR, "returns_stats.xlsx")
RISK_PATH      = os.path.join(SCRIPT_DIR, "risk_evaluation_summary.json")
CONFIG_PATH    = os.path.join(SCRIPT_DIR, "run_config.json")
SIM_PATH       = os.path.join(SCRIPT_DIR, "simulated_returns.npz")

GOAL_SHEETS = ["Minimum Variance", "Max Risk-Adjusted", "Tail-Risk CVaR"]

_SKIP = ("portfolio ",)   # resampled stat rows start with "Portfolio ..."


def _pct(x, dp=1):
    try:
        return f"{float(x) * 100:.{dp}f}%"
    except (TypeError, ValueError):
        return "  n/a"


# ── Load the shared basis (tickers, Sigma_LW, capm_mu, scenarios) ───────────────

def load_basis():
    sim = np.load(SIM_PATH, allow_pickle=True)
    sim_tickers = [str(t) for t in sim["tickers"]]
    returns = np.asarray(sim["returns"], dtype=float)        # S x n

    cov_df = pd.read_excel(RETURNS_PATH, sheet_name="Annualised Cov", index_col=0)
    capm_df = pd.read_excel(RETURNS_PATH, sheet_name="CAPM Returns")

    # Canonical order = simulation column order; align cov + capm to it.
    tickers = sim_tickers
    Sigma = cov_df.loc[tickers, tickers].to_numpy(dtype=float)
    capm_map = dict(zip(capm_df["Ticker"].astype(str),
                        capm_df["CAPM_Expected_Return"].astype(float)))
    capm_mu = np.array([capm_map.get(t, np.nan) for t in tickers], dtype=float)
    return tickers, returns, Sigma, capm_mu


# ── Weights per portfolio (aligned to `tickers`, summing to 1) ──────────────────

def _weights_from_sheet(sheet, tickers):
    df = pd.read_excel(RESAMPLED_PATH, sheet_name=sheet)
    w = {}
    for _, row in df.iterrows():
        s = str(row.get("Stock", "")).strip()
        if not s or s.lower() == "nan" or s.lower().startswith(_SKIP):
            continue
        try:
            val = float(row.get("Weight (%)"))
        except (TypeError, ValueError):
            continue
        if val > 0:
            w[s] = val / 100.0
    vec = np.array([w.get(t, 0.0) for t in tickers], dtype=float)
    tot = vec.sum()
    return vec / tot if tot > 0 else vec


def _current_weights(tickers):
    """Layer 5's current-portfolio weights: prefer the persisted Layer 5 output
    (identical vector), else reconstruct from holdings via run_all._current_weights."""
    # Primary: the exact vector Layer 5 wrote last run.
    try:
        ports = json.load(open(RISK_PATH, encoding="utf-8")).get("portfolios", {})
        wmap = ports.get("Current Portfolio", {}).get("weights")
        if wmap:
            vec = np.array([float(wmap.get(t, 0.0)) for t in tickers], dtype=float)
            if vec.sum() > 0:
                return vec / vec.sum(), "risk_evaluation_summary.json (Layer 5)"
    except Exception:
        pass
    # Fallback: reconstruct from run_config.json holdings via Layer 5's own helper.
    try:
        cfg = json.load(open(CONFIG_PATH, encoding="utf-8"))
        import run_all
        vec = run_all._current_weights(cfg, tickers)
        if vec is not None and np.asarray(vec).sum() > 0:
            return np.asarray(vec, dtype=float), "run_all._current_weights(run_config.json)"
    except Exception:
        pass
    return None, "unavailable"


# ── Per-portfolio metrics on both bases ─────────────────────────────────────────

def portfolio_row(w, returns, Sigma, capm_mu):
    vol_lw = float(np.sqrt(w @ Sigma @ w))

    port = returns @ w                              # S monthly total returns
    vol_sim = float(np.std(port) * np.sqrt(12))

    er_factor = float(w @ capm_mu)
    er_sim_arith = float(np.mean(port) * 12)

    # geometric: exp(12 * mean(log(1 + r))) - 1, guarding 1 + r <= 0
    one_plus = 1.0 + port
    valid = one_plus > 0
    dropped = int((~valid).sum())
    er_sim_geom = float(np.exp(12.0 * np.mean(np.log(one_plus[valid]))) - 1.0) \
        if valid.any() else float("nan")
    return vol_lw, vol_sim, er_factor, er_sim_arith, er_sim_geom, dropped


def main():
    for p in (RESAMPLED_PATH, RETURNS_PATH, SIM_PATH):
        if not os.path.exists(p):
            print(f"[FATAL] required artifact missing: {os.path.basename(p)}")
            sys.exit(1)

    tickers, returns, Sigma, capm_mu = load_basis()
    S, n = returns.shape

    print("=" * 92)
    print("  RISK-BASIS DIAGNOSTIC  (read-only)   optimizer basis  vs  simulation basis")
    print("=" * 92)
    print(f"  Tickers          : {', '.join(tickers)}")
    print(f"  Scenarios        : {S:,} x {n}  (monthly total returns, simulated_returns.npz)")
    print(f"  Sigma_LW source  : returns_stats.xlsx 'Annualised Cov' (already annualized)")
    print(f"  capm_mu source   : returns_stats.xlsx 'CAPM Returns' (CAPM_Expected_Return)")

    # Assemble portfolios
    portfolios = []
    for sheet in GOAL_SHEETS:
        try:
            portfolios.append((sheet, _weights_from_sheet(sheet, tickers)))
        except Exception as exc:
            print(f"  [WARN] could not load goal '{sheet}': {exc}")
    w_cur, cur_src = _current_weights(tickers)
    if w_cur is not None:
        portfolios.append(("Current", w_cur))
    print(f"  Current weights  : {cur_src}")

    # ── Table A ────────────────────────────────────────────────────────────────
    print("\n" + "-" * 92)
    print("  TABLE A  --  per portfolio   (vol_LW = sqrt(w.Sigma.w);  vol_sim = std(R@w)*sqrt(12))")
    print("-" * 92)
    wcols = "".join(f"{t.replace('.NS',''):>9}" for t in tickers)
    print(f"  {'Portfolio':<18}{wcols}"
          f"{'volLW':>8}{'volSim':>8}{'ER_fac':>8}{'ERsimA':>8}{'ERsimG':>8}")
    for name, w in portfolios:
        vol_lw, vol_sim, er_fac, er_a, er_g, dropped = portfolio_row(
            w, returns, Sigma, capm_mu)
        wstr = "".join(f"{wi * 100:>8.1f}%" for wi in w)
        print(f"  {name:<18}{wstr}"
              f"{_pct(vol_lw):>8}{_pct(vol_sim):>8}{_pct(er_fac):>8}"
              f"{_pct(er_a):>8}{_pct(er_g):>8}")
        if dropped:
            print(f"  {'':<18}  (geometric: dropped {dropped} scenarios with 1+r <= 0)")

    print("\n  Legend: volLW=Ledoit-Wolf vol | volSim=simulation vol | "
          "ER_fac=factor-model E[r]")
    print("          ERsimA=sim E[r] arithmetic (mean*12) | "
          "ERsimG=sim E[r] geometric (compounded)")

    # Quick automated read-outs
    goals_only = [(nm, w) for nm, w in portfolios if nm in GOAL_SHEETS]
    if goals_only:
        lws = {nm: float(np.sqrt(w @ Sigma @ w)) for nm, w in goals_only}
        lowest = min(lws, key=lws.get)
        print(f"\n  Lowest Ledoit-Wolf vol among goals: {lowest} "
              f"({_pct(lws[lowest])})  "
              f"-> {'as expected' if lowest == 'Minimum Variance' else 'NOT Minimum Variance (!)'}")

    # ── Table B ────────────────────────────────────────────────────────────────
    print("\n" + "-" * 92)
    print("  TABLE B  --  per stock   (factor-model vs simulation, on both risk and return)")
    print("-" * 92)
    print(f"  {'Stock':<14}{'ER_factor':>11}{'ER_simAnn':>11}"
          f"{'vol_LW':>10}{'vol_sim':>10}{'sim/LW vol':>12}")
    sim_mean_ann = returns.mean(axis=0) * 12
    sim_vol_ann  = returns.std(axis=0) * np.sqrt(12)
    lw_vol       = np.sqrt(np.diag(Sigma))
    for i, t in enumerate(tickers):
        ratio = sim_vol_ann[i] / lw_vol[i] if lw_vol[i] else float("nan")
        print(f"  {t:<14}{_pct(capm_mu[i]):>11}{_pct(sim_mean_ann[i]):>11}"
              f"{_pct(lw_vol[i]):>10}{_pct(sim_vol_ann[i]):>10}{ratio:>11.2f}x")

    # Flag the stock driving inflation (largest sim-vs-LW vol ratio)
    ratios = sim_vol_ann / np.where(lw_vol == 0, np.nan, lw_vol)
    j = int(np.nanargmax(ratios))
    print(f"\n  Largest sim/LW vol blow-up: {tickers[j]} "
          f"({ratios[j]:.2f}x; sim {_pct(sim_vol_ann[j])} vs LW {_pct(lw_vol[j])}) "
          f"-> prime suspect for the residual inflation.")
    print("=" * 92)


if __name__ == "__main__":
    main()
