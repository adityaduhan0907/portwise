#!/usr/bin/env python3
"""
tail_risk_optimizer.py  —  Layer 6 (Tail-Risk / CVaR Optimizer)

Fills module 2's Goal 3. Minimises portfolio CVaR(95%) -- the average of the
worst 5% of monthly portfolio outcomes -- over the Layer 4 scenario matrix
(simulated_returns.npz), subject to:
    w >= 0, sum(w) = 1                       (long-only, fully invested)
    annualised simulated portfolio mean >= R_min   (return floor; a parameter)

Method: the Rockafellar-Uryasev linearisation, which is EXACT for empirical CVaR.
With losses L_s = -(scenario_return_s) and level alpha = 0.95, minimise over
(w, eta, u):
        eta + 1/((1-alpha) S) * sum_s u_s
        s.t.  u_s >= L_s - eta,  u_s >= 0
Solved with scipy.optimize.linprog (HiGHS) using a sparse constraint matrix.

The return constraint uses the SIMULATED per-asset means (same scenarios) so the
objective and the constraint are consistent.

FEASIBILITY is checked before returning weights (never returns garbage):
  1. R_max = max achievable annualised mean (= 12 * best single-asset mean). If
     R_min > R_max -> INFEASIBLE with a plain message; no weights.
  2. Otherwise solve -> achieved CVaR + weights.
  3. If a max-loss TARGET L is supplied and achieved CVaR > L, keep the weights but
     ATTACH a message (L is a target, not a hard constraint -- CVaR is already
     minimised).
"""

import os

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SIM_PATH   = os.path.join(SCRIPT_DIR, "simulated_returns.npz")
TRADING_MONTHS = 12
ALPHA = 0.95


# ── Inputs ───────────────────────────────────────────────────────────────────────

def load_scenarios(source=SIM_PATH):
    """Load Layer 4 scenarios. `source` is a .npz path or an in-memory DataFrame.
    Returns (returns ndarray [S x n], tickers list)."""
    if isinstance(source, pd.DataFrame):
        return source.values.astype(float), list(source.columns)
    data = np.load(source, allow_pickle=True)
    return data["returns"].astype(float), list(data["tickers"])


def _empirical_cvar_loss(port, alpha=ALPHA):
    """Empirical CVaR(alpha) of a return vector, returned as a positive loss
    (mean of the worst (1-alpha) tail). Matches Layer 5's definition."""
    q = np.percentile(port, (1.0 - alpha) * 100.0)
    tail = port[port <= q]
    cvar_return = float(tail.mean()) if tail.size else float(q)
    return -cvar_return, cvar_return            # (positive loss, signed return)


# ── Core CVaR optimisation (Rockafellar-Uryasev LP) ─────────────────────────────

def solve_min_cvar(scenarios, tickers=None, r_min=0.0, max_loss_target=None,
                   alpha=ALPHA):
    """
    Minimise portfolio CVaR(alpha) over the scenario matrix.

    scenarios       : ndarray [S x n] or DataFrame (cols = tickers).
    r_min           : ANNUALISED minimum-return floor on the simulated mean.
    max_loss_target : optional MONTHLY max-loss target L (positive fraction, e.g.
                      0.05). Compared to the achieved CVaR; not a hard constraint.

    Returns a result dict (status / weights / achieved CVaR / expected return /
    message). Never returns garbage weights: an infeasible floor yields status
    'infeasible_return' with weights=None.
    """
    if isinstance(scenarios, pd.DataFrame):
        R, tickers = scenarios.values.astype(float), list(scenarios.columns)
    else:
        R = np.asarray(scenarios, dtype=float)
        if tickers is None:
            tickers = [f"asset_{i}" for i in range(R.shape[1])]

    S, n = R.shape
    mu_sim_m = R.mean(axis=0)                          # per-asset monthly mean
    r_max_annual = float(TRADING_MONTHS * mu_sim_m.max())   # best single asset

    base = {
        "status":          None,
        "tickers":         list(tickers),
        "weights":         None,
        "weights_array":   None,
        "alpha":           alpha,
        "r_min":           r_min,
        "r_max_annual":    r_max_annual,
        "max_loss_target": max_loss_target,
        "achieved_cvar_monthly_loss": None,
        "achieved_cvar_monthly_return": None,
        "expected_return_annual":  None,
        "expected_return_monthly": None,
        "message":         "",
    }

    # ── Feasibility check 1: is the return floor achievable at all? ────────────
    if r_min > r_max_annual + 1e-9:
        base["status"]  = "infeasible_return"
        base["message"] = (
            f"INFEASIBLE: the highest achievable annualised return with these "
            f"assets is ~{r_max_annual*100:.1f}%; your floor of {r_min*100:.1f}% "
            f"can't be met. Lower the return floor to <= ~{r_max_annual*100:.1f}%."
        )
        return base

    # ── Build the Rockafellar-Uryasev LP ──────────────────────────────────────
    # Variables x = [ w(n) , eta(1) , u(S) ].
    n_u_coef = 1.0 / ((1.0 - alpha) * S)
    c = np.concatenate([np.zeros(n), [1.0], np.full(S, n_u_coef)])

    # CVaR rows:  -R w - eta - u_s <= 0   (since u_s >= -(R_s w) - eta)
    W_block = sparse.csr_matrix(-R)                        # S x n
    eta_col = sparse.csr_matrix(-np.ones((S, 1)))          # S x 1
    U_block = -sparse.identity(S, format="csr")           # S x S
    cvar_rows = sparse.hstack([W_block, eta_col, U_block], format="csr")

    # Return floor:  -mu_sim . w <= -(R_min / 12)
    ret_row = sparse.hstack(
        [sparse.csr_matrix(-mu_sim_m.reshape(1, -1)), sparse.csr_matrix((1, 1 + S))],
        format="csr",
    )

    A_ub = sparse.vstack([cvar_rows, ret_row], format="csr")
    b_ub = np.concatenate([np.zeros(S), [-(r_min / TRADING_MONTHS)]])

    # sum(w) = 1
    A_eq = sparse.hstack(
        [sparse.csr_matrix(np.ones((1, n))), sparse.csr_matrix((1, 1 + S))],
        format="csr",
    )
    b_eq = np.array([1.0])

    bounds = [(0.0, 1.0)] * n + [(None, None)] + [(0.0, None)] * S

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")

    if not res.success:
        base["status"]  = "lp_failed"
        base["message"] = f"CVaR LP did not solve: {res.message}"
        return base

    w = np.clip(res.x[:n], 0.0, None)
    w = w / w.sum()

    port = R @ w
    cvar_loss, cvar_ret = _empirical_cvar_loss(port, alpha)
    mean_m = float(port.mean())

    base.update({
        "status":          "optimal",
        "weights":         {t: float(wi) for t, wi in zip(tickers, w)},
        "weights_array":   w,
        "achieved_cvar_monthly_loss":   cvar_loss,
        "achieved_cvar_monthly_return": cvar_ret,
        "expected_return_annual":  mean_m * TRADING_MONTHS,
        "expected_return_monthly": mean_m,
    })

    floor_note = (f" Return floor {r_min*100:.1f}% satisfied "
                  f"({mean_m*TRADING_MONTHS*100:.1f}% achieved)." if r_min > 0 else "")
    msg = (f"Minimised CVaR(95%) = {cvar_loss*100:.2f}% monthly worst-case loss."
           + floor_note)

    # ── Feasibility check 3: max-loss target (comparison, not a constraint) ────
    if max_loss_target is not None and cvar_loss > max_loss_target + 1e-9:
        msg += (
            f"\n  NOTE: the lowest worst-case loss achievable with these assets is "
            f"~{cvar_loss*100:.1f}% (monthly CVaR); your target of "
            f"{max_loss_target*100:.1f}% isn't reachable -- these assets fall "
            f"together in bad markets. Options: raise your tolerance to "
            f"~{cvar_loss*100:.1f}%, or add diversifying assets (debt funds, gold, "
            f"bonds). A security-library mix feature to source such diversifiers is "
            f"planned."
        )
        base["target_reachable"] = False
    elif max_loss_target is not None:
        base["target_reachable"] = True

    base["message"] = msg
    return base


# ── Display / export helpers ─────────────────────────────────────────────────────

def print_tail_risk_result(result):
    W = 64
    print(f"\n{'='*W}")
    print("  3 — Tail-Risk Minimization (CVaR 95%)  [Layer 6]")
    print(f"{'='*W}")

    if result["status"] == "infeasible_return":
        print(f"  STATUS : INFEASIBLE (return floor)")
        print(f"  {result['message']}")
        print(f"{'='*W}")
        return
    if result["status"] != "optimal":
        print(f"  STATUS : {result['status'].upper()}")
        print(f"  {result['message']}")
        print(f"{'='*W}")
        return

    print(f"  {'Stock':<20}  {'Weight':>10}")
    print(f"  {'-'*20}  {'-'*10}")
    for t in result["tickers"]:
        w = result["weights"][t]
        if w >= 0.0001:
            print(f"  {t:<20}  {w*100:>9.2f}%")
    print(f"  {'-'*20}  {'-'*10}")
    print(f"\n  Expected Annual Return : {result['expected_return_annual']*100:>8.2f}%")
    print(f"  Achieved CVaR 95%      : {result['achieved_cvar_monthly_loss']*100:>8.2f}%  "
          f"(monthly worst-case loss)")
    print(f"  VaR-level outcome      : {result['achieved_cvar_monthly_return']*100:>8.2f}%  (monthly)")
    print(f"\n  {result['message']}")
    print(f"{'='*W}")


def build_export_df(result):
    """Return a DataFrame in module 2's sheet layout (Stock / Weight (%) + summary
    rows) so the tail-risk goal exports like the others."""
    rows = [{"Stock": t, "Weight (%)": round(result["weights"][t] * 100, 2)}
            for t in result["tickers"]]
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"Stock": "", "Weight (%)": ""}
    df.loc[len(df)] = {"Stock": "Portfolio Return (%)",
                       "Weight (%)": round(result["expected_return_annual"] * 100, 2)}
    df.loc[len(df)] = {"Stock": "Portfolio CVaR 95% (%)",
                       "Weight (%)": round(result["achieved_cvar_monthly_loss"] * 100, 2)}
    return df


# ── Standalone verification ──────────────────────────────────────────────────────

def main():
    if not os.path.exists(SIM_PATH):
        print(f"  ERROR: {SIM_PATH} not found. Run simulation_engine.py first.")
        return
    scen, tickers = load_scenarios(SIM_PATH)
    print(f"  Scenarios: {scen.shape}  tickers={tickers}")
    result = solve_min_cvar(scen, tickers, r_min=0.0)
    print_tail_risk_result(result)


if __name__ == "__main__":
    main()
