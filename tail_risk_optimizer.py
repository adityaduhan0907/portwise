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
    (mean of the worst (1-alpha) tail) plus the signed tail return.

    Uses the MEAN OF THE WORST k = round((1-alpha)*S) outcomes rather than
    mean(port[port <= percentile]). The two agree for a continuous distribution,
    but Method-A scenarios are bootstrapped from a small pool of real months, so
    `port` carries MANY exactly-tied values. The percentile-mask form is then
    unstable at the boundary: a 1-ULP change in the cutoff (e.g. from a 1e-17
    weight perturbation) flips a whole block of tied scenarios in/out of the tail,
    swinging the reported CVaR by ~1pp for an UNCHANGED portfolio. The worst-k
    form is tie-robust and monotonic, so the achieved-CVaR readout and the
    cross-goal ordering are stable. The LP objective (Rockafellar-Uryasev) is
    unchanged -- this only affects the post-hoc measurement."""
    S = port.shape[0]
    k = max(1, int(round((1.0 - alpha) * S)))
    worst = np.partition(port, k - 1)[:k]       # k smallest returns (unordered)
    cvar_return = float(worst.mean())
    return -cvar_return, cvar_return            # (positive loss, signed return)


# ── Core CVaR optimisation (Rockafellar-Uryasev LP) ─────────────────────────────

def _capped_max_return(mu_sim_m, w_min, w_max):
    """
    Highest achievable ANNUALISED mean under the adaptive box constraints
    (w_min <= w_i <= w_max, sum=1). Water-fill: seed every asset at w_min, then
    pour the remaining budget into the highest-mean assets up to w_max. Used so
    the return-floor feasibility check is honest about the cap (you can no longer
    put 100% in the single best asset).
    """
    n = mu_sim_m.shape[0]
    w = np.full(n, w_min, dtype=float)
    leftover = 1.0 - n * w_min
    for i in np.argsort(mu_sim_m)[::-1]:               # highest mean first
        if leftover <= 1e-12:
            break
        add = min(w_max - w_min, leftover)
        w[i] += add
        leftover -= add
    return float(TRADING_MONTHS * (mu_sim_m @ w))


def solve_min_cvar(scenarios, tickers=None, r_min=0.0, max_loss_target=None,
                   alpha=ALPHA, apply_caps=True):
    """
    Minimise portfolio CVaR(alpha) over the scenario matrix.

    scenarios       : ndarray [S x n] or DataFrame (cols = tickers).
    r_min           : ANNUALISED minimum-return floor on the simulated mean.
    max_loss_target : optional MONTHLY max-loss target L (positive fraction, e.g.
                      0.05). Compared to the achieved CVaR; not a hard constraint.
    apply_caps      : when True (default) apply the SAME adaptive per-asset box
                      (3 stocks 5%-60% / 4-6 3%-35% / 7-15 2%-20%) that Max
                      Risk-Adjusted uses, so the CVaR LP can't pile into one name.
                      The schedule is reused from module2_optimiser.get_adaptive_bounds
                      (NOT duplicated). Set False for the old uncapped behaviour.

    Returns a result dict (status / weights / achieved CVaR / expected return /
    message / caps). Never returns garbage weights: an infeasible floor yields
    status 'infeasible_return' with weights=None.
    """
    if isinstance(scenarios, pd.DataFrame):
        R, tickers = scenarios.values.astype(float), list(scenarios.columns)
    else:
        R = np.asarray(scenarios, dtype=float)
        if tickers is None:
            tickers = [f"asset_{i}" for i in range(R.shape[1])]

    S, n = R.shape
    mu_sim_m = R.mean(axis=0)                          # per-asset monthly mean

    # ── Adaptive per-asset box (reused from Max Risk-Adjusted) ────────────────
    # Lazy import: module2_optimiser imports THIS module at its top, so importing
    # it at module scope would be a circular import. Importing here is safe (the
    # module is fully initialised by the time any solve runs).
    if apply_caps:
        from module2_optimiser import get_adaptive_bounds
        w_min, w_max = get_adaptive_bounds(n)
    else:
        w_min, w_max = 0.0, 1.0

    # r_max respects the cap: best single asset if uncapped, else cap water-fill.
    r_max_annual = (_capped_max_return(mu_sim_m, w_min, w_max) if apply_caps
                    else float(TRADING_MONTHS * mu_sim_m.max()))

    base = {
        "status":          None,
        "tickers":         list(tickers),
        "weights":         None,
        "weights_array":   None,
        "alpha":           alpha,
        "r_min":           r_min,
        "r_max_annual":    r_max_annual,
        "max_loss_target": max_loss_target,
        "caps":            {"applied": bool(apply_caps),
                            "w_min": w_min, "w_max": w_max, "n_assets": n},
        "achieved_cvar_monthly_loss": None,
        "achieved_cvar_monthly_return": None,
        "expected_return_annual":  None,
        "expected_return_monthly": None,
        "message":         "",
    }

    # ── Feasibility check 1: is the return floor achievable at all? ────────────
    if r_min > r_max_annual + 1e-9:
        cap_note = (f" (under the {w_max*100:.0f}% per-asset cap)" if apply_caps else "")
        base["status"]  = "infeasible_return"
        base["message"] = (
            f"INFEASIBLE: the highest achievable annualised return with these "
            f"assets{cap_note} is ~{r_max_annual*100:.1f}%; your floor of "
            f"{r_min*100:.1f}% can't be met. Lower the return floor to "
            f"<= ~{r_max_annual*100:.1f}%."
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

    # Per-asset box on w (adaptive caps + matching minimums); eta free; u_s >= 0.
    bounds = [(w_min, w_max)] * n + [(None, None)] + [(0.0, None)] * S

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
    cap_note = (f" Per-asset caps applied (n={n}): {w_min*100:.0f}%-{w_max*100:.0f}%."
                if apply_caps else " No per-asset caps (uncapped).")
    msg = (f"Minimised CVaR(95%) = {cvar_loss*100:.2f}% monthly worst-case loss."
           + floor_note + cap_note)

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
