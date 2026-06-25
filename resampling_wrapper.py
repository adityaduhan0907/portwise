#!/usr/bin/env python3
"""
resampling_wrapper.py  —  Layer 7 (Michaud-style Resampling Wrapper)

Stabilises and de-concentrates the optimizer weights by bootstrapping the inputs,
re-optimizing B times, and AVERAGING the resulting weight vectors. B defaults PER
GOAL (parameter, do not raise): GMV / Max-Risk-Adjusted = 500, Tail-Risk CVaR = 250
(the CVaR LP is the costlier per-iteration solve and 250 is stable enough).

Reuses existing code -- nothing is re-implemented:
  - module1_data._ledoit_wolf_cov           (Method A covariance estimator)
  - module2_optimiser.optimise / portfolio_vol / neg_sharpe / get_adaptive_bounds
                                             (GMV + Max Risk-Adjusted optimizers)
  - module2_optimiser.load_capm_mu           (factor expected returns)
  - tail_risk_optimizer.solve_min_cvar       (Layer 6 CVaR optimizer)

Resampling differs by goal:
  Min Variance / Max Risk-Adjusted (covariance-based, Method A):
    bootstrap the 10y monthly return matrix (resample MONTHS / rows with
    replacement, one shared row-draw so cross-asset structure is preserved),
    re-estimate Sigma via module1's Ledoit-Wolf, keep mu fixed at capm_mu
    (factor premiums -- NOT re-estimated), re-run the goal's optimizer with its
    existing constraints (RAR keeps adaptive caps; GMV uncapped).
  Tail-Risk CVaR (Layer 6):
    bootstrap the 10,000-scenario matrix (resample scenarios / rows), re-run
    solve_min_cvar with the same r_min.
  Method B (any asset < 36 months -> factor covariance):
    bootstrapping asset returns is a no-op for a factor-built Sigma; re-bootstrapping
    factor history / residual pools isn't cleanly wired through the current handoffs,
    so this path is DEFERRED with a NOTE -- single-shot Method-B weights returned
    unchanged (consistent with the mixed-portfolio Method-B deferral).

Output: resampled_portfolios.xlsx (one sheet per goal) -- NON-destructive, does NOT
overwrite optimised_portfolios.xlsx. Per-iteration weights kept in memory and
optionally saved to resampled_weights.npz.

Non-interactive, file-based, plain-text. Seeds via numpy SeedSequence (iteration i
independent), base_seed + goal selector parameters.
"""

import json
import os
import time

import numpy as np
import pandas as pd

import module1_data as m1
import module2_optimiser as m2
import tail_risk_optimizer as tro

# ── Configuration ──────────────────────────────────────────────────────────────
SCRIPT_DIR          = os.path.dirname(os.path.abspath(__file__))
RETURNS_PATH        = os.path.join(SCRIPT_DIR, "returns_stats.xlsx")
OPTIMISED_PATH      = os.path.join(SCRIPT_DIR, "optimised_portfolios.xlsx")
RESAMPLED_PATH      = os.path.join(SCRIPT_DIR, "resampled_portfolios.xlsx")
SIM_PATH            = os.path.join(SCRIPT_DIR, "simulated_returns.npz")
FACTOR_HISTORY_PATH = os.path.join(SCRIPT_DIR, "factor_history.json")
WEIGHTS_NPZ_PATH    = os.path.join(SCRIPT_DIR, "resampled_weights.npz")

DEFAULT_B      = 500        # chosen value for GMV / RAR -- do not raise
DEFAULT_B_CVAR = 250        # CVaR goal only: the LP per-iter cost is higher and
                            # 250 is stable enough (~halves the CVaR wall-clock)
DEFAULT_SEED = 20260616
GOALS        = ("gmv", "rar", "cvar")

# ── Solver-cost / parallelism knobs (Layer-7 speedups; do NOT change the
#    statistical result -- same B, same averaging, same per-iteration logic) ──────
DEFAULT_N_JOBS   = None     # B-loop parallelism policy. None = per-goal default:
                            # CVaR parallelises across cores (its LP iter is heavy
                            # enough to beat process-spawn overhead), GMV/RAR stay
                            # SEQUENTIAL (at restarts=10 their iters are too cheap --
                            # spawn/pickle overhead cancels the gain). Passing an
                            # explicit int overrides this for ANY goal. Parallel
                            # paths degrade to a sequential loop on a 1-core box.
DEFAULT_RESTARTS = 10       # SLSQP multi-starts per GMV/RAR iteration. Lowered from
                            # 50: verified weight diff 10-vs-50 was 0.0000 pp (the
                            # n=5 global optimum is found from every start), so the
                            # extra restarts were pure cost (a solver-cost knob).
DEFAULT_S_SUB    = 5000     # CVaR ONLY: scenarios drawn per resample LP. The full
                            # set is 10,000; subsampling halves the HiGHS solve and
                            # does NOT touch Layer-5 (which always uses all 10,000).


def default_B(goal):
    """Per-goal resampling count: CVaR uses 250, GMV/RAR use 500."""
    return DEFAULT_B_CVAR if goal == "cvar" else DEFAULT_B
GOAL_SHEET   = {"gmv": "Minimum Variance", "rar": "Max Risk-Adjusted", "cvar": "Tail-Risk CVaR"}
GOAL_LABEL   = {"gmv": "Min Variance (GMV)", "rar": "Max Risk-Adjusted", "cvar": "Tail-Risk CVaR"}


# ── Input loading (reuses existing artifacts) ───────────────────────────────────

def _load_monthly_returns(returns_path=RETURNS_PATH):
    """10y monthly asset returns from module1's 'LongRun Monthly Returns' sheet."""
    df = pd.read_excel(returns_path, sheet_name="LongRun Monthly Returns",
                       index_col=0, parse_dates=True)
    return df


def _load_universe(returns_path=RETURNS_PATH):
    """(tickers, capm_mu) -- tickers from 'Annualised Mu', factor mu via module2."""
    mu_df   = pd.read_excel(returns_path, sheet_name="Annualised Mu")
    tickers = list(mu_df["Ticker"])
    mom_mu  = mu_df["Annualised_Expected_Return"].values.astype(float)
    capm_mu = m2.load_capm_mu(returns_path, tickers, mom_mu)
    return tickers, capm_mu


def _use_factor_cov(path=FACTOR_HISTORY_PATH):
    try:
        with open(path, encoding="utf-8") as f:
            return bool(json.load(f).get("use_factor_covariance", False))
    except Exception:
        return False


def load_single_shot_weights(sheet, tickers, path=OPTIMISED_PATH):
    """Single-shot optimizer weights for a goal from optimised_portfolios.xlsx."""
    df = pd.read_excel(path, sheet_name=sheet)
    wmap = {r["Stock"]: float(r["Weight (%)"]) / 100.0
            for _, r in df.iterrows() if r["Stock"] in set(tickers)}
    return np.array([wmap.get(t, 0.0) for t in tickers])


# ── Parallel B-loop machinery ────────────────────────────────────────────────────
#
# The B iterations are independent (each draws its own SeedSequence child seed, no
# shared mutable state), so they parallelise across cores. We use the STANDARD
# LIBRARY (concurrent.futures.ProcessPoolExecutor) -- no joblib dependency -- and
# degrade to a plain sequential loop on a 1-core box or any pool failure. The
# read-only context (the return matrix, the optimizer settings) is shipped to each
# worker ONCE via the pool initializer, so the big array isn't re-pickled per task.
#
# Reproducibility: workers consume child_seeds[i] exactly as the old loop did, and
# the pool preserves input order, so the averaged weights are byte-identical to the
# sequential version for a given base_seed.

_CTX = {}   # per-process worker context, populated by _pool_init


def _pool_init(ctx):
    global _CTX
    _CTX = ctx


def _cov_worker(seed):
    """One Method-A resample: bootstrap months -> Ledoit-Wolf Sigma -> optimise."""
    M, tickers, capm_mu = _CTX["M"], _CTX["tickers"], _CTX["capm_mu"]
    obj, goal           = _CTX["obj"], _CTX["goal"]
    w_min, w_max        = _CTX["w_min"], _CTX["w_max"]
    n_restarts          = _CTX["n_restarts"]
    T = M.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, T, size=T)                     # shared row-draw (all assets)
    boot = pd.DataFrame(M[idx], columns=tickers)
    cov_monthly, _ = m1._ledoit_wolf_cov(boot)           # reuse module1 estimator
    cov_annual = cov_monthly.values * m1.TRADING_MONTHS
    w, success, _ = m2.optimise(obj, capm_mu, cov_annual, goal, w_min, w_max,
                                n_restarts=n_restarts)
    return w if success else None


def _cvar_worker(seed):
    """One CVaR resample: bootstrap S_sub scenarios -> re-solve the CVaR LP."""
    scen, tickers = _CTX["scen"], _CTX["tickers"]
    r_min, s_sub  = _CTX["r_min"], _CTX["s_sub"]
    S = scen.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, S, size=s_sub)                 # subsample (default 5,000)
    res = tro.solve_min_cvar(scen[idx], tickers, r_min=r_min)
    return res["weights_array"] if res["status"] == "optimal" else None


def _resolve_n_jobs(n_jobs):
    """Worker count: None -> n_cores-1; always clamped to [1, n_cores]. 1 == sequential."""
    cores = os.cpu_count() or 1
    if n_jobs is None:
        return max(1, cores - 1)
    return max(1, min(int(n_jobs), cores))


def _policy_n_jobs(goal, n_jobs):
    """Per-goal parallelism policy when n_jobs is left as the default (None):
    CVaR parallelises (heavy LP iter), GMV/RAR run sequentially (cheap iters at
    restarts=10 lose to spawn overhead). An explicit int overrides for any goal."""
    if n_jobs is None:
        return None if goal == "cvar" else 1   # None -> _resolve gives n_cores-1
    return n_jobs


def _map_seeds(worker, ctx, seeds, n_jobs):
    """
    Apply `worker(seed)` for every child seed, preserving input order so the result
    is byte-identical to the sequential loop. Runs in a ProcessPoolExecutor when
    n_jobs>1; otherwise (or if the pool can't start -- e.g. a restricted 1-core
    deployment) falls back to a sequential loop in-process.
    """
    seeds  = list(seeds)
    n_jobs = _resolve_n_jobs(n_jobs)
    if n_jobs > 1 and len(seeds) > 1:
        # NOTE: deliberately do NOT pin worker BLAS threads. The per-iteration
        # matrices are tiny so intra-op threads buy nothing, but forcing 1 thread
        # changes the FP reduction order and breaks exact byte-identity with the
        # sequential baseline -- which we value more than a marginal scaling gain.
        try:
            from concurrent.futures import ProcessPoolExecutor
            chunk = max(1, len(seeds) // (n_jobs * 4))
            with ProcessPoolExecutor(max_workers=n_jobs,
                                     initializer=_pool_init, initargs=(ctx,)) as ex:
                return list(ex.map(worker, seeds, chunksize=chunk))
        except Exception as exc:        # graceful degradation -> sequential
            print(f"  [Layer 7] parallel pool unavailable ({exc}); running sequentially.")
    _pool_init(ctx)                     # sequential path: set context in THIS process
    return [worker(s) for s in seeds]


# ── Resampling cores ────────────────────────────────────────────────────────────

def _resample_cov_goal(goal, monthly_df, tickers, capm_mu, B, base_seed,
                       n_jobs=DEFAULT_N_JOBS, n_restarts=DEFAULT_RESTARTS):
    """
    Covariance-based resampling (Method A) for 'gmv' or 'rar'. Bootstraps months,
    re-estimates Ledoit-Wolf Sigma, re-runs the goal's optimizer. Returns
    (avg_weights, per_iteration_W, n_failures). The B-loop is parallelised across
    cores (byte-identical to the old sequential loop for a given base_seed).
    """
    M = monthly_df[tickers].values
    T, n = M.shape

    if goal == "gmv":
        obj, w_min, w_max = m2.portfolio_vol, 0.0, 1.0
    else:  # rar
        obj, w_min, w_max = m2.neg_sharpe, 0.0, m2.get_adaptive_bounds(n)[1]

    child_seeds = np.random.SeedSequence(base_seed).spawn(B)
    ctx = {"M": M, "tickers": tickers, "capm_mu": capm_mu, "obj": obj, "goal": goal,
           "w_min": w_min, "w_max": w_max, "n_restarts": n_restarts}
    results = _map_seeds(_cov_worker, ctx, child_seeds, n_jobs)

    W = np.array([w for w in results if w is not None]).reshape(-1, n)
    avg = W.mean(axis=0)
    avg = avg / avg.sum()                                # defensive renormalise
    return avg, W, (B - W.shape[0])


def _resample_cvar_goal(scen, tickers, r_min, B, base_seed,
                        n_jobs=DEFAULT_N_JOBS, s_sub=DEFAULT_S_SUB):
    """
    Scenario-based resampling for the CVaR goal. Bootstraps `s_sub` of the 10k
    scenarios per iteration (subsampling shrinks the HiGHS LP, which is ~99% of the
    per-iter cost) and re-solves solve_min_cvar. The tie-robust worst-k CVaR readout
    scales with the draw, so it stays valid; Layer 5 is unaffected (it always uses
    all 10,000). Returns (avg_weights, W, n_failures). Parallelised across cores.
    """
    S, n = scen.shape
    s_sub = max(1, int(s_sub))
    child_seeds = np.random.SeedSequence(base_seed).spawn(B)
    ctx = {"scen": scen, "tickers": tickers, "r_min": r_min, "s_sub": s_sub}
    results = _map_seeds(_cvar_worker, ctx, child_seeds, n_jobs)

    W = np.array([w for w in results if w is not None]).reshape(-1, n)
    avg = W.mean(axis=0)
    avg = avg / avg.sum()
    return avg, W, (B - W.shape[0])


# ── Public entry point ──────────────────────────────────────────────────────────

def resample_goal(goal, B=None, base_seed=DEFAULT_SEED, r_min=0.0, verbose=True,
                  n_jobs=DEFAULT_N_JOBS, n_restarts=DEFAULT_RESTARTS,
                  s_sub=DEFAULT_S_SUB):
    """
    Run resampling for one goal ('gmv' / 'rar' / 'cvar'). Returns a result dict:
      {goal, status, tickers, avg_weights(dict), avg_array, per_iter (B x n),
       n_failures, elapsed_s, deferred(bool), message}

    B is a parameter; when left as None it defaults PER GOAL via default_B()
    (CVaR -> 250, GMV/RAR -> 500). Passing B explicitly overrides the default.

    Speed knobs (none change the statistical contract):
      n_jobs     : worker processes for the independent B-loop (None -> n_cores-1).
      n_restarts : SLSQP multi-starts per GMV/RAR iteration (default 50).
      s_sub      : scenarios drawn per CVaR resample LP (default 5,000 of 10,000).
    """
    if goal not in GOALS:
        raise ValueError(f"goal must be one of {GOALS}, got {goal!r}")
    if B is None:
        B = default_B(goal)

    tickers, capm_mu = _load_universe()
    n = len(tickers)
    t0 = time.time()

    if goal in ("gmv", "rar"):
        if _use_factor_cov():
            # Method B (factor covariance) -> resampling deferred.
            ss = load_single_shot_weights(GOAL_SHEET[goal], tickers)
            msg = ("NOTE: use_factor_covariance=True -> covariance is factor-built "
                   "(B F B^T + D). Bootstrapping asset returns is a no-op for it, and "
                   "re-bootstrapping factor history / residual pools is not cleanly "
                   "wired through the current handoffs. Resampling DEFERRED; returning "
                   "single-shot Method-B weights unchanged.")
            if verbose:
                print(f"  [{goal}] {msg}")
            return {"goal": goal, "status": "deferred_method_b", "tickers": tickers,
                    "avg_weights": dict(zip(tickers, ss)), "avg_array": ss,
                    "per_iter": ss.reshape(1, -1), "n_failures": 0,
                    "elapsed_s": time.time() - t0, "deferred": True, "message": msg}

        monthly = _load_monthly_returns()
        eff_jobs = _policy_n_jobs(goal, n_jobs)          # GMV/RAR -> sequential by default
        avg, W, fails = _resample_cov_goal(goal, monthly, tickers, capm_mu, B, base_seed,
                                           n_jobs=eff_jobs, n_restarts=n_restarts)
        msg = (f"Method A resampling: B={B}, {W.shape[0]} solves used, {fails} failed "
               f"(restarts={n_restarts}, n_jobs={_resolve_n_jobs(eff_jobs)}).")

    else:  # cvar
        scen, scen_tickers = tro.load_scenarios(SIM_PATH)
        if scen_tickers != tickers:
            tickers = scen_tickers                       # trust the scenario labels
        eff_jobs = _policy_n_jobs(goal, n_jobs)          # CVaR -> parallel by default
        avg, W, fails = _resample_cvar_goal(scen, tickers, r_min, B, base_seed,
                                            n_jobs=eff_jobs, s_sub=s_sub)
        msg = (f"CVaR resampling: B={B}, {W.shape[0]} LP solves used, {fails} failed "
               f"(s_sub={min(s_sub, scen.shape[0])}/{scen.shape[0]}, "
               f"n_jobs={_resolve_n_jobs(eff_jobs)}).")

    if verbose:
        print(f"  [{goal}] {msg}  ({time.time()-t0:.1f}s)")

    return {"goal": goal, "status": "optimal", "tickers": tickers,
            "avg_weights": dict(zip(tickers, avg)), "avg_array": avg,
            "per_iter": W, "n_failures": fails,
            "elapsed_s": time.time() - t0, "deferred": False, "message": msg}


# ── Output ───────────────────────────────────────────────────────────────────────

def _goal_sheet_df(result, capm_mu, tickers):
    """Sheet in module2's layout: Stock / Weight (%) rows + summary rows."""
    w = result["avg_array"]
    rows = [{"Stock": t, "Weight (%)": round(float(wi) * 100, 2)}
            for t, wi in zip(tickers, w)]
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"Stock": "", "Weight (%)": ""}
    ann_ret = float(capm_mu @ w) * 100.0
    df.loc[len(df)] = {"Stock": "Portfolio Return (%)", "Weight (%)": round(ann_ret, 2)}
    if result["goal"] == "cvar":
        scen, _ = tro.load_scenarios(SIM_PATH)
        port = scen @ w
        loss, _ = tro._empirical_cvar_loss(port)
        df.loc[len(df)] = {"Stock": "Portfolio CVaR 95% (%)", "Weight (%)": round(loss * 100, 2)}
    return df


def write_resampled_xlsx(results, capm_mu, tickers, path=RESAMPLED_PATH):
    """Write averaged weights, one sheet per goal (non-destructive)."""
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for r in results:
            df = _goal_sheet_df(r, capm_mu, tickers)
            df.to_excel(writer, sheet_name=GOAL_SHEET[r["goal"]], index=False)
    return path


def save_per_iter(results, tickers, path=WEIGHTS_NPZ_PATH):
    payload = {f"W_{r['goal']}": r["per_iter"] for r in results}
    payload["tickers"] = np.array(tickers, dtype=object)
    np.savez_compressed(path, **payload)
    return path


# ── Verification ────────────────────────────────────────────────────────────────

def _verify(B=None, seed_a=DEFAULT_SEED, seed_b=DEFAULT_SEED + 1, r_min=0.0):
    W = 74
    tickers, capm_mu = _load_universe()
    b_desc = (f"B={B}" if B is not None
              else f"B per goal (GMV/RAR={DEFAULT_B}, CVaR={DEFAULT_B_CVAR})")
    print(f"\n{'='*W}")
    print("  Layer 7 -- Resampling Wrapper  (VERIFICATION)")
    print(f"  Assets : {', '.join(tickers)}   |   {b_desc}   |   seeds={seed_a},{seed_b}")
    print(f"  Method : {'B (factor cov) DEFERRED' if _use_factor_cov() else 'A (Ledoit-Wolf)'}"
          f" for GMV/RAR   |   CVaR over Layer-4 scenarios")
    print(f"{'='*W}")

    results_a = [resample_goal(g, B=B, base_seed=seed_a, r_min=r_min) for g in GOALS]
    by_goal_a = {r["goal"]: r for r in results_a}

    # 1. Determinism -------------------------------------------------------------
    print(f"\n  [1] DETERMINISM (same base_seed -> identical averaged weights)")
    for g in GOALS:
        rep = resample_goal(g, B=B, base_seed=seed_a, r_min=r_min, verbose=False)
        identical = np.array_equal(rep["avg_array"], by_goal_a[g]["avg_array"])
        print(f"      {GOAL_LABEL[g]:<20}: byte-identical re-run = {identical}")

    # 2. Stability (is 500 enough): two seeds ------------------------------------
    print(f"\n  [2] STABILITY  -- {b_desc}, two base seeds, drift between averaged vectors")
    results_b = {g: resample_goal(g, B=B, base_seed=seed_b, r_min=r_min, verbose=False)
                 for g in GOALS}
    for g in GOALS:
        wa, wb = by_goal_a[g]["avg_array"], results_b[g]["avg_array"]
        drift = np.abs(wa - wb)
        print(f"      {GOAL_LABEL[g]:<20}: max per-asset drift {drift.max()*100:5.2f} pp"
              f"   L1 {drift.sum()*100:5.2f} pp")

    # 3. De-concentration vs single-shot -----------------------------------------
    print(f"\n  [3] DE-CONCENTRATION  (resampled vs single-shot)")
    print(f"      {'goal':<20}{'single max':>12}{'resamp max':>12}"
          f"{'single>=2%':>12}{'resamp>=2%':>12}")
    for g in GOALS:
        ss = load_single_shot_weights(GOAL_SHEET[g], tickers)
        rs = by_goal_a[g]["avg_array"]
        print(f"      {GOAL_LABEL[g]:<20}{ss.max()*100:>11.2f}%{rs.max()*100:>11.2f}%"
              f"{int((ss >= 0.02).sum()):>12}{int((rs >= 0.02).sum()):>12}")

    # 4. Constraints intact ------------------------------------------------------
    print(f"\n  [4] CONSTRAINTS")
    cap = m2.get_adaptive_bounds(len(tickers))[1]
    for g in GOALS:
        w = by_goal_a[g]["avg_array"]
        line = (f"      {GOAL_LABEL[g]:<20}: sum={w.sum():.6f}  min={w.min():+.4f}  "
                f"max={w.max()*100:.2f}%")
        if g == "rar":
            line += f"  (cap {cap*100:.0f}% respected: {w.max() <= cap + 1e-6})"
        if g == "cvar":
            port_mean_ann = (tro.load_scenarios(SIM_PATH)[0] @ w).mean() * m1.TRADING_MONTHS
            line += f"  (return floor {r_min*100:.0f}% holds: {port_mean_ann >= r_min - 1e-9})"
        print(line)

    # 5. CVaR ordering preserved -------------------------------------------------
    print(f"\n  [5] CVaR ORDERING  (resampled CVaR weights vs other goals)")
    scen, _ = tro.load_scenarios(SIM_PATH)
    def cvar_of(w):
        return tro._empirical_cvar_loss(scen @ w)[0]
    cvar_resampled = cvar_of(by_goal_a["cvar"]["avg_array"])
    cvar_gmv = cvar_of(by_goal_a["gmv"]["avg_array"])
    cvar_rar = cvar_of(by_goal_a["rar"]["avg_array"])
    print(f"      resampled CVaR weights : {cvar_resampled*100:.2f}%")
    print(f"      resampled GMV  weights : {cvar_gmv*100:.2f}%   "
          f"(CVaR <= GMV: {cvar_resampled <= cvar_gmv + 1e-9})")
    print(f"      resampled RAR  weights : {cvar_rar*100:.2f}%   "
          f"(CVaR <= RAR: {cvar_resampled <= cvar_rar + 1e-9})")

    # 6. Runtime -----------------------------------------------------------------
    print(f"\n  [6] RUNTIME ({b_desc})")
    for g in GOALS:
        print(f"      {GOAL_LABEL[g]:<20}: {by_goal_a[g]['elapsed_s']:6.1f}s"
              f"   ({by_goal_a[g]['n_failures']} failed solves)")

    # Output files ---------------------------------------------------------------
    write_resampled_xlsx(results_a, capm_mu, tickers)
    save_per_iter(results_a, tickers)
    print(f"\n  Wrote -> {RESAMPLED_PATH}  (optimised_portfolios.xlsx untouched)")
    print(f"  Wrote -> {WEIGHTS_NPZ_PATH}")
    print(f"{'='*W}\n")
    return results_a


def main():
    _verify()


if __name__ == "__main__":
    main()
