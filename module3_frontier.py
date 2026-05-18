#!/usr/bin/env python3
"""
module3_frontier.py

Reads prices.xlsx, recalculates annualised returns and covariance,
simulates 10 000 random portfolios, solves for the three optimal
portfolios (Max Sharpe, Min Volatility, Max Return), and plots the
efficient frontier saved as efficient_frontier.png.
"""

import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")          # headless render first; swap to interactive below
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

# ── Configuration ──────────────────────────────────────────────────────────────
TRADING_DAYS  = 252
MAX_WEIGHT    = 0.40
N_SIMULATIONS = 10_000
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))


def _load_risk_free_rate(fallback=0.043):
    """Load blended risk-free rate from risk_free_rates.json, or fall back to default."""
    json_path = os.path.join(SCRIPT_DIR, "risk_free_rates.json")
    try:
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)
        rate       = float(data["blended_rate"])
        fetched_at = data.get("fetched_at", "unknown")
        print(f"  Risk-free rate : {rate*100:.4f}%  "
              f"(blended, risk_free_rates.json — fetched {fetched_at})")
        return rate
    except FileNotFoundError:
        print(f"  WARNING: risk_free_rates.json not found — using fallback {fallback*100:.1f}%.")
        print("           Run module0_riskfree.py first for fresh rates.")
        return fallback
    except Exception as exc:
        print(f"  WARNING: Could not read risk_free_rates.json ({exc}). "
              f"Using fallback {fallback*100:.1f}%.")
        return fallback


RISK_FREE_RATE = _load_risk_free_rate()

# ── Portfolio maths ────────────────────────────────────────────────────────────

def portfolio_stats(weights, mu, cov):
    ret    = float(weights @ mu)
    vol    = float(np.sqrt(weights @ cov @ weights))
    sharpe = (ret - RISK_FREE_RATE) / vol if vol > 0 else float("nan")
    return ret, vol, sharpe


def neg_sharpe(w, mu, cov):
    _, vol, sharpe = portfolio_stats(w, mu, cov)
    return -sharpe if vol > 0 else 1e9


def portfolio_vol(w, mu, cov):
    return float(np.sqrt(w @ cov @ w))


def neg_return(w, mu, cov):
    return -float(w @ mu)


# ── SLSQP optimiser (identical logic to module2) ───────────────────────────────

def optimise(objective, mu, cov, label):
    n           = len(mu)
    bounds      = [(0.0, MAX_WEIGHT)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    rng         = np.random.default_rng(seed=42)
    best        = None

    for _ in range(50):
        w0 = rng.dirichlet(np.ones(n))
        w0 = np.clip(w0, 0, MAX_WEIGHT)
        w0 /= w0.sum()

        res = minimize(
            objective, w0, args=(mu, cov),
            method="SLSQP", bounds=bounds, constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 2000},
        )
        if (res.success or res.status == 0) and (best is None or res.fun < best.fun):
            best = res

    if best is None or not (best.success or best.status == 0):
        return None, False, f"Optimisation did not converge for '{label}'"

    w = np.clip(best.x, 0, None)
    w /= w.sum()
    return w, True, "OK"


# ── Random portfolio simulation ────────────────────────────────────────────────

def simulate_portfolios(mu, cov, n_sims, rng):
    """
    Generate n_sims random feasible portfolios.
    Strategy: draw Dirichlet weights, clip at MAX_WEIGHT, renormalise.
    Returns arrays: vols, rets, sharpes  (each of length n_sims).
    """
    n    = len(mu)
    vols    = np.empty(n_sims)
    rets    = np.empty(n_sims)
    sharpes = np.empty(n_sims)

    batch = 0
    attempts = 0
    max_attempts = n_sims * 20

    while batch < n_sims and attempts < max_attempts:
        # Generate a block at once for speed
        chunk = min(n_sims - batch, 2000)
        raw   = rng.dirichlet(np.ones(n), size=chunk)   # (chunk, n)
        raw   = np.clip(raw, 0, MAX_WEIGHT)
        row_sums = raw.sum(axis=1, keepdims=True)
        # Reject rows that collapsed to near-zero after clipping (extremely rare)
        valid = (row_sums.ravel() > 1e-9)
        raw   = raw[valid]
        raw  /= raw.sum(axis=1, keepdims=True)

        for w in raw:
            if batch >= n_sims:
                break
            r, v, s = portfolio_stats(w, mu, cov)
            rets[batch]    = r
            vols[batch]    = v
            sharpes[batch] = s
            batch += 1
        attempts += chunk

    if batch < n_sims:
        # Trim if we hit the attempt limit
        rets    = rets[:batch]
        vols    = vols[:batch]
        sharpes = sharpes[:batch]

    return vols, rets, sharpes


# ── Chart ──────────────────────────────────────────────────────────────────────

def build_chart(vols, rets, sharpes, optimal_points, out_path):
    """
    optimal_points: list of dicts with keys:
        label, vol, ret, sharpe, marker, color, size, zorder, edgecolor
    """
    fig, ax = plt.subplots(figsize=(13, 8))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    # ── Scatter: all simulated portfolios ─────────────────────────────────────
    # Clip extreme Sharpe values for colour normalisation so a handful of
    # outliers do not wash out the gradient across the bulk of points.
    s_min = np.nanpercentile(sharpes, 2)
    s_max = np.nanpercentile(sharpes, 98)
    norm  = plt.Normalize(vmin=s_min, vmax=s_max)

    sc = ax.scatter(
        vols * 100, rets * 100,
        c=sharpes, cmap="plasma", norm=norm,
        s=6, alpha=0.55, linewidths=0, zorder=2,
    )

    # ── Colour bar ────────────────────────────────────────────────────────────
    cbar = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.035)
    cbar.set_label("Sharpe Ratio", color="white", fontsize=11, labelpad=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=9)
    cbar.outline.set_edgecolor("#444")

    # ── Optimal portfolio markers ─────────────────────────────────────────────
    label_offsets = {          # (dx%, dy%) nudge for annotation text
        "Max Sharpe":     ( 0.5,  1.2),
        "Min Volatility": ( 0.4, -1.6),
        "Max Return":     (-4.8, -1.6),
    }

    for pt in optimal_points:
        x = pt["vol"] * 100
        y = pt["ret"] * 100

        ax.scatter(
            x, y,
            marker=pt["marker"], s=pt["size"], color=pt["color"],
            edgecolors=pt["edgecolor"], linewidths=1.5,
            zorder=pt["zorder"], label=pt["label"],
        )

        dx, dy = label_offsets.get(pt["label"], (0.5, 0.5))
        ax.annotate(
            f"{pt['label']}\n"
            f"Ret {pt['ret']*100:.1f}%  "
            f"Vol {pt['vol']*100:.1f}%  "
            f"SR {pt['sharpe']:.2f}",
            xy=(x, y), xytext=(x + dx, y + dy),
            fontsize=8.5, color="white",
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8),
            bbox=dict(boxstyle="round,pad=0.3", fc="#1f2937", ec="#555", alpha=0.85),
            path_effects=[pe.withStroke(linewidth=0, foreground="black")],
        )

    # ── Axes cosmetics ────────────────────────────────────────────────────────
    ax.set_xlabel("Annual Volatility (%)", color="white", fontsize=12, labelpad=8)
    ax.set_ylabel("Annual Return (%)",     color="white", fontsize=12, labelpad=8)
    ax.set_title("Efficient Frontier",     color="white", fontsize=16, fontweight="bold", pad=14)

    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.grid(color="#2a2a3a", linewidth=0.5, linestyle="--", alpha=0.7)

    legend = ax.legend(
        frameon=True, facecolor="#1f2937", edgecolor="#555",
        labelcolor="white", fontsize=10, loc="upper left",
        markerscale=0.8,
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n  Chart saved -> {out_path}")

    # Try to display on screen; silently skip on headless / no-display systems
    try:
        matplotlib.use("TkAgg")
        plt.switch_backend("TkAgg")
        plt.show()
    except Exception:
        try:
            matplotlib.use("QtAgg")
            plt.switch_backend("QtAgg")
            plt.show()
        except Exception:
            pass   # headless environment — chart is on disk


# ── Text summary ───────────────────────────────────────────────────────────────

def print_summary(optimal_points):
    w = 62
    print(f"\n{'='*w}")
    print("  OPTIMAL PORTFOLIO COORDINATES")
    print(f"{'='*w}")
    print(f"  {'Portfolio':<24}  {'Return':>8}  {'Volatility':>10}  {'Sharpe':>8}")
    print(f"  {'-'*24}  {'-'*8}  {'-'*10}  {'-'*8}")
    for pt in optimal_points:
        print(
            f"  {pt['label']:<24}  "
            f"{pt['ret']*100:>7.2f}%  "
            f"{pt['vol']*100:>9.2f}%  "
            f"{pt['sharpe']:>8.4f}"
        )
    print(f"{'='*w}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"\n{'='*62}")
    print("  Module 3 -- Efficient Frontier")
    print(f"{'='*62}\n")

    # ── 1. Load pre-computed mu and covariance from module1 ───────────────────
    returns_path = os.path.join(SCRIPT_DIR, "returns_stats.xlsx")
    if not os.path.exists(returns_path):
        print(f"  ERROR: '{returns_path}' not found.")
        print("  Run module1_data.py first to generate the required files.")
        sys.exit(1)

    mu      = None
    cov     = None
    tickers = None

    # Primary path: read pre-computed split-estimation mu and cov
    try:
        mu_df  = pd.read_excel(returns_path, sheet_name="Annualised Mu")
        cov_df = pd.read_excel(returns_path, sheet_name="Annualised Cov", index_col=0)
        tickers = list(mu_df["Ticker"])
        mu      = mu_df["Annualised_Expected_Return"].values.astype(float)
        cov     = cov_df.loc[tickers, tickers].values.astype(float)
        print(f"  Loaded split-estimation parameters for {len(tickers)} stocks:")
        for t in tickers:
            print(f"    {t}")
        print( "  Return source : momentum 12M-1M  (Annualised Mu sheet)")
        print( "  Risk source   : 10-year monthly  (Annualised Cov sheet)")
    except Exception:
        pass

    # Fallback: recompute from daily returns if new sheets are absent
    if mu is None:
        print("  WARNING: 'Annualised Mu'/'Annualised Cov' sheets not found.")
        print("           Re-run module1_data.py to generate split-estimation parameters.")
        print("           Falling back to daily-returns-based estimation.\n")
        prices_path = os.path.join(SCRIPT_DIR, "prices.xlsx")
        if not os.path.exists(prices_path):
            print(f"  ERROR: '{prices_path}' not found either. Run module1_data.py first.")
            sys.exit(1)
        try:
            prices_df = pd.read_excel(prices_path, index_col=0, parse_dates=True)
        except Exception as exc:
            print(f"  ERROR reading prices.xlsx: {exc}")
            sys.exit(1)
        prices_df  = prices_df.dropna(how="all").dropna(axis=1, how="all")
        returns_df = prices_df.pct_change().iloc[1:]
        clean      = returns_df.dropna()
        tickers    = list(clean.columns)
        mu         = clean.mean().values * TRADING_DAYS
        cov        = clean.cov().values  * TRADING_DAYS
        print(f"  {len(clean)} daily return observations for {len(tickers)} stocks:")
        for t in tickers:
            print(f"    {t}")

    n = len(tickers)

    min_stocks = int(np.ceil(1.0 / MAX_WEIGHT))
    if n < min_stocks:
        print(f"\n  ERROR: Need at least {min_stocks} stocks for the "
              f"{MAX_WEIGHT*100:.0f}% cap. Only {n} available.")
        sys.exit(1)

    # ── 3. Simulate random portfolios ─────────────────────────────────────────
    print(f"\n  Simulating {N_SIMULATIONS:,} random portfolios ...", end=" ", flush=True)
    rng = np.random.default_rng(seed=0)
    sim_vols, sim_rets, sim_sharpes = simulate_portfolios(mu, cov, N_SIMULATIONS, rng)
    print(f"done. ({len(sim_rets):,} generated)")

    # ── 4. Optimise three portfolios ───────────────────────────────────────────
    opt_configs = [
        ("Max Sharpe",     neg_sharpe,    "gold",        "*", 500, 6,  "white"),
        ("Min Volatility", portfolio_vol, "dodgerblue",  "D", 220, 6,  "white"),
        ("Max Return",     neg_return,    "orangered",   "^", 260, 6,  "white"),
    ]

    optimal_points = []
    any_failed     = False

    for label, obj_fn, color, marker, size, zorder, ec in opt_configs:
        print(f"  Optimising: {label} ...", end=" ", flush=True)
        w, success, msg = optimise(obj_fn, mu, cov, label)
        if not success:
            print(f"\n  WARNING: {msg}")
            any_failed = True
            continue
        r, v, s = portfolio_stats(w, mu, cov)
        print(f"done.  Ret {r*100:.2f}%  Vol {v*100:.2f}%  SR {s:.4f}")
        optimal_points.append({
            "label": label, "weights": w,
            "ret": r, "vol": v, "sharpe": s,
            "color": color, "marker": marker, "size": size,
            "zorder": zorder, "edgecolor": ec,
        })

    if not optimal_points:
        print("\n  All optimisations failed — cannot draw chart.")
        sys.exit(1)

    if any_failed:
        print("  (One or more optimisations failed — chart may be incomplete.)")

    # ── 5. Print text summary ──────────────────────────────────────────────────
    print_summary(optimal_points)

    # ── 6. Build and save chart ────────────────────────────────────────────────
    print(f"\n  Building chart ...", end=" ", flush=True)
    out_path = os.path.join(SCRIPT_DIR, "efficient_frontier.png")
    try:
        build_chart(sim_vols, sim_rets, sim_sharpes, optimal_points, out_path)
    except Exception as exc:
        print(f"\n  ERROR building chart: {exc}")
        sys.exit(1)

    print(f"\n{'='*62}\n")


if __name__ == "__main__":
    main()
