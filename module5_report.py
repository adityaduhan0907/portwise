#!/usr/bin/env python3
"""
module5_report.py  --  Module 5 report builder

Pulls the pipeline's existing outputs into ONE plain-language Markdown report that
a retail investor can read top to bottom. It is NON-INTERACTIVE and reads existing
artifacts only -- no recomputation beyond the same simple display-layer maths the
dashboard already does (past-year return, factor-model Risk Efficiency). Every input
is optional: if a file is missing or unreadable, the affected section degrades
gracefully with a clear note rather than crashing.

This report is kept IN STEP with the dashboard (app.py). In particular:
  * plain-language metric labels (Typical Worst-Year Loss / Average Loss During
    Crashes / Chance of Large Loss / Worst Expected Decline / Risk Efficiency);
  * ANNUAL VaR/CVaR figures (the monthly_* fields stay internal, never shown);
  * momentum is NOT surfaced (it still runs upstream); only the two diversification
    checks -- correlation and sector concentration -- appear, with their numbers;
  * the recommended section shows current-vs-recommended weights and TWO clearly
    separated returns (current's actual past year vs the recommended's forward
    projection), never a single before/after number;
  * the cross-goal comparison is LOAD-BEARING here -- the dashboard shows only the
    chosen goal vs the current portfolio, so this report is the only place all three
    goals are laid out side by side;
  * a historical crisis stress-test section, with honest coverage notes.

Reads (all produced upstream)
-----------------------------
  run_config.json                 run context: tickers, holdings, currency, choice
  resampled_portfolios.xlsx       chosen-goal weights + forward return (Layer 7)
  rebalancing_plan_*.xlsx (latest) buy/sell instructions + currency (Module 4)
  risk_evaluation_summary.json    chosen + all goals + Current Portfolio risk (Layer 5)
  robustness_warnings.json        the diversification checks (Layer 6)
  stress_test.json                historical crisis replay (post-Layer 5)
  returns_stats.xlsx              factor-model inputs for Risk Efficiency (Module 1)
  prices.xlsx                     price history for the current past-year return
  risk_free_rates.json            blended risk-free rate (Module 0)

Writes
------
  portfolio_report_YYYYMMDD.md    date-stamped Markdown report in the repo dir.

Usage
-----
  python module5_report.py [PORTFOLIO_NAME]
"""

import glob
import json
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
RESAMPLED_PATH = os.path.join(SCRIPT_DIR, "resampled_portfolios.xlsx")
RISK_PATH      = os.path.join(SCRIPT_DIR, "risk_evaluation_summary.json")
ROBUST_PATH    = os.path.join(SCRIPT_DIR, "robustness_warnings.json")
CONFIG_PATH    = os.path.join(SCRIPT_DIR, "run_config.json")
STRESS_PATH    = os.path.join(SCRIPT_DIR, "stress_test.json")
RETURNS_PATH   = os.path.join(SCRIPT_DIR, "returns_stats.xlsx")
PRICES_PATH    = os.path.join(SCRIPT_DIR, "prices.xlsx")
RF_PATH        = os.path.join(SCRIPT_DIR, "risk_free_rates.json")

# ── Plain-language labels for the three goals ───────────────────────────────────
GOAL_LABELS = {
    "Minimum Variance":  "Lowest Risk",
    "Max Risk-Adjusted": "Best Balance of Risk and Return",
    "Tail-Risk CVaR":    "Crash Protection",
}
GOAL_ONE_LINER = {
    "Minimum Variance":  "It aims for the smallest year-to-year swings, accepting "
                         "lower returns in exchange for a steadier ride.",
    "Max Risk-Adjusted": "It aims for the most return per unit of risk taken -- the "
                         "best trade-off between growth and bumps along the way.",
    "Tail-Risk CVaR":    "It aims to limit how much you could lose in the worst years, "
                         "trading some upside for protection against crashes.",
}
GOAL_TRADEOFF = {
    "Minimum Variance":  "Trades away some growth for the steadiest ride -- the "
                         "smallest expected swings of the three.",
    "Max Risk-Adjusted": "Chases the best growth-for-risk balance -- usually more "
                         "return than the lowest-risk option, with bigger swings.",
    "Tail-Risk CVaR":    "Built to soften the worst years -- it accepts some everyday "
                         "bumpiness to limit how deep a crash could go.",
}

# The dashboard's verbatim methodology note (under its risk table).
METHODOLOGY_NOTE = (
    "Risk is estimated by simulating thousands of possible future years based on how "
    "your holdings have behaved historically. 'Average Loss During Crashes' is the "
    "average outcome in the worst 5% of those simulated years. These estimates assume "
    "the future resembles the past and cannot predict unprecedented events."
)

# Friendly period text for each crisis window in stress_test.json.
CRISIS_PERIOD = {
    "2008 GFC":      "the 2008 global financial crisis (late 2007 to early 2009)",
    "COVID crash":   "the COVID-19 crash (February to March 2020)",
    "2022 drawdown": "the 2022 market drawdown (January to October 2022)",
    "dot-com":       "the dot-com crash (2000 to 2002)",
}

# Rows in resampled_portfolios.xlsx that are not ticker holdings.
_SKIP_LABELS = {
    "Portfolio Return (%)", "Portfolio Volatility (%)",
    "Portfolio Sharpe Ratio", "", "nan", "none",
}

NA = "_This section is unavailable: the required data could not be read._"


# ── Small helpers ───────────────────────────────────────────────────────────────

def _pct(x, dp=1):
    try:
        return f"{float(x) * 100:.{dp}f}%"
    except (ValueError, TypeError):
        return "n/a"

def _pct_raw(x, dp=1):
    """Format a value already expressed in percent (e.g. 35.0 -> '35%')."""
    try:
        return f"{float(x):.{dp}f}%"
    except (ValueError, TypeError):
        return "n/a"

def _money(amount, currency):
    """Format a money amount in the display currency (₹ for INR, $ for USD)."""
    sym = "₹" if currency == "INR" else "$"
    try:
        amt = float(amount)
    except (ValueError, TypeError):
        return f"{sym}?"
    if currency == "INR":
        return f"{sym}{amt:,.0f}"
    return f"{sym}{amt:,.2f}"

def _is_summary_label(name):
    """True for non-ticker spacer/summary rows (e.g. 'Portfolio CVaR 95% (%)')."""
    s = str(name).strip()
    return (not s) or s.lower() in {x.lower() for x in _SKIP_LABELS} \
        or s.lower().startswith("portfolio ") or "(%)" in s


# ── Loaders (each returns None / {} on failure; never raises) ────────────────────

def load_config():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def load_goal_sheet(sheet):
    """
    Parse one resampled goal sheet into:
      ({ticker: weight_pct} in sheet order, forward_return_pct or None)
    Skips spacer/summary rows; pulls the 'Portfolio Return (%)' forward estimate.
    Returns (None, None) on failure.
    """
    try:
        df = pd.read_excel(RESAMPLED_PATH, sheet_name=sheet)
    except Exception:
        return None, None
    weights, fwd_ret = {}, None
    for _, row in df.iterrows():
        stock = row.get("Stock")
        wt    = row.get("Weight (%)")
        if not isinstance(stock, str) or not stock.strip():
            continue
        if stock.strip() == "Portfolio Return (%)":
            try:
                fwd_ret = float(wt)
            except (TypeError, ValueError):
                pass
            continue
        if _is_summary_label(stock):
            continue
        try:
            w = float(wt)
        except (TypeError, ValueError):
            continue
        if w > 0:
            weights[stock.strip()] = w
    return (weights or None), fwd_ret

def list_goal_sheets():
    try:
        return list(pd.ExcelFile(RESAMPLED_PATH).sheet_names)
    except Exception:
        return []

def load_risk():
    try:
        with open(RISK_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def load_robustness():
    try:
        with open(ROBUST_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def load_stress():
    try:
        with open(STRESS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def load_blended_rf():
    """Blended risk-free rate from risk_free_rates.json (0.0 on any failure)."""
    try:
        with open(RF_PATH, encoding="utf-8") as f:
            return float(json.load(f).get("blended_rate", 0.0))
    except Exception:
        return 0.0

def load_factor_inputs():
    """
    (capm_mu_map, cov_df, rf) for the factor-model Risk Efficiency (Sharpe),
    mirroring app.py: CAPM expected returns + annualised covariance + blended rf.
    Returns (None, None, 0.0) on any failure.
    """
    try:
        capm = pd.read_excel(RETURNS_PATH, sheet_name="CAPM Returns")
        mu_map = dict(zip(capm["Ticker"], capm["CAPM_Expected_Return"].astype(float)))
        cov = pd.read_excel(RETURNS_PATH, sheet_name="Annualised Cov", index_col=0)
        return mu_map, cov, load_blended_rf()
    except Exception:
        return None, None, 0.0

def load_prices():
    """Price history DataFrame (date index, ticker columns), or empty on failure."""
    try:
        return pd.read_excel(PRICES_PATH, index_col=0, parse_dates=True)
    except Exception:
        return pd.DataFrame()


# ── Display-layer maths (identical definitions to app.py) ────────────────────────

def portfolio_sharpe(weights, mu_map, cov_df, rf):
    """
    Factor-model Risk Efficiency, computed exactly as module2/app.py do:
        (capm_mu . w  -  rf) / sqrt(wᵀ Σ w)
    `weights` is {ticker: weight} at any scale (renormalised over cov_df tickers).
    Returns float, or nan if it cannot be computed.
    """
    if not weights or mu_map is None or cov_df is None:
        return float("nan")
    tickers = [t for t in cov_df.index if t in weights and float(weights[t]) > 0]
    if not tickers:
        return float("nan")
    w = np.array([float(weights[t]) for t in tickers], dtype=float)
    if w.sum() <= 0:
        return float("nan")
    w = w / w.sum()
    mu  = np.array([float(mu_map.get(t, 0.0)) for t in tickers], dtype=float)
    Sig = cov_df.loc[tickers, tickers].values.astype(float)
    var = float(w @ Sig @ w)
    if var <= 0:
        return float("nan")
    return (float(mu @ w) - rf) / (var ** 0.5)

def one_year_return(ticker, prices_df):
    """Trailing ~1Y price return (%) for one ticker, or None (mirrors app.py)."""
    try:
        if prices_df.empty or ticker not in prices_df.columns:
            return None
        s = prices_df[ticker].dropna()
        if len(s) < 2:
            return None
        latest   = float(s.iloc[-1])
        year_ago = float(s.iloc[-252]) if len(s) >= 252 else float(s.iloc[0])
        return (latest - year_ago) / year_ago * 100.0
    except Exception:
        return None


# ── Plan + chosen-goal resolution ───────────────────────────────────────────────

def latest_plan_path():
    plans = glob.glob(os.path.join(SCRIPT_DIR, "rebalancing_plan_*.xlsx"))
    return max(plans, key=os.path.getmtime) if plans else None

def load_plan(path):
    """Return (summary_dict, instructions_df, current_holdings_df) or (None, None, None)."""
    if not path:
        return None, None, None
    try:
        summ_df = pd.read_excel(path, sheet_name="Summary")
        summary = {str(r["Field"]).strip(): r["Value"] for _, r in summ_df.iterrows()}
    except Exception:
        summary = None
    try:
        instr = pd.read_excel(path, sheet_name="Rebalancing Instructions")
    except Exception:
        instr = None
    try:
        cur = pd.read_excel(path, sheet_name="Current Holdings")
    except Exception:
        cur = None
    return summary, instr, cur

def resolve_choice(config, plan_summary, robustness, sheets):
    """Decide which goal the report is about (CLI > config > plan > robustness > first)."""
    if len(sys.argv) > 1 and sys.argv[1].strip():
        return sys.argv[1].strip()
    if config.get("portfolio_choice"):
        return config["portfolio_choice"]
    if plan_summary and plan_summary.get("Target Portfolio"):
        return str(plan_summary["Target Portfolio"]).strip()
    for v in (robustness or {}).values():
        if isinstance(v, dict) and v.get("portfolio_choice"):
            return v["portfolio_choice"]
    return sheets[0] if sheets else None


# ── Section builders (return Markdown strings) ──────────────────────────────────

def section_title(choice, report_date):
    label = GOAL_LABELS.get(choice, choice or "Your Portfolio")
    one   = GOAL_ONE_LINER.get(choice, "")
    lines = [
        "# Your Portfolio Report",
        "",
        f"**Date:** {report_date}",
        "",
        f"**Your chosen goal:** {label}"
        + (f" _(model name: {choice})_" if choice and choice in GOAL_LABELS else ""),
        "",
    ]
    if one:
        lines += [one, ""]
    return "\n".join(lines)


def section_recommended(choice, rec_weights, fwd_ret, risk, factor_inputs, prices_df):
    """
    Current-vs-recommended weights, then TWO clearly separated returns:
      * the current portfolio's ACTUAL return over the past year (backward),
      * the recommended portfolio's PROJECTED return (forward).
    For the Max Risk-Adjusted goal, a Risk Efficiency (current vs recommended) line
    is added -- matching the extra row the dashboard shows for that goal only.
    """
    lines = ["## 1. Your recommended portfolio", ""]
    if not rec_weights:
        return "\n".join(lines + [NA, ""])

    ports  = (risk or {}).get("portfolios", {})
    cur_w  = (ports.get("Current Portfolio") or {}).get("weights", {})  # fractions
    cur_w  = {t: float(w) for t, w in (cur_w or {}).items()}
    has_current = bool(cur_w)

    if has_current:
        lines.append("Here is your recommended mix, shown beside what you hold today. "
                     "Each line reads as: now → recommended.")
    else:
        lines.append("You didn't enter any current holdings, so this is a fresh "
                     "portfolio. Here is the recommended mix:")
    lines.append("")

    for t, w in rec_weights.items():          # sheet order
        if has_current:
            cur_share = cur_w.get(t, 0.0)
            now_s = _pct(cur_share, 0) if cur_share > 0 else "not held"
            lines.append(f"- **{t}** — now {now_s} → recommended {_pct_raw(w, 0)}")
        else:
            lines.append(f"- **{t}** — {_pct_raw(w, 0)}")
    lines.append("")

    # Max Risk-Adjusted only: Risk Efficiency, current vs recommended.
    if choice == "Max Risk-Adjusted" and factor_inputs:
        mu_map, cov, rf = factor_inputs
        se_rec = portfolio_sharpe(rec_weights, mu_map, cov, rf)
        se_cur = portfolio_sharpe(cur_w, mu_map, cov, rf) if has_current else float("nan")
        if se_rec == se_rec:        # not NaN
            if se_cur == se_cur:
                lines.append(
                    f"- **Risk Efficiency** — now {se_cur:.2f} → recommended "
                    f"{se_rec:.2f} (higher means more return for the risk taken)"
                )
            else:
                lines.append(
                    f"- **Risk Efficiency (recommended):** {se_rec:.2f} "
                    "(higher means more return for the risk taken)"
                )
            lines.append("")

    # Two clearly-separated returns -- never a single before/after number.
    realized = None
    if has_current and not prices_df.empty:
        num = den = 0.0
        for t, w in cur_w.items():
            r1 = one_year_return(t, prices_df)
            if r1 is not None:
                num += w * r1
                den += w
        realized = (num / den) if den > 0 else None

    if realized is not None:
        lines.append(f"- **Your current portfolio's actual return over the past year:** "
                     f"{realized:.1f}% (what already happened).")
    elif has_current:
        lines.append("- **Your current portfolio's actual return over the past year:** "
                     "not available (not enough price history).")
    if fwd_ret is not None:
        lines.append(f"- **This recommended portfolio's projected return (a forward "
                     f"estimate, not a promise):** about {_pct_raw(fwd_ret)} a year.")
    lines.append("")
    lines.append("These two numbers measure different things: the first is your current "
                 "mix's real past year; the second is a model projection for the "
                 "recommended mix. They are not a before-and-after of the same thing.")
    lines.append("")
    return "\n".join(lines)


def section_changes(plan_summary, instr, current_df, currency):
    lines = ["## 2. What to change", ""]
    if instr is None or plan_summary is None:
        return "\n".join(lines + [NA, ""])

    cur = currency or str(plan_summary.get("Display Currency", "USD")).strip()
    amt_col = "Amount (INR)" if cur == "INR" else "Amount (USD)"
    if amt_col not in instr.columns:
        amt_col = "Amount (USD)"

    holds_nothing = current_df is None or len(current_df) == 0

    actioned = instr[instr["Status"].astype(str).str.contains("Recommended", na=False)] \
        if "Status" in instr.columns else instr
    # Defensive: drop any non-ticker summary row that leaked into the instructions.
    actioned = actioned[~actioned["Ticker"].map(_is_summary_label)]

    if holds_nothing:
        lines.append("You don't hold anything yet, so this is a **fresh portfolio**. "
                     "To build it, buy:")
        lines.append("")
        for _, r in actioned.iterrows():
            lines.append(f"- **Buy {_money(r.get(amt_col), cur)}** of {r['Ticker']}")
        lines.append("")
        return "\n".join(lines)

    if len(actioned) == 0:
        lines.append("No trades are needed — your current holdings already match the "
                     "recommended mix closely enough (small trades under 1% are skipped).")
        lines.append("")
        return "\n".join(lines)

    lines.append("To move from what you hold today to the recommended mix, make these "
                 "trades (small trades under 1% have already been dropped):")
    lines.append("")
    for _, r in actioned.iterrows():
        action = str(r.get("Action", "")).strip().upper()
        verb = "Buy" if action == "BUY" else "Sell"
        lines.append(f"- **{verb} {_money(r.get(amt_col), cur)}** of {r['Ticker']}")
    lines.append("")
    return "\n".join(lines)


def _g(p, *keys):
    d = p
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d

def _cmp_phrase(cur, rec, lower_is_better=True):
    """'down from X (better)' style phrase, or '' if no current to compare."""
    if cur is None:
        return ""
    cur_s = _pct(cur)
    try:
        if rec < cur:
            tag = "lower — better" if lower_is_better else "lower — worse"
            return f", down from {cur_s} for your current mix ({tag})"
        if rec > cur:
            tag = "higher — worse" if lower_is_better else "higher — better"
            return f", up from {cur_s} for your current mix ({tag})"
        return f", unchanged from your current mix ({cur_s})"
    except TypeError:
        return f" (current mix: {cur_s})"


def section_risk(choice, risk, factor_inputs):
    lines = ["## 3. The risk picture (current vs recommended)", ""]
    ports = (risk or {}).get("portfolios", {})
    rec = ports.get(choice)
    if not rec:
        return "\n".join(lines + [NA, ""])
    cur = ports.get("Current Portfolio")

    if cur:
        lines.append("Here is how the recommended portfolio compares with what you hold "
                     "today. For the loss figures, lower is better; for Risk Efficiency, "
                     "higher is better.")
    else:
        lines.append("You have no current portfolio to compare against, so these are the "
                     "figures for the recommended portfolio. For the loss figures, lower "
                     "is better; for Risk Efficiency, higher is better.")
    lines.append("")

    # Volatility
    lines.append(
        f"- **Volatility (typical year-to-year swing): "
        f"{_pct(_g(rec, 'volatility', 'annualized'))}**"
        f"{_cmp_phrase(_g(cur, 'volatility', 'annualized') if cur else None, _g(rec, 'volatility', 'annualized'))}. "
        "This is roughly how much the portfolio's value tends to move in a typical year."
    )
    # Typical Worst-Year Loss (annual VaR 95%)
    lines.append(
        f"- **Typical Worst-Year Loss: {_pct(_g(rec, 'var_95', 'annual_loss'))}**"
        f"{_cmp_phrase(_g(cur, 'var_95', 'annual_loss') if cur else None, _g(rec, 'var_95', 'annual_loss'))}. "
        "In a bad year (worse than 19 out of 20), you could lose about this much."
    )
    # Average Loss During Crashes (annual CVaR 95%)
    lines.append(
        f"- **Average Loss During Crashes: {_pct(_g(rec, 'cvar_95', 'annual_loss'))}**"
        f"{_cmp_phrase(_g(cur, 'cvar_95', 'annual_loss') if cur else None, _g(rec, 'cvar_95', 'annual_loss'))}. "
        "Averaged across the worst 5% of simulated years, this is the typical loss."
    )
    # Chance of Large Loss
    lines.append(
        f"- **Chance of Large Loss: {_pct(_g(rec, 'chance_large_loss', 'probability'))}**"
        f"{_cmp_phrase(_g(cur, 'chance_large_loss', 'probability') if cur else None, _g(rec, 'chance_large_loss', 'probability'))}. "
        "This is the modelled chance of a loss bigger than 10% over the period."
    )
    # Worst Expected Decline (max drawdown p95)
    lines.append(
        f"- **Worst Expected Decline: {_pct(_g(rec, 'max_drawdown', 'p95_worst'))}**"
        f"{_cmp_phrase(_g(cur, 'max_drawdown', 'p95_worst') if cur else None, _g(rec, 'max_drawdown', 'p95_worst'))}. "
        "This is roughly the largest peak-to-trough drop you might see in a year."
    )
    # Risk Efficiency (factor-model Sharpe), higher is better.
    if factor_inputs:
        mu_map, cov, rf = factor_inputs
        se_rec = portfolio_sharpe(rec.get("weights", {}), mu_map, cov, rf)
        se_cur = portfolio_sharpe(cur.get("weights", {}), mu_map, cov, rf) if cur else float("nan")
        if se_rec == se_rec:
            cmp = ""
            if se_cur == se_cur:
                if se_rec > se_cur:
                    cmp = f", up from {se_cur:.2f} for your current mix (higher — better)"
                elif se_rec < se_cur:
                    cmp = f", down from {se_cur:.2f} for your current mix (lower — worse)"
                else:
                    cmp = f", unchanged from your current mix ({se_cur:.2f})"
            lines.append(
                f"- **Risk Efficiency: {se_rec:.2f}**{cmp}. "
                "This is how much return the mix earns for the risk it takes -- "
                "higher is better."
            )
    lines.append("")

    # Forward expected-return note.
    er = rec.get("expected_return", {})
    er_val = er.get("geometric_annualized") or er.get("compounded_annualized") \
        or er.get("annualized")
    if er_val is not None:
        lines.append(
            f"**Expected return:** about {_pct(er_val)} a year, as a model estimate "
            "based on past data -- not a promise. Actual results will vary, and some "
            "years will be negative."
        )
        lines.append("")

    # The dashboard's verbatim methodology note.
    lines.append(f"_{METHODOLOGY_NOTE}_")
    lines.append("")
    return "\n".join(lines)


def section_awareness(robustness):
    """
    Diversification only -- correlation and sector concentration -- with the same
    plain-language ✅/⚠️ wording and numbers the dashboard uses. Momentum is
    intentionally NOT shown here (it still runs upstream and is recorded in
    robustness_warnings.json, but the dashboard omits it).
    """
    lines = ["## 4. Things to be aware of", ""]
    if not robustness:
        return "\n".join(lines + [NA, ""])
    lines.append("We checked how well the recommended portfolio is diversified. "
                 "Here is what we found:")
    lines.append("")

    # Correlation
    corr  = robustness.get("high_correlation", {})
    pairs = corr.get("flagged_pairs", [])
    if corr.get("triggered") and pairs:
        for p in pairs:
            lines.append(
                f"- **Correlation:** ⚠️ {p['stock_a']} and {p['stock_b']} move almost "
                f"identically (correlation {p['correlation']:.2f}) — holding both adds "
                "little diversification."
            )
    else:
        thr = corr.get("threshold", 0.85)
        lines.append(
            f"- **Correlation:** ✅ Your holdings move independently enough to "
            f"diversify well (no pair above {thr:.2f})."
        )

    # Sector concentration
    sec     = robustness.get("sector_concentration", {})
    largest = sec.get("largest_sector")
    share   = sec.get("largest_share")
    eff     = sec.get("effective_sectors")
    if largest is not None and share is not None and eff is not None:
        if sec.get("triggered"):
            lines.append(
                f"- **Concentration:** ⚠️ {share * 100:.0f}% of this portfolio is in one "
                f"sector ({largest}) — effective spread about {eff:.1f} sectors."
            )
        else:
            lines.append(
                f"- **Concentration:** ✅ Well spread across sectors (largest is "
                f"{largest} at {share * 100:.0f}%; effective spread about "
                f"{eff:.1f} sectors)."
            )
    else:
        lines.append("- **Concentration:** _sector data unavailable._")
    lines.append("")
    return "\n".join(lines)


def section_options(choice, compare_goals, risk):
    """
    LOAD-BEARING cross-goal comparison. The dashboard shows only the chosen goal vs
    the current portfolio, so this is the ONLY place the user sees all three goals
    side by side. Each goal lists its key risk figures plus a plain-language note on
    what it trades off.
    """
    lines = ["## 5. Comparing all three options", ""]
    ports = (risk or {}).get("portfolios", {})
    if not compare_goals:
        return "\n".join(lines + [NA, ""])

    lines.append("Your model built three portfolios from the same holdings, each tuned "
                 f"to a different goal. You chose **{GOAL_LABELS.get(choice, choice)}**. "
                 "The dashboard only shows the one you picked, so here is how all three "
                 "compare so you can see the trade-offs:")
    lines.append("")

    for sheet in compare_goals:
        label = GOAL_LABELS.get(sheet, sheet)
        chosen_tag = "  _(your choice)_" if sheet == choice else ""
        lines.append(f"### {label}{chosen_tag}")
        lines.append("")
        p = ports.get(sheet)
        if isinstance(p, dict):
            lines.append(
                f"- Typical year-to-year swing: {_pct(_g(p, 'volatility', 'annualized'))}"
            )
            lines.append(
                f"- Typical Worst-Year Loss: {_pct(_g(p, 'var_95', 'annual_loss'))}"
            )
            lines.append(
                f"- Average Loss During Crashes: {_pct(_g(p, 'cvar_95', 'annual_loss'))}"
            )
            lines.append(
                f"- Worst Expected Decline: {_pct(_g(p, 'max_drawdown', 'p95_worst'))}"
            )
            er = p.get("expected_return", {})
            er_val = er.get("geometric_annualized") or er.get("compounded_annualized") \
                or er.get("annualized")
            if er_val is not None:
                lines.append(f"- Projected return (forward estimate): about {_pct(er_val)} a year")
        else:
            lines.append("- _Risk figures for this option are unavailable._")
        tradeoff = GOAL_TRADEOFF.get(sheet)
        if tradeoff:
            lines.append(f"- **What it trades off:** {tradeoff}")
        lines.append("")

    lines.append("In short: the lower-risk option gives up some growth for a steadier "
                 "ride, the crash-protection option focuses on softening the worst years, "
                 "and the balanced option sits between them.")
    lines.append("")
    return "\n".join(lines)


def _fmt_tickers(names):
    """Join tickers as prose: 'A', 'A and B', or 'A, B and C'."""
    names = list(names)
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _modeled_reason(detail):
    """Plain-language reason a holding had to be modeled, from its breakdown detail."""
    d = (detail or "").lower()
    if "listed later" in d or "before this window" in d:
        return "it listed after this period"
    if "this far back" in d:
        return "its price history doesn't reach this far back"
    return "it has no market history for this period"


def section_stress(choice, stress):
    """
    Historical crisis replay from stress_test.json. For each window, how the CHOSEN
    portfolio WOULD have performed -- using the NEW per-window basis tags. Each holding
    is 'actual' (real history), 'modeled' (reconstructed from its factor exposures --
    an estimate, not an observation), or 'insufficient' (excluded). A blended window
    MUST name which holdings were modeled and flag the figure as an estimate; a fully
    measured window reads plainly. Never says "based on all N holdings" for a blend.
    """
    lines = ["## 6. How this portfolio would have handled past crises", ""]
    windows = (stress or {}).get("windows") if isinstance(stress, dict) else None
    if not windows:
        return "\n".join(lines + [
            "_This section is unavailable: the historical stress test could not be read._",
            "",
        ])

    lines.append("We replayed the recommended mix through four real market crises. For "
                 "each one we show two facts about this same portfolio: where it would "
                 "have **ended over the full period**, and the **deepest drop** it would "
                 "have taken along the way (it may have partly recovered by the end). "
                 "Where a holding was trading at the time, we use its **actual** history. "
                 "Where it hadn't been listed yet, we **estimate** its return from its "
                 "factor exposures (how it tends to move with the market and similar "
                 "stocks) rather than dropping it — and we say so, per crisis. An "
                 "estimated figure is a model reconstruction, not something that actually "
                 "happened.")
    lines.append("")

    # Keep the familiar chronological-ish ordering used elsewhere.
    order = ["2008 GFC", "COVID crash", "2022 drawdown", "dot-com"]
    ordered = [w for w in order if w in windows] + \
              [w for w in windows if w not in order]

    for name in ordered:
        w = windows.get(name, {})
        period = CRISIS_PERIOD.get(name, name)
        cum = w.get("cumulative_return")
        dd  = w.get("max_drawdown")
        basis        = w.get("basis")
        actual_h     = list(w.get("actual_holdings", []) or [])
        modeled_h    = list(w.get("modeled_holdings", []) or [])
        insufficient = list(w.get("insufficient_holdings", []) or [])
        detail_of    = {h.get("ticker"): h.get("detail")
                        for h in (w.get("holdings", []) or [])}

        # No figure at all -> honest "insufficient" line, naming the excluded names.
        if cum is None:
            excl = _fmt_tickers(insufficient) or "the holdings"
            lines.append(
                f"- **{name}** — {period}: not enough data to estimate. Neither real "
                f"history nor factor data reaches this far back for {excl}, so no figure "
                "is shown — nothing is assumed."
            )
            continue

        is_blended = bool(modeled_h)
        direction = "down" if cum < 0 else "up"
        # Two facts about the SAME (recommended) portfolio: where it ended over the
        # full window, and its deepest peak-to-trough drop along the way.
        if is_blended:
            head = (f"**{name}** — {period}: your recommended portfolio would have "
                    f"ended this period {direction} about {abs(cum) * 100:.0f}% "
                    f"(estimated), having fallen as much as about {abs(dd) * 100:.0f}% "
                    "at its worst point along the way.")
        else:
            head = (f"**{name}** — {period}: your recommended portfolio would have "
                    f"ended this period {direction} about {abs(cum) * 100:.0f}%, "
                    f"having fallen as much as about {abs(dd) * 100:.0f}% at its worst "
                    "point along the way.")

        sentence = head
        if not modeled_h and not insufficient:
            # Fully measured: read plainly, no per-ticker itemisation.
            sentence += " Based on actual history for all holdings."
        else:
            # Mixed: name which holdings are actual vs modeled vs excluded.
            disc = []
            if actual_h:
                disc.append(f"{_fmt_tickers(actual_h)} from actual history")
            if modeled_h:
                reasons = {}
                for t in modeled_h:
                    reasons.setdefault(_modeled_reason(detail_of.get(t)), []).append(t)
                for reason, ts in reasons.items():
                    disc.append(f"{_fmt_tickers(ts)} modeled from "
                                f"{'its' if len(ts) == 1 else 'their'} factor exposures "
                                f"({reason})")
            if insufficient:
                disc.append(f"{_fmt_tickers(insufficient)} excluded (no data this far back)")
            # Capitalise only the first letter (str.capitalize() would lowercase
            # the tickers that follow, e.g. 'RELIANCE.NS' -> 'Reliance.ns').
            disc[0] = disc[0][0].upper() + disc[0][1:]
            sentence += " " + "; ".join(disc) + "."
            if is_blended:
                sentence += " The modeled portion is an estimate, not an observation."
        lines.append(f"- {sentence}")

    lines.append("")
    lines.append("These are historical replays, not forecasts — and where a holding is "
                 "marked *modeled*, that part is a factor-based estimate, not a measured "
                 "outcome. The next crisis may look different from any of these.")
    lines.append("")
    return "\n".join(lines)


def section_footer(report_date):
    return "\n".join([
        "---",
        "",
        "_This report is informational only and is not financial advice._  ",
        f"_Based on market data as of {report_date}. Markets change; review regularly._",
        "",
    ])


# ── Orchestration ───────────────────────────────────────────────────────────────

def build_report():
    config      = load_config()
    sheets      = list_goal_sheets()
    plan_path   = latest_plan_path()
    plan_summary, instr, current_df = load_plan(plan_path)
    risk        = load_risk()
    robustness  = load_robustness()
    stress      = load_stress()

    choice = resolve_choice(config, plan_summary, robustness, sheets)

    # Report date: prefer the plan's date, else today.
    report_date = None
    if plan_summary and plan_summary.get("Date"):
        report_date = str(plan_summary["Date"]).strip()
    if not report_date:
        report_date = datetime.now().strftime("%Y-%m-%d")

    currency = config.get("currency")
    if not currency and plan_summary:
        currency = str(plan_summary.get("Display Currency", "")).strip() or None

    rec_weights, fwd_ret = load_goal_sheet(choice) if choice else (None, None)

    # Display-layer inputs shared by the recommended + risk sections.
    factor_inputs = load_factor_inputs()
    if factor_inputs[0] is None:
        factor_inputs = None
    prices_df = load_prices()

    # Section 5 compares ALL three goals. Layer 7 resamples only the chosen goal, so
    # source the goal list from the Layer 5 risk summary (which evaluates all three);
    # fall back to resampled sheet names only if the risk summary is unavailable.
    risk_ports = (risk or {}).get("portfolios", {})
    compare_goals = [g for g in GOAL_LABELS if g in risk_ports] or sheets

    parts = [
        section_title(choice, report_date),
        section_recommended(choice, rec_weights, fwd_ret, risk, factor_inputs, prices_df),
        section_changes(plan_summary, instr, current_df, currency),
        section_risk(choice, risk, factor_inputs),
        section_awareness(robustness),
        section_options(choice, compare_goals, risk),
        section_stress(choice, stress),
        section_footer(report_date),
    ]
    return "\n".join(parts), report_date


def main():
    print(f"\n{'=' * 60}")
    print("  MODULE 5  --  PLAIN-LANGUAGE MARKDOWN REPORT")
    print(f"{'=' * 60}")

    report_md, report_date = build_report()

    stamp = datetime.now().strftime("%Y%m%d")
    out_path = os.path.join(SCRIPT_DIR, f"portfolio_report_{stamp}.md")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        print(f"  Report written -> {os.path.basename(out_path)}")
        print(f"  Market data as of: {report_date}")
    except Exception as exc:
        print(f"  [FAIL] could not write report: {exc}")
        sys.exit(1)
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
