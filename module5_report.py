#!/usr/bin/env python3
"""
module5_report.py  --  Module 5 report builder

Pulls the pipeline's existing outputs into ONE plain-language Markdown report that
a retail investor can read top to bottom. It is NON-INTERACTIVE and reads existing
artifacts only -- no recomputation. Every input is optional: if a file is missing
or unreadable, the affected section degrades gracefully with a clear note rather
than crashing.

Reads (all produced upstream)
-----------------------------
  run_config.json                 run context: tickers, holdings, currency, choice
  resampled_portfolios.xlsx       chosen-goal weights + all three goals (Layer 7)
  rebalancing_plan_*.xlsx (latest) buy/sell instructions + currency (Module 4)
  risk_evaluation_summary.json    chosen + Current Portfolio risk metrics (Layer 5)
  robustness_warnings.json        the three robustness checks (Layer 6)

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

import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ───────────────────────────────────────────────────────────────────────
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
RESAMPLED_PATH = os.path.join(SCRIPT_DIR, "resampled_portfolios.xlsx")
RISK_PATH      = os.path.join(SCRIPT_DIR, "risk_evaluation_summary.json")
ROBUST_PATH    = os.path.join(SCRIPT_DIR, "robustness_warnings.json")
CONFIG_PATH    = os.path.join(SCRIPT_DIR, "run_config.json")

# ── Plain-language labels for the three goals ───────────────────────────────────
GOAL_LABELS = {
    "Minimum Variance":  "Lowest Risk",
    "Max Risk-Adjusted": "Best Balance of Risk and Return",
    "Tail-Risk CVaR":    "Crash Protection",
}
GOAL_ONE_LINER = {
    "Minimum Variance":  "It aims for the smallest month-to-month swings, accepting "
                         "lower returns in exchange for a steadier ride.",
    "Max Risk-Adjusted": "It aims for the most return per unit of risk taken -- the "
                         "best trade-off between growth and bumps along the way.",
    "Tail-Risk CVaR":    "It aims to limit how much you could lose in the worst months, "
                         "trading some upside for protection against crashes.",
}
GOAL_DIFFERENCE = {
    "Minimum Variance":  "steadiest, with the least expected volatility",
    "Max Risk-Adjusted": "chases the best growth-for-risk trade-off",
    "Tail-Risk CVaR":    "built to soften the worst-case months",
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

def _money(amount, currency):
    """Format a money amount in the display currency."""
    sym = "₹" if currency == "INR" else "$"
    try:
        amt = float(amount)
    except (ValueError, TypeError):
        return f"{sym}?"
    if currency == "INR":
        return f"{sym}{amt:,.0f}"
    return f"{sym}{amt:,.2f}"


# ── Loaders (each returns None / {} on failure; never raises) ────────────────────

def load_config():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def load_goal_weights(sheet):
    """{ticker: weight_fraction} for one resampled sheet, or None."""
    try:
        df = pd.read_excel(RESAMPLED_PATH, sheet_name=sheet)
    except Exception:
        return None
    out = {}
    for _, row in df.iterrows():
        stock = str(row.get("Stock", "")).strip()
        if (stock.lower() in {s.lower() for s in _SKIP_LABELS}
                or stock.lower().startswith("portfolio ")):
            continue
        try:
            wt = float(row.get("Weight (%)"))
        except (ValueError, TypeError):
            continue
        if wt > 0:
            out[stock] = wt / 100.0
    return out or None

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


# ── Chosen-goal resolution ──────────────────────────────────────────────────────

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
        f"# Your Portfolio Report",
        "",
        f"**Date:** {report_date}",
        "",
        f"**Your chosen goal:** {label}"
        + (f" _(model name: {choice})_" if choice and choice in GOAL_LABELS else ""),
        "",
    ]
    if one:
        lines.append(one)
        lines.append("")
    return "\n".join(lines)


def section_recommended(choice, weights):
    lines = ["## 2. Your recommended portfolio", ""]
    if not weights:
        lines += [NA, ""]
        return "\n".join(lines)
    ordered = sorted(weights.items(), key=lambda kv: -kv[1])
    lines.append("Your recommended mix is:")
    lines.append("")
    for t, w in ordered:
        lines.append(f"- **{t}** — {_pct(w, 0)}")
    lines.append("")
    top_t, top_w = ordered[0]
    rest_items = [f"{t} ({_pct(w, 0)})" for t, w in ordered[1:]]
    if len(rest_items) > 1:
        rest = ", ".join(rest_items[:-1]) + " and " + rest_items[-1]
    else:
        rest = rest_items[0] if rest_items else ""
    if rest:
        lines.append(
            f"This mix leans most heavily on {top_t} ({_pct(top_w, 0)}), "
            f"with {rest} making up the rest."
        )
    else:
        lines.append(f"This mix is held entirely in {top_t}.")
    lines.append("")
    return "\n".join(lines)


def section_changes(plan_summary, instr, current_df, currency):
    lines = ["## 3. What to change", ""]
    if instr is None or plan_summary is None:
        lines += [NA, ""]
        return "\n".join(lines)

    # Currency + amount column
    cur = currency or str(plan_summary.get("Display Currency", "USD")).strip()
    amt_col = "Amount (INR)" if cur == "INR" else "Amount (USD)"
    if amt_col not in instr.columns:
        amt_col = "Amount (USD)"

    holds_nothing = current_df is None or len(current_df) == 0

    actioned = instr[instr["Status"].astype(str).str.contains("Actioned", na=False)] \
        if "Status" in instr.columns else instr

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
        action = str(r.get("Action", "")).strip().capitalize()
        verb = "Buy" if action.upper() == "BUY" else "Sell"
        lines.append(f"- **{verb} {_money(r.get(amt_col), cur)}** of {r['Ticker']}")
    lines.append("")
    return "\n".join(lines)


def _metric_block(name, current, recommended, plain, lower_is_better=True):
    """One risk metric: recommended value, optional before/after, plain meaning."""
    lines = []
    rec_s = _pct(recommended)
    if current is None:
        lines.append(f"- **{name}: {rec_s}.** {plain}")
        return lines
    cur_s = _pct(current)
    try:
        if recommended < current:
            direction = "lower" if lower_is_better else "higher"
            change = f"down from {cur_s} ({direction} — {'better' if lower_is_better else 'worse'})"
        elif recommended > current:
            direction = "higher" if lower_is_better else "lower"
            change = f"up from {cur_s} ({direction} — {'worse' if lower_is_better else 'better'})"
        else:
            change = f"unchanged from {cur_s}"
    except TypeError:
        change = f"(current: {cur_s})"
    lines.append(f"- **{name}: {rec_s}**, {change}. {plain}")
    return lines


def section_risk(choice, risk):
    lines = ["## 4. The risk picture (before vs after)", ""]
    ports = (risk or {}).get("portfolios", {})
    rec = ports.get(choice)
    if not rec:
        lines += [NA, ""]
        return "\n".join(lines)
    cur = ports.get("Current Portfolio")

    if cur:
        lines.append("Here is how the recommended portfolio compares with what you hold "
                     "today. Lower numbers mean less risk.")
    else:
        lines.append("You don't have a current portfolio to compare against, so these "
                     "are the figures for the recommended portfolio. Lower numbers mean "
                     "less risk.")
    lines.append("")

    def g(p, *keys):
        d = p
        for k in keys:
            if not isinstance(d, dict):
                return None
            d = d.get(k)
        return d

    lines += _metric_block(
        "Volatility (typical year-to-year swing)",
        g(cur, "volatility", "annualized") if cur else None,
        g(rec, "volatility", "annualized"),
        "This is roughly how much the portfolio's value tends to swing over a year.",
    )
    lines += _metric_block(
        "Bad-month loss (VaR 95%)",
        g(cur, "var_95", "monthly_loss") if cur else None,
        g(rec, "var_95", "monthly_loss"),
        "In a bad month (worse than 19 out of 20), you could lose about this much.",
    )
    lines += _metric_block(
        "Worst-months average loss (CVaR 95%)",
        g(cur, "cvar_95", "monthly_loss") if cur else None,
        g(rec, "cvar_95", "monthly_loss"),
        "And in those worst months, the average loss is about this much.",
    )
    lines += _metric_block(
        "Largest drop (max drawdown)",
        g(cur, "max_drawdown", "p95_worst") if cur else None,
        g(rec, "max_drawdown", "p95_worst"),
        "This is roughly the largest peak-to-trough drop you might see over a year.",
    )
    lines.append("")

    # Expected return: prefer a geometric/compounded figure if Layer 5 provides one.
    er = rec.get("expected_return", {})
    er_val = er.get("geometric_annualized") or er.get("compounded_annualized")
    geo = er_val is not None
    if er_val is None:
        er_val = er.get("annualized")
    if er_val is not None:
        kind = "a compounded long-run average" if geo else "an estimated long-run average"
        lines.append(
            f"**Expected return:** about {_pct(er_val)} a year, as {kind}. "
            "This is a model estimate based on past data, not a promise — actual "
            "results will vary, and some years will be negative."
        )
        lines.append("")
    return "\n".join(lines)


def section_awareness(robustness):
    lines = ["## 5. Things to be aware of", ""]
    if not robustness:
        lines += [NA, ""]
        return "\n".join(lines)
    order = [
        ("negative_momentum",    "Momentum"),
        ("high_correlation",     "Diversification"),
        ("sector_concentration", "Sector concentration"),
    ]
    lines.append("We ran three robustness checks on the recommended portfolio. "
                 "Here is what each found:")
    lines.append("")
    for key, title in order:
        c = robustness.get(key)
        if not isinstance(c, dict):
            lines.append(f"- **{title}:** _check result unavailable._")
            continue
        flag = "⚠️ " if c.get("triggered") else "✅ "
        msg = c.get("message", "No message.")
        lines.append(f"- {flag}**{title}:** {msg}")
    lines.append("")
    return "\n".join(lines)


def section_options(choice, sheets, risk):
    lines = ["## 6. The other options", ""]
    ports = (risk or {}).get("portfolios", {})
    if not sheets:
        lines += [NA, ""]
        return "\n".join(lines)
    lines.append("Your model produced three portfolios. You chose "
                 f"**{GOAL_LABELS.get(choice, choice)}**. Here is how all three compare:")
    lines.append("")
    for sheet in sheets:
        label = GOAL_LABELS.get(sheet, sheet)
        vol = None
        p = ports.get(sheet)
        if isinstance(p, dict):
            vol = p.get("volatility", {}).get("annualized")
        diff = GOAL_DIFFERENCE.get(sheet, "")
        chosen_tag = "  _(your choice)_" if sheet == choice else ""
        vol_s = f"typical swing about {_pct(vol)} a year" if vol is not None \
            else "risk figure unavailable"
        lines.append(f"- **{label}**{chosen_tag} — {vol_s}; {diff}.")
    lines.append("")
    lines.append("In short: lower-risk options trade away some growth for a steadier "
                 "ride, while higher-return options accept bigger swings.")
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

    weights = load_goal_weights(choice) if choice else None

    # Section 6 compares ALL three goals. Layer 7 now resamples only the chosen goal,
    # so resampled_portfolios.xlsx may hold a single sheet -- source the goal list from
    # the Layer 5 risk summary (which still evaluates all three) and fall back to the
    # resampled sheet names only if the risk summary is unavailable.
    risk_ports = (risk or {}).get("portfolios", {})
    compare_goals = [g for g in GOAL_LABELS if g in risk_ports] or sheets

    parts = [
        section_title(choice, report_date),
        section_recommended(choice, weights),
        section_changes(plan_summary, instr, current_df, currency),
        section_risk(choice, risk),
        section_awareness(robustness),
        section_options(choice, compare_goals, risk),
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
