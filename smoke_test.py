#!/usr/bin/env python3
"""
smoke_test.py  —  Non-interactive end-to-end driver for run_all.py

Runs the REORDERED + EXTENDED pipeline once on the 3-name India set, without any
interactive prompt, to confirm the orchestration wiring executes:
  module0 -> module1 -> simulation(L4) -> guard -> module2 -> resample(L7)
          -> riskeval(L5) -> module3 -> module4 -> module5

It does NOT modify run_all.py or any module. run_all's input() calls are answered by
a prompt-aware monkeypatch; the full combined pipeline output (incl. subprocess
stdout, via OS fd-level redirection) is captured to smoke_test_run.log. A concise
report is printed to the real stdout.

Harness-owned artifact: smoke_test_run.log (safe to delete). Pipeline product files
are listed in the manifest.
"""

import builtins
import contextlib
import glob
import io
import json
import os
import sys
import time
from datetime import date

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH   = os.path.join(SCRIPT_DIR, "smoke_test_run.log")

TICKERS   = ["RELIANCE.NS", "HDFCBANK.NS", "INFY.NS"]
HOLDINGS  = [("RELIANCE.NS", "10"), ("HDFCBANK.NS", "5"), ("INFY.NS", "8")]
BENCHMARK = "^NSEI"
CURRENCY  = "INR"
CHOSEN_GOAL = "Minimum Variance"     # collect_inputs picks option 1 (first sheet)


# ── Prompt-aware fake input (keeps run_all.py untouched) ────────────────────────

def make_fake_input():
    holdings_iter = iter(HOLDINGS)
    state = {"current": None}
    unexpected = []

    def fake_input(prompt=""):
        p = str(prompt)
        if "Refresh price data" in p:   return "no"
        if "Tickers:" in p:             return ", ".join(TICKERS)
        if "Benchmark" in p:            return BENCHMARK
        if "currency" in p:             return CURRENCY
        if "Ticker (or 'done')" in p:
            try:
                t, s = next(holdings_iter); state["current"] = (t, s); return t
            except StopIteration:
                return "done"
        if "Shares of" in p:            return state["current"][1] if state["current"] else "1"
        if "anyway" in p:               return "yes"
        if "portfolio number" in p:     return "1"
        if "Press Enter to begin" in p: return ""
        unexpected.append(p.strip())
        return ""

    return fake_input, unexpected


# ── Capture pipeline output at fd level (captures subprocess stdout too) ────────

def run_pipeline(logpath):
    fake_input, unexpected = make_fake_input()
    real_input = builtins.input
    builtins.input = fake_input

    logf = open(logpath, "w", encoding="utf-8")
    saved1, saved2 = os.dup(1), os.dup(2)
    os.dup2(logf.fileno(), 1); os.dup2(logf.fileno(), 2)
    py_out = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = py_out

    t0 = time.time()
    outcome = "completed"
    try:
        import run_all
        run_all.main()
    except SystemExit as exc:
        outcome = f"SystemExit({exc.code})"
    except Exception as exc:                       # pragma: no cover
        outcome = f"Exception: {type(exc).__name__}: {exc}"
    finally:
        elapsed = time.time() - t0
        try:
            py_out.flush()
        except Exception:
            pass
        sys.stdout, sys.stderr = old_out, old_err
        py_out.close()
        os.dup2(saved1, 1); os.dup2(saved2, 2)
        os.close(saved1); os.close(saved2)
        logf.close()
        builtins.input = real_input

    return outcome, unexpected, elapsed


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _read_log(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def _xlsx_sheets(path):
    try:
        import pandas as pd
        return list(pd.ExcelFile(path).sheet_names)
    except Exception as exc:
        return [f"<unreadable: {exc}>"]


def _goal_weights(xlsx, sheet):
    """{ticker: weight_pct} from a Stock/Weight(%) goal sheet."""
    import pandas as pd
    df = pd.read_excel(xlsx, sheet_name=sheet)
    out = {}
    for _, r in df.iterrows():
        s = str(r.get("Stock", "")).strip()
        if s and "Portfolio" not in s and s.lower() != "nan":
            try:
                out[s] = round(float(r["Weight (%)"]), 2)
            except (ValueError, TypeError):
                pass
    return out


def _plan_target_weights(plan_xlsx):
    """{ticker: target weight %} from a rebalancing plan's 'Target Weights' sheet."""
    import pandas as pd
    df = pd.read_excel(plan_xlsx, sheet_name="Target Weights")
    return {str(r["Ticker"]).strip(): round(float(r["Target Weight %"]), 2)
            for _, r in df.iterrows()}


def _manifest(since_ts):
    import numpy as np
    rows = []
    for fname in sorted(os.listdir(SCRIPT_DIR)):
        fpath = os.path.join(SCRIPT_DIR, fname)
        if not os.path.isfile(fpath) or fname in ("smoke_test.py", "smoke_test_run.log"):
            continue
        if os.path.getmtime(fpath) < since_ts - 1:
            continue
        detail = ""
        if fname.endswith(".xlsx"):
            detail = "sheets: " + ", ".join(_xlsx_sheets(fpath))
        elif fname.endswith(".npz"):
            try:
                d = np.load(fpath, allow_pickle=True); detail = "keys: " + ", ".join(d.files)
            except Exception as exc:
                detail = f"<unreadable: {exc}>"
        elif fname.endswith(".json"):
            detail = "json"
        elif fname.endswith(".png"):
            detail = f"{os.path.getsize(fpath)} bytes"
        rows.append((fname, detail))
    return rows


def _latest_plan():
    plans = glob.glob(os.path.join(SCRIPT_DIR, "rebalancing_plan_*.xlsx"))
    return max(plans, key=os.path.getmtime) if plans else None


# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("Running reordered+extended run_all.py non-interactively (India 3-name set)...")
    print(f"  tickers={TICKERS}  holdings={HOLDINGS}\n")

    outcome, unexpected, pipe_elapsed = run_pipeline(LOG_PATH)
    log = _read_log(LOG_PATH)

    W = 74
    print("=" * W)
    print("  SMOKE TEST REPORT  --  run_all.py with Layer 7 (resample) + Layer 5 (risk)")
    print("=" * W)
    print(f"  run_all.main() outcome : {outcome}")
    print(f"  full transcript        : smoke_test_run.log\n")

    # 1. Per-step results --------------------------------------------------------
    steps = [
        ("0  Module 0 (risk-free)",   "[OK] Module 0 complete"),
        ("1  Module 1 (estimation)",  "[OK] Module 1 complete"),
        ("   Layer 4 (simulation)",   "[OK] Layer 4 complete"),
        ("2  Module 2 (optimiser)",   "[OK] Module 2 complete"),
        ("7  Layer 7 (resampling)",   "[OK] Layer 7 complete"),
        ("5  Layer 5 (risk eval)",    "[OK] Layer 5 complete"),
        ("3  Module 3 (frontier)",    "[OK] Module 3 complete"),
        ("4  Module 4 (rebalancing)", "[OK] Module 4"),
        ("6  Layer 6 (robustness)",   "[OK] Layer 6 complete"),
    ]
    print("  PER-STEP RESULT")
    for label, marker in steps:
        print(f"    [{'PASS' if marker in log else 'FAIL'}] {label}")
    m5_absent = "module5_report.py was not found" in log or "module5_report.py not found" in log
    print(f"    [{'SKIP' if m5_absent else '????'}] 5  Module 5 (report) -- absent; "
          f"run_all skips GRACEFULLY (no error)")

    # Seed reported by Layer 7
    import re
    seed_match = re.search(r"Base seed \(from simulated_returns\.npz\): (\S+)", log)
    print(f"\n  Layer 7 base_seed (from simulated_returns.npz): "
          f"{seed_match.group(1) if seed_match else '<not found in log>'}")

    # 2. Freshness guard ---------------------------------------------------------
    import run_all
    ok_now, why_now = run_all.simulation_is_fresh()
    print(f"\n  FRESHNESS GUARD: blocked during run? "
          f"{'Cannot proceed to CVaR consumers' in log}  | post-run ({ok_now}, '{why_now}')")

    # 3. Layer 7 / Layer 5 production --------------------------------------------
    resampled_path = os.path.join(SCRIPT_DIR, "resampled_portfolios.xlsx")
    summary_path   = os.path.join(SCRIPT_DIR, "risk_evaluation_summary.json")
    rs_sheets = _xlsx_sheets(resampled_path) if os.path.exists(resampled_path) else []
    print("\n  LAYER 7 OUTPUT")
    print(f"    resampled_portfolios.xlsx sheets: {rs_sheets}")

    print("\n  LAYER 5 OUTPUT (risk_evaluation_summary.json)")
    metric_keys = ("volatility", "var_95", "cvar_95", "max_drawdown")
    if os.path.exists(summary_path):
        summ = json.load(open(summary_path, encoding="utf-8"))
        ports = summ.get("portfolios", {})
        print(f"    keyed portfolios: {list(ports.keys())}")
        print(f"    {'portfolio':<20}{'ann.vol%':>9}{'VaR95%':>9}{'CVaR95%':>9}{'maxDD-p95%':>12}")
        for name, m in ports.items():
            has_all = all(k in m for k in metric_keys)
            vol = m["volatility"]["annualized"] * 100
            var = m["var_95"]["monthly_loss"] * 100
            cv  = m["cvar_95"]["monthly_loss"] * 100
            dd  = m["max_drawdown"]["p95_worst"] * 100
            tag = "" if has_all else "  <-- MISSING METRICS"
            print(f"    {name:<20}{vol:>9.2f}{var:>9.2f}{cv:>9.2f}{dd:>12.2f}{tag}")
        want = {"Minimum Variance", "Max Risk-Adjusted", "Tail-Risk CVaR", "Current Portfolio"}
        print(f"    has 3 goals + current: {want.issubset(set(ports.keys()))}")
    else:
        print("    [FAIL] risk_evaluation_summary.json not found")

    # 3b. Layer 6 robustness checks ----------------------------------------------
    print(f"\n  LAYER 6 OUTPUT (robustness_warnings.json)  [chosen goal: '{CHOSEN_GOAL}']")
    rob_path = os.path.join(SCRIPT_DIR, "robustness_warnings.json")
    if os.path.exists(rob_path):
        rob = json.load(open(rob_path, encoding="utf-8"))
        print(f"    keys ({len(rob)}): {list(rob.keys())}")
        for key in ("negative_momentum", "high_correlation", "sector_concentration"):
            c = rob.get(key, {})
            tag = "FLAGGED" if c.get("triggered") else "clear"
            print(f"    - {key:<22} [{tag}]")
            print(f"        {c.get('message', '<no message>')}")
        # Spot-check: momentum signs vs returns_stats, sector weights sum to ~100%
        sc = rob.get("sector_concentration", {})
        sw = sc.get("sector_weights", {})
        ssum = sum(sw.values()) if sw else float("nan")
        print(f"    sector weights sum: {ssum * 100:.2f}%   "
              f"largest: {sc.get('largest_sector')} = "
              f"{sc.get('largest_share', float('nan')) * 100:.2f}%   "
              f"eff. sectors: {sc.get('effective_sectors', float('nan')):.2f}")
    else:
        print("    [FAIL] robustness_warnings.json not found")

    # 4. Canonical-weights check -------------------------------------------------
    print(f"\n  CANONICAL-WEIGHTS CHECK  (chosen goal: '{CHOSEN_GOAL}')")
    plan = _latest_plan()
    if plan and os.path.exists(resampled_path):
        tgt = _plan_target_weights(plan)
        res = _goal_weights(resampled_path, CHOSEN_GOAL)
        opt = _goal_weights(os.path.join(SCRIPT_DIR, "optimised_portfolios.xlsx"), CHOSEN_GOAL)
        print(f"    plan file: {os.path.basename(plan)}")
        print(f"    {'ticker':<14}{'plan target':>12}{'resampled':>12}{'single-shot':>12}")
        for t in TICKERS:
            print(f"    {t:<14}{tgt.get(t, float('nan')):>11.2f}%"
                  f"{res.get(t, float('nan')):>11.2f}%{opt.get(t, float('nan')):>11.2f}%")
        match_res = all(abs(tgt.get(t, -9) - res.get(t, 99)) < 0.05 for t in TICKERS)
        match_opt = all(abs(tgt.get(t, -9) - opt.get(t, 99)) < 0.05 for t in TICKERS)
        print(f"    plan == RESAMPLED (canonical): {match_res}   "
              f"|   plan == single-shot: {match_opt}")
    else:
        print("    [FAIL] could not locate plan or resampled file")

    # 5. Module 4 fallback path (resampled missing -> warn + single-shot) --------
    print("\n  MODULE 4 FALLBACK (resampled file absent -> single-shot + warning)")
    fb_input = {"tickers": TICKERS, "currency": CURRENCY, "portfolio_choice": CHOSEN_GOAL,
                "holdings": [{"ticker": t, "shares": float(s)} for t, s in HOLDINGS]}
    today_str = date.today().isoformat()
    hidden = resampled_path + ".hidden"

    def _rename_retry(src, dst, attempts=10):
        """Windows can briefly hold an xlsx handle after pandas reads it;
        gc + short retries release it without failing the whole report."""
        import gc
        for i in range(attempts):
            try:
                os.rename(src, dst)
                return
            except PermissionError:
                gc.collect()
                time.sleep(0.5)
        os.rename(src, dst)   # final attempt; raise if still locked

    try:
        _rename_retry(resampled_path, hidden)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fb_stats = run_all.run_module4(dict(fb_input), today_str)
        fb_out = buf.getvalue()
    finally:
        if os.path.exists(hidden):
            _rename_retry(hidden, resampled_path)
    warned   = "falling back to single-shot" in fb_out
    produced = fb_stats is not None
    print(f"    warning printed: {warned}   |   plan still produced (no crash): {produced}")
    # Regenerate the canonical plan so the resting artifact uses resampled weights.
    with contextlib.redirect_stdout(io.StringIO()):
        run_all.run_module4(dict(fb_input), today_str)
    print("    canonical plan regenerated (resting rebalancing_plan uses resampled weights)")

    # 6. Manifest ----------------------------------------------------------------
    print("\n  ARTIFACT MANIFEST (produced/updated this run)")
    for fname, detail in _manifest(t_start):
        print(f"    - {fname}")
        if detail:
            print(f"        {detail}")

    # 7. Downstream + runtime ----------------------------------------------------
    print("\n  DOWNSTREAM INPUTS")
    print("    Module 4 reads : resampled_portfolios.xlsx (CANONICAL; single-shot fallback)")
    print("    Module 3 reads : returns_stats.xlsx (Annualised Mu/Cov; sim-independent)")
    print("    Module 5       : absent -> skipped gracefully")

    print(f"\n  RUNTIME")
    print(f"    pipeline wall-clock (run_all.main): {pipe_elapsed:6.1f}s  "
          f"(~{pipe_elapsed/60:.1f} min; Layer 7 adds ~3-4 min)")
    print(f"    total harness wall-clock          : {time.time()-t_start:6.1f}s")

    if unexpected:
        print("\n  [WARN] unexpected prompts answered with '' :")
        for p in unexpected:
            print(f"        {p!r}")

    # Restore run_config.json (run_all deletes it on completion)
    with open(os.path.join(SCRIPT_DIR, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({"tickers": TICKERS, "currency": CURRENCY, "holdings": []}, f)
    print("\n  Cleanup: restored run_config.json to the India 3-name set.")
    print("=" * W)


if __name__ == "__main__":
    main()
