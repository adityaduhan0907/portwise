import glob
import json
import math
import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

import fetch_util

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_STOCKS = 15
MAX_NEW_POSITIONS = 5
SKIP_LABELS = {"Portfolio Return (%)", "Portfolio Volatility (%)", "Portfolio Sharpe Ratio", "", "nan"}
INDIA_SFX = (".NS", ".BO")

# ── Market constants ────────────────────────────────────────────────────────────
ERP_USA          = 0.058093
ERP_INDIA        = 0.0750
RF_INDIA         = 0.0675
IMPLIED_PE_USA   = 23.8425
IMPLIED_PE_INDIA = 20.0


@st.cache_data(show_spinner=False)
def load_company_list():
    """Load US and India company lists from company_list.xlsx. Cached for session."""
    path = os.path.join(SCRIPT_DIR, "company_list.xlsx")
    if not os.path.exists(path):
        return None
    try:
        frames = []
        xf = pd.ExcelFile(path)
        if "US Companies" in xf.sheet_names:
            us = pd.read_excel(path, sheet_name="US Companies")
            us["Market"] = "USA"
            frames.append(us)
        if "India Companies" in xf.sheet_names:
            ind = pd.read_excel(path, sheet_name="India Companies")
            ind["Market"] = "India"
            frames.append(ind)
        return pd.concat(frames, ignore_index=True) if frames else None
    except Exception:
        return None


def _filter_options(query, company_df, market):
    """Return up to 10 Search strings matching query, filtered by market (USA or India)."""
    if company_df is None or not query or not query.strip():
        return []
    df = company_df[company_df["Market"] == market]
    q = query.strip().lower()
    mask = df["Search"].str.lower().str.contains(q, na=False)
    return df[mask]["Search"].dropna().tolist()[:10]


def _extract_ticker(sel):
    """Pull ticker out of 'Company Name (TICKER)' string."""
    if sel and "(" in sel:
        return sel.rsplit("(", 1)[-1].rstrip(")")
    return ""

st.set_page_config(
    page_title="PORTWISE",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

with open(os.path.join(SCRIPT_DIR, "static", "style.css")) as f:
    css = f.read()
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# --- Session state ---
for _k, _v in [
    ("holdings", [{"ticker": "", "shares": 0.0}]),
    ("new_positions", []),
    ("results", None),
    ("error", None),
    ("mkt_active", "USA"),
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ═══════════════════════════════════════════════════════════════
# Helpers — pipeline
# ═══════════════════════════════════════════════════════════════

def get_usd_inr():
    try:
        h = fetch_util.fetch_history("USDINR=X", period="3d", what="USD/INR spot")
        if h is not None and not h.empty:
            return float(h["Close"].iloc[-1])
    except Exception:
        pass
    return 84.0


def load_risk_free_rate():
    """Read the blended risk-free rate from risk_free_rates.json."""
    try:
        with open(os.path.join(SCRIPT_DIR, "risk_free_rates.json"), encoding="utf-8") as f:
            return float(json.load(f)["blended_rate"])
    except Exception:
        return 0.043


# ═══════════════════════════════════════════════════════════════
# Helpers — new-pipeline canonical output readers
# ═══════════════════════════════════════════════════════════════

def _latest_file(pattern):
    """Newest file in SCRIPT_DIR matching a glob pattern, or None."""
    files = glob.glob(os.path.join(SCRIPT_DIR, pattern))
    return max(files, key=os.path.getmtime) if files else None


def load_resampled_weights(sheet):
    """
    Chosen goal's averaged (resampled) weights -- the CANONICAL recommendation.
    Reads resampled_portfolios.xlsx (one sheet per resampled goal). Returns
    (DataFrame, None) or (None, error_message).
    """
    path = os.path.join(SCRIPT_DIR, "resampled_portfolios.xlsx")
    if not os.path.exists(path):
        return None, "resampled_portfolios.xlsx not found"
    try:
        xf = pd.ExcelFile(path)
        if sheet not in xf.sheet_names:
            return None, f"sheet '{sheet}' not in {xf.sheet_names}"
        return pd.read_excel(path, sheet_name=sheet), None
    except Exception as exc:
        return None, str(exc)


def load_risk_summary():
    """risk_evaluation_summary.json (Layer 5) or None."""
    path = os.path.join(SCRIPT_DIR, "risk_evaluation_summary.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_robustness():
    """robustness_warnings.json (Layer 6) or None."""
    path = os.path.join(SCRIPT_DIR, "robustness_warnings.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_stress_test():
    """stress_test.json (historical crisis replay) or None."""
    path = os.path.join(SCRIPT_DIR, "stress_test.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# Friendly period text per crisis window (matches the Markdown report).
CRISIS_PERIOD = {
    "2008 GFC":      "2008 financial crisis (late 2007–early 2009)",
    "COVID crash":   "COVID-19 crash (Feb–Mar 2020)",
    "2022 drawdown": "2022 drawdown (Jan–Oct 2022)",
    "dot-com":       "dot-com crash (2000–2002)",
}
CRISIS_ORDER = ["2008 GFC", "COVID crash", "2022 drawdown", "dot-com"]


def _stress_modeled_reason(detail):
    """Plain-language reason a holding had to be modeled, from its breakdown detail
    (same wording the report uses)."""
    d = (detail or "").lower()
    if "listed later" in d or "before this window" in d:
        return "listed after this period"
    if "this far back" in d:
        return "price history doesn't reach this far back"
    return "no market history for this period"


def load_blended_rf():
    """Blended risk-free rate from risk_free_rates.json (0.0 on any failure)."""
    try:
        with open(os.path.join(SCRIPT_DIR, "risk_free_rates.json"),
                  encoding="utf-8") as f:
            return float(json.load(f).get("blended_rate", 0.0))
    except Exception:
        return 0.0


def load_factor_inputs():
    """
    (capm_mu_map, cov_df, rf) for the factor-model Sharpe ("Risk Efficiency"):
      capm_mu_map : {ticker: annualised factor-model E[r]}  (returns_stats 'CAPM Returns')
      cov_df      : annualised covariance matrix             (returns_stats 'Annualised Cov')
      rf          : blended risk-free rate                   (risk_free_rates.json)
    Returns (None, None, 0.0) on any failure.
    """
    try:
        rpath = os.path.join(SCRIPT_DIR, "returns_stats.xlsx")
        capm  = pd.read_excel(rpath, sheet_name="CAPM Returns")
        mu_map = dict(zip(capm["Ticker"], capm["CAPM_Expected_Return"].astype(float)))
        cov = pd.read_excel(rpath, sheet_name="Annualised Cov", index_col=0)
        return mu_map, cov, load_blended_rf()
    except Exception:
        return None, None, 0.0


def portfolio_sharpe(weights, mu_map, cov_df, rf):
    """
    Factor-model Sharpe ("Risk Efficiency"), computed the same way module2 does:
        (capm_mu . w  -  rf) / sqrt(wᵀ Σ w)
    using the factor-model expected return, the blended risk-free rate and the
    annualised portfolio volatility. `weights` is {ticker: weight} at any scale
    (renormalised over the tickers present in cov_df). Returns float or nan.
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


def current_weights_map():
    """
    The user's current-portfolio weights (value-weighted, normalised) as computed
    by Layer 5, from risk_evaluation_summary.json. {} if the user entered no
    holdings (then 'Current Portfolio' is absent). Currency-neutral.
    """
    r = load_risk_summary()
    if not r:
        return {}
    cp = r.get("portfolios", {}).get("Current Portfolio")
    if not cp:
        return {}
    return {t: float(w) for t, w in cp.get("weights", {}).items()}


def parse_goal_sheet(wdf):
    """
    Split a resampled goal sheet (Stock / Weight (%)) into:
      (recommended_weights {ticker: weight_pct}, expected_return_pct or None)
    Skips spacer/summary rows; pulls the 'Portfolio Return (%)' summary value.
    """
    rec, exp_ret = {}, None
    for _, row in wdf.iterrows():
        stock = row.get("Stock")
        wt    = row.get("Weight (%)")
        if not isinstance(stock, str) or not stock.strip():
            continue
        if stock.strip() == "Portfolio Return (%)":
            try:
                exp_ret = float(wt)
            except (TypeError, ValueError):
                pass
            continue
        if "(%)" in stock or stock in SKIP_LABELS:
            continue
        try:
            rec[stock] = float(wt)
        except (TypeError, ValueError):
            continue
    return rec, exp_ret


# ═══════════════════════════════════════════════════════════════
# Helpers — display layer only
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_info(ticker):
    """Returns (company_name, country). Cached 1 hour."""
    try:
        info = fetch_util.fetch_info(ticker)
        name = info.get("longName") or info.get("shortName") or ticker
        country = info.get("country") or "N/A"
        return name, country
    except Exception:
        return ticker, "N/A"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_sector(ticker):
    """Returns sector string or None. Cached 1 hour."""
    try:
        info = fetch_util.fetch_info(ticker)
        return info.get("sector") or None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_pe(ticker):
    """Returns forwardPE first, falls back to trailingPE, then None. Cached 1 hour."""
    try:
        info = fetch_util.fetch_info(ticker)
        pe = info.get("forwardPE")
        if pe is None or (isinstance(pe, float) and pe != pe):
            pe = info.get("trailingPE")
        if pe is None or (isinstance(pe, float) and pe != pe):
            return None
        pe = float(pe)
        return pe if pe > 0 else None
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def load_prices_for_display():
    """Returns (latest_prices_dict, full_df). Cached 10 min."""
    path = os.path.join(SCRIPT_DIR, "prices.xlsx")
    if not os.path.exists(path):
        return {}, pd.DataFrame()
    df = pd.read_excel(path, index_col=0, parse_dates=True)
    latest = df.ffill().iloc[-1].dropna().to_dict()
    return latest, df


def calc_1y_return(ticker, prices_df):
    try:
        if prices_df.empty or ticker not in prices_df.columns:
            return None
        series = prices_df[ticker].dropna()
        if len(series) < 2:
            return None
        latest = float(series.iloc[-1])
        year_ago = float(series.iloc[-252]) if len(series) >= 252 else float(series.iloc[0])
        return (latest - year_ago) / year_ago * 100
    except Exception:
        return None


def get_native_price(ticker, positions, prices_latest):
    """Price in native currency + whether it is an Indian stock."""
    if ticker in positions:
        return positions[ticker]["price_native"], positions[ticker]["is_indian"]
    if ticker in prices_latest:
        return prices_latest[ticker], ticker.upper().endswith(INDIA_SFX)
    return None, False


def fmt_val(usd_val, currency, usd_inr):
    if currency == "INR":
        return f"₹{usd_val * usd_inr:,.0f}"
    return f"${usd_val:,.2f}"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_benchmark_returns():
    """Fetch 1Y return for S&P 500 and Nifty 50. Cached 1 hour."""
    out = {}
    for label, sym in [("S&P 500", "^GSPC"), ("Nifty 50", "^NSEI")]:
        try:
            hist = fetch_util.fetch_history(sym, period="2y", what=f"{label} benchmark")
            if hist is None or hist.empty or len(hist) < 2:
                out[label] = None
                continue
            s = hist["Close"].dropna()
            latest   = float(s.iloc[-1])
            year_ago = float(s.iloc[-252]) if len(s) >= 252 else float(s.iloc[0])
            out[label] = (latest - year_ago) / year_ago * 100
        except Exception:
            out[label] = None
    return out




def render_table(caption, headers, rows, row_classes=None, wrap_last=False):
    if row_classes is None:
        row_classes = [""] * len(rows)
    inner_cls = "tbl wrap-last" if wrap_last else "tbl"
    html = f'<div class="tbl-card"><div class="{inner_cls}"><table><caption>{caption}</caption><thead><tr>'
    for h in headers:
        html += f'<th scope="col">{h}</th>'
    html += "</tr></thead><tbody>"
    for row, cls in zip(rows, row_classes):
        html += f'<tr class="{cls}">'
        for ci, cell in enumerate(row):
            td_cls = ' class="act"' if ci == 0 and cls in ("buy-row", "sell-row") else ""
            html += f"<td{td_cls}>{cell}</td>"
        html += "</tr>"
    html += "</tbody></table></div></div>"
    return html


# ═══════════════════════════════════════════════════════════════
# UI — Form
# ═══════════════════════════════════════════════════════════════

st.markdown(
    '<div class="portwise-header">'
    '<div class="portwise-title">PORTWISE</div>'
    '<div class="portwise-subtitle">Smart guide to rebalance your portfolio in seconds</div>'
    '<div class="portwise-accent-line"></div>'
    '</div>',
    unsafe_allow_html=True,
)
st.divider()

_ccy = st.selectbox("**Currency**", ["USD — US Dollar", "INR — Indian Rupee"])
selected_currency = "INR" if "INR" in _ccy else "USD"

# ── Market filter ─────────────────────────────────────────────────────────────
_company_df = load_company_list()

st.markdown("**Market**")
_mc1, _mc2 = st.columns(2)
with _mc1:
    if st.button(
        "USA",
        key="btn_mkt_usa",
        use_container_width=True,
        type="primary" if st.session_state.mkt_active == "USA" else "secondary",
    ):
        st.session_state.mkt_active = "USA"
        st.rerun()
with _mc2:
    if st.button(
        "India",
        key="btn_mkt_india",
        use_container_width=True,
        type="primary" if st.session_state.mkt_active == "India" else "secondary",
    ):
        st.session_state.mkt_active = "India"
        st.rerun()
_mkt_filter = st.session_state.mkt_active

st.markdown("**Your holdings**")

for i, h in enumerate(list(st.session_state.holdings)):
    c1, c2, c3 = st.columns([3, 2, 1])
    with c1:
        if i == 0:
            st.markdown("**Ticker**")
        # Search text input (doubles as direct ticker entry when no match)
        query = st.text_input(
            f"_hsearch_{i}",
            key=f"hsearch_{i}",
            placeholder="Type company name or ticker e.g. Apple or AAPL",
            label_visibility="collapsed",
        )
        opts = _filter_options(query, _company_df, _mkt_filter) if _company_df is not None else []
        if opts:
            _no_sel = "— select company —"
            _qkey   = (query or "").strip().lower()[:30]
            _sel = st.selectbox(
                f"_hsel_{i}",
                options=[_no_sel] + opts,
                index=0,
                key=f"hsel_{i}_{_qkey}",
                label_visibility="collapsed",
            )
            _extracted = _extract_ticker(_sel)
            if _extracted:
                st.session_state.holdings[i]["ticker"] = _extracted
            elif not st.session_state.holdings[i]["ticker"] and query:
                st.session_state.holdings[i]["ticker"] = query.strip().upper()
        elif query and query.strip():
            # No dropdown matches — accept typed text as direct ticker
            st.session_state.holdings[i]["ticker"] = query.strip().upper()
        # Confirmation caption
        if st.session_state.holdings[i]["ticker"]:
            st.caption(f"Selected: {st.session_state.holdings[i]['ticker']}")
    with c2:
        if i == 0:
            st.markdown("**Shares**")
        s_val = st.number_input(
            f"Shares {i+1}", value=int(h["shares"]),
            min_value=0, step=1, key=f"s_{i}",
            label_visibility="collapsed",
        )
        st.session_state.holdings[i]["shares"] = s_val
    with c3:
        if i == 0:
            st.markdown("<div style='height:56px'></div>", unsafe_allow_html=True)
        if i > 0 and st.button("Remove", key=f"rm_{i}"):
            st.session_state.holdings.pop(i)
            st.rerun()

if len(st.session_state.holdings) < MAX_STOCKS:
    if st.button("+ Add Stock", use_container_width=True):
        st.session_state.holdings.append({"ticker": "", "shares": 0.0})
        st.rerun()
else:
    st.caption("Maximum of 15 stocks reached.")

# ── Feature 1A — Stocks I want to add ────────────────────────────────────────
st.divider()
st.markdown("#### Stocks I want to add to my portfolio")
st.markdown(
    "<p style='color:#4A5568;font-size:13px;font-family:Inter,sans-serif;margin-top:-8px'>"
    "Don't own these yet — the optimizer will recommend how many units to buy "
    "based on your portfolio size</p>",
    unsafe_allow_html=True,
)

if st.session_state.new_positions:
    _nh1, _nh2, _nh3 = st.columns([3, 2, 1])
    with _nh1:
        st.markdown("**Ticker**")
    with _nh2:
        st.markdown(
            "<span style='color:#4A5568;font-size:14px;font-family:Inter,sans-serif'>New position</span>",
            unsafe_allow_html=True,
        )

for i, np_h in enumerate(list(st.session_state.new_positions)):
    nc1, nc2, nc3 = st.columns([3, 2, 1])
    with nc1:
        np_query = st.text_input(
            f"_npsearch_{i}",
            key=f"npsearch_{i}",
            placeholder="Type company name or ticker e.g. NVDA, INFY.NS",
            label_visibility="collapsed",
        )
        np_opts = _filter_options(np_query, _company_df, _mkt_filter) if _company_df is not None else []
        if np_opts:
            _np_no_sel = "— select company —"
            _np_qkey   = (np_query or "").strip().lower()[:30]
            _np_sel = st.selectbox(
                f"_npsel_{i}",
                options=[_np_no_sel] + np_opts,
                index=0,
                key=f"npsel_{i}_{_np_qkey}",
                label_visibility="collapsed",
            )
            _np_extracted = _extract_ticker(_np_sel)
            if _np_extracted:
                st.session_state.new_positions[i]["ticker"] = _np_extracted
            elif not st.session_state.new_positions[i]["ticker"] and np_query:
                st.session_state.new_positions[i]["ticker"] = np_query.strip().upper()
        elif np_query and np_query.strip():
            st.session_state.new_positions[i]["ticker"] = np_query.strip().upper()
        if st.session_state.new_positions[i]["ticker"]:
            st.caption(f"Selected: {st.session_state.new_positions[i]['ticker']}")
    with nc2:
        st.markdown(
            "<div style='padding-top:8px;color:#4A5568;font-style:italic;"
            "font-size:13px;font-family:Inter,sans-serif'>New position</div>",
            unsafe_allow_html=True,
        )
    with nc3:
        if st.button("Remove", key=f"np_rm_{i}"):
            st.session_state.new_positions.pop(i)
            st.rerun()

if len(st.session_state.new_positions) < MAX_NEW_POSITIONS:
    if st.button("+ Add Stock", key="add_new_pos", use_container_width=True):
        st.session_state.new_positions.append({"ticker": ""})
        st.rerun()
else:
    st.caption("Maximum of 5 new positions reached.")

# ── Optimization target & run ─────────────────────────────────────────────────
# New-pipeline goals -> resampled_portfolios.xlsx sheet names. Plain-language
# labels arrive in the next pass; for now the selector passes the goal through.
GOAL_SHEETS = ["Minimum Variance", "Max Risk-Adjusted", "Tail-Risk CVaR"]
selected_target = st.selectbox("**Optimization target**", GOAL_SHEETS)

st.divider()

_filled      = [h for h in st.session_state.holdings if h["ticker"].strip()]
_new_filled  = [h for h in st.session_state.new_positions if h["ticker"].strip()]
_new_tickers_set = {h["ticker"] for h in _new_filled}
_can_run     = len(_filled) >= 3

if st.button("Run Optimization", use_container_width=True,
             type="primary", disabled=not _can_run):
    st.session_state.error = None
    st.session_state.results = None

    tickers = [h["ticker"] for h in _filled] + [h["ticker"] for h in _new_filled]
    all_holdings = (
        [{"ticker": h["ticker"], "shares": float(h["shares"])} for h in _filled]
        + [{"ticker": h["ticker"], "shares": 0.0} for h in _new_filled]
    )

    # Clean progress UI: a "Running…" header, a bar, and one short stage label.
    _head_box = st.empty()
    _bar_box  = st.empty()
    _cap_box  = st.empty()
    _head_box.markdown("**Running…**")
    _bar = _bar_box.progress(0)

    def _on_progress(done, total, label):
        # Plain "Step N of M" — M is the pipeline's real stage total (the callback's
        # `total` arg), never hardcoded; the jargon stage `label` is intentionally unused.
        pct = int(min(max(done / total, 0.0), 1.0) * 100)
        _bar.progress(pct)
        _cap_box.caption(f"Step {done} of {total}")

    try:
        import run_all
        run_all.run_pipeline(
            tickers=tickers,
            holdings=all_holdings,
            currency=selected_currency,
            portfolio_choice=selected_target,
            interactive=False,
            progress_callback=_on_progress,
        )
        st.session_state.results = dict(
            target=selected_target,
            currency=selected_currency,
            new_tickers=_new_tickers_set,
        )
    except Exception as exc:
        # PipelineError (or anything else) -> friendly message, never a stack trace.
        st.session_state.error = str(exc) or exc.__class__.__name__
    finally:
        # Clear the progress UI either way; results or the error render below.
        _head_box.empty()
        _bar_box.empty()
        _cap_box.empty()

if not _can_run:
    st.caption("Enter at least 3 tickers to run.")

if st.session_state.error:
    st.error(
        f"The analysis could not complete: {st.session_state.error}\n\n"
        "Please check your tickers are valid and your internet connection is "
        "stable, then tap **Run Optimization** again."
    )


# ═══════════════════════════════════════════════════════════════
# Results — new-pipeline canonical outputs (minimal pass)
# ═══════════════════════════════════════════════════════════════

if st.session_state.results:
    R        = st.session_state.results
    target   = R["target"]
    currency = R["currency"]

    st.divider()

    # 1) Recommended portfolio — Company / Current vs Recommended position ──────
    #    Each holding's current weight (the user's entered holdings, normalised by
    #    Layer 5) beside the recommended (resampled) weight. For the Max Risk-
    #    Adjusted goal a 'Risk Efficiency' (Sharpe) row is appended for current vs
    #    recommended. The two returns (past-actual vs forward-projection) live in
    #    separate labelled bullets BELOW the table, not in a table row, so they are
    #    not read as a single before/after comparison.
    st.subheader(f"Recommended portfolio — {target}")
    _wdf, _werr = load_resampled_weights(target)
    if _wdf is not None:
        _rec_w, _rec_exp = parse_goal_sheet(_wdf)     # {ticker: weight%}, expected %
        _cur_w = current_weights_map()                # {ticker: weight frac} or {}
        _has_current = bool(_cur_w)

        _tbl_rows = []
        for _t in _rec_w:                             # sheet order
            _name = fetch_ticker_info(_t)[0]
            _cur_cell = (f"{_cur_w[_t] * 100:.2f}%"
                         if _t in _cur_w and _cur_w[_t] > 0 else "—")
            _tbl_rows.append([_name, _cur_cell, f"{_rec_w[_t]:.2f}%"])

        # Max Risk-Adjusted only: Risk Efficiency (Sharpe) for current vs recommended.
        if target == "Max Risk-Adjusted":
            _mu_map, _cov, _rf = load_factor_inputs()
            _se_rec = portfolio_sharpe(_rec_w, _mu_map, _cov, _rf)
            _se_cur = (portfolio_sharpe(_cur_w, _mu_map, _cov, _rf)
                       if _has_current else float("nan"))
            _cur_se = f"{_se_cur:.2f} (current)" if _se_cur == _se_cur else "—"
            _rec_se = f"{_se_rec:.2f} (recommended)" if _se_rec == _se_rec else "—"
            _tbl_rows.append(["Risk Efficiency", _cur_se, _rec_se])

        # Clean static table — no dataframe toolbar (search / show-hide / CSV icon).
        st.markdown(
            render_table(
                "",
                ["Company Name", "Current Position", "Recommended Position"],
                _tbl_rows,
            ),
            unsafe_allow_html=True,
        )
        if not _has_current:
            st.caption("Current Position shows '—' because no current holdings were "
                       "entered. Recommended = averaged (resampled) model weights.")

        # Returns — two clearly-separated bullets BELOW the table. One is a backward
        # actual (past year), the other a forward projection; never the same row.
        _realized = None
        if _has_current:
            _, _prices_df = load_prices_for_display()
            _num = _den = 0.0
            for _t, _w in _cur_w.items():
                _r1 = calc_1y_return(_t, _prices_df)
                if _r1 is not None:
                    _num += _w * _r1
                    _den += _w
            _realized = (_num / _den) if _den > 0 else None

        if _realized is not None:
            st.markdown(
                f"- **Your current portfolio's actual return over the past year:** "
                f"{_realized:.2f}%"
            )
        elif _has_current:
            st.markdown("- **Your current portfolio's actual return over the past "
                        "year:** not available (insufficient price history)")
        if _rec_exp is not None:
            st.markdown(
                f"- **This recommended portfolio's projected return "
                f"(forward estimate):** {_rec_exp:.2f}%"
            )
    else:
        st.warning(f"Could not load recommended weights: {_werr}")

    # 2) Risk — current vs recommended ────────────────────────────────────────
    #    Plain-language, ANNUAL figures only. VaR/CVaR are read from the annual
    #    fields (the monthly_* fields stay in the json but are never shown here).
    #    "Risk Efficiency" is the factor-model Sharpe (same definition used in the
    #    recommended table and module2): (capm_mu·w - rf)/sqrt(wᵀΣw).
    st.subheader("Risk — current vs recommended")
    _risk = load_risk_summary()
    if _risk and _risk.get("portfolios"):
        _ports = _risk["portfolios"]

        # Factor inputs for the Risk Efficiency (Sharpe) calculation.
        _mu_map, _cov, _rf = load_factor_inputs()

        def _risk_row(_name):
            _m = _ports.get(_name)
            if not _m:
                return [_name, "—", "—", "—", "—", "—", "—"]
            _sharpe = portfolio_sharpe(_m.get("weights", {}), _mu_map, _cov, _rf)
            _vol = _m.get("volatility", {}).get("annualized")
            return [
                _name,
                f"{_vol * 100:.2f}%" if _vol is not None else "—",  # Volatility (annualised)
                f"{_m['var_95']['annual_loss'] * 100:.2f}%",        # Typical Worst-Year Loss
                f"{_m['cvar_95']['annual_loss'] * 100:.2f}%",       # Average Loss During Crashes
                f"{_m['chance_large_loss']['probability'] * 100:.1f}%",  # Chance of Large Loss
                f"{_m['max_drawdown']['p95_worst'] * 100:.2f}%",    # Worst Expected Decline
                f"{_sharpe:.2f}" if _sharpe == _sharpe else "—",    # Risk Efficiency (Sharpe)
            ]

        _rows = []
        if "Current Portfolio" in _ports:
            _rows.append(_risk_row("Current Portfolio"))
        _rows.append(_risk_row(target))
        st.markdown(
            render_table(
                "",
                ["Portfolio", "Volatility (yearly swing)",
                 "Typical Worst-Year Loss", "Average Loss During Crashes",
                 "Chance of Large Loss", "Worst Expected Decline",
                 "Risk Efficiency"],
                _rows,
            ),
            unsafe_allow_html=True,
        )
        st.caption(
            "Risk is estimated by simulating thousands of possible future years "
            "based on how your holdings have behaved historically. "
            "'Average Loss During Crashes' is the average outcome in the worst 5% "
            "of those simulated years. These estimates assume the future resembles "
            "the past and cannot predict unprecedented events."
        )
    else:
        st.warning("Risk summary unavailable.")

    # 3) Robustness checks ────────────────────────────────────────────────────
    #    Momentum is intentionally NOT shown here (it still runs in
    #    robustness_checks.py and is recorded in robustness_warnings.json). We
    #    surface only the diversification checks — correlation and sector
    #    concentration — as plain-language flagged-or-fine lines with the numbers.
    st.subheader("Diversification checks")
    _rob = load_robustness()
    if _rob:
        # — Correlation —
        _corr = _rob.get("high_correlation", {})
        _pairs = _corr.get("flagged_pairs", [])
        if _corr.get("triggered") and _pairs:
            for _p in _pairs:
                st.markdown(
                    f"**Correlation:** ⚠️ {_p['stock_a']} and {_p['stock_b']} move "
                    f"almost identically (correlation {_p['correlation']:.2f}) — "
                    f"holding both adds little diversification."
                )
        else:
            _thr = _corr.get("threshold", 0.85)
            st.markdown(
                f"**Correlation:** ✅ Your holdings move independently enough to "
                f"diversify well (no pair above {_thr:.2f})."
            )

        # — Sector concentration —
        _sec = _rob.get("sector_concentration", {})
        _largest = _sec.get("largest_sector")
        _share   = _sec.get("largest_share")
        _eff     = _sec.get("effective_sectors")
        if _largest is not None and _share is not None and _eff is not None:
            if _sec.get("triggered"):
                st.markdown(
                    f"**Concentration:** ⚠️ {_share * 100:.0f}% of this portfolio is "
                    f"in one sector ({_largest}) — effective spread about "
                    f"{_eff:.1f} sectors."
                )
            else:
                st.markdown(
                    f"**Concentration:** ✅ Well spread across sectors (largest is "
                    f"{_largest} at {_share * 100:.0f}%; effective spread about "
                    f"{_eff:.1f} sectors)."
                )
        else:
            st.caption("Sector concentration data unavailable.")
    else:
        st.caption("Diversification checks unavailable.")

    # 3b) Historical stress test ──────────────────────────────────────────────
    #    Compact crisis replay from stress_test.json, using the NEW per-window basis
    #    tags (basis / actual_holdings / modeled_holdings / insufficient_holdings) —
    #    NOT the old missing[] list. A blended window VISIBLY discloses that some
    #    holdings are factor-modeled (an estimate), so an estimate is never shown as
    #    fully observed. Same honesty wiring the Markdown report follows.
    st.subheader("How this would have handled past crises")
    _stress = load_stress_test()
    _windows = (_stress or {}).get("windows") if isinstance(_stress, dict) else None
    if _windows:
        st.caption(
            "Two facts about the same **recommended** portfolio in each crisis: where it "
            "would have ended over the full period, and the deepest drop it would have "
            "taken along the way (it may have partly recovered by the end)."
        )
        _ordered = [w for w in CRISIS_ORDER if w in _windows] + \
                   [w for w in _windows if w not in CRISIS_ORDER]
        for _name in _ordered:
            _w = _windows.get(_name, {})
            _period = CRISIS_PERIOD.get(_name, _name)
            _cum = _w.get("cumulative_return")
            _dd  = _w.get("max_drawdown")
            _modeled = list(_w.get("modeled_holdings", []) or [])
            _insuff  = list(_w.get("insufficient_holdings", []) or [])
            _detail_of = {h.get("ticker"): h.get("detail")
                          for h in (_w.get("holdings", []) or [])}

            # No figure -> honest "insufficient data".
            if _cum is None:
                _excl = ", ".join(_insuff) or "the holdings"
                st.markdown(
                    f"**{_name}** — {_period}: ⚠️ insufficient data — neither real "
                    f"history nor factor data reaches this far back for {_excl}; "
                    "no figure shown."
                )
                continue

            if _modeled:
                # Blended: estimate, must name the modeled holdings.
                _reasons = {}
                for _t in _modeled:
                    _reasons.setdefault(_stress_modeled_reason(_detail_of.get(_t)),
                                        []).append(_t)
                _bits = "; ".join(f"{', '.join(_ts)} ({_r})"
                                  for _r, _ts in _reasons.items())
                _dir = "down" if _cum < 0 else "up"
                _line = (
                    f"**{_name}** — {_period}: your recommended portfolio would have "
                    f"**ended this period {_dir} ~{abs(_cum) * 100:.0f}%** (estimated), "
                    f"having fallen as much as ~{abs(_dd) * 100:.0f}% at its worst point "
                    f"along the way. · 🟡 *estimated* — {_bits} factor-modeled, so this "
                    "figure is partly an estimate, not fully observed."
                )
                if _insuff:
                    _line += f" {', '.join(_insuff)} excluded (no data this far back)."
                st.markdown(_line)
            else:
                # Fully measured.
                _dir = "down" if _cum < 0 else "up"
                st.markdown(
                    f"**{_name}** — {_period}: your recommended portfolio would have "
                    f"**ended this period {_dir} ~{abs(_cum) * 100:.0f}%**, having fallen "
                    f"as much as ~{abs(_dd) * 100:.0f}% at its worst point along the way. "
                    "· ✅ measured (all holdings have real history)."
                )
        st.caption(
            "Real historical replays. Where a holding predates a crisis its return is "
            "reconstructed from its factor exposures and shown as *estimated* — a model "
            "figure, not a measured outcome."
        )
    else:
        st.caption("Historical stress test unavailable.")

    # 4) Rebalancing plan (latest) + download ─────────────────────────────────
    #    The plan sheet carries Amount (USD) always and Amount (INR) when the book
    #    is INR. Both amounts are correctly computed by module4; here we display
    #    only the entered-currency amount, formatted with the right symbol
    #    (₹ for INR, $ for USD) — never both, never the wrong one for the book.
    st.subheader("Rebalancing plan")
    _plan = _latest_file("rebalancing_plan_*.xlsx")
    if _plan:
        try:
            _instr = pd.read_excel(_plan, sheet_name="Rebalancing Instructions")

            _use_inr = (currency == "INR") and ("Amount (INR)" in _instr.columns)
            if _use_inr:
                _amt_src, _amt_col = "Amount (INR)", "Amount (₹)"
                _fmt = lambda v: (f"₹{v:,.0f}" if isinstance(v, (int, float))
                                  and pd.notna(v) else v)
            else:
                _amt_src, _amt_col = "Amount (USD)", "Amount ($)"
                _fmt = lambda v: (f"${v:,.2f}" if isinstance(v, (int, float))
                                  and pd.notna(v) else v)

            if _amt_src in _instr.columns:
                _instr[_amt_col] = _instr[_amt_src].map(_fmt)
            # Drop the raw amount columns (keep only the formatted display one).
            _instr = _instr.drop(
                columns=[c for c in ("Amount (USD)", "Amount (INR)")
                         if c in _instr.columns and c != _amt_col],
                errors="ignore",
            )
            # Place the amount column where it reads naturally (after the diff).
            _order = ["Action", "Ticker", "Current Weight %", "Target Weight %",
                      "Weight Difference %", _amt_col, "Status", "Note"]
            _instr = _instr[[c for c in _order if c in _instr.columns]]

            # Clean static table — colour BUY/SELL rows; no dataframe toolbar.
            _headers = list(_instr.columns)
            _rows = _instr.astype(object).where(pd.notna(_instr), "").values.tolist()
            _row_cls = []
            for _r in _instr.get("Action", pd.Series([""] * len(_instr))):
                _a = str(_r).strip().upper()
                _row_cls.append("buy-row" if _a == "BUY"
                                else "sell-row" if _a == "SELL" else "")
            st.markdown(
                render_table("", _headers, _rows, row_classes=_row_cls, wrap_last=True),
                unsafe_allow_html=True,
            )
        except Exception as _exc:
            st.caption(f"(Could not preview instructions: {_exc})")
    else:
        st.caption("No rebalancing plan available yet.")

    # 5) The full plain-language report is intentionally NOT rendered inline — its
    #    key results already appear in the tables/sections above. It remains
    #    available in full via the Download control below (Markdown).

    # 6) Single Download control ───────────────────────────────────────────────
    #    One control for ALL user-facing deliverables — no per-table download
    #    buttons. Pick a file by friendly name, then download it. Only genuine
    #    deliverables that exist on disk are offered.
    st.divider()
    st.subheader("Download")
    _XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    _candidates = [
        ("Full report (Markdown)", _latest_file("portfolio_report_*.md"), "text/markdown"),
        ("Rebalancing plan (Excel)", _latest_file("rebalancing_plan_*.xlsx"), _XLSX_MIME),
        ("Portfolio weights (Excel)",
         os.path.join(SCRIPT_DIR, "resampled_portfolios.xlsx"), _XLSX_MIME),
    ]
    _deliverables = [(lbl, p, mime) for lbl, p, mime in _candidates
                     if p and os.path.exists(p)]
    if _deliverables:
        _labels = [lbl for lbl, _, _ in _deliverables]
        _choice = st.selectbox("Choose a file to download", _labels,
                               label_visibility="collapsed")
        _sel = next(d for d in _deliverables if d[0] == _choice)
        with open(_sel[1], "rb") as _f:
            st.download_button(
                f"Download — {_sel[0]}",
                data=_f.read(),
                file_name=os.path.basename(_sel[1]),
                mime=_sel[2],
                use_container_width=True,
            )
    else:
        st.caption("No downloadable files available yet.")

    st.divider()
    if st.button("Start over", key="reset"):
        st.session_state.results = None
        st.session_state.error = None
        st.rerun()
