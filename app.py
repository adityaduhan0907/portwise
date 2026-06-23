import glob
import json
import math
import os
import sys
import warnings
from datetime import datetime

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
        pct = int(min(max(done / total, 0.0), 1.0) * 100)
        _bar.progress(pct)
        _cap_box.caption(label)

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

    # 1) Recommended weights — chosen goal, resampled = CANONICAL ──────────────
    st.subheader(f"Recommended portfolio — {target}")
    _wdf, _werr = load_resampled_weights(target)
    if _wdf is not None:
        st.caption("Averaged (resampled) weights for your chosen goal "
                   "— source: resampled_portfolios.xlsx")
        st.dataframe(_wdf, use_container_width=True, hide_index=True)
    else:
        st.warning(f"Could not load recommended weights: {_werr}")

    # 2) Risk — current vs recommended ────────────────────────────────────────
    st.subheader("Risk — current vs recommended")
    _risk = load_risk_summary()
    if _risk and _risk.get("portfolios"):
        _ports = _risk["portfolios"]

        def _risk_row(_name):
            _m = _ports.get(_name)
            if not _m:
                return [_name, "—", "—", "—", "—"]
            return [
                _name,
                f"{_m['volatility']['annualized'] * 100:.2f}%",
                f"{_m['var_95']['monthly_loss'] * 100:.2f}%",
                f"{_m['cvar_95']['monthly_loss'] * 100:.2f}%",
                f"{_m['max_drawdown']['p95_worst'] * 100:.2f}%",
            ]

        _rows = []
        if "Current Portfolio" in _ports:
            _rows.append(_risk_row("Current Portfolio"))
        _rows.append(_risk_row(target))
        _risk_df = pd.DataFrame(
            _rows,
            columns=["Portfolio", "Ann. Volatility", "VaR 95% (monthly)",
                     "CVaR 95% (monthly)", "Max Drawdown (p95)"],
        )
        st.caption("source: risk_evaluation_summary.json")
        st.dataframe(_risk_df, use_container_width=True, hide_index=True)
    else:
        st.warning("Risk summary unavailable (risk_evaluation_summary.json).")

    # 3) Robustness warnings ──────────────────────────────────────────────────
    st.subheader("Robustness checks")
    _rob = load_robustness()
    if _rob:
        _any = False
        for _key in ("negative_momentum", "high_correlation", "sector_concentration"):
            _c = _rob.get(_key, {})
            _msg = _c.get("message", "")
            if not _msg:
                continue
            _any = True
            if _c.get("triggered"):
                st.warning(_msg)
            else:
                st.info(_msg)
        if not _any:
            st.caption("No robustness messages reported.")
    else:
        st.caption("No robustness_warnings.json found.")

    # 4) Rebalancing plan (latest) + download ─────────────────────────────────
    st.subheader("Rebalancing plan")
    _plan = _latest_file("rebalancing_plan_*.xlsx")
    if _plan:
        try:
            _instr = pd.read_excel(_plan, sheet_name="Rebalancing Instructions")
            st.dataframe(_instr, use_container_width=True, hide_index=True)
        except Exception as _exc:
            st.caption(f"(Could not preview instructions: {_exc})")
        with open(_plan, "rb") as _f:
            st.download_button(
                "Download Rebalancing Plan (Excel)",
                data=_f.read(),
                file_name=os.path.basename(_plan),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
    else:
        st.caption("No rebalancing_plan_*.xlsx found.")

    # 5) Plain-language report (latest markdown) ──────────────────────────────
    st.subheader("Your report")
    _report = _latest_file("portfolio_report_*.md")
    if _report:
        with open(_report, encoding="utf-8") as _f:
            st.markdown(_f.read())
    else:
        st.caption("No portfolio_report_*.md found.")

    st.divider()
    if st.button("Start over", key="reset"):
        st.session_state.results = None
        st.session_state.error = None
        st.rerun()
