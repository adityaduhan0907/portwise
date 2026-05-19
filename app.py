import importlib
import json
import math
import os
import subprocess
import sys
import warnings
from datetime import datetime

import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAX_STOCKS = 15
MAX_NEW_POSITIONS = 5
SKIP_LABELS = {"Portfolio Return (%)", "Portfolio Volatility (%)", "Portfolio Sharpe Ratio", "", "nan"}
INDIA_SFX = (".NS", ".BO")

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
]:
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ═══════════════════════════════════════════════════════════════
# Helpers — pipeline
# ═══════════════════════════════════════════════════════════════

def write_config(tickers, currency):
    with open(os.path.join(SCRIPT_DIR, "run_config.json"), "w") as f:
        json.dump({"tickers": tickers, "currency": currency, "holdings": [],
                   "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)


def run_module(name, timeout=300):
    path = os.path.join(SCRIPT_DIR, name)
    r = subprocess.run([sys.executable, path], cwd=SCRIPT_DIR,
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        err = r.stderr.strip()[-600:] if r.stderr.strip() else "No error detail available"
        raise RuntimeError(err)


def get_usd_inr():
    try:
        import yfinance as yf
        h = yf.Ticker("USDINR=X").history(period="3d")
        if not h.empty:
            return float(h["Close"].iloc[-1])
    except Exception:
        pass
    return 84.0


def load_portfolio_options():
    path = os.path.join(SCRIPT_DIR, "optimised_portfolios.xlsx")
    opts = {}
    xf = pd.ExcelFile(path)
    for sheet in xf.sheet_names:
        df = pd.read_excel(path, sheet_name=sheet)
        w = {}
        for _, row in df.iterrows():
            s = str(row.get("Stock", "")).strip()
            wt = row.get("Weight (%)", None)
            if s in SKIP_LABELS:
                continue
            try:
                val = float(wt)
                if val > 0:
                    w[s] = val
            except (ValueError, TypeError):
                pass
        if w:
            opts[sheet] = w
    return opts


def run_rebalancing(holdings, currency, target_name):
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)
    import module4_rebalance as m4
    importlib.reload(m4)

    prices = m4.load_latest_prices(os.path.join(SCRIPT_DIR, "prices.xlsx"))
    usd_inr = get_usd_inr()
    target_weights = load_portfolio_options().get(target_name, {})

    raw, bad = [], []
    for h in holdings:
        t = h["ticker"].strip().upper()
        if not t:
            continue
        resolved, price = m4.lookup_price(t, prices)
        if resolved is None:
            bad.append(t)
        else:
            raw.append((resolved, float(h["shares"]), price, m4.is_indian(resolved)))

    if not raw:
        raise ValueError("None of your tickers matched price data. Check symbols.")

    positions, total_usd = m4.build_current_portfolio(raw, usd_inr)
    instructions = m4.generate_instructions(positions, target_weights, total_usd, usd_inr)

    today = datetime.now().strftime("%Y-%m-%d")
    out_path = os.path.join(SCRIPT_DIR, f"rebalancing_plan_{datetime.now().strftime('%Y%m%d')}.xlsx")
    m4.export_to_excel(positions, total_usd, target_name, target_weights,
                       instructions, currency, usd_inr, today, out_path)

    return instructions, positions, total_usd, usd_inr, bad, out_path


# ═══════════════════════════════════════════════════════════════
# Helpers — display layer only
# ═══════════════════════════════════════════════════════════════

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_info(ticker):
    """Returns (company_name, country). Cached 1 hour."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        name = info.get("longName") or info.get("shortName") or ticker
        country = info.get("country") or "N/A"
        return name, country
    except Exception:
        return ticker, "N/A"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_sector(ticker):
    """Returns sector string or None. Cached 1 hour."""
    try:
        import yfinance as yf
        return yf.Ticker(ticker).info.get("sector") or None
    except Exception:
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_pe(ticker):
    """Returns trailingPE, then forwardPE, then None. Cached 1 hour."""
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        pe = info.get("trailingPE")
        if pe is None or (isinstance(pe, float) and (pe != pe)):  # None or NaN
            pe = info.get("forwardPE")
        if pe is None or (isinstance(pe, float) and (pe != pe)):
            return None
        pe = float(pe)
        return pe if pe > 0 else None  # negative P/E is not meaningful
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
            import yfinance as yf
            hist = yf.Ticker(sym).history(period="2y")
            if hist.empty or len(hist) < 2:
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

st.markdown("**Your holdings**")

for i, h in enumerate(list(st.session_state.holdings)):
    c1, c2, c3 = st.columns([3, 2, 1])
    with c1:
        t_val = st.text_input(
            f"Ticker {i+1}", value=h["ticker"],
            key=f"t_{i}", placeholder="e.g. AAPL, RELIANCE.NS",
            label_visibility="collapsed" if i > 0 else "visible",
        )
        st.session_state.holdings[i]["ticker"] = t_val.strip().upper()
    with c2:
        s_val = st.number_input(
            f"Shares {i+1}", value=int(h["shares"]),
            min_value=0, step=1, key=f"s_{i}",
            label_visibility="collapsed" if i > 0 else "visible",
        )
        st.session_state.holdings[i]["shares"] = s_val
    with c3:
        if i == 0:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
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
        np_val = st.text_input(
            f"New Ticker {i+1}", value=np_h["ticker"],
            key=f"np_t_{i}", placeholder="e.g. NVDA, INFY.NS",
            label_visibility="collapsed",
        )
        st.session_state.new_positions[i]["ticker"] = np_val.strip().upper()
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
_targets = {
    "Max Sharpe Ratio — Best risk-adjusted return (recommended)": "Max Sharpe Ratio",
    "Min Volatility — Lowest risk portfolio": "Min Volatility",
    "Max Return — Highest return (higher risk)": "Max Return",
}
_t_label = st.selectbox("**Optimization target**", list(_targets.keys()))
selected_target = _targets[_t_label]

st.divider()

_filled      = [h for h in st.session_state.holdings if h["ticker"].strip()]
_new_filled  = [h for h in st.session_state.new_positions if h["ticker"].strip()]
_new_tickers_set = {h["ticker"] for h in _new_filled}
_can_run     = len(_filled) >= 3

if st.button("Run Optimization", use_container_width=True,
             type="primary", disabled=not _can_run):
    st.session_state.error = None
    st.session_state.results = None
    tickers      = [h["ticker"] for h in _filled] + [h["ticker"] for h in _new_filled]
    all_holdings = _filled + [{"ticker": h["ticker"], "shares": 0.0} for h in _new_filled]
    try:
        with st.spinner("Fetching prices and optimizing — this takes about 30 seconds..."):
            write_config(tickers, selected_currency)
            run_module("module0_riskfree.py", timeout=60)
            run_module("module1_data.py", timeout=300)
            run_module("module2_optimiser.py", timeout=180)
            run_module("module3_frontier.py", timeout=180)
            instrs, positions, total_usd, usd_inr, bad, xl_path = run_rebalancing(
                all_holdings, selected_currency, selected_target
            )
        st.session_state.results = dict(
            instrs=instrs, positions=positions, total_usd=total_usd,
            usd_inr=usd_inr, bad=bad, xl_path=xl_path,
            currency=selected_currency, target=selected_target,
            new_tickers=_new_tickers_set,
        )
    except RuntimeError as e:
        st.session_state.error = str(e)
    except Exception as e:
        st.session_state.error = f"{type(e).__name__}: {str(e)[:400]}"

if not _can_run:
    st.caption("Enter at least 3 tickers to run.")

if st.session_state.error:
    st.error(
        "**Something went wrong.** "
        + st.session_state.error
        + "\n\nCheck your ticker symbols and internet connection, then try again."
    )


# ═══════════════════════════════════════════════════════════════
# Results
# ═══════════════════════════════════════════════════════════════

if st.session_state.results:
    R           = st.session_state.results
    currency    = R["currency"]
    usd_inr     = R["usd_inr"]
    positions   = R["positions"]
    instrs      = R["instrs"]
    target      = R["target"]
    total_usd   = R["total_usd"]
    new_tickers = R.get("new_tickers", set())

    if R["bad"]:
        st.warning(f"Could not find prices for: {', '.join(R['bad'])}. These were excluded.")

    st.divider()

    prices_latest, prices_df = load_prices_for_display()

    # ── TABLE 1 — Current Portfolio Position ──────────────────────────────────
    st.subheader("Current Portfolio Position")

    t1_rows, t1_classes = [], []
    wa_num, wa_den = 0.0, 0.0

    for ticker in sorted(positions.keys()):
        pos = positions[ticker]
        # Exclude new (zero-share) positions from current holdings view
        if ticker in new_tickers and pos.get("value_usd", 0) <= 0:
            continue
        name, country = fetch_ticker_info(ticker)
        weight    = pos["weight_pct"]
        value_usd = pos["value_usd"]
        ret       = calc_1y_return(ticker, prices_df)
        ret_str   = f"{ret:.1f}%" if ret is not None else "N/A"
        if ret is not None:
            wa_num += (weight / 100) * ret
            wa_den += weight / 100

        t1_rows.append([
            ticker, name, country,
            f"{weight:.1f}%",
            fmt_val(value_usd, currency, usd_inr),
            ret_str,
        ])
        t1_classes.append("")

    t1_rows.append(["TOTAL", "", "", "100.0%", fmt_val(total_usd, currency, usd_inr), ""])
    t1_classes.append("total")

    st.markdown(render_table(
        "Your current holdings, weights, values, and one-year price returns.",
        ["Ticker", "Company", "Country", "Weight", "Value", "1Y Return"],
        t1_rows, t1_classes,
    ), unsafe_allow_html=True)

    total_str = fmt_val(total_usd, currency, usd_inr)
    wa_str    = f"{wa_num:.1f}%" if wa_den > 0 else "N/A"
    st.markdown(f"- **Total Portfolio Value:** {total_str}")
    st.markdown(f"- **Weighted Average 1Y Return:** {wa_str}")

    # ── Feature 2 — Benchmark Comparison ─────────────────────────────────────
    st.markdown("---")
    st.subheader("How does your portfolio compare to the market?")

    try:
        # Determine which benchmarks are relevant based on portfolio countries
        _port_countries = set()
        for _tk in positions:
            if _tk in new_tickers and positions[_tk].get("value_usd", 0) <= 0:
                continue
            _, _ctry = fetch_ticker_info(_tk)
            if _ctry and _ctry != "N/A":
                _port_countries.add(_ctry)

        _has_us    = "United States" in _port_countries
        _has_india = "India" in _port_countries
        _has_other = bool(_port_countries - {"United States", "India"})

        if not _port_countries:
            _show_sp, _show_nifty = True, False
        elif _has_us and not _has_india and not _has_other:
            _show_sp, _show_nifty = True, False
        elif _has_india and not _has_us and not _has_other:
            _show_sp, _show_nifty = False, True
        elif _has_us and _has_india:
            _show_sp, _show_nifty = True, True
        else:
            _show_sp, _show_nifty = True, False

        bench    = fetch_benchmark_returns()
        port_ret = wa_num if wa_den > 0 else None

        def _rs(v):
            return f"{v:.1f}%" if v is not None else "N/A"

        _bench_rows = [["Your Portfolio", _rs(port_ret)]]
        if _show_sp:
            _bench_rows.append(["S&P 500", _rs(bench.get("S&P 500"))])
        if _show_nifty:
            _bench_rows.append(["Nifty 50", _rs(bench.get("Nifty 50"))])

        st.markdown(render_table(
            "1 Year Return Comparison",
            ["Name", "1Y Return"],
            _bench_rows,
        ), unsafe_allow_html=True)

        sp    = bench.get("S&P 500") if _show_sp else None
        nifty = bench.get("Nifty 50") if _show_nifty else None
        if port_ret is not None:
            beats_sp    = sp    is not None and port_ret > sp
            beats_nifty = nifty is not None and port_ret > nifty
            if _show_sp and _show_nifty:
                if beats_sp and beats_nifty:
                    _msg = "Your portfolio outperformed both benchmarks over the past year."
                elif beats_sp:
                    _msg = "Your portfolio outperformed S&P 500 but lagged Nifty 50 over the past year."
                elif beats_nifty:
                    _msg = "Your portfolio outperformed Nifty 50 but lagged S&P 500 over the past year."
                else:
                    _msg = ("Both benchmarks outperformed your portfolio over the past year. "
                            "The optimizer aims to improve this going forward.")
            elif _show_sp:
                _msg = ("Your portfolio outperformed S&P 500 over the past year." if beats_sp
                        else "S&P 500 outperformed your portfolio over the past year. "
                             "The optimizer aims to improve this going forward.")
            else:
                _msg = ("Your portfolio outperformed Nifty 50 over the past year." if beats_nifty
                        else "Nifty 50 outperformed your portfolio over the past year. "
                             "The optimizer aims to improve this going forward.")
            if _msg:
                st.markdown(_msg)

        _unavail = [b for b, shown, val in [("S&P 500", _show_sp, bench.get("S&P 500")),
                                             ("Nifty 50", _show_nifty, bench.get("Nifty 50"))]
                    if shown and val is None]
        if _unavail:
            st.caption("Benchmark data temporarily unavailable for one or more indices.")

    except Exception as _e:
        st.warning(f"Benchmark comparison unavailable: {_e}")

    # ── Feature 3 — Portfolio Health Score ───────────────────────────────────
    st.markdown("---")
    st.subheader("Portfolio Health Score")

    try:
        _countries            = set()
        _sectors_set          = set()
        _unknown_sector_count = 0
        _ret_series           = {}
        _pe_weighted_sum      = 0.0
        _pe_weight_sum        = 0.0
        _pe_missing_tks       = []
        _pe_total_count       = 0

        for _tk in positions:
            if _tk in new_tickers and positions[_tk].get("value_usd", 0) <= 0:
                continue
            _pe_total_count += 1
            _, _ctry = fetch_ticker_info(_tk)
            if _ctry and _ctry != "N/A":
                _countries.add(_ctry)
            _sect = fetch_ticker_sector(_tk)
            if _sect:
                _sectors_set.add(_sect)
            else:
                _unknown_sector_count += 1
            _pe_val = fetch_ticker_pe(_tk)
            _wt_frac = positions[_tk]["weight_pct"] / 100.0
            if _pe_val is not None:
                _pe_weighted_sum += _wt_frac * _pe_val
                _pe_weight_sum   += _wt_frac
            else:
                _pe_missing_tks.append(_tk)
            if not prices_df.empty and _tk in prices_df.columns:
                _s = prices_df[_tk].dropna()
                if len(_s) > 30:
                    _ret_series[_tk] = _s.pct_change().dropna()

        # Factor 1 — Diversification (max 2.0)
        _nc = len(_countries)
        if _nc >= 5:
            f1, f1x = 2.0, f"Your portfolio spans {_nc} countries. Excellent geographic diversification."
        elif _nc >= 3:
            f1, f1x = 1.5, (f"Your portfolio spans {_nc} countries. "
                            "Adding stocks from more regions would improve this score.")
        elif _nc == 2:
            f1, f1x = 1.0, "Your portfolio spans 2 countries. Consider adding international exposure."
        else:
            f1, f1x = 0.5, "All your stocks are from a single country. Geographic diversification could reduce risk."

        # Factor 2 — Sharpe Ratio (max 2.0)
        sharpe = None
        try:
            _sp = pd.read_excel(os.path.join(SCRIPT_DIR, "optimised_portfolios.xlsx"),
                                sheet_name=target)
            for _, _row in _sp.iterrows():
                if "Sharpe" in str(_row.get("Stock", "")):
                    try:
                        sharpe = float(_row.get("Weight (%)", 0))
                    except Exception:
                        pass
        except Exception:
            pass

        if sharpe is None and len(_ret_series) >= 2:
            try:
                _rdf = pd.DataFrame(_ret_series).dropna()
                _tks = list(_rdf.columns)
                _tw  = sum(positions[t]["weight_pct"] for t in _tks if t in positions) or 1.0
                _wts = [(positions[t]["weight_pct"] / _tw if t in positions else 0.0) for t in _tks]
                _pr  = _rdf.mul(_wts, axis=1).sum(axis=1)
                _ar  = _pr.mean() * 252
                _av  = _pr.std() * math.sqrt(252)
                sharpe = _ar / _av if _av > 0 else 0.0
            except Exception:
                sharpe = None

        if sharpe is None:
            f2, f2x = 0.8, "Sharpe ratio could not be calculated from available data."
        elif sharpe >= 2.0:
            f2, f2x = 2.0, f"Excellent risk-adjusted return (Sharpe: {sharpe:.2f})."
        elif sharpe >= 1.5:
            f2, f2x = 1.6, f"Strong risk-adjusted return (Sharpe: {sharpe:.2f})."
        elif sharpe >= 1.0:
            f2, f2x = 1.2, f"Good risk-adjusted return (Sharpe: {sharpe:.2f})."
        elif sharpe >= 0.5:
            f2, f2x = 0.8, (f"Moderate risk-adjusted return (Sharpe: {sharpe:.2f}). "
                            "The optimizer targets a higher Sharpe.")
        else:
            f2, f2x = 0.4, f"Low risk-adjusted return (Sharpe: {sharpe:.2f}). Rebalancing should help."

        # Factor 3 — Valuation: Weighted P/E (max 2.0)
        try:
            _pe_india_only = "India" in _countries and "United States" not in _countries
            _pe_benchmark  = 20.0 if _pe_india_only else 18.0
            _pe_bench_name = "Nifty 50" if _pe_india_only else "S&P 500"

            if _pe_total_count == 0 or len(_pe_missing_tks) > _pe_total_count / 2:
                f3, f3x = 0.0, "Insufficient P/E data available for this portfolio."
            elif _pe_weight_sum <= 0:
                f3, f3x = 0.0, "P/E data unavailable — factor skipped."
            else:
                _wtd_pe = _pe_weighted_sum / _pe_weight_sum
                _pe_miss_note = (
                    f" Note: {', '.join(_pe_missing_tks)} had no P/E data available "
                    f"and were excluded from this calculation."
                    if _pe_missing_tks else ""
                )
                if _wtd_pe < _pe_benchmark * 0.80:
                    f3, f3x = 2.0, (
                        f"Your portfolio's weighted P/E of {_wtd_pe:.1f} is below the "
                        f"{_pe_bench_name} long run median of {_pe_benchmark:.0f}. "
                        f"This suggests your portfolio is attractively valued relative to history."
                        + _pe_miss_note
                    )
                elif _wtd_pe <= _pe_benchmark * 1.20:
                    f3, f3x = 1.4, (
                        f"Your portfolio's weighted P/E of {_wtd_pe:.1f} is in line with the "
                        f"{_pe_bench_name} long run median of {_pe_benchmark:.0f}. "
                        f"Your portfolio appears fairly valued."
                        + _pe_miss_note
                    )
                elif _wtd_pe <= _pe_benchmark * 1.40:
                    f3, f3x = 0.8, (
                        f"Your portfolio's weighted P/E of {_wtd_pe:.1f} is above the "
                        f"{_pe_bench_name} long run median of {_pe_benchmark:.0f}. "
                        f"Your portfolio is trading at a slight premium to historical norms."
                        + _pe_miss_note
                    )
                else:
                    f3, f3x = 0.2, (
                        f"Your portfolio's weighted P/E of {_wtd_pe:.1f} is significantly above the "
                        f"{_pe_bench_name} long run median of {_pe_benchmark:.0f}. "
                        f"Consider whether the growth outlook justifies this premium valuation."
                        + _pe_miss_note
                    )
        except Exception:
            f3, f3x = 0.0, "P/E data unavailable — factor skipped."

        # Factor 4 — Sector Concentration (max 2.0)
        _ns = len(_sectors_set)
        _sect_note = (f" ({_unknown_sector_count} ticker(s) with unknown sector not counted.)"
                      if _unknown_sector_count > 0 else "")
        if _ns >= 4:
            f4, f4x = 2.0, f"Your portfolio spans 4 or more sectors. Good sector diversification.{_sect_note}"
        elif _ns == 3:
            f4, f4x = 1.4, (f"Your portfolio covers 3 sectors. "
                            f"Adding stocks from more sectors would improve this score.{_sect_note}")
        elif _ns == 2:
            f4, f4x = 0.8, f"Your portfolio is in 2 sectors. Consider diversifying into other sectors.{_sect_note}"
        elif _ns == 1:
            f4, f4x = 0.2, f"All your stocks are in the same sector. This increases risk significantly.{_sect_note}"
        else:
            f4, f4x = 0.8, "Sector data unavailable — score estimated. Check ticker symbols."

        # Factor 5 — Correlation (max 2.0)
        _hi_pairs = []
        if len(_ret_series) >= 2:
            try:
                _rdf2 = pd.DataFrame(_ret_series).dropna()
                _tks2 = list(_rdf2.columns)
                _corr = _rdf2.corr()
                for _a in range(len(_tks2)):
                    for _b in range(_a + 1, len(_tks2)):
                        _ti, _tj = _tks2[_a], _tks2[_b]
                        _cv = _corr.loc[_ti, _tj]
                        if _cv > 0.85:
                            _hi_pairs.append((_ti, _tj, _cv))
            except Exception:
                pass

        _np = len(_hi_pairs)
        if _np == 0:
            f5, f5x = 2.0, "No highly correlated stock pairs detected. Good diversification."
        elif _np == 1:
            _p = _hi_pairs[0]
            f5, f5x = 1.4, (f"{_p[0]} and {_p[1]} move closely together "
                            f"(correlation: {_p[2]:.2f}). They may not provide true diversification.")
        elif _np == 2:
            _ps = "; ".join(f"{p[0]} & {p[1]}" for p in _hi_pairs)
            f5, f5x = 0.8, (f"Two highly correlated pairs ({_ps}). "
                            "Consider swapping one for a less correlated alternative.")
        else:
            f5, f5x = 0.4, (f"{_np} highly correlated stock pairs detected. "
                            "Your portfolio may be less diversified than it appears.")

        _total_score = round(f1 + f2 + f3 + f4 + f5, 1)
        if   _total_score >= 9.0: _rating = "Excellent"
        elif _total_score >= 7.5: _rating = "Good"
        elif _total_score >= 6.0: _rating = "Fair"
        elif _total_score >= 4.0: _rating = "Needs Attention"
        else:                     _rating = "Poor"

        _progress_pct = min(_total_score / 10 * 100, 100)
        st.markdown(
            f'<div class="health-score-container" aria-live="polite">'
            f'<div style="margin:0 0 4px">'
            f'<span class="health-score-number">{_total_score}</span>'
            f'<span class="health-score-suffix">&thinsp;/ 10</span>'
            f'</div>'
            f'<div class="health-score-rating">{_rating}</div>'
            f'<div class="health-progress-track">'
            f'<div class="health-progress-fill" style="width:{_progress_pct:.1f}%"></div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(render_table(
            "Portfolio Health Score Breakdown",
            ["Factor", "Score", "Explanation"],
            [
                ["Diversification",      f"{f1:.1f} / 2.0", f1x],
                ["Sharpe Ratio",         f"{f2:.1f} / 2.0", f2x],
                ["Valuation (Weighted P/E)", f"{f3:.1f} / 2.0", f3x],
                ["Sector Concentration", f"{f4:.1f} / 2.0", f4x],
                ["Correlation",          f"{f5:.1f} / 2.0", f5x],
            ],
            wrap_last=True,
        ), unsafe_allow_html=True)

    except Exception as _e:
        st.warning(f"Health score could not be calculated: {_e}")

    # ── TABLE 2 — Adjusted Weight Portfolio ───────────────────────────────────
    st.subheader("Adjusted Weight Portfolio")

    try:
        opts = load_portfolio_options()
        all_opt_tickers = sorted({t for w in opts.values() for t in w})

        t2_rows, t2_classes = [], []
        col_totals = {"Max Sharpe Ratio": 0.0, "Min Volatility": 0.0, "Max Return": 0.0}

        for ticker in all_opt_tickers:
            name, _ = fetch_ticker_info(ticker)
            ms = opts.get("Max Sharpe Ratio", {}).get(ticker, 0.0)
            mv = opts.get("Min Volatility",   {}).get(ticker, 0.0)
            mr = opts.get("Max Return",       {}).get(ticker, 0.0)
            col_totals["Max Sharpe Ratio"] += ms
            col_totals["Min Volatility"]   += mv
            col_totals["Max Return"]       += mr
            t2_rows.append([ticker, name, f"{ms:.1f}%", f"{mv:.1f}%", f"{mr:.1f}%"])
            t2_classes.append("")

        t2_rows.append([
            "TOTAL", "",
            f"{col_totals['Max Sharpe Ratio']:.1f}%",
            f"{col_totals['Min Volatility']:.1f}%",
            f"{col_totals['Max Return']:.1f}%",
        ])
        t2_classes.append("total")

        st.markdown(render_table(
            "Recommended portfolio allocations for each optimization strategy.",
            ["Ticker", "Company", "Max Sharpe", "Min Volatility", "Max Return"],
            t2_rows, t2_classes,
        ), unsafe_allow_html=True)

        st.caption(
            "These are the recommended allocations for each strategy. "
            "The rebalancing tables below are based on your chosen target."
        )
    except Exception as e:
        st.warning(f"Could not load portfolio weights: {e}")

    # ── TABLE 3A — Action Table ───────────────────────────────────────────────
    st.subheader(f"Action Table — {target}")

    t3a_rows, t3a_classes = [], []

    for inst in instrs:
        if "Skipped" in inst["status"]:
            continue
        ticker = inst["ticker"]
        action = inst["action"]
        price_native, is_indian = get_native_price(ticker, positions, prices_latest)
        if price_native is None or price_native <= 0:
            continue

        raw_units = (inst["amount_inr"] / price_native if is_indian
                     else inst["amount_usd"] / price_native)
        units = math.ceil(raw_units) if action == "BUY" else math.floor(raw_units)
        if units <= 0:
            continue

        value_usd = (units * price_native / usd_inr) if is_indian else (units * price_native)
        name, _ = fetch_ticker_info(ticker)
        note = "New position — funded from rebalancing" if ticker in new_tickers else ""

        t3a_rows.append([action, ticker, name, str(units),
                          fmt_val(value_usd, currency, usd_inr), note])
        t3a_classes.append("buy-row" if action == "BUY" else "sell-row")

    if t3a_rows:
        st.markdown(render_table(
            f"Trades required to move your portfolio to the {target} allocation.",
            ["Action", "Ticker", "Company", "Units", f"Value ({currency})", "Note"],
            t3a_rows, t3a_classes,
        ), unsafe_allow_html=True)
    else:
        st.info("No trades required — your portfolio is already within the rebalancing threshold.")

    # ── TABLE 3B — Optimized Portfolio ────────────────────────────────────────
    st.subheader(f"Optimized Portfolio — {target}")

    final_shares = {t: pos["shares"] for t, pos in positions.items()}
    for inst in instrs:
        if "Skipped" in inst["status"]:
            continue
        ticker = inst["ticker"]
        action = inst["action"]
        price_native, is_indian = get_native_price(ticker, positions, prices_latest)
        if price_native is None or price_native <= 0:
            continue
        raw_units = (inst["amount_inr"] / price_native if is_indian
                     else inst["amount_usd"] / price_native)
        units = math.ceil(raw_units) if action == "BUY" else math.floor(raw_units)
        if units <= 0:
            continue
        if action == "BUY":
            final_shares[ticker] = final_shares.get(ticker, 0) + units
        else:
            final_shares[ticker] = max(0.0, final_shares.get(ticker, 0) - units)

    total_final_usd = 0.0
    for ticker, shares in final_shares.items():
        if shares <= 0:
            continue
        price_native, is_indian = get_native_price(ticker, positions, prices_latest)
        if price_native is None:
            continue
        value_usd = (shares * price_native / usd_inr) if is_indian else (shares * price_native)
        total_final_usd += value_usd

    t3b_rows, t3b_classes = [], []
    for ticker in sorted(final_shares.keys()):
        shares = final_shares[ticker]
        if shares <= 0:
            continue
        price_native, is_indian = get_native_price(ticker, positions, prices_latest)
        if price_native is None:
            continue
        value_usd = (shares * price_native / usd_inr) if is_indian else (shares * price_native)
        weight    = (value_usd / total_final_usd * 100) if total_final_usd > 0 else 0
        name, _   = fetch_ticker_info(ticker)
        t3b_rows.append([ticker, name, f"{shares:g}", f"{weight:.1f}%",
                          fmt_val(value_usd, currency, usd_inr)])
        t3b_classes.append("")

    t3b_rows.append(["TOTAL", "", "", "100.0%", fmt_val(total_final_usd, currency, usd_inr)])
    t3b_classes.append("total")

    st.markdown(render_table(
        "Your portfolio after all trades in the Action Table above are executed.",
        ["Ticker", "Company", "Units", "Weight", f"Value ({currency})"],
        t3b_rows, t3b_classes,
    ), unsafe_allow_html=True)

    # ── Download ──────────────────────────────────────────────────────────────
    xl = R["xl_path"]
    if os.path.exists(xl):
        with open(xl, "rb") as f:
            st.download_button(
                "Download Rebalancing Plan (Excel)",
                data=f.read(),
                file_name=os.path.basename(xl),
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    # ── Feature 1B — What if I add cash ──────────────────────────────────────
    st.markdown("---")
    st.subheader("What if I invest additional cash?")
    st.caption("See how new money would be allocated across your optimized portfolio")

    try:
        _tw_cash = load_portfolio_options().get(target, {})
        _tw_sum  = sum(_tw_cash.values()) or 1.0

        _slider_max = max(1000, int(total_final_usd * 5 / 100) * 100)
        _add_usd = st.slider(
            "Additional investment amount",
            min_value=0,
            max_value=_slider_max,
            value=0,
            step=100,
            key="cash_slider",
            help="Use arrow keys or drag to adjust",
        )

        _disp_val = (f"₹{_add_usd * usd_inr:,.0f}" if currency == "INR"
                     else f"${_add_usd:,.2f}")
        st.markdown(
            f'<p class="cash-display-value" aria-label="Additional investment: {_disp_val}">'
            f'{_disp_val}</p>',
            unsafe_allow_html=True,
        )

        if _add_usd > 0:
            _cash_data = []
            _total_add = 0.0

            for _tk in sorted(final_shares.keys()):
                _sh = final_shares[_tk]
                if _sh <= 0:
                    continue
                _pn, _ii = get_native_price(_tk, positions, prices_latest)
                if _pn is None or _pn <= 0:
                    continue
                _ow   = _tw_cash.get(_tk, 0) / _tw_sum
                _nat  = _add_usd * usd_inr if _ii else _add_usd
                _au   = math.floor(_ow * _nat / _pn)
                _av   = (_au * _pn / usd_inr) if _ii else (_au * _pn)
                _total_add += _av
                _cash_data.append({
                    "ticker": _tk, "is_indian": _ii, "price_native": _pn,
                    "shares_3b": _sh, "add_units": _au, "add_value_usd": _av,
                })

            _new_grand = total_final_usd + _total_add
            _cash_rows, _cash_cls = [], []

            for _d in _cash_data:
                _nm, _ = fetch_ticker_info(_d["ticker"])
                _nu = _d["shares_3b"] + _d["add_units"]
                _nv = (_nu * _d["price_native"] / usd_inr if _d["is_indian"]
                       else _nu * _d["price_native"])
                _nw = (_nv / _new_grand * 100) if _new_grand > 0 else 0
                _cash_rows.append([
                    _d["ticker"], _nm,
                    str(_d["add_units"]),
                    fmt_val(_d["add_value_usd"], currency, usd_inr),
                    f"{_nu:g}",
                    f"{_nw:.1f}%",
                    fmt_val(_nv, currency, usd_inr),
                ])
                _cash_cls.append("")

            _cash_rows.append([
                "TOTAL", "", "",
                fmt_val(_total_add, currency, usd_inr),
                "", "100.0%",
                fmt_val(_new_grand, currency, usd_inr),
            ])
            _cash_cls.append("total")

            _remainder = _add_usd - _total_add

            st.markdown(
                '<div aria-live="polite">'
                + render_table(
                    "Updated Allocation with Additional Cash",
                    ["Ticker", "Company", "Additional Units", "Additional Value",
                     "New Total Units", "New Total Weight", "New Total Value"],
                    _cash_rows, _cash_cls,
                )
                + "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"- **Total portfolio value after additional investment:** "
                f"{fmt_val(_new_grand, currency, usd_inr)}\n"
                f"- **Uninvested cash remainder:** "
                f"{fmt_val(_remainder, currency, usd_inr)} "
                f"(small amount left over due to whole unit rounding)"
            )

    except Exception as _e:
        st.warning(f"Cash allocation preview unavailable: {_e}")

    if st.button("Start over", key="reset"):
        st.session_state.results = None
        st.session_state.error = None
        st.rerun()
