#!/usr/bin/env python3
"""
fetch_util.py — hardened, shared yfinance access for the whole pipeline.

WHY THIS EXISTS
  A full run fires ~60-80 yfinance requests. yfinance hides server-side
  throttling / timeouts as EMPTY results (it logs "possibly delisted" and returns
  an empty frame), which the old call sites swallowed into None/NaN — indistinguishable
  from a stock that genuinely has no data. Degraded numbers then flowed silently
  into the covariance, the residuals, and module 0's Method-A/B flag.

WHAT THIS MODULE GUARANTEES, for EVERY yfinance call
  1. retry with exponential backoff + jitter and a per-call network timeout, and
  2. a hard distinction between
       - TRANSIENT failure (throttle / timeout / connection, after all retries)
           -> raise TransientFetchError  (LOUD: names the ticker AND the call), vs
       - GENUINE missing data (the call succeeded but the symbol truly has none)
           -> return None / empty, so callers keep their LEGITIMATE fallbacks
              (drop the ticker, neutral factor bucket, "Unknown" sector, ...).

HOW THE DISTINCTION IS MADE WITHOUT CHANGING THE DATA PATH
  yf.download records its per-symbol error in yfinance.shared._ERRORS even when it
  returns an empty frame. After an empty download we read that string; combined
  with the typed exceptions (YFRateLimitError = transient, YFPricesMissingError =
  genuine missing) this classifies the failure. The actual data returned is
  byte-identical to before — only the error handling changed.
"""

import json
import os
import random
import time

import pandas as pd
import yfinance as yf

try:                                            # yfinance stashes per-symbol errors here
    from yfinance import shared as _yf_shared
except Exception:                               # pragma: no cover - defensive
    _yf_shared = None

try:
    from yfinance.exceptions import (
        YFRateLimitError, YFPricesMissingError, YFTickerMissingError,
    )
except Exception:                               # pragma: no cover - older/newer yfinance
    class YFRateLimitError(Exception):
        pass

    class YFPricesMissingError(Exception):
        pass

    class YFTickerMissingError(Exception):
        pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Retry / backoff knobs ────────────────────────────────────────────────────────
ATTEMPTS   = 4         # total tries per call
BASE_DELAY = 1.0       # seconds; doubles each retry -> 1s, 2s, 4s
MAX_DELAY  = 8.0       # cap on a single backoff sleep
JITTER     = 0.4       # + random.uniform(0, JITTER) seconds, to de-sync bursts
TIMEOUT    = 15        # per-call network timeout (seconds) where yfinance accepts it

INDIA_SUFFIXES = (".NS", ".BO")

# Cross-process handoff so run_all (parent) can report WHICH ticker/call failed
# when a module subprocess dies on a transient fetch error.
FETCH_ERROR_PATH = os.path.join(SCRIPT_DIR, "fetch_error.json")

# Error-message substrings that mean GENUINE missing data (do NOT fail loud).
_MISSING_MARKERS = (
    "delisted", "pricesmissing", "no price data", "tickermissing",
    "no data found", "symbol may be delisted", "invalidperiod",
    "no timezone found", "no fundamentals",
)
# Error-message substrings that mean a TRANSIENT provider problem (fail loud
# after the retries are exhausted).
_TRANSIENT_MARKERS = (
    "ratelimit", "rate limit", "too many requests", "429",
    "timeout", "timed out", "read timed out", "connection", "max retries",
    "ssl", "temporarily", "503", "502", "504", "remotedisconnected",
    "connectionreset", "connection aborted",
)


class TransientFetchError(RuntimeError):
    """A yfinance call kept failing for network reasons (throttle/timeout/conn)."""

    def __init__(self, ticker, what, attempts, detail):
        self.ticker = ticker
        self.what = what
        self.attempts = attempts
        self.detail = str(detail)
        super().__init__(
            f"Couldn't fetch data for {ticker} ({what}) after {attempts} attempts: "
            f"{self.detail}. This is a temporary data-provider issue "
            f"(rate-limit / timeout / connection), not missing data — please retry."
        )


class _Retry(Exception):
    """Internal signal: this attempt failed transiently; retry it."""

    def __init__(self, detail):
        self.detail = detail
        super().__init__(str(detail))


def _looks_transient(msg):
    m = (msg or "").lower()
    if any(k in m for k in _MISSING_MARKERS):
        return False
    return any(k in m for k in _TRANSIENT_MARKERS)


def _classify_exc(exc):
    """True -> transient, False -> genuine-missing, None -> unknown."""
    if isinstance(exc, YFRateLimitError):
        return True
    if isinstance(exc, (YFPricesMissingError, YFTickerMissingError)):
        return False
    return True if _looks_transient(repr(exc)) else None


def fetch_with_retry(fn, *, ticker, what,
                     attempts=ATTEMPTS, base_delay=BASE_DELAY,
                     max_delay=MAX_DELAY, jitter=JITTER):
    """
    Call ``fn()`` with retry + exponential backoff + jitter on TRANSIENT failures.

    ``fn()`` may:
      * return a value         -> returned to the caller as-is (incl. None/empty,
                                  which callers use to mean GENUINE missing data),
      * raise _Retry(detail)   -> retried (transient), or
      * raise YFRateLimitError / a transient-looking exception -> retried, or
      * raise YFPricesMissingError / a missing-looking exception -> re-raised
                                  (genuine; caller decides), or
      * raise anything unknown -> treated as transient (retried, then loud).

    After ``attempts`` transient failures, raises TransientFetchError(ticker, what).
    """
    last = None
    for i in range(attempts):
        try:
            return fn()
        except _Retry as r:
            last = r.detail
        except Exception as exc:                # noqa: BLE001 - we re-raise genuine ones
            verdict = _classify_exc(exc)
            if verdict is False:
                raise                            # genuine missing/other -> caller handles
            last = repr(exc)                     # transient or unknown -> retry
        if i < attempts - 1:
            time.sleep(min(max_delay, base_delay * (2 ** i)) + random.uniform(0, jitter))
    raise TransientFetchError(ticker, what, attempts, last)


# ── Price history (one resolved symbol) ──────────────────────────────────────────

def fetch_close_series(symbol, *, start=None, end=None, period=None,
                       interval="1d", what=None, attempts=ATTEMPTS, timeout=TIMEOUT):
    """
    Download adjusted-close prices for ONE already-resolved symbol, hardened.

    Returns
      pd.Series  non-empty adjusted closes on success, or
      None       when the provider clearly reports GENUINE missing data.

    Raises
      TransientFetchError  if the provider kept failing transiently.
    """
    what = what or f"{interval} price history"

    def _attempt():
        key = symbol.upper()
        if _yf_shared is not None:
            _yf_shared._ERRORS.pop(key, None)   # clear stale error before the call
        kwargs = dict(interval=interval, auto_adjust=True, progress=False,
                      threads=False, timeout=timeout)
        if period is not None:
            kwargs["period"] = period
        else:
            kwargs["start"], kwargs["end"] = start, end
        raw = yf.download(symbol, **kwargs)
        if raw is not None and not raw.empty:
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()
            if not close.empty:
                return close
        # Empty frame -> consult yfinance's own recorded error to classify.
        err = ""
        if _yf_shared is not None:
            err = _yf_shared._ERRORS.get(key, "") or ""
        if _looks_transient(err):
            raise _Retry(err or "empty result (transient)")
        return None                              # genuine missing / unknown-empty

    return fetch_with_retry(_attempt, ticker=symbol, what=what, attempts=attempts)


# ── Generic history (rates / FX — single symbol, soft callers) ───────────────────

def fetch_history(symbol, *, period="10d", interval="1d", what=None,
                  auto_adjust=True, attempts=ATTEMPTS, timeout=TIMEOUT):
    """
    Ticker(symbol).history(...) hardened.

    Returns a DataFrame (possibly EMPTY for genuine missing data); raises
    TransientFetchError on persistent transient failure. Used for auxiliary,
    fallback-backed series (risk-free proxy, USD/INR) — those callers may catch
    TransientFetchError and fall back, since they are not portfolio risk inputs.
    """
    what = what or f"{symbol} {interval} history"

    def _attempt():
        try:
            h = yf.Ticker(symbol).history(
                period=period, interval=interval, auto_adjust=auto_adjust,
                timeout=timeout, raise_errors=True,
            )
        except Exception as exc:                # noqa: BLE001
            if _classify_exc(exc) is False:
                return pd.DataFrame()            # genuine missing
            raise _Retry(repr(exc))
        return h if h is not None else pd.DataFrame()

    return fetch_with_retry(_attempt, ticker=symbol, what=what, attempts=attempts)


# ── Ticker.info (sector / marketCap / priceToBook), memoised per process ─────────

_INFO_MEMO = {}     # symbol(upper) -> info dict; one network fetch per run


def fetch_info(symbol, *, attempts=ATTEMPTS, use_cache=True):
    """
    Fetch Ticker(symbol).info, hardened + memoised for the life of the process.

    Returns the info dict (possibly SPARSE). Callers keep their legitimate
    neutral-bucket / "Unknown"-sector fallbacks for missing FIELDS. Only a
    persistent TRANSIENT failure of the call itself raises TransientFetchError.
    """
    key = symbol.upper()
    if use_cache and key in _INFO_MEMO:
        return _INFO_MEMO[key]

    def _attempt():
        try:
            return yf.Ticker(symbol).info or {}
        except Exception as exc:                # noqa: BLE001
            if _classify_exc(exc) is False:
                return {}                        # genuine (e.g. unknown symbol) -> soft
            raise _Retry(repr(exc))

    info = fetch_with_retry(
        _attempt, ticker=symbol,
        what="company info (sector/marketCap/P/B)", attempts=attempts,
    )
    if use_cache:
        _INFO_MEMO[key] = info
    return info


# ── Symbol resolution (bare -> .NS / .BO) ────────────────────────────────────────

def candidate_symbols(raw_ticker):
    """
    Resolution candidates for a user ticker.

    A SUFFIXED ticker (contains a '.', e.g. "INFY.NS", "WIPRO.BO", "BRK.B") is used
    AS-IS — we NEVER also try the bare US symbol. That bare retry is exactly how an
    NSE/BSE name could wrongly resolve to a US ADR (e.g. INFY.NS -> INFY). Only a
    genuinely UNSUFFIXED ticker gets the .NS / .BO fallbacks.
    """
    t = raw_ticker.strip()
    if "." in t:
        return [t]
    return [t, f"{t}.NS", f"{t}.BO"]


def resolve_and_fetch(raw_ticker, start=None, end=None, interval="1d",
                      period=None, what=None, attempts=ATTEMPTS, prefer=None):
    """
    Resolve a ticker and fetch its adjusted-close Series.

    ``prefer`` (optional): a symbol already resolved elsewhere (e.g. module 0's
    handoff). When given, ONLY that symbol is fetched — no candidate probing — which
    avoids re-probing bare symbols a second time within a run.

    Returns
      (symbol, series, None)        first candidate that returns data, or
      (None, None, reason)          every candidate is GENUINELY missing.

    Raises
      TransientFetchError           no candidate returned data and at least one
                                    failed transiently (so a network blip never
                                    masquerades as a missing / short ticker).
    """
    cands = [prefer] if prefer else candidate_symbols(raw_ticker)
    transient_detail = None
    for symbol in cands:
        try:
            series = fetch_close_series(
                symbol, start=start, end=end, period=period,
                interval=interval, what=what, attempts=attempts,
            )
        except TransientFetchError as exc:
            transient_detail = exc.detail        # a sibling listing may still resolve
            continue
        if series is not None and not series.empty:
            return symbol, series, None
    if transient_detail is not None:
        raise TransientFetchError(
            raw_ticker, what or f"{interval} price history", attempts, transient_detail,
        )
    return None, None, f"No data returned (tried: {', '.join(cands)})"


# ── Cross-process fetch-error handoff (module subprocess -> run_all) ──────────────

def write_fetch_error(ticker, what, detail, module=None, path=FETCH_ERROR_PATH):
    """Persist a transient-failure marker so the parent run_all can report it."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "ticker": ticker, "call": what, "detail": str(detail),
                "module": module, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)
    except Exception:                            # pragma: no cover - best effort
        pass


def clear_fetch_error(path=FETCH_ERROR_PATH):
    """Remove any stale fetch-error marker (call at pipeline start)."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:                            # pragma: no cover
        pass


def read_fetch_error(path=FETCH_ERROR_PATH):
    """Return the fetch-error marker dict, or None if absent/unreadable."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def friendly_message(err):
    """
    Build the user-facing 'couldn't fetch TICKER' line from a marker dict or a
    TransientFetchError. Returns None if `err` carries nothing usable.
    """
    if isinstance(err, TransientFetchError):
        return (f"Couldn't fetch market data for {err.ticker} ({err.what}). "
                f"This is a temporary data-provider issue — please retry.")
    if isinstance(err, dict) and err.get("ticker"):
        return (f"Couldn't fetch market data for {err['ticker']} "
                f"({err.get('call', 'price data')}). "
                f"This is a temporary data-provider issue — please retry.")
    return None
