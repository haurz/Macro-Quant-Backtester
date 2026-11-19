"""Streamlit front end for the Macro Quant Backtester.

Run locally with ``streamlit run app.py``.
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ai.explainer import TradeExplainer
from data.data_loader import FXDataLoader
from engine.backtest import BacktestEngine
from engine.metrics import PerformanceEvaluator
from engine.signals import MovingAverageCrossover, RSIStrategy

st.set_page_config(page_title="Macro Quant Backtester", page_icon="📈", layout="wide")

TICKERS = {"EUR/USD": "EURUSD=X", "GBP/USD": "GBPUSD=X", "USD/JPY": "USDJPY=X"}
HISTORY_START = date(2015, 1, 1)
PCT_METRICS = {"CAGR", "Annualized Volatility", "Maximum Drawdown"}
MAX_AI_CALLS_PER_SESSION = 10


# --------------------------------------------------------------------------- data
@st.cache_data(ttl=6 * 3600, show_spinner="Loading market data...")
def load_data(ticker: str) -> tuple[pd.DataFrame, str]:
    """Return cleaned daily data and where it came from ('live' or 'snapshot')."""
    loader = FXDataLoader(start_date=HISTORY_START)
    try:
        return loader.fetch_data(ticker), "live"
    except Exception:
        # Yahoo Finance can rate-limit shared cloud hosts; fall back to the
        # CSV snapshot committed in data/cache so the app still works.
        cached = loader.cache_dir / f"{loader._cache_name(ticker)}.csv"
        if cached.exists():
            return loader._read_cache(cached), "snapshot"
        raise


def fmt(name: str, value: float) -> str:
    if pd.isna(value):
        return "-"
    if name in PCT_METRICS:
        return f"{value * 100:,.2f}%"
    if name in {"Total Trades", "Max Drawdown Duration (days)"}:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def get_cohere_key(typed: str) -> str | None:
    if typed.strip():
        return typed.strip()
    try:
        if "COHERE_API_KEY" in st.secrets:
            return str(st.secrets["COHERE_API_KEY"])
    except Exception:
        pass
    return os.getenv("COHERE_API_KEY")


# ------------------------------------------------------------------------ sidebar
st.sidebar.title("📈 Backtest settings")
pair = st.sidebar.selectbox("Currency pair", list(TICKERS))
ticker = TICKERS[pair]

try:
    market, source = load_data(ticker)
except Exception as exc:
    st.error(f"Could not load data for {ticker}: {exc}")
    st.stop()

first, last = market.index[0].date(), market.index[-1].date()
date_range = st.sidebar.date_input("Date range", value=(first, last), min_value=first, max_value=last)
if not (isinstance(date_range, tuple) and len(date_range) == 2):
    st.info("Select both a start and an end date.")
    st.stop()
start, end = date_range

strategy_name = st.sidebar.radio("Strategy", ["MA Crossover", "RSI"], horizontal=True)
allow_short = st.sidebar.checkbox("Allow short positions", value=True)

if strategy_name == "MA Crossover":
    fast = st.sidebar.slider("Fast SMA window", 5, 100, 20)
    slow = st.sidebar.slider("Slow SMA window", 20, 250, 50)
    if fast >= slow:
        st.sidebar.error("Fast window must be smaller than slow window.")
        st.stop()
    strategy = MovingAverageCrossover(fast_window=fast, slow_window=slow, allow_short=allow_short)
    params = {"fast_window": fast, "slow_window": slow}
    full_name = "MovingAverageCrossover"
else:
    window = st.sidebar.slider("RSI window", 5, 50, 14)
    oversold, overbought = st.sidebar.slider("Oversold / overbought levels", 5, 95, (30, 70))
    strategy = RSIStrategy(window=window, oversold=oversold, overbought=overbought, allow_short=allow_short)
    params = {"window": window, "oversold": oversold, "overbought": overbought}
    full_name = "RSIStrategy"

st.sidebar.subheader("Execution")
capital = st.sidebar.number_input("Initial capital ($)", 1_000.0, 10_000_000.0, 100_000.0, 1_000.0)
commission = st.sidebar.number_input("Commission (bps)", 0.0, 50.0, 3.0, 0.5)
slippage = st.sidebar.number_input("Slippage (bps)", 0.0, 50.0, 1.0, 0.5)

# ------------------------------------------------------------------------- engine
signals = strategy.generate(market)  # warm up on full history, then slice
signals = signals.loc[pd.Timestamp(start) : pd.Timestamp(end)]
if len(signals) < 30:
    st.warning("Please choose a longer date range (at least ~30 trading days).")
    st.stop()

result = BacktestEngine(capital, commission, slippage).run(signals)
sheet = PerformanceEvaluator(result).generate_tear_sheet()
curve = result.equity_curve
bench = curve["Close"] / curve["Close"].iloc[0] * capital

# ----------------------------------------------------------------------- header
st.title("Macro Quant Backtester")
st.caption(
    f"{pair} daily data, {start} to {end} | {strategy_name} | "
    "Signals execute on the next bar (no look-ahead). Educational research tool, not financial advice."
)
if source == "snapshot":
    st.info(f"Live data was unavailable, so a saved snapshot through {last} is being used.")

cols = st.columns(5)
for col, name in zip(cols, ["CAGR", "Sharpe Ratio", "Maximum Drawdown", "Win Rate (%)", "Total Trades"]):
    s, b = sheet.loc[name, "Strategy"], sheet.loc[name, "Benchmark"]
    delta = None
    if name in PCT_METRICS and pd.notna(b):
        delta = f"{(s - b) * 100:+.2f} pp vs buy & hold"
    elif name == "Sharpe Ratio":
        delta = f"{s - b:+.2f} vs buy & hold"
    col.metric(name, fmt(name, s), delta)

tab_perf, tab_sig, tab_trades, tab_ai, tab_about = st.tabs(
    ["Performance", "Signals", "Trades", "AI rationale", "About"]
)

# ------------------------------------------------------------------ performance
with tab_perf:
    drawdown = curve["equity"] / curve["equity"].cummax() - 1
    bench_dd = bench / bench.cummax() - 1
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=curve.index, y=curve["equity"], name=strategy_name), row=1, col=1)
    fig.add_trace(go.Scatter(x=bench.index, y=bench, name="Buy & hold", line=dict(dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown * 100, name="Strategy drawdown", fill="tozeroy"), row=2, col=1)
    fig.add_trace(go.Scatter(x=bench_dd.index, y=bench_dd * 100, name="B&H drawdown", line=dict(dash="dot")), row=2, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_layout(height=620, hovermode="x unified", legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")

    table = pd.DataFrame(
        {
            "Strategy": [fmt(n, v) for n, v in sheet["Strategy"].items()],
            "Buy & Hold": [fmt(n, v) for n, v in sheet["Benchmark"].items()],
        },
        index=sheet.index,
    )
    st.subheader("Tear sheet")
    st.dataframe(table, width="stretch")

# --------------------------------------------------------------------- signals
with tab_sig:
    ind_rows = 2 if strategy_name == "RSI" else 1
    fig = make_subplots(rows=ind_rows, cols=1, shared_xaxes=True,
                        row_heights=[0.7, 0.3] if ind_rows == 2 else [1.0], vertical_spacing=0.05)
    fig.add_trace(go.Scatter(x=signals.index, y=signals["Close"], name="Close"), row=1, col=1)
    if strategy_name == "MA Crossover":
        fig.add_trace(go.Scatter(x=signals.index, y=signals["SMA_fast"], name=f"SMA {fast}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=signals.index, y=signals["SMA_slow"], name=f"SMA {slow}"), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=signals.index, y=signals["RSI"], name="RSI"), row=2, col=1)
        fig.add_hline(y=oversold, line_dash="dash", row=2, col=1)
        fig.add_hline(y=overbought, line_dash="dash", row=2, col=1)
    log = result.trade_log
    for direction, symbol in (("LONG", "triangle-up"), ("SHORT", "triangle-down")):
        sel = log[log["direction"] == direction]
        fig.add_trace(go.Scatter(x=sel["entry_date"], y=sel["entry_price"], mode="markers",
                                 name=f"{direction.title()} entry", marker=dict(symbol=symbol, size=10)), row=1, col=1)
    fig.update_layout(height=560, hovermode="x unified", legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------------------- trades
with tab_trades:
    log = result.trade_log
    if log.empty:
        st.info("No trades were generated for these settings.")
    else:
        shown = log.copy()
        shown["entry_date"] = pd.to_datetime(shown["entry_date"]).dt.date
        shown["exit_date"] = pd.to_datetime(shown["exit_date"]).dt.date
        st.dataframe(
            shown,
            width="stretch",
            column_config={
                "entry_price": st.column_config.NumberColumn(format="%.5f"),
                "exit_price": st.column_config.NumberColumn(format="%.5f"),
                "pnl_percent": st.column_config.NumberColumn("pnl %", format="%.2f"),
                "net_pnl_dollars": st.column_config.NumberColumn("net pnl ($)", format="%.2f"),
            },
        )
        st.download_button("Download trade log (CSV)", shown.to_csv(index=False), "trade_log.csv", "text/csv")

# ----------------------------------------------------------------- AI rationale
with tab_ai:
    st.write(
        "An LLM (Cohere) translates the indicator values on a trade's entry date into plain English. "
        "It is restricted to the supplied data: no predictions, no outside news, no advice."
    )
    log = result.trade_log
    if log.empty:
        st.info("No trades to explain.")
    else:
        typed_key = st.text_input("Cohere API key (optional, never stored)", type="password")
        n = st.slider("Number of trades to explain", 1, min(5, len(log)), min(3, len(log)))
        used = st.session_state.get("ai_calls", 0)
        if st.button("Generate explanations", disabled=used >= MAX_AI_CALLS_PER_SESSION):
            explainer = TradeExplainer(api_key=get_cohere_key(typed_key))
            out = []
            with st.spinner("Asking the model..."):
                for _, trade in log.head(n).iterrows():
                    entry = pd.Timestamp(trade["entry_date"])
                    recent = signals.loc[signals.index <= entry].tail(10)
                    text = explainer.generate_rationale(trade.to_dict(), recent, full_name, params)
                    out.append((trade, text))
            st.session_state["ai_calls"] = used + n
            st.session_state["ai_out"] = out
        if used >= MAX_AI_CALLS_PER_SESSION:
            st.warning("Session limit reached. Refresh the page to start a new session.")
        for i, (trade, text) in enumerate(st.session_state.get("ai_out", []), 1):
            st.markdown(
                f"**Trade {i}: {trade['direction']}** | entry {pd.Timestamp(trade['entry_date']).date()} "
                f"@ {float(trade['entry_price']):.5f}"
            )
            st.write(text)

# ------------------------------------------------------------------------ about
with tab_about:
    st.markdown(
        """
**How it works**
- **Data:** daily FX bars from Yahoo Finance via `yfinance`, cleaned and cached as CSV.
- **Signals:** vectorised moving-average crossover and Wilder RSI strategies.
- **Execution:** each close-based signal is shifted one bar before it is traded, so no result uses future information. Commission and slippage are charged on every change of position.
- **Metrics:** CAGR, annualised volatility, Sharpe (zero risk-free rate), max drawdown and duration, win rate and profit factor, all compared with buy & hold.
- **Explainability:** an LLM describes why a mechanical signal fired, using only the data it is given.

**Limitations:** single asset, fixed position size, daily bars, no financing/swap costs. Past performance does not predict future results.
        """
    )
