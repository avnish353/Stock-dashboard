"""
Real-Time Stock Market Dashboard (Streamlit + yfinance + Plotly)

How to run:
  1. Activate your venv in Windows:
       venv\Scripts\activate
  2. Run:
       streamlit run app.py

Dependencies (requirements.txt):
  streamlit
  pandas
  plotly
  yfinance
  requests
"""

import time
from datetime import datetime
from typing import Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Real-Time Stock Dashboard", layout="wide")
st.title("📈 Real-Time Stock Market Dashboard")

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Controls")

default_tickers = ["AAPL", "GOOGL", "MSFT"]
extra_candidates = ["TSLA", "AMZN", "META", "NVDA", "NFLX", "IBM", "INTC"]

tickers = st.sidebar.multiselect(
    "Select stocks",
    options=default_tickers + extra_candidates,
    default=default_tickers,
)

period = st.sidebar.selectbox(
    "Historical range (yfinance)",
    options=["1d", "5d", "1mo", "3mo", "6mo", "1y"],
    index=0,
)

interval = st.sidebar.selectbox(
    "Candle interval",
    options=["1m", "2m", "5m", "15m", "30m", "60m", "1d"],
    index=0,
)

ma_window = st.sidebar.slider("Moving average window (candles)", 5, 100, 20)
refresh_seconds = st.sidebar.slider("Auto-refresh every (seconds)", 15, 300, 60, 15)

st.sidebar.markdown("---")
st.sidebar.markdown("Tip: smaller refresh = more 'live' but more API calls.")

# ---------------------------
# Helpers: load data & indicators
# ---------------------------
@st.cache_data(ttl=60)
def load_data_yfinance(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """
    Fetch data and automatically adjust unsupported period/interval combos.
    """

    # Auto-fix invalid combinations
    invalid = {
        "30m": ["1mo", "3mo", "6mo", "1y"],
        "60m": ["1mo"],
        "1m":  ["1mo", "3mo", "6mo", "1y"]
    }

    if interval in invalid and period in invalid[interval]:
        # Force a working combination
        st.warning(
            f"⚠ {interval} interval is not available for period={period}. "
            f"Automatically switching to period='5d'."
        )
        period = "5d"

    df = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        return df

    df = df.copy().reset_index()
    if "Datetime" in df.columns:
        df.rename(columns={"Datetime": "DateTime"}, inplace=True)
    else:
        df.rename(columns={"Date": "DateTime"}, inplace=True)
     
    return df


def add_indicators(df: pd.DataFrame, ma_window: int) -> pd.DataFrame:
    """Add simple moving average and percent change columns."""
    if df.empty:
        return df
    df = df.copy()
    # Moving average on Close
    df["MA"] = df["Close"].rolling(window=ma_window, min_periods=1).mean()
    # Percent change from previous candle
    df["Pct_Change"] = df["Close"].pct_change().fillna(0) * 100
    return df


def compute_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Return last price, change, pct change, high, low, volume (day)."""
    if df.empty:
        return {}
    # Use last and first meaningful rows; ensure arithmetic done carefully
    last_close = float(df["Close"].iloc[-1])
    prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else last_close
    change = last_close - prev_close
    pct_change = (change / prev_close * 100) if prev_close != 0 else 0.0

    # For the selected historical range, show high/low/total volume
    day_high = float(df["High"].max())
    day_low = float(df["Low"].min())
    total_volume = int(df["Volume"].sum())

    return {
        "last_close": last_close,
        "change": change,
        "pct_change": pct_change,
        "day_high": day_high,
        "day_low": day_low,
        "total_volume": total_volume,
    }


def make_price_figure(df: pd.DataFrame, ticker: str) -> go.Figure:
    """Candlestick + MA Plotly figure."""
    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=df["DateTime"],
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name=f"{ticker} Price",
            increasing_line_color="#00b894",
            decreasing_line_color="#d63031",
        )
    )

    if "MA" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["DateTime"],
                y=df["MA"],
                mode="lines",
                name=f"MA ({ma_window})",
                line=dict(width=1.6, dash="dash"),
            )
        )

    fig.update_layout(
        title=f"{ticker} Price",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
    )

    return fig


# ---------------------------
# Main layout
# ---------------------------
if not tickers:
    st.warning("Select at least one ticker from the sidebar to begin.")
    st.stop()

tabs = st.tabs(tickers)

for tab, ticker in zip(tabs, tickers):
    with tab:
        st.subheader(f"{ticker}")
        df = load_data_yfinance(ticker, period, interval)

        if df.empty:
            st.error(f"No data available for {ticker} with period='{period}' and interval='{interval}'.")
            continue

        df = add_indicators(df, ma_window)
        kpis = compute_kpis(df)

        # KPIs row
        if kpis:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Last Price", f"{kpis['last_close']:.2f}", f"{kpis['change']:.2f} ({kpis['pct_change']:.2f}%)")
            c2.metric("Range High", f"{kpis['day_high']:.2f}")
            c3.metric("Range Low", f"{kpis['day_low']:.2f}")
            c4.metric("Total Volume", f"{kpis['total_volume']:,}")

        # Chart
        fig = make_price_figure(df, ticker)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

        # Data + small indicators
        col_left, col_right = st.columns([3, 1])
        with col_left:
            with st.expander("Show raw data (last 100 rows)"):
                st.dataframe(df.tail(100), use_container_width=True)
        with col_right:
            st.markdown("*Quick Stats*")
            st.write(f"Data points: {len(df)}")
            st.write(f"First: {df['DateTime'].iloc[0]}")
            st.write(f"Last: {df['DateTime'].iloc[-1]}")

st.markdown("---")
st.caption(f"Last fetched: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Auto-refresh every {refresh_seconds} seconds")

# ---------------------------
# Auto-refresh (simple)
# ---------------------------
# Note: streamlit will sleep here for refresh_seconds then re-run the script.
time.sleep(refresh_seconds)
st.experimental_rerun()