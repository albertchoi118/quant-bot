import streamlit as st
import pandas as pd
import yfinance as yf

# Page Configuration
st.set_page_config(
    page_title="Quant Bot Live Dashboard",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ 24/7 Quant Scanner & Real-Time Tracker")
st.markdown("Automated signal logging with live market feed integration.")

# Load Historical Signal Data safely
@st.cache_data(ttl=15)
def load_signal_data():
    try:
        df = pd.read_csv("backtest_signals.csv")
        if df.empty or "Ticker" not in df.columns:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()

df_signals = load_signal_data()

# Safe Metric Computations
total_signals = len(df_signals)
latest_target = df_signals.iloc[-1]["Ticker"] if not df_signals.empty and "Ticker" in df_signals.columns else "N/A"
avg_conviction = f"{df_signals['Conviction'].mean():.1f}%" if not df_signals.empty and "Conviction" in df_signals.columns and pd.notna(df_signals['Conviction'].mean()) else "N/A"
last_scan = df_signals.iloc[-1]["Timestamp"] if not df_signals.empty and "Timestamp" in df_signals.columns else "Waiting for first scan..."

# ==========================================
# 1. TOP METRICS SUMMARY
# ==========================================
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Signals Logged", total_signals)
with col2:
    st.metric("Latest Target", latest_target)
with col3:
    st.metric("Average Conviction", avg_conviction)
with col4:
    st.metric("Last Scan Activity", last_scan)

st.divider()

# ==========================================
# 2. REAL-TIME MARKET WATCHLIST
# ==========================================
st.subheader("🔴 Live Market Feed (Triggered Symbols)")

if not df_signals.empty and "Ticker" in df_signals.columns:
    unique_tickers = df_signals["Ticker"].dropna().unique().tolist()
    if unique_tickers:
        data = []
        for t in unique_tickers:
            try:
                # Format ticker for Yahoo Finance
                yf_symbol = str(t).replace('-', '.')
                ticker_obj = yf.Ticker(yf_symbol)
                fi = ticker_obj.fast_info
                price = getattr(fi, "last_price", None)
                prev_close = getattr(fi, "previous_close", None)
                change_pct = ((price - prev_close) / prev_close) * 100 if price and prev_close else 0.0
                
                data.append({
                    "Ticker": t,
                    "Live Price": f"${price:.2f}" if price else "N/A",
                    "Day Change": f"{change_pct:+.2f}%" if price and prev_close else "0.00%",
                })
            except Exception:
                data.append({"Ticker": t, "Live Price": "N/A", "Day Change": "0.00%"})
                
        st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)
    else:
        st.info("No tickers found in signal log yet.")
else:
    st.info("ℹ️ No signals recorded yet. `backtest_signals.csv` is missing or empty. Your live market feed will populate automatically once your bot logs its first signal!")

st.divider()

# ==========================================
# 3. COMPLETE HISTORICAL SIGNAL LOG
# ==========================================
st.subheader("📊 Complete Historical Signal Log")

if not df_signals.empty:
    st.dataframe(df_signals.sort_values(by="Timestamp", ascending=False), use_container_width=True, hide_index=True)
else:
    st.info("Waiting for first high-conviction scan result...")
