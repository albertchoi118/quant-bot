import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Quant Scanner Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Live Quant Scanner & Signal Dashboard")
st.markdown("Monitoring automated institutional mean-reversion setups, historical backtests, and active trades.")

# ==========================================
# LOAD DATA
# ==========================================
CSV_FILE = "backtest_signals.csv"

@st.cache_data(ttl=60)
def load_data():
    if not Path(CSV_FILE).is_file():
        return pd.DataFrame(columns=["Timestamp", "Ticker", "Exchange", "Price", "RSI", "NLP_Score", "Conviction", "Stop_Loss", "Take_Profit"])
    df = pd.read_csv(CSV_FILE)
    return df

df = load_data()

if df.empty:
    st.warning("⚠️ No signal data found in `backtest_signals.csv`. Run the bot or add mock data.")
    st.stop()

# ==========================================
# CHRONOLOGICAL FIRST-HIT SIMULATION ENGINE
# ==========================================
@st.cache_data(ttl=300)
def evaluate_trade_outcomes(signal_df):
    results = []
    
    for _, row in signal_df.iterrows():
        ticker = row["Ticker"]
        entry_time = pd.to_datetime(row["Timestamp"])
        entry_price = float(row["Price"])
        stop_loss = float(row["Stop_Loss"])
        take_profit = float(row["Take_Profit"])
        
        status = "Active / Open"
        return_pct = 0.0
        exit_price = entry_price
        
        try:
            start_str = entry_time.strftime("%Y-%m-%d")
            end_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            bars = yf.download(ticker, start=start_str, end=end_str, interval="1h", progress=False)
            if bars.empty:
                bars = yf.download(ticker, start=start_str, end=end_str, interval="1d", progress=False)
            
            if not bars.empty:
                if isinstance(bars.columns, pd.MultiIndex):
                    bars.columns = bars.columns.get_level_values(0)
                bars = bars[bars.index >= entry_time]
        except Exception:
            bars = pd.DataFrame()
            
        if bars.empty:
            # Fallback to current price if no historical post-entry bars are available
            try:
                curr_data = yf.download(ticker, period="1d", interval="1m", progress=False)
                current_price = float(curr_data['Close'].iloc[-1]) if not curr_data.empty else entry_price
            except Exception:
                current_price = entry_price
                
            if current_price >= take_profit:
                status = "Win (TP Hit)"
                return_pct = ((take_profit - entry_price) / entry_price) * 100
                exit_price = take_profit
            elif current_price <= stop_loss:
                status = "Loss (SL Hit)"
                return_pct = ((stop_loss - entry_price) / entry_price) * 100
                exit_price = stop_loss
            else:
                status = "Active / Open"
                return_pct = ((current_price - entry_price) / entry_price) * 100
                exit_price = current_price
        else:
            # Bar-by-bar chronological check to find which threshold hit FIRST
            hit_found = False
            for _, bar in bars.iterrows():
                high = float(bar['High'])
                low = float(bar['Low'])
                
                hit_tp = high >= take_profit
                hit_sl = low <= stop_loss
                
                if hit_tp and hit_sl:
                    # Conservative tie-breaker if both hit in same bar
                    status = "Loss (SL Hit)"
                    return_pct = ((stop_loss - entry_price) / entry_price) * 100
                    exit_price = stop_loss
                    hit_found = True
                    break
                elif hit_tp:
                    status = "Win (TP Hit)"
                    return_pct = ((take_profit - entry_price) / entry_price) * 100
                    exit_price = take_profit
                    hit_found = True
                    break
                elif hit_sl:
                    status = "Loss (SL Hit)"
                    return_pct = ((stop_loss - entry_price) / entry_price) * 100
                    exit_price = stop_loss
                    hit_found = True
                    break
            
            if not hit_found:
                last_price = float(bars['Close'].iloc[-1])
                status = "Active / Open"
                return_pct = ((last_price - entry_price) / entry_price) * 100
                exit_price = last_price
                
        row_dict = row.to_dict()
        row_dict["Exit_Price"] = round(exit_price, 2)
        row_dict["Return_Pct"] = round(return_pct, 2)
        row_dict["Status"] = status
        results.append(row_dict)
        
    return pd.DataFrame(results)

with st.spinner("Analyzing trade outcomes and historical price action..."):
    perf_df = evaluate_trade_outcomes(df)

# ==========================================
# SIDEBAR FILTERS (FIXED: Empty = Everything)
# ==========================================
st.sidebar.header("🔍 Filter Signals")

all_exchanges = sorted(perf_df["Exchange"].dropna().unique().tolist())
selected_exchanges = st.sidebar.multiselect(
    "Select Exchange(s)", 
    options=all_exchanges, 
    default=[] # Empty = Everything
)

all_tickers = sorted(perf_df["Ticker"].dropna().unique().tolist())
selected_tickers = st.sidebar.multiselect(
    "Select Ticker(s)", 
    options=all_tickers, 
    default=[] # Empty = Everything
)

# Apply filters logically
filtered_df = perf_df.copy()
if selected_exchanges:
    filtered_df = filtered_df[filtered_df["Exchange"].isin(selected_exchanges)]
if selected_tickers:
    filtered_df = filtered_df[filtered_df["Ticker"].isin(selected_tickers)]

# ==========================================
# TOP METRICS SUMMARY
# ==========================================
total_signals = len(filtered_df)
wins = len(filtered_df[filtered_df["Status"].str.contains("Win")])
losses = len(filtered_df[filtered_df["Status"].str.contains("Loss")])
active = len(filtered_df[filtered_df["Status"].str.contains("Active")])
win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0
avg_return = filtered_df["Return_Pct"].mean() if total_signals > 0 else 0.0

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total Signals", total_signals)
col2.metric("Wins (TP)", wins)
col3.metric("Losses (SL)", losses)
col4.metric("Win Rate", f"{win_rate:.1f}%")
col5.metric("Active / Open", active)
col6.metric("Avg Return", f"{avg_return:+.2f}%")

st.markdown("---")

# ==========================================
# CUMULATIVE TIME SERIES CHART WITH GRANULARITY
# ==========================================
st.subheader("📊 Cumulative Performance Over Time")

granularity = st.radio(
    "Select Chart Granularity", 
    options=["Daily", "Weekly", "Monthly"], 
    horizontal=True
)

chart_df = filtered_df.copy()
chart_df["Parsed_Time"] = pd.to_datetime(chart_df["Timestamp"])

if granularity == "Daily":
    chart_df["Period"] = chart_df["Parsed_Time"].dt.date
elif granularity == "Weekly":
    chart_df["Period"] = chart_df["Parsed_Time"].dt.to_period("W").dt.start_time.dt.date
else:
    chart_df["Period"] = chart_df["Parsed_Time"].dt.to_period("M").dt.start_time.dt.strftime("%Y-%m")

# Group and compute cumulative returns
grouped = chart_df.groupby("Period")["Return_Pct"].mean().reset_index()
grouped = grouped.sort_values("Period")
grouped["Cumulative_Return"] = grouped["Return_Pct"].cumsum()

st.line_chart(grouped.set_index("Period")["Cumulative_Return"])

st.markdown("---")

# ==========================================
# LIVE ACTIVE MARKET FEED SECTION
# ==========================================
st.subheader("🟢 Live Active Market Feed (Pending Outcomes)")
active_df = filtered_df[filtered_df["Status"] == "Active / Open"]

if active_df.empty:
    st.info("No active trades currently pending. All past signals have hit either Take Profit or Stop Loss.")
else:
    active_display_cols = ["Timestamp", "Ticker", "Exchange", "Price", "Exit_Price", "Return_Pct", "Status", "RSI", "Conviction", "Stop_Loss", "Take_Profit"]
    st.dataframe(
        active_df[active_display_cols].sort_values(by="Timestamp", ascending=False),
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")

# ==========================================
# FULL SIGNAL LOG & PERFORMANCE TABLE
# ==========================================
st.subheader("📋 Complete Signal Log")

display_cols = [
    "Timestamp", "Ticker", "Exchange", "Price", "Exit_Price", 
    "Return_Pct", "Status", "RSI", "NLP_Score", "Conviction", "Stop_Loss", "Take_Profit"
]

st.dataframe(
    filtered_df[display_cols].sort_values(by="Timestamp", ascending=False),
    use_container_width=True,
    hide_index=True
)

# ==========================================
# DOWNLOAD CSV BUTTON
# ==========================================
csv_data = filtered_df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download Filtered Signals as CSV",
    data=csv_data,
    file_name="filtered_quant_signals.csv",
    mime="text/csv"
)
