import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, date

# Page Configuration
st.set_page_config(
    page_title="Advanced Quant Bot Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 24/7 Quant Scanner, Live Feed & Performance Analytics")
st.markdown("Advanced quantitative trading dashboard with backtest simulation, equity curve tracking, and multi-parameter filtering.")

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

# Helper function to categorize tickers into exchanges/universes for filtering
def categorize_ticker(ticker):
    penny_list = ["SNDL", "MULN", "BIOR", "OCGN", "SOUN", "RIG", "CLOV", "TELL", "CAN"]
    if ticker in penny_list:
        return "Penny / OTC"
    nasdaq_tech = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "NFLX", "AMD", "INTC", "QCOM"]
    if ticker in nasdaq_tech:
        return "NASDAQ (Tech/Growth)"
    return "S&P 500 / NYSE"

# ==========================================
# BACKTEST EVALUATION & METRICS ENGINE
# ==========================================
def run_backtest_simulation(df):
    if df.empty or "Ticker" not in df.columns:
        return pd.DataFrame()
    
    results = []
    
    for _, row in df.iterrows():
        ticker = str(row["Ticker"]).replace('-', '.')
        entry_time = str(row["Timestamp"])
        entry_price = float(row["Price"])
        stop_loss = float(row["Stop_Loss"])
        take_profit = float(row["Take_Profit"])
        conviction = int(row["Conviction"]) if "Conviction" in row else 78
        
        status = "Open ⏳"
        return_pct = 0.0
        
        try:
            start_date = entry_time.split()[0]
            hist = yf.Ticker(ticker).history(start=start_date, interval="1h")
            if not hist.empty:
                sub_hist = hist[hist.index >= entry_time]
                if not sub_hist.empty:
                    for _, bar in sub_hist.iterrows():
                        high = float(bar['High'])
                        low = float(bar['Low'])
                        
                        if high >= take_profit:
                            status = "Win 🎯"
                            return_pct = ((take_profit - entry_price) / entry_price) * 100
                            break
                        elif low <= stop_loss:
                            status = "Loss 🛡️"
                            return_pct = ((stop_loss - entry_price) / entry_price) * 100
                            break
                    else:
                        current_p = float(sub_hist.iloc[-1]['Close'])
                        return_pct = ((current_p - entry_price) / entry_price) * 100
        except Exception:
            pass
            
        results.append({
            "Timestamp": row["Timestamp"],
            "Date": pd.to_datetime(row["Timestamp"]).date(),
            "Ticker": row["Ticker"],
            "Exchange / Category": categorize_ticker(row["Ticker"]),
            "Entry Price": f"${entry_price:.2f}",
            "Take Profit": f"${take_profit:.2f}",
            "Stop Loss": f"${stop_loss:.2f}",
            "Conviction": conviction,
            "Status": status,
            "Return (%)": round(return_pct, 2)
        })
        
    return pd.DataFrame(results)

# Run base simulation
full_bt_df = run_backtest_simulation(df_signals)

# ==========================================
# SIDEBAR FILTER CONTROL CENTER
# ==========================================
st.sidebar.header("🎛️ Dashboard Filters")

if not full_bt_df.empty:
    min_date = pd.to_datetime(full_bt_df["Date"]).min()
    max_date = pd.to_datetime(full_bt_df["Date"]).max()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    all_tickers = sorted(full_bt_df["Ticker"].unique().tolist())
    selected_tickers = st.sidebar.multiselect("Filter by Ticker", options=all_tickers, default=all_tickers)

    all_categories = sorted(full_bt_df["Exchange / Category"].unique().tolist())
    selected_categories = st.sidebar.multiselect("Filter by Exchange / Universe", options=all_categories, default=all_categories)

    if len(date_range) == 2:
        start_d, end_d = date_range
        filtered_df = full_bt_df[
            (full_bt_df["Date"] >= start_d) & 
            (full_bt_df["Date"] <= end_d) &
            (full_bt_df["Ticker"].isin(selected_tickers)) &
            (full_bt_df["Exchange / Category"].isin(selected_categories))
        ]
    else:
        filtered_df = full_bt_df
else:
    filtered_df = pd.DataFrame()

# Calculate filtered metrics safely
if not filtered_df.empty:
    wins = len(filtered_df[filtered_df["Status"] == "Win 🎯"])
    losses = len(filtered_df[filtered_df["Status"] == "Loss 🛡️"])
    total_closed = wins + losses
    win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0
    cumulative_return = filtered_df["Return (%)"].sum()
else:
    wins, losses, win_rate, cumulative_return = 0, 0, 0.0, 0.0

# ==========================================
# 1. TOP OVERALL PERFORMANCE METRICS
# ==========================================
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("Total Signals", len(filtered_df))
with col2:
    st.metric("Strategy Win Rate", f"{win_rate:.1f}%", f"{wins}W / {losses}L")
with col3:
    st.metric("Cumulative Return", f"{cumulative_return:+.2f}%")
with col4:
    avg_conv = f"{filtered_df['Conviction'].mean():.1f}%" if not filtered_df.empty else "N/A"
    st.metric("Avg Conviction", avg_conv)
with col5:
    last_time = filtered_df.iloc[-1]["Timestamp"] if not filtered_df.empty else "Waiting for signals..."
    st.metric("Last Scan", last_time)

st.divider()

# ==========================================
# 2. OVERALL PERFORMANCE EQUITY CURVE
# ==========================================
st.subheader("📈 Overall Performance Equity Curve (Cumulative Return Over Time)")
if not filtered_df.empty:
    chart_data = filtered_df.sort_values(by="Timestamp").copy()
    chart_data["Cumulative Return (%)"] = chart_data["Return (%)"].cumsum()
    st.line_chart(chart_data.set_index("Timestamp")["Cumulative Return (%)"])
else:
    st.info("ℹ️ No performance data available yet. Waiting for signals to be logged.")

st.divider()

# ==========================================
# 3. BACKTEST PERFORMANCE ANALYTICS TABLE
# ==========================================
st.subheader("🎯 Backtest Simulation Results")
if not filtered_df.empty:
    display_cols = ["Timestamp", "Ticker", "Exchange / Category", "Entry Price", "Take Profit", "Stop Loss", "Conviction", "Status", "Return (%)"]
    st.dataframe(filtered_df[display_cols].sort_values(by="Timestamp", ascending=False), use_container_width=True, hide_index=True)
else:
    st.info("ℹ️ No historical signals logged yet. Your backtest table will populate automatically once your bot logs a signal.")

st.divider()

# ==========================================
# 4. LIVE MARKET WATCHLIST (TRIGGERED SYMBOLS)
# ==========================================
st.subheader("🔴 Live Market Feed (Triggered Symbols)")
if not filtered_df.empty:
    unique_filtered_tickers = filtered_df["Ticker"].dropna().unique().tolist()
    if unique_filtered_tickers:
        live_data = []
        for t in unique_filtered_tickers:
            try:
                yf_symbol = str(t).replace('-', '.')
                hist = yf.Ticker(yf_symbol).history(period="2d")
                if len(hist) >= 2:
                    price = float(hist['Close'].iloc[-1])
                    prev_close = float(hist['Close'].iloc[-2])
                    change_pct = ((price - prev_close) / prev_close) * 100
                elif len(hist) == 1:
                    price = float(hist['Close'].iloc[-1])
                    change_pct = 0.0
                else:
                    price, change_pct = None, 0.0
                
                live_data.append({
                    "Ticker": t,
                    "Live Price": f"${price:.2f}" if price else "N/A",
                    "Day Change": f"{change_pct:+.2f}%" if price else "0.00%",
                })
            except Exception:
                live_data.append({"Ticker": t, "Live Price": "N/A", "Day Change": "0.00%"})
                
        st.dataframe(pd.DataFrame(live_data), use_container_width=True, hide_index=True)
else:
    st.info("ℹ️ Real-time prices for triggered symbols will appear here once your bot logs a signal.")
