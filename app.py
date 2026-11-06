import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import plotly.graph_objects as go

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
        return pd.DataFrame(columns=[
            "Timestamp", "Ticker", "Exchange", "Price", "RSI", 
            "SMA200", "Stop_Loss", "Take_Profit", "Signal", 
            "Status", "Exit_Price", "Exit_Date"
        ])
    df = pd.read_csv(CSV_FILE)
    return df

df = load_data()

if df.empty:
    st.warning("⚠️ No signal data found in `backtest_signals.csv`. Run the bot or check your repository setup.")
    st.stop()

# ==========================================
# ROBUST CHRONOLOGICAL FIRST-HIT EVALUATION ENGINE
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
        
        status = row.get("Status", "OPEN")
        exit_price = row.get("Exit_Price", np.nan)
        return_pct = 0.0
        
        # If already closed in CSV history, retain its state
        if str(status).upper() not in ["OPEN", "ACTIVE / OPEN"]:
            row_dict = row.to_dict()
            if pd.notna(exit_price) and exit_price > 0:
                return_pct = ((float(exit_price) - entry_price) / entry_price) * 100
            row_dict["Return_Pct"] = round(return_pct, 2)
            results.append(row_dict)
            continue

        # Otherwise, dynamically evaluate current market outcome
        bars = pd.DataFrame()
        try:
            start_str = entry_time.strftime("%Y-%m-%d")
            end_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            bars = yf.download(ticker, start=start_str, end=end_str, interval="1d", progress=False)
            
            if not bars.empty:
                if isinstance(bars.columns, pd.MultiIndex):
                    bars.columns = bars.columns.get_level_values(0)
                bars = bars[bars.index >= pd.to_datetime(start_str)]
        except Exception:
            bars = pd.DataFrame()
            
        if bars.empty:
            try:
                curr_data = yf.download(ticker, period="5d", interval="1d", progress=False)
                if not curr_data.empty:
                    if isinstance(curr_data.columns, pd.MultiIndex):
                        curr_data.columns = curr_data.columns.get_level_values(0)
                    current_price = float(curr_data['Close'].iloc[-1])
                else:
                    current_price = entry_price
            except Exception:
                current_price = entry_price
                
            if current_price >= take_profit:
                status = "Win (TP Hit)"
                exit_price = take_profit
                return_pct = ((exit_price - entry_price) / entry_price) * 100
            elif current_price <= stop_loss:
                status = "Loss (SL Hit)"
                exit_price = stop_loss
                return_pct = ((exit_price - entry_price) / entry_price) * 100
            else:
                status = "Active / Open"
                exit_price = np.nan
                return_pct = ((current_price - entry_price) / entry_price) * 100
        else:
            hit_found = False
            for _, bar in bars.iterrows():
                high = float(bar['High'])
                low = float(bar['Low'])
                
                hit_tp = high >= take_profit
                hit_sl = low <= stop_loss
                
                if hit_tp and hit_sl:
                    status = "Loss (SL Hit)"
                    exit_price = stop_loss
                    return_pct = ((exit_price - entry_price) / entry_price) * 100
                    hit_found = True
                    break
                elif hit_tp:
                    status = "Win (TP Hit)"
                    exit_price = take_profit
                    return_pct = ((exit_price - entry_price) / entry_price) * 100
                    hit_found = True
                    break
                elif hit_sl:
                    status = "Loss (SL Hit)"
                    exit_price = stop_loss
                    return_pct = ((exit_price - entry_price) / entry_price) * 100
                    hit_found = True
                    break
            
            if not hit_found:
                last_price = float(bars['Close'].iloc[-1])
                status = "Active / Open"
                exit_price = np.nan
                return_pct = ((last_price - entry_price) / entry_price) * 100
                
        row_dict = row.to_dict()
        row_dict["Exit_Price"] = round(exit_price, 2) if pd.notna(exit_price) else np.nan
        row_dict["Return_Pct"] = round(return_pct, 2)
        row_dict["Status"] = status
        results.append(row_dict)
        
    return pd.DataFrame(results)

with st.spinner("Analyzing trade outcomes and historical price action..."):
    perf_df = evaluate_trade_outcomes(df)

# ==========================================
# SIDEBAR FILTERS
# ==========================================
st.sidebar.header("🔍 Filter Signals")

all_exchanges = sorted(perf_df["Exchange"].dropna().unique().tolist()) if "Exchange" in perf_df.columns else []
selected_exchanges = st.sidebar.multiselect(
    "Select Exchange(s)", 
    options=all_exchanges, 
    default=[]
)

all_tickers = sorted(perf_df["Ticker"].dropna().unique().tolist())
selected_tickers = st.sidebar.multiselect(
    "Select Ticker(s)", 
    options=all_tickers, 
    default=[]
)

filtered_df = perf_df.copy()
if selected_exchanges:
    filtered_df = filtered_df[filtered_df["Exchange"].isin(selected_exchanges)]
if selected_tickers:
    filtered_df = filtered_df[filtered_df["Ticker"].isin(selected_tickers)]

# ==========================================
# TOP METRICS SUMMARY
# ==========================================
total_signals = len(filtered_df)
wins = len(filtered_df[filtered_df["Status"].str.contains("Win", case=False, na=False)])
losses = len(filtered_df[filtered_df["Status"].str.contains("Loss", case=False, na=False)])
active = len(filtered_df[filtered_df["Status"].str.contains("Open|Active", case=False, na=False)])
win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0.0
avg_return = filtered_df["Return_Pct"].mean() if total_signals > 0 else 0.0

st.markdown(f"""
<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 0.75rem; margin-bottom: 1.5rem;">
    <div style="background: rgba(128,128,128,0.08); padding: 0.85rem; border-radius: 8px; text-align: center; border: 1px solid rgba(128,128,128,0.15);">
        <div style="font-size: 0.8rem; color: gray; font-weight: 500;">Total Signals</div>
        <div style="font-size: 1.4rem; font-weight: 700;">{total_signals}</div>
    </div>
    <div style="background: rgba(128,128,128,0.08); padding: 0.85rem; border-radius: 8px; text-align: center; border: 1px solid rgba(128,128,128,0.15);">
        <div style="font-size: 0.8rem; color: gray; font-weight: 500;">Wins (TP)</div>
        <div style="font-size: 1.4rem; font-weight: 700; color: #00D26A;">{wins}</div>
    </div>
    <div style="background: rgba(128,128,128,0.08); padding: 0.85rem; border-radius: 8px; text-align: center; border: 1px solid rgba(128,128,128,0.15);">
        <div style="font-size: 0.8rem; color: gray; font-weight: 500;">Losses (SL)</div>
        <div style="font-size: 1.4rem; font-weight: 700; color: #FF3B30;">{losses}</div>
    </div>
    <div style="background: rgba(128,128,128,0.08); padding: 0.85rem; border-radius: 8px; text-align: center; border: 1px solid rgba(128,128,128,0.15);">
        <div style="font-size: 0.8rem; color: gray; font-weight: 500;">Win Rate</div>
        <div style="font-size: 1.4rem; font-weight: 700;">{win_rate:.1f}%</div>
    </div>
    <div style="background: rgba(128,128,128,0.08); padding: 0.85rem; border-radius: 8px; text-align: center; border: 1px solid rgba(128,128,128,0.15);">
        <div style="font-size: 0.8rem; color: gray; font-weight: 500;">Active / Open</div>
        <div style="font-size: 1.4rem; font-weight: 700; color: #0084FF;">{active}</div>
    </div>
    <div style="background: rgba(128,128,128,0.08); padding: 0.85rem; border-radius: 8px; text-align: center; border: 1px solid rgba(128,128,128,0.15);">
        <div style="font-size: 0.8rem; color: gray; font-weight: 500;">Avg Return</div>
        <div style="font-size: 1.4rem; font-weight: 700; color: {'#00D26A' if avg_return >= 0 else '#FF3B30'};">{avg_return:+.2f}%</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# EQUITY CURVE
# ==========================================
st.subheader("📊 Strategy Cumulative Equity Curve")

chart_df = filtered_df.copy()
chart_df["Timestamp"] = pd.to_datetime(chart_df["Timestamp"])
chart_df = chart_df.sort_values("Timestamp")

if not chart_df.empty:
    chart_df["Cumulative_Return"] = ((1 + chart_df["Return_Pct"] / 100).cumprod() - 1) * 100

    fig = go.Figure()
    
    is_positive = chart_df["Cumulative_Return"].iloc[-1] >= 0 if not chart_df.empty else True
    curve_color = "#00D26A" if is_positive else "#FF3B30"
    fill_color = "rgba(0, 210, 106, 0.1)" if is_positive else "rgba(255, 59, 48, 0.1)"

    fig.add_trace(go.Scatter(
        x=chart_df["Timestamp"],
        y=chart_df["Cumulative_Return"],
        mode="lines+markers",
        name="Equity Curve",
        line=dict(color=curve_color, width=2.5),
        marker=dict(size=4),
        fill="tozeroy",
        fillcolor=fill_color
    ))

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(128, 128, 128, 0.2)",
        rangeslider_visible=True
    )
    
    fig.update_yaxes(
        title_text="Cumulative Return (%)",
        showgrid=True,
        gridcolor="rgba(128, 128, 128, 0.2)"
    )

    fig.update_layout(
        height=400,
        margin=dict(l=10, r=10, t=20, b=10),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No data available for the selected filters.")

st.markdown("---")

# ==========================================
# LIVE ACTIVE MARKET FEED SECTION
# ==========================================
st.subheader("🟢 Live Active Market Feed (Pending Outcomes)")
active_df = filtered_df[filtered_df["Status"].str.contains("Open|Active", case=False, na=False)]

if active_df.empty:
    st.info("No active trades currently pending.")
else:
    active_cols = ["Timestamp", "Ticker", "Exchange", "Price", "Exit_Price", "Return_Pct", "Status", "RSI", "Stop_Loss", "Take_Profit"]
    st.dataframe(
        active_df[active_cols].sort_values(by="Timestamp", ascending=False),
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
    "Return_Pct", "Status", "RSI", "SMA200", "Stop_Loss", "Take_Profit"
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
