import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Quant Bot Dashboard & Backtester",
    page_icon="📈",
    layout="wide"
)

st.title("📈 24/7 Quant Scanner, Live Feed & Backtest Analytics")
st.markdown("Automated signal logging with live market tracking and historical strategy backtesting.")

# Load Historical Signal Data
@st.cache_data(ttl=15)
def load_signal_data():
    try:
        return pd.read_csv("backtest_signals.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=["Timestamp", "Ticker", "Price", "RSI", "NLP_Score", "Conviction", "Stop_Loss", "Take_Profit"])

df_signals = load_signal_data()

# ==========================================
# BACKTEST EVALUATION ENGINE
# ==========================================
def run_backtest_simulation(df):
    if df.empty:
        return df, 0, 0, 0, 0.0
    
    results = []
    wins = 0
    losses = 0
    open_trades = 0
    
    for _, row in df.iterrows():
        ticker = row["Ticker"].replace('-', '.')
        entry_time = row["Timestamp"]
        entry_price = float(row["Price"])
        stop_loss = float(row["Stop_Loss"])
        take_profit = float(row["Take_Profit"])
        
        status = "Open ⏳"
        exit_price = entry_price
        return_pct = 0.0
        
        try:
            # Fetch price data from entry time to now
            hist = yf.download(ticker, start=entry_time.split()[0], progress=False, interval="1h")
            if not hist.empty:
                # Flatten multi-index columns if present in newer yfinance versions
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.get_level_values(0)
                
                # Filter rows strictly after the entry timestamp
                sub_hist = hist[hist.index >= entry_time]
                
                if not sub_hist.empty:
                    for _, bar in sub_hist.iterrows():
                        high = float(bar['High'])
                        low = float(bar['Low'])
                        
                        # Check if Take Profit was hit first
                        if high >= take_profit:
                            status = "Win 🎯"
                            exit_price = take_profit
                            return_pct = ((take_profit - entry_price) / entry_price) * 100
                            wins += 1
                            break
                        # Check if Stop Loss was hit first
                        elif low <= stop_loss:
                            status = "Loss 🛡️"
                            exit_price = stop_loss
                            return_pct = ((stop_loss - entry_price) / entry_price) * 100
                            losses += 1
                            break
                    else:
                        # If neither hit yet, use the latest available price
                        current_p = float(sub_hist.iloc[-1]['Close'])
                        return_pct = ((current_p - entry_price) / entry_price) * 100
                        open_trades += 1
                else:
                    open_trades += 1
            else:
                open_trades += 1
        except Exception:
            open_trades += 1
            
        results.append({
            "Timestamp": row["Timestamp"],
            "Ticker": row["Ticker"],
            "Entry": entry_price,
            "Target": take_profit,
            "StopLoss": stop_loss,
            "Status": status,
            "Return (%)": round(return_pct, 2)
        })
        
    res_df = pd.DataFrame(results)
    total_closed = wins + losses
    win_rate = (wins / total_closed * 100) if total_closed > 0 else 0.0
    total_return = res_df["Return (%)"].sum() if not res_df.empty else 0.0
    
    return res_df, wins, losses, win_rate, total_return

# ==========================================
# 1. TOP METRICS SUMMARY
# ==========================================
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Signals Logged", len(df_signals))

# Run backtest stats for top metrics
bt_results, wins, losses, win_rate, total_return = run_backtest_simulation(df_signals)

with col2:
    st.metric("Strategy Win Rate", f"{win_rate:.1f}%", f"{wins} Wins / {losses} Losses")
with col3:
    st.metric("Cumulative Simulated Return", f"{total_return:+.2f}%")
with col4:
    last_time = df_signals.iloc[-1]["Timestamp"] if not df_signals.empty else "Waiting..."
    st.metric("Last Scan Activity", last_time)

st.divider()

# ==========================================
# 2. BACKTEST PERFORMANCE DASHBOARD
# ==========================================
st.subheader("🎯 Strategy Backtest Performance (Take-Profit vs. Stop-Loss)")
if df_signals.empty:
    st.info("ℹ️ No historical signals logged yet. Backtest results will populate once your bot logs signals.")
else:
    st.markdown("This simulation evaluates each signal against historical hourly price movement to see if the trade hit your **+8% Take Profit target** or **-4% Stop Loss floor** first.")
    
    # Display Backtest Table
    st.dataframe(bt_results.sort_values(by="Timestamp", ascending=False), use_container_width=True, hide_index=True)

st.divider()

# ==========================================
# 3. LIVE MARKET FEED FOR TRIGGERED SYMBOLS
# ==========================================
@st.cache_data(ttl=60)
def fetch_realtime_data(tickers):
    if not tickers:
        return pd.DataFrame()
    yf_tickers = [t.replace('-', '.') for t in tickers]
    data = []
    try:
        tickers_obj = yf.Tickers(" ".join(yf_tickers))
        for original_t, yf_t in zip(tickers, yf_tickers):
            info = tickers_obj.tickers[yf_t].fast_info
            price = info.last_price
            prev_close = info.previous_close
            change_pct = ((price - prev_close) / prev_close) * 100 if prev_close else 0.0
            data.append({
                "Ticker": original_t,
                "Live Price": f"${price:.2f}" if price else "N/A",
                "Day Change": f"{change_pct:+.2f}%" if change_pct else "0.00%",
            })
    except Exception:
        pass
    return pd.DataFrame(data)

st.subheader("🔴 Live Market Feed (Triggered Symbols)")
if not df_signals.empty:
    st.dataframe(fetch_realtime_data(df_signals["Ticker"].unique().tolist()), use_container_width=True, hide_index=True)
else:
    st.info("Waiting for first signal to populate live feed...")

st.divider()

# ==========================================
# 4. COMPLETE HISTORICAL LOG
# ==========================================
st.subheader("📊 Complete Raw Signal Log")
if not df_signals.empty:
    st.dataframe(df_signals.sort_values(by="Timestamp", ascending=False), use_container_width=True, hide_index=True)
