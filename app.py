import streamlit as st
import pandas as pd
import yfinance as yf
import os

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Quant Bot Live Dashboard",
    page_icon="📈",
    layout="wide"
)

CSV_FILE = "backtest_signals.csv"

st.title("📈 24/7 Quantitative Stock Scanner Dashboard")
st.markdown("Real-time monitoring for S&P 500 & Nasdaq-100 signals with dynamic ATR risk management.")

# ==========================================
# LOAD DATA
# ==========================================
@st.cache_data(ttl=30) # Refresh data cache every 30 seconds
def load_data():
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        return pd.DataFrame()
    try:
        df = pd.read_csv(CSV_FILE)
        return df
    except Exception as e:
        st.error(f"Error loading CSV data: {e}")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.warning(f"⚠️ No signal data found in `{CSV_FILE}` yet. Waiting for the bot cron trigger to log the first trades.")
else:
    # Ensure necessary columns exist
    expected_cols = ['Timestamp', 'Ticker', 'Price', 'Stop_Loss', 'Take_Profit', 'Status', 'Exit_Price']
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    # ==========================================
    # METRICS SUMMARY BAR
    # ==========================================
    total_trades = len(df)
    open_trades_df = df[df['Status'].astype(str).str.upper().str.contains("OPEN|ACTIVE")]
    closed_trades_df = df[~df['Status'].astype(str).str.upper().str.contains("OPEN|ACTIVE")]
    
    active_count = len(open_trades_df)
    closed_count = len(closed_trades_df)
    
    # Calculate win rate & closed returns from logged Exit_Price
    wins = 0
    losses = 0
    total_pnl_pct = 0.0
    
    for _, row in closed_trades_df.iterrows():
        entry = float(row['Price'])
        exit_val = row['Exit_Price']
        status_str = str(row['Status']).upper()
        
        if pd.notna(exit_val):
            exit_price = float(exit_val)
            pct_gain = ((exit_price - entry) / entry) * 100
            total_pnl_pct += pct_gain
            if "TP" in status_str or exit_price > entry:
                wins += 1
            else:
                losses += 1

    win_rate = (wins / closed_count * 100) if closed_count > 0 else 0.0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Signals Logged", total_trades)
    col2.metric("Active Open Positions", active_count)
    col3.metric("Closed Positions", closed_count)
    col4.metric("Win Rate", f"{win_rate:.1f}%")
    col5.metric("Closed Cumulative Return", f"{total_pnl_pct:+.2f}%")

    st.markdown("---")

    # ==========================================
    # ACTIVE OPEN POSITIONS (Real-Time Intraday Live Price)
    # ==========================================
    st.subheader("🟢 Active Open Positions (Live Intraday Feed)")
    if active_count > 0:
        active_rows = []
        for _, row in open_trades_df.iterrows():
            ticker = row['Ticker']
            entry_price = float(row['Price'])
            
            # Fetch high-resolution live intraday price (1-minute intervals)
            curr_price = entry_price
            unrealized_pnl = 0.0
            try:
                t_obj = yf.Ticker(ticker)
                # Try fetching today's 1-minute intraday bars for real-time pricing
                hist = t_obj.history(period="1d", interval="1m")
                if hist.empty:
                    # Fallback to daily if market is closed / pre-market
                    hist = t_obj.history(period="2d", interval="1d")
                    
                if not hist.empty:
                    if isinstance(hist.columns, pd.MultiIndex):
                        hist.columns = hist.columns.get_level_values(0)
                    curr_price = float(hist.iloc[-1]['Close'])
                    unrealized_pnl = ((curr_price - entry_price) / entry_price) * 100
            except Exception:
                pass

            active_rows.append({
                'Timestamp': row['Timestamp'],
                'Ticker': ticker,
                'Entry Price': entry_price,
                'Live Price': round(curr_price, 2),
                'Unrealized PnL (%)': round(unrealized_pnl, 2),
                'Stop_Loss': row['Stop_Loss'],
                'Take_Profit': row['Take_Profit'],
                'RSI': row['RSI'],
                'Status': row['Status']
            })
            
        active_display_df = pd.DataFrame(active_rows)
        st.dataframe(active_display_df.reset_index(drop=True), use_container_width=True)
    else:
        info_col1, _ = st.columns([1, 4])
        with info_col1:
            st.info("No active open positions right now.")

    st.markdown("---")

    # ==========================================
    # COMPLETE SIGNAL & TRADE HISTORY LOG
    # ==========================================
    st.subheader("📜 Complete Signal & Trade History Log")
    
    # Optional search filter
    search_ticker = st.text_input("🔍 Filter by Ticker symbol:", "").upper().strip()
    
    display_df = df.copy()
    if search_ticker:
        display_df = display_df[display_df['Ticker'].str.contains(search_ticker, na=False)]
        
    st.dataframe(
        display_df.sort_values(by="Timestamp", ascending=False).reset_index(drop=True),
        use_container_width=True
    )

    # Manual refresh button
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
