#!/usr/bin/env python3
"""
Dynamic Quant Sentinel - S&P 500 & Nasdaq-100 Scanner & Position Manager
========================================================================
- Dynamically fetches full S&P 500 & Nasdaq-100 universes.
- Uses a local CSV (`active_positions.csv`) for persistent state tracking.
- Only loads and processes OPEN positions for active monitoring.
- Scans for Buy signals (skipping open positions) & saves new entries to CSV.
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import requests

try:
    import yfinance as yf
except ImportError:
    yf = None

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Configuration & Credentials
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "YOUR_TELEGRAM_CHAT_ID")
CSV_FILENAME = "active_positions.csv"

# ==========================================
# 1. CSV STATE MANAGEMENT (OPEN ONLY FILTER)
# ==========================================
def initialize_csv_if_missing():
    """Ensures the CSV file exists with the proper schema."""
    if not os.path.exists(CSV_FILENAME):
        df = pd.DataFrame(columns=[
            "ticker", "entry_date", "entry_price", 
            "take_profit_target", "stop_loss_target", "status", "alert_sent"
        ])
        df.to_csv(CSV_FILENAME, index=False)

def get_open_positions() -> pd.DataFrame:
    """Loads the CSV and filters strictly for OPEN positions in memory."""
    initialize_csv_if_missing()
    df = pd.read_csv(CSV_FILENAME)
    if df.empty:
        return df
    return df[df['status'] == 'OPEN'].copy()

def update_position_status(ticker: str, new_status: str):
    """Updates a specific ticker's status (e.g., from OPEN to CLOSED) in the full CSV."""
    initialize_csv_if_missing()
    df = pd.read_csv(CSV_FILENAME)
    df.loc[df['ticker'] == ticker, 'status'] = new_status
    df.to_csv(CSV_FILENAME, index=False)

def append_new_positions(new_rows: list):
    """Appends newly triggered buy signals to the CSV file."""
    if not new_rows:
        return
    initialize_csv_if_missing()
    df_full = pd.read_csv(CSV_FILENAME)
    df_new = pd.DataFrame(new_rows)
    df_combined = pd.concat([df_full, df_new], ignore_index=True)
    df_combined.to_csv(CSV_FILENAME, index=False)

# ==========================================
# 2. DYNAMIC UNIVERSE RETRIEVAL
# ==========================================
def get_dynamic_universe() -> list:
    """Dynamically fetches all S&P 500 and Nasdaq-100 tickers from Wikipedia."""
    tickers = set()
    
    # 1. Fetch S&P 500
    try:
        sp_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        sp_table = pd.read_html(sp_url)[0]
        sp_tickers = sp_table['Symbol'].tolist()
        tickers.update([t.replace('.', '-') for t in sp_tickers])
        logging.info(f"Loaded {len(sp_tickers)} S&P 500 tickers.")
    except Exception as e:
        logging.error(f"Failed to fetch S&P 500 universe: {e}")

    # 2. Fetch Nasdaq-100
    try:
        ndx_url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        ndx_tables = pd.read_html(ndx_url)
        ndx_table = None
        for table in ndx_tables:
            if 'Ticker' in table.columns or 'Symbol' in table.columns:
                ndx_table = table
                break
        if ndx_table is not None:
            col = 'Ticker' if 'Ticker' in ndx_table.columns else 'Symbol'
            ndx_tickers = ndx_table[col].tolist()
            tickers.update([t.replace('.', '-') for t in ndx_tickers])
            logging.info(f"Loaded Nasdaq-100 components. Total unique universe size: {len(tickers)}")
    except Exception as e:
        logging.error(f"Failed to fetch Nasdaq-100 universe: {e}")

    return list(tickers)

# ==========================================
# 3. TELEGRAM DISPATCHER
# ==========================================
def send_telegram_alert(message: str) -> bool:
    """Sends a formatted markdown message to your Telegram bot."""
    if TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN" or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram token/chat_id not configured. Printing preview:")
        print(f"\n--- [TELEGRAM PREVIEW] ---\n{message}\n--------------------------\n")
        return False
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        return True
    except Exception as e:
        logging.error(f"Failed to send Telegram notification: {e}")
        return False

# ==========================================
# 4. NOTIFICATION BUILDERS
# ==========================================
def alert_buy_signal(ticker: str, entry_price: float, rsi: float, atr: float, tp_target: float, sl_target: float):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S ET")
    message = (
        f"🟢 *[NEW BUY SIGNAL DETECTED]* 🟢\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• **Ticker:** `{ticker}`\n"
        f"• **Target Entry:** `${entry_price:.2f}`\n"
        f"• **Indicators:** RSI `{rsi:.1f}` | ATR `${atr:.2f}`\n"
        f"• **Take Profit Target:** `${tp_target:.2f}`\n"
        f"• **Stop Loss Target:** `${sl_target:.2f}`\n\n"
        f"👉 *Action:* Review chart & execute Buy order in Questrade. *(Auto-logged as OPEN)*\n"
        f"⏱️ `{timestamp}`"
    )
    send_telegram_alert(message)

def alert_take_profit(ticker: str, current_price: float, target_price: float, entry_price: float):
    gain_pct = ((current_price - entry_price) / entry_price) * 100
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S ET")
    message = (
        f"🚀 *[TAKE-PROFIT TARGET REACHED]* 🚀\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• **Ticker:** `{ticker}`\n"
        f"• **Current Price:** `${current_price:.2f}`\n"
        f"• **Target Price:** `${target_price:.2f}`\n"
        f"• **Unrealized Gain:** `+{gain_pct:.2f}%`\n\n"
        f"⚡ *ACTION REQUIRED:* Open Questrade & **SELL MANUALLY** now! *(Status updated to CLOSED)*\n"
        f"⏱️ `{timestamp}`"
    )
    send_telegram_alert(message)

# ==========================================
# 5. QUANT CALCULATIONS & DATA
# ==========================================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    return df

def get_latest_market_data(ticker: str):
    if yf is None:
        return None
    try:
        data = yf.Ticker(ticker).history(period="1y")
        if data.empty or len(data) < 200:
            return None
        return calculate_indicators(data)
    except Exception:
        return None

# ==========================================
# 6. MAIN EXECUTION WORKFLOW
# ==========================================
def main():
    logging.info("Starting Dynamic Quant Sentinel execution run...")
    
    # ------------------------------------------
    # STEP 1: Load ONLY OPEN positions for monitoring
    # ------------------------------------------
    df_open = get_open_positions()
    
    logging.info(f"Loaded {len(df_open)} OPEN positions from {CSV_FILENAME} to check for Take-Profit targets.")
    
    for idx, row in df_open.iterrows():
        ticker = row['ticker']
        tp_target = row['take_profit_target']
        entry_price = row['entry_price']
        
        df = get_latest_market_data(ticker)
        if df is None:
            continue
            
        current_price = df['Close'].iloc[-1]
        
        if current_price >= tp_target and not row.get('alert_sent', False):
            alert_take_profit(ticker, current_price, tp_target, entry_price)
            # Update status to CLOSED in the CSV file
            update_position_status(ticker, 'CLOSED')

    # ------------------------------------------
    # STEP 2: Scan full market universe for Buy signals
    # ------------------------------------------
    logging.info("Fetching full S&P 500 & Nasdaq-100 universe for scanning...")
    universe = get_dynamic_universe()
    
    # Refresh open tickers set to skip them during buy scans
    current_open_df = get_open_positions()
    open_tickers_set = set(current_open_df['ticker'].tolist()) if not current_open_df.empty else set()
    
    logging.info(f"Scanning {len(universe)} tickers (Skipping {len(open_tickers_set)} active open positions)...")
    
    new_rows = []
    for ticker in universe:
        if ticker in open_tickers_set:
            continue
            
        df = get_latest_market_data(ticker)
        if df is None:
            continue

        latest = df.iloc[-1]
        close = latest['Close']
        rsi = latest['RSI']
        atr = latest['ATR']
        sma200 = latest['SMA200']

        # Buy Strategy: Above 200 SMA and RSI oversold (< 35)
        if close > sma200 and rsi < 35:
            tp_target = round(close + (2 * atr), 2)
            sl_target = round(close - (1 * atr), 2)
            
            alert_buy_signal(ticker, close, rsi, atr, tp_target, sl_target)
            
            new_rows.append({
                "ticker": ticker,
                "entry_date": datetime.now().strftime("%Y-%m-%d"),
                "entry_price": round(close, 2),
                "take_profit_target": tp_target,
                "stop_loss_target": sl_target,
                "status": "OPEN",
                "alert_sent": False
            })

    # Append new OPEN buy signals to CSV
    if new_rows:
        append_new_positions(new_rows)
        logging.info(f"Successfully added {len(new_rows)} new OPEN positions to {CSV_FILENAME}.")

    logging.info("Dynamic Quant Sentinel run completed successfully.")

if __name__ == "__main__":
    main()
