#!/usr/bin/env python3
"""
Live Quant Bot - Combined Signal Scanner & Position Monitor
===========================================================
- Buy Signals: Scans ticker universe (e.g., S&P 500 / Nasdaq-100) using RSI & ATR.
- Sell Signals: Tracks active positions and triggers a Take-Profit alert when target is hit.
- Sends cleanly formatted Telegram notifications with clear visual distinction.
"""

import os
import sys
import logging
from datetime import datetime
import pandas as pd
import numpy as np

# Optional: Replace with your preferred market data provider (e.g., yfinance, alpaca, finnhub)
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

# Sample Universe & Active Positions Tracking (Replace with your database/CSV load)
WATCHLIST = ["AAPL", "MSFT", "NVDA", "AMD", "AMZN", "GOOGL", "META", "TSLA"]

# Active positions you currently hold and want to monitor for upside exits
# In production, load this from a database or local state file (e.g., active_positions.json)
ACTIVE_POSITIONS = [
    {
        "ticker": "AAPL",
        "entry_price": 220.50,
        "take_profit_target": 235.00,
        "stop_loss_target": 212.00,
        "alert_sent": False
    },
    {
        "ticker": "NVDA",
        "entry_price": 125.00,
        "take_profit_target": 140.00,
        "stop_loss_target": 118.00,
        "alert_sent": False
    }
]

# ==========================================
# 1. TELEGRAM DISPATCHER
# ==========================================
def send_telegram_alert(message: str) -> bool:
    """Sends a formatted markdown message to your Telegram bot."""
    if TELEGRAM_BOT_TOKEN == "YOUR_TELEGRAM_BOT_TOKEN" or not TELEGRAM_CHAT_ID:
        logging.warning("Telegram token/chat_id not configured. Message skipped.")
        print(f"\n--- [TELEGRAM OUTPUT PREVIEW] ---\n{message}\n---------------------------------\n")
        return False
    
    import requests
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    
    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        logging.info("Telegram notification sent successfully.")
        return True
    except Exception as e:
        logging.error(f"Failed to send Telegram notification: {e}")
        return False

# ==========================================
# 2. DISTINCT NOTIFICATION BUILDERS
# ==========================================
def alert_buy_signal(ticker: str, entry_price: float, rsi: float, atr: float, tp_target: float, sl_target: float):
    """Formats and dispatches a BUY opportunity signal."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S ET")
    message = (
        f"🟢 *[NEW BUY SIGNAL DETECTED]* 🟢\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• **Ticker:** `{ticker}`\n"
        f"• **Target Entry:** `${entry_price:.2f}`\n"
        f"• **Indicators:** RSI `{rsi:.1f}` | ATR `${atr:.2f}`\n"
        f"• **Take Profit Target:** `${tp_target:.2f}`\n"
        f"• **Suggested Stop Loss:** `${sl_target:.2f}`\n\n"
        f"👉 *Action:* Review chart setup and execute Buy order in Questrade.\n"
        f"⏱️ `{timestamp}`"
    )
    send_telegram_alert(message)

def alert_take_profit(ticker: str, current_price: float, target_price: float, entry_price: float):
    """Formats and dispatches a TAKE-PROFIT exit alert."""
    gain_pct = ((current_price - entry_price) / entry_price) * 100
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S ET")
    message = (
        f"🚀 *[TAKE-PROFIT TARGET REACHED]* 🚀\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"• **Ticker:** `{ticker}`\n"
        f"• **Current Price:** `${current_price:.2f}`\n"
        f"• **Target Price:** `${target_price:.2f}`\n"
        f"• **Unrealized Gain:** `+{gain_pct:.2f}%`\n\n"
        f"⚡ *ACTION REQUIRED:* Open Questrade Edge and **SELL MANUALLY** now!\n"
        f"⏱️ `{timestamp}`"
    )
    send_telegram_alert(message)

# ==========================================
# 3. QUANT LOGIC & MARKET DATA
# ==========================================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates RSI, ATR, and 200-day SMA."""
    # RSI 14
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # ATR 14
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=14).mean()

    # 200 SMA
    df['SMA200'] = df['Close'].rolling(window=200).mean()
    
    return df

def get_latest_market_data(ticker: str):
    """Fetches historical daily data for analysis using yfinance."""
    if yf is None:
        logging.error("yfinance package not installed. Install with `pip install yfinance`.")
        return None
    try:
        data = yf.Ticker(ticker).history(period="1y")
        if data.empty or len(data) < 200:
            return None
        return calculate_indicators(data)
    except Exception as e:
        logging.error(f"Error fetching market data for {ticker}: {e}")
        return None

# ==========================================
# 4. MONITORING & SCANNING LOOPS
# ==========================================
def check_take_profit_exits(active_positions: list):
    """Scans active holdings to notify when Take-Profit targets are hit."""
    logging.info("Checking active positions for Take-Profit targets...")
    for pos in active_positions:
        ticker = pos['ticker']
        tp_target = pos['take_profit_target']
        entry_price = pos['entry_price']
        
        df = get_latest_market_data(ticker)
        if df is None:
            continue
            
        current_price = df['Close'].iloc[-1]
        
        # Trigger alert if price meets/exceeds upside target and hasn't already alerted
        if current_price >= tp_target and not pos.get('alert_sent', False):
            alert_take_profit(ticker, current_price, tp_target, entry_price)
            pos['alert_sent'] = True
            
        elif current_price < tp_target and pos.get('alert_sent', False):
            # Reset toggle if price temporarily dips back below target
            pos['alert_sent'] = False

def scan_for_buy_signals(watchlist: list, active_tickers: set):
    """Scans watchlist tickers for mean-reversion buy setups."""
    logging.info("Scanning watchlist for new Buy Signals...")
    for ticker in watchlist:
        # Deduplication: skip scanning if we already own this position
        if ticker in active_tickers:
            continue
            
        df = get_latest_market_data(ticker)
        if df is None:
            continue

        latest = df.iloc[-1]
        close = latest['Close']
        rsi = latest['RSI']
        atr = latest['ATR']
        sma200 = latest['SMA200']

        # Mean Reversion Setup: Trend above 200 SMA but oversold RSI (< 35)
        if close > sma200 and rsi < 35:
            tp_target = close + (2 * atr)  # 2:1 Reward-to-Risk ratio target
            sl_target = close - (1 * atr)  # Stop loss baseline
            
            alert_buy_signal(ticker, close, rsi, atr, tp_target, sl_target)

# ==========================================
# 5. MAIN EXECUTION ENTRY POINT
# ==========================================
def main():
    logging.info("Starting Quant Bot execution run...")
    
    # 1. Track currently open tickers for deduplication
    active_tickers = {pos['ticker'] for pos in ACTIVE_POSITIONS}
    
    # 2. First check open positions for Take-Profit exits
    check_take_profit_exits(ACTIVE_POSITIONS)
    
    # 3. Next scan watchlist for new entry setups
    scan_for_buy_signals(WATCHLIST, active_tickers)
    
    logging.info("Quant Bot run completed successfully.")

if __name__ == "__main__":
    main()
