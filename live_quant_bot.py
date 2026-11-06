import os
import sys
import time
import datetime
import pytz
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CSV_FILE = "backtest_signals.csv"

# Strategy Parameters (Calibrated Sweet Spot & Volatility Scaling)
RSI_PERIOD = 14
RSI_BUY_THRESHOLD = 35.0   # Balanced institutional dip-buying threshold
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 1.5    # Stop loss distance in ATR units
ATR_TP_MULTIPLIER = 3.0    # Take profit distance in ATR units (2:1 Reward/Risk)


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def log(msg: str):
    """Prints diagnostic logs with timestamp."""
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_str}] {msg}")

def is_market_open() -> bool:
    """Checks if US stock market is currently open (9:30 AM - 4:00 PM Eastern ET)."""
    tz = pytz.timezone("America/New_York")
    now = datetime.datetime.now(tz)
    
    if now.weekday() >= 5:
        return False
        
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now <= market_close

def send_telegram_alert(message: str):
    """Sends immediate notification to Telegram with token sanitization."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log("⚠️ Telegram credentials missing in environment variables.")
        return
        
    clean_token = TELEGRAM_BOT_TOKEN.strip().strip("'").strip('"')
    clean_chat_id = TELEGRAM_CHAT_ID.strip().strip("'").strip('"')

    if clean_token.startswith("bot"):
        clean_token = clean_token[3:]
        
    url = f"https://api.telegram.org/bot{clean_token}/sendMessage"
    payload = {
        "chat_id": clean_chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code == 200:
            log("📲 Telegram alert dispatched successfully.")
        else:
            log(f"⚠️ Telegram alert failed ({res.status_code}): {res.text}")
    except Exception as e:
        log(f"❌ Error sending Telegram alert: {e}")

def get_clean_universe() -> list:
    """Scrapes clean S&P 500 and Nasdaq-100 ticker lists."""
    tickers = set()
    headers = {"User-Agent": "Mozilla/5.0"}
    
    # 1. Scrape S&P 500 Tickers
    try:
        sp_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        resp = requests.get(sp_url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})
        for row in table.find_all('tr')[1:]:
            ticker = row.find_all('td')[0].text.strip().replace('.', '-')
            if ticker:
                tickers.add(ticker)
    except Exception as e:
