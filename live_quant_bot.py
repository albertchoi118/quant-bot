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
        log(f"⚠️ Error scraping S&P 500: {e}")

    # 2. Scrape Nasdaq-100 Tickers
    try:
        ndx_url = "https://en.wikipedia.org/wiki/Nasdaq-100"
        resp = requests.get(ndx_url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, 'html.parser')
        table = soup.find('table', {'id': 'constituents'})
        if table:
            for row in table.find_all('tr')[1:]:
                ticker = row.find_all('td')[1].text.strip().replace('.', '-')
                if ticker and ticker not in ['Ticker', 'Symbol']:
                    tickers.add(ticker)
    except Exception as e:
        log(f"⚠️ Error scraping Nasdaq-100: {e}")

    if not tickers:
        log("⚠️ Scraping failed. Utilizing core tech fallback list.")
        tickers = {"AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "COST", "AMD"}
        
    return sorted(list(tickers))

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Standard 14-period Relative Strength Index."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# ==========================================
# MAIN QUANT BOT EXECUTION ENGINE
# ==========================================
def run_quant_scanner():
    if not is_market_open():
        log("⏸️ Market is currently CLOSED. Proceeding with scan using latest available daily bars...")
    else:
        log("🚀 Market is OPEN. Initializing High-Speed Quant Scan...")
    
    # Safe check: Load active/open tickers from CSV if file exists and is not empty
    open_tickers = set()
    if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
        try:
            existing_df = pd.read_csv(CSV_FILE)
            if 'Ticker' in existing_df.columns and 'Status' in existing_df.columns:
                open_mask = existing_df['Status'].astype(str).str.upper().str.contains("OPEN|ACTIVE")
                open_tickers = set(existing_df.loc[open_mask, 'Ticker'].unique())
            log(f"📋 Loaded {len(open_tickers)} active OPEN positions from {CSV_FILE} for deduplication.")
        except Exception as e:
            log(f"⚠️ Error reading existing CSV for deduplication: {e}")

    universe = get_clean_universe()
    log(f"📊 Tracking {len(universe)} unique tickers. Starting scan...")

    signals_found = []

    for ticker in universe:
        try:
            # Deduplication Check: Skip if already active in CSV
            if ticker in open_tickers:
                log(f"⏸️ {ticker} already has an active OPEN position. Skipping.")
                continue

            df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            
            if df.empty or len(df) < 200:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df['RSI'] = calculate_rsi(df['Close'], RSI_PERIOD)
            
            # Calculate True Range & ATR
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            df['ATR'] = ranges.max(axis=1).rolling(window=ATR_PERIOD).mean()

            df['SMA200'] = df['Close'].rolling(window=200).mean()

            latest = df.iloc[-1]
            price = float(latest['Close'])
            rsi = float(latest['RSI'])
            sma200 = float(latest['SMA200'])
            atr = float(latest['ATR'])

            # Fallback ATR if NaN
            if np.isnan(atr) or atr <= 0:
                atr = price * 0.02

            rsi_ok = rsi < RSI_BUY_THRESHOLD
            trend_ok = price > sma200  

            if rsi_ok and trend_ok:
                # Dynamic ATR-based risk management
                sl_price = round(price - (ATR_SL_MULTIPLIER * atr), 2)
                tp_price = round(price + (ATR_TP_MULTIPLIER * atr), 2)
                timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                log(f"🎯 BUY SIGNAL: {ticker} @ ${price:.2f} | RSI: {rsi:.1f} | ATR: ${atr:.2f} | TP: ${tp_price} | SL: ${sl_price}")

                alert_msg = (
                    f"🚨 *DYNAMIC QUANT BUY ALERT* 🚨\n\n"
                    f"📌 *Ticker:* `{ticker}`\n"
                    f"💵 *Entry Price:* `${price:.2f}`\n"
                    f"📈 *Take Profit:* `${tp_price:.2f}` (+3 ATR)\n"
                    f"🛑 *Stop Loss:* `${sl_price:.2f}` (-1.5 ATR)\n"
                    f"📊 *RSI (14):* `{rsi:.1f}`\n"
                    f"🌐 *Macro Trend:* `Above 200-SMA (${sma200:.2f})`\n\n"
                    f"⚡ *Action:* Set up bracket order on Questrade Edge Web!"
                )
                send_telegram_alert(alert_msg)

                signals_found.append({
                    "Timestamp": timestamp_str,
                    "Ticker": ticker,
                    "Exchange": "S&P/Nasdaq",
                    "Price": round(price, 2),
                    "RSI": round(rsi, 2),
                    "SMA200": round(sma200, 2),
                    "Stop_Loss": sl_price,
                    "Take_Profit": tp_price,
                    "Signal": "BUY",
                    "Status": "OPEN",
                    "Exit_Price": np.nan,
                    "Exit_Date": np.nan
                })
            else:
                log(f"🔍 {ticker} | Price: ${price:.2f} | RSI: {rsi:.1f} (Req < {RSI_BUY_THRESHOLD}: {rsi_ok}) | Trend OK: {trend_ok} -> NO BUY")

        except Exception as e:
            log(f"⚠️ Error processing {ticker}: {e}")
            continue

    if signals_found:
        new_df = pd.DataFrame(signals_found)
        if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
            try:
                existing_df = pd.read_csv(CSV_FILE)
                updated_df = pd.concat([existing_df, new_df], ignore_index=True)
                updated_df.to_csv(CSV_FILE, index=False)
            except Exception:
                new_df.to_csv(CSV_FILE, index=False)
        else:
            new_df.to_csv(CSV_FILE, index=False)
        log(f"✅ Successfully appended {len(signals_found)} new signals to {CSV_FILE}.")
    else:
        log("🏁 Scan complete. Zero new unique signals triggered during this cycle.")

if __name__ == "__main__":
    run_quant_scanner()
