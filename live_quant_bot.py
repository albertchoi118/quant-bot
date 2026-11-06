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

# Strategy Parameters (Calibrated for High-Probability Mean Reversion)
RSI_PERIOD = 14
RSI_BUY_THRESHOLD = 40.0   # Bumping to 40 captures front-run index dips
TAKE_PROFIT_PCT = 0.05     # +5% Take Profit target
STOP_LOSS_PCT = 0.03       # -3% Stop Loss target


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
    
    # Weekday check (0=Mon, 4=Fri, 5=Sat, 6=Sun)
    if now.weekday() >= 5:
        return False
        
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    
    return market_open <= now <= market_close

def send_telegram_alert(message: str):
    """Sends immediate notification to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log("⚠️ Telegram credentials missing. Skipping alert dispatch.")
        return
        
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        res = requests.post(url, json=payload, timeout=10)
        if res.status_code == 200:
            log("📲 Telegram alert dispatches successfully.")
        else:
            log(f"⚠️ Telegram alert failed: {res.text}")
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
        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text.strip().replace('.', '-')
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
            for row in table.findAll('tr')[1:]:
                ticker = row.findAll('td')[1].text.strip().replace('.', '-')
                if ticker and ticker not in ['Ticker', 'Symbol']:
                    tickers.add(ticker)
    except Exception as e:
        log(f"⚠️ Error scraping Nasdaq-100: {e}")

    # Fallback default if scraping fails
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
        log("⏸️ Market is CLOSED. Skipping scan.")
        sys.exit(0)

    log("🚀 Market is OPEN. Initializing High-Speed Quant Scan...")
    universe = get_clean_universe()
    log(f"📊 Tracking {len(universe)} unique tickers. Starting scan...")

    signals_found = []

    for ticker in universe:
        try:
            # Download recent 1-year daily bars for robust moving averages
            df = yf.download(ticker, period="1y", interval="1d", progress=False, auto_adjust=True)
            
            if df.empty or len(df) < 200:
                continue

            # Flatten MultiIndex columns if returned by yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Compute Key Technical Indicators
            df['RSI'] = calculate_rsi(df['Close'], RSI_PERIOD)
            df['SMA20'] = df['Close'].rolling(window=20).mean()
            df['SMA50'] = df['Close'].rolling(window=50).mean()
            df['SMA200'] = df['Close'].rolling(window=200).mean()

            latest = df.iloc[-1]
            price = float(latest['Close'])
            rsi = float(latest['RSI'])
            sma20 = float(latest['SMA20'])
            sma50 = float(latest['SMA50'])
            sma200 = float(latest['SMA200'])

            # Quant Entry Conditions:
            # 1. RSI oversold dip (RSI < 40)
            # 2. Macro Trend intact (Price > 200-day SMA)
            rsi_ok = rsi < RSI_BUY_THRESHOLD
            trend_ok = price > sma200  

            if rsi_ok and trend_ok:
                tp_price = round(price * (1 + TAKE_PROFIT_PCT), 2)
                sl_price = round(price * (1 - STOP_LOSS_PCT), 2)
                date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                log(f"🎯 BUY SIGNAL GENERATED: {ticker} @ ${price:.2f} | RSI: {rsi:.1f} | TP: ${tp_price} | SL: ${sl_price}")

                # Format Telegram Alert
                alert_msg = (
                    f"🚨 *QUANT BUY ALERT* 🚨\n\n"
                    f"📌 *Ticker:* `{ticker}`\n"
                    f"💵 *Entry Price:* `${price:.2f}`\n"
                    f"📈 *Take Profit:* `${tp_price:.2f}` (+5%)\n"
                    f"🛑 *Stop Loss:* `${sl_price:.2f}` (-3%)\n"
                    f"📊 *RSI (14):* `{rsi:.1f}`\n"
                    f"🌐 *Macro Trend:* `Above 200-SMA (${sma200:.2f})`\n\n"
                    f"⚡ *Action:* Set up bracket order on Questrade Edge Web!"
                )
                send_telegram_alert(alert_msg)

                # Append signal for CSV logging
                signals_found.append({
                    "Date": date_str,
                    "Ticker": ticker,
                    "Signal": "BUY",
                    "Entry_Price": round(price, 2),
                    "Take_Profit": tp_price,
                    "Stop_Loss": sl_price,
                    "RSI": round(rsi, 2),
                    "SMA200": round(sma200, 2),
                    "Status": "OPEN",
                    "Exit_Price": np.nan,
                    "Exit_Date": np.nan
                })
            else:
                log(f"🔍 {ticker} | Price: ${price:.2f} | RSI: {rsi:.1f} (Req < 40: {rsi_ok}) | SMA200: ${sma200:.2f} | Trend OK: {trend_ok} -> NO BUY")

        except Exception as e:
            log(f"⚠️ Error processing {ticker}: {e}")
            continue

    # Update CSV Log File if signals were found
    if signals_found:
        new_df = pd.DataFrame(signals_found)
        if os.path.exists(CSV_FILE):
            existing_df = pd.read_csv(CSV_FILE)
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
            updated_df.to_csv(CSV_FILE, index=False)
        else:
            new_df.to_csv(CSV_FILE, index=False)
        log(f"✅ Successfully appended {len(signals_found)} new signals to {CSV_FILE}.")
    else:
        log("🏁 Scan complete. Zero signals triggered during this cycle.")

if __name__ == "__main__":
    run_quant_scanner()
