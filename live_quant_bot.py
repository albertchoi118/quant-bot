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

# Strategy Parameters
RSI_PERIOD = 14
RSI_BUY_THRESHOLD = 35.0   
ATR_PERIOD = 14
ATR_SL_MULTIPLIER = 1.5    
ATR_TP_MULTIPLIER = 3.0    


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
# POSITION EXIT MONITOR
# ==========================================
def check_open_position_exits():
    """Checks active OPEN positions against current market price action for TP/SL hits and records exit prices."""
    if not os.path.exists(CSV_FILE) or os.path.getsize(CSV_FILE) == 0:
        return
        
    try:
        df = pd.read_csv(CSV_FILE)
        if 'Status' not in df.columns or df.empty:
            return
            
        updated = False
        today_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for idx, row in df.iterrows():
            if str(row['Status']).upper() == 'OPEN':
                ticker = row['Ticker']
                entry_price = float(row['Price'])
                sl = float(row['Stop_Loss'])
                tp = float(row['Take_Profit'])
                
                hist = yf.download(ticker, period="3d", interval="1d", progress=False, auto_adjust=True)
                if hist.empty:
                    continue
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.get_level_values(0)
                    
                latest = hist.iloc[-1]
                day_high = float(latest['High'])
                day_low = float(latest['Low'])
                current_close = float(latest['Close'])
                
                # Check Take Profit Hit
                if day_high >= tp:
                    exit_price = round(max(tp, current_close), 2)
                    df.at[idx, 'Status'] = 'CLOSED (TP)'
                    df.at[idx, 'Exit_Price'] = exit_price
                    df.at[idx, 'Exit_Date'] = today_str
                    updated = True
                    
                    log(f"🎉 TAKE PROFIT HIT: {ticker} (Entry: ${entry_price:.2f}) -> Exit Price: ${exit_price:.2f} (Target TP: ${tp:.2f})")
                    
                    alert_msg = (
                        f"🎯 *TAKE PROFIT TARGET REACHED* 🎯\n\n"
                        f"📌 *Ticker:* `{ticker}`\n"
                        f"💵 *Entry Price:* `${entry_price:.2f}`\n"
                        f"📈 *Exit Price:* `${exit_price:.2f}` (Target: `${tp:.2f}`)\n"
                        f"⚡ *Action:* Close position on Questrade Edge Web!"
                    )
                    send_telegram_alert(alert_msg)
                    
                # Check Stop Loss Hit
                elif day_low <= sl:
                    exit_price = round(min(sl, current_close), 2)
                    df.at[idx, 'Status'] = 'CLOSED (SL)'
                    df.at[idx, 'Exit_Price'] = exit_price
                    df.at[idx, 'Exit_Date'] = today_str
                    updated = True
                    
                    log(f"🛑 STOP LOSS HIT: {ticker} (Entry: ${entry_price:.2f}) -> Exit Price: ${exit_price:.2f} (Target SL: ${sl:.2f})")
                    
                    alert_msg = (
                        f"🛑 *STOP LOSS TRIGGERED* 🛑\n\n"
                        f"📌 *Ticker:* `{ticker}`\n"
                        f"💵 *Entry Price:* `${entry_price:.2f}`\n"
                        f"📉 *Exit Price:* `${exit_price:.2f}` (Target SL: `${sl:.2f}`)\n"
                        f"⚡ *Action:* Cut losses on Questrade Edge Web!"
                    )
                    send_telegram_alert(alert_msg)
                    
        if updated:
            df.to_csv(CSV_FILE, index=False)
            log("📋 CSV successfully updated with Exit_Price and Exit_Date for closed positions.")
            
    except Exception as e:
        log(f"⚠️ Error checking open position exits: {e}")


# ==========================================
# MAIN QUANT BOT EXECUTION ENGINE
# ==========================================
def run_quant_scanner():
    if not is_market_open():
        log("⏸️ Market is currently CLOSED. Proceeding with scan using latest available daily bars...")
    else:
        log("🚀 Market is OPEN. Initializing High-Speed Quant Scan...")
    
    # 1. Monitor active open positions and log exit prices if targets hit
    check_open_position_exits()

    # 2. Reload active/open tickers from CSV after exit check
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

            if np.isnan(atr) or atr <= 0:
                atr = price * 0.02

            rsi_ok = rsi < RSI_BUY_THRESHOLD
            trend_ok = price > sma200  

            if rsi_ok and trend_ok:
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
