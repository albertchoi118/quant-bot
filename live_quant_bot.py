import os
import time
import csv
from pathlib import Path
import requests
import pandas as pd
import ta
import finnhub
import alpaca_trade_api as alpaca
from datetime import datetime
from bs4 import BeautifulSoup

# ==========================================
# CONFIGURATION & ENVIRONMENT VARIABLES
# ==========================================
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Validate critical keys
if not ALPACA_API_KEY or not FINNHUB_API_KEY or not TELEGRAM_BOT_TOKEN:
    print("⚠️ Warning: Missing one or more API keys in environment variables.")

# Initialize API Clients
api = alpaca.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# ==========================================
# 1. UNIVERSE FETCHERS (Direct BeautifulSoup)
# ==========================================
def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    if not table:
        raise ValueError("Could not find S&P 500 constituents table on Wikipedia.")
        
    tickers = []
    for row in table.find_all('tr')[1:]:
        cols = row.find_all(['td', 'th'])
        if cols:
            symbol = cols[0].text.strip().replace('.', '-')
            tickers.append(symbol)
    return tickers

def get_nasdaq100_tickers():
    url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    tables = soup.find_all('table', {'class': 'wikitable'})
    if not tables:
        raise ValueError("Could not find Nasdaq-100 table on Wikipedia.")
        
    target_table = tables[0]
    tickers = []
    for row in target_table.find_all('tr')[1:]:
        cols = row.find_all(['td', 'th'])
        if cols:
            symbol = cols[0].text.strip().replace('.', '-')
            if symbol and len(symbol) <= 6:
                tickers.append(symbol)
    return tickers

def get_penny_stocks():
    # High-volume exchange-listed micro-caps / penny stocks
    return ["SNDL", "MULN", "BIOR", "OCGN", "SOUN", "RIG", "CLOV", "TELL", "CAN"]

# ==========================================
# 2. TELEGRAM & BACKTEST LOGGING SYSTEMS
# ==========================================
def send_telegram_alert(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram tokens not configured. Skipping alert dispatch.")
        return
    
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"Failed to send Telegram alert: {e}")

def log_signal_to_csv(result):
    file_path = "backtest_signals.csv"
    file_exists = Path(file_path).is_file()
    
    with open(file_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Ticker", "Price", "RSI", "NLP_Score", "Conviction", "Stop_Loss", "Take_Profit"])
        
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            result["ticker"],
            result["price"],
            result["rsi"],
            result["nlp"],
            result["conviction"],
            result["stop_loss"],
            result["take_profit"]
        ])

# ==========================================
# 3. LIVE DATA & QUANT ANALYSIS ENGINE
# ==========================================
def fetch_live_bars(ticker):
    """Fetches real-time hourly bars from Alpaca API."""
    try:
        bars = api.get_bars(ticker, alpaca.TimeFrame.Hour, limit=50).df
        if bars.empty or len(bars) < 20:
            return None
        return bars
    except Exception:
        return None

def analyze_news_sentiment(ticker):
    """Fetches breaking news from Finnhub and returns a crude sentiment score."""
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        news = finnhub_client.company_news(ticker, _from=today, to=today)
        if not news:
            return 0.50  # Neutral baseline
        
        bullish_keywords = ["upgrade", "approval", "earnings", "contract", "record", "growth", "fda", "patent"]
        score = 0.50
        for item in news[:5]:
            headline = item.get('headline', '').lower()
            if any(word in headline for word in bullish_keywords):
                score += 0.10
        return min(0.95, score)
    except Exception:
        return 0.50

def run_quant_analysis(ticker):
    bars = fetch_live_bars(ticker)
    if bars is None:
        return None

    close = bars['close']
    current_price = float(close.iloc[-1])
    
    # Calculate Technical Indicators
    rsi = float(ta.momentum.RSIIndicator(close=close, window=14).rsi().iloc[-1])
    sma_20 = float(ta.trend.SMAIndicator(close=close, window=20).sma_indicator().iloc[-1])
    sma_50 = float(ta.trend.SMAIndicator(close=close, window=50).sma_indicator().iloc[-1])

    # Filter Criteria: Oversold (RSI < 35) + MA Crossover Reversal
    if rsi < 35 and current_price > sma_20 and sma_20 > sma_50:
        nlp_score = analyze_news_sentiment(ticker)
        
        # Calculate Conviction Score (50% Technical, 50% News NLP)
        tech_score = (100 - rsi) / 100.0
        conviction = int(((tech_score * 0.5) + (nlp_score * 0.5)) * 100)

        # High-Conviction Threshold (> 78%)
        if conviction >= 78:
            stop_loss = round(current_price * 0.96, 2)
            take_profit = round(current_price * 1.08, 2)
            
            return {
                "ticker": ticker,
                "price": current_price,
                "rsi": round(rsi, 1),
                "nlp": round(nlp_score, 2),
                "conviction": conviction,
                "stop_loss": stop_loss,
                "take_profit": take_profit
            }
    return None

# ==========================================
# MAIN RUNNER (Single-Execution Mode for GitHub Actions)
# ==========================================
def main():
    print("🚀 Initializing Live Quant Bot Scan...")
    universe = list(set(get_sp500_tickers() + get_nasdaq100_tickers() + get_penny_stocks()))
    print(f"Tracking {len(universe)} real-time stock tickers.")
    
    signals_found = 0
    for ticker in universe:
        result = run_quant_analysis(ticker)
        if result:
            msg = (
                f"🚨 *HIGH-CONVICTION BUY SIGNAL* 🚨\n\n"
                f"*Ticker:* `{result['ticker']}`\n"
                f"*Current Price:* `${result['price']:.2f}`\n"
                f"*Conviction Score:* `{result['conviction']}%`\n"
                f"*RSI (14):* `{result['rsi']}`\n"
                f"*NLP News Score:* `+{result['nlp']}`\n\n"
                f"🎯 *Take Profit Target:* `${result['take_profit']}`\n"
                f"🛡️ *Stop Loss Floor:* `${result['stop_loss']}`\n"
            )
            print(f"Signal Detected: {result['ticker']} @ ${result['price']}")
            send_telegram_alert(msg)
            log_signal_to_csv(result)
            signals_found += 1
            
        # Respect Alpaca free-tier rate limits (max ~200 requests/min)
        time.sleep(0.5)
        
    print(f"Scan complete. Found {signals_found} high-conviction opportunities.")

if __name__ == "__main__":
    main()
