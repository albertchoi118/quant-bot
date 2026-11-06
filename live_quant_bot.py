import os
import time
import requests
import pandas as pd
import ta
import finnhub
import alpaca_trade_api as alpaca
from datetime import datetime

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
# 1. UNIVERSE FETCHERS
# ==========================================
def get_sp500_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        return tables[0]['Symbol'].str.replace('.', '-').tolist()
    except Exception as e:
        print(f"Error fetching S&P 500: {e}")
        return ["AAPL", "NVDA", "MSFT", "AMZN"]

def get_nasdaq100_tickers():
    try:
        url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
        tables = pd.read_html(url)
        for table in tables:
            if 'Ticker' in table.columns or 'Symbol' in table.columns:
                col = 'Ticker' if 'Ticker' in table.columns else 'Symbol'
                return table[col].str.replace('.', '-').tolist()
        return ["QQQ", "TSLA", "AMD", "META"]
    except Exception as e:
        print(f"Error fetching Nasdaq-100: {e}")
        return []

def get_penny_stocks():
    # High-volume exchange-listed micro-caps / penny stocks ($0.50 - $5.00)
    return ["SNDL", "MULN", "BIOR", "OCGN", "SOUN", "RIG", "CLOV", "TELL", "CAN"]

# ==========================================
# 2. TELEGRAM ALERT SYSTEM
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
            signals_found += 1
            time.sleep(1)  # Prevent Telegram API rate limits
            
    print(f"Scan complete. Found {signals_found} high-conviction opportunities.")

if __name__ == "__main__":
    main()