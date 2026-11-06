import os
import csv
from pathlib import Path
import requests
import pandas as pd
import ta
import finnhub
import alpaca_trade_api as alpaca
from datetime import datetime
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ==========================================
# CONFIGURATION & ENVIRONMENT VARIABLES
# ==========================================
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Initialize API & Sentiment Clients
api = alpaca.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_BASE_URL, api_version='v2')
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
vader_analyzer = SentimentIntensityAnalyzer()

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
        raise ValueError("Could not find S&P 500 table.")
    return [row.find_all(['td', 'th'])[0].text.strip().replace('.', '-') for row in table.find_all('tr')[1:] if row.find_all(['td', 'th'])]

def get_nasdaq100_tickers():
    url = 'https://en.wikipedia.org/wiki/Nasdaq-100'
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    tables = soup.find_all('table', {'class': 'wikitable'})
    if not tables:
        raise ValueError("Could not find Nasdaq-100 table.")
    return [row.find_all(['td', 'th'])[0].text.strip().replace('.', '-') for row in tables[0].find_all('tr')[1:] if row.find_all(['td', 'th']) and len(row.find_all(['td', 'th'])[0].text.strip()) <= 6]

def get_penny_stocks():
    return ["SNDL", "MULN", "BIOR", "OCGN", "SOUN", "RIG", "CLOV", "TELL", "CAN"]

# ==========================================
# 2. TELEGRAM & BACKTEST LOGGING SYSTEMS
# ==========================================
def send_telegram_alert(message):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}, timeout=5)

def log_signal_to_csv(result):
    file_path = "backtest_signals.csv"
    file_exists = Path(file_path).is_file()
    with open(file_path, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Ticker", "Exchange", "Price", "RSI", "NLP_Score", "Conviction", "Stop_Loss", "Take_Profit"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
            result["ticker"], 
            result["exchange"],
            result["price"], 
            result["rsi"], 
            result["nlp"], 
            result["conviction"], 
            result["stop_loss"], 
            result["take_profit"]
        ])

# ==========================================
# 3. LIVE DATA & ADVANCED VADER SENTIMENT ENGINE
# ==========================================
def fetch_live_bars(ticker):
    try:
        bars = api.get_bars(ticker, alpaca.TimeFrame.Hour, limit=50).df
        return bars if not bars.empty and len(bars) >= 20 else None
    except Exception:
        return None

def analyze_news_sentiment(ticker):
    try:
        today = datetime.now().strftime('%Y-%m-%d')
        news = finnhub_client.company_news(ticker, _from=today, to=today)
        if not news:
            return 0.50  # Neutral baseline if no news today
        
        compound_scores = []
        for item in news[:5]: # Evaluate top 5 recent news items
            headline = item.get('headline', '')
            summary = item.get('summary', '')
            text = f"{headline} {summary}"
            
            if text.strip():
                scores = vader_analyzer.polarity_scores(text)
                compound_scores.append(scores['compound'])
        
        if not compound_scores:
            return 0.50
            
        # Average the VADER compound scores (-1.0 to 1.0)
        avg_compound = sum(compound_scores) / len(compound_scores)
        
        # Map compound score (-1.0 to 1.0) to bot's NLP scale (0.50 to 0.95)
        # Negative news gets dampened to 0.50, positive news scales linearly up to 0.95
        normalized_score = 0.50 + (max(0.0, avg_compound) * 0.45)
        return round(max(0.50, min(0.95, normalized_score)), 2)
    except Exception:
        return 0.50

def run_quant_analysis(ticker):
    bars = fetch_live_bars(ticker)
    if bars is None:
        return None
    close = bars['close']
    current_price = float(close.iloc[-1])
    rsi = float(ta.momentum.RSIIndicator(close=close, window=14).rsi().iloc[-1])
    sma_20 = float(ta.trend.SMAIndicator(close=close, window=20).sma_indicator().iloc[-1])
    sma_50 = float(ta.trend.SMAIndicator(close=close, window=50).sma_indicator().iloc[-1])

    # Core Strategy: Oversold dip in a macro uptrend
    if rsi < 35 and current_price > sma_20 and sma_20 > sma_50:
        nlp_score = analyze_news_sentiment(ticker)
        tech_score = (100 - rsi) / 100.0
        conviction = int(((tech_score * 0.5) + (nlp_score * 0.5)) * 100)
        
        # Must pass high conviction threshold
        if conviction >= 78:
            return {
                "ticker": ticker, "price": current_price, "rsi": round(rsi, 1),
                "nlp": round(nlp_score, 2), "conviction": conviction,
                "stop_loss": round(current_price * 0.96, 2), "take_profit": round(current_price * 1.08, 2)
            }
    return None

# ==========================================
# MAIN RUNNER
# ==========================================
def main():
    print("🚀 Initializing Fast Live Quant Bot Scan with VADER NLP...")
    
    sp500_list = get_sp500_tickers()
    nasdaq_list = get_nasdaq100_tickers()
    penny_list = get_penny_stocks()
    
    sp500_set = set(sp500_list)
    nasdaq_set = set(nasdaq_list)
    penny_set = set(penny_list)
    
    universe = list(sp500_set | nasdaq_set | penny_set)
    print(f"Tracking {len(universe)} unique tickers across S&P 500, Nasdaq-100, and Penny Stocks.")
    
    signals_found = 0
    for ticker in universe:
        result = run_quant_analysis(ticker)
        if result:
            if ticker in penny_set:
                result["exchange"] = "Penny / OTC"
            elif ticker in nasdaq_set:
                result["exchange"] = "NASDAQ-100"
            else:
                result["exchange"] = "S&P 500"
                
            msg = f"🚨 *HIGH-CONVICTION BUY* 🚨\n\n*Ticker:* `{result['ticker']}`\n*Exchange:* `{result['exchange']}`\n*Price:* `${result['price']:.2f}`\n*NLP Score:* `{result['nlp']}`\n*Conviction:* `{result['conviction']}%`"
            send_telegram_alert(msg)
            log_signal_to_csv(result)
            signals_found += 1
        
    print(f"Scan complete. Found {signals_found} signals.")

if __name__ == "__main__":
    main()
