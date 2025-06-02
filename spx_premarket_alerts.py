import requests
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
import csv
import openai
from fredapi import Fred
import pytz
import re
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import yfinance as yf

# Load environment variables
load_dotenv()
EMAIL = os.getenv("MC_EMAIL")
PASSWORD = os.getenv("MC_PASSWORD")
EMAIL_TO = os.getenv("EMAIL_TO")

# Load API Keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
finnhub_api_key = os.getenv("FINNHUB_API_KEY")
marketaux_api_key = os.getenv("MARKETAUX_API_KEY")
fred_api_key = os.getenv("FRED_API_KEY")
fred = Fred(api_key=fred_api_key)

DOWNLOAD_DIR = os.path.join(os.getcwd(), "data")
if not os.path.exists(DOWNLOAD_DIR):
    os.makedirs(DOWNLOAD_DIR)


def get_implied_move_yfinance(ticker="SPY", sentiment_score=None):
    try:
        stock = yf.Ticker(ticker)
        spot_price = stock.history(period="1d")["Close"].iloc[-1]
        spot_strike = round(spot_price / 5) * 5
        expirations = stock.options
        if not expirations:
            print("⚠️ No options data available.")
            return None
        expiration = expirations[0]
        opt_chain = stock.option_chain(expiration)
        calls = opt_chain.calls
        puts = opt_chain.puts
        call_row = calls.iloc[(calls['strike'] - spot_price).abs().argsort()[:1]]
        put_row = puts.iloc[(puts['strike'] - spot_price).abs().argsort()[:1]]
        call_ask = float(call_row["ask"].values[0])
        put_ask = float(put_row["ask"].values[0])
        straddle_cost = call_ask + put_ask
        implied_move_pct = (straddle_cost / spot_price) * 100

        if sentiment_score is not None:
            sign = '+' if sentiment_score >= 0 else '-'
            return f"{sign}{round(implied_move_pct, 2)}%"
        else:
            return f"±{round(implied_move_pct, 2)}%"
    except Exception as e:
        print("❌ Error fetching from yfinance:", e)
        return None


def classify_headlines_openai_weighted(headlines):
    prompt = "\n".join([f"{i+1}. {h}" for i, h in enumerate(headlines)])
    system = (
        "You are a financial market analyst. You are given a list of market news headlines. "
        "For each headline, assess the potential impact on the U.S. stock market (specifically S&P 500) "
        "and assign a sentiment score between -5 and +5:\n\n"
        "-5 = Extremely bearish (very negative impact)\n"
        "-3 = Bearish\n"
        " 0 = Neutral or unclear\n"
        "+3 = Bullish\n"
        "+5 = Extremely bullish (very positive impact)\n\n"
        "Only provide the sentiment score for each headline in the format:\n"
        "1. +3\n"
        "2. -2\n"
        "3.  0\n\n"
        "Do not explain your reasoning or repeat the headlines."
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )
        output = response.choices[0].message.content.strip()
        print("🧠 GPT Output:\n", output)
        lines = output.splitlines()
        scores = []
        for line in lines:
            match = re.search(r"[-+]?[0-9]+", line)
            scores.append(int(match.group()) if match else 0)
        return scores
    except Exception as e:
        print("❌ Weighted classification failed:", e)
        return [0] * len(headlines)

def is_market_relevant(text):
    keywords = ["fed", "tariff", "rate", "inflation", "yields", "bond", "treasury", "earnings", "revenue",
                "stocks", "markets", "recession", "jobless", "cpi", "ppi", "gdp", "volatility"]
    return any(k in text.lower() for k in keywords)

def scrape_headlines(url, selector, base_url=""):
    headlines = []
    try:
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        soup = BeautifulSoup(res.text, 'html.parser')
        for el in soup.select(selector)[:10]:
            text = el.get_text(strip=True)
            link = el.get("href", "")
            if text and is_market_relevant(text):
                full_link = f"{base_url}{link}" if link.startswith("/") else link
                headlines.append(f"{text} - {full_link}")
    except Exception as e:
        print(f"⚠️ Error scraping {url}:", e)
    return headlines

def get_all_market_news():
    headlines_raw = []

    # CNBC (weight = 1.5)
    headlines_cnbc = scrape_headlines("https://www.cnbc.com/world/?region=world", "a.Card-title")
    headlines_raw += [{"source": "CNBC", "headline": h} for h in headlines_cnbc]

    # Finnhub
    try:
        res = requests.get(f"https://finnhub.io/api/v1/news?category=general&token={finnhub_api_key}")
        data = res.json()
        for item in data[:10]:
            if is_market_relevant(item.get("headline", "")):
                headlines_raw.append({"source": "Finnhub", "headline": f"{item['headline']} - {item['url']}"})
    except Exception as e:
        print("❌ Finnhub news fetch failed:", e)

    # Marketaux
    try:
        res = requests.get(
            f"https://api.marketaux.com/v1/news/all?symbols=SPY&filter_entities=true&language=en&api_token={marketaux_api_key}"
        ).json()
        for article in res.get("data", [])[:10]:
            if is_market_relevant(article.get("title", "")):
                headlines_raw.append({"source": "Marketaux", "headline": f"{article['title']} - {article['url']}"})
    except Exception as e:
        print("❌ Marketaux news fetch failed:", e)

    # Score only the headlines
    headlines_only = [item["headline"] for item in headlines_raw]
    scores = classify_headlines_openai_weighted(headlines_only)

    # Attach scores directly
    for i, score in enumerate(scores):
        headlines_raw[i]["score"] = score

    # Return as list of (src, score, headline)
    final_news = [(item["source"], item["score"], item["headline"]) for item in headlines_raw]

    # Optional: print for debug
    print("\n🧠 Final Scored Headlines:")
    for src, score, headline in final_news:
        print(f"[{src}] ({score:+}) → {headline}")

    return final_news


def get_price_from_investing(url, retries=2, delay=3):
    for attempt in range(retries):
        try:
            print(f"🌐 Attempt {attempt+1} to fetch price from: {url}")
            options = Options()
            options.add_argument("--headless=new")
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            driver = webdriver.Chrome(options=options)

            driver.get(url)
            time.sleep(4)  # Let JS load

            price_element = driver.find_element(By.CSS_SELECTOR, '[data-test="instrument-price-last"]')
            price_text = price_element.text.replace(",", "")
            price = float(price_text)

            driver.quit()
            return price
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed: {e}")
            if attempt == retries - 1:
                print(f"❌ Failed to fetch price from {url} after {retries} attempts.")
            time.sleep(delay)
            try:
                driver.quit()
            except:
                pass
    return None

def get_price_finnhub(symbol):
    try:
        url = f"https://finnhub.io/api/v1/quote?symbol={symbol}&token={finnhub_api_key}"
        res = requests.get(url).json()
        return res.get("c")  # 'c' is the current price
    except Exception as e:
        print(f"❌ Finnhub fallback failed for {symbol}: {e}")
        return None

def get_spx():
    price = get_price_from_investing("https://www.investing.com/indices/us-spx-500")
    if not price:
        print("🔁 Falling back to Finnhub for SPX")
        price = get_price_finnhub("^GSPC")
    return price

def get_es():
    price = get_price_from_investing("https://www.investing.com/indices/us-spx-500-futures")
    if not price:
        print("🔁 Falling back to Finnhub for ES")
        price = get_price_finnhub("ES=F")
    return price

def get_vix():
    price = get_price_from_investing("https://www.investing.com/indices/volatility-s-p-500")
    if not price:
        print("🔁 Falling back to Finnhub for VIX")
        price = get_price_finnhub("^VIX")
    return price


def get_previous_values():
    try:
        vix_series = fred.get_series("VIXCLS").dropna()
        spx_series = fred.get_series("SP500").dropna()
        if len(vix_series) >= 2 and len(spx_series) >= 2:
            return float(spx_series.iloc[-2]), float(vix_series.iloc[-2])
    except Exception as e:
        print("❌ Error fetching previous values from FRED:", e)
    return None, None

def get_weekly_trend_bias():
    try:
        spx_series = fred.get_series("SP500").dropna()
        last = spx_series.iloc[-1]
        week_ago = spx_series.iloc[-6]
        return 1 if last > week_ago else -1 if last < week_ago else 0
    except:
        return 0

def rule_based_market_bias(sentiment_score, vix, es, spx):
    bias = 0
    reasons = []

    capped_sentiment = max(min(sentiment_score, 10), -10)
    if capped_sentiment >= 3:
        bias += 1
        reasons.append("Moderate-to-strong positive sentiment (≥3)")
    elif capped_sentiment <= -3:
        bias -= 1
        reasons.append("Strong negative sentiment (≤ -3)")

    if (es - spx) / spx > 0.0015:  # ~0.15%
        bias += 1
        reasons.append("ES futures significantly above SPX")

    if vix:
        if vix < 18:
            bias += 1
            reasons.append("Low VIX (<18) - low fear")
        elif vix > 25:
            bias -= 1
            reasons.append("High VIX (>25) - market fear")

    direction = "📈 Bullish" if bias > 0 else "📉 Bearish"
    return direction


def log_market_features(spx, es, vix, prev_spx, prev_vix, implied_move, sentiment_score, bias_label):
    print("prev_spx:", prev_spx, "prev_vix:", prev_vix)
    vix_delta = (vix - prev_vix) / prev_vix if prev_vix and isinstance(vix, float) else 0
    futures_gap = es - prev_spx if prev_spx and isinstance(es, float) else 0
    trend_bias = get_weekly_trend_bias()
    chicago_time = datetime.datetime.now(pytz.timezone("America/Chicago")).strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "timestamp": chicago_time,
        "weekly_trend": trend_bias,
        "sentiment_score": sentiment_score,
        "implied_move": implied_move,
        "vix": vix,
        "vix_delta": vix_delta,
        "futures_gap": futures_gap,
        "spx": spx,
        "bias": bias_label
    }
    path = os.path.join(DOWNLOAD_DIR, "market_features.csv")
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        fieldnames = ["timestamp", "weekly_trend", "sentiment_score", "implied_move", "vix", "vix_delta", "futures_gap", "spx", "bias"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def train_logistic_model():
    path = os.path.join(DOWNLOAD_DIR, "market_features.csv")

    try:
        df = pd.read_csv(path, usecols=[
            "weekly_trend", "sentiment_score", "implied_move",
            "vix", "vix_delta", "futures_gap", "spx"
        ])
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return None

    # Ensure all data is numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(inplace=True)

    # Create binary target column based on SPX next move
    df["target"] = (df["spx"].shift(-1) > df["spx"]).astype(int)
    df.dropna(inplace=True)

    # Handle case where only one class is present (e.g., all 0 or all 1)
    if df["target"].nunique() < 2:
        print("⚠️ Cannot train model: only one class present in target column.")
        return None

    # Features for training
    features = ["weekly_trend", "sentiment_score", "implied_move", "vix", "vix_delta", "futures_gap"]
    X = df[features]
    y = df["target"]

    # Train/test split (no shuffle to preserve time series order)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LogisticRegression()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"🔍 Logistic Regression Accuracy: {accuracy:.2f}")
    return model


def predict_with_model(model, features_dict):
    if model is None:
        return "⚠️ No ML prediction due to training issue."

    import numpy as np
    try:
        features = np.array([[features_dict[k] for k in [
            "weekly_trend", "sentiment_score", "implied_move",
            "vix", "vix_delta", "futures_gap"
        ]]])
        prob = model.predict_proba(features)[0][1]
        direction = "📈 Bullish" if prob > 0.5 else "📉 Bearish"
        print(f"🤖 ML Prediction: {direction} (Prob: {prob:.2f})")
        return direction
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return "⚠️ Prediction failed"


def send_email(subject, spx, vix, es, news, direction, reasons, move_msg, to_email):
    try:
        import pytz
        # Get current time in US/Central
        central = pytz.timezone('US/Central')
        now_cst = datetime.datetime.now(central)
        current_time_cst = now_cst.strftime('%I:%M %p CT')

        # Decide market window message
        hour = now_cst.hour
        if hour < 11:
            market_window_note = "🕒 Monitoring trends for the 8:00 AM to 12:00 PM market window."
        else:
            market_window_note = "🕒 Monitoring trends for the 12:00 PM to 3:00 PM market window."

        message = MIMEMultipart("alternative")
        message["From"] = os.getenv("EMAIL_USER")
        message["To"] = to_email
        message["Subject"] = subject

        # Plaintext fallback
        body_text = f"""
📊 Pre-Market Alert for {datetime.date.today()}
🔹 SPX: {spx}  🔺 VIX: {vix}  📉 ES: {es}

📰 Headlines:
{chr(10).join([f"- [{src}] {headline} ({score:+})" for src, score, headline in news])}

📊 Market Bias: {direction}
{chr(10).join([f"- {r}" for r in reasons])}

📉 Expected Move: {move_msg}

{market_window_note}

Generated by CDUS Trading Bot • {current_time_cst}
        """

        # HTML version
        html = f"""
        <html>
        <body style="font-family: Arial, sans-serif; padding: 20px; color: #333;">
            <h2 style="color: #0d6efd;">📊 Pre-Market Alert for {datetime.date.today()}</h2>
            <p>
                <strong>🔹 SPX:</strong> {spx} &nbsp;&nbsp;
                <strong>🔺 VIX:</strong> {vix} &nbsp;&nbsp;
                <strong>📉 ES:</strong> {es}
            </p>

            <h3>📰 Headlines:</h3>
            <ul>
                {''.join(f"<li>[{src}] {headline} ({score:+})</li>" for src, score, headline in news)}
            </ul>

            <h3>📊 Market Bias: {direction}</h3>
           <!-- <h4>📈 Implied Move (SPY ATM): {move_msg}</h4> -->

            <p style="font-size: 1em; color: #333; margin-top: 20px;">
                <strong>{market_window_note}</strong>
            </p>

            <p style="font-size: 0.9em; color: #888; margin-top: 10px;">
                Generated by CDUS Trading Bot • {current_time_cst}
            </p>
        </body>
        </html>
        """

        # Attach both parts
        message.attach(MIMEText(body_text, "plain"))
        message.attach(MIMEText(html, "html"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(os.getenv("EMAIL_USER"), os.getenv("EMAIL_PASS"))
            server.send_message(message)
            print("✅ Email sent.")
    except Exception as e:
        print("❌ Email failed:", e)
        
def is_us_market_holiday():
    from pandas.tseries.holiday import USFederalHolidayCalendar
    from pandas.tseries.offsets import CustomBusinessDay

    today = pd.Timestamp.today().normalize()
    calendar = USFederalHolidayCalendar()
    holidays = calendar.holidays(start=today - pd.Timedelta(days=1), end=today + pd.Timedelta(days=1))
    return today in holidays
    
def main():
    
    if is_us_market_holiday():
        print("📅 US Market is closed today. Skipping alert.")
        return
    
    today = datetime.date.today()
    news = get_all_market_news()
    source_weights = {
    "CNBC": 1.5,
    "Finnhub": 1.0,
    "Marketaux": 1.0
}
    sentiment_score = sum(score * source_weights.get(src, 1.0) for src, score, _ in news)

    sentiment_score = sentiment_score / len(news) if news else 0
    sentiment_score = max(min(sentiment_score, 10), -10)




    implied_move = get_implied_move_yfinance("SPY", sentiment_score)
    spx, es, vix = get_spx(), get_es(), get_vix()
    prev_spx, prev_vix = get_previous_values()

    # ✅ Validate scraped and derived data
    try:
        implied_move_value = float(implied_move.strip('±+-%')) if implied_move else None
    except:
        implied_move_value = None

    if not all(isinstance(x, float) for x in [spx, es, vix]) or implied_move_value is None:
        print("⚠️ Skipping log and email: Market data is missing or invalid.")
        return

   # 📈 Rule-based signal
    direction = rule_based_market_bias(sentiment_score, vix, es, spx)
    print("📉 Rule-based Bias:", direction)

    #✅ Log market data
    bias_label = "Bullish" if "Bullish" in direction else "Bearish"
    log_market_features(spx, es, vix, prev_spx, prev_vix, implied_move_value, sentiment_score, bias_label)
    print(f"✅ Logged market data for {today} to CSV")


    # 🧠 Print headlines
    print("🧠 Classified Headlines with Sentiment:")
    for src, score, headline in news:
        print(f"[{src}] {score:+d} → {headline}")

    print(f"📊 Pre-Market Alert for {today}")
    print(f"SPX: {spx}, ES: {es}, VIX: {vix}")
    print(f"📈 Implied Move (SPY ATM): {implied_move}")
    print(f"Sentiment Score: {sentiment_score}")


    # Email reasons
    reasons = []
    if sentiment_score > 0: reasons.append("Positive sentiment score")
    if es > spx: reasons.append("ES futures are higher than SPX")
    if vix < 18: reasons.append("VIX is below 18 (low fear)")
        

    # 📧 Send email
    send_email(
        subject=f"📊 SPX Pre-Market Outlook – {today}",
        spx=spx,
        vix=vix,
        es=es,
        news=[(src, score, headline) for src, score, headline in news],
        direction=direction,
        reasons=reasons,
        move_msg=implied_move,
        to_email=EMAIL_TO
    )

    # 🤖 ML prediction
    market_data_path = os.path.join(DOWNLOAD_DIR, "market_features.csv")
    if os.path.exists(market_data_path):
        df = pd.read_csv(market_data_path, usecols=["weekly_trend", "sentiment_score", "implied_move", "vix", "vix_delta", "futures_gap", "spx"])

        if len(df) >= 10:
            model = train_logistic_model()
            vix_delta = (vix - prev_vix) / prev_vix if prev_vix else 0
            futures_gap = es - prev_spx if prev_spx else 0
            features_today = {
                "weekly_trend": get_weekly_trend_bias(),
                "sentiment_score": sentiment_score,
                "implied_move": float(implied_move_value),
                "vix": vix,
                "vix_delta": vix_delta,
                "futures_gap": futures_gap
            }
            predict_with_model(model, features_today)
        else:
            print("⚠️ Not enough data to train ML model yet. Waiting for more logs...")

if __name__ == "__main__":
    main()
