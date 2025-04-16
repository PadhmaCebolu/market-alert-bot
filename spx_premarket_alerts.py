import requests
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from dotenv import load_dotenv
import csv
import datetime


# Load environment variables
load_dotenv()

# =============================
# 📌 Utility Functions
# =============================

def tag_sentiment(text):
    negative_keywords = {
        "crash": 5, "recession": 4, "tariff": 4, "rate hike": 3,
        "selloff": 3, "inflation": 2, "conflict": 2, "drop": 2, "slide": 2,
        "war": 5, "defaults": 3, "plunge": 3
    }
    positive_keywords = {
        "rally": 3, "gain": 2, "growth": 2, "beat": 2,
        "optimism": 2, "cut rates": 3, "stimulus": 4,
        "record high": 4, "jump": 2
    }
    text = text.lower()
    score = 0
    for word, weight in negative_keywords.items():
        if word in text:
            score -= weight
    for word, weight in positive_keywords.items():
        if word in text:
            score += weight
    emoji = "📈" if score >= 3 else "📉" if score <= -3 else "🔹"
    return emoji, score

def is_market_relevant(text):
    keywords = ["fed", "tariff", "rate", "inflation", "yields", "bond", "treasury", "earnings", "revenue", "stocks", "markets", "recession", "jobless", "cpi", "ppi", "gdp", "volatility"]
    return any(k in text.lower() for k in keywords)

# =============================
# 📌 News Scrapers
# =============================

def scrape_headlines(url, selector, base_url=""):
    headlines = []
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        for el in soup.select(selector)[:10]:
            text = el.get_text(strip=True)
            link = el.get("href", "")
            if text and is_market_relevant(text):
                emoji, score = tag_sentiment(text)
                full_link = f"{base_url}{link}" if link.startswith("/") else link
                headlines.append((emoji, score, f"{emoji} {text} - {full_link}"))
    except Exception as e:
        print(f"⚠️ Error scraping {url}:", e)
    return headlines

def get_all_market_news():
    return (
        scrape_headlines("https://macenews.com/", ".elementor-heading-title") +
        scrape_headlines("https://www.cnbc.com/world/?region=world", "a.Card-title") +
        scrape_headlines("https://www.reuters.com/", "a[data-testid='Heading']", base_url="https://www.reuters.com")
    )

# =============================
# 📌 Market Data
# =============================

def get_price_from_investing(url):
    try:
        options = Options()
        options.add_argument("--headless")
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        time.sleep(4)
        price = driver.find_element(By.CSS_SELECTOR, '[data-test="instrument-price-last"]').text.replace(",", "")
        driver.quit()
        return float(price)
    except Exception as e:
        print(f"⚠️ Error retrieving price from {url}:", e)
        return "N/A"

def get_spx():
    return get_price_from_investing("https://www.investing.com/indices/us-spx-500")

def get_vix():
    return get_price_from_investing("https://www.investing.com/indices/volatility-s-p-500")

def get_es():
    return get_price_from_investing("https://www.investing.com/indices/us-spx-500-futures")

# =============================
# 📌 Analysis & Bias
# =============================

def estimate_direction(spx, es, sentiment_score, vix):
    score = 0
    reasons = []
    gap = es - spx if isinstance(spx, float) and isinstance(es, float) else 0

    if gap > 10:
        score += 1
        reasons.append("ES futures lead SPX → bullish")
    elif gap < -10:
        score -= 1
        reasons.append("ES futures lag SPX → bearish")

    if sentiment_score >= 3:
        score += 1
        reasons.append("Positive news bias")
    elif sentiment_score <= -3:
        score -= 1
        reasons.append("Negative news bias")

    if isinstance(vix, float) and vix > 30:
        score -= 1
        reasons.append("High VIX (>30) → bearish weight")

    if score >= 1:
        return "📈 Bullish", reasons
    elif score <= -1:
        return "📉 Bearish", reasons
    else:
        return "⚖️ Neutral", reasons

def calculate_vix_move(spx, vix, bias):
    try:
        if isinstance(spx, float) and isinstance(vix, float):
            move = (spx * vix / 100) / (252 ** 0.5)
            if bias == "📉 Bearish":
                move_points = -round(move, 2)
                return move_points, f"{move_points} pts drop expected"
            elif bias == "📈 Bullish":
                move_points = round(move, 2)
                return move_points, f"{move_points} pts rise expected"
            else:
                return round(move, 2), f"±{round(move, 2)} pts (~{round((move / spx) * 100, 2)}%)"
    except Exception as e:
        print(f"⚠️ Error calculating VIX move: {e}")
    return "N/A", "N/A"

# Logging function for pre-market predictions
def log_premarket_prediction(date, spx, es, vix, sentiment_score, direction, move_pts):
    log_file = os.path.join(os.getcwd(), "market_predictions.csv")
    file_exists = os.path.isfile(log_file)

    with open(log_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["date", "spx", "es", "vix", "sentiment_score", "predicted_trend", "predicted_move_pts"])
        writer.writerow([date, spx, es, vix, sentiment_score, direction, move_pts])
    print(f"📁 Logging to: {log_file}")  

# =============================
# 📧 Email Notification
# =============================

def send_email(subject, body, to_email):
    from_email = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASS")
    smtp_server = "smtp.gmail.com"
    port = 587

    message = MIMEMultipart()
    message["From"] = from_email
    message["To"] = to_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "html"))

    try:
        with smtplib.SMTP(smtp_server, port) as server:
            server.starttls()
            server.login(from_email, password)
            server.send_message(message)
            print("✅ Email sent successfully!")
    except Exception as e:
        print("❌ Email failed:", e)

# =============================
# 📌 Main Alert Function
# =============================

def main():
    today = datetime.date.today()
    spx, vix, es = get_spx(), get_vix(), get_es()
    news = get_all_market_news()
    sentiment_score = sum(score for _, score, _ in news)
    direction, reasons = estimate_direction(spx, es, sentiment_score, vix)
    move_pts, move_msg = calculate_vix_move(spx, vix, direction)


    # Create an HTML formatted email
html_message = f"""
<html>
  <body style="font-family:Arial, sans-serif; line-height:1.6;">
    <h2 style="color:#0a66c2;">📊 Pre-Market Alert for {today}</h2>
    
    <p>
      <b>🔹 SPX:</b> {spx} &nbsp;&nbsp;
      <b>🔺 VIX:</b> {vix} &nbsp;&nbsp;
      <b>📉 ES:</b> {es}
    </p>

    <h3>📰 Headlines</h3>
    <ul>
      {''.join(f"<li>{h}</li>" for _, _, h in news)}
    </ul>

    <h3 style="color:#dc3545;">📊 Market Bias: <span style="font-size: 1.2em;">{direction}</span></h3>
    <ul>
      {''.join(f"<li>{r}</li>" for r in reasons)}
    </ul>

    <h3 style="color:#800080;">📉 VIX-Derived Expected Move:</h3>
    <p style="font-size:1.1em; font-weight:bold;">{move_msg}</p>

    <br/>
    <p style="font-size: 0.9em; color: gray;">⏰ Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
  </body>
</html>
"""


    

    log_premarket_prediction(
    date=today,
    spx=spx,
    es=es,
    vix=vix,
    sentiment_score=sentiment_score,
    direction=direction,
    move_pts=move_pts
)
    # Send email (customize this call)
    send_email(subject="📊 Pre-Market Alert", body=html_message, to_email=os.getenv("EMAIL_TO"))
    

if __name__ == "__main__":
    main()
