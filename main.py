import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import os
from pathlib import Path
import pandas_ta as ta
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import plotly.graph_objects as go
import json
import urllib.parse
import requests
import random
import time
from bs4 import BeautifulSoup
import nltk
from transformers import pipeline
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Download required lexicons
nltk.download('vader_lexicon')

# Streamlit Configuration
st.set_page_config(
    page_title="Stock Analysis & Forecasting",
    page_icon="favicon.ico",
    layout="wide"
)
hide_streamlit_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# File Paths & Directories
if os.getenv("STREAMLIT_SERVER_PORT"):
    BASE_DIR = "/tmp/DATASETS"
else:
    BASE_DIR = os.path.join(os.path.dirname(__file__), "DATASETS")

DAILY_DIR = os.path.join(BASE_DIR, "Daily_data")
WEEKLY_DIR = os.path.join(BASE_DIR, "Weekly_data")
MONTHLY_DIR = os.path.join(BASE_DIR, "Monthly_data")
SECTORS_FILE = os.path.join(BASE_DIR, "sectors with symbols.csv")

# Ensure directories exist
for dir_path in [DAILY_DIR, WEEKLY_DIR, MONTHLY_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# --- GFS Analysis Functions ---
def get_latest_date():
    today = dt.date.today()
    return today.strftime("%Y-%m-%d")

def clean_and_save_data(data, filepath):
    data.reset_index(inplace=True)
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    for col in data.columns[1:]:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()
    data.to_csv(filepath, index=False)

def download_stock_data(interval, folder, custom_start_date="2020-01-01", custom_end_date=None):
    base_path = BASE_DIR
    filepath = os.path.join(base_path, "indicesstocks.csv")
    start_date = custom_start_date
    end_date = custom_end_date if custom_end_date else get_latest_date()

    total_symbols = 0
    with open(filepath) as f:
        for line in f:
            if "," not in line:
                continue
            symbols = line.split(",")
            total_symbols += len([s.strip() for s in symbols if s.strip()])

    if total_symbols == 0:
        st.warning("No symbols found in indicesstocks.csv")
        return

    progress_bar = st.progress(0)
    status_text = st.empty()
    processed = 0

    with open(filepath) as f:
        for line in f:
            if "," not in line:
                continue
            symbols = line.split(",")
            for symbol in symbols:
                symbol = symbol.strip()
                if not symbol:
                    continue
                try:
                    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
                    ticketfilename = symbol.replace(".", "_")
                    save_path = os.path.join(base_path, folder, f"{ticketfilename}.csv")
                    clean_and_save_data(data, save_path)
                    processed += 1
                    progress = processed / total_symbols
                    progress_bar.progress(progress)
                    status_text.text(f"Downloading & Updating {folder} data: {processed}/{total_symbols} ({progress:.1%})")
                except Exception as e:
                    st.error(f"Error downloading {symbol}: {e}")
                    processed += 1
                    progress_bar.progress(processed / total_symbols)

    progress_bar.empty()
    status_text.empty()
    st.success(f"{folder.replace('_', ' ').title()} download & update completed!")

def fetch_vix():
    vix = yf.Ticker("^VIX")
    vix_data = vix.history(period="1d")
    return vix_data['Close'].iloc[0] if not vix_data.empty else None

def append_row(df, row):
    return pd.concat([df, pd.DataFrame([row], columns=row.index)]).reset_index(drop=True) if not row.isnull().all() else df

def getRSI14_and_BB(csvfilename):
    if Path(csvfilename).is_file():
        try:
            df = pd.read_csv(csvfilename)
            if df.empty or 'Close' not in df.columns:
                return 0.00, 0.00, 0.00, 0.00
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df['rsi14'] = ta.rsi(df['Close'], length=14)
            bb = ta.bbands(df['Close'], length=20)
            if bb is None or df['rsi14'] is None:
                return 0.00, 0.00, 0.00, 0.00
            df['lowerband'] = bb['BBL_20_2.0']
            df['middleband'] = bb['BBM_20_2.0']
            rsival = df['rsi14'].iloc[-1].round(2)
            ltp = df['Close'].iloc[-1].round(2)
            lowerband = df['lowerband'].iloc[-1].round(2)
            middleband = df['middleband'].iloc[-1].round(2)
            return rsival, ltp, lowerband, middleband
        except Exception:
            return 0.00, 0.00, 0.00, 0.00
    else:
        return 0.00, 0.00, 0.00, 0.00

def dayweekmonth_datasets(symbol, symbolname, index_code):
    symbol_with_underscore = symbol.replace('.', '_')
    day_path = os.path.join(DAILY_DIR, f"{symbol_with_underscore}.csv")
    week_path = os.path.join(WEEKLY_DIR, f"{symbol_with_underscore}.csv")
    month_path = os.path.join(MONTHLY_DIR, f"{symbol_with_underscore}.csv")
    cday = dt.datetime.today().strftime('%d/%m/%Y')
    dayrsi14, dltp, daylowerband, daymiddleband = getRSI14_and_BB(day_path)
    weekrsi14, wltp, weeklowerband, weekmiddleband = getRSI14_and_BB(week_path)
    monthrsi14, mltp, monthlowerband, monthmiddleband = getRSI14_and_BB(month_path)
    new_row = pd.Series({
        'entrydate': cday,
        'indexcode': index_code,
        'indexname': symbolname,
        'dayrsi14': dayrsi14,
        'weekrsi14': weekrsi14,
        'monthrsi14': monthrsi14,
        'dltp': dltp,
        'daylowerband': daylowerband,
        'daymiddleband': daymiddleband,
        'weeklowerband': weeklowerband,
        'weekmiddleband': weekmiddleband,
        'monthlowerband': monthlowerband,
        'monthmiddleband': monthmiddleband
    })
    return new_row

def generateGFS(scripttype):
    indicesdf = pd.DataFrame(columns=[
        'entrydate', 'indexcode', 'indexname', 'dayrsi14', 
        'weekrsi14', 'monthrsi14', 'dltp', 'daylowerband', 
        'daymiddleband', 'weeklowerband', 'weekmiddleband', 
        'monthlowerband', 'monthmiddleband'
    ])
    try:
        with open(os.path.join(BASE_DIR, f"{scripttype}.csv")) as f:
            for line in f:
                if "," not in line:
                    continue
                symbol, symbolname = line.split(",")[0], line.split(",")[1]
                new_row = dayweekmonth_datasets(symbol.strip(), symbolname.strip(), symbol.strip())
                indicesdf = append_row(indicesdf, new_row)
    except Exception as e:
        st.error(f"Error generating GFS report: {e}")
    return indicesdf

def read_indicesstocks(file_path):
    indices_dict = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) > 1:
                    index_code = parts[0].strip()
                    indices_dict[index_code] = [stock.strip() for stock in parts[1:]]
    except Exception as e:
        st.error(f"Error reading indicesstocks.csv: {e}")
    return indices_dict

# --- LSTM Prediction Functions ---
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def predict_future(model, last_sequence, scaler, days=5):
    predictions = []
    current_sequence = last_sequence.copy()
    for _ in range(days):
        next_pred = model.predict(current_sequence.reshape(1, -1, 1))
        predicted_value = next_pred[0, 0]
        predictions.append(predicted_value)
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = predicted_value
    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()

def get_predicted_values(data, epochs=170, batch_size=32, start_date=None, end_date=None):
    df = data.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    if start_date and end_date:
        df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))]
    
    if len(df) < 50:
        st.warning("Not enough data in selected date range. Please select a wider range.")
        return None

    if 'High' in df.columns and 'Low' in df.columns:
        avg_high_gap = (df['High'] - df['Close']).mean()
        avg_low_gap = (df['Close'] - df['Low']).mean()
    else:
        avg_high_gap = 0
        avg_low_gap = 0
    
    close_data = df[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)
    
    time_step = 13
    train_size = int(len(scaled_data) * 0.8)
    train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
    
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(time_step, 1)),
        LSTM(32, return_sequences=True),
        LSTM(32),
        Dense(1)
    ])
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    with st.spinner("Training model..."):
        model.fit(X_train, y_train, validation_data=(X_test, y_test), 
                  epochs=epochs, batch_size=batch_size, verbose=1)
    
    train_pred = scaler.inverse_transform(model.predict(X_train))
    test_pred = scaler.inverse_transform(model.predict(X_test))
    
    y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    train_r2 = r2_score(y_train_actual, train_pred)
    test_r2 = r2_score(y_test_actual, test_pred)
    
    train_dates = df['Date'].iloc[time_step + 1 : train_size]
    test_start = train_size + time_step
    test_end = test_start + len(y_test_actual)
    test_dates = df['Date'].iloc[test_start : test_end]
    
    last_sequence = scaled_data[-time_step:]
    future_preds = predict_future(model, last_sequence, scaler)
    
    future_high = future_preds + avg_high_gap
    future_low = future_preds - avg_low_gap
    
    return (
        train_r2, test_r2, future_preds, future_high, future_low,
        train_dates, y_train_actual, train_pred,
        test_dates, y_test_actual, test_pred
    )

# --- News Scraping & Sentiment Analysis Functions ---
def get_session():
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

session_news = get_session()

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36",
]

@st.cache_resource
def load_finbert():
    return pipeline("text-classification", model="ProsusAI/finbert")

def analyze_sentiment(text, method="FinBERT"):
    try:
        finbert = load_finbert()
        if not text.strip():
            return "Neutral"
        result = finbert(text[:512], truncation=True)[0]
        return result['label'].capitalize()
    except Exception as e:
        if not st.session_state.error_logged:
            st.error(f"Sentiment analysis error: {e}")
            st.session_state.error_logged = True
        return "Error"

def scrape_moneycontrol_news(company):
    search_url = f"https://www.moneycontrol.com/news/tags/{str(company).replace(' ', '-').lower()}.html"
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    try:
        response = session_news.get(search_url, headers=headers, timeout=10)
        if "captcha" in response.text.lower():
            raise Exception("CAPTCHA detected on Moneycontrol")
        response.raise_for_status()
        time.sleep(random.uniform(1, 3))
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('li', class_='clearfix')[:5]
        news = []
        for article in articles:
            if article.find('h2') and article.find('a'):
                headline = article.find('h2').text.strip()
                link = article.find('a')['href']
                news.append((headline, link))
        return news
    except Exception:
        return []

def scrape_bing_news(company):
    search_url = f"https://www.bing.com/news/search?q={company.replace(' ', '+')}"
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    try:
        response = session_news.get(search_url, headers=headers, timeout=10)
        if "captcha" in response.text.lower():
            raise Exception("CAPTCHA detected on Bing News")
        response.raise_for_status()
        time.sleep(random.uniform(1, 3))
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all("div", class_="news-card")[:5]
        results = []
        for a in articles:
            link_elem = a.find("a")
            if link_elem:
                headline = link_elem.get_text(strip=True)
                link = link_elem.get("href")
                results.append((headline, link))
        return results if results else []
    except Exception:
        return []

def fetch_news_newsapi(company):
    import requests.utils
    news_api_key = "063b1b2696c24c3a867c46c94cf9b810"
    keywords = (
        '"stocks" OR "market" OR "earnings" OR "finance" OR "shares" OR '
        '"dividends" OR "profit" OR "investment" OR "revenue" OR "IPO" OR '
        '"acquisition" OR "merger" OR "valuation" OR "forecast" OR "guidance" OR '
        '"liquidity" OR "debt" OR "equity" OR "yield" OR "expense" OR '
        '"gain" OR "growth" OR "upturn" OR "surge" OR "positive" OR "favorable" OR '
        '"improvement" OR "expansion" OR "strong" OR '
        '"loss" OR "decline" OR "drop" OR "bearish" OR "downturn" OR "negative" OR '
        '"weak" OR "recession" OR "crisis" OR "slump"'
    )
    query = f'"{company}" AND ({keywords})'
    encoded_query = requests.utils.quote(query)
    search_url = f"https://newsapi.org/v2/everything?qInTitle={encoded_query}&apiKey={news_api_key}"
    try:
        response = session_news.get(search_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])
        news = []
        for article in articles[:10]:
            headline = article.get("title", "")
            link = article.get("url", "")
            publishedAt = article.get("publishedAt", None)
            if headline and link:
                news.append((headline, link, publishedAt))
        return news
    except Exception:
        return []

def filter_recent_news(news, days=20):
    filtered = []
    now = dt.datetime.utcnow()
    for item in news:
        if len(item) == 3 and item[2]:
            try:
                pub_date = dt.datetime.strptime(item[2], "%Y-%m-%dT%H:%M:%SZ")
                if (now - pub_date).days <= days:
                    filtered.append(item)
            except Exception:
                pass
        else:
            filtered.append(item)
    return filtered

def fetch_news(company):
    news = scrape_moneycontrol_news(company)
    if not news:
        time.sleep(random.uniform(1, 3))
        news = scrape_bing_news(company)
    return news

def update_filtered_indices_by_sentiment(filepath, sentiment_method="FinBERT", use_newsapi=False, aggregate_sources=False):
    df = pd.read_csv(filepath)
    companies = df["Company Name"].unique()
    sentiment_summary = {}
    updated_companies = []
    news_data = {}

    progress_bar = st.progress(0)
    for idx, company in enumerate(companies):
        progress_bar.progress((idx + 1) / len(companies))
        if aggregate_sources:
            news_api_news = fetch_news_newsapi(company)
            news_api_news = filter_recent_news(news_api_news, days=20)
            default_news = fetch_news(company)
            combined_news = news_api_news + default_news
            unique_news = []
            seen = set()
            for item in combined_news:
                if len(item) == 3:
                    headline, link, pub_date = item
                    tup = (headline, link, pub_date)
                elif len(item) == 2:
                    headline, link = item
                    tup = (headline, link, "")
                else:
                    continue
                key = (headline, link)
                if key not in seen:
                    unique_news.append(tup)
                    seen.add(key)
            news = unique_news
        elif use_newsapi:
            news = fetch_news_newsapi(company)
            news = filter_recent_news(news, days=20)
        else:
            news = fetch_news(company)
            
        company_news_details = []
        pos = neg = neu = 0
        
        if not news:
            verdict = "Neutral"
        else:
            for article in news:
                if len(article) == 3:
                    headline, link, pub_date = article
                elif len(article) == 2:
                    headline, link = article
                    pub_date = ""
                else:
                    continue
                sentiment = analyze_sentiment(headline, sentiment_method)
                company_news_details.append((headline, sentiment, link, pub_date))
                if sentiment == "Positive":
                    pos += 1
                elif sentiment == "Negative":
                    neg += 1
                else:
                    neu += 1
            total = pos + neg + neu
            if total > 0:
                if (pos / total) > 0.4:
                    verdict = "Positive"
                elif (neg / total) > 0.4:
                    verdict = "Negative"
                else:
                    verdict = "Neutral"
            else:
                verdict = "Neutral"

        sentiment_summary[company] = {"Verdict": verdict, "Positive": pos, "Negative": neg, "Neutral": neu}
        news_data[company] = company_news_details

        if verdict != "Negative":
            updated_companies.append(company)

    progress_bar.empty()
    updated_df = df[df["Company Name"].isin(updated_companies)]
    updated_df.to_csv(filepath, index=False)

    return sentiment_summary, updated_df, news_data

# --- Candlestick Pattern Recognition Functions ---
def detect_candlestick_patterns(df):
    df = df.copy()
    df['Body'] = abs(df['Close'] - df['Open'])
    df['Range'] = df['High'] - df['Low']
    df['Lower_Shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['Upper_Shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    
    df['Hammer'] = ((df['Body'] <= 0.3 * df['Range']) & 
                    (df['Lower_Shadow'] >= 2 * df['Body']) & 
                    (df['Upper_Shadow'] <= 0.1 * df['Range']))
    
    df['Doji'] = (df['Body'] <= 0.1 * df['Range'])
    
    df['Bullish_Engulfing'] = ((df['Open'].shift(1) > df['Close'].shift(1)) &
                               (df['Open'] < df['Close']) &
                               (df['Open'] < df['Close'].shift(1)) &
                               (df['Close'] > df['Open'].shift(1)))
    
    df['Bearish_Engulfing'] = ((df['Open'].shift(1) < df['Close'].shift(1)) &
                               (df['Open'] > df['Close']) &
                               (df['Open'] > df['Close'].shift(1)) &
                               (df['Close'] < df['Open'].shift(1)))
    return df

def plot_candlestick_with_patterns(df):
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Candlesticks')])
    hammer_df = df[df['Hammer']]
    doji_df = df[df['Doji']]
    bullish_df = df[df['Bullish_Engulfing']]
    bearish_df = df[df['Bearish_Engulfing']]
    
    fig.add_trace(go.Scatter(x=hammer_df['Date'], y=hammer_df['Low'],
                             mode='markers', marker=dict(color='green', size=10, symbol='triangle-up'),
                             name='Hammer'))
    fig.add_trace(go.Scatter(x=doji_df['Date'], y=doji_df['Low'],
                             mode='markers', marker=dict(color='blue', size=10, symbol='diamond'),
                             name='Doji'))
    fig.add_trace(go.Scatter(x=bullish_df['Date'], y=bullish_df['Low'],
                             mode='markers', marker=dict(color='lime', size=10, symbol='star'),
                             name='Bullish Engulfing'))
    fig.add_trace(go.Scatter(x=bearish_df['Date'], y=bearish_df['High'],
                             mode='markers', marker=dict(color='red', size=10, symbol='star'),
                             name='Bearish Engulfing'))
    fig.update_layout(title="Candlestick Pattern Recognition", xaxis_title="Date", yaxis_title="Price", dragmode="pan")
    return fig

# --- Session State Initialization ---
if "user_stock" not in st.session_state:
    st.session_state.user_stock = ""

if "gfs_output" not in st.session_state:
    st.session_state.gfs_output = None

if "lstm_output" not in st.session_state:
    st.session_state.lstm_output = None

if "single_stock_output" not in st.session_state:
    st.session_state.single_stock_output = None

if "error_logged" not in st.session_state:
    st.session_state.error_logged = False

# --- Streamlit App UI ---
st.title("Stock Analysis & Forecasting Dashboard ")
st.markdown(
    "This app performs a three-step process: first it runs a GFS analysis to filter qualified indices/stocks, "
    "then it applies a news sentiment filter (using FinBERT) before running an LSTM model for future price forecast."
)
st.sidebar.header("Information")
st.sidebar.markdown(
    """
- **Volatility Index:** High VIX indicates a volatile market.
- **GFS Analysis:** Download data, calculate technical indicators, and filter stocks.
- **News Sentiment Analysis:** Remove companies with negative news sentiment using FinBERT.
- **Stock Prediction (LSTM):** forecast's the future prices using an LSTM model.
- **Candlestick Pattern Recognition:** Detect common candlestick patterns to help time entries/exits.
"""
)

# Create Four Main Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "GFS Analysis & News Sentiment Analysis",
    "Stock Forecasting (LSTM)",
    "Single Stock Analysis",
    "Fundamental Analysis Tools"
])

# ----------- Tab 1: GFS Analysis & News Sentiment Analysis -----------
with tab1:
    st.header("GFS Analysis")
    st.markdown(
        """
        **Overview:**  
        This section downloads stock data (Daily/Weekly/Monthly) for symbols listed in `indicesstocks.csv`, 
        calculates technical indicators (RSI, Bollinger Bands), and filters stocks based on multi-timeframe criteria.
        """
    )

    if st.button("Run Full GFS Analysis"):
        with st.spinner("Fetching VIX data..."):
            vix_value = fetch_vix()
            st.session_state.vix_value = vix_value

        if vix_value is None:
            st.error("Could not fetch VIX data. Please try again later.")
        else:
            st.session_state.show_data_choice = True
            if vix_value > 20:
                st.warning(f"**High Volatility Detected (VIX: {vix_value:.2f})**  \nMarket conditions are volatile. Proceed with caution.")
            else:
                st.success(f"Market Volatility Normal (VIX: {vix_value:.2f})")

    if st.session_state.get('show_data_choice', False):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Update Data with Latest"):
                st.session_state.data_choice = 'update'
        with col2:
            st.info("Data will be downloaded fresh.")
            st.session_state.data_choice = 'update'

        if 'data_choice' in st.session_state:
            if st.session_state.data_choice == 'update':
                update_option = st.selectbox("Select dataset update option", options=["Use Current Date", "Select Custom End Date"])
                fixed_start_date = "2020-01-01"
                if update_option == "Select Custom End Date":
                    custom_end_date = st.date_input("Select Custom End Date", value=dt.date.today(), max_value=dt.date.today())
                    selected_end_date_str = custom_end_date.strftime("%Y-%m-%d")
                else:
                    selected_end_date_str = None
                if st.button("Proceed to Download and Update Data"):
                    with st.spinner("Downloading and updating data..."):
                        os.makedirs(DAILY_DIR, exist_ok=True)
                        os.makedirs(WEEKLY_DIR, exist_ok=True)
                        os.makedirs(MONTHLY_DIR, exist_ok=True)
                        download_stock_data(interval='1d', folder='Daily_data', custom_start_date=fixed_start_date, custom_end_date=selected_end_date_str)
                        download_stock_data(interval='1wk', folder='Weekly_data', custom_start_date=fixed_start_date, custom_end_date=selected_end_date_str)
                        download_stock_data(interval='1mo', folder='Monthly_data', custom_start_date=fixed_start_date, custom_end_date=selected_end_date_str)
                    
                    def generate_gfs_reports():
                        df3 = generateGFS('indicesdf')
                        df4 = df3.loc[
                            df3['monthrsi14'].between(35, 60) &
                            df3['weekrsi14'].between(35, 60) &
                            df3['dayrsi14'].between(35, 60)
                        ]
                        st.markdown("### Qualified Indices")
                        if df4.empty:
                            st.warning("No indices meet GFS criteria.")
                        else:
                            st.dataframe(df4.style.format({
                                'dayrsi14': '{:.2f}',
                                'weekrsi14': '{:.2f}',
                                'monthrsi14': '{:.2f}',
                                'dltp': '{:.2f}'
                            }), use_container_width=True)
                            df4.to_csv(os.path.join(BASE_DIR, "filtered_indices.csv"), index=False)
                        
                        st.markdown("### Qualified Stocks")
                        indices_dict = read_indicesstocks(os.path.join(BASE_DIR, "indicesstocks.csv"))
                        results_df = pd.DataFrame(columns=df3.columns)
                        for index in df4['indexcode']:
                            if index in indices_dict:
                                for stock in indices_dict[index]:
                                    if stock:
                                        new_row = dayweekmonth_datasets(stock, stock, index)
                                        results_df = append_row(results_df, new_row)
                        
                        results_df = results_df.loc[
                            results_df['monthrsi14'].between(40, 60) &
                            results_df['weekrsi14'].between(40, 60) &
                            results_df['dayrsi14'].between(40, 60)
                        ]
                        
                        sectors_df = pd.read_csv(SECTORS_FILE)
                        results_df = results_df.merge(
                            sectors_df[['Index Name', 'Company Name']],
                            left_on='indexname',
                            right_on='Index Name',
                            how='left'
                        )
                        results_df.drop(columns=['Index Name'], inplace=True, errors='ignore')
                        results_df['Company Name'] = results_df['Company Name'].fillna('N/A')
                        
                        if results_df.empty:
                            st.warning("No stocks meet GFS criteria.")
                        else:
                            st.dataframe(results_df.style.format({
                                'dayrsi14': '{:.2f}',
                                'weekrsi14': '{:.2f}',
                                'monthrsi14': '{:.2f}',
                                'dltp': '{:.2f}'
                            }), use_container_width=True)
                            results_df.to_csv(os.path.join(BASE_DIR, "filtered_indices_output.csv"), index=False)
                        
                        st.success("GFS Analysis completed!")
                        st.session_state.gfs_output = results_df
                    generate_gfs_reports()
                    del st.session_state.data_choice
                    st.session_state.show_data_choice = False

    st.markdown("## News Sentiment Analysis")
    st.markdown(
        """
        The filtered stocks from the GFS analysis are now evaluated based on recent news sentiment using FinBERT.  
        Companies with an overall negative sentiment (based on scraped news headlines) will be removed.
        """
    )
    filtered_file = os.path.join(BASE_DIR, "filtered_indices_output.csv")
    if os.path.exists(filtered_file):
        st.markdown("**Using FinBERT for sentiment analysis.**")
        use_newsapi = st.checkbox("Keyword Specific (NewsAPI.org)", value=False)
        aggregate_sources = st.checkbox("Aggregate Keywords & Multi-Sources", value=False)
        if st.button("Run News Sentiment Analysis"):
            with st.spinner("Analyzing news sentiment for each company..."):
                sentiment_summary, updated_df, news_data = update_filtered_indices_by_sentiment(
                    filtered_file,
                    sentiment_method="FinBERT",
                    use_newsapi=use_newsapi,
                    aggregate_sources=aggregate_sources
                )
            st.success("News Sentiment Analysis completed!")
            st.markdown("### Sentiment Summary")
            summary_df = pd.DataFrame.from_dict(sentiment_summary, orient='index')
            st.dataframe(summary_df)
            st.markdown("### Updated Filtered Indices")
            st.dataframe(updated_df.style.format({
                'dayrsi14': '{:.2f}',
                'weekrsi14': '{:.2f}',
                'monthrsi14': '{:.2f}',
                'dltp': '{:.2f}'
            }), use_container_width=True)
            st.markdown("### Detailed News Analysis")
            for company, articles in news_data.items():
                if articles:
                    with st.expander(f"{company} ({len(articles)} articles)"):
                        for article in articles:
                            if len(article) == 4:  # Ensure we have all 4 elements
                                headline, sentiment, link, pub_date = article
                                pub_str = f" (Published: {pub_date[:10]})" if pub_date else ""
                                st.markdown(
                                    f"**{headline}**{pub_str}<br>"
                                    f"Sentiment: `{sentiment}`<br>"
                                    f"<a href='{link}' target='_blank'>Read Article</a>",
                                    unsafe_allow_html=True
                                )
            st.session_state.news_sentiment = {"summary": summary_df, "updated": updated_df}
    else:
        st.info("GFS analysis output not found. Please run the GFS Analysis first.")

# ----------- Tab 2: Stock Prediction (LSTM) -----------
with tab2:
    st.header("Stock Forecasting using LSTM")
    st.markdown(
        """
        **Overview:**  
        This section loads the filtered stocks (output from the GFS and News Sentiment Analysis) 
        and trains an LSTM model on their daily data to forecast future prices.
        """
    )
    filtered_indices_path = os.path.join(BASE_DIR, "filtered_indices_output.csv")
    if os.path.exists(filtered_indices_path):
        selected_indices = pd.read_csv(filtered_indices_path)
        st.success("Loaded filtered indices from GFS & News Sentiment Analysis.")
    else:
        st.error("Filtered indices file not found. Please run the GFS and News Sentiment Analysis first.")
        st.stop()

    if os.path.exists(SECTORS_FILE):
        sectors_df = pd.read_csv(SECTORS_FILE)
    else:
        st.error("sectors with symbols.csv file not found at the specified path.")
        st.stop()

    daily_data = {}
    daily_files_list = [f for f in os.listdir(DAILY_DIR) if f.endswith('.csv')]
    if not daily_files_list:
        st.error("No daily data files found in the Daily_data folder.")
        st.stop()

    for file in daily_files_list:
        file_path = os.path.join(DAILY_DIR, file)
        name = os.path.splitext(file)[0].replace('_', '.')
        try:
            df_daily = pd.read_csv(file_path)
            daily_data[name] = df_daily
        except Exception as e:
            st.error(f"Error loading {file}: {e}")

    st.sidebar.header("LSTM Configuration")
    epochs_input = st.sidebar.number_input("Number of Epochs", min_value=1, max_value=500, value=170)
    start_date = st.sidebar.date_input("Select Start Date", value=dt.date(2020, 1, 1))
    end_date = st.sidebar.date_input("Select End Date", value=dt.date.today())
    if start_date > end_date:
        st.error("Error: End date must fall after start date.")
        st.stop()

    analysis_mode = st.selectbox("Select Analysis Mode", options=["Compare Multiple Stocks", "Single Stock Analysis"])

    if analysis_mode == "Single Stock Analysis":
        company_choice = st.selectbox("Select a Company", options=selected_indices['indexname'].unique())
        companies_to_process = selected_indices[selected_indices['indexname'] == company_choice]
    else:
        companies_to_process = selected_indices

    if st.button("Run LSTM Analysis"):
        results = []
        current_date = dt.datetime.now().strftime("%Y-%m-%d")
        
        if analysis_mode == "Compare Multiple Stocks":
            combined_fig = go.Figure()
        
        for _, row in companies_to_process.iterrows():
            index_name = row['indexname']
            if index_name not in daily_data:
                st.warning(f"Data for {index_name} not found. Skipping...")
                continue

            matching_rows = sectors_df[sectors_df['Index Name'] == index_name]
            company_name = matching_rows['Company Name'].iloc[0] if not matching_rows.empty else index_name

            col_header, col_button = st.columns([4, 1])
            with col_header:
                st.subheader(f"Processing {index_name} ({company_name})")
            with col_button:
                yahoo_url = f"https://finance.yahoo.com/chart/{index_name}"
                st.markdown(f"[ðŸ—  View Current Charts on Yahoo Finance]({yahoo_url})")

            result = get_predicted_values(
                daily_data[index_name],
                epochs=epochs_input,
                start_date=start_date,
                end_date=end_date
            )
            if result is None:
                st.warning(f"Not enough data for {index_name}. Skipping...")
                continue

            (
                train_r2, test_r2, future_preds, future_high, future_low,
                train_dates, y_train_actual, train_pred,
                test_dates, y_test_actual, test_pred
            ) = result

            train_plot_df = pd.DataFrame({
                'Date': pd.to_datetime(train_dates),
                'Actual': y_train_actual.flatten(),
                'Predicted': train_pred.flatten()
            })
            test_plot_df = pd.DataFrame({
                'Date': pd.to_datetime(test_dates),
                'Actual': y_test_actual.flatten(),
                'Predicted': test_pred.flatten()
            })
            last_date_in_data = pd.to_datetime(daily_data[index_name]['Date']).max()
            future_dates = pd.date_range(last_date_in_data + pd.Timedelta(days=1), periods=5, freq='B')[:5]
            future_plot_df = pd.DataFrame({
                'Date': future_dates,
                'Predicted Close': future_preds,
                'Predicted High': future_high,
                'Predicted Low': future_low
            })

            fig_train = go.Figure()
            fig_train.add_trace(go.Scatter(x=train_plot_df['Date'], y=train_plot_df['Actual'], mode='lines', name='Actual'))
            fig_train.add_trace(go.Scatter(x=train_plot_df['Date'], y=train_plot_df['Predicted'], mode='lines', name='Predicted'))
            fig_train.update_layout(title=f"Training Data for {index_name} (RÂ² = {train_r2:.4f})",
                                    dragmode="pan", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig_train, use_container_width=True)

            fig_test = go.Figure()
            fig_test.add_trace(go.Scatter(x=test_plot_df['Date'], y=test_plot_df['Actual'], mode='lines', name='Actual'))
            fig_test.add_trace(go.Scatter(x=test_plot_df['Date'], y=test_plot_df['Predicted'], mode='lines', name='Predicted'))
            fig_test.update_layout(title=f"Testing Data for {index_name} (RÂ² = {test_r2:.4f})",
                                   dragmode="pan", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig_test, use_container_width=True)

            fig_future = go.Figure()
            fig_future.add_trace(go.Scatter(x=future_plot_df['Date'], y=future_plot_df['Predicted Close'], mode='lines+markers', name='Predicted Close'))
            fig_future.add_trace(go.Scatter(x=future_plot_df['Date'], y=future_plot_df['Predicted High'], mode='lines+markers', name='Predicted High'))
            fig_future.add_trace(go.Scatter(x=future_plot_df['Date'], y=future_plot_df['Predicted Low'], mode='lines+markers', name='Predicted Low'))
            fig_future.update_layout(title=f"Future Predictions for {index_name}", dragmode="pan", xaxis_title="Date", yaxis_title="Price")
            
            if analysis_mode == "Single Stock Analysis":
                st.plotly_chart(fig_future, use_container_width=True)
                st.markdown("#### Next 5 Days Predictions")
                st.table(future_plot_df)
            else:
                combined_fig.add_trace(go.Scatter(x=future_plot_df['Date'], y=future_plot_df['Predicted Close'], mode='lines+markers', name=index_name))
                with st.expander(f"Next 5 Days Predictions for {index_name}"):
                    st.table(future_plot_df)

            results.append({
                'Run Date': current_date,
                'Index Name': index_name,
                'Company Name': company_name,
                'Model': 'LSTM',
                'Train R2 Score': train_r2,
                'Test R2 Score': test_r2,
                'Close Day 1': future_preds[0],
                'Close Day 2': future_preds[1],
                'Close Day 3': future_preds[2],
                'Close Day 4': future_preds[3],
                'Close Day 5': future_preds[4],
                'High Day 1': future_high[0],
                'High Day 2': future_high[1],
                'High Day 3': future_high[2],
                'High Day 4': future_high[3],
                'High Day 5': future_high[4],
                'Low Day 1': future_low[0],
                'Low Day 2': future_low[1],
                'Low Day 3': future_low[2],
                'Low Day 4': future_low[3],
                'Low Day 5': future_low[4]
            })

        if analysis_mode == "Compare Multiple Stocks":
            combined_fig.update_layout(title="Comparison of Future Predicted Close Prices (Next 5 Days)", dragmode="pan", xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(combined_fig, use_container_width=True)

        if results:
            result_df = pd.DataFrame(results)
            st.subheader("Prediction Results Summary")
            st.dataframe(result_df.style.format({
                'Train R2 Score': '{:.4f}',
                'Test R2 Score': '{:.4f}',
                'Close Day 1': '{:.2f}',
                'Close Day 2': '{:.2f}',
                'Close Day 3': '{:.2f}',
                'Close Day 4': '{:.2f}',
                'Close Day 5': '{:.2f}',
                'High Day 1': '{:.2f}',
                'High Day 2': '{:.2f}',
                'High Day 3': '{:.2f}',
                'High Day 4': '{:.2f}',
                'High Day 5': '{:.2f}',
                'Low Day 1': '{:.2f}',
                'Low Day 2': '{:.2f}',
                'Low Day 3': '{:.2f}',
                'Low Day 4': '{:.2f}',
                'Low Day 5': '{:.2f}'
            }))
            result_df['Verdict'] = result_df['Test R2 Score'].apply(
                lambda x: "Strong Forecast" if x >= 0.9 else ("Moderate Forecast" if x >= 0.8 else "Weak Forecast")
            )
            verdict_df = result_df[['Index Name', 'Company Name', 'Test R2 Score', 'Verdict']]
            st.subheader("Company Verdict")
            st.dataframe(verdict_df.style.format({'Test R2 Score': '{:.4f}'}))

        else:
            st.warning("No valid data found for prediction.")
        st.session_state.lstm_output = result_df

# ----------- Tab 3: Single Stock Analysis -----------
with tab3:
    st.title("Single Stock Analysis")
    st.markdown(
        """
        This section performs an end-to-end analysis on a **single stock** using:
        - **LSTM Prediction:** Predicts future prices using a preconfigured model (170 epochs by default, batch size 33).
        - **Candlestick Pattern Recognition:** Detects common candlestick patterns.
        """
    )
    
    # 1. Load the NSE stocks Excel file and create a searchable dropdown.
    try:
        stocks_file_path = os.path.join(BASE_DIR, "NSE-stocks.xlsx")
        stocks_df = pd.read_excel(stocks_file_path)
        stocks_df['Display'] = stocks_df['Company'] + " (" + stocks_df['Ticker'] + ")"
        selected_option = st.selectbox(
            "Select a Stock",
            options=stocks_df['Display'].tolist(),
            help="Start typing to search for your company."
        )
        selected_ticker = selected_option.split("(")[-1].replace(")", "").strip()
        st.session_state.user_stock = selected_ticker
    except Exception as e:
        st.error(f"Error loading NSE stocks file: {e}")
        st.stop()
    
    st.markdown("---")
    
    # 2. Run Analysis button - using default LSTM settings.
    if st.button("Run Analysis"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_steps = 3
        current_step = 0

        # Step 1: Acquire OHLCV Data
        status_text.markdown("**Step 1/3: Acquiring OHLCV Data...**")
        fixed_start_date = "2020-01-01"
        end_date_str = dt.date.today().strftime("%Y-%m-%d")
        try:
            ticker_obj = yf.Ticker(selected_ticker)
            try:
                company_info = ticker_obj.info
                company_name = company_info.get('longName', selected_ticker)
            except Exception:
                company_name = selected_ticker
        except Exception as e:
            st.error(f"Error fetching ticker info: {e}")
            st.stop()
        
        symbol_filename = company_name.replace(" ", "_")
        daily_data = yf.download(selected_ticker, start=fixed_start_date, end=end_date_str, interval='1d')
        if daily_data.empty:
            st.error("No daily data found. Please check the ticker and try again.")
            st.stop()
        file_daily = os.path.join(DAILY_DIR, f"{symbol_filename}.csv")
        daily_data.reset_index(inplace=True)
        daily_data.columns = [col[0] if isinstance(col, tuple) else col for col in daily_data.columns]
        for col in daily_data.columns[1:]:
            daily_data[col] = pd.to_numeric(daily_data[col], errors='coerce')
        daily_data.dropna(inplace=True)
        daily_data.to_csv(file_daily, index=False)
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        
        # Step 2: Run LSTM Prediction (170 epochs, batch size 33)
        status_text.markdown("**Step 2/3: Running LSTM Prediction...**")
        if os.path.exists(file_daily):
            data_daily = pd.read_csv(file_daily)
        else:
            data_daily = pd.DataFrame()
        
        if data_daily.empty:
            st.error("No valid daily data available for LSTM prediction.")
            st.stop()
        result = get_predicted_values(
            data_daily,
            epochs=170,
            batch_size=33,
            start_date=fixed_start_date,
            end_date=end_date_str
        )
        if result is None:
            st.error("Not enough data for LSTM analysis.")
            st.stop()
        (
            train_r2, test_r2, future_preds, future_high, future_low,
            train_dates, y_train_actual, train_pred,
            test_dates, y_test_actual, test_pred
        ) = result
        
        st.success(f"LSTM Model - Train RÂ²: {train_r2:.4f}, Test RÂ²: {test_r2:.4f}")
        train_plot_df = pd.DataFrame({
            'Date': pd.to_datetime(train_dates),
            'Actual': y_train_actual.flatten(),
            'Predicted': train_pred.flatten()
        })
        fig_train = go.Figure()
        fig_train.add_trace(go.Scatter(x=train_plot_df['Date'], y=train_plot_df['Actual'], mode='lines', name='Actual'))
        fig_train.add_trace(go.Scatter(x=train_plot_df['Date'], y=train_plot_df['Predicted'], mode='lines', name='Predicted'))
        fig_train.update_layout(title=f"Training Predictions for {company_name} (RÂ²={train_r2:.4f})", xaxis_title="Date", yaxis_title="Price", dragmode="pan")
        st.plotly_chart(fig_train, use_container_width=True)
        
        test_plot_df = pd.DataFrame({
            'Date': pd.to_datetime(test_dates),
            'Actual': y_test_actual.flatten(),
            'Predicted': test_pred.flatten()
        })
        fig_test = go.Figure()
        fig_test.add_trace(go.Scatter(x=test_plot_df['Date'], y=test_plot_df['Actual'], mode='lines', name='Actual'))
        fig_test.add_trace(go.Scatter(x=test_plot_df['Date'], y=test_plot_df['Predicted'], mode='lines', name='Predicted'))
        fig_test.update_layout(title=f"Testing Predictions for {company_name} (RÂ²={test_r2:.4f})", xaxis_title="Date", yaxis_title="Price", dragmode="pan")
        st.plotly_chart(fig_test, use_container_width=True)
        
        last_date = pd.to_datetime(daily_data['Date']).max()
        future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=5, freq='B')[:5]
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Close': future_preds,
            'Predicted High': future_high,
            'Predicted Low': future_low
        })
        fig_future = go.Figure()
        fig_future.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted Close'], mode='lines+markers', name='Predicted Close'))
        fig_future.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted High'], mode='lines+markers', name='Predicted High'))
        fig_future.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Predicted Low'], mode='lines+markers', name='Predicted Low'))
        fig_future.update_layout(title=f"Future Predictions (Next 5 Days) for {company_name}", xaxis_title="Date", yaxis_title="Price", dragmode="pan")
        st.plotly_chart(fig_future, use_container_width=True)
        st.markdown("**Future Predictions Table**")
        st.dataframe(future_df.reset_index(drop=True))
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        
        # Step 3: Candlestick Pattern Recognition
        status_text.markdown("**Step 3/3: Running Candlestick Pattern Recognition...**")
        data_daily['Date'] = pd.to_datetime(data_daily['Date'])
        patterns_df = detect_candlestick_patterns(data_daily)
        fig_patterns = plot_candlestick_with_patterns(patterns_df)
        st.plotly_chart(fig_patterns, use_container_width=True)
        current_step += 1
        progress_bar.progress(current_step / total_steps)
        st.markdown(
    """
    **Chart Annotations:**
    - **Green Marker (Triangle-Up):** Hammer â€“ A small candle with a long shadow at the bottom. It hints that prices might start going up.
    - **Blue Marker (Diamond):** Doji â€“ A candle where the open and close are almost the same, showing uncertainty in the market.
    - **Lime Marker (Star):** Bullish Engulfing â€“ A strong green candle that completely covers the previous red candle, suggesting a possible upward move.
    - **Red Marker (Star):** Bearish Engulfing â€“ A strong red candle that completely covers the previous green candle, suggesting a potential drop in price.
    """
)

        
        progress_bar.empty()
        status_text.markdown("**Analysis Completed!**")
        st.success("Single Stock Analysis Completed!")

# ----------- Tab 4: Fundamental Analysis Tools -----------
with tab4:
    st.title("Fundamental Analysis Tools")
    symbol = st.session_state.get("user_stock", "AAPL")
    st.markdown(f"Below is an embedded view of TradingView Financials widget for **{symbol}**. You can interact with the page without being redirected.")
    
    if symbol.endswith(".NS"):
        tv_symbol = "NSE:" + symbol.replace(".NS", "")
    else:
        tv_symbol = symbol

    params = {"symbol": tv_symbol, "colorTheme": "light", "isTransparent": False}
    encoded_params = urllib.parse.quote(json.dumps(params))
    iframe_html = f"""
    <iframe src="https://s.tradingview.com/embed-widget/financials/?locale=en#{encoded_params}" width="100%" height="600" frameborder="0"></iframe>
    """
    st.components.v1.html(iframe_html, height=600)

    st.markdown("---")
    st.header("Expanded Financial Statements")

    ticker_obj = yf.Ticker(symbol)
    try:
        quarterly_income = ticker_obj.quarterly_financials
        quarterly_balance = ticker_obj.quarterly_balance_sheet
        quarterly_cashflow = ticker_obj.quarterly_cashflow

        with st.expander("Quarterly Income Statement"):
            if not quarterly_income.empty:
                st.dataframe(quarterly_income.style.format("{:,.0f}"))
                st.markdown("**Income Statement Percentage Change (QoQ):**")
                pct_change_income = quarterly_income.pct_change(axis=1) * 100
                st.dataframe(pct_change_income.style.format("{:.2f}%"))
            else:
                st.info("Quarterly Income Statement data not available.")

        with st.expander("Quarterly Balance Sheet"):
            if not quarterly_balance.empty:
                st.dataframe(quarterly_balance.style.format("{:,.0f}"))
                st.markdown("**Balance Sheet Percentage Change (QoQ):**")
                pct_change_balance = quarterly_balance.pct_change(axis=1) * 100
                st.dataframe(pct_change_balance.style.format("{:.2f}%"))
            else:
                st.info("Quarterly Balance Sheet data not available.")

        with st.expander("Quarterly Cash Flow Statement"):
            if not quarterly_cashflow.empty:
                st.dataframe(quarterly_cashflow.style.format("{:,.0f}"))
                st.markdown("**Cash Flow Statement Percentage Change (QoQ):**")
                pct_change_cashflow = quarterly_cashflow.pct_change(axis=1) * 100
                st.dataframe(pct_change_cashflow.style.format("{:.2f}%"))
            else:
                st.info("Quarterly Cash Flow Statement data not available.")
    except Exception as e:
        st.error(f"Error fetching expanded financial statements: {e}")

    st.markdown("---")
    st.header("Key Financial Ratios")
    st.markdown("Below are some of the key financial ratios to quickly assess the company's valuation and financial health.")

    # Initialize info to ensure it is defined in the event of an error.
    info = None
    try:
        info = ticker_obj.info
        ratios = {
            "Trailing P/E": info.get("trailingPE", "N/A"),
            "Forward P/E": info.get("forwardPE", "N/A"),
            "Price/Book": info.get("priceToBook", "N/A"),
            "Return on Equity": info.get("returnOnEquity", "N/A"),
            "Dividend Yield": info.get("dividendYield", "N/A"),
            "Debt-to-Equity": info.get("debtToEquity", "N/A")
        }
        ratios_df = pd.DataFrame(list(ratios.items()), columns=["Ratio", "Value"])
        st.dataframe(ratios_df)
    except Exception as e:
        st.error(f"Error fetching key financial ratios: {e}")

    st.markdown("---")
    st.header("Financial Verdict")

    score = 50
    verdict_parts = []

    if not quarterly_income.empty:
        income_pct_change = quarterly_income.pct_change(axis=1).dropna(axis=1)
        if not income_pct_change.empty:
            avg_growth = income_pct_change.mean().mean() * 100
            if avg_growth > 5:
                verdict_parts.append(f"Income shows healthy growth with an average quarterly increase of {avg_growth:.2f}%.")
                score += 20
            elif avg_growth < 0:
                verdict_parts.append(f"Income is declining with an average quarterly decrease of {avg_growth:.2f}%.")
                score -= 20
            else:
                verdict_parts.append(f"Income is relatively stable with little change quarter-over-quarter (avg: {avg_growth:.2f}%).")
    else:
        verdict_parts.append("Income statement data is not available.")

    if info is not None:
        pe = info.get("trailingPE")
        roe = info.get("returnOnEquity")
        if pe is not None:
            if pe < 25:
                verdict_parts.append("The Price-to-Earnings ratio is low, which may indicate attractive valuation.")
                score += 10
            else:
                verdict_parts.append("The Price-to-Earnings ratio is high, which could suggest the stock is overvalued.")
                score -= 10
        if roe is not None:
            if roe > 0.15:
                verdict_parts.append("Return on Equity is high, showing good efficiency in generating returns.")
                score += 10
            else:
                verdict_parts.append("Return on Equity is moderate, suggesting average performance.")
                score -= 5
    else:
        verdict_parts.append("Financial ratios are not available for deeper analysis.")

    score = max(0, min(100, score))

    st.markdown(
        f"""
        <div style="background: #e0e0e0; border-radius: 20px; overflow: hidden; height: 25px; margin-bottom: 15px;">
            <div style="width: {score}%; height: 100%; background: linear-gradient(90deg, red, orange, green); text-align: center; line-height: 25px; color: white;">
                <strong>{score:.0f}%</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    final_statement = " ".join(verdict_parts)
    st.markdown(f"<div style='font-size: 20px; font-weight: bold; color: #333;'>{final_statement}</div>", unsafe_allow_html=True)
