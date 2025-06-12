# Main Python script
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

# Additional Imports for News, Sentiment & Candlestick Analysis
import requests
import random
import time
from bs4 import BeautifulSoup
import nltk
from transformers import pipeline
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Download required lexicons (FinBERT is used; VADER no longer used)
# This line is crucial and should remain as it handles NLTK data download
try:
    nltk.data.find('corpora/vader_lexicon.zip')
except nltk.downloader.DownloadError:
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

# File Paths & Directories - CRITICAL CHANGE FOR DEPLOYMENT
# This makes BASE_DIR relative to where your app runs on Render.
# Render will create these directories if they don't exist.
BASE_DIR = os.path.join(os.getcwd(), "DATASETS")
DAILY_DIR = os.path.join(BASE_DIR, "Daily_data")
WEEKLY_DIR = os.path.join(BASE_DIR, "Weekly_data")
MONTHLY_DIR = os.path.join(BASE_DIR, "Monthly_data")

# These files are assumed to be in your Git repository at the root level,
# or specifically placed in the DATASETS folder you create in your repo.
# For simpler deployment, include them directly in your Git repo at the same level as Main.py
# or within a 'DATASETS' subfolder if you explicitly create that in your repo.
# If they are not present, functionalities relying on them will show warnings.
INDICES_STOCKS_FILE = os.path.join(BASE_DIR, "indicesstocks.csv")
NSE_STOCKS_FILE = os.path.join(BASE_DIR, "NSE-stocks.xlsx")
SECTORS_FILE = os.path.join(BASE_DIR, "sectors with symbols.csv") # Used by GFS Qualified Stocks


# --- Helper Functions (Keep these as they are from your original code) ---

# Function to append a row to a DataFrame (utility)
def append_row(df, row):
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)

# Function to download stock data
@st.cache_data(ttl=3600) # Cache data for 1 hour
def download_stock_data(interval='1d', folder='Daily_data', custom_start_date=None, custom_end_date=None):
    # Ensure all necessary directories exist before downloading
    os.makedirs(os.path.join(BASE_DIR, folder), exist_ok=True)
    
    # Attempt to read indicesstocks.csv for symbols
    symbols_to_download = []
    if os.path.exists(INDICES_STOCKS_FILE):
        try:
            with open(INDICES_STOCKS_FILE, 'r') as file:
                for line in file:
                    parts = line.strip().split(',')
                    if parts:
                        symbols_to_download.append(parts[0].strip())
        except Exception as e:
            st.error(f"Error reading {INDICES_STOCKS_FILE}: {e}")
            return
    else:
        st.warning(f"'{INDICES_STOCKS_FILE}' not found. Cannot download data for pre-defined indices/stocks.")
        st.info("Please manually enter symbols for analysis if you wish to proceed without this file.")
        return # Exit if the critical file is missing

    if not symbols_to_download:
        st.warning("No symbols found in 'indicesstocks.csv' to download.")
        return
        
    session = requests.Session()
    retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    for symbol in symbols_to_download:
        try:
            file_path = os.path.join(BASE_DIR, folder, f"{symbol}.csv")
            data = yf.download(symbol, start=custom_start_date, end=custom_end_date, interval=interval, session=session)
            if not data.empty:
                data.to_csv(file_path)
                # st.success(f"Downloaded {symbol} data to {file_path}") # Too verbose for production
            else:
                st.warning(f"No data found for {symbol} with interval {interval}. Skipping.")
        except Exception as e:
            st.error(f"Error downloading {symbol} data: {e}")

# Function to read stock data from a file
def read_stock_data(filepath):
    if not os.path.exists(filepath):
        st.error(f"Error: Data file not found at {filepath}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        return df
    except Exception as e:
        st.error(f"Error reading {filepath}: {e}")
        return pd.DataFrame()

# Function to calculate technical indicators (RSI, Bollinger Bands)
def calculate_indicators(df):
    if df.empty:
        return df
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['BB_LOWER'], df['BB_MIDDLE'], df['BB_UPPER'] = ta.bbands(df['Close'], length=20, std=2)
    return df

# Function to get latest data points for a symbol
def get_latest_data(symbol, interval_folder):
    filepath = os.path.join(BASE_DIR, interval_folder, f"{symbol}.csv")
    df = read_stock_data(filepath)
    if df.empty:
        return None, None, None, None, None, None, None
    latest = df.iloc[-1]
    return (
        latest['RSI'] if 'RSI' in latest else None,
        latest['BB_LOWER'] if 'BB_LOWER' in latest else None,
        latest['BB_MIDDLE'] if 'BB_MIDDLE' in latest else None,
        df['Close'].iloc[-1] if 'Close' in df.columns else None,
        latest.name.strftime('%Y-%m-%d') if latest.name else None, # latest date
        latest['Open'] if 'Open' in latest else None,
        latest['Volume'] if 'Volume' in latest else None
    )

# Function to calculate day, week, month datasets
def dayweekmonth_datasets(symbol, symbolname, indexcode):
    day_rsi, day_lower, day_middle, day_ltp, day_date, _, _ = get_latest_data(symbol, "Daily_data")
    week_rsi, week_lower, week_middle, _, _, _, _ = get_latest_data(symbol, "Weekly_data")
    month_rsi, month_lower, month_middle, _, _, _, _ = get_latest_data(symbol, "Monthly_data")

    return {
        'entrydate': dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'indexcode': indexcode,
        'indexname': symbolname,
        'dayrsi14': day_rsi,
        'weekrsi14': week_rsi,
        'monthrsi14': month_rsi,
        'dltp': day_ltp,
        'daylowerband': day_lower,
        'daymiddleband': day_middle,
        'weeklowerband': week_lower,
        'weekmiddleband': week_middle,
        'monthlowerband': month_lower,
        'monthmiddleband': month_middle
    }

# Function to fetch VIX
@st.cache_data(ttl=600) # Cache VIX for 10 minutes
def fetch_vix():
    try:
        vix = yf.Ticker("^VIX").history(period="1d")
        if not vix.empty:
            return vix['Close'].iloc[-1]
        return None
    except Exception as e:
        st.error(f"Error fetching VIX: {e}")
        return None

# GFS Analysis Helper: Read indicesstocks.csv
def read_indicesstocks(file_path):
    indices_dict = {}
    if not Path(file_path).is_file():
        st.warning(f"File not found: {file_path}. GFS analysis might be limited.")
        return indices_dict
    try:
        with open(file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(',')
                if len(parts) > 1:
                    index_code = parts[0].strip()
                    indices_dict[index_code] = [stock.strip() for stock in parts[1:]]
    except Exception as e:
        st.error(f"Error reading {file_path}: {e}")
    return indices_dict

# GFS Analysis Helper: Generate GFS report
def generateGFS(scripttype_file_name):
    # This function expects a file like 'indicesdf.csv' or 'stocksdf.csv' in BASE_DIR
    indicesdf = pd.DataFrame(columns=[
        'entrydate', 'indexcode', 'indexname', 'dayrsi14', 
        'weekrsi14', 'monthrsi14', 'dltp', 'daylowerband', 
        'daymiddleband', 'weeklowerband', 'weekmiddleband', 
        'monthlowerband', 'monthmiddleband'
    ])
    filepath = os.path.join(BASE_DIR, f"{scripttype_file_name}.csv") # Example: indicesdf.csv
    
    # If the file for GFS generation (e.g., indicesdf.csv) isn't present,
    # we cannot generate the report based on it.
    if not Path(filepath).is_file():
        st.warning(f"Required file for GFS: {filepath} not found. Skipping GFS generation.")
        return indicesdf
    
    try:
        with open(filepath) as f:
            for line in f:
                if "," not in line:
                    continue
                symbol, symbolname = line.split(",")[0], line.split(",")[1]
                new_row = dayweekmonth_datasets(symbol.strip(), symbolname.strip(), symbol.strip())
                indicesdf = append_row(indicesdf, new_row)
    except Exception as e:
        st.error(f"Error generating GFS report from {filepath}: {e}")
    return indicesdf

# --- LSTM Prediction Functions (Keep as is) ---
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

@st.cache_resource # Cache the loaded model
def load_lstm_model(model_path="lstm_model.h5"):
    if not os.path.exists(model_path):
        st.error(f"LSTM model not found at {model_path}. Please train the model first.")
        return None
    from tensorflow.keras.models import load_model
    model = load_model(model_path)
    return model

def train_lstm_model(df):
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    train_size = int(len(data_scaled) * 0.8)
    train_data = data_scaled[0:train_size, :]
    test_data = data_scaled[train_size:len(data_scaled), :]

    look_back = 60
    X_train, y_train = create_dataset(train_data, look_back)
    X_test, y_test = create_dataset(test_data, look_back)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=1, epochs=1)

    # Save the trained model
    model.save("lstm_model.h5")
    return model, scaler, X_test, y_test, data_scaled

# --- News Scraping & Sentiment Analysis Functions (Keep as is) ---
@st.cache_resource # Cache the FinBERT model
def load_finbert_model():
    return pipeline('sentiment-analysis', model='ProsusAI/finbert')

# Modified to pass the FinBERT model
def get_sentiment(text, finbert_model):
    if not text:
        return "Neutral", 0.0, 0.0, 0.0 # Return default values for empty text
    try:
        result = finbert_model(text)
        sentiment_label = result[0]['label']
        sentiment_score = result[0]['score']

        # FinBERT outputs 'positive', 'negative', 'neutral'
        if sentiment_label == 'positive':
            return "Positive", sentiment_score, 0.0, 0.0
        elif sentiment_label == 'negative':
            return "Negative", 0.0, sentiment_score, 0.0
        else: # neutral
            return "Neutral", 0.0, 0.0, sentiment_score
    except Exception as e:
        st.error(f"Error during sentiment analysis: {e}")
        return "Error", 0.0, 0.0, 0.0

def scrape_google_news(query, num_articles=5):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://news.google.com/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
    
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)

    articles = []
    try:
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Google News structure often uses <a> tags with specific attributes or classes
        # This selector might need adjustment if Google News changes its HTML structure
        article_elements = soup.find_all('a', class_='JtCrunchNewsTitleLink')[:num_articles] # Example selector
        if not article_elements: # Fallback for different structures
            article_elements = soup.find_all('a', class_='VDXfz')[:num_articles] # Another common class
        if not article_elements:
             article_elements = soup.find_all('a', class_='DY5T1d RgnpHc')[:num_articles] # Yet another common class
        
        for article in article_elements:
            title = article.text.strip()
            link = "https://news.google.com" + article['href'].lstrip('.')
            articles.append({'title': title, 'link': link})
            
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching news for '{query}': {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred during news scraping: {e}")
    
    return articles

def get_news_from_newsapi(company_name, api_key):
    # This is a placeholder. You'll need to sign up for NewsAPI.org and get an API key.
    if not api_key:
        st.warning("NewsAPI.org API Key is not set. Skipping NewsAPI fetching.")
        return []

    url = f"https://newsapi.org/v2/everything?q={urllib.parse.quote_plus(company_name)}&apiKey={api_key}&language=en&sortBy=relevancy&pageSize=5"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        articles = []
        for article in data.get('articles', []):
            articles.append({
                'title': article.get('title'),
                'link': article.get('url')
            })
        return articles
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching news from NewsAPI for '{company_name}': {e}")
        return []
    except json.JSONDecodeError:
        st.error(f"Could not decode JSON response from NewsAPI for '{company_name}'.")
        return []

def update_filtered_indices_by_sentiment(filtered_file_path, sentiment_method="FinBERT", use_newsapi=False, aggregate_sources=False):
    if not os.path.exists(filtered_file_path):
        st.error(f"Filtered indices file not found: {filtered_file_path}")
        return {}, pd.DataFrame(), []

    filtered_df = pd.read_csv(filtered_file_path)
    if filtered_df.empty:
        return {}, pd.DataFrame(), []

    finbert_model = None
    if sentiment_method == "FinBERT":
        finbert_model = load_finbert_model()

    sentiment_summary = {}
    all_news_data = []
    
    # Placeholder for NEWS_API_KEY. Replace with your actual NewsAPI key if using.
    # For deployment, consider using Streamlit Secrets or Render Environment Variables for this.
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"] if "NEWS_API_KEY" in st.secrets else os.environ.get("NEWS_API_KEY", "")


    for index, row in filtered_df.iterrows():
        company_name = row['Company Name']
        index_code = row['indexcode'] # Use indexcode for news search if company name is 'N/A' or problematic
        search_query = company_name if company_name != 'N/A' else index_code

        news_articles = []
        if use_newsapi:
            news_articles.extend(get_news_from_newsapi(search_query, NEWS_API_KEY))
        
        if aggregate_sources or not use_newsapi: # Scrape Google News if aggregating or not using NewsAPI
            news_articles.extend(scrape_google_news(f"{search_query} stock news"))

        if not news_articles:
            # If no news articles were found from either source for this company
            st.warning(f"No news articles found for {company_name} ({index_code}).")
            sentiment_summary[company_name] = {'Positive': 0, 'Negative': 0, 'Neutral': 0, 'Total Articles': 0}
            continue

        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        company_news_data = [] # To store news articles for current company

        for article in news_articles:
            title = article.get('title')
            link = article.get('link')

            if title:
                sentiment, pos_score, neg_score, neu_score = get_sentiment(title, finbert_model)
                if sentiment == "Positive":
                    positive_count += 1
                elif sentiment == "Negative":
                    negative_count += 1
                else:
                    neutral_count += 1
                
                company_news_data.append({
                    'Company': company_name,
                    'Title': title,
                    'Link': link,
                    'Sentiment': sentiment,
                    'Positive Score': pos_score,
                    'Negative Score': neg_score,
                    'Neutral Score': neu_score
                })
        
        all_news_data.extend(company_news_data)

        sentiment_summary[company_name] = {
            'Positive': positive_count,
            'Negative': negative_count,
            'Neutral': neutral_count,
            'Total Articles': len(news_articles)
        }

    return sentiment_summary, filtered_df, all_news_data

# --- Candlestick Pattern Recognition Functions (Keep as is) ---
def recognize_candlestick_patterns(df):
    if df.empty:
        return df
    
    # Initialize all pattern columns to False
    patterns = [
        'CDLDOJI', 'CDLHAMMER', 'CDLINVERTEDHAMMER', 'CDLDRAGONFLYDOJI', 
        'CDLGRAVESTONEDOJI', 'CDLMORNINGSTAR', 'CDLEVENINGSTAR', 
        'CDLBULLISHENGULFING', 'CDLBEARISHENGULFING', 'CDLPIERCING', 
        'CDLDARKCLOUDCOVER', 'CDLHARAMI', 'CDLTWEEZERTOP', 'CDLTWEEZERBOTTOM',
        'CDLTHREEWHITESOLDIERS', 'CDLTHREECROWS'
    ]
    for pattern in patterns:
        df[pattern] = 0 # Initialize with 0

    # Calculate patterns using pandas_ta
    df.ta.cdl_pattern(append=True) # This calculates all recognized patterns

    # Filter out patterns with value 0 (no pattern) and map positive/negative
    recognized_patterns = []
    for index, row in df.iterrows():
        for pattern_col in [col for col in df.columns if col.startswith('CDL') and col != 'CDL_PATTERN_SUM']:
            value = row[pattern_col]
            if value != 0:
                pattern_name = pattern_col.replace('CDL', '').replace('_', ' ').strip().title()
                strength = "Bullish" if value > 0 else "Bearish"
                recognized_patterns.append({
                    'Date': index.strftime('%Y-%m-%d'),
                    'Pattern': pattern_name,
                    'Strength': strength,
                    'Value': value
                })
    return pd.DataFrame(recognized_patterns)

# --- Stock Fundamentals (Keep as is) ---
@st.cache_data(ttl=3600) # Cache fundamentals for 1 hour
def get_stock_fundamentals(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        fundamentals = {
            "Symbol": info.get('symbol'),
            "Company Name": info.get('longName'),
            "Sector": info.get('sector'),
            "Industry": info.get('industry'),
            "Market Cap": info.get('marketCap'),
            "P/E Ratio": info.get('trailingPE'),
            "Forward P/E": info.get('forwardPE'),
            "EPS": info.get('trailingEps'),
            "Dividend Yield": info.get('dividendYield'),
            "Beta": info.get('beta'),
            "Return on Equity (ROE)": info.get('returnOnEquity'),
            "Return on Assets (ROA)": info.get('returnOnAssets'),
            "Debt to Equity": info.get('debtToEquity'),
            "Current Ratio": info.get('currentRatio'),
            "Gross Margins": info.get('grossMargins'),
            "Operating Margins": info.get('operatingMargins'),
            "Profit Margins": info.get('profitMargins'),
            "52 Week High": info.get('fiftyTwoWeekHigh'),
            "52 Week Low": info.get('fiftyTwoWeekLow'),
            "Average Volume (10d)": info.get('averageDailyVolume10Day'),
            "Currency": info.get('currency'),
            "Exchange": info.get('exchange'),
            "Country": info.get('country')
        }
        return fundamentals
    except Exception as e:
        st.error(f"Could not retrieve fundamentals for {ticker_symbol}: {e}")
        return None

# --- Session State Initialization ---
if 'vix_value' not in st.session_state:
    st.session_state.vix_value = None
if 'show_data_choice' not in st.session_state:
    st.session_state.show_data_choice = False
if 'data_choice' not in st.session_state:
    st.session_state.data_choice = None
if 'gfs_output' not in st.session_state:
    st.session_state.gfs_output = pd.DataFrame()
if 'news_articles_data' not in st.session_state:
    st.session_state.news_articles_data = []
if 'error_logged' not in st.session_state:
    st.session_state.error_logged = False # To prevent repeated error messages for NewsAPI key


# --- Streamlit App UI ---
st.title("Comprehensive Stock Analysis and Forecasting")

tab1, tab2, tab3 = st.tabs(["GFS Analysis & News", "Single Stock Analysis & LSTM", "Candlestick Patterns & Fundamentals"])

# ----------- Tab 1: GFS Analysis & News Sentiment Analysis -----------
with tab1:
    st.header("GFS Analysis")
    st.markdown(
        """
        **Overview:** This section downloads stock data (Daily/Weekly/Monthly) for symbols listed in `indicesstocks.csv` 
        (if provided in your repository), calculates technical indicators (RSI, Bollinger Bands), and filters stocks based on multi-timeframe criteria.
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
                st.warning(f"**High Volatility Detected (VIX: {vix_value:.2f})** \nMarket conditions are volatile. Proceed with caution.")
            else:
                st.success(f"Market Volatility Normal (VIX: {vix_value:.2f})")

    if st.session_state.get('show_data_choice', False):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Download & Update Data (from Yahoo Finance)"):
                st.session_state.data_choice = 'update'
        with col2:
            st.info("This will download fresh data from Yahoo Finance for symbols in `indicesstocks.csv`.")
            
        if 'data_choice' in st.session_state and st.session_state.data_choice == 'update':
            update_option = st.selectbox("Select dataset update option", options=["Use Current Date", "Select Custom End Date"])
            fixed_start_date = "2020-01-01"
            selected_end_date_str = None
            if update_option == "Select Custom End Date":
                custom_end_date = st.date_input("Select Custom End Date", value=dt.date.today(), max_value=dt.date.today())
                selected_end_date_str = custom_end_date.strftime("%Y-%m-%d")

            if st.button("Proceed to Download and Generate GFS Reports"):
                # Ensure the directories for downloaded data exist
                os.makedirs(DAILY_DIR, exist_ok=True)
                os.makedirs(WEEKLY_DIR, exist_ok=True)
                os.makedirs(MONTHLY_DIR, exist_ok=True)

                # Check if indicesstocks.csv is present before attempting to download
                if not os.path.exists(INDICES_STOCKS_FILE):
                    st.error(f"Required file '{INDICES_STOCKS_FILE}' not found. Please provide it in your repository's 'DATASETS' folder to proceed with data download and GFS analysis.")
                else:
                    with st.spinner("Downloading and updating data..."):
                        # Download data for all symbols listed in indicesstocks.csv
                        download_stock_data(interval='1d', folder='Daily_data', custom_start_date=fixed_start_date, custom_end_date=selected_end_date_str)
                        download_stock_data(interval='1wk', folder='Weekly_data', custom_start_date=fixed_start_date, custom_end_date=selected_end_date_str)
                        download_stock_data(interval='1mo', folder='Monthly_data', custom_start_date=fixed_start_date, custom_end_date=selected_end_date_str)
                    
                    # Generate GFS reports based on the newly downloaded data
                    # Note: generateGFS expects a 'scripttype' which refers to a CSV file name (e.g., 'indicesdf')
                    # If you don't have a pre-existing 'indicesdf.csv' that lists all indices,
                    # this part will generate an empty DataFrame.
                    # You might need to adjust 'generateGFS' or ensure 'indicesdf.csv' is populated
                    # dynamically or included in your repo.
                    
                    st.markdown("### Generating GFS Reports...")
                    
                    # For a truly dynamic GFS, you might iterate through the symbols that were actually downloaded
                    # instead of relying on a pre-existing 'indicesdf.csv' that may not be updated.
                    # As per your original code, it expects `indicesdf.csv` to exist for `generateGFS('indicesdf')`.
                    # If you want to use the symbols from `indicesstocks.csv` for *both* download AND GFS generation,
                    # you'd need to loop through them here and call `dayweekmonth_datasets` for each.
                    # For now, let's assume `indicesdf.csv` should exist if you want this part to work as originally intended.
                    
                    # --- Logic to generate indicesdf.csv if it doesn't exist or is empty ---
                    # This is a hypothetical way to populate `indicesdf.csv` if it's missing or if you want to regenerate it.
                    # This section relies on 'indicesstocks.csv' which contains the main index codes.
                    temp_indices_df_path = os.path.join(BASE_DIR, "indicesdf.csv")
                    if os.path.exists(INDICES_STOCKS_FILE) and not os.path.exists(temp_indices_df_path):
                        st.info("Creating a temporary 'indicesdf.csv' for GFS analysis based on 'indicesstocks.csv'...")
                        try:
                            # Read the first column of indicesstocks.csv as index codes
                            temp_df_content = pd.read_csv(INDICES_STOCKS_FILE, header=None, usecols=[0], names=['IndexCode'])
                            temp_df_content['IndexName'] = temp_df_content['IndexCode'] # Use IndexCode as IndexName for simplicity
                            temp_df_content.to_csv(temp_indices_df_path, index=False, header=False)
                            st.success("Temporary 'indicesdf.csv' created.")
                        except Exception as e:
                            st.error(f"Error creating temporary 'indicesdf.csv': {e}. GFS analysis might be affected.")

                    # Now call generateGFS, which will use the (potentially new) indicesdf.csv
                    df3 = generateGFS('indicesdf') # This function now checks if indicesdf.csv exists
                    
                    df4 = df3.loc[
                        df3['monthrsi14'].between(35, 60) &
                        df3['weekrsi14'].between(35, 60) &
                        df3['dayrsi14'].between(35, 60)
                    ]
                    st.markdown("### Qualified Indices")
                    if df4.empty:
                        st.warning("No indices meet GFS criteria or required data is missing.")
                    else:
                        st.dataframe(df4.style.format({
                            'dayrsi14': '{:.2f}',
                            'weekrsi14': '{:.2f}',
                            'monthrsi14': '{:.2f}',
                            'dltp': '{:.2f}'
                        }), use_container_width=True)
                        # Save filtered indices for potential later use (e.g., news analysis)
                        df4.to_csv(os.path.join(BASE_DIR, "filtered_indices.csv"), index=False)
                    
                    st.markdown("### Qualified Stocks")
                    
                    if not os.path.exists(INDICES_STOCKS_FILE):
                        st.warning("Skipping Qualified Stocks analysis: 'indicesstocks.csv' not found.")
                        st.session_state.gfs_output = pd.DataFrame() # Set to empty
                    elif not os.path.exists(SECTORS_FILE):
                         st.warning(f"Skipping Qualified Stocks analysis: '{SECTORS_FILE}' not found. Please include this file in your 'DATASETS' folder if you want full GFS stock analysis.")
                         st.session_state.gfs_output = pd.DataFrame()
                    else:
                        indices_dict = read_indicesstocks(INDICES_STOCKS_FILE)
                        results_df = pd.DataFrame(columns=df3.columns) # Use same columns as df3 for consistency
                        
                        if not df4.empty: # Only proceed if there are qualified indices
                            for index_code in df4['indexcode']:
                                if index_code in indices_dict:
                                    for stock_symbol in indices_dict[index_code]:
                                        if stock_symbol: # Ensure stock symbol is not empty
                                            new_row = dayweekmonth_datasets(stock_symbol, stock_symbol, index_code)
                                            results_df = append_row(results_df, new_row)
                            
                            results_df = results_df.loc[
                                (results_df['monthrsi14'].between(40, 60)) &
                                (results_df['weekrsi14'].between(40, 60)) &
                                (results_df['dayrsi14'].between(40, 60))
                            ]
                            
                            try:
                                sectors_df = pd.read_csv(SECTORS_FILE)
                                # Merge based on stock symbol, not index name as original code was doing
                                # This merge logic assumes 'Company Name' in sectors_df maps to the stock symbol
                                # If 'Index Name' was intended to link to the `indexname` in results_df (which is stock symbol)
                                # then 'Index Name' in sectors_df should contain stock symbols.
                                # Adjusting to merge on indexcode/indexname if that's the intention, assuming it holds stock symbols.
                                # Given the 'Company Name' column, it's more likely `sectors_df` has symbol -> company name.
                                results_df = results_df.merge(
                                    sectors_df[['Index Name', 'Company Name']], # Assuming 'Index Name' here refers to stock symbol or index code
                                    left_on='indexname', # This column in results_df holds the stock symbol/index code
                                    right_on='Index Name',
                                    how='left'
                                )
                                results_df.drop(columns=['Index Name'], inplace=True, errors='ignore')
                                results_df['Company Name'] = results_df['Company Name'].fillna('N/A')
                            except Exception as e:
                                st.warning(f"Could not merge with sectors data (Error: {e}). Ensure '{SECTORS_FILE}' format is correct.")
                                results_df['Company Name'] = 'N/A (Sector file issue)'

                            if results_df.empty:
                                st.warning("No stocks meet GFS criteria or required data is missing for stocks.")
                            else:
                                st.dataframe(results_df.style.format({
                                    'dayrsi14': '{:.2f}',
                                    'weekrsi14': '{:.2f}',
                                    'monthrsi14': '{:.2f}',
                                    'dltp': '{:.2f}'
                                }), use_container_width=True)
                                results_df.to_csv(os.path.join(BASE_DIR, "filtered_indices_output.csv"), index=False)
                        else:
                            st.warning("No qualified indices found, so no stocks could be analyzed for GFS.")

                    st.success("GFS Analysis completed!")
                    st.session_state.gfs_output = results_df
                    del st.session_state.data_choice
                    st.session_state.show_data_choice = False

    st.markdown("## News Sentiment Analysis")
    # This section relies on 'filtered_indices_output.csv' being created by GFS.
    filtered_file = os.path.join(BASE_DIR, "filtered_indices_output.csv")
    if os.path.exists(filtered_file) and not st.session_state.gfs_output.empty:
        st.markdown("**Using FinBERT for sentiment analysis.**")
        use_newsapi = st.checkbox("Use NewsAPI.org (requires API key)", value=False)
        aggregate_sources = st.checkbox("Aggregate Google News & NewsAPI", value=False)
        
        if use_newsapi and (("NEWS_API_KEY" not in st.secrets) and ("NEWS_API_KEY" not in os.environ)) and not st.session_state.error_logged:
            st.error("NEWS_API_KEY not found in Streamlit Secrets or environment variables. NewsAPI functionality will be skipped.")
            st.session_state.error_logged = True # Log once
        elif not use_newsapi:
            st.session_state.error_logged = False # Reset if unchecked

        if st.button("Run News Sentiment Analysis"):
            if use_newsapi and (("NEWS_API_KEY" not in st.secrets) and ("NEWS_API_KEY" not in os.environ)):
                 st.error("Cannot run NewsAPI analysis. NEWS_API_KEY is missing.")
            else:
                with st.spinner("Analyzing news sentiment for each company..."):
                    sentiment_summary, updated_df, news_articles_data = update_filtered_indices_by_sentiment(
                        filtered_file,
                        sentiment_method="FinBERT",
                        use_newsapi=use_newsapi,
                        aggregate_sources=aggregate_sources
                    )
                st.success("News Sentiment Analysis completed!")
                st.markdown("### Sentiment Summary")
                if sentiment_summary:
                    summary_df = pd.DataFrame.from_dict(sentiment_summary, orient='index')
                    st.dataframe(summary_df)
                else:
                    st.warning("No sentiment summary generated. Check GFS analysis results and data availability.")
                
                st.markdown("### Detailed News Articles and Sentiment")
                if news_articles_data:
                    news_df = pd.DataFrame(news_articles_data)
                    st.dataframe(news_df, use_container_width=True)
                    st.session_state.news_articles_data = news_df.to_dict('records') # Store for potential download
                    
                    csv_news = news_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download News Articles as CSV",
                        data=csv_news,
                        file_name="news_sentiment_analysis.csv",
                        mime="text/csv",
                    )

                else:
                    st.info("No news articles were retrieved for analysis.")
    else:
        st.info("Run GFS Analysis first to generate the filtered list of companies for news analysis.")

# ----------- Tab 2: Single Stock Analysis & LSTM -----------
with tab2:
    st.header("Single Stock Analysis")
    
    # Allow user to input a stock symbol
    stock_symbol = st.text_input("Enter Stock Symbol (e.g., 'AAPL', 'RELIANCE.NS')", value="AAPL").upper()
    
    # Attempt to load symbols from NSE-stocks.xlsx if it exists
    stock_list = []
    if os.path.exists(NSE_STOCKS_FILE):
        try:
            nse_stocks_df = pd.read_excel(NSE_STOCKS_FILE)
            if 'Symbol' in nse_stocks_df.columns:
                stock_list = nse_stocks_df['Symbol'].tolist()
                stock_symbol = st.selectbox("Or select from NSE stocks:", options=[''] + stock_list, index=0)
                if stock_symbol == '':
                    stock_symbol = "AAPL" # Default if nothing selected
            else:
                st.warning(f"'{NSE_STOCKS_FILE}' found but 'Symbol' column is missing. Please check the file format.")
        except Exception as e:
            st.error(f"Error reading '{NSE_STOCKS_FILE}': {e}. Please ensure it's a valid Excel file.")
    else:
        st.warning(f"'{NSE_STOCKS_FILE}' not found. Cannot populate stock dropdown. Please manually enter a symbol.")
    
    if stock_symbol:
        st.subheader(f"Analyzing: {stock_symbol}")
        
        # Fetch fundamental data
        fundamentals = get_stock_fundamentals(stock_symbol)
        if fundamentals:
            st.markdown("### Fundamental Data")
            fundamentals_df = pd.DataFrame(fundamentals.items(), columns=["Metric", "Value"])
            st.dataframe(fundamentals_df.set_index("Metric"), use_container_width=True)
            
            # Simplified score and verdict based on available data
            pe_ratio = fundamentals.get("P/E Ratio")
            roe = fundamentals.get("Return on Equity (ROE)")
            
            # Display verdict and score
            st.markdown("#### Valuation and Efficiency Verdict:")
            verdict_parts = []
            score = 50 # Start with a neutral score

            if pe_ratio is not None or roe is not None:
                if pe_ratio is not None:
                    if pe_ratio < 25:
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
            st.info(" ".join(verdict_parts))


        # Fetch historical data for plotting and LSTM
        st.markdown("### Historical Data and LSTM Prediction")
        
        # User input for date range for historical data
        today = dt.date.today()
        default_start_date = today - dt.timedelta(days=365*2) # 2 years ago
        col_start, col_end = st.columns(2)
        start_date = col_start.date_input("Start Date", value=default_start_date)
        end_date = col_end.date_input("End Date", value=today)

        if start_date >= end_date:
            st.error("Error: End date must be after start date.")
        else:
            try:
                # Use a specific session for yfinance to handle retries
                session = requests.Session()
                retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
                adapter = HTTPAdapter(max_retries=retry)
                session.mount('http://', adapter)
                session.mount('https://', adapter)

                df = yf.download(stock_symbol, start=start_date, end=end_date, interval="1d", session=session)
                
                if df.empty:
                    st.warning(f"No historical data found for {stock_symbol} in the selected range.")
                else:
                    df = calculate_indicators(df) # Calculate RSI and BBands

                    # Plotting historical data with Candlestick, RSI, and Bollinger Bands
                    fig = go.Figure()
                    
                    # Candlestick
                    fig.add_trace(go.Candlestick(x=df.index,
                                                open=df['Open'],
                                                high=df['High'],
                                                low=df['Low'],
                                                close=df['Close'],
                                                name='Candlestick'))

                    # Bollinger Bands
                    if 'BB_LOWER' in df.columns and 'BB_MIDDLE' in df.columns and 'BB_UPPER' in df.columns:
                        fig.add_trace(go.Scatter(x=df.index, y=df['BB_UPPER'], line=dict(color='blue', width=1), name='Upper BB'))
                        fig.add_trace(go.Scatter(x=df.index, y=df['BB_MIDDLE'], line=dict(color='orange', width=1), name='Middle BB'))
                        fig.add_trace(go.Scatter(x=df.index, y=df['BB_LOWER'], line=dict(color='blue', width=1), name='Lower BB'))
                    
                    fig.update_layout(title=f'{stock_symbol} Historical Data with Bollinger Bands', xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True)

                    # RSI Plot
                    if 'RSI' in df.columns:
                        fig_rsi = go.Figure()
                        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
                        fig_rsi.add_hline(y=70, annotation_text="Overbought", line_dash="dot", line_color="red")
                        fig_rsi.add_hline(y=30, annotation_text="Oversold", line_dash="dot", line_color="green")
                        fig_rsi.update_layout(title=f'{stock_symbol} Relative Strength Index (RSI)', yaxis_title='RSI Value')
                        st.plotly_chart(fig_rsi, use_container_width=True)


                    # LSTM Prediction
                    st.markdown("#### LSTM Price Prediction")
                    if st.button(f"Train and Predict with LSTM for {stock_symbol}"):
                        if len(df) < 100: # Need sufficient data for LSTM
                            st.warning("Not enough data to train LSTM model effectively. Need at least 100 data points.")
                        else:
                            with st.spinner("Training LSTM model..."):
                                try:
                                    model, scaler, X_test, y_test, data_scaled = train_lstm_model(df.copy())
                                    
                                    # Make predictions
                                    test_predict = model.predict(X_test)
                                    test_predict = scaler.inverse_transform(test_predict)
                                    y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

                                    # Calculate R2 score
                                    r2 = r2_score(y_test_inv, test_predict)
                                    st.info(f"R-squared (R2) score for LSTM prediction: {r2:.4f}")

                                    # Plotting predictions
                                    train_predict_plot = np.empty_like(data_scaled)
                                    train_predict_plot[:, :] = np.nan
                                    # The train part needs to be plotted correctly if you want to show it.
                                    # For simplicity, focusing on test prediction.

                                    test_predict_plot = np.empty_like(data_scaled)
                                    test_predict_plot[:, :] = np.nan
                                    test_predict_plot[len(data_scaled) - len(test_predict) - 1:len(data_scaled) - 1, :] = test_predict

                                    fig_lstm = go.Figure()
                                    fig_lstm.add_trace(go.Scatter(x=df.index, y=scaler.inverse_transform(data_scaled).flatten(), mode='lines', name='Actual Price'))
                                    # Adjust index for test_predict_plot to match original df index
                                    test_plot_indices = df.index[len(data_scaled) - len(test_predict) - 1 : len(data_scaled) - 1]
                                    fig_lstm.add_trace(go.Scatter(x=test_plot_indices, y=test_predict.flatten(), mode='lines', name='Predicted Price (Test)'))
                                    
                                    fig_lstm.update_layout(title=f'{stock_symbol} LSTM Price Prediction',
                                                        xaxis_title='Date',
                                                        yaxis_title='Price')
                                    st.plotly_chart(fig_lstm, use_container_width=True)

                                    # Predict next day's closing price
                                    last_60_days = data_scaled[-60:].reshape(1, 60, 1)
                                    next_day_prediction_scaled = model.predict(last_60_days)
                                    next_day_prediction = scaler.inverse_transform(next_day_prediction_scaled)[0][0]
                                    
                                    st.success(f"Predicted closing price for the next trading day: **${next_day_prediction:.2f}**")

                                except Exception as e:
                                    st.error(f"Error during LSTM training or prediction: {e}")
                                    st.warning("Ensure you have enough historical data and that TensorFlow is correctly installed and configured.")

            except Exception as e:
                st.error(f"Error fetching data for {stock_symbol}: {e}. Please check the symbol and try again.")

# ----------- Tab 3: Candlestick Patterns & Fundamentals -----------
with tab3:
    st.header("Candlestick Pattern Recognition")
    
    # Allow user to input a stock symbol
    candlestick_symbol = st.text_input("Enter Stock Symbol for Candlestick Analysis (e.g., 'MSFT')", value="MSFT").upper()

    if candlestick_symbol:
        st.subheader(f"Candlestick Patterns for: {candlestick_symbol}")
        
        # User input for date range for historical data
        today = dt.date.today()
        default_start_date_c = today - dt.timedelta(days=365) # 1 year ago for patterns
        col_start_c, col_end_c = st.columns(2)
        start_date_c = col_start_c.date_input("Start Date for Candlesticks", value=default_start_date_c)
        end_date_c = col_end_c.date_input("End Date for Candlesticks", value=today)

        if start_date_c >= end_date_c:
            st.error("Error: End date must be after start date for candlestick analysis.")
        else:
            try:
                # Use a specific session for yfinance to handle retries
                session = requests.Session()
                retry = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
                adapter = HTTPAdapter(max_retries=retry)
                session.mount('http://', adapter)
                session.mount('https://', adapter)

                df_candlestick = yf.download(candlestick_symbol, start=start_date_c, end=end_date_c, interval="1d", session=session)
                
                if df_candlestick.empty:
                    st.warning(f"No historical data found for {candlestick_symbol} in the selected range for candlestick analysis.")
                else:
                    patterns_df = recognize_candlestick_patterns(df_candlestick.copy())
                    
                    if not patterns_df.empty:
                        st.markdown("### Detected Candlestick Patterns")
                        st.dataframe(patterns_df, use_container_width=True)
                        
                        # Plot candlestick chart with pattern markers
                        fig_candlestick = go.Figure(data=[go.Candlestick(x=df_candlestick.index,
                                                        open=df_candlestick['Open'],
                                                        high=df_candlestick['High'],
                                                        low=df_candlestick['Low'],
                                                        close=df_candlestick['Close'],
                                                        name='Candlestick')])
                        
                        # Add markers for patterns
                        for _, row in patterns_df.iterrows():
                            # Find the corresponding data point in df_candlestick
                            pattern_date = pd.to_datetime(row['Date'])
                            if pattern_date in df_candlestick.index:
                                close_price = df_candlestick.loc[pattern_date]['Close']
                                fig_candlestick.add_trace(go.Scatter(
                                    x=[pattern_date],
                                    y=[close_price],
                                    mode='markers',
                                    marker=dict(symbol='star', size=10, color='red' if row['Strength'] == 'Bearish' else 'green'),
                                    name=f"{row['Pattern']} ({row['Strength']})",
                                    hovertext=f"{row['Pattern']} ({row['Strength']}) on {row['Date']}",
                                    hoverinfo='text'
                                ))

                        fig_candlestick.update_layout(title=f'{candlestick_symbol} Candlestick Chart with Detected Patterns', xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig_candlestick, use_container_width=True)

                    else:
                        st.info("No common candlestick patterns detected in the selected period.")

            except Exception as e:
                st.error(f"Error fetching data or recognizing patterns for {candlestick_symbol}: {e}. Please check the symbol and try again.")
