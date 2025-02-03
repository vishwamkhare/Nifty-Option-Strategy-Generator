from turtle import clear
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def download_historical_data():
    data = yf.download("^NSEI", start="2010-01-01", end=datetime.datetime.now().strftime('%Y-%m-%d'))
    data['Returns'] = data['Close'].pct_change()
    data['Target'] = (data['Returns'] > 0).astype(int)  # 1 if price goes up, 0 otherwise
    return data

def add_features(data):
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['MA_200'] = data['Close'].rolling(window=200).mean()
    delta = data['Close'].diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data.dropna(inplace=True)
    return data

def train_model(data):
    X = data[['MA_50', 'MA_200', 'RSI']]
    y = data['Target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model Accuracy:", accuracy_score(y_test, model.predict(X_test)))
    return model

def fetch_real_time_data():
    nifty = yf.Ticker("^NSEI")
    real_time_data = nifty.history(period="5d")
    return real_time_data

def compute_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def fetch_nifty_option_chain():
    url = "https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive"
    }
    session = requests.Session()
    session.get("https://www.nseindia.com", headers=headers)  # Initialize session
    response = session.get(url, headers=headers)
    if response.status_code != 200:
        print("Failed to fetch option chain data. Status code:", response.status_code)
        return None
    data = response.json()
    return data['records']['data']

def generate_option_strategy(model, real_time_data, historical_data):
    latest_close = real_time_data['Close'].iloc[-1]
    ma_50 = historical_data['Close'].rolling(window=50).mean().iloc[-1]
    ma_200 = historical_data['Close'].rolling(window=200).mean().iloc[-1]
    rsi = compute_rsi(historical_data).iloc[-1]
    X_input = np.array([[ma_50, ma_200, rsi]]).reshape(1, -1)
    prediction = model.predict(X_input)[0]
    option_chain = fetch_nifty_option_chain()
    if option_chain is None:
        return "Error: Option chain data is not available. Please try again later."
    
    call_options = [opt for opt in option_chain if opt.get('CE')]
    put_options = [opt for opt in option_chain if opt.get('PE')]
    
    if prediction == 1:
        valid_calls = [opt['CE'] for opt in call_options if opt['CE']['strikePrice'] > latest_close]
        if not valid_calls:
            return "Error: No valid call options found."
        selected_call = valid_calls[0]
        strike_price = selected_call['strikePrice']
        expiry_date = selected_call['expiryDate']
        premium = selected_call['lastPrice']
        stop_loss = strike_price - (0.5 * premium)  # Example stop loss logic
        target_price = strike_price + (2 * premium)  # Example target price logic
        return {
            "Strategy": "Buy Call Option",
            "Strike Price": strike_price,
            "Expiry Date": expiry_date,
            "Premium (Cost)": premium,
            "Recommended Buy Price": premium,
            "Recommended Sell Price": target_price,
            "Stop Loss": stop_loss,
            "Reasoning": "The model predicts an upward movement in the Nifty index."
        }
    else:
        valid_puts = [opt['PE'] for opt in put_options if opt['PE']['strikePrice'] < latest_close]
        if not valid_puts:
            return "Error: No valid put options found."
        selected_put = valid_puts[-1]
        strike_price = selected_put['strikePrice']
        expiry_date = selected_put['expiryDate']
        premium = selected_put['lastPrice']
        stop_loss = strike_price + (0.5 * premium)  # Example stop loss logic
        target_price = strike_price - (2 * premium)  # Example target price logic
        return {
            "Strategy": "Buy Put Option",
            "Strike Price": strike_price,
            "Expiry Date": expiry_date,
            "Premium (Cost)": premium,
            "Recommended Buy Price": premium,
            "Recommended Sell Price": target_price,
            "Stop Loss": stop_loss,
            "Reasoning": "The model predicts a downward movement in the Nifty index."
        }

def main():
    data = download_historical_data()
    data = add_features(data)
    model = train_model(data)
    real_time_data = fetch_real_time_data()
    strategy = generate_option_strategy(model, real_time_data, data)
    print("Option Recommendation:")
    for key, value in strategy.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()