import requests

BASE = "https://www.deribit.com/api/v2"

def get_btc_options():
    url = f"{BASE}/public/get_book_summary_by_currency"
    params = {"currency": "BTC", "kind": "option"}
    return requests.get(url, params=params).json()["result"]

def get_btc_price():
    url = f"{BASE}/public/ticker"
    params = {"instrument_name": "BTC-PERPETUAL"}
    return requests.get(url, params=params).json()["result"]["last_price"]

def get_funding():
    url = f"{BASE}/public/get_funding_rate_history"
    params = {"instrument_name": "BTC-PERPETUAL", "count": 1}
    return requests.get(url, params=params).json()["result"][0]["funding_rate"]
