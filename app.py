import streamlit as st
from data.deribit import get_btc_options, get_btc_price, get_funding
from analytics.pcr import compute_pcr
from analytics.max_pain import compute_max_pain
from engine.bias_engine import get_bias

st.set_page_config(layout="wide")
st.title("BTC Derivatives Sentiment Engine")

options = get_btc_options()
price = get_btc_price()
funding = get_funding()

pcr = compute_pcr(options)
max_pain = compute_max_pain(options)
bias = get_bias(pcr, funding)

col1, col2, col3 = st.columns(3)
col1.metric("BTC Price", price)
col2.metric("PCR", pcr)
col3.metric("Funding", funding)

st.subheader("Market Bias")
st.markdown(f"## {bias}")

st.subheader("Max Pain")
st.write(max_pain)
