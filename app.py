import streamlit as st
from datetime import datetime
from dhanhq import dhanhq
import pandas as pd

from dateutil.parser import isoparse

# ----------------- Helpers -----------------

def get_dhan_client():
    client_id = st.secrets["DHAN_CLIENT_ID"]
    access_token = st.secrets["DHAN_ACCESS_TOKEN"]
    return dhanhq(client_id, access_token)

def get_nearest_weekly_expiry(client):
    under_sec_id = st.secrets.get("NIFTY_UNDER_SECURITY_ID", "13")
    resp = client.expiry_list(
        under_security_id=under_sec_id,
        under_exchange_segment=client.INDEX,
    )
    expiries = resp.get("data") or resp
    expiry_dates = [isoparse(e).date() for e in expiries]

    today = datetime.now().date()
    future_expiries = [d for d in expiry_dates if d >= today]
    if not future_expiries:
        return None
    return min(future_expiries).isoformat()

def get_nifty_option_chain(client, expiry_iso):
    under_sec_id = st.secrets.get("NIFTY_UNDER_SECURITY_ID", "13")
    resp = client.option_chain(
        under_security_id=under_sec_id,
        under_exchange_segment=client.INDEX,
        expiry=expiry_iso,
    )
    data = resp.get("data", resp)
    underlying_ltp = data.get("last_price")
    oc_array = data.get("oc", data.get("option_chain", []))

    rows = []
    for row in oc_array:
        strike = row.get("strike_price")
        ce = row.get("ce", {}) or {}
        pe = row.get("pe", {}) or {}
        rows.append({
            "strike": float(strike),
            "ce_oi": ce.get("oi", 0),
            "ce_vol": ce.get("volume", 0),
            "ce_ltp": ce.get("ltp", 0.0),
            "ce_iv": ce.get("iv", None),
            "pe_oi": pe.get("oi", 0),
            "pe_vol": pe.get("volume", 0),
            "pe_ltp": pe.get("ltp", 0.0),
            "pe_iv": pe.get("iv", None),
        })

    df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
    df["abs_diff"] = (df["strike"] - underlying_ltp).abs()
    return underlying_ltp, df

def compute_pcr(df):
    total_put_oi = df["pe_oi"].sum()
    total_call_oi = df["ce_oi"].sum()
    if total_call_oi <= 0:
        return None, "Unknown"
    pcr = total_put_oi / total_call_oi
    if pcr < 0.7:
        sentiment = "Bullish"
    elif pcr > 1.3:
        sentiment = "Bearish"
    else:
        sentiment = "Neutral"
    return pcr, sentiment

def find_oi_support_resistance(df, underlying_ltp):
    below = df[df["strike"] <= underlying_ltp]
    above = df[df["strike"] >= underlying_ltp]

    support = resistance = None
    if not below.empty:
        support_row = below.loc[below["pe_oi"].idxmax()]
        support = (float(support_row["strike"]), int(support_row["pe_oi"]))
    if not above.empty:
        resistance_row = above.loc[above["ce_oi"].idxmax()]
        resistance = (float(resistance_row["strike"]), int(resistance_row["ce_oi"]))
    return support, resistance

def select_strikes(df, underlying_ltp, count=3):
    atm_strike = df.loc[df["abs_diff"].idxmin(), "strike"]
    df["distance"] = df["strike"] - atm_strike
    window = df[df["strike"].between(atm_strike - 400, atm_strike + 400)]
    window["ce_liq"] = window["ce_vol"] + window["ce_oi"]
    window["pe_liq"] = window["pe_vol"] + window["pe_oi"]

    ce_candidates = window.sort_values(
        ["distance", "ce_liq"], ascending=[True, False]
    ).head(count)[["strike", "ce_ltp", "ce_oi", "ce_vol"]]

    pe_candidates = window.sort_values(
        ["distance", "pe_liq"], ascending=[False, False]
    ).head(count)[["strike", "pe_ltp", "pe_oi", "pe_vol"]]

    return atm_strike, ce_candidates, pe_candidates

def generate_directional_view(sentiment, support, resistance, underlying_ltp):
    if sentiment == "Bullish" and support:
        s, _ = support
        if underlying_ltp <= s + 50:
            return "Bias: Buy Call (CE) near ATM."
    if sentiment == "Bearish" and resistance:
        r, _ = resistance
        if underlying_ltp >= r - 50:
            return "Bias: Buy Put (PE) near ATM."
    if sentiment == "Neutral":
        return "Neutral: consider range trades or wait."
    return "No clear edge: avoid directional trade."

# ----------------- Streamlit UI -----------------

st.set_page_config(page_title="Nifty Option Chain Scanner", layout="wide")
st.title("Nifty Weekly Option Chain Scanner (DhanHQ)")

refresh = st.button("Refresh Data")

if "prev_chain" not in st.session_state:
    st.session_state.prev_chain = None

if refresh:
    try:
        client = get_dhan_client()
        expiry = get_nearest_weekly_expiry(client)
        if not expiry:
            st.error("No future expiry found.")
        else:
            underlying_ltp, df = get_nifty_option_chain(client, expiry)

            pcr, sentiment = compute_pcr(df)
            support, resistance = find_oi_support_resistance(df, underlying_ltp)
            atm_strike, ce_candidates, pe_candidates = select_strikes(df, underlying_ltp)

            st.subheader("Overview")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Underlying (Nifty LTP)", f"{underlying_ltp:.2f}")
            c2.metric("Nearest Expiry", expiry)
            c3.metric("PCR", f"{pcr:.2f}" if pcr else "N/A")
            c4.metric("Sentiment", sentiment)

            st.subheader("Support & Resistance (from OI)")
            col1, col2 = st.columns(2)
            if support:
                col1.write(f"Support (Put OI): {support[0]} (OI: {support[1]})")
            else:
                col1.write("Support: N/A")
            if resistance:
                col2.write(f"Resistance (Call OI): {resistance[0]} (OI: {resistance[1]})")
            else:
                col2.write("Resistance: N/A")

            st.subheader("Strike Selection (Liquid Near ATM)")
            st.write(f"ATM Strike: {atm_strike}")
            st.write("Suggested Call (CE) Strikes:")
            st.dataframe(ce_candidates)
            st.write("Suggested Put (PE) Strikes:")
            st.dataframe(pe_candidates)

            view = generate_directional_view(sentiment, support, resistance, underlying_ltp)
            st.subheader("Scanner Interpretation")
            st.write(view)
            st.caption("Educational only. Not trading advice.")

            # Store current snapshot for possible future buildup logic
            st.session_state.prev_chain = df.copy()

    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()
else:
    st.info("Click 'Refresh Data' to fetch latest Nifty option chain.")
