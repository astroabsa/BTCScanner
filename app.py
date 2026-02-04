import streamlit as st
import pandas as pd
from dhanhq import dhanhq
from datetime import datetime
from dateutil.parser import isoparse
from streamlit_autorefresh import st_autorefresh
from zoneinfo import ZoneInfo  # for IST timezone handling[web:86][web:91]

# ----------------- PAGE CONFIG -----------------

st.set_page_config(
    page_title="Nifty Weekly Option Chain Scanner (DhanHQ)",
    layout="wide"
)

# ----------------- CONSTANTS -----------------
# From Dhan Option Chain docs (Nifty index underlying and index segment).[web:19][web:40]
NIFTY_UNDER_SECURITY_ID = 13
UNDER_EXCHANGE_SEGMENT = "IDX_I"   # index options segment code


# ----------------- Dhan helpers -----------------

def get_dhan_client():
    try:
        client_id = st.secrets["DHAN_CLIENT_ID"]
        access_token = st.secrets["DHAN_ACCESS_TOKEN"]
    except KeyError:
        st.error("Please set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN in Streamlit secrets.")
        st.stop()
    return dhanhq(client_id, access_token)


def get_nearest_weekly_expiry(client) -> str | None:
    """
    expiry_list() response shape:

    {
      "status": "success",
      "remarks": "",
      "data": {
        "data": ["YYYY-MM-DD", ...],
        "status": "success"
      }
    }[web:19][web:40]
    """
    resp = client.expiry_list(
        under_security_id=NIFTY_UNDER_SECURITY_ID,
        under_exchange_segment=UNDER_EXCHANGE_SEGMENT,
    )

    outer = resp.get("data") or {}
    expiries = outer.get("data") or []
    if not expiries:
        st.error("No expiries returned from Dhan expiry_list().")
        return None

    expiry_dates = [isoparse(str(d)).date() for d in expiries]
    today = datetime.now(ZoneInfo("Asia/Kolkata")).date()
    future = [d for d in expiry_dates if d >= today]
    if not future:
        st.error("No future expiry dates found.")
        return None

    return min(future).isoformat()   # "YYYY-MM-DD"


def get_nifty_option_chain(client, expiry_iso: str):
    """
    option_chain() response is similarly double-nested in data:[web:19][web:40]

    {
      "status": "success",
      "remarks": "",
      "data": {
        "data": {
          "last_price": ...,
          "oc": { "25000.000000": { "ce": {...}, "pe": {...} }, ... },
          "status": "success"
        }
      }
    }
    """
    resp = client.option_chain(
        under_security_id=NIFTY_UNDER_SECURITY_ID,
        under_exchange_segment=UNDER_EXCHANGE_SEGMENT,
        expiry=expiry_iso,
    )

    outer = resp.get("data") or {}
    inner = outer.get("data") or outer

    underlying_ltp = float(inner.get("last_price", 0.0))

    oc_dict = inner.get("oc") or {}
    if not oc_dict:
        st.error("Option chain 'oc' section is empty in Dhan response.")
        return underlying_ltp, pd.DataFrame()

    rows = []
    for strike_str, legs in oc_dict.items():
        try:
            strike = float(strike_str)
        except ValueError:
            continue

        ce = legs.get("ce", {}) or {}
        pe = legs.get("pe", {}) or {}

        rows.append({
            "strike": strike,
            # CE
            "ce_ltp": float(ce.get("last_price", 0.0) or 0.0),
            "ce_oi": int(ce.get("oi", 0) or 0),
            "ce_prev_price": float(ce.get("previous_close_price", 0.0) or 0.0),
            "ce_prev_oi": int(ce.get("previous_oi", 0) or 0),
            "ce_vol": int(ce.get("volume", 0) or 0),
            # PE
            "pe_ltp": float(pe.get("last_price", 0.0) or 0.0),
            "pe_oi": int(pe.get("oi", 0) or 0),
            "pe_prev_price": float(pe.get("previous_close_price", 0.0) or 0.0),
            "pe_prev_oi": int(pe.get("previous_oi", 0) or 0),
            "pe_vol": int(pe.get("volume", 0) or 0),
        })

    df = pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)
    df["abs_diff"] = (df["strike"] - underlying_ltp).abs()
    return underlying_ltp, df


# ----------------- Analytics helpers -----------------

def compute_pcr(df: pd.DataFrame):
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
    return round(pcr, 2), sentiment


def find_oi_support_resistance(df: pd.DataFrame, underlying_ltp: float):
    below = df[df["strike"] <= underlying_ltp]
    above = df[df["strike"] >= underlying_ltp]
    support = resistance = None

    if not below.empty:
        s_row = below.loc[below["pe_oi"].idxmax()]
        support = (float(s_row["strike"]), int(s_row["pe_oi"]))
    if not above.empty:
        r_row = above.loc[above["ce_oi"].idxmax()]
        resistance = (float(r_row["strike"]), int(r_row["ce_oi"]))
    return support, resistance


def select_strikes(df: pd.DataFrame, underlying_ltp: float, count: int = 3):
    atm_strike = float(df.loc[df["abs_diff"].idxmin(), "strike"])
    window = df[df["strike"].between(atm_strike - 400, atm_strike + 400)].copy()

    window["ce_liq"] = window["ce_vol"] + window["ce_oi"]
    window["pe_liq"] = window["pe_vol"] + window["pe_oi"]

    ce_candidates = (
        window.sort_values(["abs_diff", "ce_liq"], ascending=[True, False])
        .head(count)[["strike", "ce_ltp", "ce_oi", "ce_vol"]]
    )
    pe_candidates = (
        window.sort_values(["abs_diff", "pe_liq"], ascending=[True, False])
        .head(count)[["strike", "pe_ltp", "pe_oi", "pe_vol"]]
    )
    return atm_strike, ce_candidates, pe_candidates


def classify_buildup(ltp, prev_price, oi, prev_oi):
    if prev_price is None or prev_oi is None:
        return "Unknown"

    price_up = ltp > prev_price
    price_down = ltp < prev_price
    oi_up = oi > prev_oi
    oi_down = oi < prev_oi

    if price_up and oi_up:
        return "Long buildup"
    if price_down and oi_up:
        return "Short buildup"
    if price_up and oi_down:
        return "Short covering"
    if price_down and oi_down:
        return "Long unwinding"
    return "Mixed"


def get_atm_buildup(df: pd.DataFrame, underlying_ltp: float):
    row = df.loc[df["abs_diff"].idxmin()]
    ce_bu = classify_buildup(row["ce_ltp"], row["ce_prev_price"], row["ce_oi"], row["ce_prev_oi"])
    pe_bu = classify_buildup(row["pe_ltp"], row["pe_prev_price"], row["pe_oi"], row["pe_prev_oi"])
    return float(row["strike"]), ce_bu, pe_bu


def generate_directional_view(sentiment, support, resistance, underlying_ltp):
    if sentiment == "Bullish" and support:
        s, _ = support
        if s <= underlying_ltp <= s + 60:
            return "Bias: Buy Call (CE) near ATM."
    if sentiment == "Bearish" and resistance:
        r, _ = resistance
        if r - 60 <= underlying_ltp <= r:
            return "Bias: Buy Put (PE) near ATM."
    if sentiment == "Neutral":
        return "Market looks range-bound; consider non-directional strategies or wait."
    return "No clear edge; avoid fresh directional trade."


# ----------------- UI + AUTO REFRESH -----------------

st.title("Nifty Weekly Option Chain Scanner (DhanHQ)")
st.caption("Uses DhanHQ Option Chain & Expiry APIs. Educational use only, not trading advice.[web:19]")

# Auto-refresh every 2 minutes (120000 ms)
refresh_count = st_autorefresh(interval=2 * 60 * 1000, key="nifty_scan_refresh")
st.write(f"Auto-refresh count: {refresh_count}")

# IST timestamp
ist_now = datetime.now(ZoneInfo("Asia/Kolkata"))
st.write(f"Last updated at (IST): {ist_now.strftime('%d-%m-%Y %H:%M:%S')}")

try:
    client = get_dhan_client()

    expiry = get_nearest_weekly_expiry(client)
    if not expiry:
        st.stop()

    underlying_ltp, df = get_nifty_option_chain(client, expiry)
    if df.empty:
        st.error("No option chain rows received.")
        st.stop()

    pcr, sentiment = compute_pcr(df)
    support, resistance = find_oi_support_resistance(df, underlying_ltp)
    atm_strike, ce_candidates, pe_candidates = select_strikes(df, underlying_ltp)
    atm_bu_strike, ce_bu, pe_bu = get_atm_buildup(df, underlying_ltp)
    view = generate_directional_view(sentiment, support, resistance, underlying_ltp)

    # -------- Overview --------
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nifty LTP", f"{underlying_ltp:.2f}")
    c2.metric("Nearest Weekly Expiry", expiry)
    c3.metric("PCR", f"{pcr:.2f}" if pcr is not None else "N/A")
    c4.metric("Sentiment", sentiment)

    # -------- Support / Resistance --------
    st.subheader("Support & Resistance (from OI)")
    col1, col2 = st.columns(2)
    if support:
        col1.write(f"Support (Put OI): **{support[0]}** | OI: {support[1]}")
    else:
        col1.write("Support: N/A")
    if resistance:
        col2.write(f"Resistance (Call OI): **{resistance[0]}** | OI: {resistance[1]}")
    else:
        col2.write("Resistance: N/A")

    # -------- Buildup --------
    st.subheader("Buildup at ATM")
    st.write(f"ATM Strike: **{atm_bu_strike}**")
    st.write(f"Call (CE) buildup: {ce_bu}")
    st.write(f"Put (PE) buildup: {pe_bu}")

    # -------- Strike Selection --------
    st.subheader("Strike Selection (Liquid near ATM)")
    st.write(f"ATM Strike used for selection: **{atm_strike}**")

    st.markdown("**Suggested Call (CE) strikes**")
    st.dataframe(ce_candidates, use_container_width=True)

    st.markdown("**Suggested Put (PE) strikes**")
    st.dataframe(pe_candidates, use_container_width=True)

    # -------- Interpretation --------
    st.subheader("Scanner Interpretation")
    st.write(view)
    st.caption(
        "Logic uses PCR, OI-based support/resistance, and ATM buildup from Dhan Option Chain snapshot.[web:19]"
    )

except Exception as e:
    st.error("Unhandled exception:")
    st.exception(e)
