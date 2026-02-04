import streamlit as st
import pandas as pd
from dhanhq import dhanhq
from datetime import datetime
from dateutil.parser import isoparse

st.set_page_config(
    page_title="Nifty Weekly Option Chain Scanner (DhanHQ)",
    layout="wide"
)

# ----------------- CONSTANTS -----------------
# From Dhan docs: UnderlyingScrip=13 and UnderlyingSeg="IDX_I" for Nifty index options.[web:19][web:40][web:58]
NIFTY_UNDER_SECURITY_ID = 13


# ----------------- BASIC DHAN TEST -----------------

def get_dhan_client():
    try:
        client_id = st.secrets["DHAN_CLIENT_ID"]
        access_token = st.secrets["DHAN_ACCESS_TOKEN"]
    except KeyError:
        st.error("Please set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN in Streamlit secrets.")
        st.stop()
    return dhanhq(client_id, access_token)


def test_dhan_calls():
    """
    Minimal test: just call expiry_list() and option_chain() once
    with hard-coded parameters. No extra logic.
    This will tell us exactly where invalid literal for int() is raised.
    """
    client = get_dhan_client()

    st.write("Using Dhan INDEX constant:", client.INDEX, type(client.INDEX))

    # ---- 1) Test expiry_list ----
    st.write("Calling expiry_list()...")
    try:
        resp = client.expiry_list(
            under_security_id=NIFTY_UNDER_SECURITY_ID,
            under_exchange_segment=client.INDEX,  # 'IDX_I'[web:58]
        )
        st.write("expiry_list() raw response:", resp)
    except Exception as e:
        st.error("Exception in expiry_list():")
        st.exception(e)
        st.stop()

    expiries = resp.get("data") or []
    if not expiries:
        st.error("No expiries returned from expiry_list().")
        st.stop()

    expiry_dates = [isoparse(str(d)).date() for d in expiries]
    today = datetime.now().date()
    future = [d for d in expiry_dates if d >= today]
    if not future:
        st.error("No future expiry from expiry_list().")
        st.stop()

    nearest_expiry = min(future).isoformat()
    st.write("Nearest expiry picked:", nearest_expiry)

    # ---- 2) Test option_chain ----
    st.write("Calling option_chain()...")
    try:
        oc_resp = client.option_chain(
            under_security_id=NIFTY_UNDER_SECURITY_ID,
            under_exchange_segment=client.INDEX,
            expiry=nearest_expiry,
        )
        st.write("option_chain() top-level keys:", list(oc_resp.keys()))
    except Exception as e:
        st.error("Exception in option_chain():")
        st.exception(e)
        st.stop()

    data = oc_resp.get("data") or {}
    st.write("option_chain() data keys:", list(data.keys()))

    ltp = float(data.get("last_price", 0.0))
    st.write("Underlying last_price:", ltp)

    oc_dict = data.get("oc") or {}
    st.write("Number of strikes in oc:", len(oc_dict))

    # Convert a few rows just to confirm everything is numeric
    rows = []
    for i, (strike_str, legs) in enumerate(oc_dict.items()):
        if i >= 5:
            break
        ce = legs.get("ce", {}) or {}
        pe = legs.get("pe", {}) or {}
        rows.append({
            "strike": float(strike_str),
            "ce_ltp": float(ce.get("last_price", 0.0) or 0.0),
            "ce_oi": int(ce.get("oi", 0) or 0),
            "pe_ltp": float(pe.get("last_price", 0.0) or 0.0),
            "pe_oi": int(pe.get("oi", 0) or 0),
        })

    df = pd.DataFrame(rows)
    st.write("Sample parsed DataFrame (first 5 strikes):")
    st.dataframe(df)


# ----------------- FULL SCANNER (same as before, but only runs if basic test passes) -----------------

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
    df = df.copy()
    df["abs_diff"] = (df["strike"] - underlying_ltp).abs()
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
    df = df.copy()
    df["abs_diff"] = (df["strike"] - underlying_ltp).abs()
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


# ----------------- STREAMLIT UI -----------------

st.title("Nifty Weekly Option Chain Scanner (DhanHQ)")
st.caption("Uses DhanHQ Option Chain & Expiry APIs. Educational use only, not trading advice.[web:19]")

debug_mode = st.checkbox("Run Dhan debug test first", value=True)

if st.button("Scan Now"):
    if debug_mode:
        # Run minimal test first; if anything blows up, you will see
        # which function and full stack trace.
        test_dhan_calls()
        st.success("Basic Dhan expiry_list() and option_chain() calls succeeded. Now you can turn off debug and re-run.")
    else:
        try:
            client = get_dhan_client()

            # First get expiry list and nearest expiry
            resp = client.expiry_list(
                under_security_id=NIFTY_UNDER_SECURITY_ID,
                under_exchange_segment=client.INDEX,
            )
            expiries = resp.get("data") or []
            expiry_dates = [isoparse(str(d)).date() for d in expiries]
            today = datetime.now().date()
            future = [d for d in expiry_dates if d >= today]
            nearest_expiry = min(future).isoformat()

            # Then full option chain
            oc_resp = client.option_chain(
                under_security_id=NIFTY_UNDER_SECURITY_ID,
                under_exchange_segment=client.INDEX,
                expiry=nearest_expiry,
            )
            data = oc_resp.get("data") or {}
            underlying_ltp = float(data.get("last_price", 0.0))
            oc_dict = data.get("oc") or {}

            rows = []
            for strike_str, legs in oc_dict.items():
                ce = legs.get("ce", {}) or {}
                pe = legs.get("pe", {}) or {}
                rows.append({
                    "strike": float(strike_str),
                    "ce_ltp": float(ce.get("last_price", 0.0) or 0.0),
                    "ce_oi": int(ce.get("oi", 0) or 0),
                    "ce_prev_price": float(ce.get("previous_close_price", 0.0) or 0.0),
                    "ce_prev_oi": int(ce.get("previous_oi", 0) or 0),
                    "ce_vol": int(ce.get("volume", 0) or 0),
                    "pe_ltp": float(pe.get("last_price", 0.0) or 0.0),
                    "pe_oi": int(pe.get("oi", 0) or 0),
                    "pe_prev_price": float(pe.get("previous_close_price", 0.0) or 0.0),
                    "pe_prev_oi": int(pe.get("previous_oi", 0) or 0),
                    "pe_vol": int(pe.get("volume", 0) or 0),
                })

            df = pd.DataFrame(rows)
            if df.empty:
                st.error("Option chain DataFrame is empty.")
                st.stop()

            pcr, sentiment = compute_pcr(df)
            support, resistance = find_oi_support_resistance(df, underlying_ltp)
            atm_strike, ce_candidates, pe_candidates = select_strikes(df, underlying_ltp)
            atm_bu_strike, ce_bu, pe_bu = get_atm_buildup(df, underlying_ltp)
            view = generate_directional_view(sentiment, support, resistance, underlying_ltp)

            # Overview
            st.subheader("Overview")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Nifty LTP", f"{underlying_ltp:.2f}")
            c2.metric("Nearest Weekly Expiry", nearest_expiry)
            c3.metric("PCR", f"{pcr:.2f}" if pcr is not None else "N/A")
            c4.metric("Sentiment", sentiment)

            # S/R
            st.subheader("Support & Resistance (from OI)")
            c1, c2 = st.columns(2)
            if support:
                c1.write(f"Support (Put OI): **{support[0]}** | OI: {support[1]}")
            else:
                c1.write("Support: N/A")
            if resistance:
                c2.write(f"Resistance (Call OI): **{resistance[0]}** | OI: {resistance[1]}")
            else:
                c2.write("Resistance: N/A")

            # Buildup
            st.subheader("Buildup at ATM")
            st.write(f"ATM Strike: **{atm_bu_strike}**")
            st.write(f"Call (CE) buildup: {ce_bu}")
            st.write(f"Put (PE) buildup: {pe_bu}")

            # Strikes
            st.subheader("Strike Selection (Liquid Near ATM)")
            st.write(f"ATM Strike used for selection: **{atm_strike}**")
            st.markdown("**Suggested Call (CE) strikes**")
            st.dataframe(ce_candidates, use_container_width=True)
            st.markdown("**Suggested Put (PE) strikes**")
            st.dataframe(pe_candidates, use_container_width=True)

            # Interpretation
            st.subheader("Scanner Interpretation")
            st.write(view)
            st.caption("Logic uses PCR, OI S/R, and ATM buildup from Dhan snapshot. Not trading advice.[web:19]")

        except Exception as e:
            st.error("Unhandled exception in scanner:")
            st.exception(e)
