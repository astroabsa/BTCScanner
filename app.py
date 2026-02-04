import streamlit as st
import pandas as pd
from dhanhq import dhanhq
from datetime import datetime
from dateutil.parser import isoparse
from streamlit_autorefresh import st_autorefresh
from zoneinfo import ZoneInfo  # IST timezone[web:86][web:91]

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
    expiry_list() response shape:[web:19][web:40]
    {
      "status": "success",
      "remarks": "",
      "data": {
        "data": ["YYYY-MM-DD", ...],
        "status": "success"
      }
    }
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


def select_strikes(df: pd.DataFrame, underlying_ltp: float, count: int = 4):
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


# ----------------- SIDEBAR (controls) -----------------

with st.sidebar:
    st.title("Settings")
    st.markdown("Configure **refresh** and **display** options here.")
    auto_refresh = st.checkbox("Auto refresh", value=True)
    refresh_secs = st.slider("Refresh interval (seconds)", 30, 300, 120, 30)
    show_full_chain = st.checkbox("Show full option chain table", value=True)
    st.markdown("---")
    st.caption("Built on DhanHQ Option Chain & Expiry APIs.[web:19]")

# Trigger auto-refresh if enabled
if auto_refresh:
    st_autorefresh(interval=refresh_secs * 1000, key="nifty_scan_refresh")


# ----------------- MAIN HEADER -----------------

st.title("Nifty Weekly Option Chain Scanner (DhanHQ)")
ist_now = datetime.now(ZoneInfo("Asia/Kolkata"))
st.write(f"Last updated (IST): **{ist_now.strftime('%d-%m-%Y %H:%M:%S')}**")

# ----------------- DATA FETCH -----------------

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

    # ----------------- TABS LAYOUT -----------------
    tab_summary, tab_strikes, tab_chain = st.tabs(
        ["Summary", "Strikes & View", "Full Option Chain"]
    )  # Tabs help organize more info with less scrolling.[web:94][web:103]

    # ===== TAB 1: SUMMARY =====
    with tab_summary:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Nifty LTP", f"{underlying_ltp:.2f}")
        m2.metric("Nearest Weekly Expiry", expiry)
        m3.metric("PCR", f"{pcr:.2f}" if pcr is not None else "N/A")
        m4.metric("Sentiment", sentiment)

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Support & Resistance")
            if support:
                st.write(f"Support (Put OI): **{support[0]}**  | OI: {support[1]}")
            else:
                st.write("Support: N/A")
            if resistance:
                st.write(f"Resistance (Call OI): **{resistance[0]}**  | OI: {resistance[1]}")
            else:
                st.write("Resistance: N/A")

        with c2:
            st.subheader("ATM Buildup")
            st.write(f"ATM Strike: **{atm_bu_strike}**")
            st.write(f"Call (CE) buildup: {ce_bu}")
            st.write(f"Put (PE) buildup: {pe_bu}")

        st.subheader("Scanner Interpretation")
        st.write(view)
        st.caption(
            "Logic uses PCR, OI-based support/resistance, and ATM buildup from Dhan Option Chain snapshot.[web:19]"
        )

    # ===== TAB 2: STRIKES & VIEW =====
    with tab_strikes:
        st.subheader("Liquid Strikes Near ATM")
        st.write(f"ATM Strike used for selection: **{atm_strike}**")

        left, right = st.columns(2)
        with left:
            st.markdown("**Suggested Call (CE) strikes**")
            st.dataframe(
                ce_candidates.set_index("strike"),
                use_container_width=True,
                height=250,
            )
        with right:
            st.markdown("**Suggested Put (PE) strikes**")
            st.dataframe(
                pe_candidates.set_index("strike"),
                use_container_width=True,
                height=250,
            )

        st.subheader("OI Snapshot Around ATM (Compact View)")
        around = df[df["strike"].between(atm_strike - 300, atm_strike + 300)].copy()
        oi_view = around[["strike", "ce_oi", "pe_oi"]].set_index("strike")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Call OI by strike")
            st.bar_chart(oi_view[["ce_oi"]], height=220)
        with c2:
            st.caption("Put OI by strike")
            st.bar_chart(oi_view[["pe_oi"]], height=220)

    # ===== TAB 3: FULL OPTION CHAIN =====
    with tab_chain:
        if show_full_chain:
            st.subheader("Full Option Chain (Current Expiry)")

            display_cols = [
                "strike",
                "ce_ltp", "ce_oi", "ce_vol",
                "pe_ltp", "pe_oi", "pe_vol",
            ]

            # Remove rows where all option values are zero (no trading interest).[web:121]
            numeric_cols = ["ce_ltp", "ce_oi", "ce_vol", "pe_ltp", "pe_oi", "pe_vol"]
            df_nonzero = df.copy()
            df_nonzero = df_nonzero[~(df_nonzero[numeric_cols] == 0).all(axis=1)]

            st.dataframe(
                df_nonzero[display_cols],
                use_container_width=True,
                height=450,
                hide_index=True,
            )
        else:
            st.info("Enable 'Show full option chain table' in sidebar to view complete chain.")

except Exception as e:
    st.error("Unhandled exception:")
    st.exception(e)
