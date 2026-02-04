import streamlit as st
import pandas as pd
from dhanhq import dhanhq
from datetime import datetime
from dateutil.parser import isoparse

# ----------------- Config -----------------

UNDER_EXCHANGE_SEGMENT = "IDX_I"  # Nifty index segment code (Dhan: UnderlyingSeg).[web:19]
DEFAULT_UNDER_SECURITY_ID = 13    # Nifty underlying security ID from Dhan docs.[web:19]

st.set_page_config(
    page_title="Nifty Weekly Option Chain Scanner (DhanHQ)",
    layout="wide"
)

# ----------------- Dhan helpers -----------------

def get_dhan_client():
    try:
        client_id = st.secrets["DHAN_CLIENT_ID"]
        access_token = st.secrets["DHAN_ACCESS_TOKEN"]
    except KeyError:
        st.error("Please set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN in Streamlit secrets.")
        st.stop()
    return dhanhq(client_id, access_token)

def get_under_security_id():
    raw = st.secrets.get("NIFTY_UNDER_SECURITY_ID", str(DEFAULT_UNDER_SECURITY_ID))
    try:
        return int(raw)
    except ValueError:
        st.error(f"NIFTY_UNDER_SECURITY_ID must be numeric, got: {raw}")
        st.stop()

def get_nearest_weekly_expiry(client, under_security_id: int) -> str | None:
    """
    Use Dhan expiry_list() to fetch all active expiries (YYYY-MM-DD) and
    pick the nearest expiry >= today.[web:19][web:40]
    """
    resp = client.expiry_list(
        under_security_id=under_security_id,
        under_exchange_segment=UNDER_EXCHANGE_SEGMENT,
    )

    expiries = resp.get("data") or []
    if not expiries:
        st.error("No expiries returned from Dhan expiry_list().")
        return None

    expiry_dates = [isoparse(str(d)).date() for d in expiries]
    today = datetime.now().date()
    future = [d for d in expiry_dates if d >= today]
    if not future:
        st.error("No future expiry available for this underlying.")
        return None

    nearest = min(future)
    return nearest.isoformat()  # "YYYY-MM-DD"

def get_nifty_option_chain(client, under_security_id: int, expiry_iso: str):
    """
    Call Dhan option_chain() and convert response to a clean DataFrame.
    Option Chain response structure (simplified):[web:19][web:40]

    {
      "data": {
        "last_price": 24964.25,
        "oc": {
          "25000.000000": {
             "ce": { "last_price": ..., "oi": ..., "previous_close_price": ..., "previous_oi": ..., "volume": ... },
             "pe": { ... }
          },
          ...
        }
      }
    }
    """
    resp = client.option_chain(
        under_security_id=under_security_id,
        under_exchange_segment=UNDER_EXCHANGE_SEGMENT,
        expiry=expiry_iso,
    )

    data = resp.get("data") or {}
    underlying_ltp = float(data.get("last_price", 0.0))

    oc_dict = data.get("oc") or {}
    if not oc_dict:
        st.error("Option chain 'oc' section is empty.")
        return underlying_ltp, pd.DataFrame()

    rows = []
    for strike_str, leg_data in oc_dict.items():
        try:
            strike = float(strike_str)
        except ValueError:
            # Skip malformed keys
            continue

        ce = leg_data.get("ce", {}) or {}
        pe = leg_data.get("pe", {}) or {}

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

    df = pd.DataFrame(rows)
    if df.empty:
        return underlying_ltp, df

    df = df.sort_values("strike").reset_index(drop=True)
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
        support_row = below.loc[below["pe_oi"].idxmax()]
        support = (float(support_row["strike"]), int(support_row["pe_oi"]))

    if not above.empty:
        resistance_row = above.loc[above["ce_oi"].idxmax()]
        resistance = (float(resistance_row["strike"]), int(resistance_row["ce_oi"]))

    return support, resistance

def select_strikes(df: pd.DataFrame, underlying_ltp: float, count: int = 3):
    # ATM strike
    atm_strike = float(df.loc[df["abs_diff"].idxmin(), "strike"])

    # Limit to ±400 points around ATM
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
    """
    Standard OI/price buildup logic using current vs previous close/OI.[web:19]
    """
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
    idx = df["abs_diff"].idxmin()
    row = df.loc[idx]

    ce_bu = classify_buildup(
        row["ce_ltp"], row["ce_prev_price"], row["ce_oi"], row["ce_prev_oi"]
    )
    pe_bu = classify_buildup(
        row["pe_ltp"], row["pe_prev_price"], row["pe_oi"], row["pe_prev_oi"]
    )

    return float(row["strike"]), ce_bu, pe_bu

def generate_directional_view(sentiment, support, resistance, underlying_ltp):
    """
    Very simple rule engine to say Buy CE / Buy PE / No clear trade.
    You can tune thresholds as you like.
    """
    if sentiment == "Bullish" and support:
        s_strike, _ = support
        if underlying_ltp >= s_strike and underlying_ltp <= s_strike + 60:
            return "Bias: Buy Call (CE) near ATM or slightly OTM."
    if sentiment == "Bearish" and resistance:
        r_strike, _ = resistance
        if underlying_ltp <= r_strike and underlying_ltp >= r_strike - 60:
            return "Bias: Buy Put (PE) near ATM or slightly OTM."
    if sentiment == "Neutral":
        return "Market looks range-bound; consider non-directional strategies or wait."
    return "No clear edge; avoid fresh directional trade."

# ----------------- Streamlit UI -----------------

st.title("Nifty Weekly Option Chain Scanner (DhanHQ)")
st.caption("Uses DhanHQ Option Chain & Expiry APIs. Educational use only, not trading advice.[web:19]")

if st.button("Scan Now"):
    try:
        client = get_dhan_client()
        under_sec_id = get_under_security_id()

        expiry = get_nearest_weekly_expiry(client, under_sec_id)
        if not expiry:
            st.stop()

        underlying_ltp, df = get_nifty_option_chain(client, under_sec_id, expiry)
        if df.empty:
            st.error("No option chain rows received.")
            st.stop()

        pcr, sentiment = compute_pcr(df)
        support, resistance = find_oi_support_resistance(df, underlying_ltp)
        atm_strike, ce_candidates, pe_candidates = select_strikes(df, underlying_ltp)
        atm_bu_strike, ce_buildup, pe_buildup = get_atm_buildup(df, underlying_ltp)
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
            col1.write(f"Support (Put OI): **{support[0]}**  | OI: {support[1]}")
        else:
            col1.write("Support: N/A")

        if resistance:
            col2.write(f"Resistance (Call OI): **{resistance[0]}**  | OI: {resistance[1]}")
        else:
            col2.write("Resistance: N/A")

        # -------- Buildup --------
        st.subheader("Buildup at ATM")
        st.write(f"ATM Strike: **{atm_bu_strike}**")
        st.write(f"Call (CE) buildup: {ce_buildup}")
        st.write(f"Put (PE) buildup: {pe_buildup}")

        # -------- Strike Suggestions --------
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
        st.error(f"Error: {e}")
else:
    st.info("Click 'Scan Now' to fetch latest Nifty weekly option chain.")
