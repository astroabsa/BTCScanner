import streamlit as st
import pandas as pd
from dhanhq import dhanhq
from datetime import datetime, timedelta
from dateutil.parser import isoparse
from streamlit_autorefresh import st_autorefresh
from zoneinfo import ZoneInfo  # IST timezone[web:86][web:91]
import plotly.graph_objects as go  # for custom-colored OI charts[web:201][web:207]

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
    option_chain() response is double-nested in data:[web:19][web:40]

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


# ----------------- Historical 5-min trend (VWAP + EMA20) -----------------

def get_5min_trend_and_vwap(client):
    """
    Uses intraday_minute_data with interval=5 to compute EMA20 + VWAP trend
    for Nifty index over today's session.[page:1][web:154][web:157]
    """
    today = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d")
    from_date = (datetime.now(ZoneInfo("Asia/Kolkata")) - timedelta(days=1)).strftime("%Y-%m-%d")

    try:
        resp = client.intraday_minute_data(
            security_id=str(NIFTY_UNDER_SECURITY_ID),
            exchange_segment=UNDER_EXCHANGE_SEGMENT,
            instrument_type="INDEX",
            interval=5,
            from_date=from_date,
            to_date=today,
        )
    except Exception:
        return "Unknown", None, None

    data = resp.get("data") or resp
    closes = data.get("close") or []
    highs = data.get("high") or []
    lows = data.get("low") or []
    vols = data.get("volume") or []

    if not closes or not highs or not lows or not vols:
        return "Unknown", None, None

    closes_s = pd.Series(closes, dtype=float)
    highs_s = pd.Series(highs, dtype=float)
    lows_s = pd.Series(lows, dtype=float)
    vols_s = pd.Series(vols, dtype=float)

    ema20 = closes_s.ewm(span=20, adjust=False).mean()
    last_close = closes_s.iloc[-1]
    last_ema = ema20.iloc[-1]
    prev_ema = ema20.iloc[-2] if len(ema20) > 1 else last_ema

    tp = (highs_s + lows_s + closes_s) / 3.0
    vwap_series = (tp * vols_s).cumsum() / vols_s.cumsum()
    last_vwap = vwap_series.iloc[-1]

    if last_close > last_ema and last_close > last_vwap and last_ema > prev_ema:
        trend = "Uptrend"
    elif last_close < last_ema and last_close < last_vwap and last_ema < prev_ema:
        trend = "Downtrend"
    else:
        trend = "Sideways"

    return trend, float(last_vwap), float(last_ema)


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


def local_pcr_around_atm(df: pd.DataFrame, atm_index: int, window: int = 2):
    """
    Local PCR using strikes in [ATM-2, ATM+2] for scalping bias.
    """
    start = max(atm_index - window, 0)
    end = min(atm_index + window, len(df) - 1)
    window_df = df.iloc[start:end + 1]

    total_put_oi = window_df["pe_oi"].sum()
    total_call_oi = window_df["ce_oi"].sum()
    if total_call_oi <= 0:
        return None, "Unknown", window_df

    pcr = total_put_oi / total_call_oi
    if pcr < 0.7:
        sentiment = "Bullish"
    elif pcr > 1.3:
        sentiment = "Bearish"
    else:
        sentiment = "Neutral"
    return round(pcr, 2), sentiment, window_df


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
    """
    Daily buildup vs previous close (informational).
    """
    row = df.loc[df["abs_diff"].idxmin()]
    ce_bu = classify_buildup(row["ce_ltp"], row["ce_prev_price"], row["ce_oi"], row["ce_prev_oi"])
    pe_bu = classify_buildup(row["pe_ltp"], row["pe_prev_price"], row["pe_oi"], row["pe_prev_oi"])
    return float(row["strike"]), ce_bu, pe_bu


def get_intraday_buildup_atm(df_current: pd.DataFrame, prev_df: pd.DataFrame | None):
    """
    Intraday buildup at ATM vs previous snapshot (for scalper bias).
    """
    if prev_df is None or prev_df.empty:
        return "Unknown", "Unknown"

    merged = df_current.merge(
        prev_df[["strike", "ce_ltp", "ce_oi", "pe_ltp", "pe_oi"]],
        on="strike",
        how="inner",
        suffixes=("", "_prev_intra"),
    )
    if merged.empty:
        return "Unknown", "Unknown"

    atm_row = merged.loc[merged["abs_diff"].idxmin()]

    ce_bu = classify_buildup(
        atm_row["ce_ltp"],
        atm_row["ce_ltp_prev_intra"],
        atm_row["ce_oi"],
        atm_row["ce_oi_prev_intra"],
    )
    pe_bu = classify_buildup(
        atm_row["pe_ltp"],
        atm_row["pe_ltp_prev_intra"],
        atm_row["pe_oi"],
        atm_row["pe_oi_prev_intra"],
    )
    return ce_bu, pe_bu


def get_intraday_delta_table(df_current: pd.DataFrame, prev_df: pd.DataFrame | None):
    """
    Mini table of ΔOI and ΔLTP for ATM-1, ATM, ATM+1 vs previous snapshot.
    """
    if prev_df is None or prev_df.empty:
        return None

    prev_idx = prev_df.set_index("strike")[["ce_ltp", "ce_oi", "pe_ltp", "pe_oi"]]
    atm_idx = df_current["abs_diff"].idxmin()
    indices = [i for i in [atm_idx - 1, atm_idx, atm_idx + 1]
               if 0 <= i < len(df_current)]

    rows = []
    for i in indices:
        row = df_current.iloc[i]
        s = row["strike"]
        if s not in prev_idx.index:
            continue
        prow = prev_idx.loc[s]
        rows.append({
            "strike": s,
            "ce_ΔOI": row["ce_oi"] - prow["ce_oi"],
            "ce_ΔLTP": round(row["ce_ltp"] - prow["ce_ltp"], 2),
            "pe_ΔOI": row["pe_oi"] - prow["pe_oi"],
            "pe_ΔLTP": round(row["pe_ltp"] - prow["pe_ltp"], 2),
        })

    if not rows:
        return None
    return pd.DataFrame(rows)


def compute_scalper_bias(trend: str, local_sentiment: str,
                         ce_bu_intra: str, pe_bu_intra: str) -> str:
    """
    Simple rule engine for scalping bias based on:
    - 5m trend (VWAP + EMA20)
    - Local PCR sentiment around ATM
    - Intraday CE/PE buildup at ATM
    """
    if trend not in ("Uptrend", "Downtrend") or local_sentiment == "Unknown":
        return "No Trade"

    ce_pos = ce_bu_intra in ("Long buildup", "Short covering")
    pe_pos = pe_bu_intra in ("Long buildup", "Short covering")

    if trend == "Uptrend" and local_sentiment == "Bullish" and ce_pos and not pe_pos:
        return "Long CE"
    if trend == "Downtrend" and local_sentiment == "Bearish" and pe_pos and not ce_pos:
        return "Long PE"
    return "No Trade"


# ----------------- SIDEBAR (controls) -----------------

with st.sidebar:
    st.title("Controls")
    auto_refresh = st.checkbox("Auto refresh", value=True)
    refresh_secs = st.slider("Refresh interval (seconds)", 30, 300, 120, 30)
    show_full_chain = st.checkbox("Show full option chain table", value=True)
    st.markdown("---")
    st.caption("Built on DhanHQ Option Chain & Expiry APIs.[web:19]")

if auto_refresh:
    st_autorefresh(interval=refresh_secs * 1000, key="nifty_scan_refresh")


# ----------------- MAIN HEADER -----------------

st.title("Nifty Weekly Option Chain Scanner (DhanHQ)")
ist_now = datetime.now(ZoneInfo("Asia/Kolkata"))
st.write(f"Last updated (IST): **{ist_now.strftime('%d-%m-%Y %H:%M:%S')}**")

# ----------------- DATA FETCH + CALC -----------------

try:
    client = get_dhan_client()

    expiry = get_nearest_weekly_expiry(client)
    if not expiry:
        st.stop()

    underlying_ltp, df = get_nifty_option_chain(client, expiry)
    if df.empty:
        st.error("No option chain rows received.")
        st.stop()

    # Maintain intraday snapshots (last 3) for ΔOI & ΔLTP
    if "snapshots" not in st.session_state:
        st.session_state["snapshots"] = []
    st.session_state["snapshots"].append(
        df[["strike", "ce_ltp", "ce_oi", "pe_ltp", "pe_oi", "abs_diff"]].copy()
    )
    if len(st.session_state["snapshots"]) > 3:
        st.session_state["snapshots"].pop(0)

    prev_intra_df = None
    if len(st.session_state["snapshots"]) >= 2:
        prev_intra_df = (
            st.session_state["snapshots"][0]
            if len(st.session_state["snapshots"]) == 3
            else st.session_state["snapshots"][-2]
        )

    # Core analytics
    pcr, sentiment = compute_pcr(df)
    support, resistance = find_oi_support_resistance(df, underlying_ltp)
    atm_strike, ce_candidates, pe_candidates = select_strikes(df, underlying_ltp)
    atm_bu_strike, ce_bu_daily, pe_bu_daily = get_atm_buildup(df, underlying_ltp)

    trend_5m, vwap_5m, ema20_5m = get_5min_trend_and_vwap(client)

    atm_index = df["abs_diff"].idxmin()
    local_pcr_val, local_sentiment, _ = local_pcr_around_atm(df, atm_index, window=2)

    ce_bu_intra, pe_bu_intra = get_intraday_buildup_atm(df, prev_intra_df)
    intra_delta_table = get_intraday_delta_table(df, prev_intra_df)

    scalper_bias = compute_scalper_bias(trend_5m, local_sentiment,
                                        ce_bu_intra, pe_bu_intra)

    # ----------------- TABS LAYOUT -----------------
    tab_summary, tab_strikes, tab_chain = st.tabs(
        ["Summary", "Strikes & View", "Full Option Chain"]
    )

    # ===== TAB 1: SUMMARY =====
    with tab_summary:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Nifty LTP", f"{underlying_ltp:.2f}")
        k2.metric("Nearest Expiry", expiry)
        k3.metric("5m Trend", trend_5m)
        k4.metric("Scalper Bias", scalper_bias)

        s1, s2, s3 = st.columns(3)
        s1.metric("Overall PCR", f"{pcr:.2f}" if pcr is not None else "N/A")
        s2.metric("Local PCR (ATM±2)", f"{local_pcr_val:.2f}" if local_pcr_val is not None else "N/A")
        s3.metric("Sentiment", local_sentiment)

        left, right = st.columns(2)

        with left:
            with st.container(border=True):
                st.subheader("Support & Resistance")
                col_a, col_b = st.columns(2)
                if support:
                    col_a.metric("Support strike", f"{support[0]:.0f}")
                    col_b.metric("Put OI at support", f"{support[1]}")
                else:
                    st.write("No clear support from Put OI.")
                if resistance:
                    col_a, col_b = st.columns(2)
                    col_a.metric("Resistance strike", f"{resistance[0]:.0f}")
                    col_b.metric("Call OI at resistance", f"{resistance[1]}")
                else:
                    st.write("No clear resistance from Call OI.")
                if vwap_5m is not None and ema20_5m is not None:
                    st.caption(f"5m VWAP: {vwap_5m:.2f}   |   EMA20: {ema20_5m:.2f}")

        with right:
            with st.container(border=True):
                st.subheader("ATM Buildup & Intraday Flow")
                st.write(f"ATM strike: **{atm_bu_strike:.0f}**")
                st.write(f"Daily vs prev close – CE: {ce_bu_daily}, PE: {pe_bu_daily}")
                st.write(f"Intraday vs last snapshot – CE: {ce_bu_intra}, PE: {pe_bu_intra}")
                if intra_delta_table is not None:
                    st.caption("ΔOI & ΔLTP (ATM−1, ATM, ATM+1)")
                    st.dataframe(
                        intra_delta_table,
                        use_container_width=True,
                        hide_index=True,
                        height=150,
                    )

        st.subheader("Scanner Interpretation")
        if scalper_bias == "Long CE":
            st.success(
                "Scalper Bias: Long CE – trend up, local PCR bullish, "
                "and intraday CE buildup supports longs."
            )
        elif scalper_bias == "Long PE":
            st.error(
                "Scalper Bias: Long PE – trend down, local PCR bearish, "
                "and intraday PE buildup supports shorts."
            )
        else:
            st.info("No aligned high-probability scalp right now; better to wait.")

        st.caption(
            "Bias combines 5m trend, local PCR, and intraday ATM CE/PE buildup from Dhan data.[web:19][web:154][web:157]"
        )

    # ===== TAB 2: STRIKES & VIEW =====
    with tab_strikes:
        st.subheader("Liquid Strikes Near ATM")
        st.write(f"ATM strike used for selection: **{atm_strike:.0f}**")

        left, right = st.columns(2)
        with left:
            st.markdown("**Suggested Call (CE) strikes**")
            st.dataframe(
                ce_candidates.set_index("strike"),
                use_container_width=True,
                height=230,
            )
        with right:
            st.markdown("**Suggested Put (PE) strikes**")
            st.dataframe(
                pe_candidates.set_index("strike"),
                use_container_width=True,
                height=230,
            )

        # -------- Combined Call/Put OI chart --------
        st.subheader("OI Snapshot Around ATM")

        around = df[df["strike"].between(atm_strike - 300, atm_strike + 300)].copy()
        oi_df = around[["strike", "ce_oi", "pe_oi"]]

        fig_oi = go.Figure()
        fig_oi.add_bar(
            name="Call OI",
            x=oi_df["strike"],
            y=oi_df["ce_oi"],
            marker_color="red",
        )
        fig_oi.add_bar(
            name="Put OI",
            x=oi_df["strike"],
            y=oi_df["pe_oi"],
            marker_color="green",
        )
        fig_oi.update_layout(
            barmode="group",
            title="Call & Put OI by strike (around ATM)",
            xaxis_title="Strike",
            yaxis_title="Open interest",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig_oi, use_container_width=True, config={"displayModeBar": False})

        # -------- ΔOI chart (change in OI vs previous snapshot) --------
        st.subheader("Change in OI (Call & Put)")

        if prev_intra_df is not None and not prev_intra_df.empty:
            prev_subset = prev_intra_df[["strike", "ce_oi", "pe_oi"]]
            merged = oi_df.merge(
                prev_subset,
                on="strike",
                how="left",
                suffixes=("", "_prev"),
            )

            merged["ce_ΔOI"] = merged["ce_oi"] - merged["ce_oi_prev"].fillna(0)
            merged["pe_ΔOI"] = merged["pe_oi"] - merged["pe_oi_prev"].fillna(0)

            fig_delta = go.Figure()
            fig_delta.add_bar(
                name="Call ΔOI",
                x=merged["strike"],
                y=merged["ce_ΔOI"],
                marker_color="red",
            )
            fig_delta.add_bar(
                name="Put ΔOI",
                x=merged["strike"],
                y=merged["pe_ΔOI"],
                marker_color="green",
            )
            fig_delta.update_layout(
                barmode="group",
                title="Change in OI by strike (vs previous snapshot)",
                xaxis_title="Strike",
                yaxis_title="ΔOI",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(t=50, b=20),
            )
            st.plotly_chart(fig_delta, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("ΔOI chart appears after at least two refreshes (so we have a previous snapshot).")

    # ===== TAB 3: FULL OPTION CHAIN (ATM ±2 only) =====
    with tab_chain:
        if show_full_chain:
            st.subheader("Full Option Chain (ATM ±2 Strikes)")
            display_cols = [
                "strike",
                "ce_ltp", "ce_oi", "ce_vol",
                "pe_ltp", "pe_oi", "pe_vol",
            ]

            df_chain = df.sort_values("strike").reset_index(drop=True)
            atm_idx_chain = df_chain["abs_diff"].idxmin()
            start = max(atm_idx_chain - 2, 0)
            end = min(atm_idx_chain + 2, len(df_chain) - 1)
            df_window = df_chain.iloc[start:end + 1].copy()

            numeric_cols = ["ce_ltp", "ce_oi", "ce_vol", "pe_ltp", "pe_oi", "PE_Vol"]
            df_window = df_window[~(df_window[numeric_cols] == 0).all(axis=1)]

            st.dataframe(
                df_window[display_cols],
                use_container_width=True,
                height=260,
                hide_index=True,
            )
        else:
            st.info("Enable 'Show full option chain table' in sidebar to view this section.")

except Exception as e:
    st.error("Unhandled exception:")
    st.exception(e)
