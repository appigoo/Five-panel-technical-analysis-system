"""
五面板專業技術分析系統 (Streamlit 版)
仿照 富途/指南針類 專業看盤軟體的五面板架構：
  面板1：K線形態 + 中期趨勢緞帶 (BBI 多空線)
  面板2：技術專家情緒指標 (模擬面板1，用隨機指標KD做情緒判斷)
  面板3：主力(鯨魚) vs 散戶 籌碼分佈 (用 MFI 資金流量指標做主力力度代理)
  面板4：MACD
  面板5：三組 RSI (9 / 14 / 24)

注意：原圖表軟體（GCAND / TE / MCDX 籌碼分佈）是廠商專有黑箱演算法，
本App用公開、可驗證的技術指標「復刻同樣的視覺邏輯與判讀規則」，
數值與原軟體不會逐一對應，但買賣訊號的判讀方式一致。
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from plotly.subplots import make_subplots
import plotly.graph_objects as go

st.set_page_config(page_title="五面板技術分析系統", layout="wide")

# ----------------------------- 側邊欄 -----------------------------
st.sidebar.title("⚙️ 設定")
ticker = st.sidebar.text_input("股票代碼 (Ticker)", value="SPY").upper().strip()
interval_label = st.sidebar.selectbox("K線週期", ["日線 (1d)", "週線 (1wk)", "月線 (1mo)"], index=1)
interval_map = {"日線 (1d)": "1d", "週線 (1wk)": "1wk", "月線 (1mo)": "1mo"}
interval = interval_map[interval_label]
period = st.sidebar.selectbox("資料範圍", ["1y", "2y", "5y", "10y", "max"], index=2)
short_ma = st.sidebar.number_input("短期均線 (緞帶)", value=20, min_value=2)
mid_ma = st.sidebar.number_input("中期均線 (緞帶)", value=60, min_value=3)
show_insider = st.sidebar.checkbox("顯示內部人買超(Insider Buying)", value=True)
run = st.sidebar.button("🔄 讀取資料", use_container_width=True)

st.title("📊 五面板專業技術分析系統")
st.caption(f"{ticker} — {interval_label} — 範圍 {period}")


# ----------------------------- 資料抓取 -----------------------------
@st.cache_data(ttl=1800, show_spinner=True)
def load_data(tk, itv, per):
    df = yf.download(tk, period=per, interval=itv, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna()
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def load_insider_buys(tk):
    """抓取內部人(董監高)買進交易紀錄(SEC Form 4)。抓不到資料時回傳空表，不影響其他功能。"""
    try:
        t = yf.Ticker(tk)
        raw = t.get_insider_transactions()
        if raw is None or raw.empty:
            return pd.DataFrame(columns=["Date", "Insider", "Position", "Shares", "Value"])
        raw = raw.copy()
        date_col = "Start Date" if "Start Date" in raw.columns else raw.columns[0]
        raw["Date"] = pd.to_datetime(raw[date_col], errors="coerce")
        trans_col = "Transaction" if "Transaction" in raw.columns else None
        if trans_col:
            is_buy = raw[trans_col].astype(str).str.contains("Purchase|Buy", case=False, na=False)
            raw = raw[is_buy]
        out = pd.DataFrame({
            "Date": raw["Date"],
            "Insider": raw.get("Insider", "未知"),
            "Position": raw.get("Position", ""),
            "Shares": raw.get("Shares", np.nan),
            "Value": raw.get("Value", np.nan),
        }).dropna(subset=["Date"]).sort_values("Date")
        return out
    except Exception:
        return pd.DataFrame(columns=["Date", "Insider", "Position", "Shares", "Value"])


# ----------------------------- 指標計算 -----------------------------
def compute_all(df, short, mid):
    d = df.copy()
    d["MA_short"] = d["Close"].rolling(short).mean()
    d["MA_mid"] = d["Close"].rolling(mid).mean()
    d["ribbon_up"] = d["MA_short"] > d["MA_mid"]

    # ---- 面板1：K線訊號分類 ----
    win = 5
    swing_low = d["Low"].rolling(win).min() == d["Low"]
    swing_high = d["High"].rolling(win).max() == d["High"]
    bullish_body = d["Close"] > d["Open"]

    cat = pd.Series("none", index=d.index)
    cat[bullish_body & swing_low] = "red_signal"          # 看漲反轉訊號
    cat[(~bullish_body) & swing_high] = "yellow_signal"    # 看跌反轉訊號
    cat[(cat == "none") & bullish_body & d["ribbon_up"]] = "darkblue_cont"     # 上漲延續
    cat[(cat == "none") & (~bullish_body) & (~d["ribbon_up"])] = "lightblue_cont"  # 下跌延續
    cat[cat == "none"] = np.where(bullish_body[cat == "none"], "darkblue_cont", "lightblue_cont")
    d["candle_cat"] = cat

    # ---- 面板2：技術專家情緒 (KD 模擬) ----
    low_n = d["Low"].rolling(14).min()
    high_n = d["High"].rolling(14).max()
    rsv = (d["Close"] - low_n) / (high_n - low_n) * 100
    d["stoch_k"] = rsv.ewm(com=2).mean()
    d["stoch_d"] = d["stoch_k"].ewm(com=2).mean()
    d["te_bar"] = d["stoch_k"]
    trend10 = d["te_bar"].rolling(10).mean()
    slope_up = trend10.diff() > 0
    d["te_trend_up"] = trend10.where(slope_up)
    d["te_trend_down"] = trend10.where(~slope_up)

    # ---- 面板3：主力(鯨魚) vs 散戶 籌碼分佈 (MFI 代理) ----
    tp = (d["High"] + d["Low"] + d["Close"]) / 3
    mf = tp * d["Volume"]
    pos_mf = mf.where(tp > tp.shift(1), 0.0)
    neg_mf = mf.where(tp < tp.shift(1), 0.0)
    mfr = pos_mf.rolling(14).sum() / neg_mf.rolling(14).sum().replace(0, np.nan)
    mfi = 100 - (100 / (1 + mfr))
    d["whale_pct"] = mfi.clip(0, 100).fillna(50)
    d["daytrader_pct"] = 8.0
    d["retail_pct"] = (100 - d["whale_pct"] - d["daytrader_pct"]).clip(lower=0)

    # ---- 面板4：MACD ----
    ema12 = d["Close"].ewm(span=12, adjust=False).mean()
    ema26 = d["Close"].ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]

    # ---- 面板5：三組 RSI ----
    def rsi(series, n):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(n).mean()
        loss = (-delta.clip(upper=0)).rolling(n).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    d["rsi9"] = rsi(d["Close"], 9)
    d["rsi14"] = rsi(d["Close"], 14)
    d["rsi24"] = rsi(d["Close"], 24)

    # ---- 面板2 補充：費氏均線扇形 (rainbow ribbon) ----
    for n in (3, 5, 8, 13, 21, 34, 55, 89):
        d[f"ema{n}"] = d["Close"].ewm(span=n, adjust=False).mean()

    # ---- 面板3 補充：主力力度快/慢趨勢線 (短期/中期買訊) ----
    d["whale_fast"] = d["whale_pct"].ewm(span=5, adjust=False).mean()
    d["whale_slow"] = d["whale_pct"].ewm(span=20, adjust=False).mean()

    return d


def detect_patterns(d):
    """偵測四種標註訊號：賣出訊號／波動性凹陷(築底)／趨勢反轉燭／急漲前紅K"""
    d = d.copy()
    atr14 = (d["High"] - d["Low"]).rolling(14).mean()
    range5 = (d["High"] - d["Low"]).rolling(5).mean()
    ribbon_width = (d["MA_short"] - d["MA_mid"]).abs()
    vol_avg20 = d["Volume"].rolling(20).mean()

    near_top = d["Close"] >= d["Close"].rolling(60).max() * 0.95
    near_bottom = d["Close"] <= d["Close"].rolling(60).min() * 1.05
    squeeze = ribbon_width <= ribbon_width.rolling(60).min() * 1.15
    tight_range = range5 <= range5.rolling(60).min() * 1.15

    # 賣出訊號：高檔 + 均線糾結收斂 (類似圖中圓圈標示的頭部糾纏區)
    d["sell_signal"] = near_top & squeeze

    # 波動性凹陷：低檔 + 波動急縮到近期最低 (築底特徵)
    d["volatility_hole"] = near_bottom & tight_range

    # 趨勢反轉燭：長實體/長影線的爆量K棒，方向與過去5日趨勢相反
    trail_chg = d["Close"] - d["Close"].shift(5)
    body_dir = np.sign(d["Close"] - d["Open"])
    big_range = (d["High"] - d["Low"]) > 1.5 * atr14
    vol_spike = d["Volume"] > 1.3 * vol_avg20
    d["trend_reversal_candle"] = big_range & vol_spike & (np.sign(trail_chg) != body_dir) & (body_dir != 0)

    # 急漲前紅K：看漲反轉K線，且未來3根K棒累積漲幅>3% (歷史回顧標註用，非即時訊號)
    future_gain = d["Close"].shift(-3) / d["Close"] - 1
    d["bullish_before_surge"] = (d["candle_cat"] == "red_signal") & (future_gain > 0.03)

    # 【新增】即時強勢紅K訊號：只用當下可得資料，不看未來，可用於盤中即時判斷
    # 條件：紅K反轉訊號 + 爆量(>1.3倍均量) + 收盤落在當日高低區間的上緣(強勢收盤)
    close_position = (d["Close"] - d["Low"]) / (d["High"] - d["Low"]).replace(0, np.nan)
    strong_close = close_position >= 0.7
    d["realtime_bullish_signal"] = (d["candle_cat"] == "red_signal") & vol_spike & strong_close

    # 【新增1】關鍵動能K棒壓力/支撐位：過去N根K棒內成交量最大的那根K棒的高點，視為關鍵價位
    window = 40
    key_level = []
    vol_vals, high_vals = d["Volume"].values, d["High"].values
    for i in range(len(d)):
        lo = max(0, i - window + 1)
        seg_vol = vol_vals[lo:i + 1]
        idxmax = int(np.argmax(seg_vol))
        key_level.append(high_vals[lo + idxmax])
    d["key_level"] = key_level
    d["near_key_level"] = (d["Close"] - d["key_level"]).abs() / d["key_level"] < 0.03

    # 【新增2】中期持有訊號：主力(鯨魚)佔比持續上升(20根K棒斜率>0.5) 且已超過散戶佔比
    def slope(s):
        y = s.values
        x = np.arange(len(y))
        if np.isnan(y).any():
            return np.nan
        return np.polyfit(x, y, 1)[0]

    d["whale_slope20"] = d["whale_pct"].rolling(20).apply(slope, raw=False)
    d["mid_term_hold_signal"] = (d["whale_slope20"] > 0.5) & (d["whale_pct"] > d["retail_pct"])

    # 【新增3】波動性凹陷(高品質版)：凹陷發生前，緞帶必須是持續下降的藍色緞帶 (至少近15根K棒中60%為藍)
    pre_descending_ribbon = (~d["ribbon_up"]).shift(1).rolling(15).mean() >= 0.6
    d["volatility_hole_confirmed"] = d["volatility_hole"] & pre_descending_ribbon.fillna(False)

    # 【新增4】關鍵動能位突破確認：收盤價由下方站上關鍵動能位 (真正的買入確認訊號，而非只是測試)
    prev_close, prev_key = d["Close"].shift(1), d["key_level"].shift(1)
    d["key_level_breakout"] = (d["Close"] > d["key_level"]) & (prev_close <= prev_key)

    # 【新增5】高品質紅K：紅K反轉訊號 + 同時主力籌碼正在上升 (單獨出現的紅K意義較弱)
    d["red_candle_with_accum"] = (d["candle_cat"] == "red_signal") & (d["whale_slope20"] > 0.3)

    # 【新增6】強勢突破買入訊號：價格突破近20根K棒的前高(整理區間壓力) + 主力籌碼同步持續上升
    # 對應「連續突破前高、伴隨鯨魚積累、順勢加碼」的多段式趨勢延續買入邏輯
    prior_high_20 = d["High"].rolling(20).max().shift(1)
    breakout_new_high = d["Close"] > prior_high_20
    d["breakout_with_accumulation"] = breakout_new_high & (d["whale_slope20"] > 0.3)

    return d


def find_momentum_levels(d, vol_window=60, top_pct=0.8, min_gap_pct=0.03):
    """找出歷史上『關鍵動能K棒』的價位清單(結構性標記，非即時訊號)，
    用於預測下一個/下下個潛在阻力目標，對應作者解讀中的 $161.60→$193→$203 邏輯。"""
    swing_high = d["High"] == d["High"].rolling(11, center=True).max()
    vol_rank = d["Volume"].rolling(vol_window, min_periods=20).apply(
        lambda s: (s.iloc[-1] >= s).mean(), raw=False)
    is_momentum_bar = swing_high.fillna(False) & (vol_rank >= top_pct).fillna(False)
    levels = sorted(d.loc[is_momentum_bar, "High"].unique())
    deduped = []
    for lv in levels:
        if not deduped or (lv - deduped[-1]) / deduped[-1] > min_gap_pct:
            deduped.append(lv)
    return deduped


def attach_insider_signals(d, insider_df, window_days=3):
    """把內部人(董監高)買進紀錄對應到最接近的K棒，並判定是否與看漲K線同步出現(內部人確認訊號)"""
    d = d.copy()
    d["insider_buy"] = False
    d["insider_buy_note"] = ""
    if insider_df is None or insider_df.empty:
        d["internal_confirmed_signal"] = False
        return d

    idx = d.index
    for _, row in insider_df.iterrows():
        buy_date = row["Date"]
        if pd.isna(buy_date):
            continue
        diffs = np.abs((idx - buy_date).days)
        min_i = int(diffs.argmin())
        if diffs[min_i] <= window_days:
            d.iloc[min_i, d.columns.get_loc("insider_buy")] = True
            note = f"{row.get('Insider', '內部人')}買進"
            existing = d.iloc[min_i, d.columns.get_loc("insider_buy_note")]
            d.iloc[min_i, d.columns.get_loc("insider_buy_note")] = f"{existing}; {note}" if existing else note

    d["internal_confirmed_signal"] = d["insider_buy"] & d["candle_cat"].isin(["red_signal", "darkblue_cont"])
    return d


def ribbon_segments(d):
    """把中期趨勢緞帶依連續同狀態切段，回傳每段 (index範圍, 是否上漲)"""
    segs = []
    state = None
    start = None
    for i in range(len(d)):
        s = bool(d["ribbon_up"].iloc[i]) if not pd.isna(d["ribbon_up"].iloc[i]) else None
        if s != state:
            if state is not None and start is not None:
                segs.append((start, i, state))
            start = i
            state = s
    if state is not None and start is not None:
        segs.append((start, len(d), state))
    return segs


COLOR_MAP = {
    "red_signal": "#ff3b3b",
    "yellow_signal": "#ffd400",
    "darkblue_cont": "#0b3d91",
    "lightblue_cont": "#7ec8ff",
}


def build_chart(d, ticker, momentum_levels=None):
    fig = make_subplots(
        rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02,
        row_heights=[0.36, 0.14, 0.16, 0.16, 0.18],
        subplot_titles=(
            f"{ticker} K線形態 + 中期趨勢緞帶", "技術專家情緒指標",
            "主力(鯨魚) vs 散戶 籌碼分佈", "MACD", "RSI (9/14/24)"
        )
    )

    x = d.index

    # --- 中期趨勢緞帶 (紅=上漲 / 藍=下跌) ---
    for start, end, up in ribbon_segments(d):
        seg = d.iloc[max(start - 1, 0):end]
        fig.add_trace(go.Scatter(
            x=list(seg.index) + list(seg.index[::-1]),
            y=list(seg["MA_short"]) + list(seg["MA_mid"][::-1]),
            fill="toself",
            fillcolor="rgba(255,0,0,0.25)" if up else "rgba(30,100,255,0.25)",
            line=dict(width=0), showlegend=False, hoverinfo="skip"
        ), row=1, col=1)

    # --- 費氏均線扇形 (對應原圖中收斂/發散的多線交叉區，用於標示賣出訊號/築底) ---
    fib_periods = [3, 5, 8, 13, 21, 34, 55, 89]
    fan_colors = ["#ffffff", "#ffe082", "#ffb74d", "#ff8a65", "#f06292", "#ba68c8", "#7986cb", "#4fc3f7"]
    for n, c in zip(fib_periods, fan_colors):
        fig.add_trace(go.Scatter(
            x=x, y=d[f"ema{n}"], line=dict(color=c, width=1), opacity=0.55,
            name=f"EMA{n}", showlegend=False, hoverinfo="skip"
        ), row=1, col=1)

    # --- 關鍵動能K棒壓力/支撐位 (過去40根K棒內最大量K棒的高點) ---
    fig.add_trace(go.Scatter(
        x=x, y=d["key_level"], mode="lines", name="關鍵動能壓力/支撐位",
        line=dict(color="#40c4ff", width=1.2, dash="dot")
    ), row=1, col=1)
    near_mask = d["near_key_level"].fillna(False)
    if near_mask.sum() > 0:
        fig.add_trace(go.Scatter(
            x=x[near_mask], y=d["key_level"][near_mask], mode="markers",
            name="測試關鍵位", marker=dict(symbol="x", size=9, color="#40c4ff")
        ), row=1, col=1)

    # --- 未來動能目標價位 (作者解讀邏輯：突破後依序挑戰下一個歷史動能K棒高點) ---
    if momentum_levels:
        last_price = d["Close"].iloc[-1]
        upcoming = [lv for lv in momentum_levels if lv > last_price][:3]
        for lv in upcoming:
            fig.add_hline(y=lv, line_dash="dash", line_color="#ffd54f", row=1, col=1,
                          annotation_text=f"目標 ${lv:,.2f}", annotation_position="right",
                          annotation_font=dict(color="#ffd54f", size=10))

    # --- 四色K線 (依分類遮罩) ---
    for cat, color in COLOR_MAP.items():
        mask = d["candle_cat"] == cat
        if mask.sum() == 0:
            continue
        fig.add_trace(go.Candlestick(
            x=x[mask], open=d["Open"][mask], high=d["High"][mask],
            low=d["Low"][mask], close=d["Close"][mask],
            increasing_line_color=color, decreasing_line_color=color,
            increasing_fillcolor=color, decreasing_fillcolor=color,
            name=cat, showlegend=True
        ), row=1, col=1)

    # --- 面板1：型態標註 ---
    pattern_style = {
        "sell_signal": ("賣出訊號", "circle-open", "#ffd400", d["High"] * 1.02),
        "volatility_hole": ("波動性凹陷", "circle-open", "#00e5ff", d["Low"] * 0.98),
        "volatility_hole_confirmed": ("波動性凹陷(高品質築底)", "circle", "#00e5ff", d["Low"] * 0.94),
        "trend_reversal_candle": ("趨勢反轉燭", "diamond-open", "#ff9100", d["High"] * 1.04),
        "bullish_before_surge": ("急漲前紅K(回顧用)", "star-open", "#ff3b3b", d["Low"] * 0.95),
        "realtime_bullish_signal": ("即時強勢紅K", "triangle-up-open", "#76ff03", d["Low"] * 0.92),
        "red_candle_with_accum": ("高品質紅K(主力同步吸籌)", "star", "#ff1744", d["Low"] * 0.90),
        "key_level_breakout": ("突破關鍵位(買入確認)", "triangle-up", "#ffd54f", d["Low"] * 0.88),
        "internal_confirmed_signal": ("內部人買超+看漲K線同步", "star", "#ffab00", d["Low"] * 0.86),
        "breakout_with_accumulation": ("強勢突破買入(突破前高+主力吸籌)", "arrow-up", "#00e676", d["Low"] * 0.84),
    }
    for col, (label, symbol, color, ypos) in pattern_style.items():
        mask = d[col].fillna(False)
        if mask.sum() == 0:
            continue
        fig.add_trace(go.Scatter(
            x=x[mask], y=ypos[mask], mode="markers", name=label,
            marker=dict(symbol=symbol, size=13, color=color, line=dict(width=2, color=color))
        ), row=1, col=1)
        last_idx = d.index[mask][-1]
        fig.add_annotation(
            x=last_idx, y=ypos[mask].iloc[-1], text=label, showarrow=True,
            arrowhead=2, ax=0, ay=-30 if col in ("sell_signal", "trend_reversal_candle") else 30,
            font=dict(color=color, size=11), row=1, col=1
        )

    # --- 內部人買超標記 (獨立於K線類別，只要有買進紀錄都標出) ---
    insider_mask = d["insider_buy"].fillna(False) if "insider_buy" in d.columns else pd.Series(False, index=d.index)
    if insider_mask.sum() > 0:
        fig.add_trace(go.Scatter(
            x=x[insider_mask], y=d["High"][insider_mask] * 1.06, mode="markers",
            name="內部人買進申報(Form 4)", text=d["insider_buy_note"][insider_mask],
            marker=dict(symbol="diamond", size=11, color="#00e676", line=dict(width=1, color="white")),
            hovertemplate="%{text}<extra></extra>"
        ), row=1, col=1)

    # --- 面板2：技術專家 (情緒柱 + 費氏均線扇形) ---
    bar_colors = np.where(d["te_bar"] >= 50, "#ff3b3b", "#00c853")
    fig.add_trace(go.Bar(x=x, y=d["te_bar"], marker_color=bar_colors, name="情緒強度"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["te_trend_up"], line=dict(color="purple", width=2), name="中期上漲趨勢線"), row=2, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["te_trend_down"], line=dict(color="green", width=2), name="中期下跌趨勢線"), row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)

    # --- 面板3：主力 vs 散戶 ---
    fig.add_trace(go.Bar(x=x, y=d["whale_pct"], marker_color="#d32f2f", name="主力(鯨魚)%"), row=3, col=1)
    fig.add_trace(go.Bar(x=x, y=d["retail_pct"], marker_color="#2e7d32", name="散戶%"), row=3, col=1)
    fig.add_trace(go.Bar(x=x, y=d["daytrader_pct"], marker_color="#fbc02d", name="日內交易者%"), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["whale_fast"], line=dict(color="#00e5ff", width=1.5), name="短期買訊線"), row=3, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["whale_slow"], line=dict(color="#e040fb", width=1.5), name="中期買訊線"), row=3, col=1)
    hold_mask = d["mid_term_hold_signal"].fillna(False)
    if hold_mask.sum() > 0:
        fig.add_trace(go.Scatter(
            x=x[hold_mask], y=d["whale_pct"][hold_mask] + 3, mode="markers",
            name="中期持有訊號", marker=dict(symbol="triangle-up", size=8, color="white")
        ), row=3, col=1)
    fig.update_layout(barmode="stack")
    fig.add_hline(y=50, line_dash="dot", line_color="white", row=3, col=1)
    fig.add_hline(y=75, line_dash="dot", line_color="orange", row=3, col=1)

    # --- 面板4：MACD ---
    hist_colors = np.where(d["macd_hist"] >= 0, "#ff3b3b", "#00c853")
    fig.add_trace(go.Bar(x=x, y=d["macd_hist"], marker_color=hist_colors, name="MACD柱"), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["macd"], line=dict(color="yellow", width=1.3), name="快線 DIF"), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["macd_signal"], line=dict(color="white", width=1.3), name="慢線 DEA"), row=4, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=4, col=1)

    # --- 面板5：RSI ---
    fig.add_trace(go.Scatter(x=x, y=d["rsi9"], line=dict(color="#ff3b3b", width=1.3), name="RSI(9)"), row=5, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["rsi14"], line=dict(color="yellow", width=1.3), name="RSI(14)"), row=5, col=1)
    fig.add_trace(go.Scatter(x=x, y=d["rsi24"], line=dict(color="white", width=1.3), name="RSI(24)"), row=5, col=1)
    fig.add_hline(y=80, line_dash="dot", line_color="cyan", row=5, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=5, col=1)
    fig.add_hline(y=20, line_dash="dot", line_color="cyan", row=5, col=1)

    fig.update_layout(
        height=1200, template="plotly_dark", xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.01),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    for i in range(1, 6):
        fig.update_xaxes(rangeslider_visible=False, row=i, col=1)
    return fig


def verdict_table(d):
    last = d.iloc[-1]
    rows = []

    rows.append(["K線訊號", last["candle_cat"], "紅=看漲反轉／黃=看跌反轉／深藍=漲勢延續／淺藍=跌勢延續"])
    rows.append(["中期趨勢緞帶", "紅(上漲)" if last["ribbon_up"] else "藍(下跌)", f"短均{last['MA_short']:.2f} vs 中均{last['MA_mid']:.2f}"])
    rows.append(["技術專家情緒", f"{last['te_bar']:.1f}", "≥50 看漲情緒／<50 看跌情緒"])
    rows.append(["主力(鯨魚)佔比", f"{last['whale_pct']:.1f}%", "≥50% 有撐；≥75% 具備噴出動能"])
    rows.append(["主力籌碼斜率(20K棒)", f"{last['whale_slope20']:.2f}", ">0.5 代表主力持續加碼(籌碼翻紅趨勢)"])
    rows.append(["關鍵動能位", f"${last['key_level']:.2f}", "近40根K棒最大量K棒高點，為壓力/支撐參考位"])
    rows.append(["散戶佔比", f"{last['retail_pct']:.1f}%", "數值持續下降＝可能正在築底"])
    rows.append(["MACD", f"DIF {last['macd']:.3f} / DEA {last['macd_signal']:.3f}", "DIF>DEA 且柱狀翻紅＝金叉看漲"])
    rows.append(["RSI(9/14/24)", f"{last['rsi9']:.1f} / {last['rsi14']:.1f} / {last['rsi24']:.1f}", "20-30超賣；80-95超買"])

    pattern_labels = {
        "sell_signal": ("賣出訊號", "高檔均線糾結收斂"),
        "volatility_hole": ("波動性凹陷", "低檔波動急縮"),
        "volatility_hole_confirmed": ("波動性凹陷(高品質築底)", "凹陷前需搭配持續下降的藍色緞帶，才是高品質築底訊號"),
        "trend_reversal_candle": ("趨勢反轉燭", "爆量長K棒方向反轉"),
        "bullish_before_surge": ("急漲前紅K(回顧用)", "⚠️用未來資料標註，僅供歷史回顧學習"),
        "realtime_bullish_signal": ("即時強勢紅K", "紅K反轉+爆量+強勢收盤，可即時使用"),
        "red_candle_with_accum": ("高品質紅K", "紅K反轉+主力籌碼同步上升，單獨出現的紅K意義較弱"),
        "key_level_breakout": ("突破關鍵位(買入確認)", "價格站上關鍵動能位，較單純測試更可靠的買入確認"),
        "mid_term_hold_signal": ("中期持有訊號", "主力籌碼持續增加且超過散戶，適合續抱"),
        "internal_confirmed_signal": ("內部人買超+看漲K線同步", "真實SEC Form 4申報買進，比技術代理指標更直接的確認"),
        "breakout_with_accumulation": ("強勢突破買入", "價格突破近20根K棒前高，且主力籌碼同步持續上升，可順勢加碼"),
    }
    for col, (label, rule) in pattern_labels.items():
        occurred = d[col].fillna(False)
        if occurred.any():
            bars_ago = len(d) - 1 - np.where(occurred.values)[0][-1]
            rows.append([label, f"最近一次：{bars_ago} 根K棒前", rule])
        else:
            rows.append([label, "近期無", rule])

    if "insider_buy" in d.columns and d["insider_buy"].any():
        bars_ago = len(d) - 1 - np.where(d["insider_buy"].fillna(False).values)[0][-1]
        rows.append(["內部人買進(任何K線)", f"最近一次：{bars_ago} 根K棒前", "來自SEC Form 4申報，真實內部人交易紀錄"])
    else:
        rows.append(["內部人買進(任何K線)", "近期無 / 無資料", "來自SEC Form 4申報，真實內部人交易紀錄"])

    return pd.DataFrame(rows, columns=["指標", "目前數值", "判讀規則"])


def generate_call(d):
    last = d.iloc[-1]
    score = 0
    reasons = []

    if last["ribbon_up"]:
        score += 1; reasons.append("中期緞帶為紅（多頭排列）")
    else:
        score -= 1; reasons.append("中期緞帶為藍（空頭排列）")

    if last["candle_cat"] == "red_signal":
        if last["whale_slope20"] > 0.3:
            score += 2; reasons.append("K線為紅K反轉，且主力籌碼同步上升 → 高品質訊號")
        else:
            score += 0; reasons.append("K線為紅K反轉，但主力籌碼未同步上升 → 單獨出現，訊號較弱")
    elif last["candle_cat"] == "darkblue_cont":
        score += 1; reasons.append("K線屬多頭延續類別")
    else:
        score -= 1; reasons.append("K線屬空頭類別")

    if last["te_bar"] >= 50:
        score += 1; reasons.append(f"情緒指標{last['te_bar']:.0f}，偏多")
    else:
        score -= 1; reasons.append(f"情緒指標{last['te_bar']:.0f}，偏空")

    if last["whale_pct"] >= 75:
        score += 2; reasons.append(f"主力佔比{last['whale_pct']:.1f}%，達噴出門檻")
    elif last["whale_pct"] >= 50:
        score += 1; reasons.append(f"主力佔比{last['whale_pct']:.1f}%，具備支撐")
    else:
        score -= 1; reasons.append(f"主力佔比僅{last['whale_pct']:.1f}%，籌碼偏弱")

    if last["macd"] > last["macd_signal"]:
        score += 1; reasons.append("MACD 金叉向上")
    else:
        score -= 1; reasons.append("MACD 死叉向下")

    if last["rsi14"] <= 30:
        score += 1; reasons.append(f"RSI(14)={last['rsi14']:.1f}，接近超賣區")
    elif last["rsi14"] >= 80:
        score -= 1; reasons.append(f"RSI(14)={last['rsi14']:.1f}，接近超買區")

    lookback = 3
    recent = d.iloc[-lookback:]
    if recent["sell_signal"].fillna(False).any():
        score -= 2; reasons.append(f"近{lookback}根K棒內出現「賣出訊號」(高檔均線糾結)")
    if recent["volatility_hole_confirmed"].fillna(False).any():
        score += 3; reasons.append(f"近{lookback}根K棒內出現「波動性凹陷+下降藍緞帶」高品質築底訊號")
    elif recent["volatility_hole"].fillna(False).any():
        score += 2; reasons.append(f"近{lookback}根K棒內出現「波動性凹陷」(低檔築底特徵)")
    if recent["trend_reversal_candle"].fillna(False).any():
        rc = recent[recent["trend_reversal_candle"].fillna(False)].iloc[-1]
        if rc["Close"] > rc["Open"]:
            score += 2; reasons.append("近期出現偏多方向的趨勢反轉燭")
        else:
            score -= 2; reasons.append("近期出現偏空方向的趨勢反轉燭")
    if recent["realtime_bullish_signal"].fillna(False).any():
        score += 1; reasons.append(f"近{lookback}根K棒內出現「即時強勢紅K」(反轉+爆量+強勢收盤)")
    if last["mid_term_hold_signal"]:
        score += 2; reasons.append(f"主力籌碼持續增加(斜率{last['whale_slope20']:.2f})且已超過散戶，屬「中期持有訊號」")
    if last["near_key_level"]:
        near_desc = "壓力" if last["Close"] < last["key_level"] else "支撐"
        reasons.append(f"目前價格接近關鍵動能位 ${last['key_level']:.2f}，該處可能形成{near_desc}，留意反應")
    if recent["key_level_breakout"].fillna(False).any():
        score += 2; reasons.append(f"近{lookback}根K棒內站上關鍵動能位 ${last['key_level']:.2f} → 買入確認訊號")
    if "internal_confirmed_signal" in recent.columns and recent["internal_confirmed_signal"].fillna(False).any():
        score += 3; reasons.append("近期出現「內部人真實買進+看漲K線」同步，屬最高等級確認訊號")
    elif "insider_buy" in recent.columns and recent["insider_buy"].fillna(False).any():
        score += 1; reasons.append("近期有內部人買進申報，但未與看漲K線同步，訊號強度中等")
    if recent["breakout_with_accumulation"].fillna(False).any():
        score += 3; reasons.append("近期出現「突破前高+主力籌碼同步上升」強勢突破買入訊號，可順勢加碼")

    if score >= 4:
        call = "🟢 偏多操作 — 順勢做多／逢回布局"
    elif score <= -4:
        call = "🔴 偏空操作 — 觀望或考慮避險／減碼"
    else:
        call = "🟡 中性 — 訊號分歧，建議縮小部位等待確認"

    price = last["Close"]
    return call, score, reasons, price


def generate_ai_prompt(d, ticker, interval_label, period, call, score, reasons, price, momentum_levels=None):
    """組合成一段結構化文字，讓使用者可複製貼給任何AI大模型(ChatGPT/Claude/Gemini等)進一步分析"""
    last = d.iloc[-1]
    date_str = str(d.index[-1].date())

    pattern_labels = {
        "sell_signal": "賣出訊號", "volatility_hole": "波動性凹陷",
        "volatility_hole_confirmed": "波動性凹陷+下降藍緞帶(高品質築底)",
        "trend_reversal_candle": "趨勢反轉燭", "bullish_before_surge": "急漲前紅K(回顧用，用未來資料標註)",
        "realtime_bullish_signal": "即時強勢紅K(可即時使用)",
        "red_candle_with_accum": "高品質紅K(紅K+主力籌碼同步上升)",
        "key_level_breakout": "突破關鍵位(買入確認)",
        "mid_term_hold_signal": "中期持有訊號(主力籌碼持續增加)",
        "internal_confirmed_signal": "內部人買超+看漲K線同步(最高等級確認)",
        "breakout_with_accumulation": "強勢突破買入(突破前高+主力吸籌)",
    }
    pattern_lines = []
    for col, label in pattern_labels.items():
        occurred = d[col].fillna(False)
        if occurred.any():
            bars_ago = len(d) - 1 - np.where(occurred.values)[0][-1]
            pattern_lines.append(f"- {label}：最近一次出現在 {bars_ago} 根K棒前")
        else:
            pattern_lines.append(f"- {label}：近期未出現")

    candle_cat_zh = {
        "red_signal": "紅K(看漲反轉訊號)", "yellow_signal": "黃K(看跌反轉訊號)",
        "darkblue_cont": "深藍K(漲勢延續)", "lightblue_cont": "淺藍K(跌勢延續)",
    }.get(last["candle_cat"], last["candle_cat"])

    targets_text = "無明確歷史動能位資料"
    if momentum_levels:
        upcoming = [lv for lv in momentum_levels if lv > price][:3]
        if upcoming:
            targets_text = " → ".join(f"${lv:,.2f}" for lv in upcoming)

    prompt = f"""你是一位專業的股票技術分析師，請根據以下由多面板技術分析系統計算出的數據，
針對「{ticker}」給我你的獨立判斷與操作建議。請特別評估目前系統給出的判讀是否合理，
如果你有不同看法請直接指出並說明理由，最後請給我明確的多/空/中性結論與理由。

【基本資訊】
股票代碼：{ticker}
資料日期：{date_str}
K線週期：{interval_label}（資料範圍：{period}）
現價：${price:,.2f}

【面板1：K線形態 + 中期趨勢緞帶】
今日K線分類：{candle_cat_zh}
中期趨勢緞帶：{"紅色(多頭排列)" if last["ribbon_up"] else "藍色(空頭排列)"}
短期均線：{last["MA_short"]:.2f}　中期均線：{last["MA_mid"]:.2f}
關鍵動能壓力/支撐位：${last["key_level"]:.2f}（近40根K棒最大量K棒高點；{"目前正在測試此關鍵位" if last["near_key_level"] else "目前價格未接近此關鍵位"}）
突破後潛在動能目標序列：{targets_text}

【面板1 自動標註型態】
{chr(10).join(pattern_lines)}

【內部人買賣紀錄 (SEC Form 4，真實資料非技術代理指標)】
{"近期有內部人買進：" + str(int(d["insider_buy"].tail(10).sum())) + " 筆(近10根K棒內)" if "insider_buy" in d.columns and d["insider_buy"].any() else "近期無內部人買進申報資料，或該股票查無公開Form 4紀錄"}

【面板2：技術專家情緒指標】
情緒強度值：{last["te_bar"]:.1f}（≥50為偏多情緒，<50為偏空情緒）

【面板3：主力(鯨魚) vs 散戶 籌碼分佈】
主力(鯨魚)佔比：{last["whale_pct"]:.1f}%（≥50%代表有撐，≥75%代表具備噴出動能）
主力籌碼20K棒斜率：{last["whale_slope20"]:.2f}（>0.5代表主力持續加碼，籌碼正從散戶翻向主力）
散戶佔比：{last["retail_pct"]:.1f}%（數值持續下降可能代表正在築底）
短期買訊線：{last["whale_fast"]:.1f}　中期買訊線：{last["whale_slow"]:.1f}

【面板4：MACD】
DIF(快線)：{last["macd"]:.3f}
DEA(慢線)：{last["macd_signal"]:.3f}
MACD柱：{last["macd_hist"]:.3f}（>0代表柱翻紅偏多，<0代表柱翻綠偏空）

【面板5：RSI】
RSI(9)：{last["rsi9"]:.1f}
RSI(14)：{last["rsi14"]:.1f}
RSI(24)：{last["rsi24"]:.1f}
（20-30為超賣區，80-95為超買區）

【本系統目前的綜合判讀】
結論：{call}
綜合評分：{score}（區間 -10 ~ +24，正值偏多、負值偏空）
判讀依據：
{chr(10).join("- " + r for r in reasons)}

請你根據以上數據，給出你自己的技術面分析、可能的風險點、以及你認為較合理的操作策略(進場點/停損點/目標價，若能給出具體數字更好)。"""
    return prompt


# ----------------------------- 主流程 -----------------------------
if ticker:
    df = load_data(ticker, interval, period)
    if df.empty:
        st.error("查無資料，請確認代碼是否正確。")
    else:
        d = compute_all(df, short_ma, mid_ma)
        d = detect_patterns(d)
        insider_df = load_insider_buys(ticker) if show_insider else pd.DataFrame(columns=["Date", "Insider", "Position", "Shares", "Value"])
        d = attach_insider_signals(d, insider_df)
        momentum_levels = find_momentum_levels(d)
        fig = build_chart(d, ticker, momentum_levels)
        st.plotly_chart(fig, use_container_width=True)

        if show_insider and not insider_df.empty:
            with st.expander(f"🟢 內部人買進紀錄 (SEC Form 4，共{len(insider_df)}筆)"):
                st.dataframe(insider_df, use_container_width=True, hide_index=True)
        elif show_insider:
            st.caption("ℹ️ 查無此股票的內部人買進公開紀錄(可能該公司近期無申報，或資料來源暫時無法取得)。")

        st.subheader("📋 當前讀值總表")
        st.dataframe(verdict_table(d), use_container_width=True, hide_index=True)

        call, score, reasons, price = generate_call(d)
        st.subheader("🎯 綜合判讀結論")
        st.markdown(f"### {call}")
        st.markdown(f"**現價：${price:,.2f}｜綜合評分：{score} (區間 -10 ~ +24)**")
        st.markdown("**判讀依據：**")
        for r in reasons:
            st.markdown(f"- {r}")

        upcoming_targets = [lv for lv in momentum_levels if lv > price][:3]
        if upcoming_targets:
            st.markdown("**突破後潛在動能目標序列：** " + " → ".join(f"${lv:,.2f}" for lv in upcoming_targets))
            st.caption("依歷史關鍵動能K棒高點由近到遠排列，僅供結構性參考，非保證價位。")

        st.info("⚠️ 本App所有『主力/散戶籌碼』與『技術專家情緒』面板為公開技術指標(MFI/隨機指標KD)之演算法代理，"
                "非原軟體(GCAND/MCDX)的黑箱公式；判讀邏輯一致，數值僅供參考，非投資建議。")

        st.subheader("🤖 AI 分析提示詞 (可複製貼給任何AI大模型)")
        st.caption("將下方文字複製，貼到 ChatGPT / Claude / Gemini 等任何AI聊天視窗，即可請它針對本次讀值給出獨立判斷。")
        ai_prompt = generate_ai_prompt(d, ticker, interval_label, period, call, score, reasons, price, momentum_levels)
        st.code(ai_prompt, language="text")
