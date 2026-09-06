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


def build_chart(d, ticker):
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

    # --- 面板1：四種型態標註 (賣出訊號/波動性凹陷/趨勢反轉燭/急漲前紅K) ---
    pattern_style = {
        "sell_signal": ("賣出訊號", "circle-open", "#ffd400", d["High"] * 1.02),
        "volatility_hole": ("波動性凹陷(築底)", "circle-open", "#00e5ff", d["Low"] * 0.98),
        "trend_reversal_candle": ("趨勢反轉燭", "diamond-open", "#ff9100", d["High"] * 1.04),
        "bullish_before_surge": ("急漲前紅K", "star-open", "#ff3b3b", d["Low"] * 0.95),
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
    rows.append(["散戶佔比", f"{last['retail_pct']:.1f}%", "數值持續下降＝可能正在築底"])
    rows.append(["MACD", f"DIF {last['macd']:.3f} / DEA {last['macd_signal']:.3f}", "DIF>DEA 且柱狀翻紅＝金叉看漲"])
    rows.append(["RSI(9/14/24)", f"{last['rsi9']:.1f} / {last['rsi14']:.1f} / {last['rsi24']:.1f}", "20-30超賣；80-95超買"])

    pattern_labels = {
        "sell_signal": "賣出訊號", "volatility_hole": "波動性凹陷(築底)",
        "trend_reversal_candle": "趨勢反轉燭", "bullish_before_surge": "急漲前紅K",
    }
    for col, label in pattern_labels.items():
        occurred = d[col].fillna(False)
        if occurred.any():
            bars_ago = len(d) - 1 - np.where(occurred.values)[0][-1]
            rows.append([label, f"最近一次：{bars_ago} 根K棒前", "型態出現代表對應轉折/警示區"])
        else:
            rows.append([label, "近期無", "型態出現代表對應轉折/警示區"])

    return pd.DataFrame(rows, columns=["指標", "目前數值", "判讀規則"])


def generate_call(d):
    last = d.iloc[-1]
    score = 0
    reasons = []

    if last["ribbon_up"]:
        score += 1; reasons.append("中期緞帶為紅（多頭排列）")
    else:
        score -= 1; reasons.append("中期緞帶為藍（空頭排列）")

    if last["candle_cat"] in ("red_signal", "darkblue_cont"):
        score += 1; reasons.append("K線屬多頭類別")
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
    if recent["volatility_hole"].fillna(False).any():
        score += 2; reasons.append(f"近{lookback}根K棒內出現「波動性凹陷」(低檔築底特徵)")
    if recent["trend_reversal_candle"].fillna(False).any():
        rc = recent[recent["trend_reversal_candle"].fillna(False)].iloc[-1]
        if rc["Close"] > rc["Open"]:
            score += 2; reasons.append("近期出現偏多方向的趨勢反轉燭")
        else:
            score -= 2; reasons.append("近期出現偏空方向的趨勢反轉燭")

    if score >= 4:
        call = "🟢 偏多操作 — 順勢做多／逢回布局"
    elif score <= -4:
        call = "🔴 偏空操作 — 觀望或考慮避險／減碼"
    else:
        call = "🟡 中性 — 訊號分歧，建議縮小部位等待確認"

    price = last["Close"]
    return call, score, reasons, price


# ----------------------------- 主流程 -----------------------------
if ticker:
    df = load_data(ticker, interval, period)
    if df.empty:
        st.error("查無資料，請確認代碼是否正確。")
    else:
        d = compute_all(df, short_ma, mid_ma)
        d = detect_patterns(d)
        fig = build_chart(d, ticker)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("📋 當前讀值總表")
        st.dataframe(verdict_table(d), use_container_width=True, hide_index=True)

        call, score, reasons, price = generate_call(d)
        st.subheader("🎯 綜合判讀結論")
        st.markdown(f"### {call}")
        st.markdown(f"**現價：${price:,.2f}｜綜合評分：{score} (區間 -10 ~ +11)**")
        st.markdown("**判讀依據：**")
        for r in reasons:
            st.markdown(f"- {r}")

        st.info("⚠️ 本App所有『主力/散戶籌碼』與『技術專家情緒』面板為公開技術指標(MFI/隨機指標KD)之演算法代理，"
                "非原軟體(GCAND/MCDX)的黑箱公式；判讀邏輯一致，數值僅供參考，非投資建議。")
