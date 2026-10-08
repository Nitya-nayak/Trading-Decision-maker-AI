import time, requests, numpy as np, pandas as pd, streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Trading Decision & Risk Analysis", page_icon="📈", layout="wide")
FEE = 0.001
BPY = {"5m": 105120, "15m": 35040, "1h": 8760, "4h": 2190, "1d": 365}
FEATS = ["rsi", "macd_h", "pctb", "ret1", "ret5", "vol20", "volchg", "trend"]
UP, DN, HOLD, INK = "#5fd1a5", "#f2776b", "#e5b95c", "#e8e4d9"
HOSTS = ["https://api.binance.com", "https://data-api.binance.vision"]
COINS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600&family=IBM+Plex+Sans:wght@400;500&display=swap');
html,body,.stApp{font-family:'IBM Plex Sans',sans-serif}
.stApp{background:#101820;color:#e8e4d9}
[data-testid=stSidebar]{background:#0c131a;border-right:1px solid #2a3b4a}
.title{font-family:Fraunces,serif;font-size:2.3rem;line-height:1.15;margin-bottom:4px}
.sub{color:#9fb0bd;margin-bottom:22px}
.sig{border:1px solid #2a3b4a;border-radius:10px;padding:28px 32px;background:#17232e;height:100%}
.sig .w{font-family:Fraunces,serif;font-size:5.5rem;line-height:1;margin:6px 0}
.sig .s{color:#9fb0bd}
.BUY{color:#5fd1a5}.SELL{color:#f2776b}.HOLD{color:#e5b95c}
.stat{border-left:2px solid #2a3b4a;padding:2px 0 2px 14px;margin-bottom:12px}
.stat .l{color:#9fb0bd;font-size:.85rem}.stat .v{font-family:Fraunces,serif;font-size:1.6rem}
</style>""", unsafe_allow_html=True)


@st.cache_data(ttl=20)
def get_klines(sym, iv, limit=1000):
    for h in HOSTS:
        try:
            r = requests.get(f"{h}/api/v3/klines", params=dict(symbol=sym, interval=iv, limit=limit), timeout=10)
            r.raise_for_status()
            df = pd.DataFrame(r.json()).iloc[:, :6]
            df.columns = ["time", "open", "high", "low", "close", "volume"]
            df["time"] = pd.to_datetime(df["time"], unit="ms")
            return df.set_index("time").astype(float)
        except Exception:
            continue
    return None


def add_ind(d):
    d = d.copy(); c = d.close; dl = c.diff()
    up, dn = dl.clip(lower=0).rolling(14).mean(), (-dl.clip(upper=0)).rolling(14).mean()
    d["rsi"] = 100 - 100 / (1 + up / dn)
    d["macd"] = c.ewm(span=12).mean() - c.ewm(span=26).mean()
    d["macd_s"] = d.macd.ewm(span=9).mean(); d["macd_h"] = d.macd - d.macd_s
    d["sma20"], d["sma50"] = c.rolling(20).mean(), c.rolling(50).mean()
    sd = c.rolling(20).std(); d["bb_u"], d["bb_l"] = d.sma20 + 2 * sd, d.sma20 - 2 * sd
    d["pctb"] = (c - d.bb_l) / (d.bb_u - d.bb_l)
    tr = pd.concat([d.high - d.low, (d.high - c.shift()).abs(), (d.low - c.shift()).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()
    d["ret1"], d["ret5"] = c.pct_change(), c.pct_change(5)
    d["vol20"] = d.ret1.rolling(20).std(); d["volchg"] = d.volume / d.volume.rolling(20).mean()
    d["trend"] = d.sma20 / d.sma50 - 1
    return d


@st.cache_resource(show_spinner="Training the model on recent candles...")
def train(sym, iv, _raw, stamp):
    d = add_ind(_raw); fut = d.close.shift(-3)
    d["y"] = (fut > d.close).astype(float).where(fut.notna())
    d = d.dropna(subset=FEATS + ["y"]); cut = int(len(d) * .7)
    m = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=10, random_state=42, n_jobs=-1)
    m.fit(d[FEATS].iloc[:cut], d.y.iloc[:cut])
    acc = (m.predict(d[FEATS].iloc[cut:]) == d.y.iloc[cut:]).mean()
    return m, acc, d.index[cut]


def backtest(m, d, start, thr):
    t = d.loc[start:].dropna(subset=FEATS).copy()
    t["p"] = m.predict_proba(t[FEATS])[:, 1]
    eq, pos, entry, curve, pnls = 1.0, 0, 0.0, [], []
    for i in range(len(t) - 1):
        p, c, n = t.p.iloc[i], t.close.iloc[i], t.close.iloc[i + 1]
        if pos == 0 and p >= thr: pos, entry, eq = 1, c, eq * (1 - FEE)
        elif pos == 1 and p <= 1 - thr: pos, eq = 0, eq * (1 - FEE); pnls.append(c / entry - 1 - 2 * FEE)
        if pos: eq *= n / c
        curve.append(eq)
    return pd.Series(curve, index=t.index[1:]), t.close.iloc[1:] / t.close.iloc[0], pnls


def risk(d, iv):
    r = d.close.pct_change().dropna(); k = np.sqrt(BPY[iv])
    cum = (1 + r).cumprod(); mdd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    ann = r.std() * k * 100; score = float(min(100, 0.6 * ann + 0.8 * abs(mdd)))
    return dict(ann=ann, var=np.percentile(r, 5) * 100, mdd=mdd, sharpe=r.mean() / r.std() * k,
                score=score, level="Low" if score < 35 else "Medium" if score < 65 else "High")


def style(fig, h=460):
    fig.update_layout(height=h, margin=dict(l=0, r=0, t=10, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color=INK), xaxis_rangeslider_visible=False,
                      legend=dict(orientation="h", y=1.06))
    fig.update_xaxes(gridcolor="#1f2f3c"); fig.update_yaxes(gridcolor="#1f2f3c")
    return fig


def jget(path, **params):
    for h in HOSTS:
        try:
            t0 = time.time(); r = requests.get(h + path, params=params, timeout=8); r.raise_for_status()
            return r.json(), h, (time.time() - t0) * 1000
        except Exception:
            continue
    return None, None, None


@st.cache_data(ttl=60)
def fear_greed():
    try:
        j = requests.get("https://api.alternative.me/fng/?limit=1", timeout=8).json()["data"][0]
        return int(j["value"]), j["value_classification"]
    except Exception:
        return None, None


@st.cache_data(ttl=300, show_spinner="Scanning all coins...")
def scan(iv, thr):
    rows = []
    for s in COINS:
        r = get_klines(s, iv)
        if r is None: continue
        m, a, _ = train(s, iv, r, str(r.index[-1]))
        dd = add_ind(r); pp = float(m.predict_proba(dd[FEATS].iloc[[-1]])[0, 1]); k = risk(dd, iv)
        rows.append({"Coin": s, "Price": round(dd.close.iloc[-1], 4), "Up-probability %": round(pp * 100),
                     "Signal": "BUY" if pp >= thr else "SELL" if pp <= 1 - thr else "HOLD",
                     "Risk": f"{k['level']} ({k['score']:.0f})", "Model accuracy %": round(a * 100, 1)})
    return pd.DataFrame(rows)


def monte_carlo(d, price, n=1000, steps=50):
    r = d.close.pct_change().dropna().values
    return price * np.cumprod(1 + np.random.choice(r, (n, steps)), axis=1)


# ---------- sidebar ----------
sym = st.sidebar.selectbox("Coin", COINS)
iv = st.sidebar.selectbox("Candle size", list(BPY), index=1)
cap = st.sidebar.number_input("Paper capital (USDT)", 1000.0, 1e7, 10000.0, step=1000.0)
thr = st.sidebar.slider("Confidence needed to act", 0.50, 0.80, 0.58, 0.01)
auto = st.sidebar.toggle("Refresh every 30 seconds")

raw = get_klines(sym, iv)
if raw is None:
    st.error("Could not reach Binance. Check your internet connection or VPN, then reload."); st.stop()
d = add_ind(raw)
model, acc, test_start = train(sym, iv, raw, str(raw.index[-1]))
L = d.iloc[-1]; price, atr = L.close, L.atr
p = float(model.predict_proba(d[FEATS].iloc[[-1]])[0, 1])
signal = "BUY" if p >= thr else "SELL" if p <= 1 - thr else "HOLD"
rk = risk(d, iv)

# ---------- paper wallet ----------
S = st.session_state
if S.get("sym") != sym:
    S.update(sym=sym, start=cap, cash=cap, qty=0.0, entry=0.0, stop=0.0, tp=0.0, log=[])


def sell(why):
    S.cash += S.qty * price * (1 - FEE)
    S.log.append(dict(time=pd.Timestamp.now().strftime("%H:%M:%S"), side="SELL", price=round(price, 2),
                      qty=round(S.qty, 6), pnl=round((price - S.entry) * S.qty, 2), reason=why))
    S.qty = 0.0


def buy():
    q = min(S.cash, S.cash * 0.01 / (2 * atr) * price)
    S.qty, S.entry, S.stop, S.tp = q * (1 - FEE) / price, price, price - 2 * atr, price + 4 * atr
    S.cash -= q
    S.log.append(dict(time=pd.Timestamp.now().strftime("%H:%M:%S"), side="BUY", price=round(price, 2),
                      qty=round(S.qty, 6), pnl=0.0, reason="AI BUY signal"))


if S.qty > 0 and price <= S.stop: sell("Stop-loss hit")
elif S.qty > 0 and price >= S.tp: sell("Take-profit hit")
equity = S.cash + S.qty * price

# ---------- header ----------
st.markdown('<div class="title">Trading decisions, with the risk shown first</div>'
            '<div class="sub">Live Binance data, a random forest signal and a paper wallet. '
            'For education only; no profit is guaranteed.</div>', unsafe_allow_html=True)
tick, host, ms = jget("/api/v3/ticker/24hr", symbol=sym)
if tick:
    st.success(f"Live from Binance ({host.replace('https://', '')}), responded in {ms:.0f} ms. "
               f"Ticker price {float(tick['lastPrice']):,.2f} USDT, 24h change {float(tick['priceChangePercent']):+.2f}%. "
               f"Last candle opened {raw.index[-1]:%d %b %H:%M} UTC (India time is UTC+5:30).")
else:
    st.warning("Binance ticker did not respond, so the data on screen may be old.")
a, b = st.columns([3, 2])
a.markdown(f'<div class="sig"><div class="s">{sym} on {iv} candles</div><div class="w {signal}">{signal}</div>'
           f'<div class="s">Chance price is higher in 3 candles: {p*100:.0f}%. Acts only above {thr*100:.0f}% or below {100-thr*100:.0f}%.</div></div>',
           unsafe_allow_html=True)
chg = (price / d.close.iloc[-2] - 1) * 100
for l, v in [("Price (USDT)", f"{price:,.2f}  ({chg:+.2f}%)"), ("Risk level", f"{rk['level']} ({rk['score']:.0f}/100)"),
             ("Model accuracy on unseen candles", f"{acc*100:.1f}%")]:
    b.markdown(f'<div class="stat"><div class="l">{l}</div><div class="v">{v}</div></div>', unsafe_allow_html=True)

t1, t2, t3, t4, t5, t6, t7 = st.tabs(["Chart", "Risk analysis", "Backtest", "Paper wallet",
                                      "Monte Carlo", "Market scanner", "Live market"])

with t1:
    x = d.iloc[-150:]
    f = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[.6, .2, .2], vertical_spacing=.03)
    f.add_trace(go.Candlestick(x=x.index, open=x.open, high=x.high, low=x.low, close=x.close, name="Price",
                               increasing_line_color=UP, decreasing_line_color=DN), 1, 1)
    for col, nm, clr in [("bb_u", "Upper band", "#5d7385"), ("bb_l", "Lower band", "#5d7385"), ("sma20", "SMA 20", HOLD)]:
        f.add_trace(go.Scatter(x=x.index, y=x[col], name=nm, line=dict(width=1, color=clr)), 1, 1)
    f.add_trace(go.Scatter(x=x.index, y=x.rsi, name="RSI", line=dict(color=INK, width=1.5)), 2, 1)
    f.add_hline(y=70, line_dash="dot", line_color=DN, row=2, col=1); f.add_hline(y=30, line_dash="dot", line_color=UP, row=2, col=1)
    f.add_trace(go.Bar(x=x.index, y=x.macd_h, name="MACD histogram", marker_color=[UP if v > 0 else DN for v in x.macd_h]), 3, 1)
    st.plotly_chart(style(f, 620), use_container_width=True)
    st.markdown("**Why the model says " + signal + "**")
    st.markdown(f"- RSI is {L.rsi:.0f}: " + ("oversold, a bounce is possible" if L.rsi < 30 else "overbought, a pullback is possible" if L.rsi > 70 else "neutral"))
    st.markdown("- MACD momentum is " + ("positive" if L.macd_h > 0 else "negative"))
    st.markdown("- SMA 20 is " + ("above" if L.trend > 0 else "below") + " SMA 50, so the short trend is " + ("up" if L.trend > 0 else "down"))
    st.markdown("- Price is " + ("near the lower Bollinger band" if L.pctb < .15 else "near the upper Bollinger band" if L.pctb > .85 else "inside the Bollinger bands"))

with t2:
    g, m = st.columns([2, 3])
    gf = go.Figure(go.Indicator(mode="gauge+number", value=rk["score"], title=dict(text=f"{rk['level']} risk"),
                                gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=INK), steps=[
                                    dict(range=[0, 35], color="#1f4d44"), dict(range=[35, 65], color="#5b4a1f"),
                                    dict(range=[65, 100], color="#5a2a28")])))
    g.plotly_chart(style(gf, 300), use_container_width=True)
    c1, c2 = m.columns(2)
    c1.metric("Yearly volatility", f"{rk['ann']:.0f}%"); c2.metric("Worst-day style loss (VaR 95%, one candle)", f"{rk['var']:.2f}%")
    c1.metric("Max drawdown", f"{rk['mdd']:.1f}%"); c2.metric("Sharpe ratio", f"{rk['sharpe']:.2f}")
    st.markdown("**Position plan for a new trade** (risk 1% of wallet cash, stop at 2 ATR, target at 4 ATR)")
    val = min(S.cash, S.cash * 0.01 / (2 * atr) * price)
    c1, c2, c3 = st.columns(3)
    c1.metric("Amount to invest", f"{val:,.0f} USDT"); c2.metric("Stop-loss", f"{price-2*atr:,.2f}"); c3.metric("Take-profit", f"{price+4*atr:,.2f}")

with t3:
    cv, bh, pn = backtest(model, d, test_start, thr)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("AI strategy return", f"{(cv.iloc[-1]-1)*100:.1f}%"); c2.metric("Buy and hold return", f"{(bh.iloc[-1]-1)*100:.1f}%")
    c3.metric("Closed trades", len(pn)); c4.metric("Win rate", f"{np.mean([x > 0 for x in pn])*100:.0f}%" if pn else "No trades yet")
    f = go.Figure([go.Scatter(x=cv.index, y=cv, name="AI strategy", line=dict(color=UP)),
                   go.Scatter(x=bh.index, y=bh, name="Buy and hold", line=dict(color=HOLD))])
    st.plotly_chart(style(f, 380), use_container_width=True)
    st.caption("Tested only on candles the model never saw during training, with a 0.1% fee on each side. Past results do not predict future results.")

with t4:
    c1, c2, c3 = st.columns(3)
    c1.metric("Wallet value", f"{equity:,.2f} USDT", f"{(equity/S.start-1)*100:+.2f}%"); c2.metric("Cash", f"{S.cash:,.2f}")
    c3.metric("Open position", f"{S.qty:.6f}" if S.qty > 0 else "None")
    b1, b2 = st.columns(2)
    if b1.button("Execute AI signal", type="primary"):
        if signal == "BUY" and S.qty == 0: buy(); st.rerun()
        elif signal == "SELL" and S.qty > 0: sell("AI SELL signal"); st.rerun()
        else: st.info(f"Signal is {signal}, so nothing to do right now.")
    if b2.button("Reset wallet"): S.sym = None; st.rerun()
    if S.qty > 0: st.caption(f"Entry {S.entry:,.2f}, stop-loss {S.stop:,.2f}, take-profit {S.tp:,.2f}. Closes automatically when hit.")
    st.dataframe(pd.DataFrame(S.log[::-1]), use_container_width=True, hide_index=True) if S.log else st.caption("No trades yet. Press Execute AI signal when the signal is BUY.")

    if S.log: st.download_button("Download trade log (CSV)", pd.DataFrame(S.log).to_csv(index=False), "trades.csv")

with t5:
    st.markdown("This simulates 1,000 possible futures by reshuffling this coin's own past returns.")
    steps = st.slider("Candles to look ahead", 10, 200, 50)
    P = monte_carlo(d, price, steps=steps)
    q = np.percentile(P, [5, 50, 95], axis=0); xs = list(range(1, steps + 1))
    f = go.Figure([go.Scatter(x=xs, y=q[2], name="Best 5%", line=dict(color=UP)),
                   go.Scatter(x=xs, y=q[1], name="Median", line=dict(color=INK)),
                   go.Scatter(x=xs, y=q[0], name="Worst 5%", line=dict(color=DN))])
    f.add_hline(y=price, line_dash="dot", line_color=HOLD)
    st.plotly_chart(style(f, 380), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Chance of ending lower", f"{(P[:, -1] < price).mean()*100:.0f}%")
    c2.metric("Worst 5% outcome", f"{(q[0, -1]/price-1)*100:.1f}%")
    c3.metric("Chance of touching the stop-loss", f"{(P.min(axis=1) <= price - 2*atr).mean()*100:.0f}%")

with t6:
    st.dataframe(scan(iv, thr), use_container_width=True, hide_index=True)
    st.caption("Every coin gets its own model. This table refreshes every 5 minutes.")

with t7:
    if tick:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("24h change", f"{float(tick['priceChangePercent']):+.2f}%"); c2.metric("24h high", f"{float(tick['highPrice']):,.2f}")
        c3.metric("24h low", f"{float(tick['lowPrice']):,.2f}"); c4.metric("24h volume (USDT)", f"{float(tick['quoteVolume'])/1e6:,.0f}M")
    ob, _, _ = jget("/api/v3/depth", symbol=sym, limit=50)
    if ob:
        bids = pd.DataFrame(ob["bids"], columns=["p", "q"]).astype(float)
        asks = pd.DataFrame(ob["asks"], columns=["p", "q"]).astype(float)
        f = go.Figure([go.Scatter(x=bids.p, y=bids.q.cumsum(), name="Buyers", fill="tozeroy", line=dict(color=UP)),
                       go.Scatter(x=asks.p, y=asks.q.cumsum(), name="Sellers", fill="tozeroy", line=dict(color=DN))])
        st.plotly_chart(style(f, 320), use_container_width=True)
        st.metric("Buyers' share of the order book", f"{bids.q.sum() / (bids.q.sum() + asks.q.sum()) * 100:.0f}%")
    v, name = fear_greed()
    if v is not None: st.metric("Market mood (Fear and Greed index)", f"{v}/100", name, delta_color="off")

if auto: time.sleep(30); st.rerun()