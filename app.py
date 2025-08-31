import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ============ Page & Style (RTL + Mobile polish) ============
st.set_page_config(page_title="داشبورد جذاب محصولات + حساسیت", layout="centered", page_icon="📊")

# Minimal CSS for RTL + nicer cards
st.markdown("""
<style>
html, body, [class*="css"]  { direction: rtl; font-family: IRANSans, Vazirmatn, Segoe UI, Roboto, sans-serif; }
.big-head { font-size: 1.3rem; font-weight: 700; margin: .4rem 0 .8rem 0; }
.kpi { border-radius: 18px; padding: 14px 16px; background: linear-gradient(145deg,#0ea5e9, #6366f1); color: #fff; box-shadow: 0 6px 20px rgba(0,0,0,.15); }
.kpi .label{opacity:.9; font-size:.9rem}
.kpi .value{font-size:1.5rem; font-weight:800; line-height:1.3}
.help {opacity:.7; font-size:.85rem}
</style>
""", unsafe_allow_html=True)

st.title("✨ داشبورد جذاب محصولات + تحلیل حساسیت")
st.caption("نمایش روی موبایل بهینه شده • داده‌ها مصنوعی اما واقع‌نما")

# ============ Synthetic Data for 5 Products ============
@st.cache_data
def make_data(seed=42, days=180):
    rng = np.random.default_rng(seed)
    start = datetime.today().date() - timedelta(days=days)
    dates = pd.date_range(start, periods=days, freq="D")
    products = [
        {"product":"Apex Lite",   "base_price":39.0,  "cost":21.0, "elastic":-0.9, "beta_mkt":0.35},
        {"product":"Nova Plus",   "base_price":59.0,  "cost":33.0, "elastic":-1.1, "beta_mkt":0.40},
        {"product":"Orion Pro",   "base_price":89.0,  "cost":52.0, "elastic":-1.2, "beta_mkt":0.45},
        {"product":"Zen Mini",    "base_price":24.0,  "cost":12.0, "elastic":-0.7, "beta_mkt":0.30},
        {"product":"Pulse Ultra", "base_price":129.0, "cost":74.0, "elastic":-1.3, "beta_mkt":0.50},
    ]
    rows=[]
    for p in products:
        # marketing with weekly wave
        mkt_base = rng.normal(1500, 400, size=days)
        week_wave = 1 + 0.15*np.sin(np.linspace(0, 6*np.pi, days))
        mkt = np.clip(mkt_base * week_wave, 200, None)

        # price noise around base
        price = p["base_price"] * (1 + rng.normal(0, 0.03, size=days))

        base_demand = 2400 / np.sqrt(p["base_price"])
        noise = rng.normal(0, 6, size=days)

        units_hat = base_demand + p["elastic"]*(price - p["base_price"]) + p["beta_mkt"]*(mkt/np.mean(mkt)) + noise
        units = np.maximum(units_hat, 0).round()

        df_p = pd.DataFrame({
            "date": dates,
            "product": p["product"],
            "price": price.round(2),
            "marketing_spend": mkt.round(0),
            "units": units.astype(int),
            "unit_cost": p["cost"],
            "elastic": p["elastic"],
            "beta_mkt": p["beta_mkt"],
        })
        df_p["revenue"] = (df_p["price"] * df_p["units"]).round(2)
        df_p["profit"]  = (df_p["revenue"] - (df_p["unit_cost"]*df_p["units"] + 0.12*df_p["marketing_spend"])).round(2)
        rows.append(df_p)
    return pd.concat(rows, ignore_index=True)

df = make_data()

# ============ Controls (compact & mobile-first) ============
with st.expander("⚙️ تنظیمات و فیلترها", expanded=True):
    prods = sorted(df["product"].unique().tolist())
    selected_products = st.multiselect("محصولات", prods, default=prods)
    min_d, max_d = df["date"].min().date(), df["date"].max().date()
    date_range = st.date_input("بازه تاریخ", (min_d, max_d), min_value=min_d, max_value=max_d)
    metric = st.selectbox("سنجه اصلی", ["revenue","profit","units"], index=0)
    smooth = st.checkbox("میانگین متحرک ۷روزه", value=True)
    dark = st.toggle("تم تیره", value=True)

plotly_template = "plotly_dark" if dark else "plotly_white"

df_f = df[df["product"].isin(selected_products)].copy()
if isinstance(date_range, tuple) and len(date_range)==2:
    d1, d2 = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    df_f = df_f[(df_f["date"]>=d1) & (df_f["date"]<=d2)]

# ============ KPI Cards ============
sum_rev = float(df_f["revenue"].sum())
sum_profit = float(df_f["profit"].sum())
sum_units = int(df_f["units"].sum())

c1,c2,c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="kpi"><div class="label">درآمد کل</div><div class="value">{sum_rev:,.0f}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="kpi" style="background:linear-gradient(145deg,#22c55e,#0ea5e9)"><div class="label">سود کل</div><div class="value">{sum_profit:,.0f}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="kpi" style="background:linear-gradient(145deg,#f97316,#ef4444)"><div class="label">تعداد فروش</div><div class="value">{sum_units:,}</div></div>', unsafe_allow_html=True)

# ============ Tabs ============
tab_dash, tab_sens, tab_drill = st.tabs(["📊 داشبورد", "🧪 تحلیل حساسیت", "🔎 دریل‌داون محصول"])

# ---- Dashboard ----
with tab_dash:
    st.markdown('<div class="big-head">مقایسه محصولات</div>', unsafe_allow_html=True)
    agg = df_f.groupby("product", as_index=False).agg(
        revenue=("revenue","sum"),
        profit=("profit","sum"),
        units=("units","sum"),
        price=("price","mean")
    )
    fig_bar = px.bar(agg.sort_values(metric, ascending=False), x="product", y=metric, text=metric,
                     title=f"{metric} بر اساس محصول", template=plotly_template)
    fig_bar.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="big-head">روند روزانه</div>', unsafe_allow_html=True)
    ts = df_f.groupby(["date","product"], as_index=False).agg(value=(metric,"sum"))
    if smooth:
        ts["value"] = ts.groupby("product")["value"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    fig_ts = px.line(ts, x="date", y="value", color="product", template=plotly_template, title=f"روند {metric}")
    st.plotly_chart(fig_ts, use_container_width=True)

    st.markdown('<div class="big-head">قیمت در برابر فروش (حباب = درآمد)</div>', unsafe_allow_html=True)
    scat = df_f.copy()
    scat["bubble"] = scat["revenue"]
    fig_sc = px.scatter(scat, x="price", y="units", size="bubble", color="product", hover_data=["product","date"],
                        template=plotly_template, title="Price vs Units (Bubble=Revenue)")
    st.plotly_chart(fig_sc, use_container_width=True)

# ---- Sensitivity ----
with tab_sens:
    st.subheader("تحلیل What-if (سراسری)")
    st.caption("سناریو روی همه محصولات اعمال می‌شود و با وضعیت مبنا مقایسه می‌گردد.")

    price_mult    = st.slider("ضریب قیمت", 0.6, 1.4, 1.00, 0.01)
    mkt_mult      = st.slider("ضریب مارکتینگ", 0.5, 2.0, 1.00, 0.05)
    cost_mult     = st.slider("ضریب هزینه واحد", 0.8, 1.5, 1.00, 0.01)
    elastic_tw    = st.slider("تغییر کشش قیمت (واحدی)", -0.5, 0.5, 0.00, 0.05)
    beta_tw       = st.slider("تغییر حساسیت مارکتینگ (واحدی)", -0.2, 0.2, 0.00, 0.01)

    base_rev = float(df_f["revenue"].sum())
    base_profit = float(df_f["profit"].sum())

    scen = df_f.copy()
    scen["price_scn"] = scen["price"] * price_mult
    scen["mkt_scn"] = scen["marketing_spend"] * mkt_mult
    scen["unit_cost_scn"] = scen["unit_cost"] * cost_mult
    scen["elastic_scn"] = scen["elastic"] + elastic_tw
    scen["beta_scn"] = scen["beta_mkt"] + beta_tw

    mkt_norm = df_f["marketing_spend"].mean() if df_f["marketing_spend"].mean()!=0 else 1.0
    units_hat = ((2400/np.sqrt(scen["price"]))
                 + scen["elastic_scn"]*(scen["price_scn"] - scen["price"])
                 + scen["beta_scn"]*(scen["mkt_scn"]/mkt_norm))
    units_pred = np.maximum(units_hat, 0).round()

    scen["revenue_scn"] = scen["price_scn"] * units_pred
    scen["profit_scn"]  = scen["revenue_scn"] - (scen["unit_cost_scn"]*units_pred + 0.12*scen["mkt_scn"])

    scen_rev = float(scen["revenue_scn"].sum())
    scen_profit = float(scen["profit_scn"].sum())

    k1, k2 = st.columns(2)
    delta_rev = 0.0 if base_rev==0 else (scen_rev-base_rev)/base_rev*100
    delta_profit = 0.0 if base_profit==0 else (scen_profit-base_profit)/base_profit*100
    with k1:
        st.metric("درآمد سناریو", f"{scen_rev:,.0f}", f"{delta_rev:+.1f}% vs مبنا")
    with k2:
        st.metric("سود سناریو", f"{scen_profit:,.0f}", f"{delta_profit:+.1f}% vs مبنا")

    cmp_df = pd.DataFrame({"حالت":["Baseline","Scenario"], "Revenue":[base_rev, scen_rev], "Profit":[base_profit, scen_profit]})
    fig_r = px.bar(cmp_df, x="حالت", y="Revenue", text="Revenue", template=plotly_template, title="درآمد: مبنا vs سناریو")
    fig_p = px.bar(cmp_df, x="حالت", y="Profit", text="Profit", template=plotly_template, title="سود: مبنا vs سناریو")
    for f in (fig_r, fig_p):
        f.update_traces(texttemplate="%{text:,.0f}", textposition="outside", cliponaxis=False)
    st.plotly_chart(fig_r, use_container_width=True)
    st.plotly_chart(fig_p, use_container_width=True)

    # Tornado (one-way sensitivity around current sliders)
    def run_sum(pm=price_mult, mm=mkt_mult, cm=cost_mult, et=elastic_tw, bt=beta_tw):
        sc=df_f.copy()
        sc["price_scn"]=sc["price"]*pm
        sc["mkt_scn"]=sc["marketing_spend"]*mm
        sc["unit_cost_scn"]=sc["unit_cost"]*cm
        sc["elastic_scn"]=sc["elastic"]+et
        sc["beta_scn"]=sc["beta_mkt"]+bt
        mkt_norm = df_f["marketing_spend"].mean() if df_f["marketing_spend"].mean()!=0 else 1.0
        u = ((2400/np.sqrt(sc["price"])) + sc["elastic_scn"]*(sc["price_scn"]-sc["price"]) + sc["beta_scn"]*(sc["mkt_scn"]/mkt_norm))
        u = np.maximum(u,0).round()
        rev = float((sc["price_scn"]*u).sum())
        prf = float((sc["price_scn"]*u - (sc["unit_cost_scn"]*u + 0.12*sc["mkt_scn"])).sum())
        return rev, prf

    base_rev_s, base_profit_s = run_sum()
    tests = [
        ("قیمت +10%", dict(pm=price_mult*1.1)),
        ("قیمت -10%", dict(pm=price_mult*0.9)),
        ("مارکتینگ +20%", dict(mm=mkt_mult*1.2)),
        ("مارکتینگ -20%", dict(mm=mkt_mult*0.8)),
        ("هزینه واحد +10%", dict(cm=cost_mult*1.1)),
        ("هزینه واحد -10%", dict(cm=cost_mult*0.9)),
        ("کشش +0.1", dict(et=elastic_tw+0.1)),
        ("کشش -0.1", dict(et=elastic_tw-0.1)),
        ("حساسیت مارکتینگ +0.05", dict(bt=beta_tw+0.05)),
        ("حساسیت مارکتینگ -0.05", dict(bt=beta_tw-0.05)),
    ]
    sens=[]
    for name,kw in tests:
        _, p = run_sum(**kw)
        sens.append({"پارامتر":name, "ΔProfit": p-base_profit_s})
    tornado = pd.DataFrame(sens).sort_values("ΔProfit", key=lambda s:s.abs(), ascending=False)
    fig_tor = px.bar(tornado, y="پارامتر", x="ΔProfit", orientation="h", template=plotly_template, title="تورنادو حساسیت (اثر روی سود)")
    st.plotly_chart(fig_tor, use_container_width=True)

# ---- Product Drilldown (2D sensitivity heatmap) ----
with tab_drill:
    psel = st.selectbox("محصول برای دریل‌داون", sorted(df_f["product"].unique()))
    sub = df_f[df_f["product"]==psel].copy()
    st.markdown('<div class="help">شبکه‌ی حساسیت دو‌بعدی: تغییر هم‌زمان قیمت × مارکتینگ و اثرش بر سود</div>', unsafe_allow_html=True)

    # grid of multipliers
    p_grid = np.linspace(0.8, 1.2, 9)
    m_grid = np.linspace(0.8, 1.2, 9)
    base_profit = float(sub["profit"].sum())

    def profit_with(pm, mm):
        sc=sub.copy()
        mkt_norm = sub["marketing_spend"].mean() if sub["marketing_spend"].mean()!=0 else 1.0
        price_scn = sc["price"]*pm
        mkt_scn   = sc["marketing_spend"]*mm
        units = ( (2400/np.sqrt(sc["price"]))
                  + sc["elastic"]*(price_scn - sc["price"])
                  + sc["beta_mkt"]*(mkt_scn/mkt_norm) )
        units = np.maximum(units,0).round()
        revenue = price_scn*units
        profit  = revenue - (sc["unit_cost"]*units + 0.12*mkt_scn)
        return float(profit.sum())

    Z = []
    for pm in p_grid:
        row=[]
        for mm in m_grid:
            row.append(profit_with(pm, mm) - base_profit)  # delta vs baseline
        Z.append(row)
    Z = np.array(Z)

    fig_hm = px.imshow(Z,
        x=[f"{m:.2f}×" for m in m_grid],
        y=[f"{p:.2f}×" for p in p_grid],
        labels=dict(x="مارکتینگ", y="قیمت", color="Δسود"),
        template=plotly_template,
        title=f"Heatmap قیمت×مارکتینگ برای {psel} (Δسود نسبت به مبنا)"
    )
    st.plotly_chart(fig_hm, use_container_width=True)

st.caption("یادداشت: مدل صرفاً آموزشی است؛ می‌توانی ضرایب را با رگرسیون/مدل واقعی جایگزین کنی.")
