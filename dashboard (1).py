"""
Flipkart iPhone 16 Supply Chain — Live Digital Twin Dashboard
Week 3 Deliverable: Streaming data + Forecasting + Live KPIs
Run with:  streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flipkart iPhone 16 — Supply Chain Digital Twin",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f, #0d2137);
        border-radius: 12px; padding: 18px 22px;
        border-left: 4px solid #2196F3;
        margin-bottom: 10px;
    }
    .metric-label { color: #90CAF9; font-size: 13px; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; }
    .metric-value { color: #FFFFFF; font-size: 32px; font-weight: 700; margin: 4px 0; }
    .metric-sub   { color: #64B5F6; font-size: 12px; }
    .alert-red  { background:#7f1d1d; border-left:4px solid #ef4444; border-radius:8px; padding:12px 16px; color:#fca5a5; font-weight:600; }
    .alert-green{ background:#14532d; border-left:4px solid #22c55e; border-radius:8px; padding:12px 16px; color:#86efac; font-weight:600; }
    .alert-yel  { background:#713f12; border-left:4px solid #f59e0b; border-radius:8px; padding:12px 16px; color:#fcd34d; font-weight:600; }
    .stMetric label { color:#90CAF9 !important; }
    .header-title { font-size:28px; font-weight:800; color:#FFFFFF; letter-spacing:1px; }
    .header-sub   { font-size:14px; color:#90CAF9; margin-top:-8px; }
    div[data-testid="stSidebar"] { background:#0d2137; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("flipkart_iphone16_multicity.csv")
    except FileNotFoundError:
        st.error("⚠️ Place flipkart_iphone16_multicity.csv in the same folder as dashboard.py")
        st.stop()
    # FIX: don't rely on parse_dates alone — on some pandas/pyarrow versions
    # (e.g. pandas on Python 3.14 used by Streamlit Cloud) it silently keeps
    # the column as a PyArrow string dtype instead of datetime64, which then
    # breaks every "<=" / "==" comparison against a Timestamp later on.
    # Explicitly converting here guarantees a real datetime64[ns] column.
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values(["Date","City"]).reset_index(drop=True)
    return df

df_all = load_data()
cities    = sorted(df_all["City"].unique())
# FIX: force these into a plain Python list of pandas Timestamps.
# df_all["Date"].unique() on a datetime column returns numpy.datetime64
# objects, which do NOT have .strftime(). Wrapping with pd.to_datetime(...)
# and converting to a list gives proper pandas Timestamps everywhere below.
all_dates = sorted(pd.to_datetime(df_all["Date"].unique()).to_pydatetime().tolist())

# ─────────────────────────────────────────────────────────────────────────────
# FORECASTING HELPER
# ─────────────────────────────────────────────────────────────────────────────
def run_forecast(series, city_name, n_forecast=15):
    """Train ARIMA + Moving Average on first 75 days, test on last 15."""
    if len(series) < 20:
        return None
    train = series[:-n_forecast]
    test  = series[-n_forecast:]

    results = {}

    # — Moving Average (7-day) —
    ma_pred = [train[-7:].mean()] * n_forecast
    results["Moving Average"] = {
        "pred": ma_pred,
        "MAE":  round(mean_absolute_error(test, ma_pred), 3),
        "RMSE": round(np.sqrt(mean_squared_error(test, ma_pred)), 3),
    }

    # — ARIMA(2,1,2) —
    try:
        model = ARIMA(train, order=(2, 1, 2))
        fit   = model.fit()
        arima_pred = fit.forecast(steps=n_forecast).tolist()
        results["ARIMA(2,1,2)"] = {
            "pred": arima_pred,
            "MAE":  round(mean_absolute_error(test, arima_pred), 3),
            "RMSE": round(np.sqrt(mean_squared_error(test, arima_pred)), 3),
        }
    except Exception:
        arima_pred = ma_pred
        results["ARIMA(2,1,2)"] = results["Moving Average"].copy()

    return {"train": train, "test": test, "results": results}

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Control Panel")
    st.markdown("---")

    selected_city = st.selectbox("🏙️ Select City", ["All Cities"] + cities)
    stream_speed  = st.slider("⚡ Stream Speed (days/sec)", 1, 10, 3)
    n_days_show   = st.slider("📅 Days to Stream", 10, 90, 90)
    forecast_city = st.selectbox("🔮 Forecast City", cities)

    st.markdown("---")
    run_live   = st.button("▶️ START LIVE STREAM", type="primary", use_container_width=True)
    show_forecast = st.button("📈 RUN FORECAST", use_container_width=True)
    st.markdown("---")

    # Supply chain map config
    st.markdown("### 🗺️ Supply Chain Map")
    st.caption("Apple Foxconn → Flipkart DC → 4 Cities")
    st.markdown("---")
    st.markdown("**Week 3 Deliverable**")
    st.caption("✅ Live streaming dashboard\n✅ Demand forecasting (ARIMA)\n✅ MAE / RMSE metrics\n✅ Stockout & KPI alerts\n✅ Supply chain map")

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(90deg,#0d2137,#1e3a5f);
            padding:20px 28px; border-radius:12px; margin-bottom:20px;
            border-bottom:3px solid #2196F3;'>
  <div class='header-title'>📦 Flipkart iPhone 16 — Supply Chain Digital Twin</div>
  <div class='header-sub'>Live Inventory · Demand Forecasting · Stockout Alerts · 4-City Network</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SUPPLY CHAIN MAP (always visible)
# ─────────────────────────────────────────────────────────────────────────────
with st.expander("🗺️ Supply Chain Network Map — Click to expand", expanded=False):
    nodes = pd.DataFrame([
        dict(name="Apple Foxconn\n(Supplier)", lat=12.9716, lon=80.2707, type="Factory",  size=20),
        dict(name="Flipkart DC\nBangalore",    lat=12.9716, lon=77.5946, type="DC",       size=16),
        dict(name="Flipkart DC\nDelhi",        lat=28.6139, lon=77.2090, type="DC",       size=16),
        dict(name="Delhi\n(Market)",           lat=28.6139, lon=77.2090, type="Customer", size=12),
        dict(name="Mumbai\n(Market)",          lat=19.0760, lon=72.8777, type="Customer", size=12),
        dict(name="Bangalore\n(Market)",       lat=12.9716, lon=77.5946, type="Customer", size=12),
        dict(name="Chennai\n(Market)",         lat=13.0827, lon=80.2707, type="Customer", size=12),
    ])
    color_map = {"Factory":"#FF6B35","DC":"#2196F3","Customer":"#4CAF50"}
    fig_map = px.scatter_mapbox(
        nodes, lat="lat", lon="lon", hover_name="name", color="type",
        color_discrete_map=color_map, size="size",
        mapbox_style="carto-darkmatter",
        zoom=4, center={"lat":18,"lon":78},
        height=420,
    )
    # Add route lines: Factory → DCs
    routes = [
        (12.9716, 80.2707, 12.9716, 77.5946, "Factory→Bangalore DC"),
        (12.9716, 80.2707, 28.6139, 77.2090, "Factory→Delhi DC"),
        (12.9716, 77.5946, 19.0760, 72.8777, "Bangalore DC→Mumbai"),
        (12.9716, 77.5946, 13.0827, 80.2707, "Bangalore DC→Chennai"),
        (28.6139, 77.2090, 28.6139, 77.2090, "Delhi DC→Delhi"),
    ]
    for lat1,lon1,lat2,lon2,name in routes:
        fig_map.add_trace(go.Scattermapbox(
            lat=[lat1,lat2], lon=[lon1,lon2], mode="lines",
            line=dict(width=2, color="#FF6B35"),
            name=name, showlegend=False,
        ))
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0},
                          paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                          legend=dict(bgcolor="#0d2137", font=dict(color="white")))
    st.plotly_chart(fig_map, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# STATIC SUMMARY (shown before stream)
# ─────────────────────────────────────────────────────────────────────────────
df_latest = df_all[df_all["Date"] == df_all["Date"].max()]

col1, col2, col3, col4 = st.columns(4)
with col1:
    total_demand = int(df_all["Daily_Demand"].sum())
    st.metric("📊 Total Demand (90d)", f"{total_demand:,} units", "All 4 cities")
with col2:
    total_stock = int(df_latest["Closing_Inventory"].sum())
    st.metric("📦 Current Inventory", f"{total_stock:,} units", "Day 90 closing")
with col3:
    stockout_days = int(df_all["Stockout"].sum())
    st.metric("🚨 Total Stockout Events", f"{stockout_days}", "Across all cities & days")
with col4:
    total_holding = int(df_all["Holding_Cost"].sum())
    st.metric("💰 Total Holding Cost", f"₹{total_holding:,}", "90 days")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# LIVE STREAM SECTION
# ─────────────────────────────────────────────────────────────────────────────
if run_live:
    st.markdown("### ⚡ LIVE DATA STREAM — Day by Day")

    dates_to_stream = all_dates[:n_days_show]

    # Placeholders
    ph_alert   = st.empty()
    ph_day     = st.empty()
    ph_kpi     = st.empty()
    ph_inv     = st.empty()
    ph_demand  = st.empty()
    ph_city    = st.empty()

    inv_history    = {c: [] for c in cities}
    demand_history = {c: [] for c in cities}
    date_history   = []
    stockout_events = []

    for day_idx, current_date in enumerate(dates_to_stream):
        # FIX: ensure current_date is a pandas Timestamp (it should already
        # be one since all_dates was built with pd.to_datetime(...).to_pydatetime(),
        # but this extra guard makes the loop robust either way).
        current_date = pd.Timestamp(current_date)

        day_data = df_all[df_all["Date"] == current_date]

        date_history.append(current_date)
        for city in cities:
            row = day_data[day_data["City"] == city]
            if not row.empty:
                inv_history[city].append(int(row["Closing_Inventory"].values[0]))
                demand_history[city].append(int(row["Daily_Demand"].values[0]))
                if row["Stockout"].values[0] > 0:
                    stockout_events.append({"date": current_date, "city": city})

        # — Alert banner —
        today_stockouts = [e["city"] for e in stockout_events if e["date"] == current_date]
        if today_stockouts:
            ph_alert.markdown(f"""
            <div class='alert-red'>🚨 STOCKOUT ALERT — {current_date.strftime('%b %d')}:
            {', '.join(today_stockouts)} — Inventory ZERO! Reorder triggered immediately.</div>
            """, unsafe_allow_html=True)
        else:
            low_inv = [c for c in cities if inv_history[c] and inv_history[c][-1] < 20]
            if low_inv:
                ph_alert.markdown(f"""
                <div class='alert-yel'>⚠️ LOW STOCK WARNING — {', '.join(low_inv)} below 20 units</div>
                """, unsafe_allow_html=True)
            else:
                ph_alert.markdown(f"""
                <div class='alert-green'>✅ All cities healthy — Day {day_idx+1}/{len(dates_to_stream)}</div>
                """, unsafe_allow_html=True)

        # — Day counter —
        progress = (day_idx + 1) / len(dates_to_stream)
        ph_day.progress(progress, text=f"Streaming Day {day_idx+1} — {current_date.strftime('%B %d, %Y')}")

        # — KPIs —
        k1, k2, k3, k4 = ph_kpi.columns(4)
        total_inv_now = sum(h[-1] for h in inv_history.values() if h)
        total_dem_now = sum(h[-1] for h in demand_history.values() if h)
        stk_count = len(stockout_events)
        reorder_count = int(df_all[df_all["Date"] <= current_date]["Reorder_Flag"].sum())

        k1.metric("📦 Total Inventory", f"{total_inv_now:,}", f"Day {day_idx+1}")
        k2.metric("📈 Today's Demand", f"{total_dem_now} units")
        k3.metric("🚨 Stockout Events", str(stk_count))
        k4.metric("🔄 Reorders Triggered", str(reorder_count))

        # — Inventory chart —
        if len(date_history) > 1:
            df_inv = pd.DataFrame(inv_history, index=date_history)
            fig_inv = go.Figure()
            colors = {"Delhi":"#2196F3","Mumbai":"#FF6B35","Bangalore":"#4CAF50","Chennai":"#9C27B0"}
            for city in cities:
                fig_inv.add_trace(go.Scatter(
                    x=date_history, y=inv_history[city], name=city,
                    line=dict(color=colors.get(city,"#fff"), width=2),
                    fill="tozeroy", fillcolor=colors.get(city,"#fff").replace("#","rgba(").replace(")",",0.08)") if False else None,
                ))
            fig_inv.add_hline(y=20, line_dash="dash", line_color="#ef4444",
                              annotation_text="🚨 Reorder Point", annotation_position="bottom right")
            fig_inv.update_layout(
                title="📦 Live Inventory Levels — All Cities",
                paper_bgcolor="#0d2137", plot_bgcolor="#0d2137",
                font=dict(color="white"), height=280,
                legend=dict(bgcolor="#0d2137"),
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"),
            )
            ph_inv.plotly_chart(fig_inv, use_container_width=True)

        # — Demand chart —
        if len(date_history) > 1:
            fig_dem = go.Figure()
            for city in cities:
                fig_dem.add_trace(go.Bar(
                    x=date_history, y=demand_history[city], name=city,
                    marker_color=colors.get(city,"#fff"),
                ))
            fig_dem.update_layout(
                title="📈 Daily Demand — All Cities",
                barmode="stack",
                paper_bgcolor="#0d2137", plot_bgcolor="#0d2137",
                font=dict(color="white"), height=260,
                legend=dict(bgcolor="#0d2137"),
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"),
            )
            ph_demand.plotly_chart(fig_dem, use_container_width=True)

        # — Per-city breakdown —
        if len(date_history) > 1:
            cols = ph_city.columns(4)
            for i, city in enumerate(cities):
                inv_val = inv_history[city][-1] if inv_history[city] else 0
                dem_val = demand_history[city][-1] if demand_history[city] else 0
                stk_c   = len([e for e in stockout_events if e["city"]==city])
                delta_c = "color:red" if inv_val < 20 else "color:green"
                cols[i].markdown(f"""
                <div style='background:#0d2137;border-radius:10px;padding:14px;
                            border:1px solid #1e3a5f;text-align:center;'>
                  <div style='color:#90CAF9;font-size:12px;font-weight:700;'>{city.upper()}</div>
                  <div style='color:white;font-size:24px;font-weight:800;'>{inv_val}</div>
                  <div style='color:#64B5F6;font-size:11px;'>Inventory</div>
                  <div style='color:#FFB74D;font-size:14px;font-weight:600;margin-top:6px;'>Demand: {dem_val}</div>
                  <div style='font-size:11px;{delta_c};'>Stockouts: {stk_c}</div>
                </div>
                """, unsafe_allow_html=True)

        time.sleep(1.0 / stream_speed)

    st.success(f"✅ Stream complete! Streamed {len(dates_to_stream)} days across 4 cities.")

# ─────────────────────────────────────────────────────────────────────────────
# FORECASTING SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
if show_forecast:
    st.markdown(f"### 🔮 Demand Forecasting — {forecast_city}")

    city_df   = df_all[df_all["City"] == forecast_city].sort_values("Date")
    demand_series = city_df["Daily_Demand"].values.astype(float)
    dates_series  = city_df["Date"].values

    fcast = run_forecast(demand_series, forecast_city)

    if fcast:
        n_train = len(fcast["train"])
        n_test  = 15
        train_dates = dates_series[:n_train]
        test_dates  = dates_series[n_train:]

        # — Accuracy metrics table —
        st.markdown("#### 📊 Model Accuracy Metrics")
        metric_rows = []
        for model_name, res in fcast["results"].items():
            metric_rows.append({"Model": model_name, "MAE": res["MAE"], "RMSE": res["RMSE"]})
        df_metrics = pd.DataFrame(metric_rows)
        st.dataframe(df_metrics.style.highlight_min(subset=["MAE","RMSE"], color="#14532d"), use_container_width=True)

        # — Best model —
        best_model = min(fcast["results"], key=lambda m: fcast["results"][m]["RMSE"])
        best_pred  = fcast["results"][best_model]["pred"]
        best_mae   = fcast["results"][best_model]["MAE"]
        best_rmse  = fcast["results"][best_model]["RMSE"]

        m1, m2, m3 = st.columns(3)
        m1.metric("🏆 Best Model", best_model)
        m2.metric("📉 MAE", best_mae, help="Mean Absolute Error — lower is better")
        m3.metric("📉 RMSE", best_rmse, help="Root Mean Square Error — lower is better")

        # — Forecast chart —
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(x=train_dates, y=fcast["train"], name="Training Data",
                                    line=dict(color="#2196F3", width=2)))
        fig_fc.add_trace(go.Scatter(x=test_dates, y=fcast["test"], name="Actual (Test)",
                                    line=dict(color="#4CAF50", width=2)))
        for model_name, res in fcast["results"].items():
            fc_color = "#FF6B35" if "ARIMA" in model_name else "#9C27B0"
            fig_fc.add_trace(go.Scatter(
                x=test_dates, y=res["pred"], name=f"{model_name} Forecast",
                line=dict(color=fc_color, width=2, dash="dash"),
            ))

        # — Reorder logic: flag if forecast > current inventory —
        current_inv = city_df["Closing_Inventory"].values[-1]
        forecast_total = sum(best_pred)
        reorder_needed = forecast_total > current_inv
        fig_fc.add_annotation(
            x=test_dates[-1], y=max(fcast["test"]),
            text="🚨 REORDER NOW!" if reorder_needed else "✅ Stock OK",
            showarrow=True, font=dict(color="red" if reorder_needed else "green", size=14),
        )

        fig_fc.update_layout(
            title=f"Demand Forecast vs Actual — {forecast_city} | Best: {best_model} (RMSE={best_rmse})",
            paper_bgcolor="#0d2137", plot_bgcolor="#0d2137",
            font=dict(color="white"), height=380,
            legend=dict(bgcolor="#0d2137"),
            xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"),
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # — Smart reorder alert —
        if reorder_needed:
            st.markdown(f"""
            <div class='alert-red'>
            🚨 SMART REORDER TRIGGERED for {forecast_city}<br>
            Forecasted demand next 15 days: <b>{forecast_total:.0f} units</b><br>
            Current closing inventory: <b>{current_inv} units</b><br>
            <b>ACTION: Place reorder of {int(forecast_total - current_inv)} units immediately!</b>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='alert-green'>
            ✅ {forecast_city} — Stock level sufficient for next 15 days.
            Forecasted demand: {forecast_total:.0f} units | Available: {current_inv} units
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# STATIC CHARTS (always visible)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📊 Full 90-Day Analysis")

tab1, tab2, tab3 = st.tabs(["📦 Inventory Trend", "🚨 Stockout Analysis", "💰 Cost Analysis"])

with tab1:
    fig1 = px.line(
        df_all, x="Date", y="Closing_Inventory", color="City",
        title="Inventory Levels — All 4 Cities (90 Days)",
        color_discrete_sequence=["#2196F3","#FF6B35","#4CAF50","#9C27B0"],
    )
    fig1.add_hline(y=20, line_dash="dash", line_color="red", annotation_text="Reorder Point")
    fig1.update_layout(paper_bgcolor="#0d2137", plot_bgcolor="#0d2137",
                       font=dict(color="white"), height=380,
                       xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"))
    st.plotly_chart(fig1, use_container_width=True)

with tab2:
    stockout_by_city = df_all.groupby("City")["Stockout"].sum().reset_index()
    stockout_by_city.columns = ["City", "Stockout Days"]
    fig2 = px.bar(stockout_by_city, x="City", y="Stockout Days",
                  color="City", title="Total Stockout Days per City",
                  color_discrete_sequence=["#2196F3","#FF6B35","#4CAF50","#9C27B0"])
    fig2.update_layout(paper_bgcolor="#0d2137", plot_bgcolor="#0d2137",
                       font=dict(color="white"), height=320,
                       xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"))
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    cost_df = df_all.groupby("City")[["Holding_Cost","Stockout_Cost"]].sum().reset_index()
    fig3 = px.bar(cost_df.melt(id_vars="City"), x="City", y="value",
                  color="variable", barmode="group",
                  title="Total Holding vs Stockout Cost per City",
                  color_discrete_sequence=["#2196F3","#ef4444"])
    fig3.update_layout(paper_bgcolor="#0d2137", plot_bgcolor="#0d2137",
                       font=dict(color="white"), height=320,
                       xaxis=dict(gridcolor="#1e3a5f"), yaxis=dict(gridcolor="#1e3a5f"))
    st.plotly_chart(fig3, use_container_width=True)

st.markdown("---")
st.markdown("<center style='color:#64B5F6;font-size:12px;'>Flipkart iPhone 16 Digital Twin · Week 3 Deliverable · Powered by Streamlit + ARIMA</center>", unsafe_allow_html=True)
