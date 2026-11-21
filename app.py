"""Dengue Early-Warning dashboard.  Run:  streamlit run app.py"""
from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import risk
from src.features import ROOT, STATIC
from src.predict import drivers, feature_table, forecast_frame, load_bundle, window_summary, whatif_row
from src.features import load_raw

st.set_page_config(page_title="Dengue Early Warning", page_icon="🦟", layout="wide")


# ---------- cached, computed once per session (no repeated work, no external API calls) ----------
@st.cache_resource
def _boot():
    return load_bundle()


@st.cache_data(show_spinner=False)
def _raw() -> pd.DataFrame:
    return load_raw()


@st.cache_data(show_spinner=False)
def _features(variant: str) -> pd.DataFrame:
    return feature_table(variant)


@st.cache_data(show_spinner=False)
def _city_forecast(variant: str, city: str, origin: str) -> pd.DataFrame:
    ft = _features(variant)
    rows = ft[(ft.city == city) & (ft.date == pd.Timestamp(origin))]
    return forecast_frame(rows, variant)


@st.cache_data(show_spinner=False)
def _metrics() -> dict:
    return json.loads((ROOT / "reports" / "metrics.json").read_text())


@st.cache_data(show_spinner=False)
def _test_preds() -> pd.DataFrame:
    return pd.read_parquet(ROOT / "reports" / "test_predictions.parquet")


_boot()
raw = _raw()
TIER_ORDER = risk.TIERS

st.title("🦟 Dengue Early-Warning System")
st.caption("Weather-lag driven ward-level outbreak risk, 2–4 weeks ahead · Dhaka North/South, Chattogram, Gazipur, Narayanganj")

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Query")
    variant = st.radio("Model", ["weather_only", "weather_plus_cases"], format_func=lambda v: {
        "weather_only": "Weather + ward profile only", "weather_plus_cases": "…plus recent reported cases"}[v])
    city = st.selectbox("City", sorted(raw.city.unique()), index=sorted(raw.city.unique()).index("Chattogram"))
    wards = sorted(raw[raw.city == city].ward_id.unique())
    ward = st.selectbox("Ward", wards)
    dates = sorted(raw.date.unique())
    default_i = max(i for i, d in enumerate(dates) if pd.Timestamp(d) <= pd.Timestamp("2025-07-07"))
    origin = st.select_slider("Forecast origin (week of data)", options=[pd.Timestamp(d).date() for d in dates],
                              value=pd.Timestamp(dates[default_i]).date())
    st.markdown("**What-if rainfall**")
    mult = st.slider("Scale last 4 weeks of rainfall", 0.0, 2.0, 1.0, 0.1,
                     help="Re-runs the lag pipeline with the recent rainfall scaled, to see how risk would change.")

origin_ts = pd.Timestamp(origin)
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Ward forecast", "🗺️ City ward ranking & heatmap", "📉 Model validation", "ℹ️ About"])

# ---------------- tab 1: ward forecast ----------------
with tab1:
    feats = _features(variant)
    if mult == 1.0:
        row = feats[(feats.ward_id == ward) & (feats.date == origin_ts)]
    else:
        row = whatif_row(raw, ward, origin_ts, mult, 4, variant)
    if row.empty or row[load_bundle()["variants"][variant]["features"]].isna().any(axis=None):
        st.warning("Not enough weather history before this date (need 8 weeks). Pick a later origin.")
        st.stop()
    fc = forecast_frame(row, variant)
    win = window_summary(fc, variant)
    color = risk.TIER_COLORS[win["tier"]]

    c1, c2, c3 = st.columns([1.2, 1, 1])
    c1.markdown(f"<div style='padding:18px;border-radius:12px;background:{color};color:white'>"
                f"<div style='font-size:14px'>{ward} · window {(origin_ts + pd.Timedelta(weeks=2)).date()} → {(origin_ts + pd.Timedelta(weeks=4)).date()}</div>"
                f"<div style='font-size:42px;font-weight:700'>{win['tier'].upper()}</div></div>", unsafe_allow_html=True)
    c2.metric("Risk score (0–100)", f"{win['risk_score']:.0f}", help="Percentile of predicted weekly cases vs. the 2016–2022 history.")
    c3.metric("Expected cases / week", f"{win['mean_weekly_cases']:.1f}", help=f"Tier cut-offs (cases/wk): {', '.join(f'{c:g}' for c in load_bundle()['variants'][variant]['cutoffs'])}")
    if mult != 1.0:
        st.info(f"What-if active: last 4 weeks of rainfall × {mult:.1f}")

    left, right = st.columns(2)
    with left:
        st.subheader("Forecast by week")
        fig = go.Figure()
        hist = raw[(raw.ward_id == ward) & (raw.date <= origin_ts + pd.Timedelta(weeks=4)) & (raw.date > origin_ts - pd.Timedelta(weeks=12))]
        fig.add_bar(x=hist.date, y=hist.dengue_cases, name="Reported cases", marker_color="#9aa5b1")
        fig.add_scatter(x=fc.target_date, y=fc.pred_cases, mode="lines+markers", name="Predicted", line=dict(color=color, width=3),
                        error_y=dict(type="data", symmetric=False, array=fc.hi - fc.pred_cases, arrayminus=fc.pred_cases - fc.lo))
        fig.add_vline(x=origin_ts, line_dash="dot", annotation_text="forecast origin")
        fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="cases / week", legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
        show = fc[["target_date", "horizon", "pred_cases", "lo", "hi", "tier"]].copy()
        show["target_date"] = show.target_date.dt.date
        show.columns = ["Week", "Lead (wk)", "Predicted", "Band low", "Band high", "Tier"]
        st.dataframe(show.round(1), hide_index=True, width="stretch")
        st.caption("Band = LightGBM 10th–90th percentile models. Empirical coverage on the 2024–25 test set is ≈95%, so it is conservative.")
    with right:
        st.subheader("Main weather drivers")
        h_sel = st.radio("Lead time", [2, 3, 4], index=1, horizontal=True, format_func=lambda h: f"{h} weeks")
        dv = drivers(row, h_sel, variant)
        grp = dv.groupby("driver_group").contribution.sum().sort_values()
        figd = go.Figure(go.Bar(x=grp.values, y=grp.index, orientation="h",
                                marker_color=["#c0141c" if v > 0 else "#2e8b57" for v in grp.values]))
        figd.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="push on risk (log-cases; right = higher)")
        st.plotly_chart(figd, width="stretch")
        top = dv.reindex(dv.contribution.abs().sort_values(ascending=False).index).head(6)
        top = top.assign(effect=top.contribution.map(lambda v: "↑ raises risk" if v > 0 else "↓ lowers risk"))
        st.dataframe(top[["feature", "value", "effect"]].round(2), hide_index=True, width="stretch")
        rl = [row[f"rain_lag{k}"].iloc[0] for k in range(9)]
        figr = px.bar(x=[f"t-{k}" for k in range(9)], y=rl, labels={"x": "weeks before origin", "y": "rainfall mm"})
        figr.update_layout(height=200, margin=dict(l=0, r=0, t=10, b=0))
        st.caption("Rainfall at each lag feeding the model (the breeding-clock inputs)")
        st.plotly_chart(figr, width="stretch")

# ---------------- tab 2: city ranking + heatmap ----------------
with tab2:
    cf = _city_forecast(variant, city, str(origin_ts.date()))
    if cf.empty:
        st.warning("No forecast for this origin date.")
    else:
        agg = cf.groupby("ward_id").agg(mean_cases=("pred_cases", "mean"), peak=("pred_cases", "max")).reset_index()
        vb = load_bundle()["variants"][variant]
        agg["tier"] = [risk.TIERS[i] for i in risk.tier_index(agg.mean_cases, vb["cutoffs"])]
        agg["risk_score"] = risk.score(agg.mean_cases, vb["ecdf"]).round(0)
        st.subheader(f"{city}: ward ranking for weeks +2 to +4 from {origin_ts.date()}")
        cnt = agg.tier.value_counts().reindex(TIER_ORDER, fill_value=0)
        cols = st.columns(4)
        for c, t in zip(cols, TIER_ORDER):
            c.metric(t, int(cnt[t]))
        ranked = agg.sort_values("mean_cases", ascending=False)
        st.dataframe(ranked.rename(columns={"ward_id": "Ward", "mean_cases": "Mean cases/wk", "peak": "Peak week cases",
                                            "tier": "Tier", "risk_score": "Score"}).round(1), hide_index=True, width="stretch")
        piv = cf.pivot(index="ward_id", columns="target_date", values="pred_cases").loc[ranked.ward_id]
        piv.columns = [str(c.date()) for c in piv.columns]
        figh = px.imshow(piv, aspect="auto", color_continuous_scale="YlOrRd", labels=dict(color="cases/wk", x="week", y="ward"))
        figh.update_layout(height=max(300, 22 * len(piv)), margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(figh, width="stretch")
        st.caption("Ward × week heatmap of predicted cases (ordered by risk). Coordinates are not in the dataset, so a geographic map is a next step.")

# ---------------- tab 3: validation ----------------
with tab3:
    m = _metrics()
    st.subheader("Chronological validation (train ≤ 2022 · early-stop 2023 · untouched test 2024–2025)")
    rows = []
    for v, hs in m["variants"].items():
        for h, r in hs.items():
            rows.append({"model": v, "lead": h, "MAE": r["model"]["MAE"], "RMSE": r["model"]["RMSE"],
                         "MAE seasonal baseline": r["seasonal_climatology_baseline"]["MAE"],
                         "tier accuracy": r["tiers"]["tier_accuracy"], "within 1 tier": r["tiers"]["within_one_tier"],
                         "High+ alert recall": r["tiers"]["alert_recall_high_or_above"],
                         "High+ alert precision": r["tiers"]["alert_precision_high_or_above"]})
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
    tp = _test_preds()
    tp = tp[(tp.variant == variant) & (tp.city == city)]
    tp = tp.assign(abs_err=(tp.actual - tp.pred).abs(), month=pd.to_datetime(tp.origin_date).dt.to_period("M").dt.to_timestamp())
    err = tp.groupby(["month", "horizon"]).abs_err.mean().reset_index()
    st.plotly_chart(px.line(err, x="month", y="abs_err", color="horizon", title=f"{city}: monthly MAE on test period, by lead (weeks)"),
                    width="stretch")
    w = tp[(tp.ward_id == ward) & (tp.horizon == 3)].sort_values("origin_date")
    fig = go.Figure()
    fig.add_scatter(x=pd.to_datetime(w.origin_date) + pd.Timedelta(weeks=3), y=w.actual, name="Actual")
    fig.add_scatter(x=pd.to_datetime(w.origin_date) + pd.Timedelta(weeks=3), y=w.pred, name="Predicted (3-wk lead)")
    fig.update_layout(title=f"{ward}: out-of-sample actual vs. predicted", height=320)
    st.plotly_chart(fig, width="stretch")

# ---------------- tab 4: about ----------------
with tab4:
    st.markdown("""
**How it works.** Weekly rainfall, temperature and humidity are converted into multi-week lag features
(rain at t-0…t-8 weeks, 2/4/8-week accumulations, rain × ward waterlogging), combined with each ward's density,
waterlogging and sanitation, and fed to LightGBM models trained per lead time (2, 3, 4 weeks).
Tiers are cut at the 60th/85th/95th percentiles of historical weekly cases.

**Cost.** No API or LLM calls. Models are loaded once from disk, features are computed once per session, and each query is a few milliseconds.
See `ARCHITECTURE.md`.
""")
