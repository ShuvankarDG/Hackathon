import pandas as pd
import plotly.express as px
import streamlit as st

from src.features import load_data
from src.predict import forecast, drivers

st.set_page_config(page_title="Dengue Early Warning", layout="wide")
st.title("Dengue Early-Warning System")
st.caption("Weather-lag based risk forecast for the next 2-4 weeks")


@st.cache_data
def get_data():
    return load_data()


df = get_data()

# ---- sidebar: choose city, ward, week ----
city = st.sidebar.selectbox("City", sorted(df.city.unique()))
ward = st.sidebar.selectbox("Ward", sorted(df[df.city == city].ward_id.unique()))
dates = sorted(df[df.ward_id == ward].date.dt.date.unique())[8:]    # need 8 weeks of history
date = st.sidebar.select_slider("Forecast week (today)", options=dates, value=pd.Timestamp("2025-07-07").date())

# ---- forecast ----
fc = forecast(ward, date)
colors = {"Low": "green", "Medium": "gold", "High": "orange", "Critical": "red"}
mean_cases = fc.predicted_cases.mean()

left, right = st.columns(2)
with left:
    st.subheader(f"{ward}: forecast from {date}")
    st.dataframe(fc, hide_index=True)
    worst = fc.sort_values("predicted_cases").iloc[-1]
    st.markdown(f"**Highest risk in the window:** :{colors[worst.risk]}[{worst.risk}] "
                f"in week {worst.target_week}")
    fig = px.bar(fc, x="target_week", y="predicted_cases", color="risk",
                 color_discrete_map=colors, labels={"predicted_cases": "cases per week"})
    st.plotly_chart(fig, width="stretch")

with right:
    st.subheader("Main drivers")
    lead = st.radio("Lead time (weeks)", [2, 3, 4], index=1, horizontal=True)
    d = drivers(ward, date, lead)
    fig2 = px.bar(d, x=d.values, y=d.index, orientation="h",
                  labels={"x": "push on risk (right = higher)", "y": ""})
    st.plotly_chart(fig2, width="stretch")