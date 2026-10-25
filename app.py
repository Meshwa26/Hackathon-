
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error

st.set_page_config(
    page_title="PetroOpt AI",
    page_icon="🛢️",
    layout="wide"
)

st.title("🛢️ PetroOpt AI")
st.subheader("AI-Based Production Optimization for Mature Oilfields")
st.caption(
    "Hackathon prototype | Synthetic demonstration data — "
    "not for real-world operating decisions"
)

@st.cache_data
def create_data():
    rng = np.random.default_rng(42)
    dates = pd.date_range("2026-01-01", periods=180, freq="D")
    records = []

    for n in range(1, 11):
        well = f"W-{n:02d}"
        initial_oil = rng.uniform(65, 180)
        initial_pressure = rng.uniform(85, 145)
        initial_water = rng.uniform(25, 65)

        for day, date in enumerate(dates):
            pressure = max(
                1, initial_pressure - 0.07 * day
                + rng.normal(0, 1.3)
            )
            water = np.clip(
                initial_water + 0.10 * day
                + rng.normal(0, 1.4), 0, 98
            )
            oil = initial_oil * np.exp(-0.0022 * day)
            oil *= (pressure / initial_pressure) ** 0.15
            oil *= (1 - 0.002 * max(water - initial_water, 0))
            oil = max(0, oil + rng.normal(0, 2.5))

            vibration = max(0.1, 2 + rng.normal(0, 0.22))
            power = max(5, 34 + rng.normal(0, 2.8))

            if day in (72, 131) and n in (3, 7):
                vibration += rng.uniform(1.5, 3)
                power += rng.uniform(8, 15)

            records.append({
                "date": date,
                "well_id": well,
                "oil_rate_bpd": round(oil, 2),
                "water_cut_pct": round(water, 2),
                "pressure_bar": round(pressure, 2),
                "pump_vibration": round(vibration, 2),
                "power_kw": round(power, 2)
            })

    return pd.DataFrame(records)

df = create_data()

wells = sorted(df["well_id"].unique())
selected_well = st.sidebar.selectbox("Select a well", wells)
days_to_show = st.sidebar.slider("History to display (days)", 30, 180, 90)

well_df = (
    df[df["well_id"] == selected_well]
    .sort_values("date")
    .tail(days_to_show)
)

latest = well_df.iloc[-1]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Latest oil rate", f"{latest['oil_rate_bpd']:.1f} bbl/day")
c2.metric("Water cut", f"{latest['water_cut_pct']:.1f}%")
c3.metric("Pressure", f"{latest['pressure_bar']:.1f} bar")
c4.metric("Pump power", f"{latest['power_kw']:.1f} kW")

st.markdown("### Production history")
st.line_chart(
    well_df.set_index("date")[["oil_rate_bpd"]]
)

st.markdown("### Operating measurements")
st.line_chart(
    well_df.set_index("date")[
        ["water_cut_pct", "pressure_bar"]
    ]
)

st.markdown("### Next-day oil production forecast")

features = [
    "oil_rate_bpd", "water_cut_pct", "pressure_bar",
    "pump_vibration", "power_kw"
]
model_data = df.sort_values(["well_id", "date"]).copy()

for col in features:
    model_data[f"{col}_lag1"] = (
        model_data.groupby("well_id")[col].shift(1)
    )

model_data["target"] = (
    model_data.groupby("well_id")["oil_rate_bpd"].shift(-1)
)
model_data = model_data.dropna()

feature_cols = [f"{c}_lag1" for c in features]
cutoff = model_data["date"].quantile(0.8)
train = model_data[model_data["date"] <= cutoff]
test = model_data[model_data["date"] > cutoff]

model = RandomForestRegressor(
    n_estimators=150,
    min_samples_leaf=2,
    random_state=42
)
model.fit(train[feature_cols], train["target"])

if not test.empty:
    predictions = model.predict(test[feature_cols])
    mae = mean_absolute_error(test["target"], predictions)
    st.metric("Holdout MAE", f"{mae:.2f} bbl/day")
    st.caption(
        "MAE is evaluated on a later time period. "
        "This score applies only to the synthetic dataset."
    )

latest_row = (
    model_data[model_data["well_id"] == selected_well]
    .sort_values("date")
    .iloc[-1]
)
forecast = model.predict(latest_row[feature_cols].to_frame().T)[0]
st.metric("Illustrative next-day forecast", f"{max(0, forecast):.1f} bbl/day")

st.markdown("### Unusual operating patterns")

anomaly_model = IsolationForest(
    contamination=0.025,
    random_state=42
)
df["anomaly"] = anomaly_model.fit_predict(df[features])
well_anomalies = df[
    (df["well_id"] == selected_well)
    & (df["anomaly"] == -1)
].sort_values("date", ascending=False)

st.write(f"Flagged observations for {selected_well}: {len(well_anomalies)}")
st.dataframe(
    well_anomalies[
        ["date", "oil_rate_bpd", "water_cut_pct",
         "pressure_bar", "pump_vibration", "power_kw"]
    ].head(10),
    use_container_width=True
)

st.info(
    "Anomaly flags are prompts for investigation, not confirmed equipment "
    "failures. Scenario optimization and real operating setpoints are not "
    "implemented in this demo. All generated values are synthetic."
)

st.download_button(
    "Download selected well data (CSV)",
    data=well_df.to_csv(index=False).encode("utf-8"),
    file_name=f"{selected_well}_demo_data.csv",
    mime="text/csv"
)
