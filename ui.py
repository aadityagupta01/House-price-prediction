"""Streamlit layout and charts. All prediction work is passed in from app.py."""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

BLUE = "#2563eb"
GREY = "#cbd5e1"
ORANGE = "#f97316"
STATE_MEDIAN = 179_700  # median house value of the full dataset


def _style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(t=30, b=10, l=10, r=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.25)")
    return fig


def render_header(metadata: dict):
    st.title("🏠 California House Price Estimator")
    st.write(
        "Change the sliders on the left. The estimate and charts update instantly. "
        f"Model: **{metadata['best_model']}**."
    )


def render_sidebar(metadata: dict) -> dict:
    """Draw the input controls and return the raw feature values."""
    rng = metadata["numeric_ranges"]

    def clamp(col, value):
        lo, hi = rng[col]
        return min(max(value, lo), hi)

    with st.sidebar:
        st.header("Describe the district")

        st.subheader("Location")
        ocean = st.selectbox(
            "Ocean proximity",
            metadata["ocean_categories"],
            index=0,
            help="How close the district is to the ocean or bay.",
        )
        lat_lo, lat_hi = rng["latitude"]
        lon_lo, lon_hi = rng["longitude"]
        latitude = st.slider("Latitude", float(lat_lo), float(lat_hi), float(clamp("latitude", 34.2)), 0.01)
        longitude = st.slider("Longitude", float(lon_lo), float(lon_hi), float(clamp("longitude", -118.5)), 0.01)

        st.subheader("People & income")
        income = st.slider(
            "Median income (× $10,000)", 0.5, 15.0, 3.5, 0.1,
            help="3.5 means about $35,000 per year.",
        )
        population = st.slider("Population", 3, 10000, 1200, 50)
        households = st.slider("Households", 1, 3000, 410, 10)

        st.subheader("Housing")
        age = st.slider("Median house age (years)", 1, 52, 29)
        rooms = st.slider("Total rooms", 2, 12000, 2100, 50)
        bedrooms = st.slider("Total bedrooms", 1, 3000, 430, 10)

    return {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": float(age),
        "total_rooms": float(rooms),
        "total_bedrooms": float(bedrooms),
        "population": float(population),
        "households": float(households),
        "median_income": float(income),
        "ocean_proximity": ocean,
    }


def render_result(price: float, values: dict, metadata: dict):
    """Three headline numbers plus any input sanity warnings."""
    cap = metadata["price_cap"]
    mae = metadata["test_mae"]

    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated median house value", f"${price:,.0f}")
    c2.metric(
        "Compared with California median",
        f"${price - STATE_MEDIAN:+,.0f}",
        help=f"California median in the data: ${STATE_MEDIAN:,}",
    )
    c3.metric(
        "Typical error of the model",
        f"± ${mae:,.0f}",
        help="Mean absolute error on the held-out test set.",
    )

    if price >= cap * 0.95:
        st.warning(
            f"The data is capped at ${cap:,}, so values near this level are likely "
            "underestimated."
        )
    if values["total_bedrooms"] > values["total_rooms"]:
        st.warning("Bedrooms are higher than total rooms. Check these two inputs.")
    if values["households"] > values["population"]:
        st.warning("Households are higher than population. Check these two inputs.")
    if values["ocean_proximity"] == "ISLAND":
        st.info("ISLAND has only 5 rows in the training data, so this estimate is unreliable.")


def render_charts(price: float, values: dict, metadata: dict, predict_many):
    """Four simple charts. `predict_many(rows)` returns one price per row."""
    st.subheader("Explore")
    tab_income, tab_ocean, tab_age, tab_map = st.tabs(
        ["Income effect", "Ocean proximity", "House age", "Map"]
    )

    with tab_income:
        grid = [x / 10 for x in range(5, 151, 5)]
        preds = predict_many([{**values, "median_income": g} for g in grid])
        fig = go.Figure()
        fig.add_scatter(x=[g * 10_000 for g in grid], y=preds, mode="lines", line=dict(color=BLUE, width=3))
        fig.add_scatter(
            x=[values["median_income"] * 10_000], y=[price], mode="markers",
            marker=dict(color=ORANGE, size=13, line=dict(color="white", width=2)),
        )
        fig.update_xaxes(title="Median income per year", tickprefix="$", tickformat=",")
        fig.update_yaxes(title="Estimated house value", tickprefix="$", tickformat=",")
        st.plotly_chart(_style(fig), use_container_width=True)
        st.caption("The orange dot is your district. The line shows what happens if only income changes.")

    with tab_ocean:
        cats = metadata["ocean_categories"]
        preds = predict_many([{**values, "ocean_proximity": c} for c in cats])
        colors = [ORANGE if c == values["ocean_proximity"] else GREY for c in cats]
        fig = go.Figure(go.Bar(
            x=cats, y=preds, marker_color=colors,
            text=[f"${p:,.0f}" for p in preds], textposition="outside", cliponaxis=False,
        ))
        fig.update_yaxes(title="Estimated house value", tickprefix="$", tickformat=",",
                         range=[0, max(preds) * 1.18])
        st.plotly_chart(_style(fig), use_container_width=True)
        st.caption("Same district, different ocean proximity. Your choice is highlighted.")

    with tab_age:
        grid = list(range(1, 53))
        preds = predict_many([{**values, "housing_median_age": float(g)} for g in grid])
        fig = go.Figure()
        fig.add_scatter(x=grid, y=preds, mode="lines", line=dict(color=BLUE, width=3))
        fig.add_scatter(
            x=[values["housing_median_age"]], y=[price], mode="markers",
            marker=dict(color=ORANGE, size=13, line=dict(color="white", width=2)),
        )
        fig.update_xaxes(title="Median house age (years)")
        fig.update_yaxes(title="Estimated house value", tickprefix="$", tickformat=",")
        st.plotly_chart(_style(fig), use_container_width=True)
        st.caption("The orange dot is your district. The line shows what happens if only house age changes.")

    with tab_map:
        st.map(pd.DataFrame({"lat": [values["latitude"]], "lon": [values["longitude"]]}), zoom=5)
        st.caption("Location of the district you described.")


def render_model_info(metadata: dict, comparison_csv: Path):
    with st.expander("How accurate is this model?"):
        c1, c2 = st.columns(2)
        c1.metric("Test R²", f"{metadata['test_r2']:.2f}", help="1.0 is perfect. Higher is better.")
        c2.metric("Test MAE", f"${metadata['test_mae']:,.0f}", help="Average size of the error.")

        if comparison_csv.exists():
            df = pd.read_csv(comparison_csv)
            fig = go.Figure(go.Bar(
                x=df["Model"], y=df["Test R²"],
                marker_color=[BLUE if m == metadata["best_model"] else GREY for m in df["Model"]],
                text=[f"{v:.2f}" for v in df["Test R²"]], textposition="outside", cliponaxis=False,
            ))
            fig.update_yaxes(title="Test R² (higher is better)", range=[0, 1])
            st.plotly_chart(_style(fig, 300), use_container_width=True)
            st.dataframe(df, use_container_width=True, hide_index=True)


def render_disclaimer():
    st.divider()
    st.caption(
        "Educational project. Based on the 1990 California census; prices are capped at $500,001 "
        "and predictions describe a whole district, not a single house. Not a property valuation."
    )


# ---------------------------------------------------------------------------
# Dataset Explorer page
# ---------------------------------------------------------------------------
def render_explorer(validation, source_label: str):
    """Show the validation outcome, data preview, cleaning summary and EDA charts."""
    st.title("📊 Dataset Explorer")
    st.caption(
        "Upload a CSV to check and explore it. This page does not retrain the model: "
        "price estimates always use the saved model."
    )

    for msg in validation.errors:
        st.error(msg)
    if not validation.ok:
        st.info(
            "Required columns: longitude, latitude, housing_median_age, total_rooms, "
            "total_bedrooms, population, households, median_income, ocean_proximity "
            "and the target median_house_value."
        )
        return

    for msg in validation.warnings:
        st.warning(msg)
    df = validation.data
    st.success(f"Dataset is valid: **{source_label}**")

    # Cleaning summary
    s = validation.summary
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows uploaded", f"{s['rows_uploaded']:,}")
    c2.metric("Rows after cleaning", f"{s['rows_after_cleaning']:,}")
    c3.metric("Duplicates removed", f"{s['duplicate_rows']:,}")
    c4.metric("Invalid target rows removed", f"{s['invalid_target_rows']:,}")

    tab_data, tab_charts = st.tabs(["Data", "Charts"])

    with tab_data:
        st.subheader("Preview")
        st.dataframe(df.head(10))
        st.caption(f"{df.shape[0]:,} rows × {df.shape[1]} columns")

        left, right = st.columns(2)
        with left:
            st.subheader("Column types")
            st.dataframe(pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str).values}), hide_index=True)
        with right:
            st.subheader("Missing values")
            missing = df.isna().sum()
            missing = missing[missing > 0]
            if missing.empty:
                st.write("No missing values.")
            else:
                st.dataframe(missing.rename("Missing").reset_index().rename(columns={"index": "Column"}), hide_index=True)
                st.caption("Missing values are kept. The model pipeline fills them (median for numbers).")

        st.subheader("Summary statistics")
        st.dataframe(df.describe().T.round(2))

    with tab_charts:
        _render_eda_charts(df)


def _render_eda_charts(df: pd.DataFrame):
    target = "median_house_value"

    st.subheader("Price distribution")
    fig = go.Figure(go.Histogram(x=df[target], nbinsx=50, marker_color=BLUE))
    fig.update_xaxes(title="Median house value", tickprefix="$", tickformat=",")
    fig.update_yaxes(title="Number of districts")
    st.plotly_chart(_style(fig, 320), use_container_width=True)
    st.caption("The tall bar at the right edge is the $500,001 price cap in the original data.")

    left, right = st.columns(2)
    with left:
        st.subheader("Income vs price")
        sample = df.sample(min(3000, len(df)), random_state=42)
        fig = go.Figure(go.Scatter(
            x=sample["median_income"], y=sample[target], mode="markers",
            marker=dict(color=BLUE, size=4, opacity=0.35),
        ))
        fig.update_xaxes(title="Median income (× $10,000)")
        fig.update_yaxes(title="Median house value", tickprefix="$", tickformat=",")
        st.plotly_chart(_style(fig, 340), use_container_width=True)
    with right:
        st.subheader("Bedrooms per district")
        cap = df["total_bedrooms"].quantile(0.99)
        fig = go.Figure(go.Histogram(x=df["total_bedrooms"].clip(upper=cap), nbinsx=40, marker_color=BLUE))
        fig.update_xaxes(title="Total bedrooms (top 1% grouped at the right edge)")
        fig.update_yaxes(title="Number of districts")
        st.plotly_chart(_style(fig, 340), use_container_width=True)

    st.subheader("Location comparison")
    by_loc = df.groupby("ocean_proximity")[target].median().sort_values()
    fig = go.Figure(go.Bar(
        x=by_loc.index, y=by_loc.values, marker_color=BLUE,
        text=[f"${v:,.0f}" for v in by_loc.values], textposition="outside", cliponaxis=False,
    ))
    fig.update_yaxes(title="Median house value", tickprefix="$", tickformat=",", range=[0, by_loc.max() * 1.18])
    st.plotly_chart(_style(fig, 340), use_container_width=True)
    st.caption("Median price by ocean proximity. ISLAND has only a handful of rows, so read it with care.")


# ---------------------------------------------------------------------------
# About page
# ---------------------------------------------------------------------------
def render_about(metadata: dict):
    st.title("ℹ️ About this project")
    st.markdown(
        f"""
**Purpose.** Estimate the median house value of a California census district
from its features. Educational project, not a property valuation.

**Dataset.** California Housing, 1990 US census: 20,640 districts, 9 input
features and the target `median_house_value`.
Prices are in **US dollars (1990)** and describe a **district median**, not one house.
The target is capped at **$500,001**.

**Model.** {metadata['best_model']}, chosen from Linear Regression, Decision Tree and
Random Forest. One shared pipeline (imputer, scaler, one-hot encoder, model) is fitted
on the training set only, so there is no data leakage.
Test R² = **{metadata['test_r2']:.2f}**, test MAE = **${metadata['test_mae']:,.0f}**.
R² is the share of price variation the model explains on unseen data, not a percentage accuracy.

**Limitations**
- 1990 data: do not use it to value real property today.
- Prices above $500,001 are unknown, so expensive districts are underestimated.
- Predictions are for a whole district, not a single house.
- `ISLAND` has only 5 rows, so its estimates are unreliable.
- The model overfits somewhat: its training R² is higher than its test R².
        """
    )