"""Streamlit entry point: loads the saved model and wires inputs -> prediction -> UI."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import ui
from src.data_preprocessing import add_features, read_csv_safely, validate_dataset

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "housing.csv"

PAGES = ["Price Estimator", "Dataset Explorer", "About"]

st.set_page_config(page_title="California House Price Estimator", page_icon="🏠", layout="wide")


@st.cache_resource
def load_artifacts():
    """Load the trained pipeline and its metadata once per session."""
    model = joblib.load(MODEL_DIR / "best_model.joblib")
    metadata = joblib.load(MODEL_DIR / "metadata.joblib")
    return model, metadata


def predict(model, metadata: dict, rows: list[dict]) -> list[float]:
    """Predict for one or more raw input rows (never below $0)."""
    df = add_features(pd.DataFrame(rows))[metadata["feature_columns"]]
    return [max(float(p), 0.0) for p in model.predict(df)]


def explorer_page():
    """Let the user upload a CSV, or fall back to the built-in dataset."""
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is not None:
        df, error = read_csv_safely(uploaded, uploaded.name)
        label = uploaded.name
    else:
        df, error = read_csv_safely(DATA_PATH, DATA_PATH.name)
        label = "built-in California Housing dataset"
    if error:
        st.error(error)
        return
    ui.render_explorer(validate_dataset(df), label)


def main():
    try:
        model, metadata = load_artifacts()
    except FileNotFoundError:
        st.error(
            "Model files not found in `models/`. Open `house_price_analysis.ipynb`, "
            "run all cells (this saves the model), then restart the app."
        )
        st.stop()

    with st.sidebar:
        page = st.radio("Go to", PAGES, label_visibility="collapsed")
        st.divider()

    if page == "Price Estimator":
        ui.render_header(metadata)
        values = ui.render_sidebar(metadata)
        price = predict(model, metadata, [values])[0]
        ui.render_result(price, values, metadata)
        ui.render_charts(price, values, metadata, lambda rows: predict(model, metadata, rows))
        ui.render_model_info(metadata, MODEL_DIR / "model_comparison.csv")
        ui.render_disclaimer()
    elif page == "Dataset Explorer":
        explorer_page()
    else:
        ui.render_about(metadata)


main()