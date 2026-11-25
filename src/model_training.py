"""Model definitions, training and saving."""
import time
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from src.data_preprocessing import (
    CAT_COLS, NUM_COLS, PRICE_CAP, RANDOM_STATE, build_preprocessor,
)

MODEL_DIR = Path("models")


def build_models() -> dict:
    """Return fresh, unfitted pipelines (preprocessing + model)."""
    estimators = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=8, min_samples_leaf=10, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE
        ),
    }
    return {
        name: Pipeline([("prep", build_preprocessor()), ("model", est)])
        for name, est in estimators.items()
    }


def train_models(pipelines: dict, X_train, y_train) -> dict:
    """Fit every pipeline on the training set. Returns fit times in seconds."""
    fit_times = {}
    for name, pipe in pipelines.items():
        start = time.time()
        pipe.fit(X_train, y_train)
        fit_times[name] = time.time() - start
        print(f"{name:<18} trained in {fit_times[name]:.2f}s")
    return fit_times


def save_artifacts(best_name, best_pipe, X_train, results, model_dir=MODEL_DIR):
    """Save the model, the metadata the app needs and the comparison table."""
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)
    joblib.dump(best_pipe, model_dir / "best_model.joblib")
    metadata = {
        "best_model": best_name,
        "feature_columns": X_train.columns.tolist(),
        "numeric_columns": NUM_COLS,
        "categorical_columns": CAT_COLS,
        "ocean_categories": sorted(X_train["ocean_proximity"].unique().tolist()),
        "price_cap": PRICE_CAP,
        "test_r2": float(results.loc[best_name, "Test R²"]),
        "test_mae": float(results.loc[best_name, "Test MAE ($)"]),
        "numeric_ranges": {c: [float(X_train[c].min()), float(X_train[c].max())] for c in NUM_COLS},
    }
    joblib.dump(metadata, model_dir / "metadata.joblib")
    results.round(4).to_csv(model_dir / "model_comparison.csv")
    return metadata