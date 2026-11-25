import joblib
import numpy as np

from src.data_preprocessing import add_features, split_data
from src.evaluation import cap_check, compare_models
from src.model_training import build_models, save_artifacts, train_models


def _fitted(tiny_df):
    X_train, X_test, y_train, y_test = split_data(add_features(tiny_df))
    pipelines = build_models()
    train_models(pipelines, X_train, y_train)
    return pipelines, X_train, X_test, y_train, y_test


def test_all_models_predict_from_raw_rows(tiny_df):
    pipelines, _, X_test, _, _ = _fitted(tiny_df)
    assert set(pipelines) == {"Linear Regression", "Decision Tree", "Random Forest"}
    for pipe in pipelines.values():
        pred = pipe.predict(X_test)
        assert pred.shape == (len(X_test),)
        assert np.isfinite(pred).all()


def test_compare_models_columns(tiny_df):
    pipelines, X_train, X_test, y_train, y_test = _fitted(tiny_df)
    results = compare_models(pipelines, X_train, y_train, X_test, y_test)
    assert list(results.index) == list(pipelines)
    assert {"Train R²", "CV R² (mean)", "Test R²", "Test MAE ($)", "Test RMSE ($)"} <= set(results.columns)
    assert (results["Test MAE ($)"] >= 0).all()


def test_cap_check_ignores_capped_rows(tiny_df):
    pipelines, _, X_test, _, y_test = _fitted(tiny_df)
    y_capped = y_test.copy()
    y_capped.iloc[0] = 500_001
    out = cap_check(pipelines, X_test, y_capped)
    assert {"MAE below cap ($)", "R² below cap"} <= set(out.columns)


def test_save_artifacts_writes_files(tiny_df, tmp_path):
    pipelines, X_train, X_test, y_train, y_test = _fitted(tiny_df)
    results = compare_models(pipelines, X_train, y_train, X_test, y_test)
    meta = save_artifacts("Linear Regression", pipelines["Linear Regression"], X_train, results, tmp_path)
    assert (tmp_path / "best_model.joblib").exists()
    assert (tmp_path / "model_comparison.csv").exists()
    assert "rooms_per_household" in joblib.load(tmp_path / "metadata.joblib")["feature_columns"]
    assert meta["best_model"] == "Linear Regression"