"""Metrics used to compare the models."""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

from src.data_preprocessing import PRICE_CAP
from src.model_training import build_models


def evaluate(name, pipe, X_train, y_train, X_test, y_test) -> dict:
    """Train R², 5-fold CV R² (training data only) and test metrics for one model."""
    pred_train = pipe.predict(X_train)
    pred_test = pipe.predict(X_test)
    cv = cross_val_score(
        build_models()[name], X_train, y_train, cv=5, scoring="r2", n_jobs=-1
    )
    return {
        "Model": name,
        "Train R²": r2_score(y_train, pred_train),
        "CV R² (mean)": cv.mean(),
        "CV R² (std)": cv.std(),
        "Test R²": r2_score(y_test, pred_test),
        "Test MAE ($)": mean_absolute_error(y_test, pred_test),
        "Test RMSE ($)": np.sqrt(mean_squared_error(y_test, pred_test)),
    }


def compare_models(pipelines, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    rows = [evaluate(n, p, X_train, y_train, X_test, y_test) for n, p in pipelines.items()]
    return pd.DataFrame(rows).set_index("Model")


def cap_check(pipelines, X_test, y_test) -> pd.DataFrame:
    """Scores on test districts below the price cap only."""
    below = y_test < PRICE_CAP
    rows = []
    for name, pipe in pipelines.items():
        pred = pipe.predict(X_test[below])
        rows.append({
            "Model": name,
            "MAE below cap ($)": mean_absolute_error(y_test[below], pred),
            "R² below cap": r2_score(y_test[below], pred),
        })
    return pd.DataFrame(rows).set_index("Model")