"""Data loading, cleaning, feature engineering, splitting and the shared preprocessing step."""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "median_house_value"
PRICE_CAP = 500_001
BASE_NUM_COLS = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
]
ENGINEERED_COLS = ["rooms_per_household", "bedrooms_per_room", "population_per_household"]
NUM_COLS = BASE_NUM_COLS + ENGINEERED_COLS
CAT_COLS = ["ocean_proximity"]
DATA_PATH = Path("data") / "housing.csv"


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV file."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows. Missing values are handled later, inside the pipeline."""
    return df.drop_duplicates().reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio features. Row-by-row maths, so it is safe to use before the split.

    A zero denominator gives NaN (not infinity); the pipeline imputer fills it.
    """
    out = df.copy()
    households = out["households"].replace(0, np.nan)
    rooms = out["total_rooms"].replace(0, np.nan)
    out["rooms_per_household"] = out["total_rooms"] / households
    out["bedrooms_per_room"] = out["total_bedrooms"] / rooms
    out["population_per_household"] = out["population"] / households
    return out


def split_data(df: pd.DataFrame, test_size: float = 0.2):
    """Return X_train, X_test, y_train, y_test (80/20 split, fixed seed)."""
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)


def build_preprocessor() -> ColumnTransformer:
    """Median-impute + scale numbers; one-hot encode the category.

    Fitted on training data only (inside the Pipeline), which avoids data leakage.
    """
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="ignore")
    return ColumnTransformer([
        ("num", numeric, NUM_COLS),
        ("cat", categorical, CAT_COLS),
    ])


# ---------------------------------------------------------------------------
# Upload validation (used by the Dataset Explorer page in the app)
# ---------------------------------------------------------------------------
REQUIRED_FEATURES = BASE_NUM_COLS + CAT_COLS
MIN_ROWS_ERROR = 30    # fewer rows than this cannot be split and used sensibly
MIN_ROWS_WARN = 100    # fewer rows than this gives unreliable results


@dataclass
class ValidationResult:
    """Outcome of validate_dataset(). `data` is the cleaned table, or None if invalid."""
    data: pd.DataFrame | None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors and self.data is not None


def read_csv_safely(source, name: str | None = None):
    """Read an uploaded CSV. Returns (DataFrame, None) or (None, error message)."""
    if name is not None and not str(name).lower().endswith(".csv"):
        return None, "Wrong file type. Please upload a .csv file."
    try:
        return pd.read_csv(source), None
    except pd.errors.EmptyDataError:
        return None, "The file is empty. Please upload a CSV that contains data."
    except (pd.errors.ParserError, UnicodeDecodeError):
        return None, "The file could not be read as a CSV. Check that it is a plain comma-separated file."


def validate_dataset(df: pd.DataFrame | None) -> ValidationResult:
    """Check an uploaded table and return a cleaned copy plus clear messages."""
    errors: list[str] = []
    warnings: list[str] = []

    if df is None or len(df) == 0:
        return ValidationResult(None, ["The file has no data rows."], [], {"rows_uploaded": 0})
    summary = {"rows_uploaded": len(df)}

    if TARGET not in df.columns:
        errors.append(f"Target column `{TARGET}` was not found. This column is required.")
    missing_cols = [c for c in REQUIRED_FEATURES if c not in df.columns]
    if missing_cols:
        errors.append("Missing required columns: " + ", ".join(f"`{c}`" for c in missing_cols))
    if errors:
        return ValidationResult(None, errors, warnings, summary)

    out = df.copy()

    # Numeric feature columns: text values become missing (the pipeline imputes them later)
    for col in BASE_NUM_COLS:
        before = out[col].isna().sum()
        out[col] = pd.to_numeric(out[col], errors="coerce")
        converted = int(out[col].isna().sum() - before)
        if converted:
            warnings.append(f"`{col}`: {converted} non-numeric value(s) were treated as missing.")

    # Target must be numeric and positive
    out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce")
    bad_target = ~(out[TARGET] > 0)
    n_bad = int(bad_target.sum())
    summary["invalid_target_rows"] = n_bad
    if n_bad == len(out):
        return ValidationResult(None, [f"`{TARGET}` has no valid numeric values."], warnings, summary)
    if n_bad:
        warnings.append(f"{n_bad} row(s) removed because `{TARGET}` is missing, non-numeric or not positive.")
        out = out[~bad_target]

    # Duplicates
    n_dup = int(out.duplicated().sum())
    summary["duplicate_rows"] = n_dup
    if n_dup:
        warnings.append(f"{n_dup} duplicate row(s) removed.")
    out = clean_data(out)

    # Missing feature values (kept; the model pipeline fills them)
    missing = out[REQUIRED_FEATURES].isna().sum()
    missing = missing[missing > 0]
    summary["missing_values"] = missing.to_dict()
    if len(missing):
        warnings.append(
            "Missing values found in " + ", ".join(f"`{c}` ({n})" for c, n in missing.items())
            + ". They are kept and would be filled by the pipeline when training."
        )

    # Dataset size
    summary["rows_after_cleaning"] = len(out)
    if len(out) < MIN_ROWS_ERROR:
        return ValidationResult(
            None, [f"Only {len(out)} usable rows. At least {MIN_ROWS_ERROR} are needed for a train/test split."],
            warnings, summary,
        )
    if len(out) < MIN_ROWS_WARN:
        warnings.append(f"Small dataset ({len(out)} rows). Results from so little data are unreliable.")

    return ValidationResult(out, [], warnings, summary)