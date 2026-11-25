import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tiny_df():
    """Small fake housing table (raw columns, no engineered features)."""
    rng = np.random.default_rng(0)
    n = 60
    households = rng.integers(50, 500, n).astype(float)
    rooms = households * rng.uniform(3, 7, n)
    df = pd.DataFrame({
        "longitude": rng.uniform(-124, -114, n),
        "latitude": rng.uniform(32, 42, n),
        "housing_median_age": rng.integers(1, 52, n).astype(float),
        "total_rooms": rooms,
        "total_bedrooms": rooms * rng.uniform(0.15, 0.3, n),
        "population": households * rng.uniform(2, 4, n),
        "households": households,
        "median_income": rng.uniform(1, 10, n),
        "ocean_proximity": rng.choice(["INLAND", "NEAR BAY", "<1H OCEAN"], n),
    })
    df["median_house_value"] = 50_000 + df["median_income"] * 40_000 + rng.normal(0, 10_000, n)
    return df