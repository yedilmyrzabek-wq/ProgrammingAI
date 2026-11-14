import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src import config

METHODS = ("Baseline", "Z-score", "Rolling IQR", "Isolation Forest")
BASELINE_QUANTILES = (0.01, 0.99)
Z_LIMIT = 3.0
IQR_WINDOW = "48h"
IQR_MIN_READINGS = 12
IQR_MULTIPLIER = 1.5
FOREST_CONTAMINATION = 0.02


def _outside(values, low, high):
    available = values.notna() & low.notna() & high.notna()
    return (values.lt(low) | values.gt(high)).astype("boolean").where(available)


def _fixed_thresholds(train, test, method):
    if method == "Baseline":
        low, high = (train.quantile(q) for q in BASELINE_QUANTILES)
    else:
        center = train.mean()
        radius = Z_LIMIT * train.std(ddof=0)
        low, high = center - radius, center + radius
    low = pd.DataFrame({name: bound for name, bound in low.items()}, index=test.index)
    high = pd.DataFrame({name: bound for name, bound in high.items()}, index=test.index)
    return _outside(test, low, high)


def _rolling_iqr(well, parameters, test_index):
    values = well.set_index("timestamp")[parameters].sort_index()
    history = values.rolling(IQR_WINDOW, min_periods=IQR_MIN_READINGS, closed="left")
    low, high = history.quantile(0.25), history.quantile(0.75)
    spread = IQR_MULTIPLIER * (high - low)
    predictions = _outside(values, low - spread, high + spread)
    return predictions.reindex(well.loc[test_index, "timestamp"]).set_axis(test_index)


def _isolation_forest(train, test):
    median = train.median().dropna()
    predictions = pd.Series(pd.NA, index=test.index, dtype="boolean", name="anomaly")
    if median.empty:
        return predictions
    train = train[median.index].dropna(how="all")
    test = test[median.index].dropna(how="all")
    model = IsolationForest(
        contamination=FOREST_CONTAMINATION, random_state=config.RANDOM_STATE
    )
    model.fit(train.fillna(median))
    if not test.empty:
        predictions.loc[test.index] = model.predict(test.fillna(median)) == -1
    return predictions


def _predict_well(well, parameters, method):
    train = well.loc[well["split"].eq("train"), parameters]
    test = well.loc[well["split"].eq("test"), parameters]
    if train.empty:
        return pd.DataFrame(pd.NA, index=test.index, columns=parameters, dtype="boolean")
    if method == "Rolling IQR":
        return _rolling_iqr(well, parameters, test.index)
    if method == "Isolation Forest":
        return _isolation_forest(train, test).to_frame()
    return _fixed_thresholds(train, test, method)


def predict(wide, method):
    """Return nullable test predictions indexed like wide, per sensor or multivariate."""
    if method not in METHODS:
        raise ValueError(f"Unknown detector: {method}")
    parameters = [name for name in config.PARAMETERS if name in wide]
    if not parameters:
        raise ValueError("The dataset contains no supported sensor columns.")
    readings = wide.loc[:, ["well_id", "timestamp", "split", *parameters]].copy()
    readings[parameters] = readings[parameters].replace([np.inf, -np.inf], np.nan)
    test_index = readings.index[readings["split"].eq("test")]
    columns = ["anomaly"] if method == "Isolation Forest" else parameters
    parts = [
        _predict_well(well, parameters, method).reindex(columns=columns)
        for _, well in readings.groupby("well_id", observed=True)
    ]
    if not parts:
        return pd.DataFrame(index=test_index, columns=columns, dtype="boolean")
    return pd.concat(parts).reindex(test_index).astype("boolean")


def hourly_anomalies(predictions):
    """Combine sensor predictions at well-hour level, preserving unscored hours."""
    return predictions.any(axis=1).astype("boolean").where(predictions.notna().any(axis=1))
