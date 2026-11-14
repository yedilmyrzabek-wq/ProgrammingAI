from contextlib import closing

import pandas as pd

from src import config, database

AREA_PARAMETER = "dynamometer_area"
REGIME_WELL = "NK-7"


def _area_shift(long_df):
    area = long_df.loc[
        long_df["parameter"].eq(AREA_PARAMETER) & long_df["flag"].fillna("").eq(""),
        ["well_id", "timestamp", "value"],
    ].copy()
    area["month"] = area["timestamp"].dt.to_period("M")
    shift = pd.Period(config.AREA_SHIFT_MONTH, freq="M")
    area = area[area["month"].isin([shift - 1, shift])]
    medians = area.pivot_table(
        index="well_id", columns="month", values="value", aggfunc="median", observed=True
    ).reindex(columns=[shift - 1, shift])
    medians.columns = ["previous_month_median", "shift_month_median"]
    denominator = medians["previous_month_median"]
    medians["ratio"] = medians["shift_month_median"].div(denominator.where(denominator.ne(0)))
    return medians.reset_index()


def _regime_change(long_df):
    readings = long_df.loc[
        long_df["well_id"].eq(REGIME_WELL), ["timestamp", "parameter", "value"],
    ].copy()
    before = readings["timestamp"].lt(pd.Timestamp(config.NK7_REGIME_START))
    readings["period"] = before.map({True: "before", False: "after"})
    readings["is_zero"] = readings["value"].eq(0)
    return readings.groupby(["parameter", "period"], observed=True, as_index=False).agg(
        observed_values=("value", "count"),
        median=("value", "median"),
        zero_values=("is_zero", "sum"),
    )


def audit_quality(long_df, db_path=config.DB_PATH):
    """Return gap, missing-value, flag and sensor drift tables."""
    flags = long_df["flag"].astype("string").fillna("").str.strip()
    flagged = long_df.loc[flags.ne(""), ["well_id", "parameter"]].assign(flag=flags)
    with closing(database.connect(db_path)) as conn:
        gaps = database.run_query("time_gaps", conn)
        missing_hours = database.run_query("missing_hours_per_well", conn)
        flag_rates = database.run_query("flag_rate_ranking", conn)
    return {
        "Missing hours by well": missing_hours,
        "Hourly gaps": gaps,
        "Null readings": long_df.loc[long_df["value"].isna()].groupby(
            ["well_id", "parameter"], observed=True, as_index=False
        ).size(),
        "Flag rates by well": flag_rates,
        "Flag codes": flagged.groupby(
            ["well_id", "parameter", "flag"], observed=True, as_index=False
        ).size().sort_values("size", ascending=False),
        f"Dynamometer area shift ({config.AREA_SHIFT_MONTH})": _area_shift(long_df),
        f"{REGIME_WELL} regime change ({config.NK7_REGIME_START})": _regime_change(long_df),
    }
