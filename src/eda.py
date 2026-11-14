import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

from src import config

PARAMS_BY_TYPE = {"fountain": config.FOUNTAIN_PARAMS, "sucker_rod_pump": config.PUMP_PARAMS}
PALETTE = "#1f4e79"


def clean_values(wide):
    clean = wide.copy()
    for parameter in config.PARAMETERS:
        flag_column = f"flag_{parameter}"
        if parameter in clean and flag_column in clean:
            clean.loc[clean[flag_column].astype(bool), parameter] = float("nan")
    return clean


def describe_by_type(wide):
    tables = []
    for well_type, parameters in PARAMS_BY_TYPE.items():
        subset = wide.loc[wide["well_type"].eq(well_type), parameters]
        stats = subset.describe().T
        stats["missing_pct"] = subset.isna().mean().mul(100).round(2)
        stats.insert(0, "well_type", well_type)
        tables.append(stats)
    return pd.concat(tables).rename_axis("parameter").reset_index()


def iqr_outlier_share(wide):
    rows = []
    for parameter in config.PARAMETERS:
        values = wide[parameter].dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        spread = 1.5 * (q3 - q1)
        outside = values.lt(q1 - spread) | values.gt(q3 + spread)
        rows.append({"parameter": parameter, "n": len(values), "outlier_pct": round(100 * outside.mean(), 2)})
    return pd.DataFrame(rows).sort_values("outlier_pct", ascending=False, ignore_index=True)


def daily_pattern(wide):
    frame = wide.assign(hour=wide["timestamp"].dt.hour)
    profiles = {}
    for well_type, parameters in PARAMS_BY_TYPE.items():
        subset = frame.loc[frame["well_type"].eq(well_type)]
        hourly = subset.groupby("hour")[parameters].mean()
        profiles[well_type] = hourly / hourly.mean()
    return profiles


def correlations(wide):
    return {
        well_type: wide.loc[wide["well_type"].eq(well_type), parameters].corr()
        for well_type, parameters in PARAMS_BY_TYPE.items()
    }


def _save(fig, name):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="#f7f6f2")
    plt.close(fig)
    return path


def plot_daily_pattern(profiles):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, (well_type, hourly) in zip(axes, profiles.items()):
        hourly.plot(ax=ax, linewidth=1.8)
        ax.axhline(1, color="grey", linewidth=0.8, linestyle="--")
        ax.set_title(well_type.replace("_", " ").title())
        ax.set_xlabel("Hour of day")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Hourly mean / daily mean")
    fig.suptitle("Daily patterns by well type")
    return _save(fig, "daily_pattern.png")


def plot_correlations(matrices):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 5]})
    for ax, (well_type, matrix) in zip(axes, matrices.items()):
        sns.heatmap(matrix, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax, cbar=False)
        ax.set_title(well_type.replace("_", " ").title())
    fig.suptitle("Parameter correlations (flagged readings removed)")
    return _save(fig, "correlations.png")


def plot_boxplots(wide):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    wide = wide.assign(well_id=wide["well_id"].astype(str))
    pumps = wide.loc[wide["well_type"].eq("sucker_rod_pump")]
    fountains = wide.loc[wide["well_type"].eq("fountain")]
    sns.boxplot(data=pumps, x="well_id", y="pump_fillage", color=PALETTE, fliersize=1.5, ax=axes[0])
    sns.boxplot(data=fountains, x="well_id", y="tubing_pressure", color=PALETTE, fliersize=1.5, ax=axes[1])
    axes[0].set_title("Pump fillage, %")
    axes[1].set_title("Tubing pressure, atm")
    for ax in axes:
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Outliers by well (IQR boxplots)")
    return _save(fig, "boxplots.png")


def plot_area_shift(wide):
    area = wide.loc[wide["well_type"].eq("sucker_rod_pump"), ["well_id", "timestamp", "dynamometer_area"]]
    area = area.loc[area["dynamometer_area"].gt(0)]
    monthly = (
        area.assign(month=area["timestamp"].dt.to_period("M").dt.to_timestamp())
        .groupby(["month", "well_id"], observed=True)["dynamometer_area"].median()
        .unstack()
    )
    fig, ax = plt.subplots(figsize=(10, 4.5))
    monthly.plot(ax=ax, marker="o", linewidth=1.5)
    ax.axvline(pd.Timestamp(f"{config.AREA_SHIFT_MONTH}-01"), color="crimson", linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_title("Dynamometer area: monthly median jumps in May 2026")
    ax.set_xlabel("")
    ax.set_ylabel("Median area (log scale)")
    ax.legend(fontsize=8, ncol=2)
    return _save(fig, "dynamometer_shift.png")


def plot_interactive_series(wide, well_id="NK-68", parameter="pump_fillage"):
    series = wide.loc[wide["well_id"].eq(well_id), ["timestamp", parameter, f"flag_{parameter}"]]
    series = series.rename(columns={f"flag_{parameter}": "flagged"})
    fig = px.line(series, x="timestamp", y=parameter, title=f"{well_id}: {parameter}")
    flagged = series.loc[series["flagged"].astype(bool)]
    fig.add_scatter(x=flagged["timestamp"], y=flagged[parameter], mode="markers",
                    marker={"color": "crimson", "size": 6}, name="quality flag")
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES_DIR / "timeseries_interactive.html"
    fig.write_html(path, include_plotlyjs="cdn")
    return path


def run_eda(wide):
    clean = clean_values(wide)
    tables = {
        "Descriptive statistics": describe_by_type(clean),
        "IQR outlier share": iqr_outlier_share(clean),
    }
    figures = [
        plot_daily_pattern(daily_pattern(clean)),
        plot_correlations(correlations(clean)),
        plot_boxplots(clean),
        plot_area_shift(clean),
        plot_interactive_series(wide),
    ]
    return tables, figures
