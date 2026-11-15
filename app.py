import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config, data_loader, database, eda, evaluation, quality
from src.detectors import METHODS, predict

INK = "#e6e6e6"
MUTED = "#6e6e6e"
SPLIT = "#9a9a9a"
ACCENT = "#ffcc00"
FLAG = "#e5533d"
GRID = "#333333"
PAPER = "#222222"
FONT = "Roboto, sans-serif"
METRIC_COLORS = {"precision": "#ffcc00", "recall": "#8c8c8c", "f1": "#e6e6e6"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@500&family=Roboto:wght@400;500;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select {
    font-family: 'Roboto', system-ui, sans-serif;
}
[data-testid="stHeader"] { background: #1a1a1a; border-bottom: 1px solid #333; }
[data-testid="stSidebar"] { background: #1a1a1a; border-right: 1px solid #333; }
[data-testid="stSidebar"] h2 { color: #ffcc00 !important; }
h1 { font-size: 1.9rem !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 0.02em;
     padding-bottom: 0.1rem !important; }
h2, h3 { font-size: 0.95rem !important; font-weight: 700 !important; text-transform: uppercase; letter-spacing: 0.08em;
         color: #bdbdbd !important; padding-top: 0.8rem !important; }
.block-container { padding-top: 2.4rem; max-width: 1280px; }
::selection { background: #ffcc00; color: #1a1a1a; }
:focus-visible { outline: 2px solid #ffcc00 !important; outline-offset: 2px; }
[data-testid="stDataFrame"], .num { font-variant-numeric: tabular-nums; }
[data-testid="stTab"] p { text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.82rem; font-weight: 500; }
.subtitle { color: #9a9a9a; margin: 0 0 1.6rem 0; font-size: 0.95rem; }
.statline { display: flex; flex-wrap: wrap; background: #2b2b2b; border: 1px solid #3a3a3a; margin: 0.4rem 0 0.8rem 0; }
.stat { display: flex; flex-direction: column; flex: 1 1 0; min-width: 9rem; padding: 0.75rem 1.1rem;
        border-right: 1px solid #444; }
.stat:last-child { border-right: none; }
.stat b { font-family: 'Roboto Mono', monospace; font-weight: 500; font-size: 1.4rem; color: #e6e6e6; }
.stat span { color: #9a9a9a; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }
.stat.hot b { color: #ffcc00; }
.note { color: #9a9a9a; font-size: 0.85rem; }
</style>
"""


@st.cache_data(show_spinner="Loading sensor data…")
def load_frames():
    long_df = data_loader.load_long()
    wide = data_loader.load_wide()
    if not config.DB_PATH.exists():
        database.build_database(long_df)
    return long_df, wide


@st.cache_data(show_spinner="Running detector on the test split…")
def detector_output(method):
    _, wide = load_frames()
    return predict(wide, method)


@st.cache_data(show_spinner="Auditing data quality…")
def quality_tables():
    long_df, _ = load_frames()
    return quality.audit_quality(long_df)


@st.cache_data(show_spinner="Comparing detectors…")
def comparison_table():
    path = config.REPORTS_DIR / "comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    _, wide = load_frames()
    comparison = evaluation.compare_methods(wide)
    evaluation.save_comparison(comparison)
    return comparison


@st.cache_data(show_spinner="Building exploration figures…")
def exploration():
    _, wide = load_frames()
    return eda.run_eda(wide)


def rounded(frame, digits=2):
    floats = frame.select_dtypes("float32").columns
    return frame.astype({column: "float64" for column in floats}).round(digits)


def units_by_parameter(long_df):
    pairs = long_df[["parameter", "unit"]].drop_duplicates()
    return dict(zip(pairs["parameter"].astype(str), pairs["unit"].astype(str)))


def sidebar_filters(wide):
    st.sidebar.header("Filters")
    wells = wide[["well_id", "well_type"]].drop_duplicates().astype(str).sort_values("well_id")
    types = dict(zip(wells["well_id"], wells["well_type"]))
    well = st.sidebar.selectbox(
        "Well", list(types), format_func=lambda w: f"{w} · {types[w].replace('_', ' ')}"
    )
    parameters = config.FOUNTAIN_PARAMS if types[well] == "fountain" else config.PUMP_PARAMS
    available = [p for p in parameters if wide.loc[wide["well_id"].eq(well), p].notna().any()]
    parameter = st.sidebar.selectbox("Parameter", available, format_func=lambda p: p.replace("_", " "))
    method = st.sidebar.radio("Detection method", METHODS, index=1)

    test = wide.loc[wide["split"].eq("test"), "timestamp"]
    first, last = wide["timestamp"].min().date(), wide["timestamp"].max().date()
    default = (test.min().date() - pd.Timedelta(days=14), last)
    period = st.sidebar.date_input("Period", value=default, min_value=first, max_value=last)
    show_flags = st.sidebar.toggle("Show quality flags", value=True)
    st.sidebar.caption("Detectors are fitted on the train split and score only the test split.")
    if not isinstance(period, tuple) or len(period) != 2:
        period = default
    return well, parameter, method, period, show_flags


def anomaly_mask(predictions, frame, parameter):
    column = "anomaly" if "anomaly" in predictions else parameter
    scored = predictions[column].reindex(frame.index)
    return scored.fillna(False).astype(bool), scored.notna()


def statline(items):
    cells = "".join(
        f'<div class="stat{" hot" if hot else ""}"><b>{value}</b><span>{label}</span></div>'
        for value, label, hot in items
    )
    st.markdown(f'<div class="statline">{cells}</div>', unsafe_allow_html=True)


def series_chart(frame, parameter, unit, anomalies, show_flags):
    flagged = frame[f"flag_{parameter}"].astype(bool)
    test_start = frame.loc[frame["split"].eq("test"), "timestamp"].min()
    fig = go.Figure()
    for split, color, width in (("train", MUTED, 1.1), ("test", INK, 1.2)):
        part = frame.loc[frame["split"].eq(split)]
        fig.add_scatter(x=part["timestamp"], y=part[parameter], mode="lines", name=f"{split} readings",
                        line={"color": color, "width": width}, connectgaps=False)
    if show_flags and flagged.any():
        fig.add_scatter(x=frame.loc[flagged, "timestamp"], y=frame.loc[flagged, parameter], mode="markers",
                        name="quality flag", marker={"color": "rgba(0,0,0,0)", "size": 9,
                                                     "line": {"color": FLAG, "width": 1.6}})
    fig.add_scatter(x=frame.loc[anomalies, "timestamp"], y=frame.loc[anomalies, parameter], mode="markers",
                    name="detected anomaly", marker={"color": ACCENT, "size": 6})
    if pd.notna(test_start):
        fig.add_vrect(x0=test_start, x1=frame["timestamp"].max(), fillcolor="#ffffff", opacity=0.03, line_width=0)
        fig.add_vline(x=test_start, line={"color": SPLIT, "dash": "dot", "width": 1})
        fig.add_annotation(x=test_start, y=1, yref="paper", text="TEST SPLIT", showarrow=False,
                           xanchor="left", xshift=6, font={"color": SPLIT, "size": 12})
    fig.update_layout(
        height=430, margin={"l": 10, "r": 10, "t": 20, "b": 10}, paper_bgcolor=PAPER, plot_bgcolor=PAPER,
        font={"family": FONT, "color": INK, "size": 13}, hovermode="x unified",
        legend={"orientation": "h", "y": -0.12, "x": 0}, yaxis_title=f"{parameter.replace('_', ' ')}, {unit}",
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def anomalies_tab(wide, units, well, parameter, method, period, show_flags):
    start, end = pd.Timestamp(period[0]), pd.Timestamp(period[1]) + pd.Timedelta(hours=23)
    frame = wide.loc[wide["well_id"].eq(well) & wide["timestamp"].between(start, end)]
    if frame[parameter].notna().sum() == 0:
        st.info("No readings for this well and parameter in the selected period. Widen the period in the sidebar.")
        return
    anomalies, scored = anomaly_mask(detector_output(method), frame, parameter)
    flagged = frame[f"flag_{parameter}"].astype(bool)
    statline([
        (f"{frame[parameter].notna().sum():,}", "hours with readings", False),
        (f"{int(scored.sum()):,}", "hours scored (test split)", False),
        (f"{int(anomalies.sum()):,}", f"anomalies · {method}", True),
        (f"{int((flagged & scored).sum()):,}", "flagged in scored hours", False),
        (f"{int((anomalies & flagged).sum()):,}", "anomalies also flagged", False),
    ])
    st.plotly_chart(series_chart(frame, parameter, units.get(parameter, ""), anomalies, show_flags),
                    width="stretch", config={"displaylogo": False})
    if not scored.any():
        st.caption("The selected period ends before the test split, so no hours were scored.")
        return
    rows = frame.loc[anomalies, ["timestamp", parameter, f"flag_{parameter}"]]
    rows = rows.rename(columns={parameter: f"value, {units.get(parameter, '')}", f"flag_{parameter}": "quality flag"})
    st.subheader(f"Detected anomalies ({len(rows):,})")
    if rows.empty:
        st.caption(f"{method} found no anomalies for this parameter in the scored hours.")
    else:
        st.dataframe(rounded(rows.sort_values("timestamp", ascending=False)), hide_index=True,
                     width="stretch",
                     column_config={"timestamp": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm")})


def comparison_chart(comparison):
    fig = go.Figure()
    for metric, color in METRIC_COLORS.items():
        fig.add_bar(x=comparison["method"], y=comparison[metric], name=metric.upper() if metric == "f1" else metric,
                    marker_color=color, text=comparison[metric].round(2), textposition="outside")
    fig.update_layout(
        barmode="group", height=380, margin={"l": 10, "r": 10, "t": 20, "b": 10}, paper_bgcolor=PAPER,
        plot_bgcolor=PAPER, font={"family": FONT, "color": INK, "size": 13},
        yaxis={"range": [0, min(1, comparison[list(METRIC_COLORS)].max().max() * 1.3)], "gridcolor": GRID,
               "title": "score vs. quality flags"},
        legend={"orientation": "h", "y": -0.15, "x": 0},
    )
    return fig


def comparison_tab():
    comparison = comparison_table()
    best_f1 = comparison.loc[comparison["f1"].idxmax()]
    best_precision = comparison.loc[comparison["precision"].idxmax()]
    fastest = comparison.loc[comparison["seconds"].idxmin()]
    st.markdown(
        f"**{best_f1['method']}** agrees best with the quality flags (F1 {best_f1['f1']:.3f}). "
        f"**{best_precision['method']}** is the most precise ({best_precision['precision']:.3f}), "
        f"and **{fastest['method']}** is the fastest ({fastest['seconds']:.2f} s)."
    )
    st.plotly_chart(comparison_chart(comparison), width="stretch", config={"displaylogo": False})
    table = comparison[["method", "precision", "recall", "f1", "seconds", "peak_traced_mib", "predicted_anomalies"]]
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "precision": st.column_config.NumberColumn(format="%.3f"),
        "recall": st.column_config.NumberColumn(format="%.3f"),
        "f1": st.column_config.NumberColumn("F1", format="%.3f"),
        "seconds": st.column_config.NumberColumn("time, s", format="%.2f"),
        "peak_traced_mib": st.column_config.NumberColumn("peak memory, MiB", format="%.1f"),
        "predicted_anomalies": st.column_config.NumberColumn("anomalies"),
    })
    hours, flagged = int(comparison["scored_hours"].iloc[0]), int(comparison["flagged_hours"].iloc[0])
    st.markdown(
        f'<p class="note">Scored on {hours:,} test well-hours, {flagged:,} of them flagged. '
        "Flags are incomplete, so these scores measure agreement, not accuracy.</p>",
        unsafe_allow_html=True,
    )


def missing_chart(missing):
    ordered = missing.sort_values("missing_pct")
    fig = go.Figure(go.Bar(x=ordered["missing_pct"], y=ordered["well_id"], orientation="h", marker_color=ACCENT,
                           text=ordered["missing_pct"].map("{:.1f}%".format), textposition="outside"))
    fig.update_layout(
        height=340, margin={"l": 10, "r": 30, "t": 10, "b": 10}, paper_bgcolor=PAPER, plot_bgcolor=PAPER,
        font={"family": FONT, "color": INK, "size": 13},
        xaxis={"title": "missing hours, %", "gridcolor": GRID, "range": [0, ordered["missing_pct"].max() * 1.2]},
    )
    return fig


def quality_tab():
    tables = quality_tables()
    shift_key = next(k for k in tables if k.startswith("Dynamometer"))
    regime_key = next(k for k in tables if "regime" in k)
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("Missing hours by well")
        st.plotly_chart(missing_chart(tables["Missing hours by well"]), width="stretch",
                        config={"displaylogo": False})
    with right:
        st.subheader("Longest gaps")
        st.dataframe(tables["Hourly gaps"].head(10), hide_index=True, width="stretch")
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader(shift_key)
        st.dataframe(rounded(tables[shift_key], 1), hide_index=True, width="stretch",
                     column_config={"ratio": st.column_config.NumberColumn("ratio, ×", format="%.1f")})
    with right:
        st.subheader(regime_key)
        st.dataframe(rounded(tables[regime_key]), hide_index=True, width="stretch")
    st.subheader("Quality flags by well")
    st.dataframe(tables["Flag rates by well"], hide_index=True, width="stretch")


def exploration_tab():
    tables, figures = exploration()
    st.subheader("Descriptive statistics (flagged readings removed)")
    st.dataframe(tables["Descriptive statistics"].round(2), hide_index=True, width="stretch")
    images = [path for path in figures if path.suffix == ".png"]
    for left_path, right_path in zip(images[::2], images[1::2]):
        left, right = st.columns(2, gap="large")
        left.image(str(left_path), width="stretch")
        right.image(str(right_path), width="stretch")
    if len(images) % 2:
        st.image(str(images[-1]), width="stretch")


def main():
    st.set_page_config(page_title="NK Well Anomaly Monitor", page_icon=":material/oil_barrel:", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    try:
        long_df, wide = load_frames()
    except (OSError, ConnectionError) as exc:
        st.error(f"Could not load the dataset: {exc}. Check the internet connection and reload the page.")
        return
    first, last = wide["timestamp"].min(), wide["timestamp"].max()
    st.title("NK field sensor anomaly monitor")
    st.markdown(
        f'<p class="subtitle">{wide["well_id"].nunique()} wells · {len(long_df):,} hourly readings · '
        f'{first:%d %b} – {last:%d %b %Y}</p>',
        unsafe_allow_html=True,
    )
    well, parameter, method, period, show_flags = sidebar_filters(wide)
    tabs = st.tabs(["Anomalies", "Method comparison", "Data quality", "Exploration"])
    with tabs[0]:
        anomalies_tab(wide, units_by_parameter(long_df), well, parameter, method, period, show_flags)
    with tabs[1]:
        comparison_tab()
    with tabs[2]:
        quality_tab()
    with tabs[3]:
        exploration_tab()


main()
