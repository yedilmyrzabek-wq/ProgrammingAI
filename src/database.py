import re
import sqlite3
import time

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from src import config

SCHEMA_PATH = config.SQL_DIR / "schema.sql"
QUERIES_PATH = config.SQL_DIR / "queries.sql"
ER_PATH = config.DOCS_DIR / "er_diagram.png"


def connect(path=config.DB_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(path)


def create_schema(conn):
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))


def split_tables(long_df):
    wells = long_df[["well_id", "well_type"]].drop_duplicates().astype(str)
    parameters = long_df[["parameter", "unit", "well_type"]].drop_duplicates("parameter").astype(str)
    readings = long_df[["well_id", "parameter", "timestamp", "value", "n_obs", "flag", "split"]].copy()
    for column in ("well_id", "parameter", "flag", "split"):
        readings[column] = readings[column].astype(str)
    readings["timestamp"] = readings["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return {"wells": wells, "parameters": parameters, "readings": readings}


def load_tables(conn, long_df, chunksize=50_000):
    for name, table in split_tables(long_df).items():
        table.to_sql(name, conn, if_exists="append", index=False, chunksize=chunksize)
    conn.commit()


def build_database(long_df, path=config.DB_PATH):
    with connect(path) as conn:
        create_schema(conn)
        load_tables(conn, long_df)
    return path


def load_queries(path=QUERIES_PATH):
    text = path.read_text(encoding="utf-8")
    blocks = re.split(r"^-- name: (\w+)\s*$", text, flags=re.MULTILINE)
    return {name: sql.strip() for name, sql in zip(blocks[1::2], blocks[2::2])}


def run_query(name, conn=None):
    queries = load_queries()
    if name not in queries:
        raise KeyError(f"Unknown query '{name}'. Available: {', '.join(queries)}")
    own_conn = conn is None
    conn = conn or connect()
    try:
        return pd.read_sql_query(queries[name], conn)
    finally:
        if own_conn:
            conn.close()


def run_all_queries():
    with connect() as conn:
        return {name: run_query(name, conn) for name in load_queries()}


def compare_pandas_sql(long_df):
    results = []
    with connect() as conn:
        start = time.perf_counter()
        pd.read_sql_query(load_queries()["parameter_stats_by_type"], conn)
        results.append({"engine": "sqlite", "seconds": round(time.perf_counter() - start, 4)})
    start = time.perf_counter()
    long_df.groupby(["well_type", "parameter"], observed=True)["value"].agg(["count", "mean", "min", "max"])
    results.append({"engine": "pandas", "seconds": round(time.perf_counter() - start, 4)})
    return pd.DataFrame(results)


def _table_box(ax, x, y, title, fields):
    height = 0.6 + 0.42 * len(fields)
    ax.add_patch(FancyBboxPatch((x, y - height), 3.4, height, boxstyle="round,pad=0.05",
                                fc="#f4f6fa", ec="#33415c", lw=1.4))
    ax.text(x + 1.7, y - 0.35, title, ha="center", va="center", fontsize=12, weight="bold")
    ax.plot([x, x + 3.4], [y - 0.65, y - 0.65], color="#33415c", lw=1)
    for i, field in enumerate(fields):
        ax.text(x + 0.15, y - 0.95 - 0.42 * i, field, va="center", fontsize=9.5, family="monospace")
    return height


def draw_er_diagram(path=ER_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    _table_box(ax, 0.2, 4.6, "wells", ["PK well_id    TEXT", "   well_type  TEXT"])
    _table_box(ax, 6.4, 4.6, "parameters", ["PK parameter  TEXT", "   unit       TEXT", "   well_type  TEXT"])
    _table_box(ax, 3.3, 2.2, "readings", [
        "PK reading_id INTEGER", "FK well_id    TEXT", "FK parameter  TEXT", "   timestamp  TEXT",
        "   value      REAL", "   n_obs      INTEGER", "   flag       TEXT", "   split      TEXT",
    ])
    arrow = dict(arrowstyle="-|>", color="#33415c", lw=1.2)
    ax.annotate("", xy=(1.9, 3.08), xytext=(3.25, 0.83), arrowprops=arrow)
    ax.annotate("", xy=(8.1, 2.66), xytext=(6.75, 0.41), arrowprops=arrow)
    ax.text(1.95, 1.9, "1 : N", fontsize=9)
    ax.text(7.65, 1.4, "1 : N", fontsize=9)
    ax.set_xlim(0, 10)
    ax.set_ylim(-2, 5)
    ax.axis("off")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
