import time
import urllib.request
from urllib.error import URLError

import pandas as pd

from src import config

KEYS = ["well_id", "well_type", "timestamp", "split"]
CSV_DTYPES = {
    "well_id": "category",
    "well_type": "category",
    "parameter": "category",
    "unit": "category",
    "value": "float32",
    "n_obs": "int16",
    "flag": "object",
}


def download_split(split, force=False):
    config.ensure_dirs()
    target = config.RAW_DIR / f"{split}.csv"
    if target.exists() and not force:
        return target
    url = f"{config.HF_BASE_URL}/{split}.csv"
    try:
        urllib.request.urlretrieve(url, target)
    except URLError as exc:
        target.unlink(missing_ok=True)
        raise ConnectionError(f"Could not download {url}: {exc.reason}") from exc
    return target


def read_split(path, split, chunksize=50_000):
    chunks = []
    reader = pd.read_csv(path, dtype=CSV_DTYPES, parse_dates=["timestamp"], chunksize=chunksize)
    for chunk in reader:
        chunks.append(chunk)
    frame = pd.concat(chunks, ignore_index=True)
    frame["split"] = split
    return frame


def optimize_dtypes(frame):
    frame = frame.copy()
    for column in ("well_id", "well_type", "parameter", "unit", "split"):
        frame[column] = frame[column].astype("category")
    frame["flag"] = frame["flag"].fillna("").astype("category")
    return frame


def load_long(force=False):
    if config.LONG_PARQUET.exists() and not force:
        return pd.read_parquet(config.LONG_PARQUET)
    parts = [read_split(download_split(split, force), split) for split in config.SPLITS]
    long_df = optimize_dtypes(pd.concat(parts, ignore_index=True))
    long_df = long_df.sort_values(["well_id", "timestamp", "parameter"], ignore_index=True)
    long_df.to_parquet(config.LONG_PARQUET, index=False)
    return long_df


def to_wide(long_df):
    long_df = long_df.assign(is_flagged=long_df["flag"].astype(str) != "")
    values = long_df.pivot_table(
        index=KEYS, columns="parameter", values="value", aggfunc="first", observed=True
    )
    flags = long_df.pivot_table(
        index=KEYS, columns="parameter", values="is_flagged", aggfunc="max", observed=True
    )
    flags = flags.fillna(False).astype(bool).add_prefix("flag_")
    wide = values.join(flags)
    wide["any_flag"] = wide[flags.columns].any(axis=1)
    wide = wide.reset_index().sort_values(["well_id", "timestamp"], ignore_index=True)
    wide.columns.name = None
    return wide


def build_datasets(force=False):
    long_df = load_long(force)
    wide_df = to_wide(long_df)
    wide_df.to_parquet(config.WIDE_PARQUET, index=False)
    return long_df, wide_df


def load_wide():
    if not config.WIDE_PARQUET.exists():
        return build_datasets()[1]
    return pd.read_parquet(config.WIDE_PARQUET)


def memory_mb(frame):
    return frame.memory_usage(deep=True).sum() / 1024**2


def compare_storage_formats(long_df):
    results = []
    raw = long_df.astype({"flag": str, "well_id": str, "parameter": str})
    targets = {
        "csv": (config.DATA_DIR / "bench.csv", raw.to_csv, pd.read_csv),
        "parquet": (config.DATA_DIR / "bench.parquet", long_df.to_parquet, pd.read_parquet),
    }
    for name, (path, writer, reader) in targets.items():
        start = time.perf_counter()
        writer(path, index=False)
        write_s = time.perf_counter() - start
        start = time.perf_counter()
        loaded = reader(path)
        read_s = time.perf_counter() - start
        results.append({
            "format": name,
            "size_mb": round(path.stat().st_size / 1024**2, 2),
            "write_s": round(write_s, 3),
            "read_s": round(read_s, 3),
            "memory_mb": round(memory_mb(loaded), 2),
        })
        path.unlink()
    return pd.DataFrame(results)
