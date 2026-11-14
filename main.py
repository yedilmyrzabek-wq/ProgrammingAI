import sqlite3
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

from src import config, data_loader, database, evaluation, quality
from src.detectors import METHODS, predict


def prepare_data():
    if config.WIDE_PARQUET.exists():
        long_df, wide = data_loader.load_long(), data_loader.load_wide()
    else:
        long_df, wide = data_loader.build_datasets()
    ensure_database(long_df)
    print(f"Ready: {len(long_df):,} readings and {len(wide):,} well-hours.")


def ensure_database(long_df):
    if config.DB_PATH.exists():
        return
    config.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=config.DB_PATH.parent) as directory:
        target = Path(directory) / config.DB_PATH.name
        with closing(database.connect(target)) as conn:
            database.create_schema(conn)
            database.load_tables(conn, long_df)
        target.replace(config.DB_PATH)


def print_table(frame):
    if frame.empty:
        print("No matching records.")
        return
    print(frame.head(20).to_string(index=False, float_format="{:.3f}".format))
    if len(frame) > 20:
        print(f"Showing 20 of {len(frame):,} rows.")


def run_audit():
    long_df = data_loader.load_long()
    ensure_database(long_df)
    for title, frame in quality.audit_quality(long_df).items():
        print(f"\n{title}")
        print_table(frame)


def run_comparison():
    comparison = evaluation.compare_methods(data_loader.load_wide())
    evaluation.save_comparison(comparison)
    print_table(comparison)
    print(f"Reports saved in {config.REPORTS_DIR}")


def run_sql():
    names = list(database.load_queries())
    name = select_option("Query number", names)
    if name is None:
        return
    if not config.DB_PATH.exists():
        ensure_database(data_loader.load_long())
    with closing(database.connect()) as conn:
        print_table(database.run_query(name, conn))


def run_eda():
    from src import eda

    tables, figures = eda.run_eda(data_loader.load_wide())
    for title, frame in tables.items():
        print(f"\n{title}")
        print_table(frame)
    print("\nSaved figures:")
    for path in figures:
        print(path)


def select_option(label, options):
    for index, option in enumerate(options, start=1):
        print(f"{index}. {option}")
    number = int(input(f"{label} (0 to return): ").strip())
    if not 0 <= number <= len(options):
        raise ValueError("Choose a number from the list.")
    return options[number - 1] if number else None


def inspect_anomalies():
    wide = data_loader.load_wide()
    wells = sorted(wide["well_id"].dropna().unique())
    well_id = select_option("Well number", wells)
    if well_id is None:
        return
    method = select_option("Detector number", METHODS)
    if method is None:
        return
    well = wide.loc[wide["well_id"].eq(well_id)]
    predictions = predict(well, method)
    if method == "Isolation Forest":
        columns = ["timestamp", "any_flag"]
        anomalies = predictions["anomaly"]
    else:
        sensors = [name for name in predictions if well[name].notna().any()]
        sensor = select_option("Sensor number", sensors)
        if sensor is None:
            return
        columns = ["timestamp", sensor, f"flag_{sensor}"]
        anomalies = predictions[sensor]
    print(f"\n{well_id} | {method} | {int(anomalies.sum()):,} anomalies")
    print_table(well.loc[anomalies.index[anomalies.fillna(False)], columns])


def main():
    actions = {
        "1": ("Prepare datasets and SQLite", prepare_data),
        "2": ("Audit data quality", run_audit),
        "3": ("Compare anomaly detectors", run_comparison),
        "4": ("Run a saved SQL query", run_sql),
        "5": ("Explore sensor data and generate figures", run_eda),
        "6": ("Inspect detected anomalies", inspect_anomalies),
    }
    while True:
        try:
            print("\nOil Well Sensor Anomaly Detection")
            for key, (label, _) in actions.items():
                print(f"{key}. {label}")
            print("0. Exit")
            choice = input("Choose an option: ").strip()
            if choice == "0":
                return
            if choice not in actions:
                print("Choose an option from the menu.")
                continue
            actions[choice][1]()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        except (OSError, ValueError, sqlite3.Error) as exc:
            print(f"Operation failed: {exc}")


if __name__ == "__main__":
    main()
