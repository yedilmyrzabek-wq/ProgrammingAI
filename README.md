# Oil Well Sensor Anomaly Detection

Final project for *Programming for Data Analysis and Database* (Applied AI, AAI-2604M), topic 22.

Detects anomalous hourly sensor readings of 10 oil wells and compares three detectors against a baseline:

| Method | Idea |
| --- | --- |
| Baseline | Fixed percentile thresholds learned on train |
| Z-score | Distance from the train mean in standard deviations |
| Rolling IQR | Tukey fences over a sliding window |
| IsolationForest | Tree-based isolation of multivariate outliers |

Quality flags from the dataset are used as a partial reference for precision / recall.

## Dataset

[NK Oil Well Sensor Monitoring v2](https://huggingface.co/datasets/Arailym-tleubayeva/NK-Oil-Well-Sensor-Monitoring-v2) —
258,387 hourly rows, 10 wells, Jan–Sep 2026, chronological train/test split. Data is downloaded on first run and is not stored in the repository.

## Setup

```bash
pip install -r requirements.txt
python main.py
streamlit run app.py
```

## Team

- Yedil Myrzabek
- Shyntassov Abdurahmans

## Results

Run `python main.py` and select **Compare anomaly detectors** to regenerate the
[comparison table](reports/comparison.csv) and [chart](reports/figures/comparison.png).
The menu also lets you prepare Parquet and SQLite, audit data quality, run
the saved SQL queries, and generate exploratory figures. Select **Inspect
detected anomalies** to choose a well and a detector and list test timestamps
that the detector marks as anomalous. Baseline, Z-score, and Rolling IQR also
prompt for a sensor.
The Isolation Forest view shows the quality flag for the entire well-hour.

The comparison uses the dataset's chronological split. The baseline uses each
well's 1st and 99th training percentiles, and Z-score uses a three-standard-deviation
threshold fitted on train. Rolling IQR uses the preceding 48 hours, excluding
the current reading. Isolation Forest fits separately for each well with 2%
contamination and uses training medians to fill missing sensor values.

All four methods were scored on 12,509 test well-hours, 1,323 of which had at least
one quality flag. Z-score had the highest F1 (0.198), close to the baseline
(0.197). Isolation Forest had the highest precision (0.310) and the lowest recall
(0.098). These scores measure agreement with incomplete quality flags. Runtime
includes fitting and prediction. Peak traced MiB covers allocations tracked by
`tracemalloc` during that run.

In the quality audit, the median of unflagged dynamometer area readings rose
14–20 times from April to May across the seven pump wells. At NK-7, median SPM
fell from 4.16 to 1.40 after the June cutoff, while the median rod weights rose.
