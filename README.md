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
