import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from src import config
from src.detectors import METHODS, hourly_anomalies, predict
from src.utils import measure, save_json


def _run_detector(wide, method):
    return hourly_anomalies(predict(wide, method))


def compare_methods(wide):
    """Compare methods against any_flag on the same scored test well-hours."""
    reference = wide.loc[wide["split"].eq("test"), "any_flag"].astype("boolean")
    runs = {method: measure(_run_detector, wide, method) for method in METHODS}
    comparable = reference.notna()
    for predictions, _, _ in runs.values():
        comparable &= predictions.notna()
    if not comparable.any():
        raise ValueError("No test hours can be scored by every detector.")
    truth = reference[comparable].astype(bool)
    rows = []
    for method, (predictions, seconds, peak_mib) in runs.items():
        scored = predictions[comparable].astype(bool)
        precision, recall, f1, _ = precision_recall_fscore_support(
            truth, scored, average="binary", zero_division=0
        )
        rows.append({
            "method": method, "precision": precision, "recall": recall, "f1": f1,
            "seconds": seconds, "peak_traced_mib": peak_mib,
            "scored_hours": len(truth), "flagged_hours": int(truth.sum()),
            "predicted_anomalies": int(scored.sum()),
        })
    return pd.DataFrame(rows)


def save_comparison(comparison, directory=config.REPORTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(directory / "comparison.csv", index=False)
    save_json(comparison.to_dict("records"), directory / "comparison.json")

    figures = directory / config.FIGURES_DIR.name
    figures.mkdir(parents=True, exist_ok=True)
    ax = comparison.set_index("method")[["precision", "recall", "f1"]].plot.bar(
        figsize=(8, 4), rot=0, ylim=(0, 1), xlabel="", ylabel="Score"
    )
    ax.set_title("Detector agreement with test quality flags")
    figure = ax.get_figure()
    try:
        figure.savefig(figures / "comparison.png", dpi=150, bbox_inches="tight")
    finally:
        plt.close(figure)
