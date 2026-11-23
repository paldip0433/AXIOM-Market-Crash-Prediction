# AXIOM — novelty assessment: 02_warning_evaluation.py
# Extracted verbatim from notebook code cells 71–76 (zero-based).
# These scripts retain notebook-global dependencies and should be run in order
# within the same initialized Colab/Python session (e.g. exec(open(path).read())).
# Original AXIOM preprocessing and model modules must already be available.
# This is an archival extraction, not a standalone refactor.


# ---------- Original notebook cell 71 ----------
import numpy as np
import pandas as pd

HORIZON = 30
ALERT_BUDGET = 0.10

price_series = (
    prices.set_index("Date")["SP500"]
    .sort_index()
)

all_dates = price_series.index
all_values = price_series.to_numpy(dtype=float)
date_positions = pd.Series(
    np.arange(len(all_dates)),
    index=all_dates
)

def choose_threshold(validation, budget=ALERT_BUDGET):
    probabilities = validation["probability"].to_numpy()
    return float(np.quantile(probabilities, 1 - budget))


def first_decline_lead(date):
    """Trading days to the first 10% decline from this date's close."""
    if date not in date_positions.index:
        return np.nan

    i = int(date_positions.loc[date])
    current = all_values[i]

    for h in range(1, HORIZON + 1):
        j = i + h

        if j >= len(all_values):
            break

        if all_values[j] <= 0.90 * current:
            return h

    return np.nan


def evaluate_warnings(predictions, threshold):

    pred = predictions.sort_values("Date").copy()
    pred["alert"] = pred["probability"] >= threshold

    # A new episode begins whenever the alert switches
    # from False to True.
    pred["alert_start"] = (
        pred["alert"]
        & ~pred["alert"].shift(fill_value=False)
    )

    pred["alert_episode"] = pred["alert_start"].cumsum()

    alert_days = pred[pred["alert"]].copy()

    episodes = []

    for episode_id, group in alert_days.groupby("alert_episode"):

        positive_alerts = group[group["label"] == 1]

        detected = len(positive_alerts) > 0

        if detected:
            first_positive_date = positive_alerts.iloc[0]["Date"]
            lead = first_decline_lead(first_positive_date)
        else:
            lead = np.nan

        episodes.append({
            "episode_id": episode_id,
            "start": group["Date"].min(),
            "end": group["Date"].max(),
            "alert_days": len(group),
            "detected": detected,
            "lead_days": lead
        })

    episode_df = pd.DataFrame(
        episodes,
        columns=[
            "episode_id", "start", "end",
            "alert_days", "detected", "lead_days"
        ]
    )

    false_episodes = int(
        (~episode_df["detected"]).sum()
    )

    # Positive-label episode detection
    label_starts = (
        (pred["label"] == 1)
        & (pred["label"].shift(fill_value=0) == 0)
    )

    pred["label_episode"] = label_starts.cumsum()

    positive = pred[pred["label"] == 1]

    detected_label_episodes = (
        positive.groupby("label_episode")["alert"]
        .any()
    )

    return {
        "threshold": threshold,
        "alert_days": int(pred["alert"].sum()),
        "alert_episodes": len(episode_df),
        "false_alert_episodes": false_episodes,
        "correct_alert_episodes": int(
            episode_df["detected"].sum()
        ),
        "label_episodes": len(detected_label_episodes),
        "detected_label_episodes": int(
            detected_label_episodes.sum()
        ),
        "label_episode_recall": (
            detected_label_episodes.mean()
            if len(detected_label_episodes) else np.nan
        ),
        "mean_lead_days": episode_df["lead_days"].mean(),
        "median_lead_days": episode_df["lead_days"].median()
    }, episode_df

# ---------- Original notebook cell 72 ----------
from pathlib import Path

warning_rows = []
all_episodes = []

for _, item in run_manifest.iterrows():

    run_dir = Path(item["metrics_path"]).parent

    validation = pd.read_csv(
        run_dir / "validation_predictions.csv",
        parse_dates=["Date"]
    )

    test = pd.read_csv(
        run_dir / "test_predictions.csv",
        parse_dates=["Date"]
    )

    threshold = choose_threshold(validation)

    metrics, episode_df = evaluate_warnings(
        test, threshold
    )

    warning_rows.append({
        "feature_group": item["feature_group"],
        "seed": item["seed"],
        **metrics
    })

    episode_df["feature_group"] = item["feature_group"]
    episode_df["seed"] = item["seed"]

    all_episodes.append(episode_df)

warning_results = pd.DataFrame(warning_rows)
warning_episodes = pd.concat(
    all_episodes, ignore_index=True
)

warning_results.to_csv(
    NOVELTY / "results" / "experiment2_warning_metrics.csv",
    index=False
)

warning_episodes.to_csv(
    NOVELTY / "results" / "experiment2_alert_episodes.csv",
    index=False
)

display(warning_results.round(3))

# ---------- Original notebook cell 73 ----------
warning_summary = (
    warning_results
    .groupby("feature_group")
    .agg(
        false_alerts_mean=("false_alert_episodes", "mean"),
        false_alerts_std=("false_alert_episodes", "std"),
        detection_mean=("label_episode_recall", "mean"),
        detection_std=("label_episode_recall", "std"),
        lead_mean=("mean_lead_days", "mean"),
        lead_std=("mean_lead_days", "std"),
        alert_days_mean=("alert_days", "mean")
    )
    .reset_index()
)

warning_summary.to_csv(
    NOVELTY / "results" / "experiment2_warning_summary.csv",
    index=False
)

display(warning_summary.round(3))

# ---------- Original notebook cell 74 ----------
import matplotlib.pyplot as plt

labels = {
    "classical": "Classical",
    "combined": "Combined",
    "combined_without_h1_change": "Combined without H1",
    "classical_plus_h1_change": "Classical + H1"
}

plot_data = warning_summary.set_index("feature_group").loc[
    list(labels.keys())
]

for metric, ylabel, filename in [
    ("lead_mean", "Mean lead time (trading days)",
     "experiment2_lead_time.png"),
    ("false_alerts_mean", "False-alert episodes (test period)",
     "experiment2_false_alerts.png")
]:

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(
        list(labels.values()),
        plot_data[metric].to_numpy(),
        color=["#64748b", "#2563eb", "#8b5cf6", "#0891b2"]
    )

    ax.set_ylabel(ylabel)
    ax.set_title("AXIOM Novelty Assessment")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    plt.tight_layout()

    path = NOVELTY / "figures" / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved:", path)

# ---------- Original notebook cell 75 ----------
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score

TARGET_FPR = 0.05

def threshold_at_fpr(validation, target_fpr=TARGET_FPR):
    negative_scores = validation.loc[
        validation["label"] == 0, "probability"
    ].to_numpy()

    if len(negative_scores) == 0:
        raise ValueError("No negative validation observations.")

    # Strictly exceed the chosen negative-score cutoff.
    cutoff = np.quantile(
        negative_scores, 1 - target_fpr
    )

    return float(np.nextafter(cutoff, np.inf))


matched_rows = []

for _, item in run_manifest.iterrows():

    folder = Path(item["metrics_path"]).parent

    val = pd.read_csv(
        folder / "validation_predictions.csv",
        parse_dates=["Date"]
    )

    test = pd.read_csv(
        folder / "test_predictions.csv",
        parse_dates=["Date"]
    )

    threshold = threshold_at_fpr(val)

    metrics, _ = evaluate_warnings(test, threshold)

    negative = test["label"] == 0
    positive = test["label"] == 1
    alerts = test["probability"] >= threshold

    matched_rows.append({
        "feature_group": item["feature_group"],
        "seed": int(item["seed"]),
        "threshold": threshold,
        "test_AP": average_precision_score(
            test["label"], test["probability"]
        ),
        "test_FPR": float(alerts[negative].mean()),
        "test_TPR": float(alerts[positive].mean()),
        **metrics
    })

matched_results = pd.DataFrame(matched_rows)

matched_results.to_csv(
    NOVELTY / "results" / "experiment2_matched_fpr_runs.csv",
    index=False
)

display(matched_results.round(4))

# ---------- Original notebook cell 76 ----------
matched_summary = (
    matched_results
    .groupby("feature_group")
    .agg(
        AP_mean=("test_AP", "mean"),
        FPR_mean=("test_FPR", "mean"),
        TPR_mean=("test_TPR", "mean"),
        false_alerts_mean=("false_alert_episodes", "mean"),
        detection_mean=("label_episode_recall", "mean"),
        lead_mean=("mean_lead_days", "mean"),
        alert_days_mean=("alert_days", "mean")
    )
    .reset_index()
)

matched_summary.to_csv(
    NOVELTY / "results" / "experiment2_matched_fpr_summary.csv",
    index=False
)

display(matched_summary.round(4))
