# AXIOM — novelty assessment: 01_h1_feature_ablation.py
# Extracted verbatim from notebook code cells 59–70 (zero-based).
# These scripts retain notebook-global dependencies and should be run in order
# within the same initialized Colab/Python session (e.g. exec(open(path).read())).
# Original AXIOM preprocessing and model modules must already be available.
# This is an archival extraction, not a standalone refactor.


# ---------- Original notebook cell 59 ----------
from pathlib import Path

experiment_root = NOVELTY / 'experiments'

for path in sorted(experiment_root.rglob('*')):
    if path.is_file():
        print(path.relative_to(NOVELTY))

# ---------- Original notebook cell 60 ----------
seeds = [42, 123, 2026]

completed = {
    ('classical', 42)
}

for group_name, columns in feature_groups.items():

    for seed in seeds:

        if (group_name, seed) in completed:
            continue

        print(
            f"\nTraining: {group_name} | seed={seed}",
            flush=True
        )

        result = dl.train_one(
            df=df,
            feature_set=group_name,
            columns=columns,
            architecture='LSTM',
            seed=seed,
            device=device,
            epochs=25
        )

        print(
            f"Completed: {group_name} | seed={seed}",
            flush=True
        )

print("\nAll scheduled experiments completed.")

# ---------- Original notebook cell 61 ----------
import pandas as pd

sample = (
    NOVELTY
    / "experiments"
    / "combined_lstm"
    / "run_004"
    / "metrics.csv"
)

metrics_sample = pd.read_csv(sample)

print("Columns:")
print(metrics_sample.columns.tolist())

print("\nContents:")
display(metrics_sample)

# ---------- Original notebook cell 62 ----------
from pathlib import Path
import pandas as pd
import json

RUN_TO_SEED = {
    4: 42,
    5: 123,
    6: 2026
}

records = []

for group_name in feature_groups:

    experiment_dir = (
        NOVELTY
        / "experiments"
        / f"{group_name}_lstm"
    )

    for run_number, seed in RUN_TO_SEED.items():

        run_dir = (
            experiment_dir
            / f"run_{run_number:03d}"
        )

        metrics_path = run_dir / "metrics.csv"
        settings_path = run_dir / "settings.json"

        if not metrics_path.exists():
            raise FileNotFoundError(metrics_path)

        metrics = pd.read_csv(metrics_path)

        records.append({
            "feature_group": group_name,
            "seed": seed,
            "run": run_number,
            "metrics_path": str(metrics_path),
            "settings_path": str(settings_path)
        })

run_manifest = pd.DataFrame(records)

display(run_manifest)

run_manifest.to_csv(
    NOVELTY / "results" / "experiment1_run_manifest.csv",
    index=False
)

# ---------- Original notebook cell 63 ----------
import pandas as pd
import numpy as np
import json

records = []

for _, item in run_manifest.iterrows():

    metrics = pd.read_csv(item["metrics_path"])

    with open(item["settings_path"], "r") as f:
        settings = json.load(f)

    # Check that the saved seed agrees with the run manifest.
    saved_seed = settings.get("seed", None)

    if saved_seed is not None:
        assert int(saved_seed) == int(item["seed"]), (
            f"Seed mismatch: {item['settings_path']}"
        )

    for _, row in metrics.iterrows():

        records.append({
            "feature_group": item["feature_group"],
            "seed": int(item["seed"]),
            "run": int(item["run"]),
            "split": row["split"],
            "best_epoch": row["best_epoch"],
            "n": row["n"],
            "positives": row["positives"],
            "AP": row["PR_AUC"],
            "ROC_AUC": row["ROC_AUC"],
            "precision_05": row["precision_05"],
            "recall_05": row["recall_05"],
            "F1_05": row["F1_05"]
        })

results = pd.DataFrame(records)

assert len(results) == 24, (
    "Expected 12 runs × 2 splits = 24 rows."
)

assert not results.duplicated(
    ["feature_group", "seed", "split"]
).any()

results.to_csv(
    NOVELTY / "results" / "experiment1_all_metrics.csv",
    index=False
)

summary = (
    results
    .groupby(["feature_group", "split"])
    .agg(
        AP_mean=("AP", "mean"),
        AP_std=("AP", "std"),
        ROC_mean=("ROC_AUC", "mean"),
        ROC_std=("ROC_AUC", "std"),
        precision_mean=("precision_05", "mean"),
        recall_mean=("recall_05", "mean"),
        F1_mean=("F1_05", "mean")
    )
    .reset_index()
)

summary.to_csv(
    NOVELTY / "results" / "experiment1_summary.csv",
    index=False
)

display(summary.round(4))

# ---------- Original notebook cell 64 ----------
import matplotlib.pyplot as plt

labels = {
    "classical": "Classical",
    "combined": "Combined",
    "combined_without_h1_change": "Combined without H1 change",
    "classical_plus_h1_change": "Classical + H1 change"
}

order = list(labels.keys())

for split in ["validation", "test"]:

    part = (
        summary[summary["split"] == split]
        .set_index("feature_group")
        .loc[order]
    )

    means = part["AP_mean"].to_numpy()
    stds = part["AP_std"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(
        range(len(order)),
        means,
        yerr=stds,
        capsize=5,
        color=["#64748b", "#2563eb", "#8b5cf6", "#0891b2"]
    )

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(
        [labels[x] for x in order],
        rotation=12,
        ha="right"
    )

    ax.set_ylabel("Average Precision")
    ax.set_title(
        f"AXIOM Novelty Assessment — {split.title()} AP"
    )

    ax.set_ylim(
        0,
        max(0.05, float(np.max(means + stds)) * 1.25)
    )

    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, means):
        ax.annotate(
            f"{value:.4f}",
            (bar.get_x() + bar.get_width()/2, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9
        )

    plt.tight_layout()

    output = (
        NOVELTY / "figures"
        / f"experiment1_{split}_ap.png"
    )

    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved:", output)

# ---------- Original notebook cell 65 ----------
test = results[results["split"] == "test"]

pivot = test.pivot(
    index="seed",
    columns="feature_group",
    values="AP"
)

pivot["H1_change_contribution"] = (
    pivot["combined"]
    - pivot["combined_without_h1_change"]
)

pivot["H1_added_to_classical"] = (
    pivot["classical_plus_h1_change"]
    - pivot["classical"]
)

display(pivot.round(4))

print("\nMean AP difference: Combined minus No H1 change")
print(pivot["H1_change_contribution"].mean())

print("\nMean AP difference: Classical + H1 minus Classical")
print(pivot["H1_added_to_classical"].mean())

pivot.to_csv(
    NOVELTY / "results" / "experiment1_h1_ablation.csv"
)

# ---------- Original notebook cell 66 ----------
sample_dir = (
    NOVELTY
    / "experiments"
    / "combined_lstm"
    / "run_004"
)

for filename in [
    "validation_predictions.csv",
    "test_predictions.csv"
]:
    path = sample_dir / filename
    predictions = pd.read_csv(path)

    print("\n", filename)
    print("Shape:", predictions.shape)
    print("Columns:", predictions.columns.tolist())

    display(predictions.head(5))

# ---------- Original notebook cell 67 ----------
from pathlib import Path
import pandas as pd

data_root = PROJECT / "data"

for path in sorted(data_root.rglob("*.csv")):
    print("\nFILE:", path.relative_to(PROJECT))
    try:
        sample = pd.read_csv(path, nrows=3)
        print("Columns:", sample.columns.tolist())
        print(sample.head(2).to_string(index=False))
    except Exception as exc:
        print("Could not preview:", exc)

# ---------- Original notebook cell 68 ----------
import numpy as np
import pandas as pd

rows = []

for _, item in run_manifest.iterrows():

    run_dir = Path(item["metrics_path"]).parent

    for split in ["validation", "test"]:

        pred = pd.read_csv(
            run_dir / f"{split}_predictions.csv",
            parse_dates=["Date"]
        ).sort_values("Date")

        rows.append({
            "feature_group": item["feature_group"],
            "seed": item["seed"],
            "split": split,
            "start": pred["Date"].min().date(),
            "end": pred["Date"].max().date(),
            "n": len(pred),
            "positives": int(pred["label"].sum()),
            "mean_probability": pred["probability"].mean(),
            "median_probability": pred["probability"].median(),
            "predicted_positive_at_05": int(
                (pred["probability"] >= 0.5).sum()
            )
        })

probability_summary = pd.DataFrame(rows)

display(
    probability_summary.sort_values(
        ["feature_group", "seed", "split"]
    )
)

probability_summary.to_csv(
    NOVELTY / "results" / "prediction_diagnostics.csv",
    index=False
)

# ---------- Original notebook cell 69 ----------
import numpy as np
import pandas as pd

prices = pd.read_csv(
    PROJECT / "data/processed/prices_aligned.csv",
    parse_dates=["Date"]
).sort_values("Date").reset_index(drop=True)

H = 30
P = prices["SP500"].to_numpy(dtype=float)

future_prices = np.column_stack([
    np.roll(P, -h) for h in range(1, H + 1)
])

# Future observations are unavailable near the dataset end.
future_prices[-H:, :] = np.nan

future_min = np.full(len(P), np.nan)
future_min[:-H] = np.min(future_prices[:-H], axis=1)

prices["reconstructed_label"] = np.where(
    np.isfinite(future_min),
    (future_min <= 0.9 * P).astype(float),
    np.nan
)

sample_predictions = pd.read_csv(
    NOVELTY / "experiments" / "combined_lstm"
    / "run_004" / "test_predictions.csv",
    parse_dates=["Date"]
)

audit = sample_predictions.merge(
    prices[["Date", "SP500", "reconstructed_label"]],
    on="Date",
    how="left",
    validate="one_to_one"
)

valid = audit["reconstructed_label"].notna()

print("Matched observations:", valid.sum())
print("Missing price/label observations:", (~valid).sum())

mismatches = (
    audit.loc[valid, "label"].astype(int).to_numpy()
    != audit.loc[valid, "reconstructed_label"].astype(int).to_numpy()
)

print("Label mismatches:", mismatches.sum())

display(audit.head())

# ---------- Original notebook cell 70 ----------
labels = audit[["Date", "label"]].copy()
labels = labels.sort_values("Date").reset_index(drop=True)

labels["episode_start"] = (
    (labels["label"] == 1)
    & (labels["label"].shift(fill_value=0) == 0)
)

labels["episode_id"] = (
    labels["episode_start"].cumsum()
)

positive_days = labels[labels["label"] == 1].copy()

episodes = (
    positive_days
    .groupby("episode_id")
    .agg(
        start=("Date", "min"),
        end=("Date", "max"),
        positive_days=("Date", "size")
    )
    .reset_index()
)

print("Number of positive-label episodes:", len(episodes))
display(episodes)
