# AXIOM — novelty assessment: 03_enhanced_h1_features.py
# Extracted verbatim from notebook code cells 77–86 (zero-based).
# These scripts retain notebook-global dependencies and should be run in order
# within the same initialized Colab/Python session (e.g. exec(open(path).read())).
# Original AXIOM preprocessing and model modules must already be available.
# This is an archival extraction, not a standalone refactor.


# ---------- Original notebook cell 77 ----------
import numpy as np
import pandas as pd

enhanced_df = df.copy()

H1_WINDOWS = [40, 80, 120]
HISTORY = 30
EPS = 1e-8

new_h1_columns = []

for w in H1_WINDOWS:

    original = f"H1_w{w}_landscape_change"

    assert original in enhanced_df.columns, (
        f"Missing original feature: {original}"
    )

    d = enhanced_df[original].astype(float)

    # Use only preceding values to establish the baseline.
    previous = d.shift(1)

    historical_mean = previous.rolling(
        HISTORY, min_periods=HISTORY
    ).mean()

    historical_std = previous.rolling(
        HISTORY, min_periods=HISTORY
    ).std(ddof=0)

    z_name = f"H1_w{w}_change_zscore"
    a_name = f"H1_w{w}_change_acceleration"

    enhanced_df[z_name] = (
        (d - historical_mean) /
        (historical_std + EPS)
    )

    enhanced_df[a_name] = d.diff()

    new_h1_columns.extend([z_name, a_name])

print("New H1 features:")
print(new_h1_columns)

print("\nNon-missing observations:")
print(enhanced_df[new_h1_columns].notna().sum())

print("Index name:", enhanced_df.index.name)
print("Index type:", type(enhanced_df.index).__name__)

display(
    enhanced_df[new_h1_columns].tail()
)

# ---------- Original notebook cell 78 ----------
new_values = enhanced_df[new_h1_columns]

print("Number of new features:", len(new_h1_columns))

print("\nMissing values:")
print(new_values.isna().sum())

print("\nInfinite values:")
print(
    pd.Series(
        np.isinf(new_values.to_numpy()).sum(axis=0),
        index=new_h1_columns
    )
)

print("\nFeature summary:")
display(
    new_values.describe().T.round(4)
)
print("\nOriginal shape:", df.shape)
print("Enhanced shape:", enhanced_df.shape)
print("Same index:", enhanced_df.index.equals(df.index))

# ---------- Original notebook cell 79 ----------
import numpy as np
import pandas as pd

# Identify the new features
enhanced_columns = list(combined) + new_h1_columns

print("Original combined features:", len(combined))
print("Enhanced combined features:", len(enhanced_columns))

assert len(combined) == 48
assert len(enhanced_columns) == 54
assert len(set(enhanced_columns)) == 54

# Check where the new features are unavailable
missing_rows = enhanced_df[new_h1_columns].isna().any(axis=1)

print("\nRows containing missing enhanced features:",
      missing_rows.sum())

print("First date:", enhanced_df.index.min())
print("Last missing date:",
      enhanced_df.index[missing_rows].max())

print("\nMissing values after first 30 rows:")
print(
    enhanced_df.iloc[30:][new_h1_columns]
    .isna().sum()
)

# Check whether the original features are unchanged
print("\nOriginal combined features unchanged:",
      enhanced_df[combined].equals(df[combined]))

# ---------- Original notebook cell 80 ----------
enhanced_clean = enhanced_df.iloc[30:].copy()

assert enhanced_clean[new_h1_columns].notna().all().all()
assert np.isfinite(
    enhanced_clean[enhanced_columns].to_numpy(dtype=float)
).all()

print("Enhanced dataset shape:", enhanced_clean.shape)

print("First available date:", enhanced_clean.index.min())
print("Last available date:", enhanced_clean.index.max())

# ---------- Original notebook cell 81 ----------
import inspect

print(inspect.getsource(dl.train_one))

# ---------- Original notebook cell 82 ----------
import numpy as np

# Preserve the original dataset length and dates
enhanced_ready = enhanced_df.copy()

# Fill only initial undefined enhanced-feature values
enhanced_ready[new_h1_columns] = (
    enhanced_ready[new_h1_columns].fillna(0.0)
)

assert len(enhanced_ready) == len(df)
assert enhanced_ready.index.equals(df.index)

assert np.isfinite(
    enhanced_ready[enhanced_columns].to_numpy(dtype=float)
).all()

# Confirm original labels and original features are unchanged
assert enhanced_ready["crash"].equals(df["crash"])
assert enhanced_ready[combined].equals(df[combined])

print("Dataset shape:", enhanced_ready.shape)
print("Enhanced feature count:", len(enhanced_columns))
print("Dates preserved:", enhanced_ready.index.equals(df.index))
print("All enhanced features finite: True")

# ---------- Original notebook cell 83 ----------
from pathlib import Path
import pandas as pd

indices = dl.split_positions(
    len(enhanced_ready),
    dl.HORIZON,
    dl.SEQUENCE_LENGTH
)

print("Enhanced experiment split dates:")

for part in ["train", "validation", "test"]:
    ids = indices[part]

    print(
        part,
        "start:", enhanced_ready.index[ids[0]].date(),
        "end:", enhanced_ready.index[ids[-1]].date(),
        "observations:", len(ids)
    )

# Compare with an existing model run
original_run = Path(
    run_manifest.loc[
        run_manifest["feature_group"] == "combined",
        "metrics_path"
    ].iloc[0]
).parent

for part in ["validation", "test"]:
    old = pd.read_csv(
        original_run / f"{part}_predictions.csv",
        parse_dates=["Date"]
    )

    new_dates = pd.DatetimeIndex(
        enhanced_ready.index[indices[part]]
    )

    assert old["Date"].reset_index(drop=True).equals(
        pd.Series(new_dates)
    ), f"{part} dates do not match!"

    assert (
        old["label"].to_numpy()
        == enhanced_ready["crash"].iloc[indices[part]].to_numpy()
    ).all(), f"{part} labels do not match!"

    print(f"{part}: dates and labels match original experiment.")

# ---------- Original notebook cell 84 ----------
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

enhanced_runs = []

for seed in [42, 123, 2026]:

    print("\n" + "=" * 65)
    print("Enhanced H1 LSTM | Seed:", seed)
    print("=" * 65)

    dl.train_one(
        df=enhanced_ready,
        feature_set="combined_enhanced_h1",
        columns=enhanced_columns,
        architecture="LSTM",
        seed=seed,
        device=device,
        epochs=25
    )

print("\nAll three enhanced H1 experiments completed.")

# ---------- Original notebook cell 85 ----------
from pathlib import Path
import pandas as pd
import numpy as np

enhanced_folder = (
    NOVELTY / "experiments" / "combined_enhanced_h1_lstm"
)

seeds = [42, 123, 2026]
rows = []

for i, seed in enumerate(seeds, start=1):

    folder = enhanced_folder / f"run_{i:03d}"

    assert (folder / "metrics.csv").exists(), folder

    metrics = pd.read_csv(folder / "metrics.csv")

    for _, r in metrics.iterrows():
        rows.append({
            "feature_group": "combined_enhanced_h1",
            "seed": seed,
            "split": r["split"],
            "PR_AUC": r["PR_AUC"],
            "ROC_AUC": r["ROC_AUC"],
            "best_epoch": r["best_epoch"]
        })

enhanced_metrics = pd.DataFrame(rows)

enhanced_metrics.to_csv(
    NOVELTY / "results" / "experiment3_enhanced_metrics.csv",
    index=False
)

print("Enhanced model metrics:")
display(enhanced_metrics.round(4))



# Read original combined-model metrics directly from saved runs

original_rows = []

original_manifest = run_manifest[
    run_manifest["feature_group"] == "combined"
]

for _, item in original_manifest.iterrows():

    metrics = pd.read_csv(item["metrics_path"])

    for _, r in metrics.iterrows():
        original_rows.append({
            "feature_group": "combined",
            "seed": int(item["seed"]),
            "split": r["split"],
            "PR_AUC": r["PR_AUC"],
            "ROC_AUC": r["ROC_AUC"]
        })

original_combined = pd.DataFrame(original_rows)

# Combine original and enhanced results
comparison = pd.concat(
    [
        original_combined,
        enhanced_metrics[
            ["feature_group", "seed", "split",
             "PR_AUC", "ROC_AUC"]
        ]
    ],
    ignore_index=True
)

comparison_summary = (
    comparison
    .groupby(["feature_group", "split"])
    .agg(
        AP_mean=("PR_AUC", "mean"),
        AP_std=("PR_AUC", "std"),
        ROC_mean=("ROC_AUC", "mean")
    )
    .reset_index()
)

print("Original vs Enhanced H1 Comparison:")
display(comparison_summary.round(4))

comparison_summary.to_csv(
    NOVELTY / "results" / "experiment3_comparison.csv",
    index=False
)

# ---------- Original notebook cell 86 ----------
from sklearn.metrics import average_precision_score

warning_rows = []

for i, seed in enumerate(seeds, start=1):

    folder = enhanced_folder / f"run_{i:03d}"

    val = pd.read_csv(
        folder / "validation_predictions.csv",
        parse_dates=["Date"]
    )

    test = pd.read_csv(
        folder / "test_predictions.csv",
        parse_dates=["Date"]
    )

    threshold = threshold_at_fpr(
        val, target_fpr=0.05
    )

    metrics, episodes = evaluate_warnings(
        test, threshold
    )

    alerts = test["probability"] >= threshold
    negatives = test["label"] == 0
    positives = test["label"] == 1

    warning_rows.append({
        "feature_group": "combined_enhanced_h1",
        "seed": seed,
        "test_AP": average_precision_score(
            test["label"], test["probability"]
        ),
        "test_FPR": float(alerts[negatives].mean()),
        "test_TPR": float(alerts[positives].mean()),
        **metrics
    })

enhanced_warnings = pd.DataFrame(warning_rows)

enhanced_warnings.to_csv(
    NOVELTY / "results" /
    "experiment3_enhanced_warning_runs.csv",
    index=False
)

print("Enhanced model warning results:")
display(enhanced_warnings.round(4))

# Compare with the earlier combined model
baseline_warnings = matched_results[
    matched_results["feature_group"] == "combined"
].copy()

warning_comparison = pd.concat(
    [baseline_warnings, enhanced_warnings],
    ignore_index=True
)

warning_summary_3 = (
    warning_comparison
    .groupby("feature_group")
    .agg(
        AP_mean=("test_AP", "mean"),
        FPR_mean=("test_FPR", "mean"),
        TPR_mean=("test_TPR", "mean"),
        false_alerts_mean=(
            "false_alert_episodes", "mean"
        ),
        detection_mean=(
            "label_episode_recall", "mean"
        ),
        lead_mean=("mean_lead_days", "mean"),
        alert_days_mean=("alert_days", "mean")
    )
    .reset_index()
)

print("\nWarning comparison:")
display(warning_summary_3.round(4))

warning_summary_3.to_csv(
    NOVELTY / "results" /
    "experiment3_warning_comparison.csv",
    index=False
)
