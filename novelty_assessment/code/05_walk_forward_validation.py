# AXIOM — novelty assessment: 05_walk_forward_validation.py
# Extracted verbatim from notebook code cells 105–122 (zero-based).
# These scripts retain notebook-global dependencies and should be run in order
# within the same initialized Colab/Python session (e.g. exec(open(path).read())).
# Original AXIOM preprocessing and model modules must already be available.
# This is an archival extraction, not a standalone refactor.


# ---------- Original notebook cell 105 ----------
import numpy as np
import pandas as pd

# Original fixed split remains untouched
original_indices = dl.split_positions(
    len(df), dl.HORIZON, dl.SEQUENCE_LENGTH
)

original_test_start = int(original_indices["test"][0])

# All supplementary folds finish before original test
historical_end = original_test_start

H = dl.HORIZON
L = dl.SEQUENCE_LENGTH

# Expanding training windows, measured in historical rows
train_fractions = [0.55, 0.62, 0.69]
validation_size = 252
test_size = 252

walk_folds = {}
fold_rows = []

for fold_number, fraction in enumerate(
    train_fractions, start=1
):

    train_boundary = int(
        historical_end * fraction
    )

    validation_start = train_boundary
    validation_end = validation_start + validation_size

    test_start = validation_end
    test_end = test_start + test_size

    if test_end > historical_end:
        raise ValueError(
            f"Fold {fold_number} extends beyond "
            "the original test boundary."
        )

    train_ids = np.arange(
        L - 1,
        train_boundary - H
    )

    validation_ids = np.arange(
        validation_start,
        validation_end - H
    )

    test_ids = np.arange(
        test_start,
        test_end - H
    )

    assert len(train_ids) > 0
    assert len(validation_ids) > 0
    assert len(test_ids) > 0

    # Labels for each earlier partition must not
    # use prices from the next partition.
    assert train_ids[-1] + H < validation_start
    assert validation_ids[-1] + H < test_start
    assert test_ids[-1] + H < original_test_start

    walk_folds[fold_number] = {
        "train": train_ids,
        "validation": validation_ids,
        "test": test_ids
    }

    fold_rows.append({
        "fold": fold_number,
        "train_start": df.index[train_ids[0]].date(),
        "train_end": df.index[train_ids[-1]].date(),
        "validation_start": df.index[validation_ids[0]].date(),
        "validation_end": df.index[validation_ids[-1]].date(),
        "test_start": df.index[test_ids[0]].date(),
        "test_end": df.index[test_ids[-1]].date(),
        "train_n": len(train_ids),
        "validation_n": len(validation_ids),
        "test_n": len(test_ids),
        "train_positives": int(
            df.iloc[train_ids]["crash"].sum()
        ),
        "validation_positives": int(
            df.iloc[validation_ids]["crash"].sum()
        ),
        "test_positives": int(
            df.iloc[test_ids]["crash"].sum()
        )
    })

walk_fold_summary = pd.DataFrame(fold_rows)

display(walk_fold_summary)

walk_fold_summary.to_csv(
    NOVELTY / "results" /
    "experiment7_walk_forward_folds.csv",
    index=False
)

print(
    "Original test begins:",
    df.index[original_test_start].date()
)

# ---------- Original notebook cell 106 ----------
print("Walk-forward fold diagnostic\n")

required_positive_count = 5

for _, row in walk_fold_summary.iterrows():

    print(f"Fold {int(row['fold'])}")

    print(
        "  Training positives:",
        row["train_positives"]
    )

    print(
        "  Validation positives:",
        row["validation_positives"]
    )

    print(
        "  Test positives:",
        row["test_positives"]
    )

    if (
        row["validation_positives"] < required_positive_count
        or row["test_positives"] < required_positive_count
    ):
        print(
            "  WARNING: Too few positive labels "
            "for a stable AP comparison."
        )

    else:
        print("  Candidate fold accepted.")

    print()

print("Original fixed splits have not been modified.")

# ---------- Original notebook cell 107 ----------
import numpy as np
import pandas as pd

original_indices = dl.split_positions(
    len(df), dl.HORIZON, dl.SEQUENCE_LENGTH
)

historical_end = int(original_indices["test"][0])

historical_labels = (
    df.iloc[:historical_end]["crash"]
    .astype(int)
)

episode_start = (
    historical_labels.eq(1)
    & historical_labels.shift(1, fill_value=0).eq(0)
)

episode_ids = episode_start.cumsum()

positive_days = pd.DataFrame({
    "Date": historical_labels.index,
    "label": historical_labels.to_numpy(),
    "episode_id": episode_ids.to_numpy()
})

positive_days = positive_days[
    positive_days["label"] == 1
]

historical_episodes = (
    positive_days
    .groupby("episode_id")
    .agg(
        start=("Date", "min"),
        end=("Date", "max"),
        positive_days=("Date", "size")
    )
    .reset_index(drop=True)
)

historical_episodes["start_year"] = (
    historical_episodes["start"].dt.year
)

print("Historical positive-label episodes:")
display(historical_episodes)

print("\nTotal episodes:", len(historical_episodes))

historical_episodes.to_csv(
    NOVELTY / "results" /
    "experiment7_historical_label_episodes.csv",
    index=False
)

# ---------- Original notebook cell 108 ----------
historical = df.iloc[:historical_end].copy()

historical["year"] = historical.index.year

annual_summary = (
    historical
    .groupby("year")
    .agg(
        trading_days=("crash", "size"),
        positive_days=("crash", "sum")
    )
    .reset_index()
)

annual_summary["positive_rate"] = (
    annual_summary["positive_days"]
    / annual_summary["trading_days"]
)

episode_counts = (
    historical_episodes
    .groupby("start_year")
    .size()
    .rename("episodes_starting")
)

annual_summary = annual_summary.merge(
    episode_counts,
    left_on="year",
    right_index=True,
    how="left"
)

annual_summary["episodes_starting"] = (
    annual_summary["episodes_starting"]
    .fillna(0)
    .astype(int)
)

print("Annual crash-label distribution:")
display(annual_summary)

annual_summary.to_csv(
    NOVELTY / "results" /
    "experiment7_annual_label_distribution.csv",
    index=False
)

# ---------- Original notebook cell 109 ----------
import numpy as np
import pandas as pd

H = dl.HORIZON
L = dl.SEQUENCE_LENGTH

fold_specs = [
    {
        "fold": 1,
        "train_end": "2003-12-31",
        "validation_start": "2004-01-01",
        "validation_end": "2006-12-31",
        "test_start": "2007-01-01",
        "test_end": "2009-12-31"
    },
    {
        "fold": 2,
        "train_end": "2009-12-31",
        "validation_start": "2010-01-01",
        "validation_end": "2012-12-31",
        "test_start": "2013-01-01",
        "test_end": "2015-12-31"
    },
    {
        "fold": 3,
        "train_end": "2015-12-31",
        "validation_start": "2016-01-01",
        "validation_end": "2018-12-31",
        "test_start": "2019-01-01",
        "test_end": "2020-12-31"
    }
]

dates = pd.to_datetime(df.index)
historical_end = int(
    dl.split_positions(
        len(df), H, L
    )["test"][0]
)

walk_folds = {}
fold_rows = []

def count_episodes(labels):
    labels = np.asarray(labels, dtype=int)
    if len(labels) == 0:
        return 0
    return int(
        np.sum(
            (labels == 1)
            & (np.r_[0, labels[:-1]] == 0)
        )
    )

for spec in fold_specs:

    fold = spec["fold"]

    validation_start = np.searchsorted(
        dates, pd.Timestamp(spec["validation_start"])
    )

    test_start = np.searchsorted(
        dates, pd.Timestamp(spec["test_start"])
    )

    test_end = np.searchsorted(
        dates,
        pd.Timestamp(spec["test_end"]),
        side="right"
    )

    train_ids = np.arange(
        L - 1,
        validation_start - H
    )

    validation_ids = np.arange(
        validation_start,
        test_start - H
    )

    test_ids = np.arange(
        test_start,
        min(test_end, historical_end) - H
    )

    assert len(train_ids) > 0
    assert len(validation_ids) > 0
    assert len(test_ids) > 0

    assert train_ids[-1] + H < validation_start
    assert validation_ids[-1] + H < test_start
    assert test_ids[-1] + H < historical_end

    walk_folds[fold] = {
        "train": train_ids,
        "validation": validation_ids,
        "test": test_ids
    }

    row = {"fold": fold}

    for split, ids in walk_folds[fold].items():

        labels = df.iloc[ids]["crash"].to_numpy()

        row[f"{split}_start"] = str(
            df.index[ids[0]].date()
        )

        row[f"{split}_end"] = str(
            df.index[ids[-1]].date()
        )

        row[f"{split}_n"] = len(ids)
        row[f"{split}_positives"] = int(labels.sum())
        row[f"{split}_episodes"] = count_episodes(labels)

    fold_rows.append(row)

walk_fold_summary = pd.DataFrame(fold_rows)

display(walk_fold_summary)

walk_fold_summary.to_csv(
    NOVELTY / "results" /
    "experiment7_final_walk_forward_folds.csv",
    index=False
)

# ---------- Original notebook cell 110 ----------
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from joblib import dump
import json
import time
import copy

WALK = NOVELTY / "experiments" / "walk_forward"
WALK.mkdir(parents=True, exist_ok=True)

walk_feature_groups = {
    "classical": {
        "data": df,
        "columns": list(feature_groups["classical"])
    },
    "combined": {
        "data": df,
        "columns": list(feature_groups["combined"])
    },
    "enhanced_h1": {
        "data": enhanced_df,
        "columns": list(enhanced_columns)
    }
}

SEEDS = [42, 123, 2026]

for name, config in walk_feature_groups.items():

    data = config["data"]
    columns = config["columns"]

    assert len(data) == len(df)
    assert data.index.equals(df.index)
    assert all(c in data.columns for c in columns)
    assert data["crash"].equals(df["crash"])

    # Check only observations actually needed
    # for the walk-forward experiments.
    for fold, splits in walk_folds.items():

        for split_name, ids in splits.items():

            start = ids[0] - dl.SEQUENCE_LENGTH + 1
            end = ids[-1] + 1

            values = data.iloc[start:end][
                columns
            ].to_numpy(dtype=float)

            assert np.isfinite(values).all(), (
                f"Non-finite features in {name}, "
                f"fold {fold}, {split_name}"
            )

    print(f"{name}: {len(columns)} features — verified")

assert len(walk_feature_groups["classical"]["columns"]) == 18
assert len(walk_feature_groups["combined"]["columns"]) == 48
assert len(walk_feature_groups["enhanced_h1"]["columns"]) == 54

print("\nAll feature groups verified.")

# ---------- Original notebook cell 111 ----------
import numpy as np
import pandas as pd

enhanced_check = enhanced_df[enhanced_columns].copy()

nonfinite_mask = ~np.isfinite(
    enhanced_check.to_numpy(dtype=float)
)

nonfinite_counts = pd.Series(
    nonfinite_mask.sum(axis=0),
    index=enhanced_columns
)

problem_columns = (
    nonfinite_counts[nonfinite_counts > 0]
    .sort_values(ascending=False)
)

print("Columns containing non-finite values:")
display(problem_columns)

print("\nTotal affected rows:",
      int(nonfinite_mask.any(axis=1).sum()))

print("\nFirst affected dates:")
display(
    enhanced_check.loc[
        nonfinite_mask.any(axis=1)
    ].head(10)
)

print("\nLast affected dates:")
display(
    enhanced_check.loc[
        nonfinite_mask.any(axis=1)
    ].tail(10)
)

# ---------- Original notebook cell 112 ----------
for fold, splits in walk_folds.items():

    print(f"\nFold {fold}")

    for split_name, ids in splits.items():

        values = enhanced_check.iloc[ids].to_numpy(
            dtype=float
        )

        affected = (~np.isfinite(values)).any(axis=1)

        print(
            f"  {split_name}: "
            f"{affected.sum()} affected rows "
            f"out of {len(ids)}"
        )

# ---------- Original notebook cell 113 ----------
L = dl.SEQUENCE_LENGTH

for fold, splits in walk_folds.items():

    print(f"\nFold {fold}")

    for split_name, ids in splits.items():

        affected_sequences = 0

        for i in ids:

            window = enhanced_check.iloc[
                i - L + 1 : i + 1
            ].to_numpy(dtype=float)

            if not np.isfinite(window).all():
                affected_sequences += 1

        print(
            f"  {split_name}: "
            f"{affected_sequences} affected sequences"
        )

# ---------- Original notebook cell 114 ----------
import numpy as np
import pandas as pd

# Identify the first row where all enhanced features
# are finite from that point onward.

feature_values = enhanced_df[
    enhanced_columns
].to_numpy(dtype=float)

finite_rows = np.isfinite(feature_values).all(axis=1)

# Find the last non-finite row
bad_positions = np.flatnonzero(~finite_rows)

warmup_end = (
    int(bad_positions.max()) + 1
    if len(bad_positions) > 0
    else 0
)

# Every input sequence must begin at or after
# the first fully valid feature row.
first_valid_prediction = (
    warmup_end + dl.SEQUENCE_LENGTH - 1
)

print("First fully valid feature row:", warmup_end)
print("First valid prediction index:", first_valid_prediction)
print(
    "First valid prediction date:",
    df.index[first_valid_prediction]
)

for fold, splits in walk_folds.items():

    original_n = len(splits["train"])

    splits["train"] = splits["train"][
        splits["train"] >= first_valid_prediction
    ]

    print(
        f"Fold {fold}: removed "
        f"{original_n - len(splits['train'])} "
        "warm-up training sequences"
    )

    # Verify every sequence, including its history.
    for split_name, ids in splits.items():

        for group_name, data, columns in [
            ("classical", df, feature_groups["classical"]),
            ("combined", df, feature_groups["combined"]),
            ("enhanced_h1", enhanced_df, enhanced_columns)
        ]:

            used_rows = np.arange(
                ids[0] - dl.SEQUENCE_LENGTH + 1,
                ids[-1] + 1
            )

            values = data.iloc[used_rows][
                columns
            ].to_numpy(dtype=float)

            assert np.isfinite(values).all(), (
                f"Non-finite values: fold {fold}, "
                f"{split_name}, {group_name}"
            )

print("\nAll walk-forward sequences are finite.")

# ---------- Original notebook cell 115 ----------
import torch.nn.functional as F

def train_walk_forward(
    fold,
    group_name,
    seed,
    epochs=25,
    batch_size=128,
    learning_rate=0.001,
    weight_decay=0.0001,
    patience=5
):

    dl.set_seed(seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    config = walk_feature_groups[group_name]
    data = config["data"]
    columns = config["columns"]

    indices = walk_folds[fold]

    train_ids = indices["train"]
    val_ids = indices["validation"]
    test_ids = indices["test"]

    run_dir = (
        WALK / f"fold_{fold}" /
        group_name / f"seed_{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)



    scaler = StandardScaler()

    # Include the history used in training sequences,
    # but never fit on validation/test observations.
    scaler_fit_end = int(train_ids[-1]) + 1

    scaler_fit_start = int(train_ids[0]) - dl.SEQUENCE_LENGTH + 1

    scaler.fit(
        data.iloc[
            scaler_fit_start:scaler_fit_end
        ][columns]
    )

    X_all = scaler.transform(
        data[columns]
    ).astype(np.float32)

    y_all = data["crash"].to_numpy(
        dtype=np.float32
    )



    X_train = dl.make_sequences(
        X_all, train_ids, dl.SEQUENCE_LENGTH
    )
    X_val = dl.make_sequences(
        X_all, val_ids, dl.SEQUENCE_LENGTH
    )
    X_test = dl.make_sequences(
        X_all, test_ids, dl.SEQUENCE_LENGTH
    )

    y_train = torch.tensor(y_all[train_ids])
    y_val = torch.tensor(y_all[val_ids])
    y_test = torch.tensor(y_all[test_ids])

    n_pos = int(y_train.sum().item())
    n_neg = len(y_train) - n_pos

    assert n_pos > 0 and n_neg > 0

    pos_weight = n_neg / n_pos

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight,
            dtype=torch.float32,
            device=device
        )
    )

    model = dl.make_model(
        "LSTM",
        input_size=len(columns),
        sequence_length=dl.SEQUENCE_LENGTH
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    train_dataset = torch.utils.data.TensorDataset(
        X_train, y_train
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = None
    stale_epochs = 0
    history = []

    start_time = time.time()



    for epoch in range(1, epochs + 1):

        model.train()
        total_loss = 0.0
        total_samples = 0

        for xb, yb in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb).reshape(-1)

            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(yb)
            total_samples += len(yb)

        train_loss = total_loss / total_samples



        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():

            for start in range(0, len(X_val), 256):

                xb = X_val[start:start + 256].to(device)
                yb = y_val[start:start + 256].to(device)

                logits = model(xb).reshape(-1)

                batch_loss = F.binary_cross_entropy_with_logits(
                    logits,
                    yb,
                    pos_weight=criterion.pos_weight,
                    reduction="sum"
                )

                val_loss_sum += batch_loss.item()

        val_loss = val_loss_sum / len(X_val)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss
        })

        print(
            f"Fold {fold} | {group_name} | "
            f"seed {seed} | epoch {epoch} | "
            f"validation loss {val_loss:.5f}"
        )

        if val_loss < best_val_loss - 1e-8:

            best_val_loss = val_loss
            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            stale_epochs = 0

        else:

            stale_epochs += 1

            if stale_epochs >= patience:
                break



    model.load_state_dict(best_state)
    model.eval()

    def get_probabilities(X):

        predictions = []

        with torch.no_grad():

            for start in range(0, len(X), 256):

                xb = X[start:start + 256].to(device)

                logits = model(xb).reshape(-1)

                probs = torch.sigmoid(logits)

                predictions.append(
                    probs.cpu().numpy()
                )

        return np.concatenate(predictions)

    val_probs = get_probabilities(X_val)
    test_probs = get_probabilities(X_test)

    val_predictions = pd.DataFrame({
        "Date": data.index[val_ids],
        "label": y_all[val_ids].astype(int),
        "probability": val_probs
    })

    test_predictions = pd.DataFrame({
        "Date": data.index[test_ids],
        "label": y_all[test_ids].astype(int),
        "probability": test_probs
    })


    def safe_metrics(predictions):

        y = predictions["label"].to_numpy()
        p = predictions["probability"].to_numpy()

        if len(np.unique(y)) < 2:
            return np.nan, np.nan

        return (
            average_precision_score(y, p),
            roc_auc_score(y, p)
        )

    val_ap, val_roc = safe_metrics(val_predictions)
    test_ap, test_roc = safe_metrics(test_predictions)



    torch.save(
        best_state,
        run_dir / "model.pt"
    )

    dump(
        scaler,
        run_dir / "scaler.joblib"
    )

    pd.DataFrame(history).to_csv(
        run_dir / "training_history.csv",
        index=False
    )

    val_predictions.to_csv(
        run_dir / "validation_predictions.csv",
        index=False
    )

    test_predictions.to_csv(
        run_dir / "test_predictions.csv",
        index=False
    )

    settings = {
        "experiment": "walk_forward",
        "fold": fold,
        "feature_group": group_name,
        "seed": seed,
        "features": columns,
        "architecture": "LSTM",
        "loss": "weighted_bce",
        "checkpoint_selection": "validation_weighted_bce",
        "positive_class_weight": pos_weight,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "sequence_length": dl.SEQUENCE_LENGTH,
        "horizon": dl.HORIZON,
        "epochs_max": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "patience": patience,
        "train_end": str(data.index[train_ids[-1]]),
        "validation_start": str(data.index[val_ids[0]]),
        "test_start": str(data.index[test_ids[0]])
    }

    with open(run_dir / "settings.json", "w") as f:
        json.dump(settings, f, indent=2)

    result = {
        "fold": fold,
        "feature_group": group_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "validation_loss": best_val_loss,
        "validation_AP": val_ap,
        "validation_ROC": val_roc,
        "test_AP": test_ap,
        "test_ROC": test_roc,
        "training_seconds": time.time() - start_time,
        "run_dir": str(run_dir)
    }

    pd.DataFrame([result]).to_csv(
        run_dir / "metrics.csv",
        index=False
    )

    return result

# ---------- Original notebook cell 116 ----------
smoke_result = train_walk_forward(
    fold=1,
    group_name="classical",
    seed=42,
    epochs=25
)

display(pd.DataFrame([smoke_result]).round(4))

# ---------- Original notebook cell 117 ----------
walk_results = []

for fold in [1, 2, 3]:

    for group_name in [
        "classical",
        "combined",
        "enhanced_h1"
    ]:

        for seed in SEEDS:

            print("\n" + "=" * 65)
            print(
                f"FOLD {fold} | "
                f"FEATURES {group_name} | "
                f"SEED {seed}"
            )
            print("=" * 65)

            if (
                fold == 1
                and group_name == "classical"
                and seed == 42
            ):
                result = smoke_result

            else:
                result = train_walk_forward(
                    fold=fold,
                    group_name=group_name,
                    seed=seed
                )

            walk_results.append(result)

            pd.DataFrame(walk_results).to_csv(
                NOVELTY / "results" /
                "experiment7_walk_forward_training_runs.csv",
                index=False
            )

print("\nAll walk-forward training runs completed.")

walk_results_df = pd.DataFrame(walk_results)

display(
    walk_results_df[
        [
            "fold",
            "feature_group",
            "seed",
            "best_epoch",
            "validation_loss",
            "test_AP",
            "test_ROC"
        ]
    ].round(4)
)

# ---------- Original notebook cell 118 ----------
walk_warning_rows = []

for result in walk_results:

    run_dir = Path(result["run_dir"])

    val = pd.read_csv(
        run_dir / "validation_predictions.csv",
        parse_dates=["Date"]
    )

    test = pd.read_csv(
        run_dir / "test_predictions.csv",
        parse_dates=["Date"]
    )

    threshold = threshold_at_fpr(
        val, target_fpr=0.05
    )

    warning_metrics, _ = evaluate_warnings(
        test, threshold
    )

    alerts = test["probability"] >= threshold

    negatives = test["label"] == 0
    positives = test["label"] == 1

    walk_warning_rows.append({
        **result,
        "threshold": threshold,
        "test_FPR": float(alerts[negatives].mean()),
        "test_TPR": (
            float(alerts[positives].mean())
            if positives.any() else np.nan
        ),
        **warning_metrics
    })

walk_warning_results = pd.DataFrame(
    walk_warning_rows
)

walk_warning_results.to_csv(
    NOVELTY / "results" /
    "experiment7_walk_forward_warning_runs.csv",
    index=False
)

walk_summary = (
    walk_warning_results
    .groupby(["fold", "feature_group"])
    .agg(
        AP_mean=("test_AP", "mean"),
        AP_std=("test_AP", "std"),
        FPR_mean=("test_FPR", "mean"),
        TPR_mean=("test_TPR", "mean"),
        detection_mean=("label_episode_recall", "mean"),
        false_alerts_mean=("false_alert_episodes", "mean"),
        lead_mean=("mean_lead_days", "mean")
    )
    .reset_index()
)

display(walk_summary.round(4))

walk_summary.to_csv(
    NOVELTY / "results" /
    "experiment7_walk_forward_summary.csv",
    index=False
)

# ---------- Original notebook cell 119 ----------
import numpy as np
import pandas as pd

paired = walk_warning_results.pivot(
    index=["fold", "seed"],
    columns="feature_group",
    values=["test_AP", "test_FPR", "false_alert_episodes",
            "label_episode_recall", "mean_lead_days"]
)

rows = []

for (fold, seed), r in paired.iterrows():

    rows.append({
        "fold": fold,
        "seed": seed,

        "AP_combined": r[("test_AP", "combined")],
        "AP_enhanced": r[("test_AP", "enhanced_h1")],

        "delta_AP": (
            r[("test_AP", "enhanced_h1")]
            - r[("test_AP", "combined")]
        ),

        "delta_FPR": (
            r[("test_FPR", "enhanced_h1")]
            - r[("test_FPR", "combined")]
        ),

        "delta_false_episodes": (
            r[("false_alert_episodes", "enhanced_h1")]
            - r[("false_alert_episodes", "combined")]
        ),

        "delta_detection": (
            r[("label_episode_recall", "enhanced_h1")]
            - r[("label_episode_recall", "combined")]
        ),

        "delta_lead": (
            r[("mean_lead_days", "enhanced_h1")]
            - r[("mean_lead_days", "combined")]
        )
    })

paired_improvements = pd.DataFrame(rows)

display(paired_improvements.round(4))

paired_improvements.to_csv(
    NOVELTY / "results" /
    "experiment7_paired_improvements.csv",
    index=False
)

print("\nPaired AP improvement summary:")

display(
    paired_improvements.groupby("fold")["delta_AP"]
    .agg(["mean", "std", "min", "max"])
    .round(4)
)

# ---------- Original notebook cell 120 ----------
walk_results = pd.read_csv(
    NOVELTY / "results" /
    "experiment7_walk_forward_training_runs.csv"
)

# ---------- Original notebook cell 121 ----------
import numpy as np
import pandas as pd

diagnostics = []

for _, run in walk_results.iterrows():

    run_dir = Path(run["run_dir"])

    val = pd.read_csv(
        run_dir / "validation_predictions.csv"
    )

    test = pd.read_csv(
        run_dir / "test_predictions.csv"
    )

    val_neg = val.loc[
        val["label"] == 0, "probability"
    ].to_numpy()

    test_neg = test.loc[
        test["label"] == 0, "probability"
    ].to_numpy()

    test_pos = test.loc[
        test["label"] == 1, "probability"
    ].to_numpy()

    threshold = threshold_at_fpr(
        val, target_fpr=0.05
    )

    diagnostics.append({
        "fold": int(run["fold"]),
        "feature_group": run["feature_group"],
        "seed": int(run["seed"]),
        "threshold": threshold,
        "validation_negative_median": np.median(val_neg),
        "test_negative_median": np.median(test_neg),
        "test_positive_median": (
            np.median(test_pos)
            if len(test_pos) else np.nan
        ),
        "validation_negative_p95": np.quantile(val_neg, 0.95),
        "test_negative_p95": np.quantile(test_neg, 0.95),
        "test_FPR": np.mean(test_neg >= threshold),
        "test_TPR": (
            np.mean(test_pos >= threshold)
            if len(test_pos) else np.nan
        ),
        "test_prevalence": test["label"].mean(),
        "test_AP": run["test_AP"]
    })

threshold_diagnostics = pd.DataFrame(diagnostics)

display(
    threshold_diagnostics.groupby(
        ["fold", "feature_group"]
    ).agg(
        threshold_mean=("threshold", "mean"),
        val_neg_p95=("validation_negative_p95", "mean"),
        test_neg_p95=("test_negative_p95", "mean"),
        FPR_mean=("test_FPR", "mean"),
        TPR_mean=("test_TPR", "mean"),
        AP_mean=("test_AP", "mean"),
        prevalence=("test_prevalence", "mean")
    ).round(4)
)

threshold_diagnostics.to_csv(
    NOVELTY / "results" /
    "experiment7_threshold_diagnostics.csv",
    index=False
)

print("Threshold diagnostics saved.")

# ---------- Original notebook cell 122 ----------
ap_lift = threshold_diagnostics.copy()

ap_lift["AP_lift"] = (
    ap_lift["test_AP"] /
    ap_lift["test_prevalence"]
)

ap_lift_summary = (
    ap_lift.groupby(["fold", "feature_group"])
    .agg(
        prevalence=("test_prevalence", "mean"),
        AP_mean=("test_AP", "mean"),
        AP_lift_mean=("AP_lift", "mean")
    )
    .reset_index()
)

display(ap_lift_summary.round(4))

ap_lift_summary.to_csv(
    NOVELTY / "results" /
    "experiment7_AP_lift.csv",
    index=False
)
