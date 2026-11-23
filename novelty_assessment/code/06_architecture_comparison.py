# AXIOM — novelty assessment: 06_architecture_comparison.py
# Extracted verbatim from notebook code cells 123–131 (zero-based).
# These scripts retain notebook-global dependencies and should be run in order
# within the same initialized Colab/Python session (e.g. exec(open(path).read())).
# Original AXIOM preprocessing and model modules must already be available.
# This is an archival extraction, not a standalone refactor.


# ---------- Original notebook cell 123 ----------
import inspect
import importlib
import torch
import pandas as pd

dl = importlib.import_module("04_deep_learning")

print("=" * 70)
print("AVAILABLE MODEL IMPLEMENTATIONS")
print("=" * 70)

print("\nmake_model() implementation:")
print(inspect.getsource(dl.make_model))

for name in ["LSTMModel", "BiLSTMModel", "TransformerModel"]:

    if hasattr(dl, name):
        print(f"\n{'=' * 70}")
        print(name)
        print("=" * 70)
        print(inspect.getsource(getattr(dl, name)))
    else:
        print(f"\n{name}: not found under this exact name")

print("\n" + "=" * 70)
print("FIXED-SPLIT VERIFICATION")
print("=" * 70)

fixed_splits = dl.split_positions(
    len(df),
    dl.HORIZON,
    dl.SEQUENCE_LENGTH
)

for split_name, ids in fixed_splits.items():

    labels = df["crash"].iloc[ids]

    print(
        f"{split_name:12s} | "
        f"n={len(ids):5d} | "
        f"positive={int(labels.sum()):4d} | "
        f"start={df.index[ids[0]].date()} | "
        f"end={df.index[ids[-1]].date()}"
    )

print("\nSequence length:", dl.SEQUENCE_LENGTH)
print("Crash horizon:", dl.HORIZON)

# ---------- Original notebook cell 124 ----------
import torch

architectures = ["LSTM", "BiLSTM", "Transformer"]

feature_counts = {
    "classical": 18,
    "combined": 48,
    "enhanced_h1": 54
}

audit_rows = []

for architecture in architectures:

    for group_name, n_features in feature_counts.items():

        try:
            model = dl.make_model(
                architecture,
                input_size=n_features,
                sequence_length=dl.SEQUENCE_LENGTH
            )

            model.eval()

            x = torch.randn(
                4,
                dl.SEQUENCE_LENGTH,
                n_features
            )

            with torch.no_grad():
                output = model(x)

            if isinstance(output, (tuple, list)):
                raise TypeError(
                    "Model returned multiple outputs"
                )

            assert isinstance(output, torch.Tensor)

            assert output.numel() == 4, (
                f"Expected 4 logits, got {tuple(output.shape)}"
            )

            assert torch.isfinite(output).all()

            audit_rows.append({
                "architecture": architecture,
                "feature_group": group_name,
                "parameters": sum(
                    p.numel() for p in model.parameters()
                ),
                "output_shape": str(tuple(output.shape)),
                "status": "PASS"
            })

        except Exception as exc:

            audit_rows.append({
                "architecture": architecture,
                "feature_group": group_name,
                "parameters": None,
                "output_shape": None,
                "status": f"FAIL: {type(exc).__name__}: {exc}"
            })

architecture_audit = pd.DataFrame(audit_rows)

display(architecture_audit)

architecture_audit.to_csv(
    NOVELTY / "results" /
    "experiment8_architecture_audit.csv",
    index=False
)

print("\nAudit saved to Google Drive.")

# ---------- Original notebook cell 125 ----------
import inspect

for name in ["LSTMModel", "BiLSTMModel", "TransformerModel"]:
    cls = dl.make_model.__globals__.get(name)

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    if cls is None:
        print("Class not found in make_model globals")
        continue

    try:
        print(inspect.getsource(cls))
    except (OSError, TypeError):
        print("Source unavailable")
        print(cls)

print("\nArchitecture parameter counts:")
display(architecture_audit[
    ["architecture", "feature_group", "parameters"]
])

# ---------- Original notebook cell 126 ----------
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import json
import time

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from joblib import dump

ARCH8 = NOVELTY / "experiments" / "architecture_comparison"
ARCH8.mkdir(parents=True, exist_ok=True)

fixed_splits = dl.split_positions(
    len(df), dl.HORIZON, dl.SEQUENCE_LENGTH
)

arch8_groups = {
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

ARCHITECTURES = ["LSTM", "BiLSTM", "Transformer"]
ARCH8_SEEDS = [42, 123, 2026]

# All fixed-split prediction dates remain unchanged.
# Only the early history used to initialize rolling features
# needs special treatment.

for group_name, config in arch8_groups.items():

    data = config["data"]
    cols = config["columns"]

    assert data.index.equals(df.index)
    assert data["crash"].equals(df["crash"])

    # The first training prediction uses a 30-day sequence.
    # Its earliest input row is index 0.
    # Enhanced features have 30 initial undefined rows.
    # For fairness, use a common training subset across groups.

feature_values = enhanced_df[
    enhanced_columns
].to_numpy(dtype=float)

bad_rows = np.flatnonzero(
    ~np.isfinite(feature_values).all(axis=1)
)

first_finite_row = (
    int(bad_rows.max()) + 1
    if len(bad_rows) else 0
)

first_valid_prediction = (
    first_finite_row + dl.SEQUENCE_LENGTH - 1
)

arch8_train_ids = fixed_splits["train"][
    fixed_splits["train"] >= first_valid_prediction
]

arch8_ids = {
    "train": arch8_train_ids,
    "validation": fixed_splits["validation"].copy(),
    "test": fixed_splits["test"].copy()
}

for group_name, config in arch8_groups.items():

    values = config["data"][
        config["columns"]
    ].to_numpy(dtype=float)

    for split_name, ids in arch8_ids.items():

        start = int(ids[0]) - dl.SEQUENCE_LENGTH + 1
        end = int(ids[-1]) + 1

        assert np.isfinite(
            values[start:end]
        ).all(), (
            f"Non-finite values in {group_name}, {split_name}"
        )

print("Original train predictions:", len(fixed_splits["train"]))
print("Common valid train predictions:", len(arch8_ids["train"]))
print("Validation predictions:", len(arch8_ids["validation"]))
print("Test predictions:", len(arch8_ids["test"]))
print("Architecture experiment directory:", ARCH8)

# ---------- Original notebook cell 127 ----------
from torch.utils.data import TensorDataset, DataLoader

def train_arch8(
    architecture,
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

    config = arch8_groups[group_name]
    data = config["data"]
    columns = config["columns"]

    run_dir = (
        ARCH8 / architecture /
        group_name / f"seed_{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ids = arch8_ids["train"]
    val_ids = arch8_ids["validation"]
    test_ids = arch8_ids["test"]

    # Training-history-only standardization.
    scaler_start = int(train_ids[0]) - dl.SEQUENCE_LENGTH + 1
    scaler_end = int(train_ids[-1]) + 1

    scaler = StandardScaler()
    scaler.fit(
        data.iloc[scaler_start:scaler_end][columns]
    )

    X_all = scaler.transform(
        data[columns].iloc[scaler_start:]
    ).astype(np.float32)

    # Restore original indexing for sequence construction.
    X_full = np.zeros(
        (len(data), len(columns)),
        dtype=np.float32
    )
    X_full[scaler_start:] = X_all

    y_all = data["crash"].to_numpy(dtype=np.float32)

    def prepare(ids):
        X = dl.make_sequences(
            X_full, ids, dl.SEQUENCE_LENGTH
        )
        y = torch.tensor(
            y_all[ids], dtype=torch.float32
        )
        return X, y

    X_train, y_train = prepare(train_ids)
    X_val, y_val = prepare(val_ids)
    X_test, y_test = prepare(test_ids)

    pos = int(y_train.sum().item())
    neg = len(y_train) - pos

    assert pos > 0 and neg > 0

    pos_weight = neg / pos

    model = dl.make_model(
        architecture,
        input_size=len(columns),
        sequence_length=dl.SEQUENCE_LENGTH
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            pos_weight, device=device
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    generator = torch.Generator().manual_seed(seed)

    loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator
    )

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0
    history = []

    started = time.time()

    for epoch in range(1, epochs + 1):

        model.train()
        train_losses = []

        for xb, yb in loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            logits = model(xb).reshape(-1)
            loss = criterion(logits, yb)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()

        with torch.no_grad():

            val_logits = []

            for start in range(0, len(X_val), 512):

                xb = X_val[start:start + 512].to(device)

                val_logits.append(
                    model(xb).reshape(-1)
                )

            val_logits = torch.cat(val_logits)

            val_loss = F.binary_cross_entropy_with_logits(
                val_logits,
                y_val.to(device),
                pos_weight=criterion.pos_weight,
                reduction="mean"
            ).item()

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "validation_loss": val_loss
        })

        print(
            f"{architecture} | {group_name} | "
            f"seed {seed} | epoch {epoch} | "
            f"val_loss {val_loss:.5f}"
        )

        if val_loss < best_loss - 1e-6:

            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )
            no_improve = 0

        else:
            no_improve += 1

        if no_improve >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()

    def predict(X):

        outputs = []

        with torch.no_grad():

            for start in range(0, len(X), 512):

                xb = X[start:start + 512].to(device)

                logits = model(xb).reshape(-1)

                outputs.append(
                    torch.sigmoid(logits).cpu().numpy()
                )

        return np.concatenate(outputs)

    val_prob = predict(X_val)
    test_prob = predict(X_test)

    def metrics(y, p):

        y = y.cpu().numpy()

        return {
            "AP": float(average_precision_score(y, p)),
            "ROC": float(roc_auc_score(y, p))
        }

    val_metrics = metrics(y_val, val_prob)
    test_metrics = metrics(y_test, test_prob)

    def save_predictions(ids, probabilities, filename):

        pd.DataFrame({
            "Date": data.index[ids],
            "label": y_all[ids].astype(int),
            "probability": probabilities
        }).to_csv(
            run_dir / filename,
            index=False
        )

    save_predictions(
        val_ids, val_prob, "validation_predictions.csv"
    )
    save_predictions(
        test_ids, test_prob, "test_predictions.csv"
    )

    torch.save(
        model.state_dict(),
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

    settings = {
        "architecture": architecture,
        "feature_group": group_name,
        "seed": seed,
        "features": columns,
        "best_epoch": best_epoch,
        "pos_weight": pos_weight,
        "checkpoint_metric": "validation_weighted_BCE",
        "sequence_length": dl.SEQUENCE_LENGTH,
        "horizon": dl.HORIZON,
        "train_start": str(data.index[train_ids[0]]),
        "train_end": str(data.index[train_ids[-1]]),
        "validation_start": str(data.index[val_ids[0]]),
        "validation_end": str(data.index[val_ids[-1]]),
        "test_start": str(data.index[test_ids[0]]),
        "test_end": str(data.index[test_ids[-1]])
    }

    with open(run_dir / "settings.json", "w") as f:
        json.dump(settings, f, indent=2)

    result = {
        "architecture": architecture,
        "feature_group": group_name,
        "seed": seed,
        "parameters": sum(
            p.numel() for p in model.parameters()
        ),
        "best_epoch": best_epoch,
        "validation_AP": val_metrics["AP"],
        "validation_ROC": val_metrics["ROC"],
        "test_AP": test_metrics["AP"],
        "test_ROC": test_metrics["ROC"],
        "training_seconds": time.time() - started,
        "run_dir": str(run_dir)
    }

    pd.DataFrame([result]).to_csv(
        run_dir / "metrics.csv",
        index=False
    )

    return result

# ---------- Original notebook cell 128 ----------
arch8_smoke = train_arch8(
    architecture="BiLSTM",
    group_name="combined",
    seed=42
)

display(
    pd.DataFrame([arch8_smoke]).round(4)
)

# ---------- Original notebook cell 129 ----------
import pandas as pd
from pathlib import Path

ARCH8_RESULTS = (
    NOVELTY / "results" /
    "experiment8_architecture_training_runs.csv"
)

all_arch8_results = []

for architecture in ARCHITECTURES:

    for group_name in arch8_groups:

        for seed in ARCH8_SEEDS:

            run_dir = (
                ARCH8 / architecture /
                group_name / f"seed_{seed}"
            )

            metrics_file = run_dir / "metrics.csv"

            print("\n" + "=" * 70)
            print(
                f"ARCHITECTURE: {architecture} | "
                f"FEATURES: {group_name} | SEED: {seed}"
            )
            print("=" * 70)

            if metrics_file.exists():

                print("Existing completed run found. Reusing.")

                result = pd.read_csv(
                    metrics_file
                ).iloc[0].to_dict()

            else:

                result = train_arch8(
                    architecture=architecture,
                    group_name=group_name,
                    seed=seed
                )

            all_arch8_results.append(result)

            # Save after every completed configuration.
            pd.DataFrame(
                all_arch8_results
            ).to_csv(
                ARCH8_RESULTS,
                index=False
            )

arch8_results = pd.DataFrame(all_arch8_results)

print("\nAll Experiment 8 training runs completed.")

display(
    arch8_results[
        [
            "architecture",
            "feature_group",
            "seed",
            "best_epoch",
            "validation_AP",
            "test_AP",
            "test_ROC"
        ]
    ].round(4)
)

print("\nSaved to:", ARCH8_RESULTS)

# ---------- Original notebook cell 130 ----------
arch8_summary = (
    arch8_results
    .groupby(["architecture", "feature_group"])
    .agg(
        AP_mean=("test_AP", "mean"),
        AP_std=("test_AP", "std"),
        ROC_mean=("test_ROC", "mean"),
        validation_AP_mean=("validation_AP", "mean"),
        parameters=("parameters", "first"),
        best_epoch_mean=("best_epoch", "mean")
    )
    .reset_index()
)

display(
    arch8_summary.sort_values(
        ["feature_group", "AP_mean"],
        ascending=[True, False]
    ).round(4)
)

arch8_summary.to_csv(
    NOVELTY / "results" /
    "experiment8_architecture_summary.csv",
    index=False
)

# ---------- Original notebook cell 131 ----------
arch8_warning_rows = []

for _, run in arch8_results.iterrows():

    run_dir = Path(run["run_dir"])

    val = pd.read_csv(
        run_dir / "validation_predictions.csv"
    )

    test = pd.read_csv(
        run_dir / "test_predictions.csv"
    )

    threshold = threshold_at_fpr(
        val,
        target_fpr=0.05
    )

    warning_metrics, episodes = evaluate_warnings(
        test,
        threshold
    )

    test_labels = test["label"].to_numpy()
    alerts = (
        test["probability"].to_numpy() >= threshold
    )

    negative_mask = test_labels == 0
    positive_mask = test_labels == 1

    row = {
        "architecture": run["architecture"],
        "feature_group": run["feature_group"],
        "seed": run["seed"],
        "test_AP": run["test_AP"],
        "test_ROC": run["test_ROC"],
        "test_FPR": float(
            alerts[negative_mask].mean()
        ),
        "test_TPR": float(
            alerts[positive_mask].mean()
        )
    }

    row.update(warning_metrics)

    arch8_warning_rows.append(row)

arch8_warnings = pd.DataFrame(
    arch8_warning_rows
)

arch8_warning_summary = (
    arch8_warnings
    .groupby(["architecture", "feature_group"])
    .agg(
        AP_mean=("test_AP", "mean"),
        FPR_mean=("test_FPR", "mean"),
        TPR_mean=("test_TPR", "mean"),
        detection_mean=("label_episode_recall", "mean"),
        false_alerts_mean=("false_alert_episodes", "mean"),
        lead_mean=("mean_lead_days", "mean"),
        alert_days_mean=("alert_days", "mean")
    )
    .reset_index()
)

display(
    arch8_warning_summary.round(4)
)

arch8_warnings.to_csv(
    NOVELTY / "results" /
    "experiment8_architecture_warning_runs.csv",
    index=False
)

arch8_warning_summary.to_csv(
    NOVELTY / "results" /
    "experiment8_architecture_warning_summary.csv",
    index=False
)

print("Architecture warning analysis saved.")
