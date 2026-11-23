# AXIOM — novelty assessment: 04_loss_function_comparison.py
# Extracted verbatim from notebook code cells 87–104 (zero-based).
# These scripts retain notebook-global dependencies and should be run in order
# within the same initialized Colab/Python session (e.g. exec(open(path).read())).
# Original AXIOM preprocessing and model modules must already be available.
# This is an archival extraction, not a standalone refactor.


# ---------- Original notebook cell 87 ----------
import inspect

print("=== MODEL FACTORY ===")
print(inspect.getsource(dl.make_model))

print("\n=== SEQUENCE BUILDER ===")
print(inspect.getsource(dl.make_sequences))

print("\n=== PREDICTION FUNCTION ===")
print(inspect.getsource(dl.predict))

print("\n=== SPLIT FUNCTION ===")
print(inspect.getsource(dl.split_positions))

# ---------- Original notebook cell 88 ----------
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalLoss(nn.Module):

    def __init__(self, pos_weight, gamma=2.0):
        super().__init__()
        self.register_buffer(
            "pos_weight",
            torch.as_tensor(pos_weight, dtype=torch.float32)
        )
        self.gamma = gamma

    def forward(self, logits, targets):

        logits = logits.reshape(-1)
        targets = targets.reshape(-1).float()

        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        pt = torch.exp(-bce)

        weights = torch.where(
            targets == 1,
            self.pos_weight,
            torch.ones_like(targets)
        )

        loss = weights * (1 - pt).pow(self.gamma) * bce

        return loss.mean()


# Numerical sanity check
test_logits = torch.tensor([-2.0, 0.0, 2.0])
test_labels = torch.tensor([0.0, 1.0, 1.0])

test_loss = WeightedFocalLoss(pos_weight=5.0)

print("Focal loss:", test_loss(test_logits, test_labels).item())

# ---------- Original notebook cell 89 ----------
import copy
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset


def train_focal(df, columns, seed, device, epochs=25, gamma=2.0):

    dl.set_seed(seed)

    feature_set = "combined_focal_gamma2"
    architecture = "LSTM"

    folder = dl.new_run_folder(
        feature_set + "_" + architecture.lower()
    )

    indices = dl.split_positions(
        len(df), dl.HORIZON, dl.SEQUENCE_LENGTH
    )

    y = df["crash"].to_numpy(dtype=int)

    # Confirm evaluation periods remain unchanged
    assert str(df.index[indices["validation"][0]].date()) == "2016-05-13"
    assert str(df.index[indices["test"][0]].date()) == "2021-06-28"

    scaler = StandardScaler()
    scaler.fit(df.iloc[indices["train"]][columns])

    x = scaler.transform(df[columns]).astype("float32")

    joblib.dump(scaler, folder / "scaler.joblib")

    sequences = {
        part: dl.make_sequences(
            x, ids, dl.SEQUENCE_LENGTH
        )
        for part, ids in indices.items()
    }

    labels = {
        part: torch.tensor(
            y[ids], dtype=torch.float32
        )
        for part, ids in indices.items()
    }

    model = dl.make_model(
        architecture,
        len(columns),
        dl.SEQUENCE_LENGTH
    ).to(device)

    positive_count = labels["train"].sum().item()
    negative_count = len(labels["train"]) - positive_count

    if positive_count == 0:
        raise ValueError("No positive crash labels in training.")

    pos_weight = negative_count / positive_count

    loss_fn = WeightedFocalLoss(
        pos_weight=pos_weight,
        gamma=gamma
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    loader = DataLoader(
        TensorDataset(
            sequences["train"],
            labels["train"]
        ),
        batch_size=128,
        shuffle=True
    )

    dl.save_settings(folder, {
        "architecture": architecture,
        "feature_set": feature_set,
        "feature_columns": list(columns),
        "seed": seed,
        "loss": "weighted_focal",
        "gamma": gamma,
        "positive_class_weight": pos_weight,
        "epochs_allowed": epochs,
        "patience": dl.PATIENCE,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 128,
        "sequence_length": dl.SEQUENCE_LENGTH,
        "horizon": dl.HORIZON,
        "crash_threshold": 0.10,
        "train_end_date": str(
            df.index[indices["train"][-1]].date()
        ),
        "validation_start_date": str(
            df.index[indices["validation"][0]].date()
        ),
        "test_start_date": str(
            df.index[indices["test"][0]].date()
        ),
        "device": device
    })

    best_ap = -1
    best_weights = None
    best_epoch = None
    patience_count = 0

    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):

        model.train()
        loss_total = 0.0

        for batch_x, batch_y in loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()

            loss_total += loss.item() * len(batch_y)

        probabilities = dl.predict(
            model,
            sequences["validation"],
            device
        )

        val_ap = average_precision_score(
            y[indices["validation"]],
            probabilities
        )

        history.append({
            "epoch": epoch,
            "train_loss": loss_total / len(labels["train"]),
            "validation_PR_AUC": val_ap
        })

        print(
            "Focal LSTM",
            "seed", seed,
            "epoch", epoch,
            "validation PR-AUC", round(val_ap, 4),
            flush=True
        )

        if val_ap > best_ap + 1e-5:

            best_ap = val_ap
            best_epoch = epoch

            best_weights = copy.deepcopy(
                model.state_dict()
            )

            patience_count = 0

        else:
            patience_count += 1

        if patience_count >= dl.PATIENCE:
            break

    model.load_state_dict(best_weights)

    torch.save({
        "state_dict": model.state_dict(),
        "architecture": architecture,
        "input_size": len(columns),
        "sequence_length": dl.SEQUENCE_LENGTH,
        "feature_columns": list(columns),
        "loss": "weighted_focal",
        "gamma": gamma
    }, folder / "model.pt")

    pd.DataFrame(history).to_csv(
        folder / "training_history.csv",
        index=False
    )

    rows = []

    for part in ["validation", "test"]:

        ids = indices[part]

        probabilities = dl.predict(
            model, sequences[part], device
        )

        rows.append({
            "split": part,
            "best_epoch": best_epoch,
            "training_seconds": round(
                time.time() - start_time, 1
            ),
            **dl.score_predictions(
                y[ids], probabilities
            )
        })

        pd.DataFrame({
            "Date": df.index[ids],
            "label": y[ids],
            "probability": probabilities
        }).to_csv(
            folder / f"{part}_predictions.csv",
            index=False
        )

    pd.DataFrame(rows).to_csv(
        folder / "metrics.csv",
        index=False
    )

    print("Saved:", folder, flush=True)

    return folder

# ---------- Original notebook cell 90 ----------
import torch
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"

assert len(combined) == 48
assert df.index.equals(enhanced_ready.index)
assert df[combined].notna().all().all()

# Confirm the folder patch remains active
assert dl.new_run_folder is novelty_run_folder, (
    "The novelty experiment folder patch is not active."
)

focal_run_folders = {}

for seed in [42, 123, 2026]:

    print("\n" + "=" * 65)
    print("FOCAL LOSS EXPERIMENT | SEED:", seed)
    print("=" * 65)

    folder = train_focal(
        df=df,
        columns=list(combined),
        seed=seed,
        device=device,
        epochs=25,
        gamma=2.0
    )

    focal_run_folders[seed] = folder

print("\nAll focal-loss experiments completed.")

# ---------- Original notebook cell 91 ----------
import pandas as pd

comparison_rows = []

# Existing weighted-BCE runs
baseline_manifest = run_manifest[
    run_manifest["feature_group"] == "combined"
]

for _, item in baseline_manifest.iterrows():

    metrics = pd.read_csv(item["metrics_path"])

    for _, r in metrics.iterrows():
        comparison_rows.append({
            "model": "Weighted BCE",
            "seed": int(item["seed"]),
            "split": r["split"],
            "AP": r["PR_AUC"],
            "ROC_AUC": r["ROC_AUC"]
        })

# New focal-loss runs
for seed, folder in focal_run_folders.items():

    metrics = pd.read_csv(folder / "metrics.csv")

    for _, r in metrics.iterrows():
        comparison_rows.append({
            "model": "Focal Loss",
            "seed": seed,
            "split": r["split"],
            "AP": r["PR_AUC"],
            "ROC_AUC": r["ROC_AUC"]
        })

loss_comparison = pd.DataFrame(comparison_rows)

loss_summary = (
    loss_comparison
    .groupby(["model", "split"])
    .agg(
        AP_mean=("AP", "mean"),
        AP_std=("AP", "std"),
        ROC_mean=("ROC_AUC", "mean")
    )
    .reset_index()
)

print("Loss function comparison:")
display(loss_summary.round(4))

loss_comparison.to_csv(
    NOVELTY / "results" /
    "experiment4_loss_comparison_runs.csv",
    index=False
)

loss_summary.to_csv(
    NOVELTY / "results" /
    "experiment4_loss_comparison_summary.csv",
    index=False
)

# ---------- Original notebook cell 92 ----------
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score

focal_warning_rows = []

for seed in [42, 123, 2026]:

    folder = Path(focal_run_folders[seed])

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

    warning_metrics, episodes = evaluate_warnings(
        test, threshold
    )

    alerts = test["probability"] >= threshold
    negatives = test["label"] == 0
    positives = test["label"] == 1

    focal_warning_rows.append({
        "feature_group": "combined_focal_gamma2",
        "seed": seed,
        "test_AP": average_precision_score(
            test["label"], test["probability"]
        ),
        "test_FPR": float(alerts[negatives].mean()),
        "test_TPR": float(alerts[positives].mean()),
        **warning_metrics
    })

focal_warnings = pd.DataFrame(focal_warning_rows)

focal_warnings.to_csv(
    NOVELTY / "results" /
    "experiment4_focal_warning_runs.csv",
    index=False
)

print("Focal-loss warning results:")
display(focal_warnings.round(4))

# ---------- Original notebook cell 93 ----------
# Original combined LSTM warning results
bce_warnings = matched_results[
    matched_results["feature_group"] == "combined"
].copy()

# Combine both models
loss_warning_comparison = pd.concat(
    [bce_warnings, focal_warnings],
    ignore_index=True
)

loss_warning_summary = (
    loss_warning_comparison
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

display(loss_warning_summary.round(4))

loss_warning_summary.to_csv(
    NOVELTY / "results" /
    "experiment4_loss_warning_comparison.csv",
    index=False
)

# ---------- Original notebook cell 94 ----------
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftF1BCELoss(nn.Module):
    def __init__(self, bce_weight=0.5, eps=1e-7):
        super().__init__()
        self.bce_weight = bce_weight
        self.eps = eps

    def forward(self, logits, targets):
        logits = logits.reshape(-1)
        targets = targets.reshape(-1).float()

        probabilities = torch.sigmoid(logits)

        tp = (probabilities * targets).sum()
        fp = (probabilities * (1 - targets)).sum()
        fn = ((1 - probabilities) * targets).sum()

        soft_f1 = (
            2 * tp + self.eps
        ) / (
            2 * tp + fp + fn + self.eps
        )

        bce = F.binary_cross_entropy_with_logits(
            logits, targets
        )

        return (
            self.bce_weight * bce
            + (1 - self.bce_weight) * (1 - soft_f1)
        )


def make_experiment_loss(loss_name, pos_weight, device):

    if loss_name == "unweighted_bce":
        return nn.BCEWithLogitsLoss()

    if loss_name == "sqrt_weighted_bce":
        return nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                pos_weight ** 0.5,
                dtype=torch.float32,
                device=device
            )
        )

    if loss_name == "soft_f1_bce":
        return SoftF1BCELoss(bce_weight=0.5)

    raise ValueError(f"Unknown loss: {loss_name}")

# ---------- Original notebook cell 95 ----------
import copy
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset


def train_alternative_loss(
    df, columns, loss_name, seed, device, epochs=25
):

    dl.set_seed(seed)

    folder = dl.new_run_folder(
        f"combined_{loss_name}_lstm"
    )

    indices = dl.split_positions(
        len(df), dl.HORIZON, dl.SEQUENCE_LENGTH
    )

    y = df["crash"].to_numpy(dtype=int)

    assert str(df.index[indices["validation"][0]].date()) == "2016-05-13"
    assert str(df.index[indices["test"][0]].date()) == "2021-06-28"

    scaler = StandardScaler()
    scaler.fit(df.iloc[indices["train"]][columns])

    x = scaler.transform(df[columns]).astype("float32")

    joblib.dump(scaler, folder / "scaler.joblib")

    sequences = {
        part: dl.make_sequences(x, ids, dl.SEQUENCE_LENGTH)
        for part, ids in indices.items()
    }

    labels = {
        part: torch.tensor(y[ids], dtype=torch.float32)
        for part, ids in indices.items()
    }

    model = dl.make_model(
        "LSTM", len(columns), dl.SEQUENCE_LENGTH
    ).to(device)

    positive_count = labels["train"].sum().item()
    negative_count = len(labels["train"]) - positive_count

    assert positive_count > 0

    pos_weight = negative_count / positive_count

    loss_fn = make_experiment_loss(
        loss_name, pos_weight, device
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    loader = DataLoader(
        TensorDataset(
            sequences["train"],
            labels["train"]
        ),
        batch_size=128,
        shuffle=True
    )

    dl.save_settings(folder, {
        "architecture": "LSTM",
        "feature_set": f"combined_{loss_name}",
        "feature_columns": list(columns),
        "loss": loss_name,
        "seed": seed,
        "epochs_allowed": epochs,
        "patience": dl.PATIENCE,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 128,
        "sequence_length": dl.SEQUENCE_LENGTH,
        "horizon": dl.HORIZON,
        "crash_threshold": 0.10,
        "original_positive_class_weight": pos_weight,
        "train_end_date": str(
            df.index[indices["train"][-1]].date()
        ),
        "validation_start_date": str(
            df.index[indices["validation"][0]].date()
        ),
        "test_start_date": str(
            df.index[indices["test"][0]].date()
        ),
        "device": device
    })

    best_ap = -1
    best_weights = None
    best_epoch = None
    patience_count = 0
    history = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):

        model.train()
        total_loss = 0.0

        for batch_x, batch_y in loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits = model(batch_x)

            loss = loss_fn(logits, batch_y)

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()

            total_loss += loss.item() * len(batch_y)

        probabilities = dl.predict(
            model, sequences["validation"], device
        )

        val_ap = average_precision_score(
            y[indices["validation"]],
            probabilities
        )

        history.append({
            "epoch": epoch,
            "train_loss": total_loss / len(labels["train"]),
            "validation_PR_AUC": val_ap
        })

        print(
            loss_name,
            "seed", seed,
            "epoch", epoch,
            "validation AP", round(val_ap, 4),
            flush=True
        )

        if val_ap > best_ap + 1e-5:
            best_ap = val_ap
            best_epoch = epoch
            best_weights = copy.deepcopy(
                model.state_dict()
            )
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= dl.PATIENCE:
            break

    model.load_state_dict(best_weights)

    torch.save({
        "state_dict": model.state_dict(),
        "architecture": "LSTM",
        "input_size": len(columns),
        "sequence_length": dl.SEQUENCE_LENGTH,
        "feature_columns": list(columns),
        "loss": loss_name
    }, folder / "model.pt")

    pd.DataFrame(history).to_csv(
        folder / "training_history.csv",
        index=False
    )

    rows = []

    for part in ["validation", "test"]:

        ids = indices[part]

        probabilities = dl.predict(
            model, sequences[part], device
        )

        rows.append({
            "split": part,
            "best_epoch": best_epoch,
            "training_seconds": round(
                time.time() - start_time, 1
            ),
            **dl.score_predictions(y[ids], probabilities)
        })

        pd.DataFrame({
            "Date": df.index[ids],
            "label": y[ids],
            "probability": probabilities
        }).to_csv(
            folder / f"{part}_predictions.csv",
            index=False
        )

    pd.DataFrame(rows).to_csv(
        folder / "metrics.csv",
        index=False
    )

    print("Saved:", folder, flush=True)

    return folder

# ---------- Original notebook cell 96 ----------
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

assert len(combined) == 48
assert dl.new_run_folder is novelty_run_folder

loss_names = [
    "unweighted_bce",
    "sqrt_weighted_bce",
    "soft_f1_bce"
]

seeds = [42, 123, 2026]

alternative_loss_runs = {}

for loss_name in loss_names:

    for seed in seeds:

        print("\n" + "=" * 65)
        print("LOSS:", loss_name, "| SEED:", seed)
        print("=" * 65)

        folder = train_alternative_loss(
            df=df,
            columns=list(combined),
            loss_name=loss_name,
            seed=seed,
            device=device,
            epochs=25
        )

        alternative_loss_runs[(loss_name, seed)] = folder

print("\nAll nine alternative-loss experiments completed.")

# ---------- Original notebook cell 97 ----------
import pandas as pd

all_loss_rows = []

# Original weighted BCE
baseline = run_manifest[
    run_manifest["feature_group"] == "combined"
]

for _, item in baseline.iterrows():

    metrics = pd.read_csv(item["metrics_path"])

    for _, r in metrics.iterrows():
        all_loss_rows.append({
            "loss": "weighted_bce",
            "seed": int(item["seed"]),
            "split": r["split"],
            "AP": r["PR_AUC"],
            "ROC_AUC": r["ROC_AUC"]
        })

# Previously trained focal loss
for seed, folder in focal_run_folders.items():

    metrics = pd.read_csv(folder / "metrics.csv")

    for _, r in metrics.iterrows():
        all_loss_rows.append({
            "loss": "focal_gamma2",
            "seed": seed,
            "split": r["split"],
            "AP": r["PR_AUC"],
            "ROC_AUC": r["ROC_AUC"]
        })

# Three new losses
for (loss_name, seed), folder in alternative_loss_runs.items():

    metrics = pd.read_csv(folder / "metrics.csv")

    for _, r in metrics.iterrows():
        all_loss_rows.append({
            "loss": loss_name,
            "seed": seed,
            "split": r["split"],
            "AP": r["PR_AUC"],
            "ROC_AUC": r["ROC_AUC"]
        })

all_loss_results = pd.DataFrame(all_loss_rows)

all_loss_summary = (
    all_loss_results
    .groupby(["loss", "split"])
    .agg(
        AP_mean=("AP", "mean"),
        AP_std=("AP", "std"),
        ROC_mean=("ROC_AUC", "mean")
    )
    .reset_index()
)

print("All loss-function comparisons:")
display(all_loss_summary.round(4))

all_loss_results.to_csv(
    NOVELTY / "results" /
    "experiment5_all_loss_runs.csv",
    index=False
)

all_loss_summary.to_csv(
    NOVELTY / "results" /
    "experiment5_all_loss_summary.csv",
    index=False
)

# ---------- Original notebook cell 98 ----------
import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    def __init__(self, pos_weight, gamma_pos=0.0, gamma_neg=2.0):
        super().__init__()
        self.register_buffer(
            "pos_weight",
            torch.tensor(float(pos_weight), dtype=torch.float32)
        )
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg

    def forward(self, logits, targets):
        logits = logits.reshape(-1)
        targets = targets.reshape(-1).float()

        p = torch.sigmoid(logits)

        positive_loss = (
            self.pos_weight
            * targets
            * (1 - p).pow(self.gamma_pos)
            * F.softplus(-logits)
        )

        negative_loss = (
            (1 - targets)
            * p.pow(self.gamma_neg)
            * F.softplus(logits)
        )

        return (positive_loss + negative_loss).mean()


class BCEPairwiseRankingLoss(nn.Module):
    def __init__(self, pos_weight, ranking_weight=0.5):
        super().__init__()
        self.register_buffer(
            "pos_weight",
            torch.tensor(float(pos_weight), dtype=torch.float32)
        )
        self.ranking_weight = ranking_weight

    def forward(self, logits, targets):
        logits = logits.reshape(-1)
        targets = targets.reshape(-1).float()

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight
        )

        positive_logits = logits[targets == 1]
        negative_logits = logits[targets == 0]

        if len(positive_logits) == 0 or len(negative_logits) == 0:
            return bce

        differences = (
            positive_logits[:, None]
            - negative_logits[None, :]
        )

        ranking_loss = F.softplus(-differences).mean()

        return (
            (1 - self.ranking_weight) * bce
            + self.ranking_weight * ranking_loss
        )


class ClassBalancedBCELoss(nn.Module):
    def __init__(self, n_pos, n_neg, beta=0.999):
        super().__init__()

        def effective_weight(n):
            return (1 - beta) / (1 - beta ** n)

        w_pos = effective_weight(n_pos)
        w_neg = effective_weight(n_neg)

        # Normalize negative weight to 1
        relative_pos_weight = w_pos / w_neg

        self.register_buffer(
            "pos_weight",
            torch.tensor(
                float(relative_pos_weight),
                dtype=torch.float32
            )
        )

    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(
            logits.reshape(-1),
            targets.reshape(-1).float(),
            pos_weight=self.pos_weight
        )


def make_advanced_loss(loss_name, n_pos, n_neg, device):

    pos_weight = n_neg / n_pos

    if loss_name == "asymmetric":
        return AsymmetricLoss(
            pos_weight=pos_weight,
            gamma_pos=0.0,
            gamma_neg=2.0
        ).to(device)

    if loss_name == "bce_pairwise":
        return BCEPairwiseRankingLoss(
            pos_weight=pos_weight,
            ranking_weight=0.5
        ).to(device)

    if loss_name == "class_balanced":
        return ClassBalancedBCELoss(
            n_pos=n_pos,
            n_neg=n_neg,
            beta=0.999
        ).to(device)

    raise ValueError(f"Unknown loss: {loss_name}")

# ---------- Original notebook cell 99 ----------
import copy
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset


def train_advanced_loss(
    df, columns, loss_name, seed, device, epochs=25
):

    dl.set_seed(seed)

    folder = dl.new_run_folder(
        f"combined_{loss_name}_lstm"
    )

    indices = dl.split_positions(
        len(df), dl.HORIZON, dl.SEQUENCE_LENGTH
    )

    y = df["crash"].to_numpy(dtype=int)

    assert str(df.index[indices["validation"][0]].date()) == "2016-05-13"
    assert str(df.index[indices["test"][0]].date()) == "2021-06-28"

    scaler = StandardScaler()
    scaler.fit(df.iloc[indices["train"]][columns])

    x = scaler.transform(df[columns]).astype("float32")

    joblib.dump(scaler, folder / "scaler.joblib")

    sequences = {
        part: dl.make_sequences(
            x, ids, dl.SEQUENCE_LENGTH
        )
        for part, ids in indices.items()
    }

    labels = {
        part: torch.tensor(
            y[ids], dtype=torch.float32
        )
        for part, ids in indices.items()
    }

    model = dl.make_model(
        "LSTM",
        len(columns),
        dl.SEQUENCE_LENGTH
    ).to(device)

    n_pos = int(labels["train"].sum().item())
    n_neg = len(labels["train"]) - n_pos

    loss_fn = make_advanced_loss(
        loss_name, n_pos, n_neg, device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.0001
    )

    loader = DataLoader(
        TensorDataset(
            sequences["train"],
            labels["train"]
        ),
        batch_size=128,
        shuffle=True
    )

    dl.save_settings(folder, {
        "architecture": "LSTM",
        "feature_set": f"combined_{loss_name}",
        "feature_columns": list(columns),
        "loss": loss_name,
        "seed": seed,
        "epochs_allowed": epochs,
        "patience": dl.PATIENCE,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "batch_size": 128,
        "sequence_length": dl.SEQUENCE_LENGTH,
        "horizon": dl.HORIZON,
        "crash_threshold": 0.10,
        "train_end_date": str(
            df.index[indices["train"][-1]].date()
        ),
        "validation_start_date": str(
            df.index[indices["validation"][0]].date()
        ),
        "test_start_date": str(
            df.index[indices["test"][0]].date()
        ),
        "device": device,
        "loss_parameters": {
            "asymmetric": {
                "gamma_pos": 0.0,
                "gamma_neg": 2.0,
                "pos_weight": n_neg / n_pos
            },
            "bce_pairwise": {
                "ranking_weight": 0.5,
                "pos_weight": n_neg / n_pos
            },
            "class_balanced": {
                "beta": 0.999
            }
        }[loss_name]
    })

    best_ap = -1
    best_weights = None
    best_epoch = None
    patience_count = 0
    history = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):

        model.train()
        total_loss = 0.0

        for batch_x, batch_y in loader:

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            logits = model(batch_x)

            loss = loss_fn(logits, batch_y)

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )

            optimizer.step()

            total_loss += loss.item() * len(batch_y)

        probabilities = dl.predict(
            model,
            sequences["validation"],
            device
        )

        val_ap = average_precision_score(
            y[indices["validation"]],
            probabilities
        )

        history.append({
            "epoch": epoch,
            "train_loss": total_loss / len(labels["train"]),
            "validation_PR_AUC": val_ap
        })

        print(
            loss_name,
            "seed", seed,
            "epoch", epoch,
            "validation AP", round(val_ap, 4),
            flush=True
        )

        if val_ap > best_ap + 1e-5:
            best_ap = val_ap
            best_epoch = epoch
            best_weights = copy.deepcopy(
                model.state_dict()
            )
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= dl.PATIENCE:
            break

    model.load_state_dict(best_weights)

    torch.save({
        "state_dict": model.state_dict(),
        "architecture": "LSTM",
        "input_size": len(columns),
        "sequence_length": dl.SEQUENCE_LENGTH,
        "feature_columns": list(columns),
        "loss": loss_name
    }, folder / "model.pt")

    pd.DataFrame(history).to_csv(
        folder / "training_history.csv",
        index=False
    )

    rows = []

    for part in ["validation", "test"]:

        ids = indices[part]

        probabilities = dl.predict(
            model,
            sequences[part],
            device
        )

        rows.append({
            "split": part,
            "best_epoch": best_epoch,
            "training_seconds": round(
                time.time() - start_time, 1
            ),
            **dl.score_predictions(
                y[ids], probabilities
            )
        })

        pd.DataFrame({
            "Date": df.index[ids],
            "label": y[ids],
            "probability": probabilities
        }).to_csv(
            folder / f"{part}_predictions.csv",
            index=False
        )

    pd.DataFrame(rows).to_csv(
        folder / "metrics.csv",
        index=False
    )

    print("Saved:", folder, flush=True)

    return folder

# ---------- Original notebook cell 100 ----------
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

assert len(combined) == 48
assert dl.new_run_folder is novelty_run_folder

advanced_loss_names = [
    "asymmetric",
    "bce_pairwise",
    "class_balanced"
]

advanced_loss_runs = {}

for loss_name in advanced_loss_names:

    for seed in [42, 123, 2026]:

        print("\n" + "=" * 65)
        print("ADVANCED LOSS:", loss_name, "| SEED:", seed)
        print("=" * 65)

        folder = train_advanced_loss(
            df=df,
            columns=list(combined),
            loss_name=loss_name,
            seed=seed,
            device=device,
            epochs=25
        )

        advanced_loss_runs[(loss_name, seed)] = folder

print("\nAll nine advanced-loss experiments completed.")

# ---------- Original notebook cell 101 ----------
comparison_rows = []

# Previous five losses
for _, r in all_loss_results.iterrows():
    comparison_rows.append({
        "loss": r["loss"],
        "seed": int(r["seed"]),
        "split": r["split"],
        "AP": r["AP"],
        "ROC_AUC": r["ROC_AUC"]
    })

# Three advanced losses
for (loss_name, seed), folder in advanced_loss_runs.items():

    metrics = pd.read_csv(folder / "metrics.csv")

    for _, r in metrics.iterrows():
        comparison_rows.append({
            "loss": loss_name,
            "seed": seed,
            "split": r["split"],
            "AP": r["PR_AUC"],
            "ROC_AUC": r["ROC_AUC"]
        })

eight_loss_results = pd.DataFrame(comparison_rows)

eight_loss_summary = (
    eight_loss_results
    .groupby(["loss", "split"])
    .agg(
        AP_mean=("AP", "mean"),
        AP_std=("AP", "std"),
        ROC_mean=("ROC_AUC", "mean")
    )
    .reset_index()
)

display(eight_loss_summary.round(4))

eight_loss_results.to_csv(
    NOVELTY / "results" /
    "experiment6_eight_loss_runs.csv",
    index=False
)

eight_loss_summary.to_csv(
    NOVELTY / "results" /
    "experiment6_eight_loss_summary.csv",
    index=False
)

# ---------- Original notebook cell 102 ----------
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score

advanced_warning_rows = []

for (loss_name, seed), folder in advanced_loss_runs.items():

    folder = Path(folder)

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

    warning_metrics, episodes = evaluate_warnings(
        test, threshold
    )

    alerts = test["probability"] >= threshold
    negatives = test["label"] == 0
    positives = test["label"] == 1

    advanced_warning_rows.append({
        "loss": loss_name,
        "seed": seed,
        "test_AP": average_precision_score(
            test["label"], test["probability"]
        ),
        "test_FPR": float(alerts[negatives].mean()),
        "test_TPR": float(alerts[positives].mean()),
        **warning_metrics
    })

advanced_warnings = pd.DataFrame(advanced_warning_rows)

advanced_warnings.to_csv(
    NOVELTY / "results" /
    "experiment6_advanced_warning_runs.csv",
    index=False
)

display(advanced_warnings.round(4))

# ---------- Original notebook cell 103 ----------
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score



def rebuild_loss_warnings(run_dict):

    rows = []

    for (loss_name, seed), folder in run_dict.items():

        folder = Path(folder)

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

        warning_metrics, _ = evaluate_warnings(
            test, threshold
        )

        alerts = test["probability"] >= threshold

        negatives = test["label"] == 0
        positives = test["label"] == 1

        rows.append({
            "loss": loss_name,
            "seed": seed,
            "test_AP": average_precision_score(
                test["label"],
                test["probability"]
            ),
            "test_FPR": float(alerts[negatives].mean()),
            "test_TPR": float(alerts[positives].mean()),
            **warning_metrics
        })

    return pd.DataFrame(rows)




baseline_warning = matched_results[
    matched_results["feature_group"] == "combined"
].copy()

baseline_warning["loss"] = "weighted_bce"




focal_warning = focal_warnings.copy()
focal_warning["loss"] = "focal_gamma2"




if "new_loss_warnings" not in globals():

    print("Reconstructing Experiment 5 warning results...")

    new_loss_warnings = rebuild_loss_warnings(
        alternative_loss_runs
    )

previous_warning = new_loss_warnings.copy()




if "advanced_warnings" not in globals():

    print("Reconstructing Experiment 6 warning results...")

    advanced_warnings = rebuild_loss_warnings(
        advanced_loss_runs
    )




eight_loss_warnings = pd.concat(
    [
        baseline_warning,
        focal_warning,
        previous_warning,
        advanced_warnings
    ],
    ignore_index=True
)




eight_loss_warning_summary = (
    eight_loss_warnings
    .groupby("loss")
    .agg(
        AP_mean=("test_AP", "mean"),
        FPR_mean=("test_FPR", "mean"),
        TPR_mean=("test_TPR", "mean"),
        detection_mean=("label_episode_recall", "mean"),
        false_alerts_mean=(
            "false_alert_episodes", "mean"
        ),
        lead_mean=("mean_lead_days", "mean"),
        alert_days_mean=("alert_days", "mean")
    )
    .reset_index()
    .sort_values("AP_mean", ascending=False)
)




eight_loss_warnings.to_csv(
    NOVELTY / "results" /
    "experiment6_all_warning_runs.csv",
    index=False
)

eight_loss_warning_summary.to_csv(
    NOVELTY / "results" /
    "experiment6_all_warning_summary.csv",
    index=False
)




print("Complete warning comparison — eight losses:")

display(eight_loss_warning_summary.round(4))

print("\nNumber of runs:", len(eight_loss_warnings))

assert len(eight_loss_warnings) == 24, (
    "Expected 8 losses × 3 seeds = 24 runs."
)

# ---------- Original notebook cell 104 ----------
paired_ap = (
    eight_loss_results[
        eight_loss_results["split"] == "test"
    ]
    .pivot(
        index="seed",
        columns="loss",
        values="AP"
    )
)

selected_losses = [
    "weighted_bce",
    "bce_pairwise",
    "asymmetric"
]

print("Test AP by seed:")
display(paired_ap[selected_losses].round(4))

paired_differences = pd.DataFrame({
    "seed": paired_ap.index,
    "ranking_minus_BCE": (
        paired_ap["bce_pairwise"]
        - paired_ap["weighted_bce"]
    ).values,
    "asymmetric_minus_BCE": (
        paired_ap["asymmetric"]
        - paired_ap["weighted_bce"]
    ).values
})

print("\nPaired differences:")
display(paired_differences.round(4))

paired_differences.to_csv(
    NOVELTY / "results" /
    "experiment6_paired_seed_differences.csv",
    index=False
)
