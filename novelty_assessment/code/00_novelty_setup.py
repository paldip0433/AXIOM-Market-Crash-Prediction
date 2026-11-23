# AXIOM — novelty assessment: 00_novelty_setup.py
# Extracted verbatim from notebook code cells 49–58 (zero-based).
# These scripts retain notebook-global dependencies and should be run in order
# within the same initialized Colab/Python session (e.g. exec(open(path).read())).
# Original AXIOM preprocessing and model modules must already be available.
# This is an archival extraction, not a standalone refactor.


# ---------- Original notebook cell 49 ----------
from google.colab import drive
from pathlib import Path
import os

drive.mount('/content/drive')

PROJECT = Path(
    '/content/drive/MyDrive/'
    'AXIOM_IE643_CourseProject_IntensiveAssessment'
)

NOVELTY = PROJECT / 'novelty_assessment'

for folder in [
    'code',
    'experiments',
    'results',
    'figures',
    'poster',
    'notes'
]:
    (NOVELTY / folder).mkdir(
        parents=True,
        exist_ok=True
    )

os.chdir(PROJECT)

print("Original project:", PROJECT)
print("Novelty experiments:", NOVELTY)

# ---------- Original notebook cell 50 ----------
import pandas as pd
import numpy as np

DATA_FILE = PROJECT / 'data/processed/features_combined.csv'

df = pd.read_csv(
    DATA_FILE,
    index_col='Date',
    parse_dates=True
).sort_index()

all_features = [
    c for c in df.columns
    if c not in ['crash', 'future_drawdown']
]

classical = [
    c for c in all_features
    if not c.startswith(('H0_', 'H1_'))
]

tda = [
    c for c in all_features
    if c.startswith(('H0_', 'H1_'))
    and not (
        c.startswith('H0_')
        and c.endswith('_count')
    )
]

h1_change = [
    c for c in tda
    if c.startswith('H1_')
    and c.endswith('_landscape_change')
]

combined = classical + tda

print("Observations:", len(df))
print("Date range:", df.index.min(), df.index.max())
print("Classical features:", len(classical))
print("TDA features:", len(tda))
print("Combined features:", len(combined))

print("\nExisting H1 landscape-change features:")
for c in h1_change:
    print(" ", c)

print("\nPositive crash rate:", df['crash'].mean())

assert len(classical) == 18
assert len(tda) == 30
assert len(h1_change) == 3
assert len(combined) == 48
assert not df[combined + ['crash']].isna().any().any()

# ---------- Original notebook cell 51 ----------
feature_groups = {
    'classical': classical,

    'combined': combined,

    'combined_without_h1_change': [
        c for c in combined
        if c not in h1_change
    ],

    'classical_plus_h1_change': (
        classical + h1_change
    )
}

for name, columns in feature_groups.items():
    print(name, len(columns))

# ---------- Original notebook cell 52 ----------
import sys
import importlib
import torch

sys.path.insert(0, str(PROJECT / 'code'))

dl = importlib.import_module('04_deep_learning')

from config import HORIZON, SEQUENCE_LENGTH
from utils import split_positions


def novelty_run_folder(experiment_name):
    base = NOVELTY / 'experiments' / experiment_name
    base.mkdir(parents=True, exist_ok=True)

    number = 1

    while (base / f'run_{number:03d}').exists():
        number += 1

    folder = base / f'run_{number:03d}'
    folder.mkdir()
    return folder

dl.new_run_folder = novelty_run_folder

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Device:", device)

seeds = [42, 123, 2026]

for group_name, columns in feature_groups.items():

    print("\nExperiment:", group_name)

    for seed in seeds:

        print("Training seed:", seed)

        dl.train_one(
            df=df,
            feature_set=group_name,
            columns=columns,
            architecture='LSTM',
            seed=seed,
            device=device,
            epochs=25
        )

# ---------- Original notebook cell 53 ----------
import sys
import importlib
import inspect

sys.path.insert(0, str(PROJECT / "code"))

dl = importlib.import_module("04_deep_learning")

print("train_one signature:")
print(inspect.signature(dl.train_one))

print("\nExperiment folder function:")
print(inspect.getsource(dl.new_run_folder))

# ---------- Original notebook cell 54 ----------
for name, columns in feature_groups.items():
    print(f"{name}: {len(columns)} features")

print("\nH1 landscape-change columns:")
for column in h1_change:
    print(column)

# ---------- Original notebook cell 55 ----------
import torch

print("PyTorch:", torch.__version__)
print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ---------- Original notebook cell 56 ----------
import inspect

source = inspect.getsource(dl.train_one)

for line in source.splitlines():
    if any(word in line for word in [
        'new_run_folder',
        'EXPERIMENT',
        'run_folder',
        'output_dir',
        'save',
        'to_csv'
    ]):
        print(line)

# ---------- Original notebook cell 57 ----------
print(dl.new_run_folder.__name__)

# ---------- Original notebook cell 58 ----------
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("Using device:", device)

pilot_result = dl.train_one(
    df=df,
    feature_set='classical',
    columns=feature_groups['classical'],
    architecture='LSTM',
    seed=42,
    device=device,
    epochs=25
)

print("Pilot training completed.")
print("Returned result:", pilot_result)
