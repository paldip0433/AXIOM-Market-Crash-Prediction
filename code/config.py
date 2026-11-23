from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'
EXPERIMENTS = ROOT / 'experiments'
FIGURES = ROOT / 'figures'
HORIZON = 30
CRASH_THRESHOLD = 0.10
TDA_WINDOWS = [40, 80, 120]
TDA_LAYERS = 5
TDA_GRID_POINTS = 160
SEQUENCE_LENGTH = 30
EPOCHS = 25
PATIENCE = 5
SEEDS = [42, 123, 2026]
SEED = 42
