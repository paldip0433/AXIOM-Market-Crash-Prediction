from pathlib import Path
import json
import random
import platform
import numpy as np


def new_run_folder(experiment_name):
    from config import EXPERIMENTS
    base = EXPERIMENTS / experiment_name
    base.mkdir(parents=True, exist_ok=True)
    number = 1
    while True:
        folder = base / f"run_{number:03d}"
        try:
            folder.mkdir()
            return folder
        except FileExistsError:
            number += 1


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_settings(folder, settings):
    settings = dict(settings)
    settings['python'] = platform.python_version()
    try:
        import torch
        settings['torch'] = torch.__version__
        settings['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    except ImportError:
        settings['gpu'] = 'not checked'
    (folder / 'settings.json').write_text(json.dumps(settings, indent=2, default=str))


def score_predictions(labels, probabilities):
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
    labels = np.asarray(labels).astype(int)
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= 0.5).astype(int)
    return {'n': len(labels), 'positives': int(labels.sum()),
            'PR_AUC': float(average_precision_score(labels, probabilities)),
            'ROC_AUC': float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
            'precision_05': float(precision_score(labels, predictions, zero_division=0)),
            'recall_05': float(recall_score(labels, predictions, zero_division=0)),
            'F1_05': float(f1_score(labels, predictions, zero_division=0))}


def split_positions(n, horizon, sequence_length=1):
    import numpy as np
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)
    return {'train': np.arange(sequence_length - 1, train_end - horizon),
            'validation': np.arange(train_end, valid_end - horizon),
            'test': np.arange(valid_end, n)}
