"""Collect results without changing old experiment folders."""
import json
import pandas as pd
from config import ROOT, EXPERIMENTS


def main():
    rows = []
    for path in sorted(EXPERIMENTS.glob('*/run_*/metrics.csv')):
        settings_file = path.parent / 'settings.json'
        if not settings_file.exists():
            continue
        settings = json.loads(settings_file.read_text())
        # Exclude 2-epoch debugging runs from the scientific summary.
        if settings.get('architecture') and settings.get('epochs_allowed', 0) < 25:
            continue
        metrics = pd.read_csv(path)
        for _, metric in metrics.iterrows():
            rows.append({'experiment': path.parent.parent.name,
                         'run': path.parent.name,
                         'run_folder': str(path.parent),
                         'seed': settings.get('seed'),
                         'architecture': settings.get('architecture', settings.get('model')),
                         'feature_set': settings.get('feature_set'),
                         **metric.to_dict()})
    if not rows:
        print('No completed full experiments yet.')
        return
    table = pd.DataFrame(rows)
    folder = ROOT / 'figures'
    folder.mkdir(exist_ok=True)
    table.to_csv(folder / 'all_experiment_results.csv', index=False)
    test = table[table.split == 'test']
    summary = test.groupby(['architecture', 'feature_set'], dropna=False).agg(
        runs=('PR_AUC', 'size'), mean_PR_AUC=('PR_AUC', 'mean'),
        sd_PR_AUC=('PR_AUC', 'std'), mean_ROC_AUC=('ROC_AUC', 'mean')
    ).reset_index()
    summary.to_csv(folder / 'test_mean_std.csv', index=False)
    print(summary.to_string(index=False))
    print('Seed means are descriptive, not a significance test.')


if __name__ == '__main__':
    main()
