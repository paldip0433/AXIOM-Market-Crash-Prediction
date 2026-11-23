"""Three logistic regression baselines using the same dates."""
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from config import PROCESSED, HORIZON, SEED
from utils import new_run_folder, save_settings, score_predictions, split_positions


def main():
    df = pd.read_csv(PROCESSED / 'features_combined.csv', index_col='Date', parse_dates=True)
    all_features = [c for c in df.columns if c not in ('crash', 'future_drawdown')]

    tda_features = [
    c for c in all_features
    if c.startswith(('H0_', 'H1_'))
    and not (c.startswith('H0_') and c.endswith('_count'))
]
    classical_features = [
    c for c in all_features
    if not c.startswith(('H0_', 'H1_'))
]

    groups = {
    'classical': classical_features,
    'tda': tda_features,
    'combined': classical_features + tda_features
}
    positions = split_positions(len(df), HORIZON)
    y = df['crash'].to_numpy()
    for name, fields in {
    'classical': classical_features,
    'tda': tda_features,
    'combined': classical_features + tda_features
}.items():
        folder = new_run_folder(name + '_logistic')
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED))
        model.fit(df.iloc[positions['train']][fields], y[positions['train']])
        joblib.dump(model, folder / 'model.joblib')
        save_settings(folder, {'model': 'logistic', 'feature_set': name, 'features': fields, 'seed': SEED,
                               'horizon': HORIZON, 'train_end': str(df.index[int(.7 * len(df))]),
                               'validation_end': str(df.index[int(.85 * len(df))])})
        rows = []
        for split in ('validation', 'test'):
            ids = positions[split]
            prob = model.predict_proba(df.iloc[ids][fields])[:, 1]
            row = {'split': split, **score_predictions(y[ids], prob)}
            rows.append(row)
            pd.DataFrame({'Date': df.index[ids], 'label': y[ids], 'probability': prob}).to_csv(folder / (split + '_predictions.csv'), index=False)
        pd.DataFrame(rows).to_csv(folder / 'metrics.csv', index=False)
        print(name, folder, rows, flush=True)


if __name__ == '__main__':
    main()
