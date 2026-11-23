"""Train the LSTM, bidirectional LSTM and Transformer with different seeds."""
import argparse
import copy
import time
import json
import numpy as np
import pandas as pd
import torch
import joblib
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from config import PROCESSED, HORIZON, SEQUENCE_LENGTH, EPOCHS, PATIENCE, SEEDS
from models import make_model
from utils import new_run_folder, save_settings, score_predictions, split_positions, set_seed


def make_sequences(data, indices, length):
    windows = [data[i-length+1:i+1] for i in indices]
    return torch.tensor(np.stack(windows), dtype=torch.float32)


def predict(model, sequences, device):
    model.eval()
    result = []
    with torch.no_grad():
        for start in range(0, len(sequences), 256):
            batch = sequences[start:start+256].to(device)
            result.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(result)


def train_one(df, feature_set, columns, architecture, seed, device, epochs):
    set_seed(seed)
    folder = new_run_folder(feature_set + '_' + architecture.lower())
    indices = split_positions(len(df), HORIZON, SEQUENCE_LENGTH)
    y = df['crash'].to_numpy(dtype=int)

    scaler = StandardScaler()
    scaler.fit(df.iloc[indices['train']][columns])
    x = scaler.transform(df[columns]).astype('float32')
    joblib.dump(scaler, folder / 'scaler.joblib')

    sequences = {part: make_sequences(x, ids, SEQUENCE_LENGTH)
                 for part, ids in indices.items()}
    labels = {part: torch.tensor(y[ids], dtype=torch.float32)
              for part, ids in indices.items()}

    model = make_model(architecture, len(columns), SEQUENCE_LENGTH).to(device)
    positive_count = labels['train'].sum().item()
    negative_count = len(labels['train']) - positive_count
    if positive_count == 0:
        raise ValueError('No positive crash labels in training data.')
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negative_count / positive_count, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    loader = DataLoader(TensorDataset(sequences['train'], labels['train']),
                        batch_size=128, shuffle=True)
    save_settings(folder, {
        'architecture': architecture, 'feature_set': feature_set,
        'feature_columns': columns, 'seed': seed,
        'epochs_allowed': epochs, 'patience': PATIENCE,
        'learning_rate': 0.001, 'batch_size': 128,
        'sequence_length': SEQUENCE_LENGTH, 'horizon': HORIZON,
        'crash_threshold': 0.10,
        'train_end_date': str(df.index[indices['train'][-1]].date()),
        'validation_start_date': str(df.index[indices['validation'][0]].date()),
        'test_start_date': str(df.index[indices['test'][0]].date()),
        'device': device,
    })
    best_ap = -1
    best_weights = None
    best_epoch = None
    patience_count = 0
    history = []
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        loss_total = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_total += loss.item() * len(batch_y)
        probabilities = predict(model, sequences['validation'], device)
        val_ap = average_precision_score(y[indices['validation']], probabilities)
        history.append({'epoch': epoch, 'train_loss': loss_total / len(labels['train']),
                        'validation_PR_AUC': val_ap})
        print(feature_set, architecture, 'seed', seed,
              'epoch', epoch, 'validation PR-AUC', round(val_ap, 4), flush=True)
        if val_ap > best_ap + 1e-5:
            best_ap = val_ap
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= PATIENCE:
            break

    model.load_state_dict(best_weights)
    torch.save({'state_dict': model.state_dict(), 'architecture': architecture,
                'input_size': len(columns), 'sequence_length': SEQUENCE_LENGTH,
                'feature_columns': columns}, folder / 'model.pt')
    pd.DataFrame(history).to_csv(folder / 'training_history.csv', index=False)
    rows = []
    for part in ['validation', 'test']:
        ids = indices[part]
        probabilities = predict(model, sequences[part], device)
        rows.append({'split': part, 'best_epoch': best_epoch,
                     'training_seconds': round(time.time() - start_time, 1),
                     **score_predictions(y[ids], probabilities)})
        pd.DataFrame({'Date': df.index[ids], 'label': y[ids],
                      'probability': probabilities}).to_csv(
                          folder / (part + '_predictions.csv'), index=False)
    pd.DataFrame(rows).to_csv(folder / 'metrics.csv', index=False)
    print('Saved', folder, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    parser.add_argument('--models', nargs='+', default=['LSTM', 'BiLSTM', 'Transformer'])
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    args = parser.parse_args()
    df = pd.read_csv(PROCESSED / 'features_combined.csv', index_col='Date', parse_dates=True)
    all_columns = [c for c in df if c not in ['crash', 'future_drawdown']]
    tda_columns = [
    c for c in all_columns
    if c.startswith(('H0_', 'H1_'))
    and not (c.startswith('H0_') and c.endswith('_count'))
]
    classical_columns = [
    c for c in all_columns
    if not c.startswith(('H0_', 'H1_'))
]

    groups = {
    'classical': classical_columns,
    'tda': tda_columns,
    'combined': classical_columns + tda_columns
}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Device:', device, 'seeds:', args.seeds)
    for feature_set, columns in groups.items():
        for architecture in args.models:
            for seed in args.seeds:
                train_one(df, feature_set, columns, architecture, seed, device, args.epochs)

if __name__ == '__main__':
    main()
