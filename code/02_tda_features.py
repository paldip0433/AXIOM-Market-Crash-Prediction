"""Persistence diagrams and persistence landscapes of rolling 4D market returns."""
import argparse
import time
import numpy as np
import pandas as pd
from ripser import ripser
from config import PROCESSED, TDA_WINDOWS, TDA_LAYERS, TDA_GRID_POINTS


def landscape(diagram, x, layers=5):
    # Every (birth, death) pair makes a triangle, with height (death-birth)/2.
    # At each x position, landscape 1 is the tallest triangle, etc.
    pairs = np.asarray(diagram)
    if len(pairs) == 0:
        return np.zeros((layers, len(x)))
    pairs = pairs[np.isfinite(pairs).all(axis=1)]
    if len(pairs) == 0:
        return np.zeros((layers, len(x)))
    values = np.maximum(0, np.minimum(x[None, :] - pairs[:, 0, None],
                                  pairs[:, 1, None] - x[None, :]))
    values.sort(axis=0)
    answer = np.zeros((layers, len(x)))
    count = min(layers, len(values))
    answer[:count] = values[-count:][::-1]
    return answer


def diagram_summary(diagram, prefix, layers, points):
    pairs = np.asarray(diagram)
    pairs = pairs[np.isfinite(pairs).all(axis=1)] if len(pairs) else np.empty((0, 2))
    lifetimes = pairs[:, 1] - pairs[:, 0] if len(pairs) else np.array([])
    xmax = float(pairs[:, 1].max()) if len(pairs) else 1.0
    grid = np.linspace(0, max(xmax, 1e-8), points)
    land = landscape(pairs, grid, layers)
    return {
        prefix + '_count': len(pairs),
        prefix + '_total_persistence': float(lifetimes.sum()),
        prefix + '_max_persistence': float(lifetimes.max()) if len(lifetimes) else 0.0,
        prefix + '_L1': float(np.trapezoid(np.abs(land).sum(axis=0), grid)),
        prefix + '_L2': float(np.sqrt(np.trapezoid((land ** 2).sum(axis=0), grid)))
    }


def landscape_change(current, previous, layers, points):
    # L2 distance between landscapes from consecutive windows.
    if previous is None:
        return np.nan
    a = np.asarray(current)
    b = np.asarray(previous)
    a = a[np.isfinite(a).all(axis=1)] if len(a) else np.empty((0, 2))
    b = b[np.isfinite(b).all(axis=1)] if len(b) else np.empty((0, 2))
    xmax = max(float(a[:, 1].max()) if len(a) else 0,
               float(b[:, 1].max()) if len(b) else 0, 1e-8)
    grid = np.linspace(0, xmax, points)
    difference = landscape(a, grid, layers) - landscape(b, grid, layers)
    return float(np.sqrt(np.trapezoid((difference ** 2).sum(axis=0), grid)))


def compute_window(returns, window, max_rows=None):
    save_file = PROCESSED / f'tda_w{window}.csv'

    if max_rows is not None:
        save_file = PROCESSED / f'tda_w{window}_trial.csv'
    old = pd.read_csv(save_file, index_col='Date', parse_dates=True) if save_file.exists() else pd.DataFrame()
    old.index.name = 'Date'
    if len(old):
        print('Resuming', save_file.name, 'from', len(old), 'saved rows', flush=True)

    dates = returns.index[window - 1:]
    if max_rows is not None:
        dates = dates[:max_rows]
    old = old.loc[old.index.intersection(dates)] if len(old) else old
    finished = set(old.index)


    if len(old) > 0:
        saved_dates = list(old.index)
        expected_dates = list(dates[:len(saved_dates)])

        if saved_dates != expected_dates:
            raise ValueError(
                f"Saved TDA dates are not continuous for window {window}. "
                "Please check the CSV before continuing."
            )

    rows = []
    previous_h1 = None
    start = time.time()
    for number, date in enumerate(dates):
        position = returns.index.get_loc(date)
        cloud = returns.iloc[position - window + 1:position + 1].to_numpy(dtype=float) * 100

        if date in finished:

            if number + 1 < len(dates) and dates[number + 1] not in finished:
                previous_h1 = ripser(cloud, maxdim=1)['dgms'][1]
            continue
        diagrams = ripser(cloud, maxdim=1)['dgms']
        row = {'Date': date}
        row.update(diagram_summary(diagrams[0], f'H0_w{window}', TDA_LAYERS, TDA_GRID_POINTS))
        row.update(diagram_summary(diagrams[1], f'H1_w{window}', TDA_LAYERS, TDA_GRID_POINTS))
        row[f'H1_w{window}_landscape_change'] = landscape_change(
            diagrams[1], previous_h1, TDA_LAYERS, TDA_GRID_POINTS)
        previous_h1 = diagrams[1]
        rows.append(row)
        if len(rows) % 100 == 0:
            combined = pd.concat([old, pd.DataFrame(rows).set_index('Date')]).sort_index()
            combined.index.name = 'Date'
            combined.to_csv(save_file)
            print('window', window, 'completed', len(combined), '/', len(dates),
                  'elapsed minutes', round((time.time()-start)/60, 1), flush=True)
    if rows:
        old = pd.concat([old, pd.DataFrame(rows).set_index('Date')]).sort_index()
    old.index.name = 'Date'
    old.to_csv(save_file)
    print('Saved:', save_file, 'rows:', len(old), flush=True)
    return old


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial', type=int, default=None,
                        help='Compute first N windows in separate trial files')
    parser.add_argument('--windows', nargs='+', type=int, default=TDA_WINDOWS)
    args = parser.parse_args()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    returns = pd.read_csv(PROCESSED / 'log_returns.csv', index_col='Date', parse_dates=True).dropna()
    parts = []
    for window in args.windows:
        part = compute_window(returns, window, args.trial)
        parts.append(part)
    if args.trial is not None:
        print('Trial finished. Full features were NOT replaced.')
        return
    classical = pd.read_csv(PROCESSED / 'features_classical.csv', index_col='Date', parse_dates=True)
    full = classical.join(parts, how='inner').replace([np.inf, -np.inf], np.nan).dropna()
    full.to_csv(PROCESSED / 'features_combined.csv')
    print('Combined dataset:', len(full), 'rows,', len(full.columns), 'columns')
    print('Positive fraction:', round(full.crash.mean(), 4))


if __name__ == '__main__':
    main()
