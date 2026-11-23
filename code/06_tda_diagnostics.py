"""Pictures and optional bottleneck-distance experiment for the report."""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ripser import ripser
from config import PROCESSED, FIGURES, TDA_WINDOWS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bottleneck', action='store_true')
    parser.add_argument('--step', type=int, default=20)
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    prices = pd.read_csv(PROCESSED / 'prices_aligned.csv', index_col='Date', parse_dates=True)
    returns = pd.read_csv(PROCESSED / 'log_returns.csv', index_col='Date', parse_dates=True).dropna()
    for w in TDA_WINDOWS:
        file = PROCESSED / f'tda_w{w}.csv'
        if not file.exists():
            continue
        df = pd.read_csv(file, index_col='Date', parse_dates=True)
        fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
        axes[0].plot(prices.index, prices.SP500)
        axes[0].set_ylabel('S&P 500 close')
        axes[1].plot(df.index, df[f'H1_w{w}_L1'], label='H1 landscape L1')
        axes[1].plot(df.index, df[f'H1_w{w}_L2'], label='H1 landscape L2')
        axes[1].legend()
        axes[2].plot(df.index, df[f'H1_w{w}_landscape_change'])
        axes[2].set_ylabel('H1 landscape change')
        axes[2].set_xlabel('Date')
        fig.suptitle(f'Topological indicators: {w}-day windows')
        fig.tight_layout()
        fig.savefig(FIGURES / f'tda_window_{w}.png', dpi=150)
        plt.close(fig)

    if args.bottleneck:
        from persim import bottleneck
        rows = []
        for w in TDA_WINDOWS:
            dates = returns.index[w-1::args.step]
            prev = None
            for date in dates:
                end = returns.index.get_loc(date)
                cloud = returns.iloc[end-w+1:end+1].to_numpy() * 100
                diagram = ripser(cloud, maxdim=1)['dgms'][1]
                if prev is not None:
                    rows.append({'Date': date, 'window': w,
                                 'H1_bottleneck_to_previous_sample': float(bottleneck(prev, diagram))})
                prev = diagram
        pd.DataFrame(rows).to_csv(FIGURES / 'sampled_bottleneck_distances.csv', index=False)
        print('Saved sampled bottleneck distances; samples are', args.step, 'trading days apart.')
    print('TDA plots saved in', FIGURES)


if __name__ == '__main__':
    main()
