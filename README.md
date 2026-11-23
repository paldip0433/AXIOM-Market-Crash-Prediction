# AXIOM: Topological Early-Warning Signals for Market Crash Prediction

**IE643 — Deep Learning: Theory and Practice | IIT Bombay**  
**Dipendu Pal | MSc course project**

## About the project

In this project, I tried to predict large falls in the US stock market using deep learning and topological data analysis (TDA).

The main question was simple: **Can the shape of market data tell us something useful about a possible crash, apart from the usual financial indicators?**

I used daily data from four US stock indices, calculated classical financial features, and then added features from persistent homology. I compared the models using chronological train, validation and test periods. I also ran more experiments to check whether changes in topological features give better or earlier warnings.

This is an experimental course project, not a trading system.

## Data and prediction task

I used daily closing-price data for:

- S&P 500
- Dow Jones
- NASDAQ
- Russell 2000

The source files cover 1990–2026. After aligning the data and preparing the features, the main dataset has **8,596 labelled observations**.

For each trading day, the target is 1 if the S&P 500 falls by **at least 10% from that day's level at any point in the next 30 trading days**. Otherwise, the target is 0.

This gives a difficult classification problem because positive labels are uncommon and nearby labels can overlap.

## What I did

### 1. Classical features

I started with **18 financial features**, including information from returns and volatility.

### 2. Topological features

I converted the daily returns of the four indices into four-dimensional point clouds. I used rolling windows of **40, 80 and 120 trading days** and computed Vietoris–Rips persistent homology.

The main idea is to study connected components ($H_0$) and loops ($H_1$) in these point clouds as the distance scale changes. I used persistence summaries and persistence landscapes to turn these into numerical features.

The original setup has **30 usable TDA features**. Together with the classical features, this gives **48 combined features**.

In the later novelty assessment, I focused more on changes in the $H_1$ persistence landscapes between successive windows. I also tested an enhanced set of **54 features**, including extra change-based features.

### 3. Models

I compared logistic regression, LSTM, bidirectional LSTM and Transformer models. The neural networks use **30-day input sequences**.

The experiments use chronological splits instead of random shuffling. Scaling is fitted on the training data, and a **30-trading-day gap** is used at split boundaries to reduce leakage from the forward-looking labels. The neural-network experiments were repeated with three seeds.

## Main results

The first set of experiments compared classical features, TDA features and their combination.

| LSTM feature set | Mean test average precision (AP) |
|---|---:|
| Classical only | 0.1340 |
| TDA only | 0.0877 |
| Classical + TDA | **0.2459** |

These are averages over three seeds on the original held-out test period. Average precision (AP, also called PR-AUC here) is useful because crash labels are rare.

The combined LSTM had the highest test AP in this comparison. But its **validation AP was lower** than the classical-only LSTM (0.0690 versus 0.1106). So I cannot say that adding TDA always improves prediction.

## Novelty assessment

After the first experiments, I looked more closely at the $H_1$ landscape-change features.

### Feature ablation

| Feature set | Mean test AP | Mean test ROC-AUC |
|---|---:|---:|
| Classical only | 0.1340 | 0.7562 |
| Combined without $H_1$ change | 0.2240 | 0.8637 |
| Combined with $H_1$ change | **0.2459** | **0.8732** |
| Combined with enhanced $H_1$ features | 0.1951 | Not aggregated |

Adding the original $H_1$ change features improved the average test AP compared with removing them. The improvement was not the same for every seed. Adding more enhanced features did not improve the original fixed-split AP.

### Warning signals

I also checked warnings, not just classification scores. The warning threshold was chosen from the validation period and then applied to the test period.

| Model | Test false-positive rate | Detected positive-label episodes | False-alert episodes |
|---|---:|---:|---:|
| Classical | 0.0550 | 0.2593 | 5.67 |
| Combined | 0.0915 | 0.8148 | 5.33 |
| Enhanced $H_1$ | 0.1296 | 0.9259 | 6.67 |

The enhanced model detected more labelled episodes, but it also had more false alarms. These episodes are groups of consecutive positive labels, not necessarily separate market crashes.

### Other experiments

I also compared eight loss-function settings, LSTM/BiLSTM/Transformer architectures, and three historical walk-forward folds.

- Weighted binary cross-entropy had the highest mean test AP (**0.2459**) in the original loss comparison.
- In the matched architecture comparison, the enhanced feature set improved mean AP within each architecture. The LSTM gave the highest AP among the three architectures in that comparison.
- In the walk-forward tests, enhanced $H_1$ improved the average AP in each fold, but the results varied by seed and market period. Warning lead time and false alarms were not consistently better.

The matched architecture runs used a different checkpoint-selection rule, so their AP values should not be compared directly with the original fixed-split baseline.

## Repository contents

```text
code/                 Main Python scripts
data/                Raw and processed data
demo/                Small demo inputs and outputs
experiments/         Original experiment results and saved artifacts
figures/             Plots and result summaries
final_results/       Final result tables
models/              Saved models and related files
novelty_assessment/  Extra experiments, results and figures
report/              Report source and supporting files
```

The root folder also contains the final report, presentation slides and the four original index CSV files. The two MP4 recordings were kept outside GitHub because of their size.

## Running the project

The project was developed in **Google Colab**, with the files stored in Google Drive.

The basic order is:

1. Prepare and align the four index series.
2. Generate classical financial features.
3. Compute TDA features for the rolling windows.
4. Prepare the 30-day model sequences and chronological splits.
5. Train the baseline and deep learning models.
6. Check the saved metrics, predictions and figures.

The full TDA calculation can take time. In the original configuration, `MAX_TDA_WINDOWS = 200` in `code/config.py` can be used for a quick check, while `None` is used for the full calculation. A quick run is not the same as reproducing the reported results.





## What I concluded

The results suggest that topological features can contain information that is not fully captured by the classical indicators. The combined LSTM performed better on the original test period, but that improvement did not carry over consistently to validation and all the later experiments.

I therefore see TDA as a possible additional set of features, **not as a proven early-warning system**. More testing across different market periods and independent crash events would be needed before making a stronger claim.

---

*IE643 course project, Indian Institute of Technology Bombay.*
