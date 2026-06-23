# Experiment Protocol

## Objective

Evaluate WT-TBiLSTM for short-term network traffic prediction under a reproducible and leakage-safe experimental setting.

## Models

1. GRU
2. LSTM
3. WT-LSTM
4. BiLSTM
5. CNN-LSTM
6. WT-BiLSTM
7. WT-TBiLSTM

## Datasets

- UK_Ac
- EU_Co

## Train/Test Split

Chronological split, default 80% training and 20% testing.

## Sliding Window

Default `time_steps=10`.

For each sample:

```text
X_t = [x_{t-10}, ..., x_{t-1}]
y_t = x_t
```

## Wavelet Transform

- Wavelet basis: db8
- Decomposition level: 3
- Thresholding: universal soft thresholding

Wavelet processing is applied after chronological split to reduce data leakage risk.

## Metrics

- RMSE
- MAE
- MAPE

## Repeated Runs

Default seeds: 1, 2, 3, 4, 5.

The script generates both individual runs and mean/std summaries.

## Statistical Test

The default comparison is WT-TBiLSTM versus selected baseline models. 
Wilcoxon signed-rank test is used when possible; otherwise paired t-test is used.
