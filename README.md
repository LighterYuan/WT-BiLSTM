# WT-BiLSTM: Wavelet-Reconstruction-Enhanced BiLSTM for Short-Term Network Traffic Forecasting

This repository accompanies the WT-BiLSTM study on one-step-ahead network traffic forecasting. The framework combines deterministic wavelet-based reconstruction with bidirectional LSTM (BiLSTM) encoding over a strictly observed historical input window.

The repository is organized as a reproducibility package: it contains the major-revision experiment configuration, exported metrics and predictions, figure-generation utilities, and—when included in the full repository—the scripts required to rerun baseline, sensitivity, and preprocessing-ablation experiments.

> **Final reported WT-BiLSTM configuration**
>
> - Wavelet basis: Daubechies-8 (`db8`)
> - Decomposition level: `2`
> - Threshold mode: `soft`
> - Threshold strategy: `universal`
> - Threshold scale: `1.0`
> - Historical input length: `10`
> - Forecast horizon: one step ahead
> - Main evaluation seeds: `1, 2, 3, 4, 5`
> - Datasets: `UK_Ac` and `EU_Co`

## Overview

Operational traffic traces are often nonlinear, non-stationary, noisy, and bursty. WT-BiLSTM first reconstructs the observed traffic sequence through wavelet processing and then feeds a short reconstructed historical window to a BiLSTM predictor.

For an observed window

\[
X_t = \{x_{t-L+1}, x_{t-L+2}, \ldots, x_t\},
\]

the model predicts the next traffic value

\[
\hat{x}_{t+1}.
\]

The BiLSTM processes only the observations contained in \(X_t\): its forward and backward recurrent branches encode the same historical window in opposite directions. Neither branch receives \(x_{t+1}\) or any later sample.

## Main Features

- Deterministic wavelet-based reconstruction before recurrent forecasting.
- One-step-ahead, short-window univariate traffic forecasting.
- Chronological 80/20 train-test split with training-only Min-Max scaling.
- Benchmark comparison against recurrent, convolutional, Transformer, and linear forecasting baselines.
- Parameter sensitivity experiments for the input-window length, wavelet basis, decomposition level, threshold mode, threshold strategy, and threshold scale.
- Preprocessing diagnostics comparing raw BiLSTM, WT-BiLSTM, moving-average BiLSTM, and Savitzky-Golay BiLSTM.
- Publication-ready figure generation from exported CSV metrics and prediction files.

## Repository Layout

The exact layout may vary slightly by release. The following structure matches the supplied major-revision configuration and output package.

```text
WT-BiLSTM/
├── config_major_revision.yaml
├── generate_wt_bilstm_sci_figures.py
├── README.md
│
├── src/                                  # Present in the full experiment repository
│   ├── run_major_revision_baselines.py
│   ├── run_major_revision_sensitivity.py
│   ├── run_major_revision_ablation.py
│   └── plot_major_revision_figures.py
│
├── scripts/                              # Present in the full experiment repository
│   ├── run_major_revision_experiments.sh
│   └── run_major_revision_experiments.bat
│
├── data/
│   ├── raw/
│   │   ├── UK_Ac.csv
│   │   └── EU_Co.csv
│   ├── processed_major_revision/
│   └── wavelet_major_revision/
│
├── results_major_revision/
│   ├── metrics/
│   │   ├── sota_baselines_summary.csv
│   │   ├── sensitivity_grouped_summary.csv
│   │   ├── denoising_ablation_summary.csv
│   │   └── ...
│   ├── predictions/
│   │   ├── sota_baseline_UK_Ac_*_seed1_pred.csv
│   │   └── sota_baseline_EU_Co_*_seed1_pred.csv
│   └── figures/
│
├── results_major_revision_level2_confirm/
│   ├── metrics/
│   ├── predictions/
│   │   ├── wt_bilstm_level2_confirm_UK_Ac_WT-BiLSTM_seed1_pred.csv
│   │   └── wt_bilstm_level2_confirm_EU_Co_WT-BiLSTM_seed1_pred.csv
│   └── figures/
│
├── figures/                              # Optional manuscript-ready figure folder
├── metrics/                              # Optional flat export folder
├── predictions/                          # Optional flat export folder
└── tables/                               # Optional manuscript-ready table folder
```

## Installation

Create and activate a virtual environment, then install the scientific Python stack used by the workflow.

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install numpy pandas pyyaml scikit-learn PyWavelets matplotlib scipy tensorflow
```

The plotting utility requires `PyWavelets` for the reconstructed-signal overlay in Fig. 2.

## Data Format

Place the raw datasets under `data/raw/`.

| Dataset | Expected file | Time column | Target column | Split |
|---|---|---:|---:|---:|
| UK_Ac | `UK_Ac.csv` | `Time` | `traffic` | chronological 80/20 |
| EU_Co | `EU_Co.csv` | `Time` | `traffic` | chronological 80/20 |

The scaler must be fitted on the training partition only and then applied to both training and test partitions.

## Important Configuration Note

Some earlier major-revision exports retain `level: 3` inside `config_major_revision.yaml`, because that file was used while producing the broader sensitivity package. **Do not use that value for the final reported WT-BiLSTM main result.**

For the final confirmatory WT-BiLSTM model, use:

```yaml
preprocessing:
  time_steps: 10
  scaler: minmax
  leakage_safe: true

wavelet:
  wavelet: db8
  level: 2
  threshold_mode: soft
  threshold_strategy: universal
  threshold_scale: 1.0
```

## Reproducing the Major-Revision Experiments

The commands below apply when the full source-code directories (`src/` and `scripts/`) are present.

### Quick smoke test

```bash
python src/run_major_revision_baselines.py \
  --config config_major_revision.yaml \
  --quick \
  --datasets UK_Ac \
  --models TCN Transformer \
  --seeds 1 \
  --quiet

python src/run_major_revision_sensitivity.py \
  --config config_major_revision.yaml \
  --quick \
  --datasets UK_Ac \
  --experiments window basis \
  --seeds 1 \
  --quiet

python src/run_major_revision_ablation.py \
  --config config_major_revision.yaml \
  --quick \
  --datasets UK_Ac \
  --seeds 1 \
  --quiet
```

### Full major-revision run

Linux/macOS:

```bash
bash scripts/run_major_revision_experiments.sh
```

Windows:

```bat
scripts\run_major_revision_experiments.bat
```

The major-revision add-on writes to separate paths, such as `results_major_revision/`, `models_major_revision/`, and `logs_major_revision/`, so that original outputs remain intact.

## Regenerating Manuscript Figures

The figure-generation script searches recursively under the supplied input directory for exported CSV files.

```bash
python generate_wt_bilstm_sci_figures.py \
  --input-dir . \
  --output-dir ./sci_figures \
  --dpi 600
```

Typical required inputs include:

```text
wt_bilstm_level2_confirm_summary.csv
wt_bilstm_level2_confirm_raw.csv
repeated_runs.csv
sota_baselines_summary.csv
sensitivity_grouped_summary.csv
denoising_ablation_summary.csv
complexity.csv
wt_bilstm_level2_confirm_UK_Ac_WT-BiLSTM_seed1_pred.csv
wt_bilstm_level2_confirm_EU_Co_WT-BiLSTM_seed1_pred.csv
```

If `PyWavelets` reports `ValueError: buffer source array is read-only`, make the input array writable before calling `pywt.wavedec()`:

```python
values = np.array(values, dtype=np.float64, copy=True)
values = np.ascontiguousarray(values)
values.setflags(write=True)
```

## Evaluated Models

The benchmark includes:

- GRU
- LSTM
- WT-LSTM
- BiLSTM
- CNN-LSTM
- TCN
- Transformer
- DLinear
- NLinear
- PatchTST-style predictor
- WT-BiLSTM
- WT-TBiLSTM

WT-TBiLSTM is retained as a topology-augmented architectural ablation. It increases model complexity but does not provide stable accuracy gains under the evaluated short-window univariate setting.

## Main Results

Results are reported as mean ± standard deviation over five random seeds.

| Dataset | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| UK_Ac | 73.7714 ± 1.7881 | 53.4556 ± 1.2648 | 1.2673% ± 0.0316% |
| EU_Co | 1.0962 × 10^8 ± 9.0305 × 10^5 | 8.0432 × 10^7 ± 9.7179 × 10^5 | 2.6236% ± 0.0782% |

Relative to raw BiLSTM, WT-BiLSTM reduces RMSE by 17.90% on UK_Ac and 20.00% on EU_Co. Sensitivity analysis identifies two-level decomposition as consistently effective on both datasets; the preferred wavelet basis, input-window length, threshold strategy, and threshold scale remain dataset-dependent.

## Reproducibility Notes

- The main benchmark uses five random seeds: `1, 2, 3, 4, 5`.
- Sensitivity and diagnostic smoothing experiments use three random seeds.
- The final main configuration is `db8 + level 2 + soft + universal + scale 1.0 + window 10`.
- The BiLSTM causal interpretation applies to the recurrent encoder: it receives only the observed historical window. A practical online deployment should also evaluate wavelet reconstruction under strictly rolling causal preprocessing constraints.
- Moving-average and Savitzky-Golay results are diagnostic preprocessing comparisons and should not be interpreted automatically as deployable online forecasting procedures.

## Citation

Please replace the placeholder below with the final bibliographic record and DOI after publication.

```bibtex
@article{yuan2026wtbilstm,
  title   = {Wavelet-Reconstruction-Enhanced Bidirectional LSTM for Robust Short-Term Network Traffic Forecasting},
  author  = {Yuan, Jinliang and Yang, Yu and Wang, Mingqi},
  year    = {2026},
  note    = {Manuscript under review}
}
```

## License

No license file is included in the supplied package. Add an explicit license before distributing the repository publicly.
