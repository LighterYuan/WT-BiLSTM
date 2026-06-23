# WT-TBiLSTM Reproducible Experimental Project

本工程用于复现修改后论文手稿中的主要实验流程：

- 两个主数据集：`UK_Ac`、`EU_Co`
- 七个预测模型：`GRU`、`LSTM`、`WT-LSTM`、`BiLSTM`、`CNN-LSTM`、`WT-BiLSTM`、`WT-TBiLSTM`
- 结果输出：整体指标、消融实验、完整测试集预测、复杂度分析、重复实验统计检验、论文图表数据

> 说明：本工程是对原始零散实验脚本的规范化重构。`archive/original_code_reference/` 中保留了原始脚本作为参考，主实验以 `src/` 目录代码为准。

---

## 1. 快速开始

建议使用 Python 3.9–3.11。

```bash
cd WT_TBiLSTM_Reproducible_Project
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```


运行前可先检查环境：

```bash
python src/environment_check.py
```

运行一次快速冒烟测试：

```bash
python src/run_all_experiments.py --config config.yaml --quick --seeds 1
```

运行完整实验：

```bash
python src/run_all_experiments.py --config config.yaml --seeds 1 2 3 4 5
python src/run_ablation.py --config config.yaml
python src/run_robustness.py --config config.yaml
python src/run_complexity.py --config config.yaml
python src/statistical_test.py --config config.yaml
python src/plot_figures.py --config config.yaml
```

或者直接运行：

```bash
bash scripts/run_full_experiment.sh
```

Windows 用户可运行：

```bat
scripts\run_full_experiment.bat
```

---

## 2. 数据说明

工程已经从你上传的 `code.zip` 中整理出：

| 工程文件 | 原始文件 | 用途 |
|---|---|---|
| `data/raw/UK_Ac.csv` | `isp.csv` | UK academic network traffic |
| `data/raw/EU_Co.csv` | `inttraffic.csv` | EU core network traffic |

默认时间列为 `Time`，预测目标列为 `traffic`。

---

## 3. 主要输出文件

完整运行后会生成：

```text
results/metrics/overall_metrics.csv
results/metrics/repeated_runs.csv
results/metrics/wavelet_ablation.csv
results/metrics/topology_ablation.csv
results/metrics/complexity.csv
results/metrics/statistical_tests.csv

results/predictions/*_pred.csv
results/figures/*.png
results/tables/*.csv
data/processed/*_windows*.npz
data/wavelet/*_denoised.csv
```

---

## 4. 重要实验设计

### 4.1 避免数据泄漏

本工程默认采用严格时间序列流程：

1. 先按时间顺序划分训练集和测试集；
2. `MinMaxScaler` 只在训练集上 `fit`；
3. 测试集只调用 `transform`；
4. 小波去噪分别作用于训练段和测试段，避免测试未来信息进入训练过程；
5. 测试窗口允许使用训练末尾的历史点作为上下文，但预测目标只来自测试段。

### 4.2 WT-TBiLSTM 中的 topology-aware attention

工程实现了一个可运行、可复现的拓扑增强注意力层：

- 从每个时间窗口中提取局部趋势、曲率、局部极值、局部持久性强度、位置编码等拓扑启发式特征；
- BiLSTM 输出时序隐藏状态；
- TopologyAttention 根据隐藏状态与拓扑特征联合计算注意力权重；
- 输出加权上下文向量用于最终预测。

实现文件：

```text
src/topology_features.py
src/topology_attention.py
```

---

## 5. 与论文图表的对应关系

见：

```text
docs/MANUSCRIPT_RESULT_MAP.md
```

---

## 6. 注意事项

- 完整实验需要一定训练时间，建议使用 GPU。
- 若只检查流程，请使用 `--quick`。
- 若需要更强 SCI 版本，建议在本工程基础上进一步增加 Transformer / PatchTST / DLinear / GNN 等强基线。
