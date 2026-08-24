# PCE-KA-GNN

基于给体与受体分子图预测有机光伏器件功率转换效率（PCE）的双分支 Kolmogorov–Arnold 图神经网络。

本项目基于 [LongLee220/KA-GNN](https://github.com/LongLee220/KA-GNN) 改编。原项目面向单分子分类任务；本项目将其扩展为“给体 SMILES + 受体 SMILES → 连续 PCE”的回归任务。项目保留了原作者的 Fourier-KAN 核心层与分子图构建方法，并增加了分子对数据处理、双图编码与融合、回归训练、MAE/RMSE/R² 评估、缓存和结果导出。

仓库已经包含运行所需的 `Active_Database.csv`，克隆后不需要再单独下载数据。Python 环境仍需使用者自行创建。

## 项目结构

```text
PCE-KA-GNN/
├── main_pce.py                 # 训练与评估入口
├── config/                     # 完整、快速和冒烟配置
├── data/
│   ├── raw/Active_Database.csv # 仓库内置 OPV-DB 数据
│   └── processed/              # 运行时自动生成图缓存
├── model/
│   ├── ka_gnn.py               # 来自原 KA-GNN 项目的 Fourier-KAN 层
│   └── pce_ka_gnn.py           # 双分支 PCE 回归模型
├── pce/                        # 配对数据与训练工具
├── tests/                      # 自动测试与小型冒烟数据
├── results/baseline/           # 已验证的两轮 CPU 基线结果
└── docs/                       # 设计与实现说明
```

## 环境配置

已验证环境为 Windows、Python 3.10.20 和 CPU 版 PyTorch 2.0.1。

```bash
conda create -n pce-kagnn python=3.10 -y
conda activate pce-kagnn
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
```

如需 GPU，请根据自己的 CUDA 版本从 PyTorch 和 DGL 官方渠道安装对应构建，其他依赖保持不变。

## 运行

先运行小型冒烟测试，确认环境和完整流程正常：

```bash
python main_pce.py --config config/pce_smoke.yaml
```

运行真实数据的两轮快速基线：

```bash
python main_pce.py --config config/pce_quick.yaml
```

运行默认 100 轮实验：

```bash
python main_pce.py --config config/pce.yaml
```

首次真实数据运行会规范化 SMILES，并使用 RDKit 生成三维构象和非共价边，因此 CPU 建图可能持续数小时。生成的图缓存写入 `data/processed/pce_graphs.pt`，之后运行会自动复用。该缓存约 131 MB，超过 GitHub 普通仓库 100 MB 的单文件限制，因此没有提交；它完全可以由仓库内置 CSV 重建。

运行输出默认保存在 `outputs/`，包括：

- `best_model.pt`：验证集 MAE 最佳的模型；
- `canonical_pairs.csv`：规范化、按有序给体–受体组合聚合的数据；
- `prepared_pairs_with_split.csv`：最终划分；
- `training_history.csv`：逐轮训练记录；
- `test_predictions.csv`：测试集真实值和预测值；
- `summary.json`：数据审计、配置摘要和测试指标。

## 数据处理与模型

程序读取 `donor_smiles`、`acceptor_smiles` 和 `pce` 三列。相同有序给体–受体组合的重复记录使用 PCE 中位数聚合，以降低未纳入模型的制备条件带来的极端值影响。唯一组合按照固定随机种子划分为训练、验证和测试集，目标标准化只使用训练集统计量。

给体和受体分别通过共享参数的 KA-GNN 编码器，随后融合两个图向量、绝对差和逐元素乘积。无 Sigmoid 的 Fourier-KAN 回归头输出标准化 PCE，最终报告原始 PCE 单位下的 MAE、RMSE 和 R²。

## 当前基线

两轮 CPU 快速运行中，38,849 条原始记录整理为 5,877 个唯一组合。受原作者三维建图方法限制，最终 470 个组合可用于该基线，划分为 376/47/47：

| 指标 | 测试集结果 |
|---|---:|
| MAE | 3.0582 |
| RMSE | 3.4442 |
| R² | -0.4890 |

该结果只证明流程可运行，不代表达到可发表性能。负 R² 表明快速基线尚未超过均值预测；三维建图成功率、数据划分、更多制备条件输入以及系统基线对比都是后续研究重点。

## 测试

```bash
python -m unittest discover -s tests -v
```

## 来源、许可与引用

代码改编自 LongLee220 的 KA-GNN 仓库，原代码采用 MIT License。详细改编边界见 [NOTICE.md](NOTICE.md)。数据来自 OPV-DB，采用 CC BY 4.0，详细信息见 [DATA_LICENSE.md](DATA_LICENSE.md)。

使用 KA-GNN 方法时请引用：

```bibtex
@article{li2025kagnn,
  title   = {Kolmogorov--Arnold graph neural networks for molecular property prediction},
  author  = {Li, Longlong and Zhang, Yipeng and Wang, Guanghui and Xia, Kelin},
  journal = {Nature Machine Intelligence},
  volume  = {7},
  pages   = {1346--1354},
  year    = {2025},
  doi     = {10.1038/s42256-025-01087-7}
}
```

使用仓库内置数据时请引用 OPV-DB：

```bibtex
@dataset{qiu2026opvdb,
  author    = {Qiu, Jiangjie},
  title     = {OPV-DB: a literature-mined organic photovoltaic device performance database},
  year      = {2026},
  publisher = {Zenodo},
  version   = {1.0.0},
  doi       = {10.5281/zenodo.20841543}
}
```
