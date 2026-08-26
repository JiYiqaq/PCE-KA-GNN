# PCE-KA-GNN

基于给体与受体分子图、材料性质、器件结构和制备条件预测有机光伏器件功率转换效率（PCE）的多模态 Kolmogorov–Arnold 图神经网络。

本项目基于 [LongLee220/KA-GNN](https://github.com/LongLee220/KA-GNN) 改编。原项目面向单分子分类；本项目保留 Fourier-KAN 核心思想，将任务改为逐器件 PCE 回归，并新增双分子图编码、条件特征编码、按给体–受体对分组的数据划分、GPU 训练、回归评估和实验审计。

仓库已包含 `Active_Database.csv` 及由它确定性生成的逐器件表，克隆后不需要额外下载数据；使用者只需配置 Python/CUDA 环境。

## 当前模型预测什么

每条样本对应一个器件，而不是一个去重后的分子对。模型综合以下信息预测该器件的 PCE：

- 分子输入：给体和受体的规范化 SMILES 拓扑图；
- 材料性质：给体/受体 HOMO、LUMO；
- 器件与制备数值条件：活性层厚度、退火温度、给受体比例、添加剂比例；
- 类别条件：常规/反型器件、ETL、HTL、溶剂和添加剂。

`Voc`、`Jsc`、`FF`、重新计算/平均/最佳 PCE 等目标衍生字段被明确排除，防止数据泄漏。数值缺失值使用仅在训练集拟合的中位数填充，并附加缺失掩码；类别缺失值和测试时未见类别使用不同标记。

## 项目结构

```text
PCE-KA-GNN/
├── main_pce.py                       # 正式训练与评估入口
├── config/
│   ├── pce.yaml                      # 多模态正式实验
│   ├── pce_material_only.yaml        # 仅给体/受体分子图消融实验
│   ├── pce_quick.yaml                # 同一 CUDA 栈的短训练配置
│   └── pce_smoke.yaml                # 同一 CUDA 栈的最小验证配置
├── data/
│   ├── raw/Active_Database.csv       # 仓库内置 OPV-DB 数据
│   └── processed/device_records.csv  # 可复现的逐器件数据缓存
├── model/pce_ka_gnn.py               # 双图与条件融合模型
├── pce/                              # 数据、拓扑图、条件预处理和训练工具
├── scripts/                          # 条件覆盖率和成对实验比较
├── tests/                            # 自动测试
├── results/                          # 已验证的正式 GPU 结果与检查点
└── docs/                             # 设计和实现说明
```

## 环境配置

正式环境已在 Windows、Python 3.10.20、NVIDIA GeForce GTX 1650 Ti 上验证：PyTorch `2.1.2+cu118`、DGL `2.2.1+cu118`、torchdata `0.7.1`。全部配置都要求 CUDA；CUDA 不可用或 DGL 图不能进入显卡时程序会立即报错，不会静默改用 CPU。

```bat
conda create -n kagnn python=3.10 -y
conda activate kagnn
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements-gpu.txt
python -m pip check
python -c "import torch,dgl; g=dgl.graph(([0],[1])).to('cuda'); print(torch.__version__, dgl.__version__, torch.cuda.get_device_name(0), g.device)"
```

最后一条命令必须显示 `2.1.2+cu118`、`2.2.1+cu118`、实际 NVIDIA 显卡名称和 `cuda:0`，之后才能开始实验。

## 运行

先做生产运行栈验证。它只减少样本和 epoch，仍使用正式 CUDA 依赖与实际 GPU：

```bat
conda activate kagnn
python -u main_pce.py --config config/pce_smoke.yaml
```

正式运行多模态模型：

```bat
python -u main_pce.py --config config/pce.yaml
```

运行使用相同数据划分的仅分子消融模型：

```bat
python -u main_pce.py --config config/pce_material_only.yaml
```

两组实验结束后执行成对比较与条件覆盖率审计：

```bat
python scripts/compare_experiments.py
python scripts/audit_context_coverage.py
```

训练输出位于 `outputs/pce_multimodal/` 或 `outputs/pce_material_only/`，包括最佳检查点、逐 epoch 记录、逐器件测试预测、带 split 的设备表和运行摘要。已经验证过的正式结果保存在 [results/](results/) 中。

## 数据与建图

38,849 条原始记录中有 26,501 条同时具备有限 PCE 和可用给体/受体 SMILES。模型不会把相同给体–受体组合取中位数合并，而是保留每种器件条件。所有相同有序分子对必须进入同一 split，从而避免同一对同时出现在训练集和测试集。

建图使用 RDKit 的确定性二维拓扑，不再把三维构象生成作为前置条件。图包含显式氢、双向化学键、92 维 CGCNN 原子特征和 21 维键特征；节点最终输入为 113 维。3 个含虚拟原子 `*` 的 SMILES 和 2 个超过 500 个重原子的数据库异常结构被明确拒绝，最终保留 3,982/3,987 个分子、26,488/26,501 条器件记录和 5,872 个分子对。

`data/processed/pce_topology_graphs.pt` 约 515 MB，超过 GitHub 普通仓库单文件 100 MB 限制，因此不提交。首次运行会用 8 个 CPU 进程从仓库内置数据确定性重建，在本机约 105 秒；之后复用缓存约 4 秒。这里只使用 CPU 是因为 RDKit 该拓扑构建路径没有 CUDA 实现。神经网络前向、反向、验证与正式训练全部在 GPU 上执行。

正式划分固定 seed 42，共 21,183/2,653/2,652 条训练/验证/测试记录，对应 4,698/587/587 个互不重叠的给体–受体对。正式配置使用 batch size 256、float32、确定性 CUDA、最多 100 epochs、early stopping patience 20，并将全部唯一分子图预加载到显卡；GTX 1650 Ti 上训练峰值显存约 1.33 GB。

## 正式结果

两种模型使用完全相同的测试记录和分子对划分：

| 模型 | 测试 MAE | 测试 RMSE | 测试 R² | 最佳 epoch | 完成 epoch |
|---|---:|---:|---:|---:|---:|
| 仅分子图 | 1.8206 | 2.5507 | 0.5861 | 12 | 32 |
| 分子图 + 器件条件 | **1.6858** | **2.3500** | **0.6486** | 59 | 79 |

加入器件条件后，MAE 下降 0.1349（7.41%），RMSE 下降 0.2007，R² 提高 0.0626。以给体–受体对为独立簇的 bootstrap 得到 MAE 改善 95% CI `[0.0224, 0.3843]`；587 个测试分子对上的双侧 Wilcoxon 检验为 `p = 6.80e-5`，秩二列相关为 0.190。该比较只覆盖一个固定训练 seed，尚未量化不同训练 seed 带来的模型波动，因此是正式工程结果而不是可直接用于论文的最终统计结论。

旧的 470 个分子对三维建图结果保留在 `results/baseline/` 作为历史记录，但它使用不同样本集和旧数据流程，不能与上表直接比较。

## 测试

```bat
python -m unittest discover -s tests -v
```

测试包含真实 CUDA 前向与反向检查；需要在上述 `kagnn` GPU 环境中执行。

## 来源、许可与引用

代码改编自 LongLee220 的 KA-GNN 仓库，原代码采用 MIT License，详细改编边界见 [NOTICE.md](NOTICE.md)。仓库数据来自 OPV-DB，采用 CC BY 4.0，详细信息见 [DATA_LICENSE.md](DATA_LICENSE.md)。

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
