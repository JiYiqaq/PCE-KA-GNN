# Verified production results

本目录保存 seed 42、PyTorch 2.1.2+cu118、DGL 2.2.1+cu118、NVIDIA GeForce GTX 1650 Ti 上完成的正式 CUDA 实验。两个当前模型使用完全相同的 2,652 条测试器件记录和 587 个测试给体–受体对。

| 目录 | 模型 | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| `multimodal/` | 分子图 + 器件条件 | 1.6858 | 2.3500 | 0.6486 |
| `material_only/` | 仅给体/受体分子图 | 1.8206 | 2.5507 | 0.5861 |

每个当前结果目录包含：

- `best_model.pt`：验证集 MAE 最优检查点，含权重、模型配置、目标缩放器和条件预处理器；
- `summary.json`：数据、图、split、GPU、配置和最终指标审计；
- `training_history.csv`：逐 epoch 训练与验证指标；
- `test_predictions.csv`：逐条测试记录的真实 PCE、预测 PCE 和绝对误差。

`experiment_comparison.json` 是按相同测试记录配对、以给体–受体对为独立簇的比较。多模态模型相对仅分子模型的 MAE 降低 0.1349（7.41%），pair-cluster bootstrap 95% CI 为 `[0.0224, 0.3843]`。`context_coverage_audit.json` 记录仅由训练集拟合的条件词表，以及各 split 的缺失/未知比例。

这些结果只有一个固定训练 seed；统计检验反映测试分子对间差异，不包含训练随机性。发表前仍应进行多 seed 重复实验，并报告均值、标准差和预先设定的模型比较方案。

`baseline/` 是旧三维建图和分子对中位数流程的历史结果。它只保留 470 个分子对，样本与当前实验不同，不能与当前模型直接比较。
