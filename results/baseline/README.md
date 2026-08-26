# Historical baseline (not directly comparable)

本目录保留改造前的确定性 CUDA 运行，仅用于追踪项目演变。旧流程把重复给体–受体器件取 PCE 中位数，并把三维构象生成作为建图前提，最终只保留 470/5,877 个分子对；测试指标为 MAE 2.2126、RMSE 2.7700、R² 0.2163。

当前正式流程保留逐器件条件，并使用稳健的二维拓扑图。因此本目录与 `results/multimodal/`、`results/material_only/` 的数据样本和 split 均不同，不能把指标差值解释为模型提升。当前可比较的消融实验和统计结果见上一级 [README](../README.md)。
