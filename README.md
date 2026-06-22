# Special Subgraph — 连接组噪声鲁棒稀疏子图发现

从斑马鱼全脑连接组中，通过可微搜索发现具有噪声鲁棒性的稀疏子图结构（motif），并验证其在稀疏 MLP 中作为连接掩码的泛化能力。

## 核心思路

1. 从 52 节点有向加权连接组出发，提取 fan-in hub 锚点
2. 在 hub 邻域内，用 Gumbel-Sigmoid 可微搜索选出 8 节点 / 22 边的子图掩码
3. 搜索目标：同时优化 clean accuracy 和 noise robustness（多 σ 高斯噪声）
4. 发现的 DSS-Motif 在 MNIST / Fashion-MNIST 上显著优于随机稀疏和度保持基线

## 流水线概览

| 阶段 | 脚本 | 功能 |
|------|------|------|
| Stage 1 | `stage1_fan_extraction.py` | 从连接组提取 Fan-α/β/γ 结构掩码 |
| Stage 2 | `stage2_baseline_generation.py` | 生成 Random-Sparse 和 Degree-Preserved 基线掩码 |
| Stage 3 | `stage3_training.py` | 在 MNIST/Fashion-MNIST 上训练稀疏 MLP 并评估鲁棒性 |
| DSS Hub | `dss_hub_enumeration.py` | 枚举所有 fan-in hub 及其候选节点池 |
| DSS Search | `dss_search.py` | 可微子图搜索（v3: 多 σ 随机噪声目标） |
| DSS Validate | `dss_validate.py` | 验证搜索得到的 hard mask 的训练表现 |
| Eval Fashion | `eval_fashion.py` | Fashion-MNIST 上验证 top-3 DSS-Motif |
| Eval Seeds | `eval_extended_seeds.py` | 扩展种子 (seeds 5–9) 交叉验证 |
| Fan Curves | `fan_curves_eval.py` | 从 Stage 3 checkpoint 重新计算 Fan 的逐 σ 曲线 |
| Figures | `generate_paper_figs.py` | 生成论文图表（robustness curves, bar charts, tables） |

## 执行顺序

```bash
# 0. 准备连接组数据
#    将 conn_matrix_complete.npy 放入 data/

# 1. 结构提取与基线
python stage1_fan_extraction.py
python stage2_baseline_generation.py

# 2. 基线训练评估
python stage3_training.py

# 3. DSS 可微搜索
python dss_hub_enumeration.py
python dss_search.py
python dss_validate.py

# 4. 扩展验证
python eval_fashion.py
python eval_extended_seeds.py
python fan_curves_eval.py

# 5. 生成论文图表
python generate_paper_figs.py
```

## 依赖

- Python ≥ 3.12
- PyTorch ≥ 2.11
- torchvision ≥ 0.26
- NumPy ≥ 2.0
- NetworkX ≥ 3.3
- matplotlib（仅 `generate_paper_figs.py` 需要）

安装：

```bash
uv sync
```

## 关键参数

| 参数 | 值 | 含义 |
|------|------|------|
| G | 8 | 子图节点数 / MLP 分组数 |
| EDGES | 22 | 每个掩码的边数 |
| Q | 0.25 | 连接组二值化阈值（top 25% 边） |
| HUB_MIN_IN | 4 | hub 最低入度 |
| NOISE_SIGMAS | 0.20–0.50 | AvgRatio 评估的噪声范围 |
| SEARCH_EPOCHS | 12 | DSS 搜索训练轮数 |
| Training EPOCHS | 20 | 验证训练轮数 |

## 输出结构

```
outputs/
├── stage1/          Fan-α/β/γ 掩码 + 候选统计
├── stage2/          Random-Sparse + Degree-Preserved 基线
├── stage3/          训练 checkpoint + main_results.csv
├── meta_search/     Hub 锚点枚举
├── meta_search_v3/  DSS 搜索结果 (hard mask + 验证)
└── paper/           扩展验证结果
figs/                论文图表
```
