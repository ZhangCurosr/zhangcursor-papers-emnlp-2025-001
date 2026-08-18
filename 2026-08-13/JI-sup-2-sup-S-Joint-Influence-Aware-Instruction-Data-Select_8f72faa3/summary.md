---
title: "JI-sup-2-sup-S-Joint-Influence-Aware-Instruction-Data-Select"
source: https://aclanthology.org/2025.emnlp-main.26.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:44:08"
---

# 论文速读：JI-sup-2-sup-S-Joint-Influence-Aware-Instruction-Data-Select

## 一句话总结
论文提出 JI²S 框架，通过离散导数联合建模指令样本的边际影响与二阶联合影响，从大规模 Alpaca 数据中自动筛选高质量子集；仅用 1k 样本微调 LLaMA2/Mistral 系列模型，在多项基准上超越全量 52k 训练及主流 GPT 打分基线，验证了“数据质量优于数量”与“高阶交互不可忽视”的核心结论。

## 研究问题与动机
1. **指令微调极度依赖数据质量**：自动生成的指令数据集（如 Alpaca-52k）包含大量噪声、重复或低质样本，直接全量训练反而拖累下游性能。
2. **GPT 打分方法存在系统性偏差**：AlpaGasus、DEITA 等依赖外部强模型打分筛选，易受位置偏置、冗长偏置与自我增强偏置影响，且与目标模型的实际优化轨迹脱钩。
3. **现有影响分析方法忽略样本交互**：LESS 等基于梯度的影响函数方法仅计算单样本边际贡献（假设贡献可加），无法捕捉样本间的冗余与非加性高阶交互，导致筛选出的子集仍含重复信息。
4. **高质量人工数据构建成本高昂**：LIMA 等精标数据集证明少量高质量数据优于大量噪声数据，但手工策展不可扩展，亟需自动化筛选范式。

## 核心贡献（创新点）
1. **提出基于离散导数的联合影响估计算法**：将一阶边际影响与二阶联合影响统一纳入影响函数框架，首次量化指令样本间的非加性交互效应。
2. **设计 JI²S 代理测试集策略**：摒弃外部 GPT 打分，改用人工高质量 LIMA-1k 作为代理评估集，使影响计算直接对齐目标模型的真实优化方向。
3. **实现低算力开销的可扩展筛选流水线**：结合 LoRA 预热、随机投影降维（8,192 维）与 Adam 更新项近似，在保留梯度保真度的同时显著降低 Hessian 相关计算成本。
4. **多维度实证验证有效性**：在 LLaMA2-7B/13B、Mistral-7B 上系统评测，仅用 1k 筛选样本即全面超越全量训练及 LIMA、AlpaGasus、DEITA 等强基线，并给出 λ 超参敏感性分析与多样性可视化证据。

## 方法详解
JI²S 采用四阶段流水线，核心围绕“影响评分 → 排序 → 截断选取”展开：

1. **LoRA Warm-Up（预热）**：从原始训练集 $\mathcal{D}_{train}$ 随机采样 5% 样本，使用 LoRA（rank=128, α=512, dropout=0.1）微调 4 个 epoch，获取逼近全量训练梯度方向的中间参数 $\theta^t$。
2. **梯度提取与降维**：在每一训练步 $t$，计算训练样本 $z_i$ 与代理测试集样本 $z'$ 在 LoRA 参数空间的梯度，并通过随机投影将高维梯度压缩至 8,192 维子空间，缓解维度灾难。
3. **联合影响计算**：
   - **边际项**（一阶）：沿用适配 Adam 的 influence formula，$\mathrm{Inf}^1(z_i) \approx -\sum_{t} \eta_t \sum_{z'\in\mathcal{D}_{test}} \Gamma(z_i,\theta^t)^\top \nabla_\theta \mathcal{L}(z',\theta^t)$，衡量单样本对测试损失的独立贡献。
   - **联合项**（二阶）：基于离散导数定义，经二阶泰勒展开近似为 $\mathrm{Inf}^2(z_i,z_j) \approx -\sum_{t} \eta_t^2 \Gamma(z_i,\theta^t)^\top H_{z'} \Gamma(z_j,\theta^t)$，其中 $H_{z'}$ 为测试损失 Hessian 矩阵的近似，刻画 $z_i$ 与 $z_j$ 组合产生的非加性交互。
   - **统一评分**：$\mathrm{Inf}(z_i) = \sum_{t=1}^T \eta_t \left( \mathrm{Inf}^1(z_i) + \lambda \sum_{z_j \neq z_i} \mathrm{Inf}^2(z_i,z_j) \right)$，$\lambda$ 控制二阶交互权重；得分越低表示该样本对目标模型性能提升越显著。
4. **样本筛选**：对所有训练样本按 $\mathrm{Inf}(z_i)$ 升序排列，截取 Top-K（默认 K=1,000）构成高质量子集，用于最终指令微调。

## 实验与结果
- **数据集与基线**：训练集 Alpaca-52k，筛选目标 1k；基线包括全量 Alpaca-52k、LIMA-1k、AlpaGasus-1k、DEITA-1k。
- **评估基准**：Open LLM Benchmarks（ARC、HellaSwag、MMLU、Winogrande）、MT-Bench、GPT-4 五数据集成对胜率评测。
- **核心结果**：
  - **LLaMA2-7B**：JI²S-1k 平均得分 **64.62**，超越 Alpaca-52k（63.34）、LIMA-1k（64.40）、DEITA-1k（62.33）；MT-Bench 达 **5.40** 为全场最高。
  - **Mistral-7B**：JI²S-1k 平均 **72.52**，显著超越 Alpaca-52k（67.70）及所有 1k 筛选基线；MT-Bench 同样登顶 5.40。
  - **LLaMA2-13B**：JI²S-1k 平均 **69.65**，仅次于 LIMA-1k（70.09），体现方法在更大模型上的稳健性。
- **关键提升**：Mistral-7B 较全量训练提升约 **+4.82** 分；LLaMA2-7B 较全量提升 **+1.28** 分，印证“少而精”优于“多而噪”。
- **消融与扩展**：λ=0.1 时 LLaMA2-7B 综合最优；即使剔除联合项（λ=0）已强于多数基线；将子集扩至 2k/3k 无显著增益；UMAP 可视化显示筛选样本均匀覆盖嵌入空间核心区与边缘区，保留语义多样性。

## 相关工作脉络
1. **GPT 打分筛选（AlpaGasus, DEITA）**：依赖外部大模型独立评分指令质量，存在系统性偏差且与目标模型优化轨迹解耦；JI²S 转向数据对目标模型的真实梯度贡献，提升泛化与可解释性。
2. **边际影响函数（LESS, Koh & Liang）**：基于 SGD/Adam 梯度估算单样本边际贡献，计算高效但假设贡献可加，忽略样本冗余；JI²S 引入二阶联合影响弥补高阶交互盲区。
3. **精标小数据集（LIMA）**：证明 1k 高质量数据可超越 52k 自动生成数据，但依赖人工策展；JI²S 将 LIMA 转为代理测试集而非直接训练集，实现自动化高质量筛选。
4. **数据规模-质量权衡（Textbooks are all you need, Phi-2）**：强调数据质量主导模型上限；JI²S 提供可复现的自动化筛选范式，避免手工策展瓶颈。
5. **Shapley 交互与离散导数（Faith-Shap, Simfluence）**：理论框架相似但计算呈指数复杂度；JI²S
