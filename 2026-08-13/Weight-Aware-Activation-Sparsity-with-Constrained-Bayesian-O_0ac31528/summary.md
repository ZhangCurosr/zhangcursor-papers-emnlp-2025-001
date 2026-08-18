---
title: "Weight-Aware-Activation-Sparsity-with-Constrained-Bayesian-O"
source: https://aclanthology.org/2025.emnlp-main.57.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:45:34"
field: "大语言模型推理加速与压缩"
keywords: ["activation sparsity", "model compression", "LLM inference acceleration", "training-free pruning", "Bayesian optimization", "GPU kernel"]
innovations: ["权重感知激活稀疏阈值策略", "约束TPE块间稀疏分配", "定制Triton GPU kernel支持非均匀稀疏"]
benchmarks: ["WikiText2", "C4", "ARC-Easy", "ARC-Challenge", "HellaSwag", "PIQA", "Winogrande", "GSM8K", "MMLU"]
---

# 论文速读：Weight-Aware Activation Sparsity with Constrained Bayesian Optimization Scheduling for Large Language Models

## 一句话总结
本文提出 WAS（Weight-Aware Activation Sparsity），一种无需训练的激活稀疏框架，通过联合考虑激活值与对应权重的耦合关系来确定稀疏阈值，并引入约束贝叶斯优化为不同 Transformer 块动态分配块级稀疏率，在 60% 模型级稀疏下实现与最强基线相当的性能，在 75% 高稀疏下显著优于现有技术，最高获得 1.68× 推理加速。

## 研究问题与动机
1. **现代 LLM 缺乏天然激活稀疏性**：早期 ReLU 模型（如 OPT）因 ReLU 函数特性天然产生稀疏激活，但现代 LLM 采用 SwiGLU 等 GLU 变体激活函数，不再具备这种天然稀疏性，需额外训练（如 ReLUfication）才能恢复，成本高昂。
2. **现有方法仅依赖激活幅度**：CATS、TEAL 等 training-free 方法仅根据激活值大小决定稀疏化，忽略了与激活相乘的权重也同等重要——相同激活值乘以大权重和小权重对输出误差的贡献截然不同。
3. **统一稀疏率忽视块级敏感性差异**： prior work 对所有 Transformer 块应用相同稀疏率，但论文观察到浅层块对稀疏更敏感（降低浅层稀疏率带来更大性能收益），深层块相对不敏感，统一稀疏并非最优策略。
4. **高稀疏场景下性能退化严重**：在 75% 极端稀疏下，现有方法（如 WANDA、CATS）性能严重退化，需更精细的稀疏决策机制。

## 核心贡献（创新点）
1. **权重感知稀疏化阈值策略**：首次将权重列范数引入激活稀疏化决策，数学推导证明稀疏误差上界为 `|x_i| × ||W_{i,:}^T||_1`，与仅用激活幅度的方法本质不同，能在相同激活值下保留对大权重更重要的激活。
2. **约束贝叶斯优化实现块间稀疏分配**：提出基于 TPE（Tree-structured Parzen Estimator）的约束优化算法，利用"浅层敏感度更高"的单调性假设对搜索空间施加约束（前一层稀疏率作为后一层下界），在 50 次迭代内高效找到最优块级稀疏配置，而非暴力搜索或均匀分配。
3. **定制 GPU 稀疏 Kernel 支持端到端加速**：基于 Triton 实现自定义 Kernel，将权重感知 mask 生成直接嵌入计算流水线，支持非均匀稀疏率和块内 Q/K/V 等子结构差异化阈值，在 A800 GPU 上实现最高 1.68× 单批解码加速。
4. **无需训练、无需权重更新**：方法完全 training-free，仅需少量校准序列（10 条 2048-token Alpaca 序列）确定阈值，可直接部署于现有 LLM 推理系统。

## 方法详解
### 3.2 权重感知激活稀疏化
传统稀疏化仅比较 `|x_i|` 与阈值 τ，本文推导稀疏误差上界：
```
L = ||xW^T - x'W^T|| ≤ Σ_i |x_i - x'_i| × ||W_{i,:}^T||_1
```
因此 mask 决策改为：
```
mask(x_i, W) = 0, if |x_i| × ||W_{i,:}^T||_1 < τ; else 1
```
其中 `||W_{i,:}^T||_1` 为权重矩阵第 i 列的 L1 范数，可离线预计算为 m 维向量，几乎不增加开销。

阈值 τ_p 由目标稀疏率 p 通过缩放后激活值的分布分位数确定：
```
(1/m) Σ_i P(|x_i| × ||W_{i,:}^T||_1 ≤ τ_p) = p
```
论文验证缩放后激活值仍保持零均值对称单峰分布（类高斯/拉普拉斯），使分位数阈值可靠。

### 3.3 块间稀疏分配（Inter-Block）
定义模型级稀疏率为各块稀疏率的均值：`S = (1/L) Σ_l p_l`。利用 TPE 采样块稀疏率向量 s ∈ ℝ^L，约束条件为 `s_i ≥ s_{i-1}`（单调递增），且满足 `|(1/L) Σ s_i - r| ≤ ε`（r 为目标稀疏率）。每次 trial 用 WikiText2 perplexity 评估，选最优配置。

### 3.4 块内稀疏分配（Intra-Block）
对块内 Q/K/V/MLP 等子组件，因阈值操作不可微，STE 梯度回传失败，改用贪心搜索（Algorithm 2）：逐步将稀疏增量分配给 MSE 下降最小的组件，直至达到目标块级稀疏率。

### 3.5 GPU Kernel
基于 DejaVu 的 Triton kernel 改进：① 将权重范数纳入 mask 生成；② 支持非均匀稀疏；③ 支持块内子结构差异化阈值；FP16 累加沿 SplitK 维度降内存，cache-aware 调度优先复用跨 thread block 的激活。

## 实验与结果
**模型**：LLaMA-2 (7B/13B/70B)、LLaMA-3 (8B/70B)、LLaMA-3.1 (8B/70B)、Mistral-7B  
**数据集**：WikiText2、C4（perplexity）；ARC-Easy/Challenge、HellaSwag、PIQA、Winogrande、GSM8K、MMLU（zero/few-shot）  
**基线**：WANDA（weight pruning）、TEAL（training-free activation sparsity SOTA）、CATS（仅 40% 稀疏有效）

**主要结果**（Table 1, WikiText2）：
- 60% 稀疏：WAS 与 TEAL 相当，LLaMA-2-7B 上 6.56 vs 6.80，LLaMA-3-8B 上 8.30 vs 10.04
- **75% 极端稀疏**：WAS 显著领先，LLaMA-2-7B 上 12.76 vs TEAL 42.15（降低 29.39），LLaMA-3-8B 上 28.33 vs 87.48

**推理加速**（Figure 4, A800 GPU）：
- 60% 稀疏：最高 1.52×
- 75% 稀疏：最高 1.68×

**消融**（Table 3, LLaMA-2-7B @75%）：
- Greedy + 无权重 + 无 TPE（TEAL）：Wiki-2 42.15
- + 约束 TPE：21.13（↓21.02）
- + 权重范数：19.21（↓2.92）
- + 两者结合：**12.76**（总↓29.39）

**约束优化有效性**（Table 4）：75% 稀疏下，约束 TPE 比无约束 TPE 性能更好且收敛更快（50 vs >100 trials）。

## 相关工作脉络
1. **WANDA (Sun et al., 2023)**：权重量化剪枝 SOTA，通过权重量级×激活幅度选择重要权重列；本文与之区别在于关注激活稀疏而非权重剪枝，且显式推导误差上界证明权重范数必要性。
2. **TEAL (Liu et al., 2024)**：当前 training-free 激活稀疏 SOTA，对全模型组件输入应用稀疏化；本文比其多一层权重感知和块间优化，75% 稀疏下 perplexity 降低 29-36。
3. **CATS (Lee et al., 2024)**：仅稀疏 MLP gate 输出，40% 稀疏即严重退化（Wiki-2 46.87），因其仅覆盖 MLP 子集；WAS 作用于全模型。
4. **DejaVu (Liu et al., 2023)**：基于残差连接的上下文稀疏，用轻量 MLP 预测低范数神经元；本文不依赖残差假设，适用于更广泛架构。
5. **ReLUfication (Mirzadeh et al., 2023) / ProSparse (Song et al., 2024)**：通过替换激活函数+持续预训练恢复稀疏性，成本高昂；本文完全 training-free。
6. **LLM_Pruner (Ma et al., 2023) / OWL (Yin et al., 2023)**：块级/层权重剪枝；本文聚焦激活侧稀疏，与权重剪枝正交。

## 局限性与未来方向
1. **激活分布假设限制**：方法依赖激活值缩放后仍保持零均值对称单峰分布，对非对称/多峰分布的模型（如某些 MoE 或新兴架构）可能失效。
2. **仅覆盖 training-free 场景**：受硬件资源限制未探索含训练的扩展，融入微调可能进一步提升高稀疏性能。
3. **TPE 优化依赖校准集**：阈值确定需 10 条 Alpaca 序列做校准，不同校准集可能影响泛化。
4. **块内贪心搜索复杂度**：Intra-block 贪心算法随组件数量线性增长，大模型（70B）块内组件多时搜索开销较大。
5. **未来方向**：扩展至更广泛架构（MoE、多模态）、探索训练辅助版本、自适应稀疏率（按 token/位置动态调整）。

## 研究启发与可借鉴点
1. **误差上界推导指导稀疏决策**：从数学上证明"激活×权重范数"比单一激活幅度更优，这种从近似误差出发的设计思路可迁移至其他稀疏化/剪枝场景。
2. **约束优化降维策略**：将单调性先验（浅层更敏感）转化为优化约束，既缩小搜索空间又提升稳定性，对高维超参搜索有普遍借鉴价值。
3. **distribution shape 稳定性验证**：论文反复验证缩放后激活分布形状不变（Appendix B Figures 6-8），这种对统计正则性的依赖与验证方法值得复用。
4. **Kernel 与算法协同设计**：将权重感知 mask 直接嵌入 Triton kernel 而非后处理，避免额外内存读写，对工程落地有高参考价值。
5. **块级敏感性分析的可视化呈现**：Figure 1b 用"分组交换稀疏率"直观展示位置敏感性，这种消融式设计可复用为理解模型架构特征的通用方法。

## 关键术语表
**Activation Sparsity**：推理时根据激活值动态置零不重要的中间激活，跳过对应权重计算，实现输入依赖的动态稀疏。  
**Training-free**：无需额外训练或微调，仅利用校准数据和模型现有权重完成稀疏化配置的方法类别。  
**TPE (Tree-structured Parzen Estimator)**：贝叶斯优化的一种变体，用核密度估计替代高斯过程，更适合高维超参搜索。  
**Constrained Bayesian Optimization**：在优化过程中加入先验约束（如单调性）以减少搜索空间、加速收敛并避免局部最优。  
**WANDA Score**：权重量化剪枝中用于排序的指标 `|x_i| × ||W_{i,:}^T||_1`，本文将其推广至激活稀疏决策。  
**SplitK**：GPU 并行计算中沿输出维度切分累加的策略，用于减少共享内存压力。  
**Cache-aware Scheduling**：GPU kernel 调度策略，优先处理被多个 thread block 复用的数据以提升缓存命中率。  
**Perplexity (PPL)**：语言模型评估指标，PPL = exp(-平均 log-likelihood)，越低越好。

## 可复现要素
- **代码**：已开源，https://github.com/HITSZ-Miao-Group/WAS
- **数据集**：WikiText2、C4、Alpaca（校准用）均为公开数据集
- **模型**：LLaMA-2/3/3.1、Mistral-7B 均为开源模型
- **关键超参**：TPE 迭代次数 50；校准序列 10 条 × 2048 token；贪心步长 α（论文未明确给出数值，见 Appendix D）；目标稀疏率 r = 0.4/0.6/0.75
- **硬件**：A800 GPU；LLaMA-3-70B 使用 TP2（Tensor Parallelism 2）
- **评估框架**：lm-evaluation-harness
- **精度**：FP16
