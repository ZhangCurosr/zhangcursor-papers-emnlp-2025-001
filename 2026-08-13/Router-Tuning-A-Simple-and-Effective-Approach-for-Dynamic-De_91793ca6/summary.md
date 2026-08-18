---
title: "Router-Tuning-A-Simple-and-Effective-Approach-for-Dynamic-De"
source: https://aclanthology.org/2025.emnlp-main.99.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:34:41"
field: "大语言模型效率优化"
keywords: ["Dynamic Depth", "Mixture of Depths", "Router-Tuning", "Attention Layer Skipping", "MoE Optimization", "Parameter-Efficient Fine-tuning"]
innovations: ["仅微调路由器的轻量化MoD方案，训练时间<15分钟，比DLO快1000倍", "系统探索Attention层动态深度的有效性，实现21%推理加速且仅0.2%性能下降", "扩展至MoE层动态专家跳过，缓解专家负载不均衡"]
benchmarks: ["LM-Harness", "ARC-C", "MMLU", "GSM8K", "HellaSwag"]
---

# 论文速读：Router-Tuning-A-Simple-and-Effective-Approach-for-Dynamic-De

## 一句话总结
论文提出了 Router-Tuning，一种仅微调路由网络（router）即可实现动态深度（MoD）的高效方法，通过冻结骨干模型参数大幅降低训练成本，并在 Attention 层和 MoE 层上验证了其有效性，实现了 21% 推理加速且仅 0.2% 性能下降。

## 研究问题与动机
1. **训练成本过高**：现有 MoD 方法（如 Raposo et al., 2024 从头预训练、Tan et al., 2024 持续训练）需要训练整个大语言模型，计算开销巨大，难以高效集成到已有 LLM 中。
2. **性能下降风险**：传统 MoD 通常作用于 Transformer Block 或 MLP 层，这些层对跳过分分钟敏感，跳过重要层会导致显著性能退化。
3. **静态层删除缺乏灵活性**：静态层删除方法（如 Layer Drop）无法适应不同输入序列的复杂度差异，对复杂任务表现不佳。
4. **如何平衡效率与性能**：如何在最小化额外训练开销的同时，实现动态深度且保持模型性能。

## 核心贡献（创新点）
1. **仅微调路由器的轻量化 MoD 方案**：Router-Tuning 只训练单层的轻量路由网络（参数量不到总参数的 0.01%），无需更新骨干模型，训练时间仅需约 15 分钟（A6000），比 DLO 快 1000 倍。与已有工作的本质区别在于完全冻结骨干，避免了灾难性遗忘。
2. **系统化探索路由作用目标与粒度**：首次系统对比了 Router-Tuning 在 Block、MLP、Attention 等不同模块以及 token/sequence 粒度的效果，发现 Attention 层动态深度在保持性能的同时可获得最大加速。
3. **扩展至 MoE 架构的动态专家跳过**：将 Router-Tuning 应用于 MoE 层的专家级别，通过动态跳过不重要的 token 来降低过载专家的负载，平衡专家利用率。
4. **与 LoRA 微调无缝兼容**：证明 Router-Tuning 可与 LoRA 联合训练，在保持计算效率增益的同时进一步提升下游任务性能。

## 方法详解
1. **路由器设计**：对于输入 $\pmb{x} \in \mathbb{R}^{L \times d}$，路由器计算重要性分数：
   - Token 级：$\pmb{R}(\pmb{x})_i = \text{GATE}(\pmb{x}_i)$
   - Sequence 级：$\pmb{R}(\pmb{x}) = \text{GATE}(\frac{1}{L}\sum_{i=1}^{L}\pmb{x}_i)$
   其中 $\text{GATE}(\pmb{x}) = \text{Sigmoid}(\pmb{W}\pmb{x})$，$\pmb{W}$ 是路由器的唯一可训练参数。

2. **二值掩码与直通估计器（STE）**：根据阈值 $\tau$ 生成二值掩码 $\pmb{M}$（1 保留，0 跳过），使用 STE 使梯度可通过二进制选择机制传播：$\frac{\partial \hat{\pmb{M}}}{\partial \pmb{R}} = 1$。

3. **MoD 输出计算**：训练时 $\pmb{y} = \pmb{M} \odot \pmb{F}(\pmb{x}) + \pmb{x}$；推理时直接跳过被路由为 0 的输入：$\pmb{y} = \pmb{F}(\pmb{x}) + \pmb{x}$（若 $\pmb{R}(\pmb{x}) \geq \tau$）或 $\pmb{y} = \pmb{x}$（否则）。

4. **MoE 扩展**：对每个专家 $E_i$，根据路由器分数决定是否执行：$\hat{E}_i(\pmb{x}) = E_i(\pmb{x})$（若 $\pmb{R}(\pmb{x}) \geq \tau$）或 $0$（否则）。整体输出为 Top-K 专家的加权和。

5. **训练目标**：$\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \cdot \mathcal{L}_{\text{MoD}}$，其中 $\mathcal{L}_{\text{task}}$ 为标准任务损失（如交叉熵），$\mathcal{L}_{\text{MoD}} = \text{ReLU}(\|\pmb{M}\|_0 - s)$ 为 $l_0$-范数正则项，控制 MoD 容量（非跳过输入比例）至目标稀疏度 $s$，$\lambda$ 为权衡系数。

## 实验与结果
1. **数据集与模型**：使用 Llama-Pro 作为训练数据；在 Llama-3-8B、Llama-2-13B、Mistral-7B、Qwen-2.5-7B/14B、DeepSeek-MoE、OLMoE 上评估；基准为 LM-Harness（ARC-C、BoolQ、HellaSwag、MMLU、OBQA、PIQA、RTE、WinoGrande、GSM8K）。
2. **模块选择实验**：Block/MLP 跳过导致严重性能下降（Llama-3-8B 平均 61.4 vs 69.7），而 Attention 层动态深度仅造成微小损失（69.4 vs 69.7），sequence 级跳过获得 1.21× 加速。
3. **对比动态跳过基线**：Router-Tuning（平均 61.4）优于 DLO（60.5）和 Skip Transformer（60.8）。
4. **静态 vs 动态**：在相同跳过率下，Router-Tuning 在 GSM8K 上比静态 Attention Drop 提升 6.5%（25% 跳过率）。
5. **效率提升**：训练时间 < 15 分钟（单卡 A6000），比 DLO（36 小时 A100）快 1000 倍；推理加速达 21%，KV Cache 减少 8GB（Llama-3-8B，序列长度 2048，batch=64）。
6. **MoE 实验**：在 DeepSeek-MoE 和 OLMoE 上，Router-Tuning 平均性能优于 Expert Drop（如 DeepSeek-MoE: 62.4 vs 61.1）。
7. **LoRA 集成**：Router-Tuning + LoRA 联合训练在保持 1.21× 加速的同时，Llama-3-8B 平均性能达 70.9，优于单独 LoRA（71.0）或 Router-Tuning（69.5）。

## 相关工作脉络
1. **Mixture of Depths (MoD)**：Raposo et al. (2024) 提出 MoD 框架，但需从头预训练；本文聚焦于在已有预训练模型上高效适配。
2. **Dynamic Layer Operation (DLO)**：Tan et al. (2024) 在预训练 Llama 上通过持续训练应用 MoD；本文方法训练成本降低三个数量级。
3. **Layer Drop / 静态层删除**：He et al. (2024b) 证明注意力层冗余；本文将其推广为动态、输入依赖的跳过机制。
4. **Early-Exit 策略**：Bae et al. (2023)、Elhoushi et al. (2024) 通过提前退出加速；本文提供更具灵活性的中间层跳过方案。
5. **Expert Drop**：He et al. (2024a) 静态丢弃不重要专家；本文实现动态专家跳过，保留所有专家潜力。
6. **参数高效微调（PEFT）**：LoRA 等；本文证明 Router-Tuning 与 PEFT 方法正交且可互补。

## 局限性与未来方向
1. **训练策略探索不足**：作者承认仅微调路由器是简化方案，更复杂的训练策略可能进一步提升性能。
2. **实验范围有限**：受计算资源限制，仅在少量模型和任务上验证，泛化性有待更广泛架构和应用场景的检验。
3. **阈值设置依赖经验**：当前使用固定阈值 $\tau = 0.5$，可能存在自适应阈值优化的空间。
4. **仅针对特定层**：目前主要聚焦 Attention 层，对其他组件（如归一化层）的动态跳过尚未探索。

## 研究启发与可借鉴点
1. **轻量级路由器的设计范式**：单层投影 + GATE 函数 + STE 的简洁设计可有效实现可微二进制选择，可迁移至其他动态计算场景。
2. **Attention 层作为动态深度的优先目标**：因二次复杂度和 KV Cache 内存开销，Attention 层跳过带来的加速收益最大，这一发现对后续工作具有指导意义。
3. **训练数据鲁棒性**：仅需 5K 样本即可有效训练路由器，且不同数据集（Alpaca、Evol-Instruct 等）对性能影响较小，降低了数据获取门槛。
4. **与 LoRA 的联合训练范式**：证明效率优化方法与性能优化方法可协同工作，为多目标模型压缩提供了新思路。
5. **MoE 负载平衡视角**：Router-Tuning 在 MoE 中不仅提升效率，还缓解了专家负载不均衡问题，为 MoE 优化提供了新角度。

## 关键术语表
**Mixture of Depths (MoD)**：动态深度框架，根据输入复杂度选择性激活/跳过模型层，以平衡计算效率与性能。
**Router-Tuning**：本文提出的方法，仅微调轻量路由器网络实现动态深度，冻结骨干模型参数。
**Straight-Through Estimator (STE)**：允许梯度通过不可微的二值选择操作的近似训练技术。
**MoD Capacity**：非跳过输入的比例，由 $l_0$-范数正则项控制，越低表示跳过越多、效率越高。
**Token-level vs Sequence-level**：两种路由粒度，前者逐 token 决策，后者对整个序列统一决策。
**KV Cache**：注意力层存储的键值缓存，跳过 Attention 层可显著减少其内存占用。

## 可复现要素
- **数据集**：训练使用 Llama-Pro（公开）；评估使用 LM-Harness 基准（公开）
- **代码**：已开源，链接 https://github.com/CASE-Lab-UMD/Router-Tuning
- **关键超参**：阈值 $\tau = 0.5$；学习率搜索范围 {1e-5, 2e-5, 5e-5, 1e-4, 2e-4}；$\lambda$ 搜索范围 {0, 0.001, 0.01, 0.1}；训练样本 5K 即可；目标稀疏度 $s$ 通过网格搜索确定
- **硬件环境**：NVIDIA RTX A6000
