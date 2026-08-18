---
title: "Efficient-Model-Development-through-Fine-tuning-Transfer"
source: https://aclanthology.org/2025.emnlp-main.131.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:48:09"
---

# 论文速读：Efficient Model Development through Fine-tuning Transfer

## 一句话总结
本文提出了一种跨模型版本的微调迁移方法，通过提取源模型微调产生的权重差值向量（diff vector）并直接叠加至目标基础模型，无需额外训练即可显著提升目标模型的指令遵循、推理与多语言能力；该迁移向量还可作为更高效的微调初始化起点，并支持迭代式累积以适配持续版本开发场景。

## 研究问题与动机
- 现代 LLM 开发普遍依赖“预训练 + 后对齐/微调”两阶段流水线，但每次基础模型版本更新都需重复昂贵的对齐过程，在领域或语言专属场景中成本尤为突出。
- 现有模型合并（Model Merging）与方法多假设共享同一基础模型，聚焦跨任务/跨领域复用；缺乏针对**不同模型版本**之间完整微调权重更新迁移的系统性研究。
- 实际业务中希望复用历史对齐知识以跳过重训，但迁移有效性的边界条件（如参数空间距离、基座能力阈值、对泛化的影响）尚未被充分验证。
- 多语言与持续演进场景下，需要一种低开销的“版本升级兼容”机制，避免为新基座重新从头训练多语言/专业指令模型。

## 核心贡献（创新点）
- **提出 Diff Vector 跨版本微调迁移框架**：直接计算 $\Delta_s = m'_s - m_s$ 并加到目标基座 $m_t$ 上实现零训练迁移；与 Task Arithmetic 等仅作用于同基座多任务合并的方法本质不同，本文解决的是异构版本间的通用能力继承。
- **实证刻画迁移有效性边界**：通过 OLMo 2 中间检查点的受控实验证明，迁移效果最佳当源/目标模型处于参数空间的线性连通区域内，且目标基座需跨越一定能力门槛才能有效吸收迁移更新。
- **提出 Transferring-then-Finetuning 范式**：将合并模型作为后续微调的起始 checkpoint，可显著加速收敛并提升 Seen/Unseen 任务准确率，且不引发泛化退化。
- **设计 Iterative Recycling-then-Finetuning 策略**：面向连续版本迭代场景，逐步累积历史 diff vector 并向下游传递，在提升性能的同时进一步压缩对齐算力开销。

## 方法详解
- **Diff Vector 计算与直接迁移**：对同源架构的源模型，计算微调前后参数差值 $\Delta_s = m'_s - m_s$。该向量编码了特定对齐过程（如指令微调、偏好优化）引入的参数适应量。直接执行 $m_t + \Delta_s$ 得到目标近似微调模型，全程无需梯度更新。
- **线性模式连通性理论支撑**：基于 Frankle 等人提出的线性模式连通性（Linear Mode Connectivity），假设 $m'_s$ 与 $m'_t$ 可在参数空间中由低损失线性路径相连：$m(\lambda) = (1-\lambda)m'_s + \lambda m'_t$。代入 $\Delta$ 定义并近似 $\Delta_s \approx \Delta_t$，推得 $m'_t \approx m_t + \Delta_s$，为直接加法迁移提供理论保证。
- **迁移-微调联合训练（Transferring-then-Finetuning）**：在 $m_t + \Delta_s$ 基础上执行少量步数的监督微调（FT），利用迁移向量提供的良好初始化加速收敛，并通过后续梯度进一步弥合与全量重训模型的差距。
- **迭代回收算法（Algorithm 1）**：对版本序列 $\mathcal{M}_1, \mathcal{M}_2, \ldots, \mathcal{M}_n$，第 $i$ 步将上一轮累积的 diff vector $\Delta_{i-1}^{iter}$ 加到当前基座 $\mathcal{M}_i$ 后微调；新微调模型再计算新一轮 diff vector 传递至 $\mathcal{M}_{i+1}$，实现历史对齐知识的滚动复用。

## 实验与结果
- **数据集与基准**：Llama 3.0/3.1 8B 指令微调版本；多语言场景使用 Aya（Malagasy/Sinhala）与 InstrucTurca（Turkish）；控制实验采用 OLMo 2 7B 五个预训练中间检查点（$\mathcal{M}_1$–$\mathcal{M}_5$）。评测覆盖 GSM8K、MATH/MATH500、ARC_C、GPQA、MMLU、IFEval、HumanEval+、MBPP+、LiveCodeBench、BigCodeBench 及 Global MMLU。
- **零训练跨版本迁移**：将 Llama 3.0 8B Instruct 的 diff vector 迁移至 Llama 3.1 8B，IFEval 绝对提升 **46.9%**，Live-CodeBench 提升 **15.7%**，整体表现超越或持平 Llama 3.1 8B Instruct。
- **多语言高效开发**：将 Llama 3.0 Instruct 的 Malagasy/Turkish 专属微调 diff vector 回收至 Llama 3.1 Instruct，Global MMLU 上 Malagasy 提升 **4.7%**、Turkish 提升 **15.5%**，优于或持平源语言模型，Sinhala 因 3.1 已更强而未获增益。
- **有效性边界（OLMo 2 实验）**：强基座（$\mathcal{M}_4, \mathcal{M}_5$）更能利用迁移更新，$\mathcal{M}_5 + \Delta_4$ 在 GSM8K 上达 77.1%，反超直接微调的 75.5%；源/目标检查点处于同参数连通区域（如均属于 Stage 1 或 Stage 2 子群）时迁移增益最大，跨群迁移可能退化。
- **迁移-微调与迭代回收**：以迁移模型为起点微调，GSM8K 最高提升
