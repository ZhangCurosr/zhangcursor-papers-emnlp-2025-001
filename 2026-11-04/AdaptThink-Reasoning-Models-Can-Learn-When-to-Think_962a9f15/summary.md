---
title: "AdaptThink-Reasoning-Models-Can-Learn-When-to-Think"
source: https://aclanthology.org/2025.emnlp-main.184.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:24:54"
field: "大模型高效推理"
keywords: ["推理模型", "强化学习", "思考模式选择", "推理效率", "自适应推理", "NoThinking"]
innovations: ["提出约束优化目标在保持准确率前提下鼓励NoThinking", "重要性采样策略解决双模式RL训练的冷启动问题", "实验验证模型能按难度自适应切换Thinking/NoThinking模式"]
benchmarks: ["GSM8K", "MATH500", "AIME2024", "MMLU"]
---

# 论文速读：AdaptThink: Reasoning Models Can Learn When to Think

## 一句话总结
论文提出了一种名为 AdaptThink 的新型强化学习算法，使推理模型能够根据问题难度自适应地选择 Thinking 或 NoThinking 模式，在大幅降低推理成本的同时进一步提升准确率。

## 研究问题与动机
- 大型推理模型（如 DeepSeek-R1、OpenAI o1）通过生成冗长的 Chain-of-Thought 提升推理能力，但导致巨大的推理开销和延迟。
- 现有方法（长度惩罚 RL、偏好微调、模型合并）仍对所有问题一律使用 Thinking，忽视了问题难度差异带来的效率冗余。
- 论文发现 NoThinking（跳过思考直接生成最终答案）在简单问题上准确率可比甚至优于 Thinking，且显著节省 token。
- 核心问题：如何让模型学会根据输入问题的难度自适应选择最优思考模式？

## 核心贡献（创新点）
- **简化 NoThinking 并验证其优势**：论文将 NoThinking 简化为强制首个 token 为 `</think>`，证明在简单问题上无需冗长思考即可获得相当或更好的性能。
- **提出 AdaptThink 约束优化目标**：通过约束优化公式在确保整体准确率不下降的前提下，最大化模型选择 NoThinking 的概率，与仅压缩 Thinking 长度的已有工作本质不同。
- **重要性采样解决冷启动**：通过重要性采样策略使训练初期同时获得 Thinking 和 NoThinking 样本，避免模型因初始策略只能生成 Thinking 而永远无法学习 NoThinking。
- **系统性实验验证自适应选择**：在 GSM8K、MATH500、AIME2024 及 OOD 的 MMLU 数据集上，证明模型确实按难度自适应切换模式。

## 方法详解
- **约束优化目标**：目标是在保证整体准确率不低于参考模型的前提下最大化 NoThinking 概率，转化为带惩罚项的无约束优化：`max E[1(y₁=</think>)·δ + R(x,y) - R̄_ref(x)]`，其中 δ 控制对 NoThinking 的偏好程度，R̄_ref 通过参考模型预采样得到。
- **PPO 风格损失函数**：采用截断策略梯度（PPO-clip）形式，优势函数为 `A(x,y) = 1(y₁=</think>)·δ + R(x,y) - R̄_ref(x)`，无需 KL 惩罚。
- **重要性采样策略**：定义采样分布 π_IS，在第一步以 50% 概率采样 `</think>`（NoThinking）和 50% 概率采样常见起始词 w_start（Thinking），t>1 时直接使用 π_θ_old；最终损失为重要性加权的 PPO-clip 形式，确保训练全程两种模式均被探索和利用。
- **算法流程**：首先从参考模型预采样 K 个响应计算基线奖励，随后每个训练步更新旧策略，按重要性采样分布生成混合思维模式的批次，利用 AdaptThink 损失更新模型，共训练 1 epoch。

## 实验与结果
- **数据集与模型**：使用 DeepScaleR（40K 数学题）训练，评估于 GSM8K、MATH500、AIME2024，基线模型为 DeepSeek-R1-Distill-Qwen-1.5B 和 7B。
- **主要结果（1.5B 模型）**：AdaptThink 相比原始 Thinking 将平均响应长度降低 **53.0%**，同时准确率提升 **+2.4%**；在 GSM8K 上长度降 50.9%（86.9% NoThinking）、MATH500 上降 63.5%（76.8% NoThinking）、AIME2024 上降 44.7%（40.4% NoThinking）。
- **主要结果（7B 模型）**：平均长度降低 **40.1%**，准确率提升 **+2.3%**；在 GSM8K 上 NoThinking 比例达 99.6%，AIME2024 上仍保留 6.3% NoThinking。
- **对比基线**：AdaptThink 在精度-长度综合指标上均优于 DPO_Shortest、OverThink、DAST、O1-Pruner、TLMRE、ModelMerging、RFT_MixThinking 等方法，是最优结果。
- **OOD 泛化**：在 MMLU 上 Reduce 38.8% 长度，Accuracy 提升 +6.5%（1.5B）和 +0.2%（7B），NoThinking 比例约 16%。
- **δ 敏感性**：δ 越大 NoThinking 比例越高、长度越短，但精度增益逐步下降；即使 δ=0 也能在简单问题上超过 50% NoThinking，说明 NoThinking 本身有内在优势。

## 相关工作脉络
- **NoThinking（Ma et al., 2025）**：提出用伪思考提示替代真实思考，本文进一步简化为空思考段并证明其在简单问题上优于 Thinking，同时扩展至自适应选择范式。
- **DPO_Shortest / OverThink / DAST / O1-Pruner / TLMRE**：均通过长度惩罚或偏好学习在 Thinking 模式下压缩响应，不改变思考模式本身；本文从根本上引入模式选择维度。
- **ModelMerging（Wu et al., 2025）**：通过权重平均融合推理/非推理模型，会牺牲模型原有能力；本文基于 RL 微调，保持推理能力同时增强效率。
- **RFT_MixThinking**：虽也使用双模式数据，但属于 reject sampling 后的监督微调，无法在推理时动态决策；本文通过 RL 实现在线自适应选择。
- **L1（Aggarwal and Welleck, 2025）**：通过 RL 控制思考长度而非思考模式，本文从"何时思考"角度切入，是正交互补方向。

## 局限性与未来方向
- 实验仅覆盖 1.5B 和 7B 模型，更大规模模型的效果尚待验证。
- 训练数据仅限于数学领域（易于获得精确奖励信号），虽 OOD 实验显示一定泛化，但期待更多通用可验证奖励任务上的验证。
- NoThinking 模式下仍存在少量隐式思考（implicit thinking），可通过对含思考关键词的响应施加零奖励来进一步抑制。

## 研究启发与可借鉴点
- **难度自适应决策框架**：约束优化形式（奖励保障+效率偏好）可迁移至其他"多模式推理/任务选择"场景，如代码生成中的分步调试 vs 直接生成。
- **重要性采样冷启动技巧**：对任何需要探索新行为模式（如多工具调用、多策略路由）的 RL 训练，均可采用类似的采样分布扰动策略。
- **离线参考奖励估计**：用参考模型预采样计算基线奖励 R̄_ref 替代 online KL 约束，可在不引入额外正则项的情况下实现稳定训练。
- **可结合团队方向**：若团队关注多模态推理或工具使用，可将 AdaptThink 思想扩展为"思考/工具调用/直接回答"的多模式自适应选择。

## 关键术语表
- **Large Reasoning Models (LRMs)**：通过强化学习或蒸馏获得深度思考能力的大型语言模型，如 DeepSeek-R1、OpenAI o1。
- **Thinking 模式**：模型生成冗长链式思考（Chain-of-Thought）后输出最终答案的推理方式。
- **NoThinking 模式**：模型跳过思考直接生成最终答案的模式，本文通过强制首 token 为 `</think>` 实现。
- **AdaptThink**：本文提出的 RL 算法，使模型能根据问题难度自适应选择 Thinking 或 NoThinking。
- **约束优化目标**：在保持整体准确率不低于参考模型的前提下，最大化 NoThinking 选择概率的优化形式。
- **重要性采样策略**：通过修改采样分布在训练初期同时生成 Thinking 和 NoThinking 样本，解决冷启动问题。
- **隐式思考（Implicit Thinking）**：NoThinking 模式下模型仍可能产生的类思考特征（如自我反思），本文评估其对纯 NoThinking 质量的潜在影响。
- **DeepScaleR**：本文使用的训练数据集，包含来自 AIME、AMC、Omni-Math 等来源的 40K 数学问题。

## 可复现要素
- **训练数据集**：DeepScaleR（公开）；**评估数据集**：GSM8K、MATH500、AIME2024、MMLU（均为公开）
- **代码**：基于 VeRL 框架构建，论文未开源
- **模型权重**：使用 DeepSeek-R1-Distill-Qwen-1.5B/7B（公开），AdaptThink 训练后权重论文未开源
- **关键超参**：K=16，δ=0.05，ε=0.2，学习率=2e-6，batch size=128，context size=16K，训练 1 epoch（314 steps）
