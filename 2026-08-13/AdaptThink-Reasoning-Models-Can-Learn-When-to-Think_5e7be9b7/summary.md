---
title: "AdaptThink-Reasoning-Models-Can-Learn-When-to-Think"
source: https://aclanthology.org/2025.emnlp-main.184.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:39:56"
field: "大模型高效推理"
keywords: ["推理效率", "自适应思考", "强化学习", "Large Reasoning Models", "NoThinking"]
innovations: ["提出 AdaptThink RL 算法实现 Thinking/NoThinking 模式自适应选择", "设计约束优化目标与重要性采样解决冷启动问题"]
benchmarks: ["GSM8K", "MATH500", "AIME2024", "MMLU"]
---

# 论文速读：AdaptThink-Reasoning-Models-Can-Learn-When-to-Think

## 一句话总结
本文提出 AdaptThink，一种新的强化学习算法，教推理模型根据输入问题难度自适应选择 Thinking（长链思考）或 NoThinking（直接输出答案）模式，在保证甚至提升准确率的同时显著降低推理开销。

## 研究问题与动机
1. **推理效率瓶颈**：现有大推理模型（如 DeepSeek-R1、OpenAI o1）依赖冗长的链式思考过程，导致推理延迟高、计算开销大，对简单问题产生不必要的冗余思考。
2. **全统一思考模式的不足**：现有高效推理方法（如 DPO、长度惩罚 RL）仅在 Thinking 模式内部压缩响应长度，未考虑"是否应该思考"这一更根本的决策问题。
3. **NoThinking 被低估**：前作 NoThinking（Ma et al., 2025）仅在低 token 预算下验证有效性，本文进一步简化为空 thinking 段提示，并系统证明在简单问题上 NoThinking 可超越 Thinking 同时大幅节省 token。
4. **冷热启动挑战**：若直接从 Thinking-only 策略开始训练，模型难以探索 NoThinking 模式，需要显式的样本平衡机制。

## 核心贡献（创新点）
1. **提出 AdaptThink 自适应思考模式选择算法**：通过 RL 教模型根据问题难度动态选择 Thinking 或 NoThinking，与现有"仅压缩 Thinking 长度"的方法本质不同。
2. **设计约束优化目标**：将"最大化 NoThinking 比例"作为主目标，"准确率不低于参考模型"作为约束，实现效率与性能的双赢。
3. **提出重要性采样策略解决冷启动**：通过在初始步强制生成一半 Thinking/一半 NoThinking 样本，避免模型坍缩至单一模式，贯穿整个训练过程的探索。
4. **系统性验证 NoThinking 在简单问题上的优势**：在 MATH500 五个难度等级上进行详细分析，揭示 Thinking 收益仅在难题上显著。

## 方法详解
**整体框架**：基于 PPO-style on-policy RL，在数学问题数据集上训练推理模型。

**核心组件一：约束优化目标（Section 4.1）**
- 定义指标函数 $\mathbb{1}(y_1 = \texttt{</think>})$ 判断是否为 NoThinking 响应（首 token 为结束标记）。
- 优化目标：最大化 NoThinking 概率，同时约束平均准确率不低于参考模型 $\pi_{\theta_{ref}}$：
$$\max \mathbb{E}[\mathbb{1}(y_1=\texttt{</think>})] \quad s.t. \quad \mathbb{E}[R(x,y)] \geq \mathbb{E}[R(x,y')]$$
- 拉格朗日变换后得到 advantage 函数：$A(x,y) = \mathbb{1}(y_1=\texttt{</think>}) \cdot \delta + R(x,y) - \bar{R}_{ref}(x)$，其中 $\delta$ 控制 NoThinking 偏好强度，$\bar{R}_{ref}$ 为参考模型预采样均值奖励。
- 最终使用无 KL 惩罚的 PPO clip loss。

**核心组件二：重要性采样（Section 4.2）**
- 问题：初始 $\pi_\theta$ 对所有样本生成 Thinking，无法自然探索 NoThinking。
- 解法：构造重要性采样分布 $\pi_{IS}$，在第一步 $t=1$ 强制以 0.5 概率生成 `</think>` 或起始词 $w_{start}$，后续 token 仍按 $\pi_{\theta_{old}}$ 采样。
- 效果：每个训练步均获得混合 Thinking/NoThinking 样本，支持全程探索与利用。
- 最终 loss（Equation 9）：
$$\mathcal{L}_{AT}(\theta) = -\mathbb{E}_{x,y\sim\pi_{IS}}\left[\min\left(\frac{\pi_\theta(y|x)}{\pi_{IS}(y|x)}A, \text{clip}\left(\frac{\pi_\theta(y|x)}{\pi_{IS}(y|x)}, 1-\epsilon, 1+\epsilon\right)A\right)\right]$$

**理论分析（Section 4.3）**：
- 当 $\bar{R}_{nothink} + \delta > \bar{R}_{think}$ 且 $\bar{R}_{nothink} + \delta > \bar{R}_{ref}$ 时，模型倾向于选择 NoThinking，即简单问题满足此条件。

## 实验与结果
**训练数据**：DeepScaleR（40K 数学题，来自 AIME、AMC、Omni-Math、STILL）。

**评估基准**：GSM8K（初等）、MATH500（高中竞赛）、AIME2024（奥赛），温度 0.6，上下文 16K。

**模型**：DeepSeek-R1-Distill-Qwen-1.5B 与 7B。

**关键结果（Table 1）**：
| 模型 | 数据集 | Accuracy↑ | Length↓ | NT Ratio |
|------|--------|-----------|---------|----------|
| **1.5B** | GSM8K | 83.1 (+4.1) | 480 (-50.9%) | 86.9% |
| | MATH500 | 82.0 (+1.4) | 1782 (-63.5%) | 76.8% |
| | AIME2024 | 31.0 (+1.6) | 6679 (-44.7%) | 40.4% |
| **平均** | — | **+2.4%** | **-53.0%** | — |
| **7B** | GSM8K | 91.0 (+3.1) | 309 (-54.7%) | 99.6% |
| | MATH500 | 92.0 (+1.8) | 1875 (-49.0%) | 76.6% |
| | AIME2024 | 55.6 (+2.1) | 8599 (-16.6%) | 6.3% |
| **平均** | — | **+2.3%** | **-40.1%** | — |

**基线对比**：优于 DPO_Shortest、OverThink、DAST、O1-Pruner、TLMRE、ModelMerging、RFT_MixThinking 等所有基线。

**超参分析（Table 2）**：$\delta=0.05$ 为最佳平衡点；即使 $\delta=0$，GSM8K/MATH500 上仍选择 50%+ NoThinking，说明无奖励偏移时 NoThinking 本身具有内在优势。

**重要性采样验证（Figure 4）**：对比 naive GRPO，AdaptThink 在训练后期持续降低长度至 3000 以下，而 GRPO 长度先降后升。

**隐式思考分析（Table 3）**：NoThinking 响应中隐式思考比例仅略增（7B 模型从 0.7%→4.2%），长度增幅可控。

**OOD 泛化（Table 4）**：在 MMLU（14K 选择题，57 领域）上，AdaptThink 长度降低 31.9%-38.8%，准确率持平或略升，NT ratio 约 16%。

## 相关工作脉络
1. **Efficient Reasoning for LRMs**：DPO_Shortest、OverThink、DAST、O1-Pruner、TLMRE 等均通过长度惩罚/偏好优化在 Thinking 模式内压缩响应，未涉及模式切换；AdaptThink 从"何时思考"这一更高维度解决问题。
2. **NoThinking（Ma et al., 2025）**：首次提出 reasoning model 可跳过思考直接输出答案，但仅在低 token budget 场景验证；本文简化提示方式（空 thinking 段）并系统研究难度自适应选择。
3. **Model Merging（Wu et al., 2025; Team et al., 2025）**：通过权重平均合并推理/非推理模型实现长度压缩，但损失推理能力；AdaptThink 保留完整推理能力并按需启用。
4. **RL-based Reasoning**：DeepScaleR、Kimi K1.5 等通过 RL 提升推理能力但未考虑效率；本文在 RL 框架内引入效率-性能联合优化。

## 局限性与未来方向
1. **规模限制**：仅在 1.5B/7B 模型上验证，更大规模模型的效果待探索。
2. **训练数据域局限**：仅使用数学数据集（因易获取且奖励可验证），通用领域需更多可验证奖励数据。
3. **隐式思考问题**：NoThinking 响应中偶现隐含思考关键词（如 "Wait"、"Alternatively"），可通过零奖励惩罚进一步抑制。
4. **$\delta$ 超参敏感**：需根据任务类型调优，缺乏自动调节机制。

## 研究启发与可借鉴点
1. **约束优化思想可迁移**：将"性能不降级"作为硬约束、"效率指标"作为软目标的建模方式，可复用于其他资源受限的推理场景（如多模态、代码生成）。
2. **重要性采样解决探索瓶颈**：当模型初始策略完全偏向某模式时，人工注入混合样本的策略设计具有普适价值，可推广至工具调用、搜索策略等离散决策学习。
3. **难度自适应范式**：除数学外，可探索在代码理解、科学 QA 等具有天然难度梯度的任务上验证该思路。
4. **隐式思考检测与抑制**：本文提出的关键词检测法可作为评估指标，未来可设计显式正则项引导纯 NoThinking 输出。

## 关键术语表
**Thinking Mode**：模型生成包含反思、回溯、自我验证的长链式思考过程后再输出答案的模式。
**NoThinking Mode**：模型跳过思考过程，直接输出最终解决方案的模式，通过空 `<think></think>` 提示触发。
**AdaptThink**：本文提出的 RL 算法，教模型根据问题难度自适应选择 Thinking/NoThinking 模式。
**Importance Sampling**：在训练初期强制生成混合 Thinking/NoThinking 样本的技术，解决冷启动与模式坍缩问题。
**Constrained Optimization**：以 NoThinking 比例为优化目标、以准确率为约束的拉格朗日优化框架。
**DeepScaleR**：本文训练数据集，包含 40K 数学题，源自 AIME、AMC、Omni-Math、STILL。
**Ratio_NT**：模型生成 NoThinking 响应的比例，衡量效率提升程度。
**Implicit Thinking**：NoThinking 模式下仍出现的隐含推理行为，表现为答案中包含思考关键词。

## 可复现要素
- **数据集**：训练集 DeepScaleR（40K 题），评估集 GSM8K、MATH500、AIME2024、MMLU；论文未明确声明开源，但均使用公开数据集。
- **代码/权重**：基于 VeRL 框架实现；论文未声明代码开源，模型权重未提供下载链接。
- **关键超参**：K=16（每样本采样数），δ=0.05（NoThinking 偏好权重），ε=0.2（PPO clip 范围），学习率 2e-6，batch size 128，训练 1 epoch（314 steps）。
