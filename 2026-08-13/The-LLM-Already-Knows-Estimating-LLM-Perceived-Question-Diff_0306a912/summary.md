---
title: "The-LLM-Already-Knows-Estimating-LLM-Perceived-Question-Diff"
source: https://aclanthology.org/2025.emnlp-main.61.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:41:33"
field: "大语言模型推理优化"
keywords: ["LLM难度估计", "隐藏表示", "马尔可夫链", "值函数", "自适应推理", "Self-Consistency"]
innovations: ["提出仅利用目标LLM隐藏表示的难度估计方法，无需输出采样", "将自回归生成建模为马尔可夫链并定义值函数估计期望输出质量", "训练独立轻量级TD网络拟合值函数，不损害主模型通用能力"]
benchmarks: ["MMBench", "ScienceQA", "MathVista", "StrategyQA", "gsm8k", "commonsenseQA", "RLHF-V", "VLFeedback"]
---

# 论文速读：The-LLM-Already-Knows-Estimating-LLM-Perceived-Question-Diff

## 一句话总结
论文提出了一种仅利用目标LLM隐藏表示即可估计问题难度的方法，通过将自回归生成过程建模为马尔可夫链并训练一个轻量级值函数网络，在不生成任何输出token的情况下实现高效的难度估计，并成功应用于Self-Consistency、Best-of-N等自适应推理策略中。

## 研究问题与动机
- **核心问题**：如何在不依赖多次输出采样、辅助模型或微调目标模型的前提下，准确估计LLM对输入问题的感知难度？
- **现有方法不足**：（1）基于输出一致性的方法（如AG）需要多次采样，计算开销大；（2）使用辅助LLM评估的方法（如LLMs-Ranking）无法准确捕捉目标模型自身的感知；（3）微调目标模型的方法（如HaluSearch）可能损害模型的通用能力和安全性。
- **动机来源**：隐藏表示蕴含比最终输出更细粒度、语义更丰富的模型预测逻辑信息；初步可视化（Figure 1）表明，同一LLM对easy/hard问题的最后层隐藏表示存在明显分离。

## 核心贡献（创新点）
- **提出基于隐藏表示的零采样难度估计方法**：仅利用目标模型对输入的hidden representations，无需任何输出生成即可估计难度；与已有工作的本质区别在于完全避免了test-time重复采样。
- **将LLM生成过程建模为马尔可夫链并定义值函数**：把token级生成轨迹形式化为马尔可夫决策过程，通过Bellman方程定义值函数估计每个隐状态的期望输出质量；与已有工作的本质区别在于直接从模型内部状态推断难度，而非依赖外部输出或辅助模型。
- **设计了基于TD学习的轻量级训练目标**：使用两层全连接网络拟合值函数，通过时间差分（TD）误差最小化进行训练；与已有工作的本质区别在于不需要微调主LLM，仅训练一个独立轻量子网络。
- **将难度估计无缝集成到自适应推理策略中**：提出了Difficulty-Aware版本的Self-Consistency、Best-of-N和Self-Refine，在保持或提升准确率的同时显著减少token消耗。

## 方法详解
- **马尔可夫链建模**：将自回归生成过程 $H_{t+1}, y_{t+1} = f_\theta(H_t, y_t)$ 视为状态转移，其中 $s_t = \{H_t, y_t\}$ 为Markov状态，$H_t$ 为到时刻t为止的hidden representation序列，$y_t$ 为生成的token；初始状态 $s_0$ 由输入问题x唯一确定。
- **奖励函数设计**：仅在生成结束（$y_t = \text{EOS}$）时根据输出质量打分（采用ORM或正确性判定），其余时刻奖励为0。
- **值函数与Bellman方程**：定义 $V(s_t) = \mathbb{E}_{s_{t+1}}[R(s_t) + \gamma V(s_{t+1})]$，当 $y_t \neq \text{EOS}$ 时退化为 $V(s_t) = \gamma \mathbb{E}[V(s_{t+1})]$，终态直接返回奖励；$V(s_0)$ 即反映输入问题的感知难度（值越低越困难）。
- **训练目标（TD Loss）**：使用两层全连接网络 $\hat{F}_\phi$ 近似值函数，TD误差为 $\delta_t = \gamma \cdot \text{Reward}(\mathbf{y}) - \hat{F}_\phi(s_t)$（终态）或 $\gamma \hat{F}_\phi(s_{t+1}) - \hat{F}_\phi(s_t)$（中间态），最小化 $\mathcal{L}_{TD} = \mathbb{E}[\sum_t \delta_t^2]$；实际以最后一层hidden state $h_t$ 作为状态表征。
- **自适应推理策略**：在Self-Consistency/Best-of-N/Self-Refine中，若 $\hat{F}_\phi(s_0) > \tau$ 则直接单次生成，否则启用多次采样或迭代细化。

## 实验与结果
- **数据集**：MMBench、ScienceQA、MathVista（多模态+数学推理）；StrategyQA、gsm8k、commonsenseQA（纯文本）；以及RLHF-V、VLFeedback（开放式任务）。
- **基线方法**：prompt（指令模型自评）、AG（输出一致性）、LLMs-Ranking（辅助LLM评分）、HaluSearch-Gen/Critic（微调目标模型）。
- **评估指标**：ROC-AUC、Macro-F1、Easy-Acc、Hard-Acc、准确率、输出token数。
- **最强结果**：在Qwen2.5-VL-7B-Instruct上，MMBench ROC-AUC达**94.15%**、Macro-F1达**80.68%**；ScienceQA ROC-AUC达**93.09%**、Macro-F1达**79.48%**；整体在六项基准中均优于所有基线。
- **效率优势**：测试时只需前向传播获取hidden state，耗时远低于需要多次采样或调用辅助模型的方法（Figure 2）；训练时间仅需1.56分钟（MMBench），仅为HaluSearch-Gen的约1/3（Table 6）。
- **自适应推理效果**：在SC/BoN/SR策略下，本文方法以更少或相当的token消耗获得最高准确率（Figure 3）。

## 相关工作脉络
- **AG（Lee et al., 2025）**：基于目标模型多次采样的一致性评估难度；本文不依赖任何采样，仅用隐藏表示即可估计。
- **LLMs-Ranking（Wang et al., 2024b）**：引入辅助LLM作为裁判评估难度；本文直接利用目标模型自身内部信号，无需外部模型。
- **HaluSearch-Gen/Critic（Cheng et al., 2025）**：通过微调目标模型赋予其难度感知能力；本文仅训练一个独立轻量子网络，不损害主模型通用能力。
- **Self-Consistency / Best-of-N / Self-Refine**：标准重复采样推理策略；本文在此基础上引入难度感知的动态调度机制，实现计算资源的高效分配。
- **Representation-based interpretability工作**（Kong et al., 2024; Zhang et al., 2024b）：关注hidden representations的语义与对齐特性；本文首次将其系统性地用于难度估计任务。

## 局限性与未来方向
- **需要访问hidden representations**：对于部分闭源系统（如API仅返回输出），无法直接获取中间层表示，限制了适用范围。
- **跨模型泛化能力有限**：在Qwen2.5-7B上训练后直接迁移到LLaMA-3.1-8B时，ROC-AUC出现显著下降（Table 5），需要为每个目标模型单独训练轻量级值函数。
- **仅支持单轮对话**：当前方法针对单轮输入设计，尚未扩展到多轮交互式场景。
- **未来方向**：探索跨架构迁移能力、推广至多轮对话场景、扩展至更多未见领域。

## 研究启发与可借鉴点
- **隐藏表示作为内部信号的价值**：证明了模型内部状态蕴含丰富的难度相关信号，为其他内部状态分析任务（如置信度估计、不确定性量化）提供了新思路。
- **马尔可夫链+值函数的建模范式**：可将LLM生成过程形式化为MDP，用于其他需要序列状态建模的任务（如过程监督、中间步骤奖励设计）。
- **轻量级辅助网络的训练策略**：不修改主模型、仅训练独立小网络的设计，在保持通用能力的同时实现特定功能，是一种值得推广的工程范式。
- **自适应推理的统一框架**：Difficulty-Aware SC/BoN/SR的简单设计模式可快速复用于其他推理策略，具有较高可迁移性。
- **与团队方向的结合机会**：该方法可直接集成到当前的推理加速管线中，用于动态分配计算预算；也可与reward model研究结合，探索更丰富的内部信号利用方式。

## 关键术语表
- **Hidden Representations（隐藏表示）**：LLM各层Transformer输出的向量，编码模型对输入的当前理解和推理状态。
- **Markov Chain（马尔可夫链）**：具有无后效性的随机过程，本文用于建模LLM token级生成过程中的状态转移。
- **Value Function（值函数）**：从某一状态出发，估计未来累积奖励的期望值，用于量化该状态的"好坏"。
- **Temporal Difference Learning（时间差分学习）**：通过比较连续预测之间的差异（TD误差）来更新价值估计的强化学习算法。
- **Outcome Reward Model (ORM)**：仅对最终输出结果打分的评价模型，本文作为困难/容易的二元标签来源。
- **Self-Consistency（自洽性）**：多次独立采样推理路径后，选择出现次数最多的答案作为最终输出。
- **Best-of-N（最优N选）**：生成N个候选答案，选择被验证器或模型评为质量最高的一个。
- **Self-Refine（自我精炼）**：迭代地让模型对自身输出进行批评和修改，逐步提升答案质量。

## 可复现要素
- **数据集**：MMBench、ScienceQA、MathVista、StrategyQA、gsm8k、commonsenseQA、RLHF-V、VLFeedback；论文未提及代码开源状态，需自行查看论文附带的repository信息。
- **关键超参**：训练集占比50%（部分数据集用官方split）、验证集5%（用于确定阈值τ）、测试集45%；采样温度T=0.5；两层全连接网络，学习率1×10⁻⁴；difficulty阈值τ在验证集上确定；ground truth通过3次独立推理判定。
- **硬件环境**：单卡NVIDIA A100 GPU，Python 3.9.21 + PyTorch 2.5.1。
