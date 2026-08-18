---
title: "reWordBench-Benchmarking-and-Improving-the-Robustness-of-Rew"
source: https://aclanthology.org/2025.emnlp-main.167.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:45:58"
---

# 论文速读：reWordBench-Benchmarking-and-Improving-the-Robustness-of-Reward-Models-with-Transformed-Inputs

## 一句话总结
论文提出了 reWordBench 基准，通过语义或排名保持的输入变换系统评估奖励模型的鲁棒性，发现 SOTA 奖励模型在轻微变换下排名准确率大幅下降甚至低于随机水平；并提出了通过正则化释义响应得分相似性来提升奖励模型鲁棒性的方法，该方法还能改善下游对齐任务的输出质量。

## 研究问题与动机
- 现有奖励模型在标准基准（如 RewardBench）上取得高准确率，但可能存在过拟合，导致在实际应用中因输入微小变化而产生不当偏好翻转。
- 奖励模型训练数据量小且常含虚假相关（如响应长度偏好），易过拟合到风格伪影；且作为评估器和对齐组件时，鲁棒性要求极高。
- 缺乏系统衡量奖励模型在意义或排名保持变换下一致性的基准，现有工作多关注分类/回归任务鲁棒性，未专门针对奖励模型的排名一致性进行度量。
- 现有提升鲁棒性的方法多依赖对抗训练或特定变换，而奖励模型需要泛化到多种自然、无意的变换场景。

## 核心贡献（创新点）
- 提出了 reWordBench 基准，包含28种分类（控制、自然、领域针对性）的意义/排名保持输入变换，用于系统评估奖励模型的鲁棒性；与以往通用NLP鲁棒性基准不同，本基准专门针对奖励模型的排名一致性设计，并覆盖代码、数学、安全等子集。
- 实证揭示了 SOTA 奖励模型在轻微变换下存在严重脆弱性，排名准确率大幅下降至随机水平以下，且最佳模型在原始 RewardBench 上的排名与变换后不一致；区别于之前仅观察分类准确率下降的研究，本文聚焦奖励模型特有的“偏好翻转”现象及其对齐后果。
- 提出通过正则化原响应与释义响应的得分相似性来训练鲁棒奖励模型（目标函数含额外一致性损失）；与单纯数据增强不同，该方法显式约束等价输入的评分一致性，且该正则化意外泛化至多种未见过的变换类型。
- 证明经正则化训练的鲁棒奖励模型在下游对齐任务（best-of-n 和 RAFT）中产生更高质量输出，LM judge 获胜率最高达59%；此前研究多关注鲁棒性本身，本文首次将奖励模型鲁棒性与对齐效用直接关联。

## 方法详解
- **reWordBench 变换设计**：将 RewardBench 实例分为三类变换：
  1. *控制变换*：手动模板确保意义不变，如 Add Quotes（添加引号）、Punct. Spaces（标点空格）、Twitter Handle/URL、StressTest（追加无意义字符串）、Ignore Above/Below、Rot-13/Rot-2。
  2. *自然变换*：模拟真实噪声，使用 Llama-3-70B-instruct 进行释义、回译（英→西→英）、回转录（文本→音频→文本）、同形字替换、字符交换/替换/插入/删除、词删除，并保证余弦相似度≥0.7。
  3. *领域针对性变换*：针对代码（Minification、Comment Bad/Good、Append Other Code）、数学（Swap Format：交换 `\boxed{}` 与 `# Answer` 格式）、安全（Jailbreak prompts 从 JailbreakChat 数据集）。
- **鲁棒奖励模型训练**：
  - 基础目标：点wise回归损失 $\mathbb{E}[(RM(x,y)-s)^2]$（公式1）。
  - 正则化目标：对每个响应 y 自动释义得到 $\tilde{y}$，构建增强数据集 $\tilde{D}=\{(x,y,\tilde{y},s)\}$，优化带正则项的联合损失：
    $$\mathbb{E}[(RM(x,y)-s)^2 + \alpha(RM(x,y)-RM(x,\tilde{y}))^2] \quad \text{(公式3)}$$
    其中 $\alpha$ 为正则化强度系数（默认 $\alpha=10$）。
  - 对比基线：简单数据增强（将释义响应作为独立样本训练，最小化 $(RM(x,y)-s)^2+(RM(x,\tilde{y})-s)^2$，公式4）。
- **评估指标**：主要关注排名鲁棒性，即对变换前后 $y_w$ 和 $y_l$ 的偏好一致性（指示函数相等）；量化绝对排名准确率下降，micro-平均 across 所有实例；同时检查原始奖励变化（附录F）。

## 实验与结果
- **数据集**：基于 RewardBench（含 Chat、Chat Hard、Safety、Reasoning 子集），使用 HelpSteer2 数据集训练鲁棒奖励模型。
- **评估基线**：7 个分类奖励模型（GRM-Llama3.2-3B、GRM-Llama3-8B、Skywork 系列 8B/27B、URM-LLaMa-3.1-8B、QRM-Llama3.1-8B、internlm2-20b）、3 个生成式奖励模型（Skywork-Critic-8B/70B、facebook/Self-taught-evaluator-70B）以及 GPT-4o。
- **主要结果**：
  - SOTA 奖励模型在 reWordBench 变换下排名准确率大幅下降，许多降至随机水平（<50%）以下；例如 Skywork/Skywork-Reward-Gemma-2-27B-v0.2 在 Ignore Above 变换下准确率从95%跌至81%（图1）。
  - 原始 RewardBench 表现最佳的模型在18/28种变换后不再是最优，表明基准排名不可靠。
  - 正则化奖励模型在 reWordBench 上鲁棒性显著提升：Chat Hard 子集准确率下降从16.6%（标准）降至8.7%（正则化），降幅减少约一半；在 Chat、Reasoning 子集上也表现最佳（表5）。
  - 正则化提升泛化：仅用释义正则化训练，但对 Ignore Above 等完全不同变换也有效（图3，α增大鲁棒性单调提升）。
- **下游对齐实验**：
  - 使用 best-of-n（n=64）和 RAFT 方法在 RewardBench 和 UltraFeedback 提示上对齐。
  - 正则化奖励模型生成的输出在 LM judge（Llama-3-70B-Instruct、Qwen2.5-72B-Instruct）下获胜率最高达59%（图4），显著优于标准奖励模型。
  - 长度控制实验（响应差异≤3 tokens）仍显示正则化模型占优（Appendix D），排除长度偏差。

## 相关工作脉络
- **奖励模型鲁棒性**：Shen et al. (2024a) 也训练正则化奖励模型，但目标函数不同（关注偏好一致性而非评分相似性），且未系统评估多种变换；本文聚焦排名一致性与对齐效用。
- **输入变换鲁棒性基准**：CheckList (Ribeiro et al., 2020)、Adversarial GLUE (Wang et al., 2021a)、RoTBench (Ye et al., 2024) 等评估 NLP 模型鲁棒性，但本文首个针对奖励模型专门设计，覆盖代码、数学、安全领域。
- **奖励模型过拟合与虚假相关**：Singhal et al. (2024) 指出长度相关性；本文扩展至多种变换下的脆弱性，并证明正则化可缓解。
- **提升模型鲁棒性的训练方法**：一致性正则化（Tack et al., 2022）、对比指令调优（Yan et al., 2024）等；本文借鉴释义一致性正则化，但应用于奖励模型评分而非分类。
- **对齐中的奖励黑客**：Gao et al. (2023) 分析奖励模型过度优化；本文显示鲁棒性不足会加剧该问题，并提出改进对齐输出的方法。
- **自动评估与 LM Judge**：Li et al. (2023) 的 AlpacaEval；本文使用类似协议评估对齐输出，并验证 judge 长度偏差不影响结论。

## 局限性与未来方向
- 变换种类虽多，但仍可能存在其他变换类型可揭示奖励模型新特性。
- 释义等变换依赖 ML 模型生成，缺乏严格的语义等价保证，仅通过小规模人工检查验证合理性。
- 使用自动 LM judge 评估对齐输出质量，尽管已验证长度偏差影响不大，但人工评估可能提供进一步洞察。
- 未来方向：探索更丰富的变换（如对抗性学习变换）、将正则化扩展至其他奖励模型架构、研究鲁棒性与对齐性能的理论联系。

## 研究启发与可借鉴点
- **正则化范式可迁移**：释义一致性正则化（公式3）可应用于其他需要鲁棒评分的场景，如排序模型、多轮对话评估器。
- **变换分类设计**：控制/自然/领域针对性三分法为构建专业基准提供模板，未来可扩展至代码生成、数学推理等垂直领域。
- **对齐效用评估**：将鲁棒性提升与下游对齐质量直接挂钩（best-of-n、RAFT），为奖励模型改进提供实用目标，而不仅关注基准分数。
- **长度控制实验**：Appendix D 中严格匹配响应长度的比较方法，可有效排除 LM judge 的长度偏好偏差，值得在类似评估中复用。
- **创新机会**：结合本团队方向，可探索将正则化与多模态奖励模型结合，或研究不同 α 值对各类变换的泛化效果。

## 关键术语表
- **Reward Model (RM)**：用于对语言模型生成的响应进行评分或偏好的神经网络，是 RLHF 和对齐算法的核心组件。
- **reWordBench**：本文提出的基准，通过意义或排名保持的输入变换系统评估奖励模型的鲁棒性。
- **Ranking Accuracy Drop**：变换前后奖励模型偏好一致性下降的百分比，用于量化鲁棒性。
- **Paraphrase Regularization**：通过正则化使原响应与释义响应得分相似的训练方法，以提升奖励模型一致性。
- **Best-of-n**：推理时采样 n 个响应并用奖励模型选择最高分响应对齐策略。
- **RAFT**：使用奖励模型选出的最佳响应对 SFT 模型进行监督微调的训练型对齐方法。
- **Reward Hacking**：策略模型利用奖励模型的虚假相关来优化评分而非真正提升质量的现象。
- **Jailbreak**：旨在绕过安全限制诱导模型生成有害内容的对抗性提示。

## 可复现要素
- **数据集**：RewardBench（公开）、HelpSteer2（公开）、UltraFeedback（公开）、JailbreakChat（公开）；reWordBench 变换细节在附录 B 提供。
- **代码/权重**：论文未明确声明开源，但使用 OpenRLHF 框架训练，代码可能需自行实现。
- **关键超参**：正则化强度 α=10（默认）；训练轮数2 epoch；batch size 128；学习率9×10⁻⁶；输入截断长度2048 tokens；best-of-n 采样数 n=64；对齐训练3 epoch，batch size 64，学习率5×10⁻⁶，weight decay 0.1。

<!--META
{"keywords": ["Reward Models", "Robustness", "Benchmarking", "Regularization", "Alignment", "reWordBench"], "field": "语言模型对齐与评估", "innovations": ["提出reWordB
