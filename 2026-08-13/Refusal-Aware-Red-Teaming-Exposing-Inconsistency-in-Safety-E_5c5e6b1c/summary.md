---
title: "Refusal-Aware-Red-Teaming-Exposing-Inconsistency-in-Safety-E"
source: https://aclanthology.org/2025.emnlp-main.49.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:34:21"
field: "大语言模型安全评估"
keywords: ["red teaming", "LLM safety", "refusal gap", "reinforcement learning", "curiosity-driven exploration", "safety evaluation"]
innovations: ["首次将refusal gap形式化为可量化的red teaming优化目标", "提出基于hidden states的refusal probe机制识别内部拒绝行为", "设计层次化好奇心驱动机制同时优化refusal gap与测试用例多样性"]
benchmarks: ["WildJailbreak", "HARMBench", "ADV-BENCH", "MaliciousInstruct"]
---

# 论文速读：Refusal-Aware-Red-Teaming-Exposing-Inconsistency-in-Safety-E

## 一句话总结
本文首次形式化定义了LLM内部拒绝行为与外部安全评估之间的"refusal gap"概念，并提出了一种基于强化学习的自动化red teaming框架，通过层次化好奇心驱动机制生成多样化测试用例，以系统性地暴露模型与安全评估器之间的不一致性。

## 研究问题与动机
- **核心问题**：LLM部署需要严格的安全评估，但模型内部拒绝机制与外部安全评估器之间可能存在系统性不一致（refusal gap），导致安全隐患。
- **现有方法不足**：当前red teaming方法主要关注单一评估器下的有害内容生成，缺乏对refusal gap这一特定不一致性的系统性挖掘与量化表征。
- **两种不一致风险**：过度拒绝（over-refusal，拒绝安全内容）降低模型实用性；拒绝不足（under-refusal，未拒绝危险内容）造成安全风险。
- **稀疏奖励挑战**：直接优化refusal gap可能导致测试用例多样性不足，且面临稀疏奖励信号问题。

## 核心贡献（创新点）
- **首次将refusal gap形式化为可量化的red teaming优化目标**：定义了over-refusal和under-refusal两个维度，并与已有工作关注单一安全评估器的方法形成本质区别。
- **提出基于refusal probe的内部拒绝检测机制**：利用模型hidden states（拒绝方向投影或线性分类器）替代脆弱的子串匹配，更鲁棒地识别模型内部拒绝行为。
- **设计层次化好奇心驱动机制**：同时优化refusal gap最大化、主题多样性和语义新颖性三个维度，克服了单一奖励信号稀疏和探索不足的缺陷。
- **实现adaptive weighting策略**：根据batch中over-refusal和under-refusal的实证频率动态调整权重，确保两类gap的均衡探索。

## 方法详解
- **Refusal Probe设计**：采用两种方案——(1)在激活空间识别"refusal direction"并投影hidden states判断；(2)在模型最后一层hidden states上训练线性分类器区分拒绝/非拒绝样本。
- **Refusal Gap公式**：$\mathcal{G}(x,y) = \lambda_{\oplus} \cdot \mathbb{I}(x \in \mathcal{G}_{\oplus}) \cdot (1 - s_{\phi}(x)) + \lambda_{\ominus} \cdot \mathbb{I}(x \in \mathcal{G}_{\ominus}) \cdot s_{\phi}(x)$，其中$\lambda_{\oplus}$和$\lambda_{\ominus}$根据当前batch中各类gap的频率自适应更新。
- **层次化奖励结构**：主奖励最大化refusal gap，次级奖励包括Topic Diversity（基于JSD度量主题分布差异）和语义多样性（Self-BLEU和cosine similarity）。
- **PPO训练框架**：目标函数为$\max_{\pi} \mathbb{E}[\mathcal{G}(x) - \beta D_{KL}(\pi||\pi_{ref}) + \sum \lambda_i B_i(x)]$，参考策略$\pi_{ref}$为未对齐的LLaMA-3-8B-Lexic-Uncensored模型。

## 实验与结果
- **数据集**：使用ADVBENCH、MALICIOUSINSTRUCT、TDC2023、HARMBENCH构建不安全指令集，ALPACA构建安全指令集。
- **评估基线**：Zero-shot、Few-shot、RL（Perez et al., 2022）、RL+TDiv、RL+Curiosity（CRT, Hong et al., 2024）。
- **主要结果**：RL+HCuriosity方法在生成质量（IoU指标）和测试用例多样性（1-cos和1-SelfBLEU）上均显著优于所有基线方法；Refusal Direction机制相比Linear CLS获得更高的IoU峰值。
- **消融实验**：三种多样性奖励（Self-BLEU、cosine similarity、topic diversity）具有叠加效应，topic diversity对整体多样性贡献最大。
- **初始模型选择**：未对齐的LLaMA-3-8B作为初始red team模型可显著提升测试用例多样性。

## 相关工作脉络
- **RL-based Red Teaming (Perez et al., 2022)**：开创性使用RL训练red team模型，但关注单一安全评估器的jailbreak成功率，未考虑模型内部拒绝行为与外部评估的不一致。
- **Curiosity-driven Exploration (Hong et al., 2024, CRT)**：引入好奇心机制提升prompt多样性，但未针对refusal gap这一特定目标设计层次化奖励结构。
- **Representation Engineering for Safety (Arditi et al., 2024; Zou et al., 2023)**：发现refusal行为与激活空间的单一方向相关，本文将其应用于refusal probe设计。
- **HarmBench (Mazeika et al., 2024)**：标准化red teaming评估框架，本文在其基础上增加了refusal gap这一新评估维度。
- **MART (Ge et al., 2023)**：结合奖励函数和novelty reward，本文扩展为refusal gap + 层次化好奇心双目标优化。

## 局限性与未来方向
- **Evaluator依赖性**：方法效能受限于所用安全评估模型的能力，evaluator本身可能存在误报或漏报。
- **单模态局限**：当前仅针对文本LLM，未扩展到多模态模型（如GPT-4V）的安全评估。
- **未来方向**：集成多个evaluator模型或引入human-in-the-loop流程提升评估准确性；拓展至多模态red teaming场景。

## 研究启发与可借鉴点
- **Refusal gap量化框架**：可将内部模型行为与外部评估的不一致性形式化为优化目标，迁移至其他安全评估场景。
- **Hidden states-based refusal detection**：利用模型内部表示而非表面token匹配来识别拒绝行为，提升了检测的鲁棒性和泛化能力。
- **层次化好奇心机制**：将多样性分解为词汇、语义、主题多个维度并设计针对性奖励，有效缓解稀疏奖励问题，可借鉴至其他RL-based生成任务。
- **Adaptive weighting策略**：根据实证频率动态调整优化权重，确保各类gap的均衡探索，适用于多目标RL优化场景。
- **实验设计借鉴**：采用动态阈值评估多样性（multi-harmfulness levels）、结合lexical和semantic多样性指标的综合评估体系。

## 关键术语表
- **Refusal Gap**：LLM内部拒绝决策与外部安全评估器判断之间的不一致性，包含over-refusal和under-refusal两种情形。
- **Refusal Probe**：基于模型hidden states识别内部拒绝行为的检测机制，可通过拒绝方向投影或线性分类器实现。
- **Hierarchical Curiosity Mechanism**：层次化好奇心驱动探索机制，同时优化refusal gap最大化、主题多样性和语义新颖性。
- **Over-refusal**：模型拒绝本应被评估器判定为安全的内容，导致不必要的拒绝。
- **Under-refusal**：模型未拒绝被评估器判定为危险的内容，存在安全隐患。
- **IoU (Intersection over Union)**：用于评估red teaming质量的指标，衡量目标测试用例与生成测试用例的重合度。

## 可复现要素
- **数据集**：ADVBENCH、MALICIOUSINSTRUCT、TDC2023、HARMBENCH（公开基准）；ALPACA（公开）
- **代码开源声明**：论文声明"committed to open-sourcing our code"
- **初始模型**：LLaMA-3-8B-Lexic-Uncensored（未对齐版本）
- **训练环境**：8× NVIDIA A100 GPU (80GB)
- **超参数**：temperature=0.7（初始模型）；PPO优化；KL系数β（论文未明确给出）
- **评估模型**：WildGuard（安全评估器）；RoBERTa（有害性检测）
