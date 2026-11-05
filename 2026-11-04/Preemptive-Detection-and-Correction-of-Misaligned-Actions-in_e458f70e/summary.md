---
title: "Preemptive-Detection-and-Correction-of-Misaligned-Actions-in"
source: https://aclanthology.org/2025.emnlp-main.12.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:54:50"
field: "LLM Agent 安全与对齐"
keywords: ["LLM Agents", "Theory of Mind", "Alignment Detection", "Belief Reasoning", "Human-Agent Collaboration", "Agent Safety"]
innovations: ["首次将 ToM 信念推理引入 agent 对齐检测，区分于 self-reflection", "设计两阶段任务验证单元（完成验证+进展评估）避免中途误报", "提出 InferAct-verb/prob 两种变体支持有无 dev 集场景"]
benchmarks: ["WebShop", "HotPotQA", "ALFWorld", "R-Judge"]
---

# 论文速读：Preemptive-Detection-and-Correction-of-Misaligned-Actions-in-LLM-Agents

## 一句话总结
本文提出 **InferAct**，一种基于大语言模型心智理论（Theory of Mind）信念推理能力的检测方法，用于在 LLM Agent 执行关键动作前预判其行为是否与用户意图对齐，并通过人机协作流程实现误对齐动作的及时纠正。

## 研究问题与动机
- **核心问题**：LLM Agent 在实际部署中可能因误解用户意图而执行具有负面后果的关键动作（如误购商品、损坏物品），目前缺乏有效的**事前检测与纠正机制**。
- **现有方法不足**：
  1. 当前 agent 系统（如 See-Act）依赖人工逐条验证，认知负担重且限制 agent 自主性。
  2. 仿真环境中的风险评估方法难以建模真实复杂场景（如网购）。
  3. 已有 LLM-as-judge 方法多用于事后评估，缺乏对中间过程的对齐检测。
  4. 自反思（self-reflection）等内部验证方式缺乏外部信号，无法有效纠正错误行为。

## 核心贡献（创新点）
1. **首次将 ToM 信念推理引入 agent 对齐检测**：利用 LLM 从第三方视角推断 agent 行为背后的意图，而非让 agent 自我反思，减少自我服务偏见。
2. **设计两阶段任务验证单元**：先判断已完成的推断任务是否 entail 用户目标，再判断未完成阶段是否处于正确进展轨道，避免中途误报。
3. **提出高效协作工作流**：仅当 agent 触发预定义关键动作时激活检测器，结合人类反馈实现迭代纠正，使 Agent 成功率提升 10.4%，同时减少 50% 人工审核负担。
4. **系统性基准评测**：在 WebShop、HotPotQA、ALFWorld 三个主流 agent 基准上，InferAct 在 11/12 设置中达到最优，Macro-F1 最高提升 20%。

## 方法详解
InferAct 由两个核心单元组成：

**1. Task Inference Unit（任务推理单元）**
- 输入：agent 的行为轨迹 $S = \{a_1, o_1, ..., a_m, o_m\}$（动作-观察序列）
- 通过提示 $P^i$ 引导 LLM 进行信念推理，推断 agent 的实际意图 $T'$：
  $$T' = LLM(P^i, S)$$
- 采用第三人称视角，与 self-reflection 的自我 introspection 形成对比。

**2. Task Verification Unit（任务验证单元）**
- **完成对齐验证**：判断推断任务 $T'$ 是否 entail 用户真实任务 $T^*$：
  $$Y^c = LLM(P^c, S, T^*, T') \in \{True, False\}$$
  使用单向蕴涵关系（fine-grained 动作可蕴涵 coarse-grained 指令）。
- **进展评估**：若未完成任务，进一步判断 $T'$ 是否处于正确进展轨道：
  $$Y^p = LLM(P^p, S, T^*, T') \in \{True, False\}$$
- 两阶段设计确保不误报仍在执行中的合法中间步骤。

**3. 协作工作流**
- 仅在 agent 尝试**预定义关键动作**时触发检测（如 Buy Now、Clean、Heat、finish[answer]）。
- 检测到 misalignment 后，向用户发送警报，用户可提供二进制或自然语言反馈进行纠正。
- 支持迭代优化循环。

## 实验与结果
**数据集**：WebShop（网购，300 tasks）、HotPotQA（多跳问答，300 tasks）、ALFWorld（家务任务，134 tasks）

**评估指标**：Macro-F1、TDE（总检测错误）、ER（有效可靠性）、PR-AUC

**主要结果**：
| 方法 | WebShop Macro-F1 | HotPotQA Macro-F1 | ALFWorld Macro-F1 |
|------|------------------|-------------------|-------------------|
| Direct Prompt | 0.400 | 0.385 | 0.360 |
| Multi-step | 0.531 | 0.624 | 0.655 |
| **InferAct-prob (Llama-3-70B)** | **0.590** | **0.599** | **0.827** |
| **InferAct-verb (Qwen3-4B)** | 0.500 | 0.527 | 0.710 |

- InferAct 在 11/12 设置中达到 SOTA，平均比最强基线高 8% Macro-F1
- InferAct-verb 在无 dev 集场景下表现最佳（无需阈值调优）
- **R-Judge 编码安全任务**：InferAct-prob Macro-F1 达 0.612，优于 Multi-step 的 0.530
- **校准性能**：InferAct-prob 在 ECE 指标上表现最稳定，尤其对 GPT-3.5-Turbo 和 Llama-3-70B 效果显著

**人机协作效果**：
- Actor + InferAct + NL反馈：WebShop 成功率 47.7%，较 Token Entropy 提升 5%
- 仅需 50% 人工干预即可达到 Full Validation 仅差 3.5% 的性能
- 真实用户研究中，InferAct + 人类反馈达到 50.3%±1.2 成功率

## 相关工作脉络
1. **LLM-as-judge 方法**（Zheng et al., 2023）：本文采用 single-answer grading 范式作为 Direct Prompt 基线，但创新在于引入 ToM 信念推理而非直接判断轨迹对错。
2. **不确定性度量**（Han et al., 2024; Lin et al., 2024）：使用 token entropy 作为基线，InferAct 通过显式意图推断超越单纯概率建模。
3. **Machine Theory-of-Mind**（Kosinski, 2023; Bubeck et al., 2023; Strachan et al., 2024）：前人主要评估 LLM 的 ToM 能力，本文首次将其应用于 agent 对齐检测。
4. **Self-reflection**（Shinn et al., 2023）：对比实验显示 self-reflection 无法带来性能提升（Table 9），InferAct 的第三人称视角更有效。
5. **Agent 安全性评估**（Ruan et al., 2024; Hua et al., 2024）：仿真/沙箱方法难以捕捉真实风险，InferAct 直接在动态环境中实时检测。
6. **多步评估方法**（Pan et al., 2024）：本文关注实时检测与纠正，而非事后评估以提升性能，且探索了 evaluator-agent-user 三方协作动态。

## 局限性与未来方向
1. **成本建模简化**：仅将 FN 和 FP 简单相加，未考虑实际场景中不同错误类型的差异化代价（如网购中误购 vs. 误报警的成本差异）。
2. **未处理恶意意图**：方法聚焦于缓解意图 misalignment，但对用户本身有害的意图（如制造炸弹）不做处理。
3. **关键动作需手动定义**：当前在小动作空间中手动定义高风险动作，开放域中自动发现关键动作仍需探索。
4. **规模扩展性有限**：human-in-the-loop 方法在开放域中可能难以无缝扩展。

## 研究启发与可借鉴点
1. **ToM 信念推理用于 agent 监控**：将心理学概念引入 ML 系统监控，提供了一种新的评估视角，可迁移到其他需要意图推断的场景（如代码审查、医疗决策支持）。
2. **两阶段验证设计**："完成验证 + 进展评估"的思路值得借鉴，适用于长程任务中区分"已完成"与"进行中"的状态判断。
3. **混合模型策略**：Figure 4 展示不同组件可使用不同模型（如 Task Inference 用 GPT-3.5，Task Verification 用 GPT-4），在性能和成本间取得平衡。
4. **threshold-free 变体**：InferAct-verb 无需 dev 集调参，适合资源受限或无标注数据的场景。
5. **安全与效率的权衡设计**：仅在关键动作时触发检测，而非全量监控，这一设计原则可用于其他 high-stakes agent 系统。

## 关键术语表
- **Theory of Mind (ToM)**：人类推断他人心理状态（信念、意图）以预测行为的能力，本文将其引入 LLM agent 监控。
- **Belief Reasoning**：基于可观察行为推断主体潜在意图的认知过程，是 InferAct 的核心机制。
- **Misaligned Action**：agent 执行但与用户真实意图不符的关键动作，可能导致负面后果。
- **Critical Action**：具有高潜在风险的动作（如 Buy Now、Clean、Heat），需触发检测器评估。
- **Macro-F1**：平衡检测灵敏度与可用性的指标，同等重视 FN 和 FP。
- **Effective Reliability (ER)**：$(TP - FP)/(TP + FP)$，衡量检测结果中真阳性相对假阳性的优势。
- **Estimated Calibration Error (ECE)**：评估概率输出校准程度的指标，越低表示置信度越准确。
- **Self-consistency**：通过多次采样 Majority Vote 提高稳定性的方法，本文作为基线对比。

## 可复现要素
- **数据集**：WebShop、HotPotQA、ALFWorld、R-Judge（均为公开基准）
- **代码/权重**：论文未明确提及代码开源声明，附录提供完整 prompt 设计（Figures 7-24）
- **关键超参**：
  - Temperature: GPT 模型 Self-Consistency 设为 0.7，其余方法 0.0；Llama-3-70B 使用 greedy search
  - 采样次数 m=5（Self-Consistency）
  - Dev set 大小：50（用于阈值调优）
  - 模型 backbone：GPT-4-Turbo、GPT-3.5-Turbo、Llama-3-70B、Qwen3-4B
