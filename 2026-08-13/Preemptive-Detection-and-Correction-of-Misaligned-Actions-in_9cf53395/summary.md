---
title: "Preemptive-Detection-and-Correction-of-Misaligned-Actions-in"
source: https://aclanthology.org/2025.emnlp-main.12.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:32:44"
field: "LLM Agent Trustworthiness"
keywords: ["Theory of Mind", "LLM Agent", "misalignment detection", "belief reasoning", "human-agent collaboration", "agent safety"]
innovations: ["首次将LLM心智理论信念推理应用于Agent关键操作预错检测", "提出两阶段任务验证机制（完成对齐+进度评估）以区分中间步骤与最终目标偏移", "构建InferAct-Actor-User协同工作流实现检测与纠正闭环"]
benchmarks: ["WebShop", "HotPotQA", "ALFWorld", "R-Judge"]
---

# 论文速读：Preemptive Detection and Correction of Misaligned Actions in LLM Agents

## 一句话总结
本文提出 **InferAct**，一种基于大语言模型心智理论（Theory-of-Mind）信念推理能力的预错检测方法，在 Agent 执行关键操作前识别其与用户意图的对齐偏差，并通过人机协作流程实现及时纠正，在三个基准任务上 Macro-F1 最高提升 20%。

## 研究问题与动机
- **核心问题**：LLM-based Agent 在实际部署中，其多步行为可能与用户意图发生"不对齐"（misalignment），导致执行不可逆的关键操作后产生负面后果（如误购物品、损坏物品）。
- **现有方法不足**：
  1. 传统 jailbreak 研究关注恶意指令注入风险，而非用户意图是正常时 Agent 自行偏离意图的问题。
  2. 现有 Agent 系统缺乏"执行前"自动检测机制，如需人工逐操作审核（如 See-Act），给用户带来巨大认知负担。
  3. 已有的 Agent 评估方法（LLM-as-judge、self-reflection 等）缺少对长程任务中"进行中"状态与"最终完成"状态的区分能力，易产生误报。
  4. 模拟沙盒评估无法真实反映复杂现实环境（如网页购物）中的对齐风险。

## 核心贡献（创新点）
1. **首次将 LLM 的心智理论信念推理能力应用于 Agent 行为检测**——通过第三人称视角推断 Agent 行为背后的意图，而非依赖 Agent 自身自省，本质上区别于 Self-Reflection 方法的自我服务偏差风险。
2. **提出两阶段任务验证机制（完成对齐 + 进度评估）**——区别于一次性二分类判断，能区分"任务已完成"和"仍在推进中"两种状态，有效减少中间步骤的误报。
3. **构建 InferAct-Actor-User 协同工作流**——仅在关键（高风险）操作前触发检测，既保障安全又保留 Agent 自主性；配合人工反馈可形成迭代改进闭环。
4. **系统性评估与开源实践指导**——在 WebShop、HotPotQA、ALFWorld 三个主流基准上全面验证，并提供 prompt 模板，便于跨场景迁移。

## 方法详解
**InferAct 框架由两大组件构成：**

**1. Task Inference Unit（任务推断单元）**
- 输入：Agent 的行为序列 $S = \{a_1, o_1, ..., a_m, o_m\}$（动作-观测交替）
- 通过 prompt $P^i$ 引导 LLM 进行信念推理，推断 Agent 意图执行的任务 $T'$：
  $$T' = LLM(P^i, S)$$
- 与 Self-Reflection 的本质区别：采用第三人称视角，基于可观察行为推断意图，避免 Agent 自我解释时的主观偏差。

**2. Task Verification Unit（任务验证单元，两阶段）**
- **阶段一：完成对齐验证**——判断 $T'$ 是否已满足用户真实任务 $T^*$：
  $$Y^c = LLM(P^c, S, T^*, T')$$
  $Y^c \in \{True, False\}$，采用单向蕴含关系（fine-grained 满足 implies coarse-grained 满足）。
- **阶段二：进度评估**（当 $Y^c = False$ 时触发）——判断 $T'$ 是否仍在通往 $T^*$ 的正确道路上：
  $$Y^p = LLM(P^p, S, T^*, T')$$
  $Y^p \in \{True, False\}$。
- 两阶段组合逻辑：$Y^c=True$ → 对齐；$Y^c=False, Y^p=True$ → 仍在推进；$Y^c=False, Y^p=False$ → 触发警报。

**3. 协同工作流设计**
- 预先定义关键动作集 $\mathcal{A}_{crit}$（如 WebShop 中的 `click[Buy Now]`、ALFWorld 中的 `Clean/Heat/Cool`），仅在触及这些动作前激活检测。
- 检测触发后，Human 介入验证并提供反馈（Binary 或 NL），Agent 据此修正后续行为，形成迭代循环。

## 实验与结果
**数据集**：WebShop（网购，300 task）、HotPotQA（问答，300 task）、ALFWorld（家居，134 task）

**评估指标**：Macro-F1、Total Detection Errors (TDE)、Effective Reliability (ER)、PR-AUC

**主要结果（Llama-3-70B backbone）**：
| 任务 | InferAct-prob Macro-F1 | 最强基线 Macro-F1 | 提升幅度 |
|------|----------------------|------------------|---------|
| WebShop | **0.590** | Multi-step 0.487 | +10.3pp |
| HotPotQA | **0.599** | Multi-step 0.569 | +3.0pp |
| ALFWorld | **0.827** | Multi-step 0.767 | +6.0pp |

- 在 **11/12 设置**（3任务×4模型）中取得最优 Macro-F1，平均超越最强基线 8pp。
- **InferAct-prob** 在 ECE 校准误差上表现最优（如 ALFWorld 与 GPT-3.5-Turbo: 0.116 vs Token Prob 0.583）。
- 跨风险等级（低/中/高）分析：InferAct-verb 在所有风险级别均保持最低漏报率（False Negative Rate）。
- **协同效果**：Actor + InferAct 经 3 轮迭代后，NL 反馈下 WebShop 成功率 70.7%，较第二优方法（Token Entropy 42.7%）提升 28pp；相对 Full Validation 仅下降 3.5%。
- **R-Judge 编码安全基准**：InferAct-prob Macro-F1 达 0.612，PR-AUC 0.740。
- **效率**：单次检测耗时 4.1s，成本 $0.0122（GPT-4-Turbo），低于 Self-Consistency（$0.0128, 6.0s）。

## 相关工作脉络
1. **LLM-as-Judge / Direct Prompt**：Zheng et al. (2023) 提出单一答案评分范式，本文将其作为基础基线，但 InferAct 通过信念推理和两阶段验证显著优于直接判定轨迹正确性的做法。
2. **Self-Reflection**：Shinn et al. (2023) 的 Reflexion 让 Agent 自我反思行为并调整，本文证明外部信念推理（第三人称）比自我解释更客观有效（Table 9）。
3. **Token Entropy / Probability**：Han et al. (2024)、Lin et al. (2024) 探索模型不确定性量化，本文对比表明仅靠 token 级概率/熵无法捕捉任务级意图对齐。
4. **Multi-step Evaluation**：逐步评估各 step 正确性后聚合（Product 聚合最优），但无法区分"中间合理步骤"与"整体目标偏移"。
5. **Machine Theory-of-Mind**：Kosinski (2023)、Bubeck et al. (2023) 等仅评估 LLM 的 ToM 能力，本文首次将其应用于 Agent 行为监控这一实际场景。
6. **Agent 信任与安全**：Ruan et al. (2024) 的 ToolEmu 和 Hua et al. (2024) 的 Agent Constitution 依赖沙盒模拟，无法处理真实环境中的动态意图偏离。

## 局限性与未来方向
1. **成本建模简化**：仅以 TDE（FN+FP 之和）衡量代价，未区分不同风险类型的具体经济损失（如退货成本 vs 客户流失）。
2. **不包含恶意意图场景**：方法仅针对用户意图正常但 Agent 偏离的情况，不解决对抗性/有害用户意图的防范。
3. **关键动作需人工预定义**：当前依赖领域专家手动标注 $\mathcal{A}_{crit}$，在开放域大动作空间中如何实现人机协同的自动发现是未来方向。
4. **样本量有限**：部分实验（如用户研究）仅在 3 名用户、100 个 WebShop 样本上进行，需更大规模验证。
5. **模型规模并非单调增益**：Appendix A 显示增大模型规模不一定提升 Task Inference/Verification 性能，需更深入机制分析。

## 研究启发与可借鉴点
1. **ToM 信念推理用于 Agent 可解释性监控**：可将第三人称意图推断思路迁移至多 Agent 协作场景中的行为归因与责任追踪。
2. **两阶段验证策略（完成 vs 进度）**：适用于任意长程任务的状态评估，尤其适合当前流行 Agentic RL/ReAct 类方法的在线纠错。
3. **混合模型部署策略**：Figure 4 证明不同组件使用不同规模模型可在性能与成本间取得最优平衡（如轻量模型做推断 + 大模型做验证）。
4. **关键动作过滤机制**：仅在高危操作前触发检测的设计思路，可推广至代码生成 Agent（`rm -rf`、`sudo`）、API 调用等高风险操作场景。
5. **校准友好的概率输出**：InferAct-prob 在不同模型上均展现稳定 ECE，表明意图推理框架天然适合概率化输出，可用于阈值自适应决策。

## 关键术语表
**Theory of Mind (ToM)**：人类推断他人心理状态（信念、意图）以预测行为的能力，本文将其引入 LLM 以实现对外部 Agent 行为的意图解读。

**Belief Reasoning**：基于观察到的行为序列推断目标背后意图的认知过程，InferAct 的核心机制，采用第三人称视角避免自我偏差。

**Misaligned Action**：Agent 执行的操作与其用户真实意图不一致的行为，可能导致不可逆负面后果（如误购买、误删除）。

**Macro-F1**：宏平均 F1 分数，同时衡量检测器在"正确识别不对齐"和"不误报正常行为"两方面的平衡性能。

**Effective Reliability (ER)**：$(TP - FP) / (TP + FP)$，衡量检测结果中真阳性相对假阳性的优势程度。

**Total Detection Errors (TDE)**：假阴性与假阳性之和，近似表征实际部署中的综合代价。

**Estimated Calibration Error (ECE)**：衡量模型输出概率与真实准确率之间的一致性，ECE 越低表示概率校准越好。

## 可复现要素
- **数据集**：WebShop、HotPotQA、ALFWorld 均为公开基准；R-Judge 亦为开源安全基准。
- **代码/权重**：论文未提供开源代码仓库链接，但附录提供了完整 Prompt 模板（Appendix C），可据此复现。
- **关键超参**：
  - Temperature：Self-Consistency 设为 0.7，其余方法为 0.0；Llama-3-70B 使用 greedy search。
  - 概率方法阈值调优：基于 50 条 dev 数据（Appendix D, Table 11）。
  - 最大干预配额：Oracle 最多评估 50% 任务（Table 12）。
  - Self-Consistency 采样数 $m=5$。
