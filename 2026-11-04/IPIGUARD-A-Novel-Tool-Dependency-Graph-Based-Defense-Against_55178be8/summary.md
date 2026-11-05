---
title: "IPIGUARD-A-Novel-Tool-Dependency-Graph-Based-Defense-Against"
source: https://aclanthology.org/2025.emnlp-main.53.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:52:54"
field: "LLM Agent 安全与鲁棒性"
keywords: ["Indirect Prompt Injection", "LLM Agents", "Tool Dependency Graph", "Agent Security", "Prompt Injection Defense"]
innovations: ["提出以 TDG 遍历为核心的执行中心防御范式，从源头阻断恶意工具调用", "设计 Argument Estimation、Node Expansion、Fake Tool Invocation 三个机制解决规划-执行解耦挑战"]
benchmarks: ["AgentDojo"]
---

# 论文速读：IPIGUARD-A-Novel-Tool-Dependency-Graph-Based-Defense-Against

## 一句话总结
本文提出 IPIGUARD，一种面向 LLM 智能体的间接提示注入（IPI）防御框架，通过将任务执行建模为工具依赖图（TDG）的遍历过程，实现"规划-执行"解耦，从源头阻断恶意工具调用。

## 研究问题与动机
- **核心问题**：LLM 智能体在与不可信外部数据源交互时，工具响应中嵌入的恶意指令可诱导智能体执行非预期的工具调用，导致 IPI 攻击成功。
- **现有方法不足**：
  1. 主流防御依赖模型内在校验或辅助检测模型，本质仍假设模型本身具备安全性，缺乏对执行轨迹的结构化约束。
  2. 自适应攻击（如 Zhan et al. 2025）可稳定绕过模型级防御，证明"prompt 加固"路线存在根本性缺陷。
  3. 智能体拥有无限制的工具调用能力，攻击者可通过劫持参数或触发额外工具调用绕过安全护栏。

## 核心贡献（创新点）
1. **提出 IPIGUARD 执行范式**：将任务执行过程形式化为 TDG 遍历，从执行层面而非模型层面阻断 IPI 攻击。
2. **引入 Argument Estimation 机制**：在 TDG 拓扑序执行过程中动态推断未确定的工具参数，解决"规划时参数未知"的核心挑战。
3. **设计 Node Expansion 机制**：基于 CQRS 思想将工具划分为 Query/Command 两类，仅在执行阶段允许 Query 类工具扩展，在保持安全性的同时提升适应性。
4. **提出 Fake Tool Invocation 机制**：针对用户任务与注入任务共享工具的场景，通过伪造完成信号消除注入指令对参数估计的干扰。

## 方法详解
IPIGUARD 由两阶段构成：

**阶段一：TDG 规划（Planning）**
- 输入：用户指令、工具描述、系统上下文。
- LLM 生成一个有向无环图（DAG），节点表示工具调用，边表示数据依赖关系。
- 节点分类：
  - **Deterministic Node**：所有参数在规划阶段已确定。
  - **Pending Node**：部分参数标记为 `<unknown>: data_type`，需在执行中推断。

**阶段二：TDG 遍历（Execution）**
- 按拓扑序依次执行节点，维护执行上下文。
- **Argument Estimation**：对 Pending Node，从历史工具响应中检索信息，填充 `<unknown>` 参数，转化为 Resolved Node 后执行。
- **Node Expansion**：执行某节点后，若需额外查询信息，可生成 Query Expanded Node（仅允许 Query 工具），继承当前节点的后续依赖。
- **Fake Tool Invocation**：当检测到上下文含注入指令时，先构造一个模拟的 Command 工具响应（伪造完成），使 Agent 认为指令已被处理，再基于原始用户意图更新参数。

**关键约束**：执行阶段禁止引入新的 Command 工具调用，仅允许 Query 扩展。

## 实验与结果
- **数据集**：AgentDojo benchmark（97 任务，4 域：Workspace/Slack/Travel/Banking，629 测试用例）。
- **评估模型**：6 个 LLM（GPT-4o、GPT-4o-mini、Claude 3.5 Sonnet、Qwen2.5-7B-Instruct、Qwen3-32B、o4-mini）。
- **攻击基准**：Ignore Previous、InjecAgent、Tool Knowledge、Important Instruction。
- **基线方法**：Detector、Tool Filter、Spotlight、Sandwich。
- **主要指标**：Benign Utility (BU)、Utility under Attack (UA)、Attack Success Rate (ASR)。

| 方法 | 平均 ASR ↓ | 平均 UA ↑ |
|------|------------|-----------|
| No Defense | 13.16% | 54.30% |
| Detector | 4.43% | 26.50% |
| Tool Filter | 2.06% | 53.36% |
| Spotlight | 11.31% | 55.09% |
| Sandwich | 5.25% | 44.01% |
| **IPIGUARD** | **0.69%** | **58.77%** |

- **最强结果**：IPIGUARD 在 GPT-4o-mini 上实现平均 ASR 0.69%、UA 58.77%，较无防御方案 ASR 降低 12.47 个百分点，UA 提升 4.47 个百分点。
- **开销**：Token 用量约为无防御方案的 2 倍，任务完成时间约 13.88s（对比无防御 7.13s）。
- **Ablation**：FTI + NE 联合使用效果最优（BU 69.07%，UA 57.07%，ASR 0.64%）。

## 相关工作脉络
1. **Prompt 级防御**：Spotlight（Hines et al. 2024）、Sandwich（Prompting 2024）通过 prompt 格式化/重复强化用户意图，属于模型层防御，易被自适应攻击绕过。
2. **辅助检测模型**：Detector（ProtectAI 2024）使用 DeBERTa 分类器检测注入，但成本高且误杀严重（UA 仅 26.50%）。
3. **工具预选择**：Tool Filter（Willison 2023）限制工具子集，但无法应对参数篡改类攻击。
4. **LLM-as-Judge**：Task Shield（Jia et al. 2024）用 LLM 监控中间步骤，但若 judge 本身被注入则失效。
5. **系统性防御**：CaMel（Debenedetti et al. 2025）通过系统级隔离增强鲁棒性，但依赖静态规划，适应性差。
6. **本文定位**：IPIGUARD 转向"执行中心"范式，通过结构化约束而非内容检测防御 IPI，与上述工作形成互补。

## 局限性与未来方向
- **防御范围**：仅防御干扰工具使用的 IPI 攻击，未覆盖纯文本输出操纵场景。
- **模型依赖性**：需要较强规划能力的 LLM，对弱模型/资源受限场景适用性有限。
- **实验规模**：受 LLM 调用成本限制，未评估 o3 等更强模型。
- **Fake Tool Invocation 边界情况**：极少数 corner case 下伪造失败导致 ASR 不为零，需进一步改进。

## 研究启发与可借鉴点
1. **执行中心防御思路**：将安全约束从"内容识别"转向"行为结构约束"，可为其他 agent 安全场景（如 tool hijacking、state poisoning）提供范式参考。
2. **CQRS 思想迁移**：Query/Command 工具二分法可用于设计更精细的权限控制策略，值得在多工具场景中推广。
3. **规划-执行分离架构**：使用不同强弱的 LLM 分别承担 planner 和 executor 角色，可在成本与性能间取得灵活权衡（见 Table 5）。
4. **TDG 形式化表示**：将工具调用序列建模为 DAG，便于后续引入形式化验证或静态分析技术。

## 关键术语表
- **Indirect Prompt Injection (IPI)**：将恶意指令嵌入智能体访问的外部数据源中，间接操控 agent 行为的攻击方式。
- **Tool Dependency Graph (TDG)**：以 DAG 形式表达任务执行中各工具调用的依赖关系与执行顺序。
- **Argument Estimation**：在执行阶段根据历史工具响应动态推断 Pending Node 中未知参数的机制。
- **Node Expansion**：允许在执行过程中为只读查询目的动态添加新工具调用的机制。
- **Fake Tool Invocation**：通过注入模拟工具响应来"消耗"注入指令，防止其干扰真实参数估计的机制。
- **Benign Utility (BU)**：无攻击时智能体成功完成任务的比例。
- **Utility under Attack (UA)**：遭受攻击时智能体仍正确完成用户任务的比例。
- **Attack Success Rate (ASR)**：攻击者目标被成功实现的比例。

## 可复现要素
- **数据集**：AgentDojo benchmark，公开可用。
- **代码**：论文未明确声明代码开源状态，需进一步确认。
- **模型**：GPT-4o、GPT-4o-mini、Claude 3.5 Sonnet、Qwen2.5-7B-Instruct、Qwen3-32B、o4-mini。
- **超参**：temperature=0；reasoning effort level=medium（默认）。
- **提示模板**：附录 A 提供了 TDG 构建、参数估计、节点扩展的完整 prompt 模板。
