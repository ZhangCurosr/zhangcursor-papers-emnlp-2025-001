---
title: "IPIGUARD-A-Novel-Tool-Dependency-Graph-Based-Defense-Against"
source: https://aclanthology.org/2025.emnlp-main.53.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:42:52"
---

# 论文速读：IPIGUARD-A-Novel-Tool-Dependency-Graph-Based-Defense-Against-Indirect-Prompt-Injection-in-LLM-Agents

## 一句话总结
针对LLM Agent在处理不可信外部数据时易受间接提示注入（IPI）攻击的问题，本文提出IPIGUARD，通过将任务执行建模为工具依赖图（TDG）的拓扑遍历，显式解耦动作规划与外部数据交互，从执行结构层面阻断恶意工具调用，在保持高任务可用性的同时实现接近零攻击成功率的安全防御。

## 研究问题与动机
- **核心问题**：能否通过主动禁止与用户任务无关的工具调用，从源头上缓解IPI攻击？
- **现有方法缺陷**：当前防御多依赖高级提示策略、辅助检测模型或LLM-as-judge，本质上仍假设模型具备内在安全性，缺乏对Agent行为的结构性约束，易被自适应IPI攻击（如Tool Knowledge、Important Instruction）绕过。
- **架构脆弱性**：传统“边执行边规划”模式允许Agent在处理工具响应时随时发起新调用，攻击者只需在响应中嵌入注入指令即可劫持工具调用序列。
- **研究动机**：借助LLM日益增强的规划能力，在任务执行前预先确定工具调用链并施加严格执行约束，将安全防线从“模型层”前移至“执行层”。

## 核心贡献（创新点）
1. **提出基于TDG的执行中心防御范式**：将任务执行过程形式化为有向无环图遍历，显式解耦规划与外部数据交互，从根本上切断注入指令触发恶意工具调用的路径。
2. **Argument Estimation（参数估计）机制**：通过拓扑排序动态解析含未知参数的待处理节点，利用已执行工具的响应补全参数，解决规划阶段信息不全的核心挑战。
3. **Node Expansion（节点扩展）机制**：基于CQRS原则将工具分为Query与Command两类，仅允许执行阶段动态插入只读查询节点以收集上下文，在维持系统适应性的同时封锁写操作工具。
4. **Fake Tool Invocation（伪造工具调用）机制**：针对用户任务与注入任务共享工具的场景，通过注入模拟响应“消化”注入指令，强制Agent回归原始意图进行参数估计，避免参数被恶意篡改。

## 方法详解
- **TDG构建（Planning Phase）**：执行前，LLM结合用户指令、工具描述与系统上下文生成TDG。图中每个节点代表一次工具调用（含工具名与参数），有向边 `(u, v)` 表示节点v依赖节点u的返回结果。节点分为**Deterministic Nodes**（参数全已知）与**Pending Nodes**（含`<unknown>`参数，需后续推断）。
- **Argument Estimation（应对C1）**：按拓扑序遍历TDG。确定性节点直接执行；待处理节点执行时，LLM从执行上下文中检索前置工具的响应，推断并填充未知参数，转化为已解决节点后调用工具，新响应加入上下文继续后续节点解析。
- **Node Expansion（应对C2）**：当前节点执行完毕后，LLM评估是否缺失必要信息。若需，仅允许生成新的Query Tool调用作为扩展节点（Read-only），继承原节点后继关系并追加至执行队列；明确禁止Command Tool引入，平衡安全性与动态信息获取需求。
- **Fake Tool Invocation（应对C3）**：当待处理节点上下文中出现注入指令时，不直接修改原工具参数，而是提示LLM优先调用一个虚拟工具处理该指令。系统向执行上下文注入预定义的伪造响应（如“新任务已完成，将继续执行原任务”），使注入指令表面被“满足”实则失效，Agent随后基于原始用户意图完成参数估计。
- **执行闭环**：TDG构建 → 拓扑排序遍历 → 参数估计/节点扩展/伪造调用交替处理 → 整合最终环境状态输出。规划器与执行器可独立配置不同规模LLM以优化成本。

## 实验与结果
- **数据集**：AgentDojo benchmark（97个任务，629个测试用例，覆盖Workspace、Slack、Travel、Banking四大领域，强调多轮状态交互与对抗注入）。
- **模型**：GPT-4o、GPT-4o-mini、Claude 3.5 Sonnet、Qwen2.5-7B-Instruct、Qwen3-32B、OpenAI o4-mini。
- **攻击类型**：Ignore Previous、InjecAgent、Tool Knowledge、Important Instruction。
- **基线方法**：No Defense、Detector、Tool Filter、Spotlight、Sandwich。
- **评估指标**：Benign Utility (BU)、Utility under Attack (UA)、Attack Success Rate (ASR)。
- **核心结果**：
  - **安全性**：IPIGUARD在四种攻击下平均ASR仅**0.69%**，各场景ASR均≤2.78%，显著优于No Defense（13.16%）、Detector（4.43%）与Spotlight（11.31%）。
  - **可用性**：平均UA达**58.77%**，为所有方法最高；无攻击时BU为**67.01%**，接近No Defense的68.04%。
  - **开销**：Token消耗约为无防御的2倍（输入14,605 vs 6,165），平均耗时13.88秒（vs 7.13秒）。规划成本仅占总开销约20%，采用“强规划器+弱执行器”组合可显著优化性价比（如o4-mini规划 + GPT-4o-mini执行，UA提升至64.39%，成本仅增加$1.26）。
  - **消融实验**：仅用Node Expansion时ASR略升至4.77%；叠加FTI后ASR降至0.64%，BU/UA进一步升至69.07%/57.07%，验证两者互补必要性。

## 相关工作脉络
- **Prompt-based防御（Spotlight, Sandwich）**：依赖分隔符或目标重复引导模型忽略注入内容，属模型层软约束；IPIGUARD转为执行结构硬约束，不依赖模型服从性，对抗适应性更强。
- **Detection-based防御（Detector）**：使用BERT分类器检测注入并中止执行；IPIGUARD无需额外检测模型，通过TDG拓扑强制隔离敏感操作，避免漏检导致的级联失败。
- **LLM-as-a-Judge（Task Shield）**：监控中间步骤验证意图对齐，但judge本身可能被注入劫持；IPIGUARD从源头封禁非计划工具调用，规避了judge可信度问题。
- **工具使用基准（AgentDojo vs InjecAgent）**：AgentDojo聚焦多轮真实API环境与状态追踪；本文在此更复杂场景下验证防御泛化性，弥补了早期单轮基准的不足。
- **传统规划范式（ReAct, HuggingGPT）**：边执行边规划；本文借鉴其工具调度思想，但重构为“预规划+受控遍历”以注入安全边界，实现安全与功能的结构化解耦。

## 局限性与未来方向
- 仅防御干扰工具使用的IPI，未覆盖纯文本输出操纵型攻击（虽在实际工具环境中风险较低，但理论完整性有限）。
- 实验规模受限于LLM调用成本，未涵盖o3等更多模型，跨架构泛化性需进一步验证。
- 依赖具备较强规划能力的LLM，在资源受限或弱模型部署场景下适用性受限。
- 未来可探索轻量级专用规划器、自动化TDG安全性验证机制，以及向多Agent协作与开放互联网环境的扩展。

## 研究启发与可借鉴点
1. **执行中心安全范式可迁移**：将安全约束从模型提示层下沉至执行控制层（DAG遍历、权限白名单）是防御Agent类应用的通用设计原则，可直接复用于RAG流水线、Multi-Agent工作流等场景。
2. **CQRS思想引入Agent规划**：严格区分读操作（允许动态扩展）与写操作（静态限制），为Agent框架的安全-灵活权衡提供了可工程化的标准参考。
3. **伪造响应消解注入的巧思**：FTI通过“表面满足-实质拦截”策略适配强指令跟随的LLM，比单纯拒绝更稳健，该思路可迁移至对抗上下文劫持、越权指令注入等变体攻击。
4. **规划-执行分离的成本优化模型**：实验证明了强规划器+弱执行器的性价比优势，其配置方法与成本-效用评估框架可直接作为后续Agent安全研究的实验基线。
5. **动态参数补全流程**：Argument Estimation解决了静态规划的信息盲区，其“基于前置响应推断未知参数”的流程可抽象为通用模板，适用于需要链式调用的复杂业务编排。

## 关键术语表
- **Indirect Prompt Injection (IPI)**：将恶意指令嵌入Agent交互的外部数据源中，
