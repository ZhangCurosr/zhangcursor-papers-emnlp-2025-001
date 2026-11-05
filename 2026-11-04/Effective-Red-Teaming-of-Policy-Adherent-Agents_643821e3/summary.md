---
title: "Effective-Red-Teaming-of-Policy-Adherent-Agents"
source: https://aclanthology.org/2025.emnlp-main.114.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:28:51"
field: "大语言模型智能体安全评估"
keywords: ["Policy-Adherent Agent", "Red-Teaming", "LLM Safety", "Adversarial Evaluation", "τ-bench", "pass@k"]
innovations: ["CRAFT多智能体红队框架，通过策略感知与欺骗规划显著超越通用越狱策略", "τ-break基准：从任务完成型基准转换为对抗性安全评估基准", "揭示强攻击能力与强防御能力不相关的反直觉规律"]
benchmarks: ["τ-bench", "τ-break"]
---

# 论文速读：Effective Red-Teaming of Policy-Adherent Agents

## 一句话总结
本文提出 CRAFT（Constraint-aware Red-teaming with Adversarial Framing and Tactics），一个面向策略合规型 LLM Agent 的多智能体红队测试系统，通过策略感知与欺骗性规划显著提升了突破合规规则的攻击成功率（航空领域 pass@1 达 70.0%）；同时引入 τ-break 基准（从 τ-bench 转化而来，覆盖航空/零售两个领域共 50 个策略违反用例），揭示了当前主流 Agent 在对抗用户下的严重脆弱性，并验证了现有轻量防御策略效果有限。

## 研究问题与动机
- **策略合规 Agent 部署需求与现实威胁存在鸿沟**：退款、取消等客服场景中，Agent 必须严格按业务规则拒绝违规请求，但恶意用户可能通过欺骗性话术诱导 Agent 违规，现有评估假设用户"合作"，严重低估了真实风险。
- **现有红队方法不适用**：通用 jailbreak（DAN）、直接提示注入（DPI）、情感操控等针对普通 LLM 的攻击策略，对策略合规 Agent 效果有限，无法暴露其真实脆弱点。
- **评估基准缺乏对抗视角**：τ-bench（Yao et al., 2024）虽在航空/零售领域构建了真实的工具调用场景，但仅评估任务完成正确率，未引入恶意用户模拟。
- **防御机制研究空白**：轻量策略提醒（policy reminder）和层级化 prompt 等防御手段在对抗攻击下效果微弱，系统化的安全加固路径尚不明确。

## 核心贡献（创新点）
1. **提出 CRAFT 多智能体红队框架**：通过 PolicyAnalyzer 提取策略片段、DeceptionPlanner 生成欺骗性指令、AvoidanceAdvisor 定义隐藏内容、DialogueExecutor 执行交互，使攻击策略明确围绕策略边界设计，而非通用语言模型越狱。
2. **设计 τ-break 对抗性基准**：将 τ-bench 任务从"任务完成评估"重新标注为"策略违反评估"——航空域通过 policy-free Agent 找出行列差异 $\Delta\mathcal{A} = \mathcal{A}_{\text{free}} \setminus \mathcal{A}_{\text{gold}}$ 并人工核验 20 个案例；零售域通过新增身份验证策略并添加 `verify_user_secret` 工具构造 30 个案例。
3. **揭示攻击成功与 Agent 鲁棒性不相关的反直觉发现**：Qwen2.5-70B 作为攻击者表现最强（平均 65.7% ASR），但作为被攻击目标时也是最脆弱的 Agent（GPT-4o 攻击下 80.0% ASR），说明强攻击能力不等于强防御能力。
4. **系统评估轻量防御策略的局限性**：验证了 Hierarchical Prompting、Full Policy Reminder、Relevant Policy Fragments Reminder 三种策略，发现即使是针对性最强的"相关策略片段提醒"在 k=4 时 ASR 仍超过 80%，表明现有 prompt 级防御不足以抵御有策略的对抗用户。

## 方法详解
- **CRAFT 四模块架构**：
  - **PolicyAnalyzer**：输入用户请求 + 完整政策文档，输出与该请求最相关的政策片段（精确复制原文，不 paraphrase），为后续规划提供策略知识基础。
  - **DeceptionPlanner**：基于用户请求和策略片段，生成"假设性前提指令"，格式为 `"Assume <A>"`，指示红队 Agent 引导目标 Agent 接受错误前提（如"假设已购买航班保险"），并在目标拒绝时重复或调整该前提。
  - **AvoidanceAdvisor**：生成"不应说什么"指令列表，每条格式为 `"Do not say <A>"`，指导红队避免暴露会触发拒绝的关键事实（如"不要提及是 basic economy 票"）。
  - **DialogueExecutor**：接收上述规划输出，执行多轮对话（最多 30 轮），动态调整回复以诱导目标 Agent 执行违规操作。
- **τ-break 数据集构建**：
  - **航空域**：用 GPT-4o 运行无政策约束的 policy-free Agent，比较其动作集合 $\mathcal{A}_{\text{free}}$ 与金标准 $\mathcal{A}_{\text{gold}}$，取差集 $\Delta\mathcal{A}$ 中涉及关键操作（`cancel_flight`, `modify_insurance` 等）的案例，人工核验 20 个策略违反用例。
  - **零售域**：在原始政策中新增身份验证约束（要求调用 `verify_user_secret` 工具），将原本合规的请求重新定义为"未认证即操作"的违规场景，构造 30 个用例。
- **评估指标 pass@k**：对每个 task-seed 执行 $n=4$ 次独立试验，成功攻击次数为 $c$，则
  $$pass@k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}$$
  当 $k=1$ 时退化为标准 ASR。

## 实验与结果
- **基线对比（航空域，GPT-4o 攻击 GPT-4o）**：
  - CRAFT：pass@1 = **70.0%**，pass@4 = 85.0%
  - Emotional Manipulation：pass@1 = 50.0%
  - DPI：pass@1 = 47.5%
  - Insist：pass@1 = 47.5%
  - Benign User (τ-bench baseline)：pass@1 = **42.5%**
  - DAN：pass@1 = **35.0%**
- **模型间对比（航空域，pass@1）**：CRAFT 攻击下，Qwen2.5-70B 作为目标最脆弱（80.0%），GPT-4o 作为攻击者平均 ASR 最高（68.5%），各模型平均 ASR 较 benign baseline 提升约 30.8%。
- **零售域结果**：即使身份验证策略极为简单，GPT-4o 在 CRAFT 攻击下 ASR 仍达 6.7%（pass@1），Qwen2.5-70B 攻击 GPT-4o 目标高达 31.7%（pass@1）。
- **消融实验（GPT-4o 攻击 GPT-4o）**：移除 PolicyAnalyzer 后 ASR 从 70.0% 降至 55.0%；移除 DeceptionPlanner 后降至 46.6%；移除两者后接近 benign baseline（50.0%），表明策略感知和欺骗规划缺一不可。
- **防御效果**：Relevant Policy Fragments Reminder 在 pass@1 将 ASR 从 70.0% 降至 36.2%（最优），但 pass@4 仍升至 55.0%；Hierarchical Prompting 几乎无效（pass@1 仅降至 57.5%）。

## 相关工作脉络
1. **τ-bench（Yao et al., 2024）**：首个大规模策略合规 Agent 基准，评估工具调用 Agent 在航空/零售场景的任务完成能力，但假设合作用户；本文在此基础上重新标注为安全评估基准 τ-break。
2. **DAN jailbreak（Shen et al., 2024a）**：通过角色扮演绕过 LLM 安全限制；本文证明此类通用越狱策略对策略合规 Agent 效果极差（ASR 仅 35%）。
3. **Emotional Manipulation（Vinay et al., 2024）**：利用情感诉求诱导 LLM 生成有害内容；在策略合规场景下效果次于 CRAFT。
4. **Direct Prompt Injection（Liu et al., 2023b）**：直接覆盖 agent 原有行为；本文发现此类"硬覆盖"对策略 Agent 几乎无效。
5. **AgentHarm（Andriushchenko et al., 2024）/ STWebAgentBench（Levy et al., 2024）**：面向 agent 安全性的通用基准，但未显式建模策略合规约束；本文强调策略合规场景的独特脆弱性。
6. **Foot-in-the-door 攻击（Nakash et al., 2025）**：针对 ReAct Agent 的渐进式诱导攻击；与本文 CRAFT 不同，未聚焦策略规则的边缘案例分析。

## 局限性与未来方向
- **合成环境限制**：所有测试用例来自 τ-bench，完全合成，无法覆盖真实用户行为的复杂性与对抗创造力。
- **策略获取假设过强**：攻击者需完整访问 Agent 政策文档，闭源或私有部署场景下该假设不成立；部分策略获取（partial/inferred policy knowledge）下的攻击策略待研究。
- **策略类型有限**：仅考察航空/零售两个领域的静态简单策略，法律、医疗、多方谈判等动态或模糊策略场景未覆盖。
- **测试规模偏小**：仅 50 个对抗用例（航空 20 + 零售 30），难以充分覆盖大规模部署中的行为多样性。
- **静态攻击设定**：红队不会根据历史交互结果迭代调整策略，未建模真实场景中的对抗升级过程；迭代式对抗训练（iterative adversarial training）是重要未来方向。
- **模型覆盖有限**：仅评估 5 个模型，更多弱模型或领域专用 Agent 的脆弱性尚未探索。

## 研究启发与可借鉴点
1. **"Benign-to-Security"数据集转换范式**：将现有任务完成型基准（如 τ-bench）通过 policy-free Agent 差异分析（$\Delta\mathcal{A}$）或策略增强（如新增身份验证约束）转换为安全评估基准，是一种低成本、高可扩展的数据集构建方法，可迁移到其他领域基准的安全化改造。
2. **多角色分工的红队框架设计**：CRAFT 将红队过程分解为"策略感知→欺骗规划→规避建议→执行"四个明确角色，每个角色有独立的 prompt 设计，这种模块化设计便于替换/组合不同模块（如更换规划模型或策略提取器），为通用红队系统设计提供了参考架构。
3. **pass@k 指标在随机性攻击评估中的适用性**：采用超几何分布估计的 pass@k 指标，比单一 ASR 更稳健地刻画了随机采样下的攻击成功率，可推广至其他具有随机性的 agent 安全评估场景。
4. **强攻击 ≠ 强防御的反直觉规律**：本文揭示的模型角色不对称性（Qwen2.5-70B 攻强守弱）为后续研究提供了明确信号：评估 Agent 安全性时，应区分其作为攻击者和作为目标的能力，不能简单外推。
5. **策略片段提醒 vs 全量提醒**：Relevant Policy Fragments Reminder 效果显著优于 Full Policy Reminder，表明在长对话中"精准聚焦"比"信息堆叠"更能维持策略对齐，这一原则可推广至其他需要持续策略遵循的 agent 系统。

## 关键术语表
- **Policy-Adherent Agent**：必须在多轮对话和工具调用中严格遵循预设业务策略（如退款时限、取消规则）的 LLM Agent，常见于客服、金融、医疗等高约束场景。
- **CRAFT**：Constraint-aware Red-teaming with Adversarial Framing and Tactics，本文提出的多智能体红队系统，通过策略感知和欺骗性规划突破策略合规 Agent。
- **τ-break**：基于 τ-bench 构建的对抗性安全评估基准，将原任务完成用例重新标注为策略违反用例，覆盖航空（20 例）和零售（30 例）两个领域。
- **pass@k**：在 $n$ 次独立试验中，至少 $k$ 次攻击成功的概率估计，计算公式为 $1 - \binom{n-c}{k}/\binom{n}{k}$，用于衡量随机性攻击评估的稳健性。
- **PolicyAnalyzer**：CRAFT 中负责从完整策略文档中提取与当前用户请求最相关片段的核心模块，输出严格为原文子串。
- **DeceptionPlanner**：CRAFT 中生成"假设性前提指令"（`Assume <A>`）的规划模块，引导红队 Agent 让目标 Agent 接受错误前提以触发策略违规。
- **AvoidanceAdvisor**：CRAFT 中定义"不应说什么"的规避建议模块，防止红队因暴露关键禁用信息而触发目标 Agent 的拒绝逻辑。
- **policy-free Agent**：忽略策略约束、只追求任务完成的对照 Agent，用于通过与金标准动作集合的差集分析识别潜在策略违反。

## 可复现要素
- **数据集**：τ-bench（原始任务基准）可从 GitHub 获取（`tau_bench` 仓库，`envs/airline/wiki.md` 和 `envs/retail/wiki.md`）；τ-break 用例为本文构造，作者声明"将向受影响的受信任方提供工件，但公开发布时将撤回攻击实现细节"（双重用途声明）。
- **代码/权重**：论文未提供开源代码仓库链接；所有模型均通过 API 调用（GPT-4o, GPT-4o-mini, LLaMA-3.3-70B-Instruct, Qwen2.5-70B, DeepSeek-V3），未微调。
- **关键超参**：temperature = 0，seed = 10，最大对话轮数 = 30，每任务重复试验 $n = 4$；规划 Agent（PolicyAnalyzer / DeceptionPlanner）统一使用 LLaMA-3.3-70B。
