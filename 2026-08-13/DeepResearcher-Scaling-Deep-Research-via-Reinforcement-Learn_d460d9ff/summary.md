---
title: "DeepResearcher-Scaling-Deep-Research-via-Reinforcement-Learn"
source: https://aclanthology.org/2025.emnlp-main.22.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:47:18"
field: "LLM agent with tool use"
keywords: ["reinforcement learning", "web search agent", "deep research", "GRPO", "multi-agent", "open-domain QA"]
innovations: ["首个真实web环境端到端RL训练框架", "多agent浏览架构与主agent策略解耦", "涌现规划/交叉验证/反思等认知行为"]
benchmarks: ["NQ", "TQ", "HotpotQA", "2Wiki", "MuSiQue", "Bamboogle", "PopQA"]
---

# 论文速读：DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments

## 一句话总结
本文提出 DeepResearcher，首个在真实网络搜索环境中通过端到端强化学习（GRPO）大规模训练 LLM 深度研究 agent 的框架，使模型能够自主导航开放网络、跨源检索与综合信息，显著提升开放域 QA 性能并涌现规划、交叉验证等认知行为。

## 研究问题与动机
- **现有方法的局限**：当前基于 LLM 的深度研究系统主要依赖人工设计的 prompt 工作流（如 ReAct、Search-o1）或在受限的本地 RAG 环境中进行 RL 训练，无法应对真实开放网络的噪声、动态性和异构性。
- **RAG 环境的缺陷**：基于固定语料库的 RAG 系统存在信息时效性衰减、领域适应性差、存储瓶颈等问题，且无法模拟真实搜索中的抗爬机制、API 限流和网络延迟等挑战。
- **RL 扩展的潜力未被充分释放**：尽管 GRPO 等方法在数学和代码推理上取得突破，但在真实 web 搜索环境中进行端到端 RL 扩展的训练框架仍属空白。
- **缺乏可复现的研究框架**：商业产品（如 Gemini Deep Research、Grok3 DeeperSearch）功能强大但封闭，开源社区缺乏系统化、可复现的深度研究 agent 训练方案。

## 核心贡献（创新点）
1. **首个真实 web 环境下的端到端 RL 训练框架**：DeepResearcher 直接在真实搜索引擎上进行 GRPO 训练，区别于 RAG 方案依赖固定语料库，使 agent 学会在噪声环境中进行动态信息检索与综合。
2. **多 agent 浏览架构设计**：引入专门的 browsing agent，从完整网页结构中增量提取相关信息，而非简单检索预处理的文本片段，解决了真实网页内容异构的提取难题。
3. **工程挑战的系统性解决**：通过 50 节点分布式 CPU 集群处理高并发搜索请求、实现重试机制与搜索结果缓存、采用分段浏览策略以应对反爬机制，使真实环境 RL 扩展成为可能。
4. **涌现认知行为的实证分析**：通过 case study 展示 RL 训练后 agent 自发形成规划、交叉验证、反思重定向和诚实回答（承认无答案）等高级认知行为，而非依赖人工设计的工作流。

## 方法详解
- **轨迹结构**：每个 trajectory 包含 reasoning（`<thought>` 标签包裹）、web_search（JSON 格式调用，返回 title/URL/snippet）、web_browse（由内部多 agent 处理 URL 并提取信息）、最终 answer（`<answer>` 标签输出）四个阶段，每轮最多 10 次 tool call。
- **RL 算法**：采用 Group Relative Policy Optimization (GRPO)，不引入 critic 网络，用 G 条 rollout 的奖励均值作为基线估计优势函数，优化目标为：
  $$\mathcal{I}(\theta) = \mathbb{E} \frac{1}{G} \sum_{i=1}^{G} [\min(ratio \cdot A_i, \text{clip}(ratio, 1-\epsilon, 1+\epsilon) \cdot A_i)] - \beta D_{KL}(\pi_\theta \| \pi_{\theta_{ref}})$$
- **奖励设计**：格式错误惩罚 -1；格式正确时以词级 F1 score 为奖励，匹配短答案 QA 场景需求。
- **Observation Masking**：对 tool 返回的 observation 内容进行 masking，仅让模型的 action/response 参与训练，避免 observation 被模型记忆。
- **多 agent 浏览工具**：内部包含多个 Reading Agent 并行处理不同 URL 的分段内容，Synthesis Agent 合并结果；采用顺序处理与早期停止策略，假设初始段落无关则跳过整页。
- **训练数据构建**：从 NQ、TQ、HotpotQA、2Wiki 四数据集构建 8 万条样本，经过两级过滤（排除时间敏感/主观/有害问题 + 用 base model pass@10 检测记忆污染），多跳问题占比 75%。
- **工程实现**：基于 verl 框架， backbone 为 Qwen2.5-7B-Instruct，每步采样 256 prompts × 16 rollouts，mini-batch size 4096。

## 实验与结果
- **数据集**：ID 集（NQ、TQ、HotpotQA、2Wiki）+ OOD 集（MuSiQue、Bamboogle、PopQA），每集随机采样 512 样本（Bamboogle 全量 125 样本）。
- **评估指标**：词级 F1（与奖励对齐）+ Model-Based Evaluation（MBE，使用 GPT-4o-mini 作为 judge 判断正确性）。
- **主要结果（MBE）**：
  - DeepResearcher 在 TQ 上达 85.0（较 CoT+RAG 的 75.8 提升 9.2 分），在 2Wiki 上达 66.6（较 R1-Searcher 的 65.8 提升 0.8 分）。
  - OOD 泛化：Bamboogle 达 72.8（较 R1-Searcher 的 65.6 提升 7.2 分），显著超越所有 baselines。
  - 相较 prompt-based 方法提升最高达 28.9 分（CoT 在 TQ 仅 48.2）。
- **消融验证**：DeepResearcher (Local RAG) 在相同框架但本地语料下训练，性能全面下滑（如 TQ MBE 仅 55.5 vs 真实 Web 的 85.0），证明真实动态环境训练的必要性。
- **训练动态**：F1 从 0.375 逐步提升至 0.55；随训练推进，复杂问题的 tool call 次数和响应长度持续增长，未出现饱和。

## 相关工作脉络
- **Prompt-based 方法**（OpenResearcher、AirRAG、Search-o1、ReAct-style Agent）：依赖人工设计工作流和提示模板，缺乏自适应学习能力；本文通过 RL 自动探索最优策略。
- **RAG-based RL 方法**（Search-R1、ReSearch、R1-Searcher）：在静态本地语料库上训练，搜索空间受限且无法应对开放网络的噪声与多样性；本文强调真实 web 环境的训练价值。
- **SFT for RAG**（CoRAG、Auto-RAG）：通过监督微调优化检索流程，但仍受限于固定知识库架构；本文端到端 RL 训练使模型自主发现检索与推理策略。
- **推理增强 RL**（DeepSeek-R1、Kimi K1.5）：主要在数学/代码领域验证 GRPO 扩展效果；本文首次将其扩展至真实网络搜索这一动态、噪声丰富的工具交互场景。
- **定位差异**：本文是首个在真实 web 环境中进行大规模 RL 训练的研究框架，填补了从静态 RAG 到开放网络 agent 的训练鸿沟。

## 局限性与未来方向
- **模型规模受限**：当前实验仅基于 7B 参数模型（Qwen2.5-7B-Instruct），未探索更大规模模型的潜在性能增益与涌现能力。
- **奖励函数局限性**：当前 F1 + 格式惩罚的奖励设计仅适用于短答案 QA，难以直接迁移至开放式深度研究任务（需要长篇综合输出、定义模糊的问题空间）。
- **数据多样性不足**：训练数据主要来自英文通用 QA 数据集，未来需扩展至更广泛的领域和语言。
- **评估指标局限**：当前的诚实行为（主动拒绝无答案问题）未被现有 QA 评估体系捕捉，需要新指标衡量 agent 可靠性。

## 研究启发与可借鉴点
- **真实环境训练的必要性**：实验证明在噪声、动态的真实 web 环境中训练，比在静态 RAG 环境中训练更能培养可泛化的检索与推理能力，这一结论可推广至其他工具交互 agent 训练。
- **多 agent 工具内部架构与主 agent 解耦**：browsing tool 内部的多 agent 设计对主 agent 策略透明，主 agent 只需学习何时调用工具，这一解耦设计值得在复杂工具链中借鉴。
- **数据污染检测的双重过滤策略**：通过 base model pass@10 检测记忆污染 + 质量过滤构建训练集的方法，可作为后续 agent 训练数据工程的标准流程。
- **Observation Masking 技术**：在 tool-use RL 训练中屏蔽 observation 仅训练 action 的策略，可避免模型记忆工具输出而非学习推理过程，适用于其他工具交互场景。
- **涌现行为的可分析性**：通过 case study 识别规划、交叉验证、反思、诚实等涌现行为，为 agent 认知能力分析提供了可复用的定性分析框架。

## 关键术语表
- **DeepResearcher**：本文提出的端到端 RL 训练框架，用于在真实 web 搜索环境中训练 LLM 深度研究 agent。
- **GRPO（Group Relative Policy Optimization）**：一种无需 critic 网络的 RL 算法，通过多条 rollout 的奖励均值估计基线来优化策略。
- **RAG（Retrieval-Augmented Generation）**：检索增强生成，将外部知识库检索结果作为上下文输入 LLM 以提升生成质量的方法。
- **MBE（Model-Based Evaluation）**：使用 LLM-as-a-Judge 方法评估模型回答正确性的模型驱动评估指标。
- **Web Browsing Agent**：DeepResearcher 中专门负责从完整网页中提取相关信息的内部多 agent 子系统。
- **Observation Masking**：在 RL 训练中屏蔽工具返回的 observation 内容，仅让模型的 action 参与梯度更新的技术。
- **Cross-validation Behavior**：agent 在首次找到答案后仍继续检索验证的自发性行为，增强最终回答的可靠性。

## 可复现要素
- **数据集**：训练集由 NQ、TQ、HotpotQA、2Wiki 构建的 8 万条样本（未公开原始构建脚本，但提供了过滤 prompt 模板）；评测集使用各数据集官方 dev set 采样。
- **代码开源**：是，完整训练框架已开源：https://github.com/GAIR-NLP/DeepResearcher
- **模型权重**：论文未提及是否开源训练后权重。
- **关键超参**：backbone Qwen2.5-7B-Instruct；每步 256 prompts × 16 rollouts；mini-batch size 4096；每 rollout 最多 10 次 tool call；搜索 top-k=10；训练框架 verl；缓存有效期 7 天。
