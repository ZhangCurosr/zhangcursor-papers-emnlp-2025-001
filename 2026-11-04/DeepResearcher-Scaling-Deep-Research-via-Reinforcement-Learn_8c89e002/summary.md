---
title: "DeepResearcher-Scaling-Deep-Research-via-Reinforcement-Learn"
source: https://aclanthology.org/2025.emnlp-main.22.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:28:18"
field: "大语言模型智能体与强化学习"
keywords: ["reinforcement learning", "web search", "deep research", "multi-agent", "GRPO", "LLM agent"]
innovations: ["首个在真实网络环境中端到端RL规模化训练深度研究代理的框架", "解决真实web搜索高并发、反爬、API限速等技术挑战的多智能体架构", "涌现规划、交叉验证、反思、诚实等认知行为的端到端RL训练方法"]
benchmarks: ["NQ", "TQ", "HotpotQA", "2Wiki", "MuSiQue", "Bamboogle", "PopQA"]
---

# 论文速读：DeepResearcher: Scaling Deep Research via Reinforcement Learning in Real-world Environments

## 一句话总结
本文提出了**DeepResearcher**，首个在真实网络环境中通过端到端强化学习（RL）规模化训练大语言模型（LLM）深度研究代理的综合框架，解决了现有RAG方法和提示工程方法无法处理动态、嘈杂开放网络搜索环境的局限。

## 研究问题与动机
- **核心问题**：当前LLM深度研究能力主要依赖脆弱的提示工程或RAG-based强化学习，但这些方法在受控环境中训练，无法捕捉真实网络的动态性、噪声和异构性。
- **现有方法不足**：
  1. **提示工程方法**（如Search-o1、ReAct）依赖人工设计工作流，缺乏自适应能力，泛化性差
  2. **RAG-based RL方法**（如Search-R1、ReSearch）依赖静态本地语料库，搜索空间受限，无法处理实时信息和多跳推理
  3. **缺乏真实环境训练**：现有RL研究多在仿真或受控环境中验证，未解决真实网络搜索中的高并发、反爬机制、API限速等技术挑战
  4. **信息提取瓶颈**：RAG系统简单检索预处理的文本片段，无法理解复杂网页结构并增量提取相关信息

## 核心贡献（创新点）
1. **首个端到端RL规模化训练框架**：DeepResearcher直接在真实网络环境中训练LLM代理，通过仅使用基于结果的奖励（outcome-based rewards）进行端到端强化学习，与需要SFT规划数据的显式规划方法本质不同。
2. **真实网络环境适配技术**：解决了真实web搜索引入的独特挑战（高并发I/O管理、API速率限制、反爬机制、网络延迟），而RAG环境无需处理这些问题。
3. **多智能体信息提取架构**：采用专门的多智能体架构，其中浏览智能体（browsing agents）从整个网页中提取相关信息并维护短期记忆，而非RAG系统简单检索预处理的文本片段。
4. **涌现认知行为**：端到端RL训练使模型自发涌现出规划（planning）、交叉验证（cross-validation）、反思（self-reflection）和诚实（honesty）等认知行为，无需人工设计的工作流约束。
5. **开放源码与全面实验**：开源完整训练框架，在7个开放域QA数据集上验证，相比提示工程基线最高提升28.9分，相比RAG-based RL代理最高提升7.2分。

## 方法详解
### 3.1 深度研究轨迹
- **推理过程**：每个推理步骤包裹在`<think>`标签中（遵循DeepSeek-R1设定），模型在采取行动前进行推理。
- **网络搜索工具**：通过生成JSON格式请求调用`web_search`工具，返回结构化结果（title、URL、snippet），固定top-k=10。
- **网页浏览智能体**：
  - 接收`web_browse`请求，处理URL的第一个网页片段
  - 维护每个查询的短期记忆库
  - 决定继续读取下一个URL/片段还是停止，并将相关信息追加到短期记忆
  - 停止时编译所有新增信息返回给主代理
  - **内部多智能体实现**：多个"Reading Agents"并行处理不同网页，"Synthesis Agent"合并 findings 返回给主代理

### 3.2 真实环境挑战应对
- **挑战I：高并发请求**：创建50节点分布式CPU服务器集群管理RL rollout过程中的工具请求（如4096次采样），核心需求是高并发I/O管理。
- **挑战II：爬取与API限制**：
  - 实现稳健的重试机制应对反爬措施
  - 引入搜索结果缓存策略（7天内相同查询直接返回缓存结果）
- **挑战III：信息提取优化**：
  - 将长网页分割为小片段顺序处理
  - 假设：如果初始片段主要包含无关信息，则网页可能无产出，可跳过
  - 多URL并行处理提高效率

### 3.3 RL训练框架
- **算法选择**：采用Group Relative Policy Optimization (GRPO)
- **损失函数**：
  $$\mathcal{I}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, \{y_i\}_{i=1}^G \sim \pi_{\theta_{old}}(\cdot|x)} \frac{1}{G} \sum_{i=1}^G [\min(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)} A_i, \text{clip}(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)}, 1-\epsilon, 1+\epsilon)A_i) - \beta \mathbb{D}_{KL}(\pi_\theta || \pi_{\theta_{ref}})]$$
- **观察掩码**：工具输出为观察（observation）而非期望结果，应用掩码防止观察参与训练，仅让模型响应贡献训练过程。

### 3.4 奖励机制
- **主要奖励**：F1分数（基于短答案开放域QA数据集）
- **奖励规则**：
  $$\text{reward} = \begin{cases} -1 & \text{if format is incorrect} \\ \text{F1 score} & \text{if format is correct} \end{cases}$$
- **格式惩罚**：格式错误（缺少标签或结构错误）惩罚-1
- **F1奖励**：基于词级F1分数评估生成答案与参考答案的准确性

## 实验与结果
### 数据集与评估
- **训练数据**：80,000个样本，两阶段过滤（低质量提问过滤+污染检测），NQ:TQ:HotpotQA:2Wiki比例1:1:3:3（75%多跳场景）
- **基线模型**：Qwen2.5-7B-Instruct
- **评估基准**：
  - **域内（ID）**：NQ、TQ、HotpotQA、2Wiki（各采样512个样本）
  - **域外（OOD）**：MuSiQue、Bamboogle、PopQA
- **评估指标**：F1分数（基于训练奖励）、MBE（Model-based Evaluation，使用GPT-4o-mini作为judge）

### 主要结果
| 方法 | NQ MBE | TQ MBE | HotpotQA MBE | 2Wiki MBE |
|------|--------|--------|--------------|-----------|
| CoT + RAG | 59.6 | 75.8 | 43.8 | 24.8 |
| Search-o1 | 55.1 | 69.5 | 42.4 | 37.7 |
| ReAct-style Agent | 39.6 | 49.2 | 26.2 | 17.6 |
| Search-r1-base | 60.0 | 76.2 | 63.0 | 47.9 |
| R1-Searcher | 52.3 | 79.1 | 53.1 | 65.8 |
| DeepResearcher (Local RAG) | 46.3 | 55.5 | 35.4 | 27.5 |
| **DeepResearcher** | **61.9** | **85.0** | **64.3** | **66.6** |

- **最强结果**：DeepResearcher在TQ上MBE达85.0，较CoT+RAG提升9.2分，较Search-r1-base提升8.8分
- **域外泛化**：在MuSiQue（29.3）、Bamboogle（72.8）、PopQA（52.7）上全面领先所有基线
- **真实环境重要性**：与Local RAG版本对比显示性能大幅下降，验证了嘈杂动态真实网络的必要性训练场
- **提升幅度**：较提示工程基线最高提升28.9分，较RAG-based RL代理最高提升7.2分

## 相关工作脉络
1. **Search-R1 (Jin et al., 2025)**：RL方法，但在训练和推理时使用检索器搜索Wikipedia，限制搜索空间；DeepResearcher不限制特定领域，允许模型自主选择URL。
2. **R1-Searcher (Song et al., 2025)**：通过 appending "site:en.wikipedia.org" 进行Bing搜索并总结top 3结果；DeepResearcher在真实环境中训练，不强制总结top 3结果。
3. **CoT + RAG / Plan*RAG (Verma et al., 2025)**：基于提示工程的检索增强方法，依赖人工设计的检索和推理流程；DeepResearcher通过端到端RL训练自发学习策略。
4. **Search-o1 (Li et al., 2025a)**：多步推理基线，在本地RAG环境下表现良好；但在真实web搜索环境中性能下降，说明真实环境训练的重要性。
5. **ReAct-style Agent**：零样本提示基线，使用提供的工具回答；缺乏端到端训练，无法适应复杂多跳场景。
6. **DeepResearcher (Local RAG)**：消融实验版本，使用相同RL框架但限制在本地RAG环境；性能全面落后，证明真实网络环境训练的必要性。

## 局限性与未来方向
- **模型规模局限**：实验仅使用7B参数模型（Qwen2.5-7B-Instruct），未探索更大模型的潜在性能提升或涌现能力。
- **奖励机制局限**：当前F1分数奖励适用于短答案事实性问题，但不适用于开放式深度研究问题（涉及模糊问题空间、大量信息综合、长形式 nuanced 输出）。
- **未来方向**：
  1. 将框架扩展到更大规模基础模型，探索scaling law
  2. 开发针对开放式深度研究的reward结构和训练方法
  3. 解决long-form答案的评估问题（当前F1不适用于长文本）
  4. 探索LLM驱动的动态top-k优化（当前固定top-k=10）

## 研究启发与可借鉴点
1. **端到端RL训练范式**：可借鉴"仅使用基于结果的奖励+观察掩码"的端到端训练策略，避免人工设计工作流对模型策略的约束，让模型自发学习搜索和推理策略。
2. **真实环境挑战应对技术**：高并发I/O管理（分布式集群）、API缓存策略、重试机制、反爬应对等技术可直接迁移到任何需要真实网络交互的agent系统。
3. **数据污染检测流程**：两阶段过滤（低质量提问过滤+pass@10污染检测）可有效防止模型利用参数知识"作弊"，确保真正学习搜索策略。
4. **多智能体信息提取架构**：浏览智能体维护短期记忆、顺序处理网页、假设跳过无产出网页的策略，可应用到任何需要网页理解的agent系统。
5. **涌现行为分析方法**：通过case study识别规划、交叉验证、反思、诚实等涌现认知行为，为RL训练agent提供可解释性分析框架。

## 关键术语表
- **GRPO (Group Relative Policy Optimization)**：群体相对策略优化算法，通过利用参考策略和rollout轨迹估计基线，无需单独训练critic网络。
- **Outcome-based Rewards**：基于结果的奖励，仅根据最终答案正确性给予奖励，不干预中间推理过程。
- **Observation Masking**：观察掩码技术，防止工具输出（observation）参与训练，仅让模型响应贡献梯度更新。
- **MBE (Model-based Evaluation)**：基于模型的评估，使用LLM-as-a-Judge（如GPT-4o-mini）评估答案正确性，适用于长文本评估。
- **Multi-hop Reasoning**：多跳推理，需要整合多个信息来源才能回答的复杂问题，占训练数据75%。
- **Search-dependent Training Data**：依赖搜索的训练数据，通过pass@10污染检测确保模型无法仅凭参数知识回答。
- **Cross-validation**：交叉验证行为，模型在首次找到答案后仍继续验证准确性，提升最终答案可靠性。
- **Honesty Behavior**：诚实行为，模型能识别未找到正确答案的情况并适当拒绝回答，减少幻觉。

## 可复现要素
- **数据集**：训练集80,000样本（NQ/TQ/HotpotQA/2Wiki比例1:1:3:3），两阶段过滤；评估集从各基准dev set随机采样512样本（Bamboogle全量125样本）
- **代码开源**：是，https://github.com/GAIR-NLP/DeepResearcher
- **权重开源**：论文未明确提及，仅开源代码
- **关键超参**：
  - 骨干模型：Qwen2.5-7B-Instruct
  - 训练框架：verl
  - 每步采样：256 prompts × 16 rollouts
  - 每rollout：最多10次工具调用 + 最终答案步骤
  - mini-batch size：4,096
  - top-k搜索结果：10
  - 缓存周期：7天
  - 服务器集群：50节点CPU分布式集群
