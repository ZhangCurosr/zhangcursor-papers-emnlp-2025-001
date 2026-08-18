---
title: "Dialect-SQL-An-Adaptive-Framework-for-Bridging-the-Dialect-G"
source: https://aclanthology.org/2025.emnlp-main.178.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:47:16"
field: "Text-to-SQL与数据库交互"
keywords: ["Text-to-SQL", "SQL Dialect", "ORM Code", "Intermediate Language", "Bootstrapping Data Synthesis", "Large Language Models"]
innovations: ["首次引入ORM代码作为中间语言桥接Text-to-SQL方言差异，无需微调即可适配多数据库", "提出可控引导式迭代合成方法从少量种子示例构建高质量Text-to-Code演示池", "系统化验证ORM中间语言在五类主流关系数据库上的跨方言适配能力"]
benchmarks: ["Spider", "BIRD"]
---

# 论文速读：Dialect-SQL-An-Adaptive-Framework-for-Bridging-the-Dialect-Gap

## 一句话总结
论文提出了 Dialect-SQL，一个使用 ORM 代码作为统一中间语言的自适应框架，通过 Text-to-Code 范式解决 Text-to-SQL 中不同数据库方言（dialect）之间的适配鸿沟，无需针对特定方言进行微调即可在多种数据库系统上保持一致的生成性能。

## 研究问题与动机
- **方言鸿沟问题**：关系型数据库系统各自实现不同的 SQL 方言，同一查询在不同数据库中的标识符格式（如 SQLite 省略双引号、PostgreSQL 使用双引号、SQL Server 使用方括号）和语法结构（如取第6行的方式）存在显著差异，导致专为某一数据库生成的 SQL 无法在其他数据库中正确执行。
- **现有方法局限性**：当前 Text-to-SQL 研究主要聚焦于 SQLite（因 WikiSQL、Spider、BIRD 等主流数据集均基于 SQLite），所提方法缺乏方言适应性；例如 Llama-3.1-70B 在 BIRD 数据集上从 SQLite 迁移到 PostgreSQL 时，EX 下降高达 38.59%。
- **缺乏高质量 Text-to-Code 数据集**：公共数据集中仅有 gold SQL 标注，缺少对应的 ORM 代码，且 SQL 到 ORM 代码的自动转换尚无有效方法，导致缺乏可用于大语言模型训练的 Text-to-Code 对齐数据。

## 核心贡献（创新点）
1. **提出以 ORM 代码为中间语言的 Text-to-Code 范式**：首次引入 ORM（Object Relational Mapping）代码作为统一中间表示来桥接方言差异，与已有方法需针对每种方言训练不同模型的本质区别在于无需任何微调即可适配多种数据库。
2. **提出可控的引导式（bootstrapping）ORM 代码合成方法**：从仅 5 个手工构造的种子示例出发，通过执行反馈迭代生成并验证高质量 question-ORM code 对，逐步构建覆盖不同难度级别的演示池，解决了 Text-to-Code 数据稀缺问题。
3. **系统性地验证了跨方言适配能力**：在 Spider 及 BIRD（覆盖 SQLite、PostgreSQL、SQL Server、Oracle、MySQL 五种数据库）上的实验表明，Dialect-SQL 相较于直接 SQL 生成方法显著提升方言适应性，在 PostgreSQL 和 SQL Server 上 EX 平均下降仅 1.53% 和 2.12%。

## 方法详解
Dialect-SQL 包含两个阶段：

**离线阶段——引导式 ORM 代码合成（Bootstrapping ORM Code Synthesis）**：
- 初始演示池 $\mathcal{D}_{pool}$ 包含 5 个手工构造的种子示例 $\mathcal{D}_{seed}$。
- 迭代过程中，对训练集中的每条 (schema, question, gold SQL) 三元组，利用当前演示池引导 LLM 生成 ORM 代码片段 $\tilde{y}$ 及对应 SQL $\hat{y}$。
- 通过执行验证 $\hat{y}$ 与 gold SQL $y$ 结果是否一致，若等价则将 $(q, \tilde{y})$ 加入临时池 $\Delta\mathcal{D}$。
- 每轮结束后将 $\Delta\mathcal{D}$ 合并入 $\mathcal{D}_{pool}$，使已验证的正确示例持续增强 LLM 生成更难示例的能力，直到满足停止条件。

**在线阶段——方言自适应 SQL 生成（Dialect-Adaptive SQL Generation）**：
- **检索**：使用嵌入模型（bge-large-en-v1.5）将输入问题 $q$ 与演示池中所有问题编码为向量，通过余弦相似度检索 top-K 相关示例 $\mathcal{D}_q$。
- **ORM 代码生成**：将数据库 schema（以 class 形式定义）、检索到的示例 $\mathcal{D}_q$ 和自然语言问题 $q$ 以代码风格 prompt 格式输入 LLM，生成 ORM 代码片段 $\tilde{y}$：
  $$\tilde{y} = \arg\max_{y'} p_{LLM}(y' | S, \mathcal{D}_q, q)$$
- **SQL 转换**：使用 Python 的 SQLAlchemy ORM 框架将生成的 ORM 代码解析为面向目标方言的可执行 SQL 查询 $\hat{y}$；若转换失败（语法错误等），最多重试 $L$ 次重新生成 ORM 代码。

## 实验与结果
- **数据集**：Spider（SQLite）和 BIRD（扩展至 PostgreSQL、SQL Server、Oracle、MySQL 五种方言）；引导式合成后 Spider 训练集生成 7,930 条、BIRD 训练集生成 8,248 条 ORM code 演示样本。
- **评估基线**：Direct-SQL（直接生成目标方言 SQL，包含再生机制）；以及 SQLGlot 转译基线（将源方言 SQL 转译为 target dialect）。
- **主要结果（Llama-3.1-70B-Instruct）**：
  - Spider（SQLite）：Dialect-SQL 的 EX 达 **79.8%**，较 Direct-SQL 提升 **+2.5%**；EM 为 34.7%，VES 为 79.10%。
  - BIRD（SQLite）：Dialect-SQL 的 EX 达 **53.32%**，较 Direct-SQL 提升 **+4.36%**。
  - 方言适应性：相比 SQLite，在 PostgreSQL 上 EX 平均下降仅 **1.53%**，在 SQL Server 上下降 **2.12%**，显著优于 Direct-SQL。
- **最强结果**：gpt-4o-2024-11-20 上的 Dialect-SQL 在 BIRD 五种方言上的平均 EX 达到 **61.13%**，显著优于所有直接生成基线及转译基线（转译基线平均 EX 53.75%-57.39%）。
- **消融结论**：去掉再生机制后 Dialect-SQL 在 Spider 上 EX 下降 2.3%；使用含 hard examples（16%比例）的演示池可使 BIRD EX 提升 **1.89%**；实验还验证了 gemini-2.5-flash 上的结果趋势一致。

## 相关工作脉络
1. **LLM-based Text-to-SQL 研究**：主流方法聚焦提示工程（如 DIN-SQL、PTD-SQL）或模型训练（如 CODE-S，OmniSQL），但大多针对 SQLite 或单一方言设计；本文与 Pourreza et al. (2024, sql-gen) 均尝试解决方言问题，但 sql-gen 仍需针对新方言重新训练，而本文无需微调。
2. **中间语言方法**：Nat-SQL、SemQL 等使用简化版 SQL 作为中间表示，Pandas-like code 使用编程 API 作为推理轨迹；本文与这些方法的本质区别在于引入 ORM 代码专门针对**方言差异**而非仅简化表示。
3. **Text-to-Code 研究**：Qu et al. (2024, 2025) 使用 Pandas-like 代码缓解幻觉；本文使用 SQLAlchemy ORM 代码同时解决方言适配问题，且具有明确的执行转换器保障。
4. **SQL 转译（Transpilation）方法**：本文在 4.5 节将 Dialect-SQL 与基于 SQLGlot 的方言间转译方法进行对比，证明单一可靠源方言不存在，而 ORM 中间语言无需选择源方言即可实现跨方言适配。

## 局限性与未来方向
- **数据库类型局限**：仅在 SQLite、PostgreSQL、SQL Server、Oracle、MySQL 五种关系型数据库上验证，未测试 NoSQL 或其他数据库系统。
- **模型规模受限**：受成本限制仅测试了有限数量的 LLM，未探索更大/更小参数规模模型的表现。
- **中间语言单一**：仅使用 Python + SQLAlchemy，未探索其他编程语言或 ORM 框架（如 C# + EF Core，但后者不支持 SQL 输出）作为中间语言的可能性。

## 研究启发与可借鉴点
1. **引导式迭代数据合成策略**：从极少量种子示例出发、通过执行反馈逐步构建高质量演示池的方法，可迁移到其他缺乏标注数据的 Text-to-Code/Text-to-API 任务中。
2. **代码中间语言桥接异构表示**：将 ORM 代码作为"方言中立"的中间表示，这一思路可扩展到其他需要处理多版本/多方言输出的领域（如 Text-to-NoSQL、Text-to-DSL）。
3. **离线-在线两阶段框架设计**：离线数据构建与在线推理解耦的架构具有通用性，可在其他需要动态构建 in-context 示例的任务中复用。
4. **执行验证确保生成质量**：通过实际执行验证中间表示的正确性，再转化为最终输出，这一"生成-验证-转换"链路可推广到代码生成类任务的可靠性保障中。

## 关键术语表
- **Text-to-SQL**：将自然语言问题自动翻译为对应关系型数据库 SQL 查询的任务。
- **SQL Dialect（方言）**：不同数据库系统对 SQL 标准的具体实现差异，包括语法、标识符格式、内置函数等。
- **ORM（Object Relational Mapping）代码**：一种将数据库操作抽象为面向对象编程的代码形式，如 Python 的 SQLAlchemy，屏蔽底层 SQL 方言细节。
- **Bootstrapping（引导式合成）**：从少量种子示例出发，通过迭代生成、执行验证、累积正确样本的方式逐步构建大规模训练/演示数据。
- **Execution Accuracy（EX）**：生成查询与实际执行结果与 gold SQL 执行结果完全一致的百分比。
- **In-context Learning（上下文学习）**：通过在 prompt 中提供 few-shot 示例引导 LLM 生成符合要求的输出，无需微调模型参数。

## 可复现要素
- **数据集**：Spider 和 BIRD（原始基于 SQLite）；本文扩展 BIRD 至 PostgreSQL、SQL Server、Oracle、MySQL 五种方言，并公开了合成后的演示池数据。
- **代码开源**：是，项目代码和数据已开源（https://github.com/jieshi10/orm-sql）。
- **模型**：Llama-3.1-70B-Instruct、DeepSeek-R1-Distill-Qwen-32B、gpt-4o-2024-11-20、claude-3-7-sonnet-20250219、gemini-2.5-flash。
- **嵌入模型**：bge-large-en-v1.5。
- **ORM 框架**：Python SQLAlchemy。
- **关键超参**：检索示例数 K（图7a显示 K 变化对性能影响不显著）；最大重试次数 L；Llama-3.1 的 temperature=0.6、max tokens=256。
