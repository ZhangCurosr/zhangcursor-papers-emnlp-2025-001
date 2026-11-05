---
title: "Autoformalization-in-the-Wild-Assessing-LLMs-on-Real-World-M"
source: https://aclanthology.org/2025.emnlp-main.90.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:48:56"
field: "形式化数学与大语言模型"
keywords: ["autoformalization", "LLM", "Isabelle/HOL", "mathematical definitions", "theorem proving", "formal verification"]
innovations: ["引入 Def_Wiki 和 Def_ArXiv 两个面向真实数学定义的自动形式化基准", "提出分类细化（CR）针对 SYN/UDF/TUF 三类错误进行结构化指令约束", "提出形式化定义接地（FDG）通过后处理或提示增强 LLM 与形式化库的对齐能力"]
benchmarks: ["Def_Wiki", "Def_ArXiv", "miniF2F", "Lean4"]
---

# 论文速读：Autoformalization-in-the-Wild-Assessing-LLMs-on-Real-World-M

## 一句话总结
本文针对现实世界中复杂的数学定义，引入了两个新的自动形式化基准数据集（Def_Wiki 和 Def_ArXiv），系统评估了 LLM 将其转换为 Isabelle/HOL 的能力，并提出了分类细化（Categorical Refinement）和形式化定义接地（Formal Definition Grounding）两种增强策略，显著降低了 LLM 在自我修正和连接形式化数学库方面的失败率。

## 研究问题与动机
- **现有基准过于简单**：已有的自动形式化基准（如 miniF2F）主要包含基础算术运算和基本数学对象（整数、分数、实数），缺乏对更复杂、更抽象的自然出现数学陈述的评估能力。
- **数学定义的特殊挑战**：数学定义是数学话语的基础构件，通常高度依赖上下文、涉及抽象概念（向量、矩阵、概率等），要求模型同时形式化定义本身及其隐含的前置条件，挑战远高于简单习题。
- **LLM 自我修正能力不足**：初步实验表明，即使是最强的 LLM（GPT-4o）在收到定理证明器的二元反馈后，小模型的自我修正效果仍然有限，甚至可能退化。
- **形式化库对齐困难**：LLM 在生成 Isabelle/HOL 代码时常常引用不存在的理论文件或无法正确识别相关数学库，导致大量未定义项错误（UDF）。

## 核心贡献（创新点）
1. **引入两个新基准数据集**：收集了来自 Wikipedia（Def_Wiki，56 个定义）和 arXiv ML 论文（Def_ArXiv，30 个定义）的真实数学定义，填补了自动形式化领域缺乏复杂、真实世界数学陈述评测资源的空白；与 miniF2F 相比，这些定义的 token 数（均值 200 vs 70）、数学对象数（均值 7.6 vs 4.8）和公式数均显著更高。
2. **系统性错误分析与干预策略**：针对 GPT-4o 在 Isabelle/HOL 自动形式化中的三类主要错误（SYN/UDF/TUF）进行了定性分析，并提出了分类细化（CR）、符号细化（SR）和形式化定义接地（FDG）三种改进方法，其中 CR+FDG 组合可将 UDF 错误降低 43%、将自我修正能力提升最高达 16%。
3. **跨形式化系统的泛化验证**：将分类细化方法扩展至 Lean4，验证了其在不同定理证明器上的通用性，发现定义形式化在 Lean4 上同样面临比 miniF2F 更高的挑战。

## 方法详解
- **自动形式化任务定义**：将自然语言/LaTeX 符号映射到形式语言 F 的变换函数 f: S→F，通过 LLM 实现：f(s) = LLM(p_auto, {(s_i, φ_i)}, s)。
- **零样本提示（ZS）**：直接向 LLM 提供 LaTeX 格式的数学定义，要求输出 Isabelle/HOL 形式化代码，不给出示例。
- **二元细化（Binary Refinement）**：在 ZS 基础上，向 LLM 提供已生成的形式化代码及证明器返回的二元正确性反馈（correct/incorrect），尝试引导模型自我修正。
- **详细细化（Detailed Refinement）**：向 LLM 提供每个错误的具体类型、错误信息和所在代码行号，而非简单的二元反馈。
- **分类细化（Categorical Refinement, CR）**：针对 SYN/UDF/TUF 三类错误分别设计结构化指令约束——SYN 指令要求使用有效 Isabelle 符号（以 "\<" 开头必须以 "\>" 结尾），避免关键词冲突，正确书写运算符；UDF 指令要求所有提及项必须在局部上下文或导入的理论文件中有明确定义；TUF 指令要求操作数的类型必须与函数定义中声明的类型完全匹配。
- **符号细化（Symbolic Refinement, SR）**：基于规则的后处理方法，对 SYN 错误进行符号级修正：(1) 若符号以 "\<" 开头但缺少结尾 "\>"，则自动补全；(2) 将不存在的数学字体符号替换为 Isabelle 中对应符号。
- **形式化定义接地（Formal Definition Grounding, FDG）**：从 Isabelle/HOL 库中提取外部形式化定义及其来源，筛选与目标任务相关的定义，通过两种方式使用：Post-FDG（后处理）直接替换 LLM 生成的 preamble；Prompt-FDG（提示式）在上下文提供形式化定义，分为 Soft（允许 LLM 自行决定是否使用）和 Hard（强制 LLM 使用）两种变体。
- **评估指标**：主要指标为通过 Isabelle 证明器检查的成功率（Pass）；辅助指标包括首次错误出现位置（FEO，以正确行数比例衡量，值越高越好）及三类错误的百分比（SYN/UDF/TUF，越低越好）；时间超时（TRO）和无效输入（IVI）也作为监控指标。

## 实验与结果
- **数据集**：Def_Wiki（56 个定义，ML 领域 Wikipedia 条目）、Def_ArXiv（30 个定义，ICML 会议论文）、miniF2F-Test（244 个样本，用作对比基准）；评估统计量基于 GPT-2 tokenizer（token 数）和 GPT-4o 提示（数学对象数、公式数）。
- **评估模型**：DeepSeekMath-7B（数学专用小模型）、Llama3-8B（通用小模型）、GPT-4o（当前最强商业模型）；生成策略均为贪心解码。
- **主要结果**：
  - **定义比 miniF2F 更难**：GPT-4o 在 Def_Wiki 上 Pass 率仅 10.87%（vs miniF2F 的 25.41%，下降约 57%），FEO 从 48.90 降至 16.12。
  - **专业小模型可与大模型媲美**：DeepSeekMath-7B 在 Def_Wiki 上 Pass 率 10.87%，与 GPT-4o 持平，且 UDF 错误率更低（26.09% vs 50.00%），但 TRO 更高（34.78% vs 8.70%）。
  - **二元细化效果有限且对小模型有害**：GPT-4o 在 miniF2F 上通过二元细化 Pass 率略有提升（25.41%→29.10%），但在 Def_Wiki 上 DeepSeekMath-7B 的 Pass 率从 10.87% 下降至 6.52%。
  - **分类细化显著降错**：GPT-4o + CR-UDF + Detailed 在 Def_Wiki 上 UDF 错误率从 50.00% 降至 8.70%（下降 43%）；CR-SYN 将 SYN 错误从 19.57% 降至 8.70%。
  - **Post-FDG 大幅减少 UDF**：GPT-4o 在 miniF2F 上 Post-FDG 将 UDF 从 23.77% 降至 2.87%；在 Def_Wiki 上从 50.00% 降至 19.57%。
  - **Prompt-FDG 反而有害**：在 Def_Wiki 上，提供上下文形式化定义的 Soft-IFDC 和 Hard-IFDC 策略均导致性能下降（Pass 率从 34.78% 降至 19.57%），表明当前 LLM 难以有效利用上下文中的形式化定义。
  - **跨系统泛化**：在 Lean4 上，CR 策略同样有效，Def_Wiki 上 CR-UDF 使 Pass 率从 5.36% 提升至 10.71%。

## 相关工作脉络
- **Wu et al. (2022)**：开创性地将 LLM 应用于 Isabelle 自动形式化，本文在其提示策略基础上扩展了细化方法和错误分析视角。
- **Zheng et al. (2022, miniF2F)**：当前最主流的自动形式化基准，聚焦于竞赛级别数学问题，本文指出其样本过于简单（token 少、对象少），不足以反映真实场景的挑战，本文提出的定义数据集作为互补和进阶基准。
- **Azerbayev et al. (2023, ProofNet)**：涵盖本科到奥赛水平的非正式-形式化配对数据，但仍以习题风格为主，本文聚焦于真实数学文本中的定义，强调上下文依赖性和抽象性。
- **Zhang et al. (2024)**：研究通过符号等价和语义一致性进行自动形式化，本文采用定理证明器验证句法正确性、LLM-as-Judge 评估语义一致性的评测框架与其呼应。
- **Jiang et al. (2023, Draft, Sketch, and Prove)**：将自动形式化作为自动定理证明的中间步骤，本文专注于定义形式化这一前置但独立的任务。
- **Liu et al. (2025b, Atlas)**：通过数据生成流水线构建大规模并行语料微调 LLM，本文采用小规模高质量真实定义数据，强调数据质量而非规模。

## 局限性与未来方向
- **数据集规模较小**：Def_Wiki 仅 56 个定义，Def_ArXiv 仅 30 个，虽然规模不大但足以暴露核心挑战，扩展至更多科学领域和数据量是未来方向。
- **错误分析局限于 Isabelle/HOL**：部分结果可能无法直接推广至其他定理证明器（如 Lean、Coq），尽管验证了 CR 在 Lean4 上的泛化性，但 SR 和 FDG 的方法需要系统特定设计。
- **语义一致性尚未充分评估**：当前主要评估句法正确性（可通过定理证明器自动验证），语义对齐仍依赖人工检查，缺乏可靠的定量评估方法，是一个开放研究问题。
- **Prompt-FDG 效果不佳**：直接将形式化定义放入上下文中反而损害性能，说明 LLM 尚无法有效链接和利用上下文中的形式化知识，如何成功利用上下文形式化定义是一个重要开放问题。

## 研究启发与可借鉴点
- **错误驱动的细化设计**：先进行定性的错误分类分析（SYN/UDF/TUF），再针对每类错误设计具体的指令约束，这种"分析→干预"范式可迁移到其他形式化或代码生成任务中。
- **Post-processing 优于 in-context 学习**：对于形式化库引用问题，后处理（Post-FDG）比在提示中提供形式化定义（Prompt-FDG）更有效，提示工程应谨慎考虑上下文信息的呈现方式。
- **小模型专用化的价值**：DeepSeekMath-7B 在数学定义形式化上与 GPT-4o 表现相当，且 UDF 错误更少，说明在特定领域的小模型仍可具有竞争力，值得在资源受限场景下探索。
- **跨形式化系统的验证策略**：选择 CR 仅通过 prompt 修改即可适配 Lean4，而 SR 和 FDG 需要系统特定设计，这为后续工作提供了明确的扩展路径和优先级参考。
- **无需 ground-truth 的形式化评测**：本文采用定理证明器验证句法正确性 + LLM-as-Judge 评估语义一致性，避免了 ground-truth 形式化代码可能带来的数据泄露风险，这一思路可推广至其他形式化任务。

## 关键术语表
**Autoformalization**：将自然语言数学陈述转换为机器可验证的形式语言（如 Isabelle/HOL）的任务。
**Isabelle/HOL**：一种基于高阶逻辑的定理证明器，具有声明式证明语言，广泛用于形式化数学。
**Categorical Refinement (CR)**：针对特定错误类别设计的结构化指令细化方法，为 LLM 提供纠正特定类型错误的约束性指导。
**Formal Definition Grounding (FDG)**：将形式化数学库中与目标任务相关的定义作为辅助前提，增强 LLM 形式化代码中外部引用准确性的策略。
**Undefined Item Errors (UDF)**：L 生成的形式化代码中引用了未在局部上下文或导入理论文件中定义的项所产生的错误。
**Type Unification Failed Errors (TUF)**：操作数或参数的类型与 Isabelle/HOL 中已定义操作符/函数的类型声明不匹配而产生的错误。
**Symbolic Refinement (SR)**：基于规则的符号级后处理方法，自动修复 Isabelle 符号格式错误（如缺失闭合括号）。
**Post-FDG**：FDG 的一种实现方式，在 LLM 生成代码后通过后处理替换 preamble 中的理论导入。

## 可复现要素
- **数据集**：Def_Wiki（56 个定义）、Def_ArXiv（30 个定义），论文已声明发布，来源为 Wikipedia 机器学习类别和 arXiv cs.LG/stat.ML 子类别的 ICML 论文。
- **代码/权重**：论文未提供代码仓库链接，但使用了 DeepSeekMath-7B、Llama3-8B 和 GPT-4o，前两者开源，后者为商业 API。
- **关键超参**：所有设置均采用贪心解码（greedy decoding）；Prompt-FDG 设置中 Hard 版本强制使用上下文形式化定义，Soft 版本允许模型自由选择。
- **评估工具**：Isabelle/HOL 定理证明器用于句法正确性检查；Lean4 用于跨系统验证。
