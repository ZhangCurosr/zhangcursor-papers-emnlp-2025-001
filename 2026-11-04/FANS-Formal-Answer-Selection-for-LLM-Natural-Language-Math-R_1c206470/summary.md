---
title: "FANS-Formal-Answer-Selection-for-LLM-Natural-Language-Math-R"
source: https://aclanthology.org/2025.emnlp-main.158.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:52:34"
field: "LLM数学推理与形式化方法"
keywords: ["形式化验证", "数学推理", "答案选择", "Lean4", "大语言模型", "Long-CoT", "测试时扩展"]
innovations: ["提出首个将Lean4形式化验证引入LLM NL数学推理答案选择的三阶段框架FANS", "设计基于迁移学习的Long-CoT翻译器训练方法解决无Long CoT标注数据的翻译任务", "引入外部强模型翻译检查与ORM/FANS组合策略提升答案选择可靠性"]
benchmarks: ["MATH500", "AMC23", "Minerva Math", "Olympiad Bench"]
---

# 论文速读：FANS-Formal-Answer-Selection-for-LLM-Natural-Language-Math-R

## 一句话总结
论文提出了 FANS（Formal ANswer Selection）框架，通过引入形式化数学语言 Lean4，将大语言模型（LLM）生成的自然语言数学答案翻译为形式化定理并自动证明验证，从而在多种测试中显著提升 LLM 数学推理的答案选择准确性。

## 研究问题与动机
1. **自然语言推理的不可验证性**：LLM 在数学推理中面临严重幻觉问题，自然语言本身存在固有模糊性，导致生成答案缺乏可验证性和可信支撑。
2. **现有答案选择方法的局限**：多数投票（Majority Vote）过于简单无法充分释放 pass@n 潜力；基于奖励模型（Reward Model）的最佳-N排序方法存在分布偏移（OOD）泛化困难的问题。
3. **形式化语言的潜力未被充分利用**：尽管 Lean4 等 formal language（FL）已在定理证明领域取得进展，但极少有研究利用 FL 反向增强 LLM 的自然语言数学推理能力。
4. **测试时计算扩展（Test-time Scaling）的需求**：LLM 在有限尝试内难以找到正确解，但多次尝试后往往能给出正确答案，如何从多个候选答案中准确筛选成为关键挑战。

## 核心贡献（创新点）
1. **首个将 Lean4 形式化验证引入 LLM NL 数学推理答案选择的框架**：FANS 通过 NL→FL 翻译、形式化证明生成与编译器验证三阶段流程，为答案选择提供计算机可验证的数学基础，区别于纯 LLM 驱动的投票或奖励模型方法。
2. **提出基于迁移学习的 Long-CoT 翻译器训练方法**：针对缺乏 Long CoT 标注的 NL-FL 翻译数据，采用"训练时禁止 Long CoT + 推理时启用 Long CoT"的迁移学习策略，使 LoT-Solver 基座模型学会 Faithful 翻译。
3. **设计了可组合的答案选择增强机制**：FANS 可与多数投票、奖励模型等方法联合使用，在代数与数论等 Lean4 库支持良好的子领域取得最高 7.90% 的精度提升，在 AMC23 数据集上最多提升 8.33%。
4. **引入外部强模型进行翻译忠实度检查**：通过 QwQ-32B 等更强模型对 NL-FL 翻译结果进行二次校验，有效降低误译导致的假阳性证明，提升整体框架可靠性。

## 方法详解
FANS 框架包含三个协同阶段：

**阶段一：NL 到 FL 翻译（LeanTranslator）**
- 从 Lean-Workbook 和 DeepSeek-Prover-v1 数据集中收集 162,181 条 NL-FL 对齐的定理陈述数据。
- 以 LoT-Solver 为基座模型，采用迁移学习策略：训练时在系统提示中明确要求"不使用 Long CoT"并提供空的 Long CoT 内容；推理时则指令模型使用 Long CoT 进行分析后再翻译。
- 可选引入 Kimina-Autoformalizer 作为替代翻译器，并使用 QwQ-32B 进行翻译忠实度二次检查。

**阶段二：形式化证明生成与验证**
- 将翻译得到的 Lean4 定理陈述传入 Lean4 证明器（如 DeepSeek-Prover-v1.5/v2、Goedel Prover），生成形式化证明。
- 使用 Lean4 编译器（Verifier，如 Santos et al. 2025）进行机械验证，基于归纳构造演算（calculus of inductive constructions）的类型检查系统确保逻辑一致性。
- 验证通过的证明即为正确答案的可验证证据。

**阶段三：答案选择与输出**
- 对于成功验证的答案进行多数投票；若最高票数低于预设阈值，则回退到原始多数投票以避免误译导致的假阳性影响。
- 当奖励模型可用时，将 FANS 验证结果与 ORM 结合，利用 ORM 分数判断问题难度并动态选择最佳策略。

## 实验与结果
**数据集**：MATH500（500 道高中数学题，含 7 个子领域）、AMC23（40 道数学竞赛题）、Minerva Math、Olympiad Bench。

**基线模型**：Mistral-7B、DeepSeek-Math-7B-Instruct、Qwen-2.5-Math-1.5B/7B-Instruct；奖励模型统一使用 Qwen-RM-72B。

**主要结果**：
- 在 MATH500 上，FANS 相对于多数投票最高提升 **1.91%**（DeepSeek-Math），相对于 ORM+FANS 在数论子领域提升 **7.90%**。
- 在 AMC23 上，Mistral+FANS 相对多数投票提升 **20.00%**，DeepSeek-Math+FANS 相对 ORM 提升 **8.33%**。
- 在 Qwen-2.5-Math-7B 上，ORM+FANS 在代数子领域提升 **1.67%**，在数论提升 **1.75%**。
- 使用新翻译器（Wang et al. 2025a）和 SOTA 证明器（Ren et al. 2025）时，FANS w/ external check 在 MATH500 上达到 **61.00%**（Llama-3.2-3B）。

**最强结果**：Qwen-2.5-Math-7B + ORM + FANS 在 MATH500 上达到 **88.80%**（FANS remove 即去除假阳性后的上限）。

## 相关工作脉络
1. **Zhou et al. (2024) Don't trust: Verify**：提出了类似的自动形式化框架，但依赖弱基座模型和纯 prompt-based 方法，而 FANS 训练了专用 Long-CoT 翻译器并使用更先进的证明器。
2. **MA-LoT (Wang et al. 2025b)**：利用 Long CoT 和多智能体框架解决 FL 问题，FANS 借鉴其迁移学习思想但目标不同——FANS 用 FL 增强 NL 推理而非直接解 FL 问题。
3. **DeepSeek-Prover 系列 (Xin et al. 2024b,a; Ren et al. 2025)**：大规模数据标注训练的形式化证明器，FANS 将其作为可替换的证明引擎，体现了框架的解耦设计。
4. **Lean-Dojo (Yang et al. 2024c)**：基于检索的方法增强定理证明，FANS 关注的是如何反向利用已证明的 FL 结果服务 NL 答案选择。
5. **传统答案选择方法（Majority Vote, ORM-based Best-of-N）**：FANS 指出这些方法在 OOD 场景下的泛化缺陷，提供了一条基于形式化验证的替代路径。
6. **过程奖励模型（Process Reward Model）**：如 Zhang et al. (2024b) 提出的 generative verifiers，FANS 强调其验证基于形式化逻辑而非 LLM 自身评分，提供更强的可信保证。

## 局限性与未来方向
1. **翻译假阳性问题**：NL-FL 翻译过程中可能出现定理表述错误（如将方程求解问题误译为恒等式），导致 vacuous proof 和错误答案选择。
2. **证明器覆盖率不足**：Lean4 库生态偏向代数与数论，几何、组合数学等领域支持有限；当前 SOTA 证明器在形式化基准上仍有约 40% 的问题无法证明。
3. **框架性能与理论上限的差距**：当前 FANS 性能距离 pass@n 的理论上限仍有较大空间，需要同时提升翻译忠实度和证明成功率。
4. **未来方向**：改进翻译器的约束保持能力和优化条件显式处理；通过迭代微调和改进证明搜索策略增强证明器能力；扩展 Lean4 库覆盖更多数学分支。

## 研究启发与可借鉴点
1. **迁移学习训练策略可复用于其他翻译任务**："训练时禁止 Long CoT + 推理时启用 Long CoT"的范式，可有效利用无 Long CoT 标注数据训练具备链式推理能力的翻译器，适用于 NL-to-FL 或其他形式化语言翻译场景。
2. **形式化验证与统计方法的组合策略**：FANS 展示了将严格的形式化验证与灵活的奖励模型/多数投票相结合的设计思路，为多方法融合提供借鉴——可用形式化验证处理高置信度案例，用统计方法兜底。
3. **二次翻译检查机制**：引入更强外部模型对翻译结果进行忠实度校验，是一种低成本高效率的错误缓解策略，可迁移到任何涉及自动形式化的 pipeline 中。
4. **子领域性能差异分析**：FANS 在代数/数论上表现优异揭示了形式化语言选型与领域适配的重要性，后续工作可借鉴此思路分析不同形式化系统的领域优势。
5. **解耦的三阶段框架设计**：翻译器、证明器、验证器三者高度解耦，允许独立替换升级各组件，这种模块化设计值得在类似多阶段 pipeline 中参考。

## 关键术语表
**FANS（Formal ANswer Selection）**：一种利用 Lean4 形式化语言验证 LLM 生成的数学答案正确性的三阶段框架。
**Long-CoT Translator（长链式思维翻译器）**：能够利用详细推理步骤将自然语言数学问题翻译为 Lean4 定理陈述的模型。
**Lean4**：最新的 Lean 定理证明器，基于归纳构造演算，支持机械验证数学证明。
**ORM（Optimized Reward Model）**：经过优化的奖励模型，用于对 LLM 生成答案进行质量评分和排序。
**Pass@n**：在 n 次独立生成尝试中至少有一次正确的概率度量，反映测试时计算扩展的效果。
**False Positive（假阳性）**：在此语境中指翻译错误导致本应错误的答案被形式化证明"验证通过"的错误情况。
**Calculus of Inductive Constructions（CIC）**：Lean4 所基于的类型论基础，确保每个证明都能通过类型检查系统机械验证。
**Test-time Scaling（测试时扩展）**：在推理阶段增加计算资源（如多次采样）以提升模型性能的策略。

## 可复现要素
- **数据集**：MATH500、AMC23、Lean-Workbook、DeepSeek-Prover-v1 数据公开可用。
- **代码**：GitHub 仓库 https://github.com/MaxwellJryao/FANS 已开源。
- **基座模型**：LoT-Solver（翻译器）、DeepSeek-Prover-v1.5/v2、Goedel Prover、Qwen-RM-72B（奖励模型）。
- **训练数据量**：162,181 条 NL-FL 对齐语句（21,967 条来自 DeepSeek-v1，140,214 条来自 Lean-Workbook）。
- **硬件**：NVIDIA H200 用于训练和推理，Lean4 验证在 CPU 上进行。
- **关键超参**：vLLM 推理温度 0.6（答案生成）、0（翻译）、1.0（证明）；最大 token 数 4096（答案）、2048（翻译/证明）。
