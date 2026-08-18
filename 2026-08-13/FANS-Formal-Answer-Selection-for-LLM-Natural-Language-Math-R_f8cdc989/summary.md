---
title: "FANS-Formal-Answer-Selection-for-LLM-Natural-Language-Math-R"
source: https://aclanthology.org/2025.emnlp-main.158.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:28:49"
field: "大语言模型数学推理与形式化验证"
keywords: ["Lean4", "formal verification", "answer selection", "math reasoning", "Long-CoT", "NL-to-FL translation", "test-time scaling"]
innovations: ["首次提出利用Lean4形式化验证增强LLM自然语言数学答案选择的FANS框架", "提出Long-CoT迁移学习方法训练NL-to-FL翻译器（无Long-CoT标注数据训练、推理时激活）", "设计形式化验证与奖励模型/多数投票正交组合的可组合答案选择机制"]
benchmarks: ["MATH500", "AMC23", "Minerva Math", "Olympiad Bench"]
---

# 论文速读：FANS-Formal-Answer-Selection-for-LLM-Natural-Language-Math-R

## 一句话总结
论文提出了 **FANS**（Formal ANswer Selection）框架，首次将 Lean4 形式化数学语言引入 LLM 自然语言数学推理的答案选择阶段，通过将 NL 问答翻译为可机器验证的 Lean4 定理并调用形式化证明器求解，为 LLM 生成的多个候选答案提供比多数投票或奖励模型更可靠的形式化验证基础。

---

## 研究问题与动机

1. **LLM 数学推理的可信度问题**：自然语言（NL）固有的模糊性使得 LLM 生成的数学答案缺乏连贯且可验证的逻辑支撑，引发"LLM 是否真正具备推理能力"的广泛争论。
2. **现有答案选择方法的局限**：多数投票（Majority Vote）过于简单无法充分挖掘 pass@n 潜力；基于奖励模型（ORM）的最佳-N 排序仍是纯 LLM-based 方法，无法利用数学推理的形式化根基，且难以泛化到分布外（OOD）领域/模型。
3. **形式化语言研究的不对称性**：既有工作几乎全部聚焦"用 NL 增强 FL 能力"，鲜有利用 Lean4 等 FL 反向增强 LLM 的 NL 数学推理。
4. **测试时计算扩展的答案甄别需求**：随 test-time scaling 成为提升 LLM 数学性能的主流范式，如何从大量候选输出中准确筛选正确答案成为关键瓶颈。

---

## 核心贡献（创新点）

1. **首创 FL-to-NL 反向增强框架**：提出 FANS，首次系统性地将 Lean4 形式化验证引入 LLM 数学答案选择，提供可计算机验证的答案置信度评估，而非仅依赖奖励模型评分。
2. **Long-CoT 迁移学习训练 NL→FL 翻译器**：提出一种 transfer learning 方法，利用无 Long CoT 标注的 162,181 条 NL-FL 对齐数据训练 Long-CoT 翻译器（LeanTranslator），推理时激活 Long-CoT 能力以实现更忠实翻译——据称是首个此类方法。
3. **可组合的正交答案选择机制**：FANS 可与现有多数投票和 ORM 方法无缝组合，形成"形式化验证 + 统计/奖励"的双重保障，在 MATH500 上较 ORM 基线最高提升 1.91%，在 AMC23 上提升 8.33%。
4. **引入翻译忠实度检查（self/external check）**：针对翻译失真导致的虚假正解问题，设计了通过原始模型自身或更强外部模型（QwQ-32B）验证翻译保真度的额外环节，显著降低误判率。

---

## 方法详解

FANS 框架分为三个协同阶段：

### 阶段一：NL → FL 翻译（LeanTranslator）
- 收集 **Lean-Workbook** 中的 NL-FL 对齐语句（140,214 条）+ 用 Qwen-2.5-72B 将 DeepSeek-Prover-v1 数据集的 FL 定理反译为 NL（21,967 条），共 **162,181 条**无 Long-CoT 标注数据。
- 以 **LoT-Solver** 为底座，采用 transfer learning 策略：**训练时**在 system prompt 中明确指示"不使用 Long CoT"并填充空占位符；**推理时**指示模型使用 Long-CoT 进行逐步分析再输出 Lean4 语句。
- 翻译后引入忠实度检查：用 QwQ-32B（或自身模型）判断翻译是否与原始 NL 语义一致。

### 阶段二：FL 证明生成与验证
- 将翻译得到的 Lean4 定理陈述输入 Lean4 证明器（DeepSeek-Prover-v1.5 / Goedel-Prover / DeepSeek-Prover-v2），使用 few-shot prompt 适配 NL 衍生形式。
- 利用 **Lean4 类型检查系统**（基于归纳构造演算 CIC）对证明进行机械验证，确保每一步逻辑一致性，消除人工直觉推理的不稳定性。
- 使用 Santos et al. (2025) 的 Kimina Lean Server 降低验证开销。

### 阶段三：答案选择与回退策略
- **形式化通过的答案**：直接进入最终选择池。
- **多数投票回退**：若通过验证的答案票数低于阈值，则回退到整体多数投票，缓解翻译错误导致的假阳性影响。
- **ORM 融合**：若 reward model 可用，先比较 ORM 得分识别高难度问题（ORM 在此类问题上往往失效），再借助形式化验证精确定位正确答案。
- 框架设计高度解耦，三种阶段可灵活替换各自子组件。

---

## 实验与结果

### 数据集
- **MATH500**（Hendrycks et al., 2021）：500 道高中数学题，7 个子领域（代数、数论、三角等）
- **AMC23**（Qwen-2.5-Math 配套）：40 道更高难度竞赛题
- **Minerva Math**、**Olympiad Bench**

### 基线模型
Mistral-7B、DeepSeek-Math-7B、Qwen-2.5-Math-1.5B/7B；ORM 统一使用 Qwen-RM-72B。

### 主要结果

| 设置 | MATH500 | MATH-代数 | MATH-数论 | AMC23 |
|------|---------|-----------|-----------|-------|
| Mistral-MV → Mistral-FANS | 33.80→**36.40** (+7.69%) | 42.74→**45.97** (+7.56%) | 33.87→**35.48** (+4.75%) | 12.50→**15.00** (+20.00%) |
| DeepSeek-ORM → +FANS | 62.60→**63.80** (+1.91%) | 82.26→82.26 (—) | 61.29→**66.13** (+7.90%) | 30.00→**32.50** (+8.33%) |
| Qwen-2.5-ORM → +FANS | 80.80→**81.80** (+1.25%) | 96.77→**98.39** (+1.67%) | 91.94→**93.55** (+1.75%) | 70.00→70.00 (—) |

- 使用新翻译器（Wang et al., 2025a）+ SOTA 证明器（Ren et al., 2025）后，**FANS w/ external check** 在 MATH500 上进一步提升。
- **最强结果**：AMC23 上 Mistral-7B + FANS 达到 **15.00%**（相对 MV 基线提升 **20.00%**）；MATH500 数论子领域 DeepSeek-ORM+FANS 提升 **7.90%**。
- 代数/数论领域表现最佳，与 Lean4 现有库支持度高度相关。

---

## 相关工作脉络

1. **Zhou et al. (2024) Don't Trust: Verify**：类似框架但重度依赖 prompt engineering 且基于较弱底座模型，本文通过训练专用 Long-CoT 翻译器并使用先进证明器显著超越。
2. **Lean-Dojo / MA-LoT / DeepSeek-Prover 系列**：聚焦"用 NL 增强 FL 证明能力"，本文反向利用 FL 验证能力增强 NL 推理答案选择，方向互补。
3. **DeepSeek-Math / Qwen-2.5-Math / Llemma**：专注于 NL 数学推理模型本身的能力提升，未触及答案选择的可验证性问题，FANS 作为后处理增强层与之正交可叠加。
4. **Reward Modeling（DPO/KTO/SimPO 等）**：纯 LLM-based 答案排序方案，存在 OOD 泛化难题；FANS 提供不依赖领域分布偏移的形式化判别依据。
5. **Test-time Scaling 工作（rstar-math、s1、Inference Scaling Laws）**：揭示扩大测试时计算可显著提升性能，但未解决"如何从大量候选中选出正确答案"的核心问题，FANS 填补此空白。
6. **LeanTranslator vs. Kimina-Autoformalizer（Wang et al., 2025a）**：本文自研翻译器采用 Long-CoT transfer learning 训练策略，相比直接使用 Kimina-Autoformalizer 在忠实翻译上更具针对性优化。

---

## 局限性与未来方向

1. **翻译假阳性率高**：形式化翻译存在语义偏差风险（如将方程求解误译为重言式、极值问题约束处理不当），导致空洞证明和错误答案选择。
2. **证明器不完整**：即使 SOTA 证明器（Goedel-Prover）在 miniF2F 等基准上仍有约 40% 的定理无法完成证明，限制了整体覆盖。
3. **Lean4 生态领域偏斜**：现有库对代数/数论支持良好，但几何、组合等子领域覆盖不足，制约 FANS 在这些领域的表现。
4. **理论上限差距**：当前性能与 pass@N 的理论上限仍有显著差距，受限于翻译成功率 $p$ 和证明成功率 $q$ 的乘积效应（理论增益约为 $(r_2 - r_1) \cdot pq$）。

**未来方向**：改进 NL-FL 翻译的鲁棒性（尤其约束条件和优化目标的显式建模）；扩展 Lean4 库覆盖更多数学分支；通过迭代微调和改进证明搜索策略增强证明器能力。

---

## 研究启发与可借鉴点

1. **Long-CoT 迁移学习训练范式**：利用无 Long-CoT 标注的对齐数据训练基础能力，推理时通过 prompt 激活 Long-CoT——这一"训练-推理分离"的 transfer learning 策略可迁移至其他 NL↔FL 双向翻译任务。
2. **形式化验证作为可组合的后处理模块**：FANS 与 MV/ORM 完全解耦的设计思路，提示我们可在任何需要多候选答案筛选的场景中嵌入形式化验证层，而不必修改底层生成模型。
3. **翻译忠实度检查机制**：引入外部强模型对翻译结果进行二次校验，是降低 false positive 的有效工程手段，适用于任何 NL↔FL 转换管线。
4. **子领域表现差异的结构化分析**：FANS 在代数/数论上的显著优势揭示了"形式化工具生态成熟度"对方法效果的决定性影响，启发后续工作在选择目标形式化语言时应优先考虑领域库丰富度。
5. **测试时计算扩展的"验证闭环"**：将 test-time scaling 的大量候选输出与形式化验证结合，形成"生成→翻译→证明→验证→选择"的完整流水线，为后续研究提供了可复用的端到端架构模板。

---

## 关键术语表

**FANS**：Formal ANswer Selection，本文提出的基于 Lean4 形式化验证的 LLM 数学答案选择框架。

**Long-CoT（Long Chain-of-Thought）**：扩展版链式思维推理，让模型生成更长的逐步推理过程，本文用于提升 NL→FL 翻译的忠实度。

**Lean4**：最新的 Lean 定理证明器/形式化语言，基于归纳构造演算（CIC），提供机械可验证的数学证明能力。

**ORM（Optimized Reward Model）**：优化后的奖励模型，用于对 LLM 生成的多个候选答案进行评分排序。

**NL-FL 对齐翻译**：将自然语言数学问题与对应形式化语言定理语句配对，本文核心预处理步骤。

**Pass@N**：在 N 次独立生成中至少有一次正确的准确率，衡量测试时计算扩展效果的标准指标。

**Minority Vote（MV）**：多数投票法，取 N 次生成中出现频次最高的答案作为最终输出。

**Transfer Learning for Long-CoT**：本文提出的训练策略，在无 Long-CoT 标注数据上训练基础模型，推理时激活 Long-CoT 能力。

---

## 可复现要素

- **数据集**：MATH500（公开）、AMC23（来自 Qwen-2.5-Math 仓库）、Lean-Workbook（公开）、DeepSeek-Prover-v1 数据集（公开）；训练数据共 162,181 条 NL-FL 对齐语句。
- **代码/权重**：代码已开源于 https://github.com/MaxwellJryao/FANS；LeanTranslator 基于 LoT-Solver 微调。
- **关键超参**：vLLM 推理温度 0.6，max tokens 4096；翻译阶段温度 0，max new tokens 2048；证明器温度 1.0，max new tokens 2048。
- **硬件**：训练/推理使用 NVIDIA H200；Lean4 验证在 CPU 上进行。
- **证明器选项**：DeepSeek-Prover-v1.5、Goedel-Prover、DeepSeek-Prover-v2；验证器为 Kimina Lean Server（Santos et al., 2025）。

---
