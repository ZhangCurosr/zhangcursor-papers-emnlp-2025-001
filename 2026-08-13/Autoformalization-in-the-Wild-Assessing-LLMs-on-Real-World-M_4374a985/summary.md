---
title: "Autoformalization-in-the-Wild-Assessing-LLMs-on-Real-World-M"
source: https://aclanthology.org/2025.emnlp-main.90.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:41:54"
field: "自动形式化与形式化验证"
keywords: ["autoformalization", "large language models", "formal verification", "Isabelle/HOL", "Lean4", "mathematical definitions"]
innovations: ["提出Def_Wiki和Def_ArXiv两个真实世界数学定义数据集，评估LLM在复杂定义形式化上的能力", "设计分类细化（CR）策略，通过结构化指令针对性降低语法、未定义和类型错误", "提出形式化定义接地（FDG）方法，通过Post-FDG大幅降低未定义错误率"]
benchmarks: ["miniF2F", "Def_Wiki", "Def_ArXiv"]
---

# 论文速读：Autoformalization-in-the-Wild-Assessing-LLMs-on-Real-World-M

## 一句话总结
本文针对LLM自动形式化（autoformalization）任务，引入两个来自Wikipedia和arXiv的真实世界数学定义数据集（Def_Wiki和Def_ArXiv），系统评估多个LLM向Isabelle/HOL和Lean4形式化复杂数学定义的潜力与局限，并提出结构化细化与形式化定义接地等改进策略。

## 研究问题与动机
- **现有基准过于简单**：已有autoformalization基准（如miniF2F）主要包含基础算术运算和简单数学对象，难以反映真实场景中的复杂性。
- **定义形式化具有独特挑战**：数学定义是数学话语的核心基础构件，但通常复杂、抽象且依赖上下文前提，对LLM的挑战不同于常规定理证明。
- **LLM自我纠错能力有限**：现有工作未充分探索LLM在获取证明助手反馈后的 refine 能力，尤其是结构化细化的作用。
- **形式化库接地机制不明确**：如何将自然语言定义中的数学对象与正式库中的形式化定义有效关联，仍是开放问题。

## 核心贡献（创新点）
1. **提出两个真实世界数学定义数据集**：Def_Wiki（56条）和Def_ArXiv（30条），聚焦机器学习领域的定义，比miniF2F更具复杂性和多样性。
2. **系统性误差分析与干预策略**：识别SYN（语法错误）、UDF（未定义项错误）、TUF（类型统一失败）三大错误类别，并设计对应的结构化细化（CR）方法。
3. **形式化定义接地（FDG）策略**：通过Post-FDG将外部正式库的上下文信息作为辅助前提，显著降低未定义错误率（最高降幅43%）。
4. **跨形式化语言验证**：将方法推广至Lean4，证明分类细化（CR）具有跨形式化系统的泛化能力。
5. **小规模但高信息量**的数据集设计验证：虽规模较小，但能充分暴露真实场景下的核心挑战。

## 方法详解
- **自动形式化任务定义**：将非形式化数学语句 $s$ 映射到形式化语言 $\mathcal{F}$，即 $f(s) = \mathrm{LLM}(p_{\text{auto}}, \{(s_i, \phi_i)\}, s)$。
- **零样本提示（ZS）**：直接提示LLM生成Isabelle/HOL代码，不进行任何反馈或修正。
- **二元细化（Binary Refinement）**：提供零样本生成的代码及证明助手的二元正确性反馈（correct/incorrect），让LLM尝试自我修正。
- **详细细化（Detailed Refinement）**：提供证明助手的详细错误信息（类型、消息、位置），辅助LLM修正。
- **分类细化（CR）**：针对特定错误类别（SYN/UDF/TUF）设计结构性指令约束，例如：
  - **SYN-CR**：确保所有Isabelle符号完整（如以`<`开头必须以`>`结尾），避免使用保留字。
  - **UDF-CR**：确保代码中提及的每一项都有明确引用。
  - **TUF-CR**：确保操作数类型与定义中的类型完全匹配。
- **符号细化（SR）**：基于规则的后处理方法，修复Isabelle符号格式错误（如补全`>`）和替换不存在的数学字体符号。
- **形式化定义接地（FDG）**：从Isabelle/HOL库中提取相关形式化定义，通过Post-FDG方式替换/增强LLM生成的preamble，或通过Prompt-FDG方式将外部定义作为上下文提示。

## 实验与结果
- **数据集**：
  - miniF2F-Test：244个样本
  - Def_Wiki-Test：46个样本（开发集10个）
  - Def_ArXiv：30个样本
- **评估模型**：DeepSeekMath-7B、Llama3-8B、GPT-4o
- **评估指标**：Pass（通过证明助手检查的比例）、FEO（首次错误发生位置）、TRO（运行超时）、IVI（无效输入）、SYN/UDF/TUF各类错误率
- **关键结果**：
  - **真实定义挑战更大**：GPT-4o在Def_Wiki和Def_ArXiv上的Pass率比miniF2F平均低13.78%，FEO平均低31.90%。
  - **专业小模型表现优异**：DeepSeekMath-7B在多个数据集上达到与GPT-4o相近的Pass率。
  - **二元自我纠错效果有限**：GPT-4o在二元细化后略有提升，但DeepSeekMath-7B和Llama3-8B性能下降。
  - **详细细化优于二元细化**：在miniF2F和Def_Wiki上，详细细化修正更多错误。
  - **分类细化最有效**：SYN-CR使GPT-4o在Def_Wiki上的Pass率从10.87%提升至36.96%（+26.09%），UDF错误率从50%降至17.39%。
  - **Post-FDG显著减少UDF错误**：在Def_ArXiv上，UDF错误率从56.66%降至13.33%（降幅43%）。
  - **跨语言泛化**：CR方法在Lean4上同样有效，Def_ArXiv在Lean4上的Pass率从0%提升至6.67%。

## 相关工作脉络
1. **Autoformalization with LLMs**：Wu et al. (2022) 首次探索LLM在Isabelle上的自动形式化；本文扩展至真实世界定义并系统分析误差。
2. **MiniF2F基准**：Zheng et al. (2022) 提出的奥林匹克级数学问题基准；本文指出其过于简单，无法反映真实场景。
3. **ProofNet**：Azerbayev et al. (2023) 的本科级数学自动形式化基准；本文强调定义形式化的独特挑战。
4. **Lean4形式化**：Yang et al. (2023) 探索基于检索增强的Lean4定理证明；本文验证CR在Lean4上的泛化。
5. **符号等价自动形式化**：Li et al. (2024) 提出基于符号等价和语义一致性的方法；本文关注结构化细化与接地机制。
6. **数据增强方法**：Jiang et al. (2024) 探索多语言多样性对自动形式化的帮助；本文强调高质量定义数据的价值。

## 局限性与未来方向
- **数据集规模较小**：Def_Wiki（56条）和Def_ArXiv（30条）样本有限，可能限制结论的普遍性。
- **误差分析局限于Isabelle/HOL**：部分结果可能无法直接推广至其他证明助手（如Coq、Agda）。
- **语义一致性未充分评估**：当前评估主要关注句法正确性，语义对齐仍需开放研究。
- **Prompt-FDG效果不佳**：直接将外部定义放入提示中反而降低性能，说明LLM尚难有效利用上下文形式化定义。
- **未来方向**：扩展至更多科学领域、开发语义一致性评估方法、研究LLM与形式化库的有效交互机制。

## 研究启发与可借鉴点
1. **误差驱动的方法设计**：先进行细致的误差分析（SYN/UDF/TUF），再针对性设计干预策略，而非盲目尝试通用优化。
2. **结构化细化优于简单反馈**：提供详细错误信息+分类指令的效果显著优于仅二分正确性反馈。
3. **形式化接地（Grounding）的实用价值**：Post-FDG通过替换preamble即可大幅降低UDF错误，成本低且效果显著。
4. **跨形式化语言的通用性验证**：在Isabelle上验证的方法需进一步在Lean4等系统上检验，增强结论的稳健性。
5. **小规模高质量数据集的设计思路**：即使样本有限，只要覆盖复杂、真实的场景，就能有效暴露模型局限。

## 关键术语表
- **Autoformalization**：将非形式化数学语句（自然语言+LaTeX）自动转换为形式化语言（如Isabelle/HOL、Lean4）的任务。
- **Isabelle/HOL**：基于高阶逻辑的证明助手，广泛用于形式化数学验证。
- **Lean4**：新一代交互式定理证明器，基于依赖类型论，近年来在数学形式化领域广泛应用。
- **Categorical Refinement (CR)**：针对特定错误类别（语法、未定义、类型）设计结构化指令，引导LLM修正形式化代码。
- **Formal Definition Grounding (FDG)**：将外部形式化库中与自然语言定义相关的形式化定义作为上下文，辅助LLM生成正确的形式化代码。
- **Post-FDG**：通过后处理方式替换LLM生成的preamble，引入正确的库导入声明。
- **Symbolic Refinement (SR)**：基于规则的符号级后处理方法，修复Isabelle符号格式错误。
- **Undefined Item Error (UDF)**：形式化代码中引用的项在局部上下文或导入库中未定义导致的错误。

## 可复现要素
- **数据集**：Def_Wiki（56条）、Def_ArXiv（30条）——论文声明将发布，但未提供具体链接；miniF2F-Test为公开基准。
- **代码**：论文未明确声明代码开源，但提供了详细的方法和提示模板（见附录）。
- **关键超参**：贪心解码（greedy decoding）用于所有实验设置。
- **评估工具**：Isabelle/HOL证明助手用于句法正确性验证。
