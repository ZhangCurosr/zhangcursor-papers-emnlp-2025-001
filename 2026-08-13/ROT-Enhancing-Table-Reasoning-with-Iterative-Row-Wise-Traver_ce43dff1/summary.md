---
title: "ROT-Enhancing-Table-Reasoning-with-Iterative-Row-Wise-Traver"
source: https://aclanthology.org/2025.emnlp-main.29.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:39:09"
---

# 论文速读：ROT-Enhancing-Table-Reasoning-with-Iterative-Row-Wise-Traversals

## 一句话总结
论文提出训练免费的 **Row-of-Thought (ROT)** 方法，通过迭代式逐行遍历表格并结合反射机制，使通用大模型在无需微调的情况下即可超越推理大模型（RLLMs）的 Long CoT 表现，在降低表格幻觉的同时减少推理 token 消耗，并在 WikiTableQuestions 与 TableBench 上取得同等规模 SOTA。

## 研究问题与动机
1. **核心问题**：表格推理任务中，基于长思维链（Long CoT）的推理大模型虽性能领先，但存在训练成本高昂与生成可靠性不足的双重瓶颈。
2. **高成本局限**：Long CoT 能力依赖高质量思维链数据进行 SFT/RL 训练，数据收集与微调开销巨大，难以普惠中等规模模型。
3. **低可靠性局限**：随着推理链不断延长，模型极易丢失表格输入中的关键结构化信息，引发跨行混淆、单元格虚构等严重幻觉。
4. **研究动机**：设计一种无需训练、能强制模型充分对齐表格结构、并支持动态推理扩展的提示框架，以低成本实现高可靠的表格问答。

## 核心贡献（创新点）
1. **提出训练免费的迭代逐行遍历推理框架 ROT**：通过结构化提示驱动模型按行扫描并累积中间结果，无需 SFT/RL 即可将非推理模型推向接近推理模型的表格处理能力。
2. **以结构性行级注意力替代堆砌式长思维链**：强制模型逐行读取并融入反思机制，从根源上切断 Long CoT 因上下文丢失导致的表格内容幻觉路径。
3. **实现推理效率与准确性的双重帕累托改进**：在 WikiTableQuestions 和 TableBench 取得同等规模 SOTA，且相比 Long CoT 推理 token 消耗减少 7%~50%，避免过度思考（Overthinking）带来的算力浪费。

## 方法详解
- **输入表示**：将表格统一转为 Markdown 格式，结合指令 $I$、问题 $Q$、表格 $U$（$M$ 行 $N$ 列）及上下文示例 $D$，模型输出推理链 $R$ 与答案 $A$。
- **逐行遍历（Traversal）**：推理以行为基本单位。第 $i$ 次遍历中，模型对第 $j$ 行执行 $R_{i,j}$ 并产出中间结果 $A_{i,j}$，逐行累加得到本轮结果 $A_i$。该设计将复杂问题分解为细粒度步骤，迫使模型持续关注表格实际内容。
- **迭代与动态终止（Iteration）**：单轮遍历不足以应对多跳问题，模型在完成一轮遍历后可选择延续推理或基于反思发起新一轮遍历。总次数 $T$ 不预设，由模型根据是否已推导至最终答案自主判定终止。
- **提示策略**：non-RLLMs 采用 One-shot 提示（含一个 WalkThrough 示例），RLLMs 采用 Zero-shot 提示（Appendix 证明 Few-shot 会导致 RLLM 性能暴跌）。全程无参数更新，纯 Prompt-driven。

## 实验与结果
- **数据集**：WikiTableQuestions（平坦表 QA）、HiTab（层级结构表）、TableBench（多样化复杂 QA）。
- **评估模型**：non-RLLMs（Llama3.1-8B, Llama3.3-70B, Qwen2.5-7B, Qwen2.5-32B）与 RLLMs（DeepSeek-R1-Distill-Llama/Qwen 系列对应尺寸）。
- **主要结果**：
  - ROT (non-RLLM) 平均超越 Long CoT (RLLM) **4.3%**；ROT 应用于 RLLM 亦平均提升 **2.4%**。
  - **WikiTQ** 达 **78.7**（SOTA，超越 TableMaster 的 78.0）；**TableBench** 达 **44.8**（SOTA，超越 Wu et al. 2024 的 43.9）；**HiTab** 为 **76.6**（与 SS-CoT 的 79.1 相当）。
  - **效率优势**：ROT 正确推理的 token 数少于 Long CoT，错误推理长度也显著缩短；整体推理 token 消耗比 Long CoT 减少 **7%~50%**。
- **关键结论**：ROT 以更低成本实现了更可靠的表格推理，验证了“结构化遍历+反思”替代“堆砌长链”的可行性。

## 相关工作脉络
1. **表格推理微调方法（StructGPT, Chain-of-Table, TableGPT2 等）**：依赖大量表格指令数据或专用架构微调；本文转向训练免费、基于 Prompt 的推理过程控制，避开 SFT/RL 的高数据与算力成本。
2. **Long CoT 推理大模型（DeepSeek-R1, OpenAI o1 等）**：通过超长思维链提升复杂任务表现；本文指出 Long CoT 在表格场景易引发幻觉与过度思考，主张用结构化遍历约束推理发散。
3. **Few-shot In-Context Learning**：Appendix B.1 显示 RLLM 在 Long CoT 下使用 Few-shot 性能暴跌（如 WikiTQ 62.7→45.1）；本文据此确立 RLLM 用 Zero-shot、非推理模型用 One-shot 的差异化策略。
4. **专用表格大模型（TableLlama, TableBench LLM 等）**：针对表格预处理与解析进行架构优化；本文证明同等规模通用模型配合 ROT 可在多数榜单追平甚至超越这些专用模型。
5. **幻觉缓解技术（Context-aware decoding, Self-verification 等）**：常需外挂检索或校验模块；ROT 仅通过强制逐行扫描的提示结构即能内化缓解幻觉，无需额外组件。

## 局限性与未来方向
1. 实验仅限英文单轮问答数据集，未验证多轮表格对话（Multi-turn Table QA）场景。
2. 对层级表（HiTab）的结构适配较弱，未内置层级表头绑定机制，与专门设计层级解析的方法仍有差距。
3. 当表格行数极大时，逐行遍历可能累积超出模型上下文窗口限制，导致提前截断或失败。
4. 未来方向：拓展至多语言与多轮表格交互；结合列/单元格遍历的自适应单元选择；与工具调用或代码执行结合处理超大规模表格。

## 研究启发与可借鉴点
1. **推理长度的结构化缩放**：将 CoT 长度控制从“堆砌 token”转为“受结构约束的遍历次数”，可迁移至文档解析、代码生成等需严格对齐输入的领域。
2. **One-shot vs Zero-shot 的差异化使用**：非推理模型靠示例引导模式识别，推理模型靠零样本激发内生能力，该提示设计原则对后续模型选型与 Prompt 工程具参考价值。
3. **动态终止优于固定步数**：消融实验证明固定遍历次数（=1 或 =3）均劣于动态判定，提示框架应赋予模型自主决策权而非硬性截断。
4. **幻觉归因分析框架**：将错误拆分为 Hallucination、Misunderstanding、Locating 三类并定量统计分布，为后续表格推理工作提供清晰的评测与改进基准。
5. **遍历单元的可解释性选择**：行遍历普适最优，但在含密集层级列头的 HiTab 上列遍历更优，启发未来工作可按数据集结构特征自适应切换遍历粒度。

## 关键术语表
**ROT (Row-of-Thought)**：本文提出的训练免费推理方法，通过迭代式逐行遍历表格并结合反思机制生成答案。
**Long CoT (Long Chain-of-Thought)**：推理大模型依赖的超长思维链推理范式，通过扩展推理步长提升复杂任务表现，但易引发幻觉与成本激增。
**non-RLLM**：未经过专门推理增强（SFT/RL）的通用大语言模型，本文证明配合 ROT 可超越部分
