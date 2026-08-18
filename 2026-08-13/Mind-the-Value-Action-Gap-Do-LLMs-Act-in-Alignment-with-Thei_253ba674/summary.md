---
title: "Mind-the-Value-Action-Gap-Do-LLMs-Act-in-Alignment-with-Thei"
source: https://aclanthology.org/2025.emnlp-main.154.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:30:02"
field: "AI价值观对齐与安全"
keywords: ["价值观对齐", "价值观-行为差距", "大语言模型评估", "跨文化AI公平性", "Schwartz价值观理论"]
innovations: ["首次提出ValueActionLens框架系统评估LLM声明价值观与行为对齐", "构建VIA数据集（14,784样本）覆盖12国家×11社会主题×56价值观", "设计三维度对齐度量体系（F1、Manhattan距离、排序）并揭示跨文化不对齐风险"]
benchmarks: ["VIA数据集", "Schwartz基本价值观理论", "Likert量表评估"]
---

# 论文速读：Mind-the-Value-Action-Gap: Do LLMs Act in Alignment with Their Values?

## 一句话总结
本文提出**ValueActionLens**框架，首次系统评估大语言模型的**声明价值观与其价值知情行为之间的对齐程度**，揭示了LLM普遍存在显著的"价值观-行为差距"（Value-Action Gap），且该差距因文化背景和社会主题而异，可能引发歧视、自主权侵犯等潜在风险。

## 研究问题与动机
- **核心问题**：LLM声称的价值观是否与其在实际情境中的行为保持一致？即LLM是否存在类似于人类的"价值观-行为差距"？
- **现有方法不足**：已有研究多通过Likert量表评估LLM的**声明价值观**（stated values），如ValueBench、GPV等，但忽略了"说什么"与"做什么"之间的潜在差异。
- **理论依据**：借鉴环境心理学中的Value-Action Gap理论（Godin et al., 2005），强调个体声明价值观与实际行为之间的脱节。
- **现实风险**：若LLM的价值观声明与行为不一致，可能导致其在真实应用场景中产生隐性危害（如放大刻板印象、强化偏见算法）。

## 核心贡献（创新点）
1. **首个价值观-行为对齐评估框架**：提出ValueActionLens，首次在跨文化情境下系统量化LLM声明价值观与价值知情行为之间的差距，区别于仅评估声明值的研究。
2. **大规模价值知情行为数据集VIA**：构建包含14,784个样本的数据集，覆盖12个国家×11个社会主题×56个Schwartz价值观，附有人工标注的行为归因与自然语言解释。
3. **多维度对齐度量体系**：设计三个互补指标——对齐率（F1）、对齐距离（Manhattan距离）、对齐排序（按值类型降序排列），实现从比例到细节的完整分析。
4. **揭示跨文化对齐差异与潜在风险**：发现GPT-4o-mini、Claude-Sonnet-4、Llama-3.3-70B在非洲和亚洲语境下的对齐率显著低于北美和欧洲，并系统分类了7,106个不对齐样本的个体/交互/社会层面风险。

## 方法详解
**框架三阶段**：

1. **情境化价值观生成**：基于Schwartz基本价值观理论，构建132个情境（12国家×11社会主题），每个情境配对56个价值观的Agree/Disagree两种倾向，生成14,784个价值知情行为（VIA dataset）。

2. **双任务评估**：
   - **Task 1（声明价值观）**：采用SVS/PVQ风格提示，让LLM对56个价值观进行Likert评分（1-4级：strongly agree到strongly disagree），使用8个提示变体取平均。
   - **Task 2（价值知情行为选择）**：给定情境和价值观，让LLM从VIA数据集中的Agree/Disagree两个行为中选择其一（选项顺序打乱），同样使用8个提示变体。

3. **三指标计算**：
   - **对齐率（Alignment Rate）**：将Task1和Task2的响应二值化（Agree=0, Disagree=1），计算F1分数。
   - **对齐距离（Alignment Distance）**：计算两矩阵间逐元素Manhattan距离（L1 Norm）：$D_{ik} = |\nu_{ik} - a_{ik}|$，可按国家/主题聚合。
   - **对齐排序（Alignment Ranking）**：按距离降序排列56个价值观，识别最大差距的值类型。

**数据生成流程**：采用人机协同Pipeline——先构建8个提示变体→AI专家评估选择最优提示（Cohen's Kappa=0.7073）→跨文化人工标注验证（27名标注者，正确率87.78%，无害性80%）。

## 实验与结果
**数据集**：VIA（Value-Informed Actions），14,784个样本，12国家×11社会主题×56价值观。

**评估模型**：GPT-4o-mini、GPT-4o、GPT-3.5-turbo、Llama-3.3-70B、Gemma-2-9B、Deepseek-r1、Claude-Sonnet-4（temperature=0.2）。

**主要结果**：
- **GPT-4o-mini表现最佳**，整体对齐率F1=0.564；**GPT-3.5-turbo最差**，F1=0.179。
- **跨文化差异显著**：GPT-4o-mini在非洲（Nigeria 0.54, Uganda 0.51）和亚洲（Philippines 0.47）的对齐率低于北美（US 0.67, Canada 0.59）和欧洲（UK 0.65）。
- **跨主题差异**：Health和Leisure主题对齐率较高（GPT-4o-mini: 0.652, 0.644），Religion较低（0.519）。
- **不对齐样本**：共7,106个，占比约15-19%，涵盖歧视（334例）、自主权侵犯（42例）、误导性解释（1例）、极化（75例）等风险。

**最强结果**：GPT-4o-mini在Politics主题（0.594）和Leisure主题（0.644）表现最佳，但整体仍仅0.564，表明LLM价值观-行为对齐仍存在较大提升空间。

## 相关工作脉络
1. **ValueBench**（Ren et al., 2024）：评估LLM价值观取向，但仅关注声明值，未涉及行为对齐。
2. **GPV**（Ye et al., 2025b）：基于生成心理测量学测量人类与AI价值观，但未系统分析价值观与行为的差距。
3. **Schwartz价值观评估方法**：传统心理学工具（SVS/PVQ）用于评估人类价值观，本文首次将其适配为LLM价值观-行为对齐评估框架。
4. **LLM价值观对齐研究**：如ValueCompass（Shen et al., 2024b）关注双向对齐，但未量化价值观声明与行为输出的一致性。
5. **红队测试与安全评估**：如Ganguli et al. (2022)的red-teaming方法，本文补充了价值观层面的系统性对齐评估。
6. **文化偏见研究**：如GLOBE框架、LLM-Globe（Karinshak et al., 2024）评估文化嵌入，本文进一步分析跨文化价值观-行为对齐差异。

## 局限性与未来方向
- **价值观理论覆盖有限**：依赖Schwartz理论，可能遗漏特定文化背景下的 emergent values。
- **二值化简化**：将Likert响应二值化为Agree/Disagree，可能丢失细微的价值表达。
- **静态评估**：仅评估单轮生成，未考虑动态对话或交互式场景中的价值观一致性。
- **未来方向**：扩展到自由生成行为、对话式交互评估；支持其他价值观理论（如World Values Survey）；开发场景敏感的自适应对齐方法。

## 研究启发与可借鉴点
1. **双任务对齐评估设计**：Task1（声明）+ Task2（行为）的分离评估范式，可直接迁移至其他AI价值观/态度研究。
2. **跨文化对齐度量体系**：三指标（F1、距离、排序）的互补使用，为公平性/偏见评估提供多维分析框架。
3. **人机协同数据生成Pipeline**：提示变体构建→专家评估→跨文化标注的质量控制流程，可复用于高价值数据集构建。
4. **风险分类taxonomy**：个体/交互/社会三层风险分类（Table 5）及定义（Table 15），可直接用于LLM危害评估。
5. **解释辅助预测**：Appendix F发现结合 reasoned explanations 可显著提升价值推断准确率（F1从0.795→0.830），为价值观可解释性研究提供新思路。

## 关键术语表
- **Value-Action Gap**：声明价值观与实际行为之间的不一致现象，源自环境心理学。
- **ValueActionLens**：本文提出的评估LLM价值观与行为对齐程度的框架。
- **VIA（Value-Informed Actions）**：包含14,784个价值知情行为的大规模数据集。
- **Schwartz基本价值观理论**：包含56个跨文化普世价值观的心理测量学理论框架。
- **Alignment Rate**：基于F1分数的价值观-行为对齐比例指标。
- **Alignment Distance**：基于Manhattan距离的价值观-行为不对齐程度量化。
- **Theory of Reasoned Action**：Ajzen提出的社会心理学理论，用于解释态度与行为之间的关系。
- **Cross-task Inconsistency**：Task1（声明）与Task2（行为）响应不一致的情况。

## 可复现要素
- **数据集**：VIA dataset（14,784样本），论文声明将开源（"the dataset and code will be released"）。
- **代码**：论文未提供具体代码链接，但声明将在学术用途下开源。
- **模型权重**：评估使用商业API（GPT系列、Claude-Sonnet-4）及开源模型（Llama-3.3-70B、Gemma-2-9B、Deepseek-r1）。
- **关键超参**：temperature=0.2，8个提示变体取平均，Likert 1-4级评分。
- **评估指标**：F1、Manhattan距离（L1 Norm）、Cohen's Kappa（0.7073）。
