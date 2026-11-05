---
title: "CodeMixBench-Evaluating-Code-Mixing-Capabilities-of-LLMs-Acr"
source: https://aclanthology.org/2025.emnlp-main.109.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:49:55"
field: "多语言大语言模型评估"
keywords: ["code-mixing", "multilingual LLM", "benchmark", "synthetic data generation", "low-resource languages", "equivalence constraint"]
innovations: ["首个覆盖18种语言8个任务的LLM code-mixing综合基准CodeMixBench", "基于EC theory与GPT-4词替换指令的大规模合成pipeline", "三层过滤框架（词级/语义级/模型对齐级）确保合成数据质量"]
benchmarks: ["CodeMixBench", "LinCE", "GLUECoS"]
---

# 论文速读：CodeMixBench-Evaluating-Code-Mixing-Capabilities-of-LLMs-Acr

## 一句话总结
本文提出了 CodeMixBench，一个覆盖 18 种语言、8 个任务的综合性 code-mixing 评测基准，并设计了一种基于 GPT-4 的词替换合成方法来大规模生成多语混合文本；评估结果显示 LLM 在跨语系 code-mixing 场景下普遍存在性能下降，但增大模型规模、增加训练数据量和引入 few-shot 学习可显著改善表现。

## 研究问题与动机
- **现有 benchmark 覆盖不足**：LinCE 仅覆盖 4 个语言对和 5 个传统 NLP 任务，GLUECoS 仅 2 个语言对且缺少 MT 任务，无法全面评估 LLM 的 code-mixing 能力。
- **LLM 在 code-mixing 上表现不佳**：Zhang et al. (2023) 指出，尽管 LLM 在单语任务上表现强劲，但在 code-mixed 任务上往往不如小规模微调模型。
- **现有合成方法质量受限**：基于语言学规则的方法（如 EC theory）依赖词对齐和句法解析工具的准确性；基于神经网络的数据驱动方法受限于数据集规模和语言多样性；早期使用 LLM 直接生成 code-mixed 文本的方式未充分利用其指令遵循能力。
- **低资源语言被忽视**：现有 benchmark 几乎不涵盖中低资源语言（如 Marathi、Nepali、Tamil、Malayalam 等），而这些语言在多语种用户群体中具有重要实际应用价值。

## 核心贡献（创新点）
1. **首个面向 LLM 的多语言 code-mixing 综合基准**：CodeMixBench 覆盖 8 个任务（3 个 LLM 专属任务 + 5 个传统 NLP 任务）和 18 种语言、7 个语系，远超 LinCE 和 GLUECoS 的覆盖范围。
2. **新型大规模合成 pipeline**：首次将词替换机制与 GPT-4 指令提示相结合，基于平行语料库生成大规模、语法连贯的多语言 code-mixed 数据，解决了现有合成方法依赖解析工具且语言多样性不足的缺陷。
3. **系统的多维度过滤框架**：设计了词级（M-index、I-index）、语义级（LaBSE 余弦相似度）和模型对齐级（GPT-4 评估自然度/连贯性/可读性）三层过滤，确保合成数据质量，相较以往工作仅依赖单一指标更为全面。
4. **全面的 LLM code-mixing 能力发现**：揭示了跨语系混合对 LLM 性能的显著负面影响、模型规模/训练数据量的正向效应、以及 few-shot 学习在不同任务上的差异化效果，为后续研究提供了系统性基线。

## 方法详解
**合成数据 Pipeline（三阶段）**：

1. **平行语料构建**：从 Opaki 获取 MMLU 的 12 语种平行测试集（4,018 道多选题）；使用 GPT-4 Turbo 将 GSM8K 和 TruthfulQA 翻译成 zh/es/hi/ar，分别获得 1,319 和 817 条平行样本；从 TED2013 提取 en/zh/es/ar 平行翻译语料 4,344 条用于 MT 任务。

2. **基于 EC theory 的指令合成**：借鉴 Poplack (1980) 的 Equivalence Constraint 理论——code-mixing 发生在两种语言语法结构对齐的位置。将平行句对中结构对应的词/短语随机替换，生成语法连贯的 code-mixed 句子。Prompt 要求 GPT-4 Turbo 输出替换后的混合句及替换对应关系（格式：`<<<word/phrase→word/phrase>>>`）。

3. **三级过滤**：
   - **词级过滤**：计算 M-index = (1 - Σp_j²)/((k-1)Σp_j²) 和 I-index = ΣS(i,i+1)/(n-1)，阈值设为 0.1，过滤单语或混合频率过低的样本。
   - **语义过滤**：使用 LaBSE 计算 CM-L1、CM-L2、L1-L2 三者间的余弦相似度，低于 0.8 的样本被剔除。
   - **模型对齐过滤**：用 GPT-4 Turbo 从连贯性(coherence)、自然度(naturalness)、可读性(readability)三个维度评分（1-3分），任一维度得分为 1 即剔除。人工评估与 GPT 评分的平均一致率达 91.4%。

**实验设置**：所有实验在 one-shot 下进行，LLM 评测涵盖 GPT 系列（GPT-3.5-Turbo-Instruct、GPT-3.5-Turbo、GPT-4-Turbo、GPT-4o）、LLaMA 系列（LLaMA2-Chat 7B/13B/70B、LLaMA3-Base/Instruct 8B/70B）和 Mistral 系列（Mistral 7B、Mixtral 8x7B、Mixtral 8x22B）。

## 实验与结果
**数据集**：合成 22 个数据集（CM-MMLU 11 对、CM-GSM8K 4 对、CM-TruthfulQA 4 对、MT 3 对），整合 30 个开源 code-mixed 数据集；总成本 \$718.45。

**核心结果（one-shot accuracy）**：
- **CM-MMLU**：GPT-4o 最优（en-only 85.60%，各语言对平均 80.97%）；LLaMA3-70B-Instruct 第二（72.79%）；LLaMA2-70B-Chat 最低（47.14%）。
- **CM-GSM8K**：GPT-4-Turbo 最优（en-only 95.23%，平均 93.42%）；LLaMA3-70B-Instruct 第二（90.61%）；LLaMA2-7B-Chat 最低（18.47%）。
- **CM-TruthfulQA**：GPT-4-Turbo 最优（en-only 83.72%，平均 80.49%）；LLaMA3-70B-Instruct 第二（67.41%）。
- **传统 NLP 任务**（GPT 系列，Appendix I）：GPT-4o 在 LID 平均 83.13%、NER F1 平均 91.61%、MT BLEU 平均 50.86；SA 任务各模型表现最差，平均准确率 54.71%-60.06%。

**关键发现**：
- **跨语系混合损害显著**：zh-en 下降 2.63 分，hi-en 下降 3.47 分，bn-en 下降 7.32 分，ta-en 下降最多达 14.83 分；而 es-en/fr-en/de-en/nl-en 与 en-only 接近（同语系优势）。
- **模型规模效应**：LLaMA3-8B-Base vs LLaMA2-7B-Chat 在 CM-MMLU 上提升 25.29 分；GPT-3.5-Turbo vs GPT-3.5-Turbo-Instruct 在 CM-GSM8K 上提升 14.54 分。
- **Few-shot 效果**：高级模型（GPT-4o、LLaMA3-70B）在知识推理和真实性任务上随 k 增大持续提升；但 CM-GSM8K 数学推理任务 few-shot 增益有限，Mixtral-8x22B 甚至在 k 增大时出现性能下降。

## 相关工作脉络
- **LinCE (Aguilar et al., 2020)**：覆盖 4 个语言对、5 个传统 NLP 任务的 benchmark，本文在语言对数量（16 对）、任务覆盖（新增 3 个 LLM 专属任务）和语言多样性（18 种语言 vs 4 种）上全面超越。
- **GLUECoS (Khanuja et al., 2020b)**：仅 2 个语言对且无 MT 任务，本文填补了其对 LLM 评估能力的空白。
- **Zhang et al. (2023)**：首次指出多语 LLM 在 code-switching 上下文中表现逊于小规模微调模型，本文在其基础上提供了更系统、更全面的 benchmark 和量化分析。
- **Bhat et al. (2016); Pratapa et al. (2018)**：基于 EC theory 和词对齐/句法解析工具的早期合成方法，本文继承其理论但用 GPT-4 指令替代了易出错的解析工具链。
- **Yong et al. (2023)**：尝试用 LLM 直接生成东南亚语言 code-mixed 文本，但未充分利用指令遵循能力；本文通过将 EC theory 嵌入 prompt 中的词替换指令，实现了更高语法质量的合成。
- **Winata et al. (2019)**：使用序列到序列模型从平行句生成 code-mixed 数据，受限于数据集规模和语言多样性；本文方法可扩展至更多语言对。

## 局限性与未来方向
- **合成数据质量不完美**：尽管有三层过滤，22 个合成数据集和 30 个开源数据集仍可能存在质量问题，18 种语言的质量控制一致性难以完全保证。
- **未充分评估 bias**：论文承认未深入评估 code-mixing 场景下的潜在偏见（如语言歧视、资源不平等），引用 Ravfogel et al. (2020) 和 Peng et al. (2025) 指出这是未来方向。
- **LLaMA2 的 few-shot 退步**：LLaMA2 在 code-mixed 场景下一到两样本时性能显著低于 zero-shot，揭示了特定模型架构对混合语言的脆弱性，需进一步研究。
- **低资源语言数据稀缺**：Marathi、Nepali、Tamil、Malayalam 等低资源语言的评测数据规模有限，限制了在这些语言上的结论可靠性。

## 研究启发与可借鉴点
1. **EC theory + LLM prompt 的合成范式**：将语言学理论（Equivalence Constraint）转化为 LLM 指令中的结构化替换规则，而非让 LLM 自由生成，可有效兼顾语法正确性和语言多样性；此思路可迁移至其他低资源语言的语料合成。
2. **三级质量过滤框架**：词级统计指标（M/I-index）+ 语义相似度 + 模型对齐人工评分的组合，为 NLP 数据合成提供了可复用的质量控制模板。
3. **跨语系 vs 同语系的性能差异分析**：本文通过对照实验揭示了语言家族相似度对 code-mixing 性能的关键影响，建议后续研究将此作为 baseline 对比维度。
4. **K-shot 在不同任务类型上的差异化分析**：发现 few-shot 在数学推理任务上增益有限且在某些模型上产生负面效应，提示 future work 需区分任务复杂度来选择 prompt 策略。
5. **与多语言代码生成的结合机会**：Chai et al. (2023a) 的 ERNIE-code 和 Peng et al. (2024) 的 HumanEval-XL 均涉及多语言代码生成，CodeMixBench 的 code-mixing 评测框架可直接应用于评估 LLM 在多语言编程场景下的混合语言能力。

## 关键术语表
**Code-mixing**：在单一话语或对话中交替使用两种及以上语言的语言现象，与 code-switching 同义。
**Equivalence Constraint (EC theory)**：Poplack (1980) 提出的语言学理论，规定 code-mixing 只能发生在两种语言语法结构对齐的位置。
**M-index**：衡量文本中语言比例平衡度的指标，值域 [0,1]，0 为单语，1 为语言比例完全均衡。
**I-index**：衡量句子中语言切换频率的指标，值域 [0,1]，0 为无切换，1 为每个相邻 token 均来自不同语言。
**CM-MMLU / CM-GSM8K / CM-TruthfulQA**：分别为将 MMLU、GSM8K、TruthfulQA 数据集通过 code-mixing 合成的多语言评测子集，用于评估 LLM 的知识推理、数学推理和真实性能力。
**LaBSE**：Language-agnostic BERT Sentence Embedding，一种训练于 109 种语言的句级嵌入模型，用于计算多语文本的语义相似度。

## 可复现要素
- **数据集**：22 个合成数据集 + 30 个整合的开源数据集；代码和数据集已公开于 https://github.com/Jeromeyluck/CodeMixBench。
- **代码/权重**：评测代码开源；模型权重为开源或 API 访问（GPT 系列需 API，LLaMA/Mistral 系列为开源权重）。
- **关键超参**：GPT 系列 top-p=0.95, temperature=0.8；LLaMA/Mistral 系列使用 greedy decoding；M-index 和 I-index 过滤阈值均为 0.1；LaBSE 相似度阈值 0.8；GPT-4 模型对齐评分阈值（任一维度得分=1 则剔除）。
