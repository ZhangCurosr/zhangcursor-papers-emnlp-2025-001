---
title: "CodeMixBench-Evaluating-Code-Mixing-Capabilities-of-LLMs-Acr"
source: https://aclanthology.org/2025.emnlp-main.109.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:45:11"
field: "多语言大语言模型评测"
keywords: ["code-mixing", "multilingual LLM", "benchmark", "synthetic data generation", "few-shot learning", "cross-lingual"]
innovations: ["首个覆盖18语言8任务的代码混合LLM综合评测基准CodeMixBench", "首次将词汇替换与GPT-4提示结合的大规模合成数据管线", "词级指标+语义相似度+LLM对齐的多级自动化数据过滤机制"]
benchmarks: ["CodeMixBench", "LinCE", "GLUECoS", "CM-MMLU", "CM-GSM8K", "CM-TruthfulQA"]
---

# 论文速读：CodeMixBench-Evaluating-Code-Mixing-Capabilities-of-LLMs-Acr

## 一句话总结
本文提出了 CodeMixBench，首个涵盖 18 种语言、8 类任务的多语言代码混合（code-mixing）大语言模型评测基准，并设计了一种结合词汇替换与 GPT-4 提示的大规模合成数据生成方法，揭示了跨语系代码混合对 LLM 性能的显著负面影响。

## 研究问题与动机
- **代码混合现象重要但评测缺失**：多语言用户在真实场景中频繁交替使用多种语言，但现有评测基准难以充分评估 LLM 在此场景下的能力。
- **既有基准覆盖不足**：LinCE（4 语言对、5 传统 NLP 任务）和 GLUECoS（2 语言对、6 传统任务）语言覆盖有限且不含面向 LLM 的推理任务。
- **合成数据质量低下**：既有代码混合数据生成方法依赖词对齐和句法解析工具，或受限于数据集规模与语言多样性，LLM 生成尝试也未能充分利用指令遵循能力。
- **LLM 在代码混合上表现存疑**：已有研究表明 LLM 在代码混合任务上表现弱于小规模微调模型，但缺乏系统性跨语言、跨任务评估。

## 核心贡献（创新点）
- **首个全面的多语言代码混合 LLM 评测基准**：整合 22 个合成数据集与 30 个开源数据集，覆盖 8 任务、18 语言、7 语系；不同于 LinCE/GLUECoS 仅关注传统 NLP 任务，本文引入知识推理、数学推理、真实性等 LLM 专属评测维度。
- **首次将词汇替换与 LLM 提示相结合的大规模合成管线**：基于平行语料库，利用 GPT-4 Turbo 在表层结构对齐处进行词汇替换生成代码混合文本，避免直接生成导致的质量与多样性问题。
- **多层级质量过滤机制**：结合 M-index/I-index 词级指标、LaBSE 语义相似度过滤（阈值 0.8）与 GPT-4 对齐过滤（连贯性/自然性/可读性评分），保障合成数据质量。
- **系统性的多模型代码混合能力评估**：评测 GPT、LLaMA、Mistral 三大家族共 13 个模型在 11 种代码混合语言对上的表现，揭示跨语系代码混合对模型性能的普遍负面影响。

## 方法详解
**合成数据管线（三阶段）**：

1. **平行语料库构建**：从 Opaki 获取 12 语言 MMLU 平行题库（4,018 题），用 GPT-4 Turbo 将 GSM8K 和 TruthfulQA 翻译为 zh/es/hi/ar，并从 TED2013 提取 4,344 样本的英/中/西/阿平行 MT 语料。

2. **指令合成**：基于 POPLACK（1980）等价约束理论，提示 GPT-4 Turbo 在平行句表层结构对齐处随机替换词汇/短语，生成符合语法连贯性的代码混合文本，并返回替换对照。

3. **三级过滤**：
   - **词级过滤**：计算 M-index（语言混合均衡度）和 I-index（语言切换频率），阈值均设为 0.1 剔除单语或混合不足样本。
   - **语义过滤**：用 LaBSE 计算合成文本与原两种语言文本的余弦相似度，低于 0.8 则剔除。
   - **模型对齐过滤**：用 GPT-4 Turbo 从连贯性（coherence）、自然性（naturalness）、可读性（readability）三方面 1-3 分打分，任意维度得 1 分则剔除。

**评测设置**：CM-MMLU/CM-TruthfulQA 采用多选型准确率；CM-GSM8K 采用 CoT + 正则提取最终答案；LID/POS/NER/SA 要求 JSON 格式输出并计准确率；MT 计 BLEU 分数。所有评测为一少样本（one-shot）设定，额外进行 k-shot（k ∈ {0,1,2,5}）分析。

## 实验与结果
- **数据集**：合成 22 个数据集（CM-MMLU 11 对/12,156 题，CM-GSM8K 4 对/4,367 题，CM-TruthfulQA 4 对/3,122 题，MT 3 对/2,711 句），收集 30 个开源数据集。合成数据加权平均 M-index=0.81，I-index=0.25，LaBSE 相似度均 >0.91，GPT 对齐过滤合格率约 91.4%。

- **基线模型**：GPT-3.5-Turbo-Instruct、GPT-3.5-Turbo、GPT-4-Turbo、GPT-4o；LLaMA2-Chat（7B/13B/70B）、LLaMA3-Base/Instruct（8B/70B）；Mistral 7B、Mixtral 8x7B、Mixtral 8x22B。

- **主要结果**：
  - **模型规模效应显著**：GPT-4o 在 CM-MMLU 各语言对上均最高（en-only 85.6%，zh-en 82.97%，es-en 86.30%）；LLaMA3-8B-Base 较 LLaMA2-7B-Chat 在 CM-MMLU 平均提升 25.29 分，在 CM-GSM8K 提升 57.29 分。
  - **跨语系代码混合损害性能**：同一语系混合（如 es-en、fr-en、de-en、nl-en）性能接近 en-only；跨语系混合（zh-en、hi-en、ar-en、ta-en）准确率显著下降，低资源语言（mr-en、ne-en、ta-en）降幅达 7.62-14.83 分。
  - **推理类型差异**：Few-shot 学习对知识推理和真实性推理有效，但对数学推理（CM-GSM8K）增益有限，部分模型甚至因 k 增大而下降。
  - **GPT 系列内部差异**：GPT-3.5-Turbo 较 Instruct 版在多数任务上优 2-14 分；LLaMA3-Instruct 较 Base 在 CM-GSM8K 上高 7.73 分。

## 相关工作脉络
- **LinCE / GLUECoS**：早期代码混合基准，仅覆盖 4/2 语言对与 5/6 传统 NLP 任务，缺乏 LLM 专属推理任务评估。
- **基于句法解析的合成方法（Bhat et al., 2016; Pratapa et al., 2018）**：依赖等价约束理论与词对齐/句法解析工具，质量受工具性能制约。
- **神经网络生成方法（GAN/VAE/Seq2Seq+Pointer）**：受限于数据集规模与语言多样性。
- **Yong et al. (2023) 直接 LLM 生成**：尝试直接用 LLM 生成东南亚语言代码混合文本，但未充分利用指令遵循能力，且多样性有限。
- **Zhang et al. (2023)**：指出 multilingual LLM 在代码混合任务上表现弱于小规模微调模型，但未提供系统性跨语言基准。

## 局限性与未来方向
- **合成数据质量仍存隐患**：尽管经过多级过滤，22 个合成数据集和 30 个开源数据集各自存在潜在质量问题，18 种语言的一致性质量控制难度较大。
- **未涉及偏差与公平性**：代码混合场景下潜在的文化/语言偏差尚未评估与缓解。
- **语言覆盖仍有局限**：18 种语言虽覆盖 7 语系，但相对于全球语言多样性仍显不足，尤其低资源语言样本量偏少。
- **未来方向**：扩展至更多语言对、引入人工标注验证、开发针对代码混合场景的微调方法、研究模型偏差缓解策略。

## 研究启发与可借鉴点
- **基于平行语料+LLM 提示的合成范式**：利用平行文本的结构对齐性指导词汇替换，结合 LLM 的指令遵循能力生成高质量代码混合数据，可迁移至其他低资源多语言场景的数据增强。
- **多级自动化过滤机制**：词级语言学指标（M-index/I-index）+ 语义相似度 + LLM 对齐评分的组合策略，为合成数据质量控制提供了可复用的评估框架。
- **跨语系 vs 同语系对比分析**：通过 WALS 词序特征与语系归属解释性能差异，为理解多语言 LLM 的语言泛化规律提供了分析维度。
- **k-shot 跨任务对比**：在知识推理、数学推理、真实性三类任务上分别评估 few-shot 效果，揭示了不同推理类型对示例数量的敏感性差异，值得在后续评测中借鉴。
- **与团队方向的结合点**：可借鉴其合成管线用于多语言代码生成、跨语言指令遵循等方向的数据构建；其 LID/POS/NER 的 JSON 输出评测方式可用于多语言序列标注任务标准化。

## 关键术语表
**Code-mixing / Code-switching**：同一对话或话语中交替使用两种或多种语言的语言现象。
**M-index（Multilingual Index）**：衡量文本中语言混合均衡度的指标，值域 [0,1]，0 为单语，1 为各类语言等比例混合。
**I-index（Probability of Switching）**：衡量相邻 token 语言切换频率的指标，值域 [0,1]，反映句内语言交替密度。
**POPLACK 等价约束理论**：认为代码混合发生在两种语言句法结构对齐的位置，是本文合成方法的理论依据。
**LaBSE（Language-agnostic BERT Sentence Embedding）**：支持 109 种语言的句子级 BERT 嵌入模型，用于计算跨语言语义相似度。
**Chain-of-Thought (CoT)**：引导模型逐步推理的提示技术，本文用于数学推理任务（CM-GSM8K）。
**MoE（Mixture of Experts）**：专家混合架构，Mixtral 系列模型采用此架构实现参数扩展。
**One-shot / k-shot**：提示中提供 1 个 / k 个示例的少样本学习设定。

## 可复现要素
- **数据集**：CodeMixBench 22 个合成数据集 + 30 个收集数据集，代码和数据已公开（https://github.com/Jeromeyluck/CodeMixBench）。
- **代码**：开源，含合成管线、过滤脚本、评测 prompt 模板。
- **模型权重**：评测使用开源 LLaMA/Mistral 系列及 API 访问的 GPT 系列，非全部本地可复现。
- **关键超参**：GPT 解码 top-p=0.95、temperature=0.8；LLaMA/Mistral 采用 greedy decoding；M-index/I-index 过滤阈值均为 0.1；LaBSE 相似度阈值 0.8；GPT 对齐评分 1 分剔除。
- **合成成本**：约 $718.45。
