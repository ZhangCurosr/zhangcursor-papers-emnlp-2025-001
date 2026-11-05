---
title: "On-Relation-Specific-Neurons-in-Large-Language-Models"
source: https://aclanthology.org/2025.emnlp-main.52.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 22:06:40"
field: "大语言模型机制可解释性"
keywords: ["relation-specific neurons", "mechanistic interpretability", "factual knowledge", "neuron cumulativity", "controlled generation", "multilingual generalization"]
innovations: ["首次在LLM中系统识别并验证关系特异性神经元", "揭示神经元的累积性、多功能性与干扰性三大特性", "提出解耦实体与关系影响的统计识别与控制生成验证范式"]
benchmarks: ["LLaMA-2-7B/13B", "Gemma-7B", "LRE dataset", "12 factual relations"]
---

# 论文速读：On Relation-Specific Neurons in Large Language Models

## 一句话总结
该论文首次在 LLM（LLaMA-2 系列）中系统性地识别并验证了**关系特异性神经元（RelSpec neurons）**的存在，证明模型中存在一类独立于实体、专注于抽象关系本身的神经元，并通过控制生成实验揭示了其三大特性：累积性、多功能性与干扰性。

## 研究问题与动机
- **核心问题**：LLM 预训练期间学习的 factual knowledge 通常以三元组（subject, relation, object）形式存在，但模型内部是否存在**专注于关系本身**（而非具体事实或实体）的神经元，此前尚不明确。
- **现有方法不足**：
  1. 早期知识神经元研究（如 Dai et al., 2022）主要定位编码**具体事实**的神经元，缺乏对关系级别抽象表征的探索。
  2. 语言特异性神经元已获验证（Kojima et al., 2024），但**关系级别**的神经元机制研究仍处于空白。
  3. 现有 mechanistic interpretability 工作多关注 attention heads 或单事实神经元，未系统分析关系泛化能力。

## 核心贡献（创新点）
- **首次验证 RelSpec 神经元存在**：通过统计关联方法在 LLaMA-2 中识别出专注于特定关系的神经元，并证明其去激活会导致对应关系事实召回率显著下降。
- **揭示神经元三大特性**：提出并验证了**神经元累积性**（多个神经元协同工作）、**神经元多功能性**（神经元可跨关系/跨语言共享）与**神经元干扰性**（去激活一个关系的神经元反而可能提升其他关系的表现）。
- **解耦实体与关系的影响**：通过构建无实体重叠的评估集，证明识别出的神经元确实关注关系本身而非特定实体。
- **开源代码与数据**：将识别方法与实验数据全部公开，为后续研究提供可复现基础。

## 方法详解
- **数据集处理**：采用 Hernandez et al. (2024) 的 LRE 数据集，筛选事实数>300 的 12 个关系；每个关系随机划分 50 个三元组作为 held-out evaluation set $\mathcal{D}^{\mathrm{eva}}$，其余作为 detection set $\mathcal{D}^{\mathrm{det}}$。
- **提示词构建**：为每个三元组生成仅包含 subject 和 relation 的 zero-shot prompt（如"The CEO of NVIDIA is? Answer:"），不包含 object；通过模型生成前 2 个 token 判断 prompt 是否被正确回答，剔除错误样本。
- **神经元定义**：聚焦 FFN 层（up_proj、gate_proj、down_proj）的列向量作为神经元；将每个 prompt 的神经元输出值按 token 平均得到最终激活值 $\hat{o}_{r_i}^{m,j}$。
- **关系特异性识别**：将每个神经元视为分类器，以正样本（$\mathcal{E}^+$）和负样本（$\mathcal{E}^-$，随机采样 4 倍于正样本的其他关系 prompt）计算**平均精度（Average Precision, AP）**；选取 AP 最高的 top-k 神经元作为该关系的 RelSpec 神经元（实验中 k=3000）。
- **控制生成实验**：在推理时将 identified RelSpec 神经元的输出值强制置为 0，比较去激活前后模型在 $\mathcal{P}^{\mathrm{eva}}$ 上的准确率，分别评估 intra-relation（同一关系）和 inter-relation（不同关系）影响。

## 实验与结果
- **模型与数据集**：LLaMA-2-7B（32层，835,584个FFN神经元）与 LLaMA-2-13B（40层，1,310,720个FFN神经元）；12 个关系类别。
- **关键结果**：
  - **Intra-relation**：去激活 3000 个 RelSpec 神经元后，所有关系的准确率均显著下降，但**未降至零**（除 13B 的 landmark_country 外），证明神经元具有**累积性**。
  - **Inter-relation**：去激活某一关系的神经元不仅影响该关系，还会影响其他关系（包括语义相近和相距较远的关系），体现**多功能性**；部分情况下去激活反而**提升**其他关系准确率（如 7B 中 person_mother 在去激活 5 个其他关系神经元后提升），体现**干扰性**。
  - **跨语言泛化**：基于英语 prompt 识别的 RelSpec 神经元在德、西、法、中、日五种语言上去激活后同样导致准确率下降，证明其**多语言通用性**。
  - **非模板依赖**：更换 prompt template 后去激活效应依然一致，排除句法结构混淆。
  - **一般语言建模未受损**：在 relation-neutral 上下文中计算 perplexity，去激活后未见系统性恶化。
- **最强结果**：去激活 3000 个 RelSpec 神经元在 7B 模型上使多数关系准确率下降 10–30 个百分点，而随机去激活相同数量神经元几乎无影响，证明识别方法的特异性。

## 相关工作脉络
- **知识神经元（Knowledge Neurons）**：Dai et al. (2022) 定位编码具体事实的神经元；本文扩展至**关系级别**的抽象神经元，而非单一事实。
- **语言特异性神经元**：Kojima et al. (2024) 发现语言相关神经元主要位于首尾层；本文 RelSpec 神经元集中于**中间层**，且关注关系语义。
- **Fact Editing / Mechanistic Interpretability**：Meng et al. (2022, 2023) 通过编辑权重修改事实；本文从**神经元级解耦**角度分析关系表征机制。
- **Polysemanticity & Superposition**：Elhage et al. (2022b)、Scherlis et al. (2025) 理论预测神经元多功能性；本文提供实证支持。
- **Multilingual Neuron Sharing**：Stanczak et al. (2022)、Zhang et al. (2024) 研究跨语言神经元；本文验证关系神经元同样具有跨语言泛化能力。

## 局限性与未来方向
- **关系数量有限**：仅覆盖 12 个常见关系，未来需扩展至更广泛、更多样的关系类型。
- **多语言覆盖不足**：仅测试 5 种语言，低资源语言未涉及，未来可探索更多语言的 RelSpec 神经元行为。
- **模型范围限制**：主要基于 LLaMA-2 家族（7B/13B），Gemma-7B 有初步验证；未来需研究更大规模模型及 instruction-tuning 后的变化。
- **事实频率近似误差**：使用 Dolma 语料近似预训练频率，可能与模型实际接触频率存在偏差。
- **未来方向**：探索 RelSpec 神经元在 instruction tuning 后的稳定性；研究关系神经元与概念神经元的交互；开发基于 RelSpec 神经元的精准事实编辑方法。

## 研究启发与可借鉴点
- **解耦实体与关系的实验设计**：通过构造无实体重叠的 prompt 集合，有效分离实体影响与关系影响，该方法可直接迁移至其他知识表示研究。
- **统计关联 + 控制生成的验证范式**：采用 AP 指标识别神经元，再通过强制置零验证因果效应，兼顾可扩展性与因果严谨性。
- **累积性量化方法**：通过逐层增加去激活神经元数量并观察准确率变化曲线，定义并验证"累积性"指标，可推广至其他神经元特性分析。
- **跨语言泛化验证流程**：翻译 prompt 并保持实体对齐，验证同一组神经元的跨语言效应，为多语言 mechanistic interpretability 提供标准流程。
- **干扰性效应的利用潜力**：发现去激活某些神经元反而提升其他关系表现，启示未来可通过"剪枝干扰神经元"优化模型特定能力。

## 关键术语表
- **Relation-Specific Neurons (RelSpec neurons)**：专注于识别和生成特定关系（如 company_ceo、person_mother）而非具体实体或完整事实的神经元。
- **Neuron Cumulativity**：关系知识由多个神经元共同编码，单个神经元无法独立承载完整关系信息，去激活越多神经元准确率下降越显著。
- **Neuron Versatility**：同一组神经元可参与多个相关或无关关系的处理，且能在不同语言间共享。
- **Neuron Interference**：某些神经元在处理目标关系时可能"干扰"其他关系的处理，去激活后反而提升其他关系性能。
- **Controlled Generation**：在推理时将特定神经元输出强制置零，以观察其对模型行为的因果影响。
- **Average Precision (AP)**：将神经元激活值作为分类分数，在正负样本上计算 precision-recall 曲线下面积，用于衡量神经元对特定关系的区分能力。

## 可复现要素
- **数据集**：Labeled Relations in Entities (LRE) 数据集（Hernandez et al., 2024），论文已公开处理方式与 prompt 模板。
- **代码与数据**：已开源至 https://github.com/cisnlp/relation-specific-neurons。
- **模型**：LLaMA-2-7B、LLaMA-2-13B（HuggingFace 官方权重）；附加实验包含 Gemma-7B。
- **关键超参数**：FFN 神经元 top-3000；负样本采样倍数为正样本的 4 倍；最大生成长度为 2 tokens；去激活阈值为 0。
- **评估指标**：准确率（预测 2 tokens 是否为 object 的前缀）；Perplexity（评估一般语言建模能力）。
