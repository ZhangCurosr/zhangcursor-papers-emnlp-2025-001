---
title: "Analyzing-the-Effects-of-Supervised-Fine-Tuning-on-Model-Kno"
source: https://aclanthology.org/2025.emnlp-main.25.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:25:07"
field: "大语言模型微调机制分析"
keywords: ["supervised fine-tuning", "model knowledge", "parameter restoration", "closed-book QA", "KL divergence", "LLaMA"]
innovations: ["揭示SFT数据规模与类别对CBQA性能的意外影响", "发现90%以上参数更新冗余并提出有效恢复策略", "从token和参数双层面对照分析SFT知识变化机制"]
benchmarks: ["ENTITYQUESTIONS", "XSum", "GSM8K"]
---

# 论文速读：Analyzing-the-Effects-of-Supervised-Fine-Tuning-on-Model-Knowledge-from-Token-and-Parameter-Levels

## 一句话总结
论文系统分析了监督微调（SFT）对大语言模型内部知识的影响，发现微调数据规模与类别会引发反常的性能波动（如1920样本优于240样本时性能下降14%），并从token和参数两个层面揭示了高达90%的参数更新是冗余甚至有害的，可通过恢复这些参数有效提升模型知识准确性。

## 研究问题与动机
- **SFT如何改变模型知识**：现有研究多关注微调后输出表现，但对微调过程中知识如何被修改的内在机制缺乏深入理解。
- **数据规模与类别的意外影响**：实践中增加微调数据量并不总能提升性能，甚至可能损害模型原有知识，而微调数据的知识掌握程度（model mastery）如何影响性能尚未被系统揭示。
- **参数更新的必要性**：微调会引入大量参数变化，但这些变化中有多少是真正有助于知识增强的，有多少是冗余或有害的，仍不明确。
- **可干预性**：如果存在冗余更新，是否有简单有效的方法来恢复或避免它们，从而保护或提升模型知识？

## 核心贡献（创新点）
1. **系统揭示了SFT数据规模与类别对CBQA性能的意外影响**：相比已有工作仅关注数据质量或数量，本文首次发现微调数据量增加至1920样本时性能反而比240样本下降14%，且不同知识掌握程度的数据可造成超过12%的性能波动。
2. **从token层面建立了logits分布变化与性能下降的关联**：通过KL散度分析证明，微调导致token预测分布偏离预训练模型的程度与性能下降正相关，尤其是使用低掌握度数据时过度偏移会引发知识灾难性遗忘。
3. **从参数层面提出并验证了参数恢复策略**：发现SFT引入的参数更新高度集中（前1%参数贡献超70%总变化），且高达90%的更新是冗余的；将这些参数恢复至预训练值可在不损害甚至提升训练集和测试集性能的同时保持泛化能力。
4. **提供了跨任务验证与机制解释**：在XSum和GSM8K任务上复现了参数恢复的正向效果，并进一步分析冗余参数在层与模块间的分布规律（集中在早期层和FFN层），为高效微调设计提供直接依据。

## 方法详解
- **数据分类**：基于预训练模型对知识点的掌握程度$R_k^\mathcal{M}$（通过多模板补全成功率计算）将训练数据分为5类（$i=0\sim4$，对应0%、0~25%、25~50%、50~75%、75~100%掌握度），共18,456条位置相关数据。
- **实验设置**：使用LLaMA-2（7B/13B/70B）和LLaMA-3（8B/70B）共5个模型，在ENTITYQUESTIONS数据集上进行SFT（batch size=8，1 epoch，lr=1e-5，AdamW+cosine调度），评估在in-domain（$\mathcal{D}_\text{test}$）和out-of-domain（$\mathcal{D}_\text{testood}$）上的准确率。
- **Token-level分析**：计算微调后与预训练模型首token logits分布的KL散度；为消除常用dummy词干扰，引入logits re-normalization：提取微调模型top-10 logits及其对应预训练logits，分别做softmax后计算$ s_\text{KL}(p\|p') = -\sum_i p_i\log\frac{p_i'}{p_i} $。
- **Parameter-level分析**：定义参数相对变化率$r_i = |s_i - p_i|/|p_i|$，按$r_i$降序排序，逐步将前$k\%$（如1%~60%）更新幅度最大的参数恢复至预训练值，观察性能变化，并记录各比例参数贡献的总更新量占比。

## 实验与结果
- **数据集**：ENTITYQUESTIONS（Wikipedia提取，24个主题），in-domain取10个地理位置主题，out-of-domain取剩余14个主题；训练数据按掌握度分5类（表1）。
- **基线模型**：LLaMA-2-7B/13B/70B、LLaMA-3-8B/70B；对比策略包括原始SFT、混合数据SFT、按掌握度划分数据SFT、LoRA微调等。
- **主要结果**：
  - **规模效应**：LLaMA-3-8B使用240样本微调后性能最优，增至1920样本时$\mathbf{Acc}_\text{test}^\mathcal{M}$下降8.86%（$\mathcal{D}_\text{train-0}^\mathcal{M}$）和13.69%（$\mathcal{D}_\text{train-2}^\mathcal{M}$），全量数据时性能接近最低（图2、表2）。
  - **类别效应**：使用1920样本时，$\mathcal{D}_\text{train-0}^\mathcal{M}$与$\mathcal{D}_\text{train-2}^\mathcal{M}$微调模型的$\mathbf{Acc}_\text{test}^\mathcal{M}$相差12.06%，高掌握度数据在低掌握度测试集上表现次优（表2）。
  - **Token-level**：KL散度随数据量先降后升（图5），且$\mathcal{D}_\text{test-4}^\mathcal{M}$上的性能下降与KL上升强相关（图4），证明过度知识迁移有害。
  - **Parameter-level**：前1%参数贡献超70%总更新（表3）；恢复20%参数即可使所有类别性能提升，如$\mathcal{D}_\text{train-0}^\mathcal{M}$恢复后提升9.85%（表4）；恢复比例越高改善越明显，但过度恢复（如60%）会破坏已学技能。
  - **跨任务验证**：在XSum和GSM8K上，恢复少量参数（如10%）可提升ROUGE分数或数学准确率（表5）。
- **最强结果**：参数恢复策略在LLaMA-3-8B + $\mathcal{D}_\text{train-2}^\mathcal{M}$ + 1920样本设置下达到$\mathbf{Acc}_\text{test}^\mathcal{M}$ 62.21%（表7），较原始SFT（58.80%）提升3.41个百分点，且优于LoRA（57.82%）。

## 相关工作脉络
- **CBQA与知识评估**：Zhang et al. (2024)、Wen et al. (2024) 使用CBQA评估模型事实知识；本文在此基础上进一步分析SFT过程中知识的动态变化机制，而非仅关注静态性能。
- **SFT数据质量与规模**：Muennighoff et al. (2023)、Zhou et al. (2023) 强调高质量小数据的有效性；本文补充证明数据掌握度分类比单纯增加数据量更重要，且存在最佳规模阈值。
- **微调与知识保留**：Ren et al. (2024) 强调保持预训练分布一致性；本文从token分布偏移（KL散度）和参数更新集中度两个角度给出量化解释。
- **幻觉成因**：Gekhman et al. (2024) 指出过拟合导致幻觉；本文进一步揭示低掌握度数据会引发过度参数更新，进而破坏原有知识。
- **高效微调**：Hu et al. (2022) LoRA等方法减少更新参数量；本文提出后处理式的参数恢复，直接针对冗余更新进行修正，与参数高效方法互补。

## 局限性与未来方向
- **未提出端到端高效微调策略**：研究聚焦于现象分析与机制解释，未将参数恢复集成到训练流程中，未来可设计自适应更新机制以避免冗余。
- **模型范围有限**：仅验证了LLaMA-2/3系列，虽初步观察到其他家族结论类似，但需在更多架构（如Mistral、Qwen）和数据类型（非知识性任务）上验证泛化性。
- **恢复比例的自动确定**：当前依赖人工扫描恢复比例（1%~60%），未来可探索基于验证集性能或更新幅度分布自动选择最优恢复比例。
- **知识退化因果性**：KL散度与性能下降的相关性尚未完全确立因果关系，需进一步控制变量实验（如扰动特定参数）验证。

## 研究启发与可借鉴点
- **多模板补全掌握度度量**：通过$N_\text{map}=21$个模板和$N_\text{sample}=10$次采样计算$R_k^\mathcal{M}$，可复用为数据筛选指标，优先选择模型掌握度适中的数据（如$i=2,3$）进行微调。
- **参数恢复作为微调后处理**：在常规SFT后，计算参数相对变化率并恢复前k%（如20%）更新，以低成本提升知识准确性，尤其适用于小样本微调场景。
- **KL散度监控**：在微调过程中定期计算logits KL散度，可实时检测知识退化风险，及时调整学习率或停止训练。
- **冗余参数分布规律**：发现冗余更新集中在早期层（0–2）和FFN层（down/up/gate），未来可针对这些模块设计低秩更新或正则化约束。

## 关键术语表
- **Closed-book Question Answering (CBQA)**：不依赖外部检索资料的封闭式问答，用于评估模型内部存储的事实知识准确性。
- **KL Divergence（KL散度）**：衡量两个概率分布差异的信息论指标，本文用于量化微调前后token logits分布的变化程度。
- **Parameter Restoration（参数恢复）**：将微调后变化幅度最大的参数按比例恢复至预训练值，以减少冗余更新、提升知识保持。
- **Knowledge Mastery Level（知识掌握程度）**：预训练模型对特定知识点正确回答的比例，用于将微调数据划分为不同类别（0~4）。
- **Logits Re-normalization（Logits重新标准化）**：提取微调模型top-10 logits及对应预训练logits，分别做softmax后计算KL散度，以排除高频虚词干扰。
- **Relative Parameter Change（参数相对变化率）**：$r_i = |s_i - p_i|/|p_i|$，用于排序参数更新幅度，识别冗余更新。
- **In-domain / Out-of-domain**：in-domain指微调数据相同主题分布的测试集，out-of-domain指不同主题的测试集，用于评估知识泛化与干扰。

## 可复现要素
- **数据集**：ENTITYQUESTIONS（Sciavolino et al., 2021），公开可从[ACL Anthology](https://aclanthology.org/2021.emnlp-main.533/)获取。
- **代码/权重**：论文未明确开源代码或权重，但提供了完整的实验设置、公式与超参数。
- **关键超参数**：batch size=8，epochs=1，learning rate=1e-5，AdamW优化器，cosine learning rate schedule，temperature=0.7（模板补全采样），max new tokens=32（训练生成）、16（测试生成），随机采样5个训练数据集取均值。
