---
title: "Analyzing-the-Effects-of-Supervised-Fine-Tuning-on-Model-Kno"
source: https://aclanthology.org/2025.emnlp-main.25.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:41:30"
field: "大语言模型微调与知识机制"
keywords: ["SFT", "CBQA", "参数恢复", "KL散度", "知识保持", "LLaMA"]
innovations: ["揭示SFT中90%参数更新冗余", "提出参数恢复策略提升CBQA性能", "建立token/parameter双层分析框架解释知识退化"]
benchmarks: ["ENTITYQUESTIONS"]
---

# 论文速读：Analyzing-the-Effects-of-Supervised-Fine-Tuning-on-Model-Knowledge-from-Token-and-Parameter-Levels

## 一句话总结
本文通过分析LLaMA系列模型在CBQA任务上的表现，揭示了SFT过程中大量冗余参数更新会损害模型知识；提出参数恢复策略，通过还原90%非必要更新可显著提升性能。

## 研究问题与动机
- SFT对LLM内部知识的具体影响机制尚不清楚，限制了预测和控制知识变化的能力。
- 现有研究多关注数据集特征（质量、规模），却忽视了微调过程的内部动态。
- 微调数据类别（模型先验知识掌握程度）和规模如何影响知识变化尚不明确。
- 过度微调可能引发知识退化甚至灾难性遗忘，但缺乏系统量化分析。

## 核心贡献（创新点）
1. **揭示SFT数据规模和类别对CBQA性能的意外影响**：性能峰值出现在仅240个样本时，增加至1920样本反而下降14%，不同类别数据导致性能波动超12%。
2. **Token级分析揭示logits分布偏移与性能下降的关联**：通过KL散度度量，发现数据量过大或低掌握度数据会加剧模型偏离预训练分布，导致性能退化。
3. **发现90%的SFT参数更新是冗余甚至有害的**：参数变化高度集中，超70%的总更新集中在不到1%的参数上；恢复这些参数可提升性能。
4. **提出参数恢复策略并验证其泛化性**：通过按变化幅度排序后逐步还原参数，有效缓解SFT带来的知识损失，且在XSum、GSM8K等下游任务上同样有效。

## 方法详解
**数据集构建与分类**：
- 使用ENTITYQUESTIONS数据集（24个Wikipedia主题），选取10个位置相关主题作为in-domain数据，14个主题作为out-of-domain数据。
- 根据预训练模型对知识点的掌握程度，将数据按正确完成率$R_k^{\mathcal{M}}$分为5类：$\mathcal{D}^{\mathcal{M}}_{train-0}$（0%）至$\mathcal{D}^{\mathcal{M}}_{train-4}$（75%-100%）。
- 采用21个mapping模板，temperature=0.7，每模板采样10次计算掌握度。

**Token级分析**：
- 计算微调模型与预训练模型首token logits分布的KL散度：$s_{KL}(p\|p') = -\sum_i p_i \log\frac{p'_i}{p_i}$。
- 引入logits re-normalization：取微调模型top-10 logits，对应提取预训练模型的值，再计算softmax概率和KL散度，避免dummy words干扰。

**Parameter级分析**：
- 定义参数相对变化率：$r_i = \frac{|s_i - p_i|}{|p_i|}$，按$r_i$降序排列。
- 逐步将最大幅更新的参数还原为预训练值，观察性能变化。

**实验设置**：
- 模型：LLaMA-2-7B/13B/70B、LLaMA-3-8B/70B。
- 训练：batch size=8，1 epoch，AdamW，lr=$1\times10^{-5}$，cosine schedule。
- 测试：greedy decoding，最大长度16，5次随机采样取均值±方差。

## 实验与结果
**主要数据集**：ENTITYQUESTIONS（Sciavolino et al., 2021）。

**核心发现**：
- **Phenomenon 1**：无论数据类别，模型在240样本时达到最优，增至1920样本后性能显著下降。如$\mathcal{D}^{\mathcal{M}}_{train-0}$微调时，LLaMA-3-8B从240→1920样本下降8.86%。
- **Phenomenon 2**：当数据量达1920时，不同掌握度类别导致性能差异超12%。低掌握度数据（$\mathcal{D}^{\mathcal{M}}_{train-0}$）对高掌握度测试集损害最大。
- **Token级结果**：KL散度随数据量先降后升，低掌握度数据下峰值更明显；KL散度增大与性能下降强相关。
- **Parameter级结果**：>70%更新集中在<1%参数中；恢复20%参数即可提升全部模型性能，如$\mathcal{D}^{\mathcal{M}}_{train-0}$微调1920样本后恢复20%提升9.85%。恢复40%参数时，1920样本模型仍获益，而240样本模型开始下降。
- **泛化验证**：在XSum（ROUGE-L最高+0.20）和GSM8K（+2.08）上也观察到恢复参数后性能提升。

**最强结果**：LLaMA-3-8B在$\mathcal{D}^{\mathcal{M}}_{train-2}$上微调1920样本并恢复20%参数后，$\mathbf{Acc}_{test}^{\mathcal{M}}$达62.21%，优于未恢复的58.80%（+3.41%）。

## 相关工作脉络
1. **Ren et al. (2024)**：研究SFT对预训练知识一致性的影响，强调稳定性；本文进一步量化了内部动态机制。
2. **Gekhman et al. (2024)**：指出过拟合是幻觉主因，低掌握度数据加剧问题；本文从其机制角度给出解释。
3. **Ye et al. (2024c)**：提出60样本即可微调QA；本文在此基础上探索了规模上限及数据类别影响。
4. **Muennighoff et al. (2023)、Zhou et al. (2023)**：强调高质量小数据集有效性；本文从参数层面解释了为何"少即是多"。
5. **Allen-Zhu & Li (2025)**：论证预训练知识模块化存储；本文延伸研究SFT如何扰动这些模块。

## 局限性与未来方向
- 未基于发现提出自适应微调策略，仅停留在现象分析。
- 仅验证LLaMA-2/3系列，其他架构泛化性待确认（附录提及初步验证有效）。
- 仅使用ENTITYQUESTIONS数据集，覆盖场景有限。
- 未来可设计抑制冗余更新的微调算法。

## 研究启发与可借鉴点
1. **参数变化集中度度量可作为效率指标**：通过$r_i$排序分析参数冗余，为微调效率评估提供新视角。
2. **知识掌握度分类策略可迁移**：基于预训练模型先验将数据分级的方法，适用于其他知识增强任务。
3. **参数恢复可作为后处理技巧**：在SFT后选择性地还原冗余更新，是一种简单有效的性能修复手段。
4. **实验设计严谨**：多模型家族、多规模、多次随机采样的设置，增强了结论可信度。

## 关键术语表
**CBQA**：Closed-Book Question Answering，闭卷问答，评估模型在不借助外部资料时的 factual 知识能力。
**SFT**：Supervised Fine-Tuning，监督微调，使用标注数据调整预训练模型以适应特定任务。
**KL散度**：Kullback-Leibler divergence，衡量两个概率分布差异的指标。
**参数恢复**：将微调后发生大幅变化的参数还原为预训练值，以保留原有知识。
**知识掌握度**：预训练模型对特定知识点的正确完成比例，用于数据分类。
**logits re-normalization**：为消除 dummy words 干扰，将微调模型与预训练模型的top-k logits进行对齐后再计算KL散度。

## 可复现要素
- 数据集：ENTITYQUESTIONS（开源，Sciavolino et al., 2021）。
- 模型权重：LLaMA-2/3系列（需申请访问）。
- 代码：论文未提供公开代码仓库，但实验细节描述充分。
- 关键超参：batch_size=8，epochs=1，lr=$1\times10^{-5}$，temperature=0.7，N_map=21，N_sample=10。
