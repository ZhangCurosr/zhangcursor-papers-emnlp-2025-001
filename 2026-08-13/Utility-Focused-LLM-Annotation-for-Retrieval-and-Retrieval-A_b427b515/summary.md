---
title: "Utility-Focused-LLM-Annotation-for-Retrieval-and-Retrieval-A"
source: https://aclanthology.org/2025.emnlp-main.88.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:43:33"
field: "信息检索与检索增强生成"
keywords: ["retrieval", "retrieval-augmented generation", "LLM annotation", "document utility", "dense retrieval", "curriculum learning"]
innovations: ["提出面向RAG目标的LLM效用标注流程（RelSel/UtilSel/UtilRank）", "设计SumMargLH损失以稳健处理LLM多正样本标注中的噪声", "通过课程学习融合LLM标注与20%人工标注实现接近全量人工标注的检索与RAG性能"]
benchmarks: ["MS MARCO", "BEIR", "TREC DL 19/20", "NQ", "HotpotQA"]
---

# 论文速读：Utility-Focused LLM Annotation for Retrieval and Retrieval-Augmented Generation

## 一句话总结
本文探索使用LLM为检索和RAG训练标注文档效用（utility）以减少对人类标注的依赖，提出了一种基于效用判定的LLM自动标注流程及相应训练策略（含新的SumMargLH损失和课程学习），在MS MARCO上完成约50万查询的标注，并证明LLM标注的检索器在跨领域零样本检索和RAG任务上均优于或媲美仅用人标注训练的模型，且仅需20%人工标签即可达到接近全量标注的性能。

## 研究问题与动机
- **检索目标与RAG目标的错位**：检索器以主题相关性为目标，但RAG需要参考文档对生成过程真正"有用"，仅靠相关性标注训练出的检索器可能无法支撑高质量的RAG。
- **基于下游任务性能标注的局限**：如REPLUG等利用答案生成似然作为效用信号的方法，虽有效但高度依赖大规模人工QA标注，且跨任务/跨指标泛化困难，非事实性问题更难评估。
- **LLM直接选择效用文档的扩展性不足**：让LLM从已检索结果中选有用文档无需人工标注，但推理时成本高，无法扩展到全量语料库。
- **LLM用于大规模自动标注的研究空白**：已有LLM辅助标注工作多聚焦小规模评测数据集，缺少在完整训练集上系统验证其可扩展性和训练有效性的研究。

## 核心贡献（创新点）
1. **提出面向检索和RAG的LLM效用标注流程**：通过RelSel/UtilSel/UtilRank三种提示策略让LLM在候选集中判定文档的生成效用，而非仅判相关性，从而直接对齐RAG目标；区别于以往只用LLM做相关性判断的工作。
2. **提出SumMargLH损失函数以稳健处理多正样本**：针对LLM可能引入错误正样本的情况，将多个正样本的边际似然之和取负对数作为优化目标，避免JointLH因单条噪声标注而严重误导训练。
3. **设计LLM与人工标注的课程学习融合策略**：先用低质量但覆盖更广的LLM标注进行初始训练，再用高质量人工标注进行精修，实现了以少量人工标签复现甚至超越全量标注的效果。
4. **构建大规模LLM效用标注基准**：使用Qwen-2.5-32B-Int8为MS MARCO中约50万查询标注了平均每个查询2.9个正样本，开源数据集与代码，便于社区研究弱监督和检索评估。

## 方法详解
- **候选池构建**：对每个query，聚合来自BM25和CoCondenser检索的硬负样本$ \{d_i^-\}_{i=1}^n $（取n=30），连同唯一人工正样本$ d^+ $组成候选集，从中随机采样m=15个硬负与$ d^+ $一起进入LLM标注。
- **三种LLM标注策略**：
  - **RelSel**：先让LLM选出与query主题相关的子集，再基于伪答案对这些子集进行列表式效用评估。
  - **UtilSel**：直接让LLM选出对生成有用的文档子集。
  - **UtilRank**：让LLM按效用对所有候选文档排序，取前k%（主实验k=10）作为正样本。
- **损失函数设计**：
  - 标准打分：$ P(d|q,D^+,D^-)=\frac{\exp(s(q,d))}{\sum_{d'\in\{d^+\}\cup D^-}\exp(s(q,d'))} $
  - **SingleLH**：$ \mathcal{L}_s=-\log P(d^+|q,d^+,D^-) $，只适用于单正样本。
  - **Rand1LH**：每epoch随机抽一个正样本并用SingleLH。
  - **JointLH**：$ \mathcal{L}_s=-\log\prod_{d^+\in D^+}P(d^+|q,D^+,D^-) $，对噪声正样本敏感。
  - **SumMargLH（新）**：$ \mathcal{L}_s=-\log\sum_{d^+\in D^+}P(d^+|q,D^+,D^-) $，最大化所有正样本的边际概率之和，容忍部分假正。
- **课程学习（CL）**：第一阶段用LLM标注数据训练，第二阶段切换至含部分人工标注的数据继续训练；相比简单合并正样本集，分阶段训练更能利用高质量人工标签的校正作用。
- **RAG评估**：将检索器输出的top-k passages直接输入生成器（Llama-3.1-8B / Qwen-2.5-32B-Int8），在MS MARCO QA（top-1）、NQ和HotpotQA（top-5）上用BLEU、ROUGE-L、BERT-score或EM/F1评估。

## 实验与结果
- **数据集**：训练集为MS MARCO passage set（~503K query，8.8M passage）；检索评测用MS MARCO Dev、TREC DL 19/20及BEIR 14个零样本数据集；RAG评测用MS MARCO QA、NQ和HotpotQA。
- **基线**：Human（仅用人工标注+SingleLH）、REPLUG（用答案生成似然作为效用标签并通过KL散度训练）、以及各UtilSel/UtilRank与CL的组合。
- **领域内检索**：
  - 仅用LLM标注（UtilSel/UtilRank）在Human Test上略低于Human，但在更公平的Hybrid Test（合并GPT-4omini重标）上显著优于Human。
  - **结合20%人工标注的课程学习**可达到与100%人工标注相近的MRR@10和Recall@1000（如RetroMAE backbone下MRR@10达38.2 vs Human的38.6），节省约80%人工成本。
  - 结合100%人工标注可进一步显著超越仅用人工标注的模型。
- **领域外检索（BEIR零样本）**：
  - 仅用LLM标注的训练显著优于仅用人标注，表明人标注易导致在MS MARCO上过拟合；UtilSel平均NDCG@10达45.3%，显著高于Human的43.1%和BM25的42.9%。
  - 加入人工标注后OOD性能下降，再次印证LLM标注对提升泛化性的价值。
- **RAG性能**：
  - 在MS MARCO QA上，UtilSel (CL 100%) 取得最高的BERT-score（68.0%）和ROUGE-L（14.8%）。
  - 在跨域NQ上，UtilSel (0%人工) 取得最佳EM（59.8%）和F1（45.9%），显著优于Human和REPLUG。
  - 整体趋势与检索结果一致，验证了效用导向标注对下游生成的正面迁移。
- **成本分析**：人工标注约137万美元，REPLUG约4.5万美元（需人工答案），本文方法仅约339美元（GPU计算），性价比极高。

## 相关工作脉络
- **DPR/Contriever/RetroMAE**：基于预训练语言模型的密集检索器，使用人工相关性标注和对比损失训练；本文在其基础上引入LLM效用标注以更好地对齐RAG目标。
- **REPLUG (Shi et al., 2024)**：用地面真实答案在给定query+passage下的生成似然作为效用标签，通过KL散度对齐检索器的相关性分布；本文指出该方法需要大量人工QA标注且跨任务迁移受限，而LLM效用标注无需此类标签且更泛化。
- **Zhang et al. (2024a,b)**：利用LLM从已检索结果列表中挑选有用文档进行RAG；本文认为这种"选优"方式在推理时无法扩展到全量语料库，因此将效用判定下沉到训练阶段的标注过程。
- **LLM-assisted relevance annotation**：Thomas et al.、Rahmani et al.、Ni et al.等研究LLM用于相关性判定；本文区别于这些工作的关键在于聚焦"生成效用"而非"主题相关性"，并在大规模训练集上进行系统验证。
- **Curriculum Learning in IR**：Bengio et al.提出课程学习思想；本文将其引入检索训练，先利用覆盖广的LLM标注建立基础表征，再用高精度人工标注进行局部校准，形成成本低且效果好的混合训练范式。

## 局限性与未来方向
- **候选池构建依赖已有检索器**：当前利用BM25和CoCondenser检索的hard negatives构建标注池，不能完全反映真实场景中仅用无监督方法检索的候选分布。
- **未使用更强的LLM进行标注**：受资源限制，主实验采用Qwen-2.5-32B-Int8而非更大或最新模型，潜在标注精度上限未充分挖掘。
- **主要实验集中在MS MARCO**：虽然验证了跨域BEIR和NQ/HotpotQA的RAG，但更多样化的RAG数据集和更长尾语言的覆盖仍有待探索。
- **未来方向**：将效用标注流程扩展到更多RAG基准、使用更强LLM（如Qwen3-32B）进行交叉验证、探索动态课程学习中人工标注比例的自适应选择机制。

## 研究启发与可借鉴点
- **效用标注替代相关性标注的新思路**：将LLM的判定目标从"是否相关"转向"是否对生成有用"，为RAG场景下的检索器训练提供了更贴合下游目标的数据构建范式。
- **SumMargLH损失的通用性**：该损失适用于任何存在多正样本且可能存在标注噪声的场景（如弱监督学习、主动学习），可作为通用鲁棒多正训练技巧复用。
- **课程学习在弱监督与强监督数据融合中的策略**：先利用廉价但噪声较大的LLM标注建立表征，再用少量高质量人工标注精调，这一"粗到细"的训练次序设计值得推广到其他标注成本差异大的任务。
- **Hybrid Test评估协议的公平性**：使用GPT-4omini重标合并原始人工标签作为"金标准"，更公允地比较不同标注来源训练的模型，可为后续研究提供评估范式的参考。
- **与检索增强生成团队的结合机会**：本方法可直接用于团队现有RAG系统中的检索器初始化阶段，尤其是面对新语料库且缺乏人工标注时，能以极低成本获得具备良好跨域泛化能力的检索模型。

## 关键术语表
- **Utility（效用）**：指文档对于下游生成任务（如问答）的实际帮助程度，区别于传统检索中基于主题的"相关性"。
- **RelSel / UtilSel / UtilRank**：三种LLM标注策略，分别通过相关性过滤+效用评估、直接效用选择、按效用排序取Top-K%的方式来标记正样本。
- **SumMargLH**：本文提出的损失函数，最大化多个正样本的边际概率之和的负对数，相比JointLH对假正样本更鲁棒。
- **Curriculum Learning (CL)**：课程学习，本文指先在LLM标注数据上训练检索器，再在人工标注数据上继续训练的阶段性训练策略。
- **REPLUG**：一种利用答案生成似然作为效用信号并通过KL散度训练检索器的方法，需依赖人工标注答案。
- **Hybrid Test**：将原始人工标注正样本与GPT-4omini重标结果合并形成的评估标签集，用于更公平地比较不同标注来源训练的检索器。
- **BEIR**：包含14个不同领域检索数据集的零样本评测基准，用于评估检索器的跨领域泛化能力。
- **Hard Negative**：从检索结果中筛选出的与query看似相关但实际不相关的负样本，用于提升密集检索器的判别能力。

## 可复现要素
- **数据集**：MS MARCO passage set（公开）、BEIR（公开）、NQ（公开）、HotpotQA（公开）；论文额外构建了约50万查询的LLM效用标注集并已开源。
- **代码**：已开源，链接 https://github.com/Trustworthy-Information-Access/Utility-Focused-LLM-Annotation。
- **模型权重**：使用的预训练检索器RetroMAE和Contriever为公开模型；LLM标注器使用Qwen-2.5-32B-Int8（GPTQ量化版本，开源）和Llama-3.1-8B-Instruct（开源）。
- **关键超参**：Retriever训练2 epochs（CL第二阶段1 epoch），batch size=16，learning rate=3e-5，AdamW优化器；LLM标注候选池n=30，采样m=15，UtilRank阈值k=10%；评估指标MRR@10、Recall@1000、NDCG@10、BLEU、ROUGE-L、BERT-score、EM、F1。
- **硬件**：8张Nvidia A800 80GB GPU。
