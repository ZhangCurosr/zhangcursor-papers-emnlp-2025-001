---
title: "RuCCoD-Towards-Automated-ICD-Coding-in-Russian"
source: https://aclanthology.org/2025.emnlp-main.129.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:57:01"
field: "临床自然语言处理与医疗编码自动化"
keywords: ["ICD coding", "entity linking", "diagnosis prediction", "RAG", "PEFT", "low-resource clinical NLP", "Russian biomedical NLP", "EHR"]
innovations: ["发布首个俄语 ICD-10 EHR 编码与诊断预测双数据集 RuCCoD/RuCCoD-DP", "提出人工与 AI 双标签对照训练设计并量化标注质量增益", "系统揭示跨术语实体链接在细粒度 ICD 分级中的迁移边界"]
benchmarks: ["RuCCoD ICD coding entity-level and EHR-level", "RuCCoD-DP diagnosis prediction (weighted F1 by code frequency)", "strict/relaxed acc@k entity linking"]
---

# 论文速读：RuCCoD-Towards-Automated-ICD-Coding-in-Russian

## 一句话总结
本文发布了首个俄语电子健康记录（EHR）ICD-10编码数据集 RuCCoD，系统评估了 BERT、LLaMA+LoRA 与 RAG 等多种方法，并证明用 AI 自动生成的 ICD 编码重标注训练诊断预测模型可使加权 F1 显著提升 **28%**。

## 研究问题与动机
- 俄语属于生物医学低资源语言：UMLS 俄语词汇总量仅占英语 UMLS 约 1.96%，临床领域标注资源稀缺。
- 真实临床场景中医师兼做 ICD 编码易出现细粒度分级错误，影响编码质量与后续利用。
- 现有 ICD 编码工作主要集中于英文（如 MIMIC-III/IV），缺少面向俄语的多任务评测基准与可复现方案。
- ICD 的细粒度层级结构使跨语料/跨术语（如 UMLS → ICD-10）的直接迁移存在明显噪声与性能衰减。

## 核心贡献（创新点）
1. 发布 RuCCoD（10k+ 实体、1500+ 唯一 ICD 码）与 RuCCoD-DP（86.5 万份 EHR，16.4 万患者）两个俄语临床编码/诊断预测数据集，填补低资源语言基准空白。
2. 建立覆盖 NER+EL、PEFT+RAG、零样本 RAG 等多样化方案的综合基准，并给出端到端 EHR 级别去重聚合评估。
3. 提出“医师原始编码 vs. AI 自动链接编码”双标签实验设计，量化人工规范化难度与 AI 辅助重标注的收益。
4. 系统揭示跨术语迁移的边界：UMLS 词汇扩展在监督设定下会引入显著噪声（acc@1 下降 8–14%），而疾病实体 NER 更具可迁移性。
5. 将最佳编码模型应用于大规模真实 EHR，证明重标注数据可在训练早期更快收敛并获得更稳定的多标签预测性能。

## 方法详解
- **任务定义**
  - ICD Coding：从医师书写的主诉/诊断结论中识别疾病实体并链接到 ICD-10-CM 编码，最终进行 EHR 级别去重聚合。
  - Diagnosis Prediction (DP)：基于患者历史 EHR 多标签预测当前就诊的 ICD 编码，不使用当次医师结论文本。
- **RuCCoD 标注流程**
  - 由三名医学专家独立标注（训练集每千份各标 1000 份；测试集 500 份三人共标），至少两名一致方可采纳；均值配对 JSC 约 0.33。
  - 采用 BRAT 标注平台，结合 ICD-10-CM 词典与外部资源核查。
- **BERT-based IE 管线**
  - NER：基于 Ru-BioBERT + softmax；引入 NEREL-BIO / RuCCoN / RuCCoD 及组合进行迁移试验。
  - EL：使用 SapBERT / CODER / BERGAMOT 对实体与词典词条做平均池化编码后按欧氏距离检索 top-k；通过 BioSyn 的同义词边缘化（synonym marginalization）微调。
  - 文档级评估：去重后基于 TP/FP/FN 计算 micro-precision/recall/F1/accuracy，并报告 strict 与 relaxed（截断至三级码）两种 acc@k。
- **LLM with PEFT / RAG**
  - NER 阶段：使用 LLaMA3 / Phi3_5_mini / Llama3-Med42-8B / Mistral-Nemo / Qwen2.5-7B 等，配合 LoRA 微调与任务 prompt。
  - 字符串匹配容错：精确匹配与编辑距离阈值 ≤2。
  - EL 阶段采用 RAG：三种向量库策略——ICD dict.；ICD dict.+RuCCoD 实体；BERGAMOT 在 RuCCoD 上微调后的嵌入；使用 BGE/BERGAMOT 嵌入 + FAISS 检索 top-15，再由 LLM 做最终选择。
  - 训练时随机打乱诊断列表以缓解类别不平衡。
- **Diagnosis Prediction 模型**
  - 基于 Longformer，以私人多机构 EHR 预训练 BERT 初始化并继续长序列预训练；采用加权 F1（权重取两训练集并集中各类别占比）。
  - 比较两组训练信号：original（医师原始编码）与 linked（AI 管线重链接编码）。

## 实验与结果
- **数据规模**
  - RuCCoD：train 3000 / test 500；实体 8769 / 1557；唯一 ICD 码 train 1455 / test 548；平均每份 3 个码。
  - RuCCoD-DP：865,539 份就诊，164,527 名患者；公共测试集 494 份 / 450 名患者。
- **ICD Coding（Entity-level）**
  - 最优 BERT 方案：Precision 0.512 / Recall 0.542 / F1 0.525 / Acc 0.356（NER: NEREL-BIO+RuCCoD，EL: RuCCoD）。
  - 最优 LLM+RAG 零样本：LLaMA3-8b-Instruct + ICD dict.+RuCCoD，F1 0.458。
  - 最优 PEFT 微调：Phi3_5_mini + ICD dict.+RuCCoD，F1 0.480 / Acc 0.316；Llama3-Med42-8B F1 0.435。
  - 跨术语结论：UMLS 同义词扩展使监督 acc@1 分别下降 SapBERT 8.1%、CODER 8.4%、BERGAMOT 14.3%；跨语料直接训练未能超越零样本；NER 迁移可行（组合 NEREL-BIO+RuCCoD 得 F1 0.72）。
- **Diagnosis Prediction**
  - 加权 F1：linked 训练峰值 **0.48**，original 训练早 plateau 至 **0.2**，差距达 **0.28（+28%）**。
  - 频率鲁棒性：top10% 高频码 linked 中位数 F1 提升约 0.3（0.7 vs 0.4）；bottom10% 低频码提升约 6 倍（0.6 vs 0.1）。
  - 混淆矩阵分析（common test）：
    - Top10%：linked 相对 original，FN 减少 40%，TP 增加 43%。
    - Bottom10%：FN 下降 53%，FP 下降 91%，TN 上升 45%，TP 上升 73%。
  - 个案退化：I25.2 被错误重链接至 I11.9 等，症状重叠导致跃迁收益低（0.02），区分度高的跃迁（如 I25.2→E11.9）可达 0.48。

## 相关工作脉络
- 面向英文 ICD 编码的 MIMIC-III/IV 系列与大量 BERT/CNN/LSTM 多标签方法，本文聚焦俄语低资源场景并提供端到端评测。
- 已有生物医学实体链接工作（SapBERT、CODER、BERGAMOT、BioSyn）主要在 UMLS 体系上评估；本文将其适配至 ICD-10-CM 并揭示跨术语边界。
- 临床 ICD 编码的 LLM/RAG 应用（如 MedCodER、Klang 等）强调检索增强提升精度；本文与之对齐并在俄语 EHR 上验证，同时给出 PEFT 与零样本 RAG 的系统对比。
- 诊断预测中利用患者历史的深度学习路线（NECHO、DPSS、MERA 等）关注结构化编码序列；本文以 Longformer 直接处理长文本 EHR，并比较人工 vs. AI 标签质量。
- 本文与 RuCCoN、NEREL-BIO 的关系：三者共享俄语背景但分别面向临床记录、PubMed 抽象与 ICD 链接，本文证实跨语料直接迁移并非“银弹”。

## 局限性与未来方向
- 数据集临床场景与人群代表性有限，跨专科/机构分布不均可能影响泛化。
- 长尾 ICD 编码仍显不足，低频疾病召回受限。
- 跨语料语义差异的系统性分析未展开（仅做初步迁移实验）。
- 未来计划引入知识图谱等外部医学知识，并进一步研究 LLM 在罕见码上的泛化能力。

## 研究启发与可借鉴点
- **双标签对照设计**值得广泛复用：同一文本源生成“人工规范化”与"AI 规范化”两套标签，可客观量化标注质量对下游预测的影响。
- **EHR 级别去重聚合评估**更贴近临床目标，避免重复提及造成的度量偏差，可作为 ICD 类任务的标准报告方式。
- **NER 与 EL 的可迁移性分化**结论具有普适参考价值：实体识别通常比细粒度概念链接更易跨域/跨术语迁移，训练策略应有所区分。
- **RAG + PEFT 的组合范式**在低资源语言中展现出可扩展性：检索保证词典覆盖，微调强化任务适配，二者协同优于单一路径。
- **加权 F1 + 混淆分量 + 频率分层**的三维评估体系，可帮助识别模型在高低频码上的结构性差异，指导采样与损失设计。

## 关键术语表
- **ICD-10-CM**：国际疾病分类第十版临床修订本，用于标准化诊断编码。
- **Entity Linking (EL)**：将文本中的实体提及链接到知识库/词典中的规范概念。
- **RAG**：检索增强生成，先检索相关知识片段再交由模型生成或选择答案。
- **PEFT / LoRA**：参数高效微调，通过低秩适配器在不全量更新权重的前提下微调大模型。
- **Weighted F1**：按类别出现频率加权计算的 F1，缓解长尾多标签数据不平衡。
- **Strict / Relaxed 评估**：严格匹配与截断至更高层级码组的宽松匹配。
- **Synonym Marginalization**：在训练中对同义词表示进行边缘化处理以提升链接鲁棒性。
- **JSC（Jaccard Similarity Coefficient）**：用于衡量多标签集合重叠程度的相似度指标。

## 可复现要素
- 数据集：RuCCoD 已开源并以 CC BY 4.0 发布；RuCCoD-DP 为内部私有 EHR 数据，未公开。
- 代码：https://github.com/auto-icd-coding/ruccod（论文声明开源）。
- 关键超参（论文 Table 5）：
  - NER（RuBioBERT）：LR=1e-5，Epochs=20，BS=32，Cosine scheduler，WD=0.01。
  - EL（BERGAMOT+BioSyn）：LR=2e-5，Epochs=20，BS=32，Adam，WD=0.01。
  - LLM tuning（LoRA）：LR=5e-5，Epochs=33，BS=2，Linear with Warmup，WD=0.01。
  - ICD code prediction（Longformer）：LR=5e-5，Epochs=2，BS=4，Linear with Warmup，WD=0.01。
- 检索配置：top-k=15，温度=0，FAISS 索引；嵌入可选 BGE 与 BERGAMOT。
- 训练硬件：V100/A100 GPU（论文给出单次评估耗时参考）。
