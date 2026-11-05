---
title: "Non-Existent-Relationship-Fact-Aware-Multi-Level-Machine-Gen"
source: https://aclanthology.org/2025.emnlp-main.186.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 22:06:41"
field: "机器生成文本检测"
keywords: ["机器生成文本检测", "事实幻觉", "实体图", "多层特征融合", "知识图谱", "门控机制"]
innovations: ["首个引入真实世界实体关系核查的机器生成文本检测方法", "设计事实实体图与文本实体图的双图比较模块捕捉非存在关系", "提出实体-句子-文档三级门控渐进式特征融合架构"]
benchmarks: ["GROVER", "GPT-2", "SemEval-2024 Task 8"]
---

# 论文速读：Non-Existent-Relationship-Fact-Aware-Multi-Level-Machine-Gen

## 一句话总结
本文提出**FAML**模型，首次将**实体关系事实核查**引入机器生成文本检测，通过对比文本实体图与Wikipedia事实实体图的差异，结合多层级特征融合架构，有效识别LLM生成文本中的事实幻觉问题。

## 研究问题与动机
- **现有方法局限**：当前监督学习方法主要依赖Transformer微调提取粗粒度文本特征，或仅关注文本内实体共现模式，忽视实体关系的**事实真实性**。
- **事实幻觉问题**：LLM生成文本虽语义连贯，但常包含与现实知识不符的实体关系（如Wikipedia中不存在的连接），这是LLM事幻觉的直接体现。
- **上下文完整性不足**：RoBERTa等模型输入长度限制（如512 tokens）导致长文本信息丢失，无法构建完整的实体图。
- **统计与知识方法的不足**：统计方法（如GLTR、DetectGPT）依赖压缩的token概率特征；知识方法（如CompareNet）使用LSTM编码文本，表达能力弱于Transformer。

## 核心贡献（创新点）
1. **首个引入真实世界实体关系检测的框架**：不同于FAST/COCO仅关注文本内实体共现一致性，本文通过对比文本实体图与知识库实体图，直接捕捉LLM的**非存在关系（Non-Existent Relationship）**幻觉。
2. **事实感知的实体图比较模块**：设计双图GCN编码器分别提取文本/事实实体特征，通过差值计算得到**事实差异特征**，量化实体关系不一致程度。
3. **多层级门控特征融合架构**：提出实体→句子→文档三层渐进式融合机制，学习性门控单元自适应调节粗粒度与细粒度特征的贡献权重。
4. **多数据集验证与可解释性分析**：在GROVER、GPT-2、SemEval三个公开数据集上均超越SOTA，可视化分析证实模型能准确捕捉机器生成文本中更大的实体图差异。

## 方法详解
**FAML模型包含特征提取与特征融合两大部分：**

**（1）特征提取**
- **文档与句子级**：使用RoBERTa提取文档表示 $\mathbf{h}_{CLS}$ 和句子表示矩阵 $\mathbf{H}_s^{ini} \in \mathbb{R}^{n \times d}$。
- **实体图构建**：
  - 用REBEL模型从文本抽取关系三元组，构建**文本实体图** $\mathcal{G}_t = \langle \mathcal{V}_t, \mathcal{E}_t \rangle$。
  - 用TAGME进行实体链接到Wikipedia，查询实体间是否存在真实关系，构建**事实实体图** $\mathcal{G}_f = \langle \mathcal{V}_f, \mathcal{E}_f \rangle$，其中 $\mathcal{V}_t = \mathcal{V}_f$。
- **实体表示**：使用两层GCN分别编码两个图，节点初始嵌入为Word2Vec平均向量：
  $$\mathbf{H}_1 = \sigma(\hat{\mathbf{A}}\mathbf{H}\mathbf{W}_1), \quad \mathbf{H}_2 = \sigma(\hat{\mathbf{A}}\mathbf{H}_1\mathbf{W}_2)$$
  得到文本实体表示 $\mathbf{H}_t$ 和事实实体表示 $\mathbf{H}_f$。

**（2）特征融合**
- **实体比较**：计算事实差异特征 $\mathbf{H}_{diff} = (\mathbf{H}_t - \mathbf{H}_f)\mathbf{W}_{diff}$，经max pooling得到 $\mathbf{h}_{diff} \in \mathbb{R}^d$。
- **实体→句子融合**：对每句内实体做平均池化 $\mathbf{H}_{e2s}$，通过门控单元融合：
  $$\mathbf{g}_s = sigmoid([ \mathbf{H}_{e2s}; \mathbf{H}_s^{ini}\mathbf{W}_s ]\mathbf{W}_{g_s})$$
  $$\mathbf{H}_s^e = \mathbf{g}_s \odot \mathbf{H}_{e2s} + (1-\mathbf{g}_s) \odot \mathbf{H}_s^{ini}$$
- **句子→文档融合**：用Transformer encoder更新句子表示，再经门控融合文档表示：
  $$\mathbf{g}_d = sigmoid([ \mathbf{h}_S; \mathbf{h}_{CLS}\mathbf{W}_d ]\mathbf{W}_{g_d})$$
  $$\mathbf{h}_D = \mathbf{g}_d \odot \mathbf{h}_S + (1-\mathbf{g}_d) \odot \mathbf{h}_{CLS}$$
- **预测**：拼接 $\mathbf{h}_D$ 与 $\mathbf{h}_{diff}$ 输入分类器：
  $$\hat{y} = softmax(\sigma([ \mathbf{h}_D; \mathbf{h}_{diff} ]\mathbf{W}_{l_1})\mathbf{W}_{l_2})$$
  损失函数为标准交叉熵。

## 实验与结果
- **数据集**：GROVER（新闻风格，5k训练）、GPT-2（WebText风格，25k训练）、SemEval-2024 Task 8（多模型生成，6k训练）。
- **基线方法**：GLTR、DetectGPT（统计类）；CompareNet（知识类）；XLNet、GPT-2、RoBERTa（纯微调）；FAST、COCO（实体共现类）；USTC-BUPT（竞赛模型）。
- **主要结果**：
  | 数据集 | 模型 | ACC | F1 |
  |--------|------|-----|-----|
  | GROVER | FAML | **0.9340** | **0.9339** |
  | GPT-2 | FAML | **0.9551** | **0.9550** |
  | SemEval | FAML | **0.9116** | **0.9116** |
- **提升幅度**：相比最强基线FAST，在GROVER上F1提升2.31%（0.9108→0.9339），GPT-2上提升3.28%（0.9222→0.9550），SemEval上提升4.68%（0.8648→0.9116）。
- **消融实验**：完整模型DSEF优于仅文档级D模型约4%，验证各组件有效性。
- **融合方式对比**：门控单元（0.9340 ACC）优于拼接（0.8670）和Add&Norm（0.9176）。

## 相关工作脉络
- **FAST**（Zhong et al., 2020）：构建文本实体图学习实体共现一致性，但未验证关系真实性；本文在其基础上引入**事实核查**维度。
- **COCO**（Liu et al., 2023）：在FAST基础上添加对比学习增强低资源鲁棒性；本文强调**跨文本-知识库图对比**而非仅文本内对比。
- **CompareNet**（Hu et al., 2021）：面向假新闻检测，使用LSTM编码文本；本文针对**机器生成文本**，采用Transformer+GCN架构。
- **GNNfakeNews检测**：传统工作聚焦知识图谱与文本一致性；本文独特之处在于利用**实体关系非存在性**作为幻觉信号。
- **Fact Verification方法**（如Hover、FactKG）：验证命题真假；本文任务为**二分类检测**，更关注关系结构的统计差异。

## 局限性与未来方向
- **关系提取精度依赖**：REBEL模型抽取不完整或错误会影响实体图质量，需探索更鲁棒的关系抽取器。
- **知识库覆盖有限**：Wikipedia API频率限制需本地构建DBpedia，查询可能不完整；实体链接也可能遗漏。
- **幻觉类型受限**：仅检测**实体关系不一致**幻觉，无法识别逻辑矛盾或非实体相关的事实错误。
- **长期有效性存疑**：随着LLM事实性改进，此类幻觉可能减少，需持续调整策略。

## 研究启发与可借鉴点
- **事实核查信号的可迁移性**：将外部知识库作为"真相锚点"的思想可迁移至其他生成内容检测任务（如代码生成、学术写作检测）。
- **多层级门控融合设计**：实体→句子→文档的渐进融合模式可应用于文档级分类任务（如立场检测、虚假信息识别）。
- **差异特征提取策略**：双图对比产生差异向量的方法可推广至跨模态一致性检测（如图文匹配、视频描述验证）。
- **实验设计参考**：消融实验设计系统化（逐组件移除），融合方式对比实验（门控vs拼接vsAdd&Norm）为模型设计验证提供范例。
- **可解释性分析**：可视化实体比较特征分布（机器生成文本差异更大）为模型决策提供直观证据，值得借鉴。

## 关键术语表
- **Non-Existent Relationship**：文本中描述但现实中不存在的实体关系，是LLM事实幻觉的表现形式。
- **FAML**：Fact-Aware Multi-Level模型的缩写，本文提出的机器生成文本检测框架。
- **Entity Graph Comparison**：通过对比文本实体图与事实实体图的边差异，量化事实不一致性。
- **Gating Unit**：基于sigmoid的门控机制，自适应调节多层级特征融合权重。
- **REBEL**：Relation Extraction By End-to-end Language generation模型，用于关系三元组抽取。
- **TAGME**：实体链接工具，将文本实体映射到Wikipedia页面。
- **Fact-Level Feature**：通过文本与事实实体图差异提取的幻觉感知特征。
- **DBpedia**：结构化知识库，本文用于离线存储Wikipedia实体关系数据。

## 可复现要素
- **数据集**：GROVER、GPT-2、SemEval-2024 Task 8均为公开数据集。
- **代码/权重**：论文未提及代码开源情况。
- **关键超参**：学习率1e-5，batch size 16，表示维度128，Transformer heads=4/layers=6，RoBERTa-Large，GCN两层，最大训练epoch 30，early stopping patience 7。
- **环境**：NVIDIA A100 GPU，Ubuntu服务器。
- **工具**：REBEL-large模型、TAGME实体链接工具、DBpedia知识库（本地构建）。
