---
title: "Emotion-Transfer-with-Enhanced-Prototype-for-Unseen-Emotion"
source: https://aclanthology.org/2025.emnlp-main.31.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:29:26"
field: "对话情绪识别"
keywords: ["情绪识别", "零样本学习", "对话理解", "原型迁移", "条件随机场", "大语言模型"]
innovations: ["首次提出UERC任务并构建跨数据集未见情绪识别评测框架", "设计LLM增强描述（LED）模块利用隐式示例增强情绪原型语义", "提出参数无关的高斯自注意力（GSA）与注意力维特比解码（AVD）实现已知情绪向未见情绪的转移预测"]
benchmarks: ["IEMOCAP", "EmoryNLP", "MELD"]
---

# 论文速读：Emotion-Transfer-with-Enhanced-Prototype-for-Unseen-Emotion

## 一句话总结
本文首次提出对话中未见情绪识别（UERC）任务，旨在利用训练阶段已知情绪的迁移能力预测未见情绪；提出 ProEmoTrans 原型迁移框架，通过 LLM 增强描述、高斯自注意力机制和改进的 Viterbi 解码，在三个公开数据集上显著超越现有基线。

## 研究问题与动机
- **心理学情绪分类缺乏共识**：Ekman、Plutchik、Cowen & Keltner 等理论存在多种分类体系，各数据集（IEMOCAP、EmoryNLP、MELD）的情绪标签重叠度极低，导致单数据集模型难以跨域泛化。
- **隐式情绪表达难以定义**：对话中的情绪往往以隐式方式表达，仅依赖词典定义难以获得鲁棒且可靠的原型向量。
- **长对话话语编码困难**：现有 ERC 方法依赖额外的参数化模块（LSTM、GCN、GAT），在零样本设定下易过拟合到训练集已知情绪，损害泛化能力。
- **情绪转移的马尔可夫流难以迁移**：情绪具有马尔可夫特性（当前情绪受前序影响），但未见情绪无法预学转移分数矩阵。

## 核心贡献（创新点）
1. **首次提出 UERC 任务**：构建跨数据集训练-测试的新范式，填补了零样本情绪识别研究的空白。
2. **LLM 增强描述方法（LED）**：利用 LLM 上下文学习能力生成隐式情绪示例语句，增强复杂情绪原型表征，解决词典定义不够丰富的问题。
3. **参数无关的高斯自注意力机制（GSA）**：基于距离的离散高斯分布加权聚合跨话语信息，无需额外参数即可建模话语间关系，避免过拟合。
4. **改进的注意力维特比解码（AVD）**：在 CRF 框架下学习已知情绪间的转移概率，并通过原型相似度将转移概率扩展至未见情绪，实现情绪序列迁移。

## 方法详解
- **任务定义**：训练集 $\mathcal{D}_s$ 含已知情绪集合 $\mathcal{E}_s$，测试集 $\mathcal{D}_u$ 含未见情绪集合 $\mathcal{E}_u$，二者无交集；目标是从 $\mathcal{D}_s$ 学习并迁移预测每个话语的 $e_i^{uns} \in \mathcal{E}_u$。
- **情绪原型编码**：从 Wiktionary 获取情绪词定义 $X_i^{desc}$，用 LLM（ChatGPT-3.5）生成两条隐式情绪语句 $X_i^{llm}$，拼接为 $X_i^{see} = \{[CLS], e_i^{see}, X_i^{desc}, X_i^{llm}, [SEP]\}$，经 Encoder_E（Bert-base-uncased）取 [CLS] 得到原型 $h_i^{see}$。
- **高斯自注意力（GSA）**：对每个话语 token $h_i$，计算高斯注意力分数 $A_i = \text{Softmax}(\frac{h_i H^T}{d}) \cdot \mathcal{N}_i$，其中 $\mathcal{N}_i \sim \mathcal{N}(i, \sigma^2)$，$\sigma=0.5$；更新表示 $h_i^{utte} = h_i + A_i H$，无参数，距离越近的话语获得越高权重。
- **对比相似度与 CRF 训练**：定义对比相似度 $s_{ij}^{see} = \frac{e^{\cos(h_i^{utte}, h_j^{see})/\tau}}{\sum_{j'} e^{\cos(h_i^{utte}, h_{j'}^{see})/\tau}}$（$\tau=0.02$），将 $S^{see}$ 输入 CRF，转移矩阵 $M$，最小化序列负对数似然 $\mathcal{L} = -\log p(\hat{y})$。
- **注意力维特比解码（AVD）与推理**：在已知情绪集上进行 AVD 解码，得到概率矩阵 $\mathcal{P}$，用 Eq.13 融合已知原型增强话语表示 $h_i'$，最后通过最近邻搜索 $\arg\max_j \cos(h_i', h_j^{uns})$ 预测未见情绪。

## 实验与结果
- **数据集**：IEMOCAP（6 类情绪，4810/1000/1623 话语）、EmoryNLP（7 类，7551/954/984）、MELD（7 类，9989/1109/2610），仅使用文本模态；采用六组跨数据集迁移设置。
- **基线**：DialogueGCN、DialogueCRN、DualGAT、CPTC（Few-shot）、SACL-LSTM、SCCL、EACL（Contrastive），以及 Llama-3.1-8b、Qwen-2.5-7b、GPT-4o、DeepSeek-V3（LLM）。
- **评估指标**：仅针对未见情绪的加权平均 F1（wF1）。
- **最强结果**：ProEmoTrans 平均 wF1 为 **32.16**，较最佳基线 DeepSeek-V3 的平均提升显著；各设置提升幅度为 2.05%～11.58%。具体：$\varepsilon \to \mathcal{T}$ 达 37.27，$\mathcal{M} \to \mathcal{T}$ 达 32.36，$\mathcal{T} \to \varepsilon$ 达 28.34，$\mathcal{M} \to \varepsilon$ 达 20.73，$\mathcal{T} \to \mathcal{M}$ 达 38.59，$\varepsilon \to \mathcal{M}$ 达 35.64。
- **Few-shot 结果（16-shot）**：ProEmoTrans 在三个数据集上均超越 CPTC，分别提升 1.89%、1.38%、2.01%。
- **消融结论**：去除 LED 导致平均 wF1 下降 **14.3%**，去除 CRF+AVD 下降 **6.93%**，去除 GSA 下降 1.06%，证明三者均重要；LED 使用 2 条描述已充分，3 条提升微弱（+0.33%）。

## 相关工作脉络
- **DialogueGCN / DialogueCRN / DualGAT**：传统参数化 GNN/RNN 方案，在 UERC 中因过拟合已知情绪而性能很差（wF1 < 8），本文方法通过无参数 GSA 避免此问题。
- **SACL-LSTM / SCCL / EACL**：对比学习改进 ERC，利用语义区分性原型，在零样本设定下优于特征基线，但本文进一步引入原型迁移和 CRF 转移建模，性能更优。
- **CPTC（Xu et al., 2023）**：交叉任务 Few-shot 方法，主要改善相似情绪跨任务识别，未见情绪识别效果不佳；本文首次系统探索零样本未见情绪识别。
- **ZS-BERT（Chen & Li, 2021）/ RE-matching（Zhao et al., 2023）**：零样本关系抽取中的原型对齐方法，本文借鉴其原型匹配思路并将其迁移到 ERC 领域。
- **EmotionFlow / SPCL（Song et al., 2022a,b）**：关注情绪转移和对比学习，但未考虑未见情绪场景，本文在此基础上引入 AVD 解码迁移转移概率。

## 局限性与未来方向
- LED 模块依赖人工设计的 prompt 模板，效果未在更复杂情绪上验证，未来可探索自动 prompt 调优。
- 方法仅使用文本模态，未融合多模态信息（如面部表情、语音），多模态融合是潜在改进方向。
- 未见情绪类别占比低时，ProEmoTrans 性能下降明显（Figure 4），对类别分布更敏感，需进一步增强小样本鲁棒性。

## 研究启发与可借鉴点
- **LLM 增强原型描述**：将 LLM 生成隐式示例作为辅助信息融入原型学习，这一思路可迁移至其他零样本/少样本 NLP 任务（如关系抽取、情感分类）。
- **无参数距离感知注意力**：GSA 用离散高斯分布替代可学习参数来建模位置相关关系，设计简洁且防过拟合，可在长序列建模场景中复用。
- **CRF + 原型迁移的解码策略**：AVD 将已知标签间的转移概率通过原型相似度推广到未见类别，这一"已见→未见"的转移扩展策略具有普适性，可应用于其他序列标注任务的零样本扩展。

## 关键术语表
- **UERC（Unseen Emotion Recognition in Conversation）**：对话中未见情绪识别，首次在已知情绪集上训练、在未见情绪集上评测的零样本 ERC 任务。
- **ProEmoTrans**：本文提出的原型情绪迁移框架，通过原型对齐与 CRF 转移实现跨情绪标签集的迁移预测。
- **LED（LLM-Enhanced Description）**：利用 LLM 上下文学习能力为情绪生成隐式示例语句，增强原型语义表征的方法。
- **GSA（Gaussian Self-Attention）**：无参数的距离感知自注意力机制，使用离散高斯分布对相邻话语加权聚合信息。
- **AVD（Attention Viterbi Decoding）**：改进的维特比解码算法，输出每个话语对各已知情绪的概率，并扩展至未见情绪迁移。
- **CRF（Conditional Random Field）**：条件随机场，用于建模话语间情绪转移的序列标注概率模型。
- **InfoNCE / 对比相似度**：基于温度缩放余弦相似度的对比学习损失，拉近同类话语-原型、推远异类对。

## 可复现要素
- **数据集**：IEMOCAP、EmoryNLP、MELD，均为公开数据集；文本模态数据可从官方渠道获取。
- **代码/权重**：论文未声明代码开源。
- **关键超参**：Encoder 使用 Bert-base-uncased；LLM 使用 ChatGPT-3.5；AdamW 优化器，batch size=4，learning rate=2e-5，10 epochs，warmup=100 步；高斯方差 $\sigma=0.5$，对比温度 $\tau=0.02$；LED 生成 2 条描述；评估指标为未见情绪的加权平均 F1。
- **实验环境**：NVIDIA RTX 8000。
