---
title: "Molecular-String-Representation-Preferences-in-Pretrained-LL"
source: https://aclanthology.org/2025.emnlp-main.56.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:54:24"
---

# 论文速读：Molecular-String-Representation-Preferences-in-Pretrained-LL

## 一句话总结
系统评测了四种主流通用 LLM 在五类分子字符串表示下的零样本/少样本属性预测能力，发现模型显著偏好 InChI 和 IUPAC 名称，直接挑战了“分子必须转为 SMILES 输入 LLM”的既有假设，并揭示了 tokenization 粒度、原子计数显式性与预训练语料覆盖率共同驱动该偏好的内在机制。

## 研究问题与动机
- 通用大语言模型已展现出处理分子属性预测自然语言任务的能力，但输入字符串的表示形式对预测性能的影响机制尚未被系统厘清。
- 业界默认将分子转换为 SMILES 字符串喂给 LLM，但不同化学表示法在结构化粒度、tokenization 友好度及预训练语料分布上存在本质差异，盲目沿用 SMILES 可能并非最优策略。
- 分子属性数据集普遍较小（测试集仅 65–194 条），传统机器学习/GNN 易过拟合且缺乏可解释性；明确 LLM 的表示偏好可为药物设计与化学教育中的 LLM 集成提供更可靠的工程指导。
- 现有 SMILES 变体（DeepSMILES/SELFIES）主要为提升生成合法性而设计，其在 LLM 零/少样本推理场景下的真实效用及上下文学习补偿潜力未被充分评估。

## 核心贡献（创新点）
1. **构建首个面向通用 LLM 的多表示分子属性预测对比基准**。与以往追求微调刷榜或仅使用单一 SMILES 的评测不同，本文在相同零/少样本、CoT 提示下横向对比五种字符串表示，填补了 LLM 化学表示偏好研究的空白。
2. **发现并验证 InChI 与 IUPAC 名称的统计显著偏好**。通过 ASO 检验证明两者在零样本下显著优于 SMILES，直接反驳了“SMILES 是 LLM 分子输入最优格式”的传统经验，为后续化学 NLP 提示工程提供了反直觉的指导原则。
3. **提出基于 Tanimoto 相似度的 Few-shot 示例检索范式**。不同于固定模板或随机选取，本文利用 Morgan 指纹相似度从训练集检索与目标分子结构最相近的 5 个示例，证明该策略可有效激活 DeepSMILES/SELFIES 等原本弱势表示的预测能力。
4. **多维度归因解析偏好成因并量化 tokenization 瓶颈**。结合预训练语料统计、原子计数准确率实验与 token 长度分析，指出偏好并非源于字符串更短，而是由语料覆盖度、显式化学式提供原子数、以及 SMILES token 割裂导致的计数困难共同决定。
5. **验证现有 token 操纵策略对 SMILES 的局限性**。尝试空格/逗号分隔与显式键合并等文本扰动，发现其对预测性能改善有限，提示单纯依赖后处理 token 拆分无法根治 SMILES 的 LLM 理解瓶颈，需面向化学语义的底层 tokenizer 改进。

## 方法详解
- **模型与数据集**：评测 GPT-4o、Gemini 1.5 Pro、Llama 3.1 405b、Mistral Large 2；任务来自 MoleculeNet 基准，包含 BBBP、BACE、ClinTox（二分类，ROC-AUC↑）与 ESOL、FreeSolv（回归，RMSE↓），测试集规模 65–194。原始数据均为 SMILES，通过 `pubchempy`、`deepsmiles`、`selfies` 及 STOUT 神经机器翻译模型转换为其余四种表示。
- **提示与推理设置**：采用 Chain-of-Thought 提示，要求模型输出推理过程后以 `"Decision: ..."` 格式给出最终答案。零样本直接提问；5-shot 少样本从训练集中检索 Top-5 相似样本作为上下文。
- **相似性检索策略**：使用 Morgan 指纹计算目标分子与训练集分子的 Tanimoto 相似度，确保上下文示例在化学结构上与目标高度相关，利用 LLM 的结构类比推理能力提升预测。
- **显著性检验协议**：因数据集小，采用 Almost Stochastic Order (ASO) 检验
