---
title: "Molecular-String-Representation-Preferences-in-Pretrained-LL"
source: https://aclanthology.org/2025.emnlp-main.56.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:30:23"
field: "化学信息学与LLM交叉"
keywords: ["分子属性预测", "大语言模型", "SMILES", "InChI", "IUPAC名称", "零样本学习", "少样本学习", "分子表示"]
innovations: ["系统揭示LLM对InChI/IUPAC名称的零/少样本显著偏好，挑战SMILES默认假设", "通过ASO检验与原子计数实验从多因素角度归因表示偏好成因", "验证Tanimoto相似度检索提升少样本性能及token操纵对SMILES帮助有限"]
benchmarks: ["MoleculeNet", "BBBP", "BACE", "ClinTox", "ESOL", "FreeSolv"]
---

# 论文速读：Molecular-String-Representation-Preferences-in-Pretrained-LL

## 一句话总结
本文系统评估了四种前沿LLM（GPT-4o、Gemini 1.5 Pro、Llama 3.1 405b、Mistral Large 2）在五类分子属性预测任务上对五种分子字符串表示（SMILES、DeepSMILES、SELFIES、InChI、IUPAC名称）的零样本与少样本偏好，发现模型显著偏好InChI和IUPAC名称，挑战了"分子应以SMILES呈现给LLM"的传统假设。

## 研究问题与动机
- 通用LLM已能推理分子结构，但现有工作默认使用SMILES作为输入表示，缺乏对不同表示方式的系统性比较。
- SMILES存在非规范性（同一分子可有多重合法字符串）、tokenization困难（原子计数需逐字符解析）、以及DeepSMILES/SELFIES在预训练语料中稀缺等问题。
- 分子属性预测数据集规模小（测试集仅65–194个样本），需要严格的统计显著性检验以支撑结论。
- 理解LLM的表示偏好对药物设计、化学问答等下游应用具有重要的实践指导价值。

## 核心贡献（创新点）
- **首次系统揭示LLM在分子属性预测中对InChI和IUPAC名称的显著偏好**，与既往"SMILES最优"假设相悖。
- **通过ASO（almost stochastic order）显著性检验量化表示差异**：零样本下InChI（ε_min=0.17）和IUPAC（ε_min=0.06）均显著优于SMILES；少样本下IUPAC（ε_min=0.16）仍显著占优。
- **从多个维度归因偏好成因**：预训练语料中IUPAC提及率最高（43%）、InChI显式提供原子计数、tokenization效率等，而非单一因素。
- **验证基于Tanimoto相似度的in-context示例检索策略有效提升少样本性能**，部分变体表示（DeepSMILES）在5-shot下可与SMILES持平甚至超越。
- **提出显式键表示的SMILES token操纵实验**，证明现有token级扰动手段对属性预测帮助有限，强化了"表示本身"而非"tokenization技巧"是关键的因素。

## 方法详解
- **模型**：评测GPT-4o、Gemini 1.5 Pro、Llama 3.1 405b、Mistral Large 2，涵盖闭源与开源模型。
- **数据集**：MoleculeNet的5个标准任务——BBBP（血脑屏障穿透，二分类）、BACE（β-分泌酶抑制，二分类）、ClinTox（临床毒性，二分类）、ESOL（水溶性log值，回归）、FreeSolv（水合自由能，回归）。
- **表示转换**：原始数据仅提供SMILES，通过deepsmiles、selfies、pubchempy库转换为其他表示；无法从PubChem获取IUPAC名称时，使用STOUT神经网络翻译生成。
- **提示格式**：采用chain-of-thought推理+结构化输出（Classification: "Decision: Yes/No"；Regression: "Decision: X"）。
- **In-context学习**：从训练集按Tanimoto相似度选取5个与目标分子结构最接近的示例作为context，以提升少样本性能。
- **统计检验**：采用ASO检验（τ=0.2）比较各表示相对SMILES的随机占优关系，并进行Bonferroni多重比较校正；混合模型与任务层面的原子计数准确率与任务性能做Pearson相关性分析。

## 实验与结果
- **零样本最强结果**：Llama 3.1 405b在BBBP上用SMILES达85.4 ROC-AUC；GPT-4o在BBBP上用IUPAC达84.2；多数任务中InChI/IUPAC优于SMILES。
- **少样本最强结果**：Llama 3.1 405b在BBBP上SMILES达85.4→87.4（SELFIES），InChI达84.4；IUPAC在BACE上（67.1→73.1）和ESOL上（1.07→0.9）显著提升。
- **显著性结论**：零样本下InChI和IUPAC均显著优于SMILES；少样本下IUPAC显著优于SMILES，InChI呈优势但未达τ=0.2显著阈值。
- **DeepSMILES/SELFIES**：零样本表现普遍较差，但5-shot后可追平或略超SMILES，印证预训练语料稀缺是主因。
- **与传统基线对比**：本文LLM零/少样本性能可与部分fine-tuned专用模型（如MolTRES、Moleco）竞争，且具有chain-of-thought可解释性优势。
- **原子计数实验**：InChI下模型计数准确率接近100%（因化学式直接给出）；IUPAC约40%；SMILES显著更低。FreeSolv任务中计数准确率与预测误差呈显著负相关（r=-0.49, p=4e-05）。

## 相关工作脉络
- **Guo et al. (2023, NeurIPS)**：最早系统评估LLM化学能力，发现SELFIES零样本性能差，本文在此基础上扩展至更多模型与表示，并深入分析偏好成因。
- **Jablonka et al. (2024, Nat Mach Intell)**：微调GPT-3做多表示联合训练，本文关注零/少样本场景下通用LLM的固有能力，而非任务微调。
- **MolTRES / Moleco (Park et al., 2024)**：针对SMILES的专用化学语言模型，性能更高但需fine-tune且不可解释；本文强调通用LLM+合适表示即可达到有竞争力水平。
- **ChemBERTa / MolT5 / Galactica / BioT5 / nach0**：领域专用预训练模型，参数规模较小、缺乏指令微调；本文证明无化学预训练的通用LLM经适当表示选择同样有效。
- **Zhang et al. (2024) / Schwartz et al. (2024)**：指出tokenization影响LLM计数能力，本文通过实验验证在分子场景下，单纯token操纵对性能提升有限，表示本身的语义显式性更重要。

## 局限性与未来方向
- **任务数量有限**（仅5个）：虽覆盖物理化学、生物物理、生理学类别，但未能覆盖更大规模数据集（如QM9）以验证泛化性。
- **未测试更小的同系模型**：作者认为偏好不因模型尺寸变化，但未提供实证。
- **未包含化学专用模型**（如Galactica、nach0）：因先前工作表明其在属性预测上弱于通用LLM。
- **未探索 Wiswesser Line Notation、Group-based Molecular Representation、MDL表格表示、2D/3D结构图像**等其他表示形式。
- **结论可能不适用于分子描述生成或分子生成任务**，未来需在其他化学NLP任务上验证。

## 研究启发与可借鉴点
- **表示选择策略**：在将LLM应用于分子/化学任务时，优先尝试InChI或IUPAC名称，而非默认SMILES，可显著提升零样本性能。
- **In-context示例检索**：基于Tanimoto相似度从训练集检索相似分子作为context，是一种简单有效的少样本增强手段，可迁移至其他科学问答场景。
- **原子计数作为能力探针**：将"原子计数准确率"与任务性能做相关性分析，为理解LLM化学推理瓶颈提供了可量化的诊断工具。
- **预训练语料 prevalence 分析**：通过子串匹配统计各表示在公开语料（Dolma）中的出现频率，可作为解释模型偏好的辅助证据，此方法可复用于其他领域。
- **ASO显著性检验在小数据集上的应用**：为分子性质预测等小样本场景提供了比传统t检验更稳健的模型比较框架。

## 关键术语表
- **SMILES**：Simplified Molecular Input Line Entry System，将分子结构编码为线性字符串的化学表示法，非规范且存在多种合法等价形式。
- **InChI**：International Chemical Identifier，IUPAC制定的规范化学标识符，采用分层结构显式提供分子式与连接信息。
- **IUPAC Name**：国际纯粹与应用化学联合会制定的系统命名法，以自然语言词汇描述分子结构（如"dichloromethane"）。
- **SELFIES**：SELF-referencIng Embedded Strings，保证100%语法合法的分子字符串表示，每个字符串对应有效分子。
- **DeepSMILES**：SMILES的变体，通过修改环和分支编码减少生成无效分子的概率。
- **ASO检验**：Almost Stochastic Order检验，比较两个算法在给定置信水平下是否存在随机占优关系的统计方法。
- **Chain-of-Thought (CoT)**：引导模型先输出推理过程再给出最终答案的提示策略，提升可解释性。
- **Tanimoto相似度**：基于分子指纹（如Morgan指纹）计算的两个分子结构相似度的常用指标。

## 可复现要素
- **数据集**：MoleculeNet五任务标准测试集（BBBP/BACE/ClinTox/ESOL/FreeSolv），论文使用文献中的标准划分。
- **代码/权重**：模型通过API调用（GPT-4o、Gemini 1.5 Pro）或开源权重（Llama 3.1 405b、Mistral Large 2）；转换库为deepsmiles、selfies、pubchempy、STOUT；论文未提供统一代码仓库链接。
- **关键超参**：少样本示例数k=5；ASO检验阈值τ=0.2；Bonferroni校正；置信水平95%。
- **Prompt模板**：详见附录A，包含各任务的零样本与in-context prompt。
