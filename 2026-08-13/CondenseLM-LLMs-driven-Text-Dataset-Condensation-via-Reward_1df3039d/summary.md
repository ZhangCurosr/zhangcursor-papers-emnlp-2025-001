---
title: "CondenseLM-LLMs-driven-Text-Dataset-Condensation-via-Reward"
source: https://aclanthology.org/2025.emnlp-main.65.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:40:05"
field: "自然语言处理中的数据效率"
keywords: ["数据集压缩", "文本数据挖掘", "LLM生成", "奖励匹配", "数据效率", "核心集选择"]
innovations: ["首次提出LLM驱动的文本级数据集压缩框架CondenseLM，突破优化方法在离散文本上的瓶颈", "设计LDSC模块提取判别性特征并生成信息密集的紧凑样本，结合奖励匹配机制最大化代表性覆盖率", "证明优化方法在文本级压缩中的固有局限性，提出高效且跨模型泛化的替代范式"]
benchmarks: ["SST-2", "MNLI", "AG News", "IMDB"]
---

# 论文速读：CondenseLM-LLMs-driven-Text-Dataset-Condensation-via-Reward

## 一句话总结
论文提出了CondenseLM，一个基于LLM的文本级数据集压缩（Dataset Condensation）框架，通过LLM驱动的子集压缩（LDSC）模块提取判别性特征并生成信息密集的紧凑样本，结合奖励匹配（Reward Matching）机制引导迭代生成过程以最大化代表性和覆盖率，在四个文本分类基准上显著优于现有优化方法和核心集选择方法，同时将压缩时间从24小时降至15分钟以内。

## 研究问题与动机
1. **现有文本级数据集压缩方法效果有限**：Discretization和DiLM等优化方法因文本离散性导致信息压缩能力受限，梯度相似度仅比核心集选择略高（DiLM仅+0.12%），性能优势微弱
2. **优化方法计算成本过高**：DiLM生成10-DPC数据集需超过24小时，但相比K-Center核心集选择仅提升0.4%准确率，性价比极差
3. **嵌入级方法缺乏实用价值**：此类方法优化输入词嵌入而非离散文本，缺乏可解释性且无法跨模型泛化
4. **LLM驱动的数据生成研究尚未探索数据集压缩任务**：现有LLM数据生成工作集中于数据增强，直接迁移到压缩任务效果不佳（Table 7消融验证）

## 核心贡献（创新点）
1. **首次提出LLM驱动的有效且高效的文本级数据集压缩框架CondenseLM**，突破了优化方法在离散文本上的固有瓶颈，与DiLM等方法本质区别在于放弃梯度匹配优化而采用LLM生成+奖励引导的新范式
2. **设计了LLM驱动的子集压缩（LDSC）模块**，让LLM识别判别性特征与普遍特征后生成更紧凑、信息更密集的样本，与传统数据增强中单纯依靠prompting生成数据的本质区别在于以"信息压缩"和"偏差缓解"为双重目标
3. **引入奖励匹配（Reward Matching）机制**，训练代表性奖励模型（Heavy Regularization）和覆盖率奖励模型，以R_rep和R_cov分数指导Retrieval Stage和Condensation Stage，区别于既有数据集压缩中的梯度/轨迹/分布匹配损失函数
4. **系统论证了优化方法在文本级压缩中的局限性**（Table 1梯度相似度分析），并提出LLM驱动方案作为更有效的替代路径

## 方法详解
**整体框架**：迭代式子集压缩流程，共T轮迭代，每轮包含Retrieval Stage和Condensation Stage，最终D_syn = ∪S_syn^(t)，N_syn = K'·T

**LDSC模块**：
- 给定原始子集S_real^(t)，让LLM作为"encoder"识别其中的判别性特征（discriminative features，对分类关键的模式）和普遍特征（universal features，跨类别共享的中性模式）
- 再以"decoder"角色生成更紧凑的子集S_syn^(t)（通常为原大小的20%-50%），每个样本浓缩多个判别性特征，同时保持普遍特征的类别平衡

**奖励匹配机制**：
- **代表性奖励模型M_rep**：在原始数据集D_real上用重度正则化（dropout=0.3，early stopping 1 epoch）训练，只擅长识别常见可泛化模式
  - R_rep(x_i, y_i) = M_rep(x_i)_y_i（模型对正确标签的置信度）
- **覆盖率奖励模型M_cov^(t-1)**：在已有压缩子集的并集上训练
  - R_cov(x_i, y_i) = R_rep(x_i, y_i) - M_cov^(t-1)(x_i)_y_i（衡量新样本带来的未覆盖信息）
- **Retrieval Stage筛选**：保留满足R_rep ≥ θ_rep且R_cov ≥ θ_cov的样本后，用K-Center选取子集
- **Condensation Stage优化**：对同一子集运行LDSC多次（N个候选），用Best-of-N选择累计得分Σ[R_rep + w·R_cov]最高的候选作为S_syn^(t)

**关键超参**（Table 13）：θ_rep=0.8，θ_cov=0.3，w=0.2，N=10，子集压缩率20%-50%

## 实验与结果
**数据集**：SST-2（67.3k，2类）、MNLI（392k，3类）、AG News（120k，4类）、IMDB（25k，2类）

**基线**：
- 文本级压缩：Discretization、DiLM
- 核心集选择：Random、Herding、K-Center、EL2N、Moderate、G-DIG
- 测试模型：BERT_base、RoBERTa_base、BERT_large、XLNet_base

**主要结果**（20-DPC vs BERT_base）：
- SST-2：CondenseLM **82.4±2.5%** vs DiLM 80.3±2.8% vs K-Center 76.9±4.4% vs Full 92.7%
- MNLI：CondenseLM **51.3±1.6%** vs DiLM 48.7±2.6% vs K-Center 45.3±3.0% vs Full 86.7%
- AG News：CondenseLM **85.7±0.9%** vs DiLM 84.9±2.1%
- IMDB：CondenseLM **79.0±4.4%** vs K-Center 78.3±3.2%

**最强结果与提升幅度**：
- 5-DPC时优势最明显：SST-2上超DiLM 4.7%，MNLI上超DiLM 6.3%
- 跨模型泛化：MNLI+XLNet_base上达54.7%，比DiLM（44.7%）提升**10.0%**，比K-Center（43.5%）提升**11.2%**
- 高效性：DiLM需>24小时，CondenseLM <15分钟

**消融实验**（Table 7，20-DPC）：去掉LDSC后降至79.2±4.3；去掉RepRM降至78.9±4.9；去掉CovRM降至79.9±3.6；去除HeavyReg降至80.2±3.6

**小模型验证**（Table 8，5-DPC）：Qwen2.5-14B在SST-2上达75.0%，Gemma2-9B达75.9%，均超所有基线

## 相关工作脉络
1. **Dataset Condensation初始提出**（Wang et al., 2018）：首次将数据集压缩定义为生成紧凑合成数据集的问题，后续工作集中于图像领域，本文首次系统化探索文本领域的LLM驱动方案
2. **Gradient Matching方法**（Zhao et al., 2021; Maekawa et al., 2024/DiLM）：通过匹配梯度对齐真实数据与合成数据，DiLM是首个文本级优化方法，但本文证明其在离散文本上信息压缩效率有限
3. **Embedding-level Distillation**（Li & Li, 2021; Tao et al., 2024/DaLLM）：在嵌入空间优化后映射回文本，但缺乏可解释性和跨模型泛化能力，本文选择直接在文本层面操作
4. **Coreset Selection**（Sener & Savarese, 2018/K-Center; Paul et al., 2021/EL2N; Xia et al., 2023/Moderate; Pan et al., 2024/G-DIG）：选择真实子集而非生成合成数据，在极低数据预算（5-20 DPC）下性能远逊于CondenseLM
5. **LLM-driven Data Generation for Augmentation**（Long et al., 2024综述; Wang et al., 2023; Lee et al., 2024）：聚焦数据增强而非压缩，本文指出Error Extrapolation等框架过度强调罕见困难样本，不适合压缩任务
6. **Discretization方法**（Sucholutsky & Schonlau, 2021; Sahni & Patel, 2023）：将优化嵌入投影到最近词汇token，本文Table 4显示其信息损失严重，性能接近随机选择

## 局限性与未来方向
1. **任务范围受限**：当前仅适用于文本分类任务，论文自述可扩展至翻译、摘要、问答等更广任务
2. **未集成最新Prompt工程**：论文承认未采用最新prompt优化技术，认为集成后可进一步提升生成质量
3. **商业LLM成本**：主力实验使用GPT-4o，虽验证了小模型可行性，但成本仍是实际部署的障碍
4. **潜在预训练偏见传播**：LLM可能继承并传播web数据的偏见（如刻板印象），与伦理声明一致认为是领域共性挑战
5. **未来方向**：扩展至多语言和更多NLP任务、集成先进prompt技术、与去偏算法/数据过滤结合

## 研究启发与可借鉴点
1. **LDSC的特征提取-压缩思路可迁移至其他领域**：让LLM先识别判别性/普遍特征再生成紧凑样本的策略，可用于小规模数据场景下的数据蒸馏或质量过滤，值得借鉴到图像或其他模态任务
2. **Heavy Regularization构造代表性奖励信号的设计精巧**：用dropout=0.3+early stopping 1 epoch强制模型只学习常见可泛化模式，此思路可用于构建偏好常见模式而非边缘样本的质量评估器
3. **R_cov差分奖励设计可用于主动学习/数据选择**：用M_rep - M_cov的差值衡量"新信息"的创意，可迁移至课程学习或在线数据选择场景
4. **Best-of-N采样结合奖励筛选的框架设计通用**：多候选+奖励排序的选择策略可有效提升LLM生成质量，适用于需要多次生成的合成数据任务
5. **跨模型泛化评估的严谨性值得借鉴**：论文在BERT_base压缩后测试RoBERTa/XLNet，验证了方法不依赖于特定模型参数，此评估范式可作为文本压缩方法的标准化测试流程

## 关键术语表
**Dataset Condensation**：数据集压缩，指通过优化生成一个规模远小于原始数据集的合成数据集，使得在该合成数据上训练的模型性能接近在原始数据上训练的效果

**LDSC (LLMs-driven Subset Condensation)**：LLM驱动的子集压缩模块，让LLM从原始子集中提取判别性特征和普遍特征后生成更紧凑、信息密度更高且偏差更低的子集

**Reward Matching**：奖励匹配，利用代表性奖励模型和覆盖率奖励模型对生成样本打分并指导迭代压缩过程的机制

**Representability Reward Model**：代表性奖励模型，用重度正则化训练的分类模型，对常见可泛化模式给出高分，用于评估样本的代表性

**Coverage Reward Model**：覆盖率奖励模型，在已有压缩子集上训练的模型，与代表性模型预测差值衡量样本的新信息覆盖能力

**Data Per Class (DPC)**：每类数据点数，衡量压缩后数据集规模的常用指标，如10-DPC表示每类10个样本

**Best-of-N Sampling**：最佳N采样，从N个候选生成结果中按奖励得分选择最优者，常用于LLM输出优化

**Gradient Matching Loss**：梯度匹配损失，数据集压缩中常用的优化目标，通过最小化合成数据与真实数据训练梯度之间的差异来对齐学习行为

## 可复现要素
- **数据集**：SST-2、MNLI、AG News、IMDB（均为公开标准数据集）
- **代码开源**：论文未提及代码开源状态
- **权重开源**：未提及
- **关键超参**：
  - 生成模型：GPT-4o（也测试Qwen2.5-14B、Gemma2-9B）
  - 奖励模型：BERT_base
  - 子集压缩率：20%-50%（Table 3详列）
  - θ_rep=0.8，θ_cov=0.3，w=0.2，N=10候选
  - M_rep训练：AdamW，lr=1e-5，dropout=0.3，1 epoch，early stopping
  - M_cov训练：AdamW，lr=1e-4，dropout=0.1，200 steps
  - 评估训练：lr=1e-4，batch=64，200 steps
