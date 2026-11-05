---
title: "CondenseLM-LLMs-driven-Text-Dataset-Condensation-via-Reward"
source: https://aclanthology.org/2025.emnlp-main.65.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:27:11"
field: "文本数据集压缩与数据效率"
keywords: ["dataset condensation", "text data distillation", "LLM-driven data generation", "reward matching", "coreset selection", "low-resource NLP"]
innovations: ["LLM驱动的迭代子集condensation范式（LDSC）突破优化型方法在离散文本上的信息压缩瓶颈", "双奖励模型（Representability+Coverage）奖励匹配机制实现模型无关的condensation指导", "Best-of-N采样结合阈值过滤的迭代框架兼顾效率与生成质量"]
benchmarks: ["SST-2", "MNLI", "AG News", "IMDB"]
---

# 论文速读：CondenseLM: LLMs-driven Text Dataset Condensation via Reward Matching

## 一句话总结
本文提出 **CondenseLM**，首个针对文本数据集condensation的LLM驱动范式，通过LLM驱动的子集压缩（LDSC）模块结合奖励匹配（Reward Matching）机制，突破了优化基文本condensation在信息压缩效率上的瓶颈，在SST-2、MNLI等四个数据集上显著超越现有方法及Coreset基线，同时大幅降低计算成本（从24小时缩短至15分钟内）。

## 研究问题与动机
- **文本级dataset condensation效果有限**：现有优化型condensation方法（如DiLM）在离散文本上难以通过梯度匹配损失有效压缩信息，在5–20 DPC低预算下仅比K-Center高出约0.4%，计算成本高但收益微薄。
- **嵌入级方法可解释性差且泛化弱**：Discretization等方法优化word embedding后强制映射到词汇表，导致信息丢失，且无法跨模型泛化。
- **优化型condensation在文本上的根本瓶颈**：连续像素空间的信息压缩能力无法直接迁移至离散token空间，gradient similarity实验显示嵌入级DiLM（97.98%）远高于文本级（92.97%），但仍不如K-Center（92.85%）。
- **实际应用场景受限**：边缘设备（移动/IoT）面临存储和计算资源约束，需要高效的数据压缩方案。

## 核心贡献（创新点）
1. **提出CondenseLM——首个兼顾有效性和效率的文本级dataset condensation框架**，通过LLM驱动替代传统的双层优化，将condensation时间从24+小时降至15分钟以内，性能提升4%以上。
2. **设计LDSC（LLM-driven Subset Condensation）模块**，将LLM先作为"编码器"提取判别性特征和普遍性特征，再作为"解码器"重组生成更紧凑且更均衡的样本；与Naive few-shot prompting的本质区别在于：后者直接生成数据无法保证信息压缩和偏差抑制。
3. **引入Reward Matching机制（含Representability和Coverage两个奖励模型）**，在每轮迭代中引导生成样本同时满足"代表原有数据分布中常见通用模式"和"补充已有子集未覆盖的模式"双重目标；与已有工作（如DiLM的梯度匹配）的本质区别在于：该方法不依赖特定模型参数，实现了模型无关（model-agnostic）的设计。
4. **提出迭代子集式condensation框架（Iterative Subset-wise Condensation）**，将全量condensation分解为多轮子集检索→压缩→聚合，通过Best-of-N采样选择最优候选子集，解决了LLM无法一次性处理大规模数据集的问题。
5. **系统性验证了跨模型泛化优势**，在不同测试模型（RoBERTa、BERT_large、XLNet）上均显著优于基线，尤其在架构差异大的XLNet上提升10%–11.2%，而DiLM泛化表现较差。

## 方法详解
**整体框架**：CondenseLM采用T轮迭代流程，每轮包含Retrieval Stage和Condensation Stage两个阶段，最终合成 $\mathcal{D}_{\text{syn}} = \bigcup_{i=1}^{T} S_{\text{syn}}^{(i)}$，其中 $N_{\text{syn}} = K' \cdot T$。

**LDSC模块**：
- **Step 1**：LLM分析选定的原始子集 $S_{\text{real}}^{(t)}$，识别其中的 **discriminative features**（对分类关键的判别特征，如情感倾向词）和 **universal features**（跨类共享的普遍特征，如电影类型、叙事风格）。
- **Step 2**：LLM重新生成更紧凑的子集（通常为原大小的20%–50%），要求每个样本融合多个判别特征，同时确保各类别的普遍特征分布均衡。
- 关键超参：压缩比20%–50%，候选生成数N=10。

**Reward Matching**：
- **Representability Reward Model $\mathcal{M}_{\text{rep}}$**：在原始数据集上用**重度正则化**（dropout=0.3，early stopping after 1 epoch）训练得到，偏好能准确反映通用/常见模式但难以处理困难样本的模型预测。奖励分数：$\mathcal{R}_{\text{rep}}(x_i, y_i) = \mathcal{M}_{\text{rep}}(x_i)_{y_i}$（模型对正确标签的置信度）。
- **Coverage Reward Model $\mathcal{M}_{\text{cov}}^{(t-1)}$**：在第t轮用已有合成子集 $\bigcup_{i=1}^{t-1} S_{\text{syn}}^{(i)}$ 训练，衡量已覆盖程度。奖励分数：$\mathcal{R}_{\text{cov}}(x_i, y_i) = \mathcal{M}_{\text{rep}}(x_i)_{y_i} - \mathcal{M}_{\text{cov}}^{(t-1)}(x_i)_{y_i}$，越高表示该样本引入了原始数据中存在但未被现有子集覆盖的新模式。
- **Retrieval Stage过滤**：保留同时满足 $\mathcal{R}_{\text{rep}} \geq \theta_{\text{rep}}$（0.8）和 $\mathcal{R}_{\text{cov}} \geq \theta_{\text{cov}}$（0.3）的样本，再用K-Center选取子集。
- **Condensation Stage Best-of-N**：对同一子集生成N个候选压缩子集，按加权分数 $\mathcal{R}_{\text{rep}} + w \cdot \mathcal{R}_{\text{cov}}$（w=0.2）选最优。

## 实验与结果
- **数据集**：SST-2（67.3k, 2类）、MNLI（392k, 3类）、AG News（120k, 4类）、IMDB（25k, 2类）。
- **评估设置**：压缩至5/10/20 DPC，使用 $\text{BERT}_{\text{base}}$ 训练200步（lr=1e-4, batch=64）。
- **主要结果（SST-2）**：
  - 5-DPC：CondenseLM **77.2±4.9** vs DiLM 72.5±5.9（+4.7%），vs K-Center 70.8±4.1
  - 10-DPC：**79.9±3.1** vs DiLM 76.3±4.6（+3.6%），vs K-Center 75.9±4.7
  - 20-DPC：**82.4±2.5** vs DiLM 80.3±2.8（+2.1%），vs K-Center 79.8±3.5
- **主要结果（MNLI 20-DPC）**：CondenseLM **51.3±1.6** vs DiLM 48.7±2.6（+2.6%），vs K-Center 45.3±3.0
- **最强结果**：在5-DPC下提升幅度最大（SST-2 +4.7%，MNLI +6.3%），低数据预算场景优势最为显著。
- **效率对比**：CondenseLM耗时<15分钟，DiLM需>24小时；GPU峰值显存占用也大幅降低。
- **跨模型泛化（20-DPC）**：在XLNet_base上MNLI达到**54.7±1.7**，比DiLM（44.7±2.7）高**10.0%**。
- **小模型适用性**：Qwen2.5-14B和Gemma2-9B在5-DPC下仍超越所有基线（SST-2至少+2.5%，MNLI至少+2.2%）。
- **去污染验证**：最大测试集重叠率均<40%，平均重叠率远低于训练集重叠率，排除了数据泄露的质疑。

## 相关工作脉络
1. **优化型dataset condensation**（Wang et al., 2018; Zhao et al., 2021）：图像领域的gradient matching、trajectory matching等方法；本文指出这些方法在离散文本上失效（gradient similarity下降），转而探索LLM驱动范式。
2. **DiLM**（Maekawa et al., 2024）：当前文本condensation SOTA，通过优化代理模型生成文本样本；本文证明其在SST-2上仅比K-Center高0.4%，且耗时>24小时，而CondenseLM更快且性能更高。
3. **Embedding-level condensation**（Li & Li, 2021; Sucholutsky & Schonlau, 2021; Sahni & Patel, 2023; Tao et al., 2024/DaLLME）：优化word embedding后用最近邻映射回token；本文指出此类方法存在严重信息损失（gradient similarity仅90.19%）且无跨模型泛化能力。
4. **Coreset selection**（K-Center, Herding, EL2N, Moderate, G-DIG）：传统子集选择方法；本文表明在极低端数据预算（5-20 DPC）下，现有先进coreset方法（EL2N/Moderate/G-DIG）退化严重，而CondenseLM保持优越性能。
5. **LLM-driven data generation/augmentation**（Long et al., 2024; Wang et al., 2023; Lee et al., 2024; Li et al., 2025）：已有的LLM数据生成/增强框架；本文通过实验（Table 9）证明直接套用error extrapolation框架在condensation任务上效果不佳，因为该框架过度强调困难样本而忽视compactness需求。

## 局限性与未来方向
- **任务范围受限**：目前仅适用于文本分类，推广到翻译、摘要、QA等任务需进一步研究。
- **未结合最新prompt engineering技巧**：LLM生成质量有优化空间，融入前沿prompt方法可能进一步提升效果。
- **商业LLM成本**：最佳结果使用GPT-4o，API调用成本较高；虽然小模型也能取得不错效果但仍有差距。
- **LLM预训练偏见风险**：生成的合成数据可能继承网络数据的偏见（如刻板印象），需要与去偏算法结合使用。
- **未来方向**：扩展到多语言、更多NLP任务；探索更高效的小规模开源LLM；结合去偏技术；应用于资源受限的边缘设备场景。

## 研究启发与可借鉴点
1. **LDSC的"特征分解-重组"范式**：将数据分解为判别性特征和普遍性特征再重新组合，这一思路可迁移到其他模态的数据压缩/蒸馏任务，或在序列到序列任务中探索。
2. **双奖励模型的设计**：Representability（基于重正则化模型的置信度）和Coverage（新旧模型预测差值）两阶段奖励设计精巧地平衡了"代表性"与"多样性"，可借鉴到数据选择、主动学习等场景。
3. **重度正则化作为奖励信号来源**：用heavy regularization训练的模型天然偏好常见通用模式、忽略困难样本，这种"能力受限"反而成为高质量奖励信号——这一反直觉设计值得在其他数据质量评估任务中尝试。
4. **Best-of-N采样与迭代子集聚合框架**：将大规模condensation分解为多轮小批量处理，每轮内部通过Best-of-N选择最优候选，这种分治策略可有效降低计算复杂度，适用于其他需要LLM参与的大规模数据处理任务。
5. **去污染验证实验设计**：通过词汇重叠率分析排除LLM预训练数据泄漏的质疑，这一验证流程可作为LLM驱动数据生成工作的标准检查项。

## 关键术语表
- **Dataset Condensation（数据集蒸馏/condensation）**：生成一个紧凑的合成数据集，使其训练出的模型性能接近使用原始全量数据集训练的效果。
- **Coreset Selection（核心集选择）**：从原始数据集中直接选取一个代表性子集，而非生成新样本。
- **LDSC（LLM-driven Subset Condensation）**：本文提出的子集condensation模块，利用LLM识别并重组判别性特征与普遍性特征来生成更紧凑且更少偏差的样本。
- **Reward Matching（奖励匹配）**：通过训练奖励模型对样本进行打分，在迭代condensation过程中引导生成样本同时最大化representability和coverage。
- **Representability Reward Model**：在原始数据集上经重度正则化训练的模型，用于评估样本对通用/常见模式的捕捉能力。
- **Coverage Reward Model**：在当前已合成子集上训练的模型，与representability模型配合衡量样本带来的新信息增益。
- **DPC（Data Per Class）**：每个类别包含的数据点数量，用于衡量低数据预算下的condensation效果。
- **Best-of-N Sampling**：从多个候选生成结果中选择综合奖励分数最高的一个，本文用于在Condensation Stage筛选最优子集。

## 可复现要素
- **数据集**：SST-2、MNLI、AG News、IMDB——均为公开数据集。
- **代码/权重**：论文未提及开源代码或预训练权重。
- **关键超参**：
  - 生成模型：GPT-4o；奖励模型：$\text{BERT}_{\text{base}}$
  - $\mathcal{M}_{\text{rep}}$：dropout=0.3，early stopping after 1 epoch
  - $\mathcal{M}_{\text{cov}}$：dropout=0.1，训练200 steps，lr=1e-4
  - $\theta_{\text{rep}}$=0.8，$\theta_{\text{cov}}$=0.3，w=0.2
  - 子集压缩比：20%–50%（视数据集而定）
  - Best-of-N候选数：N=10
  - 训练设置：AdamW，lr=1e-4，batch=64，200 steps
