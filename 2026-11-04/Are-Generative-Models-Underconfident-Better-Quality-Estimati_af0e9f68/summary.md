---
title: "Are-Generative-Models-Underconfident-Better-Quality-Estimati"
source: https://aclanthology.org/2025.emnlp-main.166.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:25:20"
field: "质量估计与模型不确定性"
keywords: ["Quality Estimation", "Model Probability", "Underconfidence", "Dominant Tokens", "Speech Translation", "Machine Translation"]
innovations: ["发现自由文本生成中模型概率的underconfidence现象并提出形式化分析", "提出BOOSTEDPROB方法，通过跳变检测聚合主tokens概率质量以提升QE效果", "揭示模型质量与其自评估能力之间的正相关关系"]
benchmarks: ["Fleurs", "WMT22 General", "HJQE", "XSum", "GSM8K", "SciEx"]
---

# 论文速读：Are-Generative-Models-Underconfident-Better-Quality-Estimati

## 一句话总结
论文发现生成模型在自由文本生成（如翻译、摘要）中存在**概率"过于保守"**（underconfident）的现象——当存在多个合理输出选项时，概率质量被分散，导致正确token的单点概率偏低。为此提出 **BOOSTEDPROB** 方法，通过识别并聚合主tokens（dominant tokens）的概率质量来改进质量估计，在多个任务上相比原始模型概率平均提升 **+0.194**（Pearson相关系数），且在某些设置下接近或超越监督/集成类方法。

---

## 研究问题与动机
1. **模型概率并非总是"过自信"**：以往研究主要关注模型概率的 overconfidence 问题，但本文指出在自由文本生成任务中，模型概率反而可能表现出 **underconfidence**。
2. **歧义任务导致概率分散**：在 Speech Translation、Machine Translation 等歧义任务中，同一位置可能存在多个合理输出选项，概率质量被分散到多个正确token上，使得每个正确token的单点概率偏低，但这并不代表输出质量差。
3. **现有概率基线的不足**：Raw Probability 只看最终选中token的概率，忽略了概率分布的整体结构；Probability Entropy 只看整体熵，却忽略了最终被选中的token。两者均未能充分利用概率分布信息。
4. **需求低成本的无参考质量估计方法**：监督 QE 依赖大量人工标注，仅适用于翻译等少数任务；集成/Ensemble 方法计算成本高。亟需一种**低成本、通用、仅需模型概率分布**的质量估计方法。

---

## 核心贡献（创新点）
1. **发现并分析模型概率在自由文本生成中的 underconfidence 现象**：通过自动分析与人工标注验证，歧义任务中每个输出步往往存在多个合法token，导致概率分散，这是与 epistemic uncertainty 不同的 aleatoric uncertainty 的来源。
2. **提出 BOOSTEDPROB 方法**：通过检测概率分布中的显著跳变来识别 dominant tokens，并将其概率质量聚合后作为质量分数，无需额外训练或推理开销。
3. **系统评估证明 BOOSTEDPROB 优于原始概率和熵基线**：在 ST、MT、Summarization、QA 等多个任务和模型上，平均 Pearson 相关系数相比 Raw Probability 提升 **+0.194**，相比 Entropy 提升 **+0.065**。
4. **揭示了模型质量与自评估能力的正相关关系**：使用 BOOSTEDPROB 时，模型翻译质量越高，其自我质量估计能力也越强（弱模型除外）。

---

## 方法详解

### 核心思路
当输出token属于 **dominant cluster**（即概率分布中存在多个显著token）时，不单独使用该token的概率作为质量分数，而是使用**整个 dominant cluster 的总概率质量**作为质量分数，从而缓解 underconfidence。

### 步骤一：识别 Dominant Tokens（跳变检测启发式）

对输出步 $t$ 的概率分布 $\mathcal{P} = (p_1, p_2, \ldots, p_{|V|})$ 排序得到 $\mathcal{P}_{\text{sorted}}$，计算相邻概率的差值 $\mathcal{P}_{\text{diff}}$，定义**显著跳变**需同时满足两个条件：

1. **相对条件**：$p_{(i)} - p_{(i+1)} > p_{(i)} \times x\%$
2. **绝对条件**：$p_{(i)} - p_{(i+1)} > \epsilon$

最后一个满足条件的跳变位置 $c$ 即为切割点，位置 $1 \sim c$ 对应的token为 dominant tokens。

### 步骤二：提取 Token 级质量分数

$$
QE(w_{(i)}) = 
\begin{cases}
p_{(i)}, & \text{if } i > c \quad (\text{非主token，使用自身概率}) \\
\displaystyle\sum_{j=1}^{c} p_{(j)}, & \text{if } i \leq c \quad (\text{主token，使用cluster总概率})
\end{cases}
$$

### 步骤三：序列级质量估计

对输出序列 $Y = y_1, \ldots, y_{|Y|}$，取各 token 质量分数的**平均值**：

$$
QE(Y) = \frac{1}{|Y|} \sum_{t=1}^{|Y|} QE(y_t)
$$

### 超参数
- $x = 30\%$（相对跳变阈值）
- $\epsilon = 0.005$（绝对跳变阈值）
- 在 Whisper、DeltaLM Large、Tower 三个模型上验证，结果稳健。

---

## 实验与结果

### 数据集
| 任务 | 数据集 | 样本数 | 语言 |
|------|--------|--------|------|
| ST | Fleurs | 350 | vi-en, de-en, es-en, zh-en |
| MT | ParaCrawl | 5000 | en-de, zh-en |
| MT | WMT22 General | 2000 | en-de, zh-en |
| MT | HJQE | 1000 | en-de, en-zh |
| Summarization | XSum | 3000 | en |
| QA (Math) | GSM8K | 3000 | en |
| QA (Exam) | SciEx | 1120 | en, de |

### 模型
Whisper Large V3 (1.55B), DeltaLM Large (1.37B), NLLB (3.3B), Tower (7B), Bloomz (560M), Llama 3.2 (3B), Llama 3.3 (70B)

### 主要结果（Pearson Correlation）
- **平均性能**（Table 3）：Raw Probability = 0.083，Entropy = 0.208，**BOOSTEDPROB = 0.268**
- **最佳单项**：NLLB 在 WMT22 General zh-en 上达到 **0.688**（远超 Raw Probability 0.253 和 Entropy 0.082）
- **显著提升案例**：Whisper 在 Fleurs zh-en 上从 0.053 提升至 0.424；DeltaLM 在 WMT22 zh-en 上从 0.005 提升至 0.240

### 与高成本基线对比（Table 4）
- 在 MT en-de 上，BOOSTEDPROB（0.525）**超越** Monte Carlo Entropy（0.303）和 LLM Self-Judge
- 在 Summarization 上，与 Monte Carlo Entropy 持平或接近
- 在 Word-level QE（HJQE）上，BOOSTEDPROB（MCC=0.164）接近 Supervised QE（MCC=0.239），且仅需单次推理

### 与 Prism 结合（Table 5）
- BOOSTEDPROB + Prism 在 NLLB en-de Worst MT 上达到 **0.453**，接近 Supervised QE 的 0.453
- 缩小了原始 Prism 与监督 QE 之间的差距

### 弱模型问题（Section 6.4）
- Whisper Tiny 等极弱模型使用 BOOSTEDPROB 后 QE 性能反而下降，原因是弱模型存在 hallucination 且过度自信，BOOSTEDPROB 放大了这一问题

---

## 相关工作脉络
1. **模型概率过自信问题**（Nguyen et al., 2015; Li et al., 2021）：以往研究聚焦于模型过度自信，本文补充指出自由文本生成中存在相反的 underconfidence 现象。
2. **基于熵的质量估计**（Fomicheva et al., 2020）：Entropy 考虑了整个分布但忽略了最终选中token，BOOSTEDPROB 兼顾两者。
3. **Monte Carlo Sequence Entropy**（Malinin & Gales, 2021; Kuhn et al., 2023）：需要多次采样，计算成本高；BOOSTEDPROB 单次推理即可。
4. **Perturbation-based QE**（Dinh & Niehues, 2023）：通过对输入施加扰动来评估质量，需要额外推理；BOOSTEDPROB 无此开销。
5. **Supervised QE（CometKiwi）**（Rei et al., 2022）：依赖人工标注数据，仅适用于翻译；BOOSTEDPROB 通用且无需训练。
6. **Prism**（Thompson & Post, 2020）：用目标语言模型对翻译强制解码评分；本文证明在其基础上引入 BOOSTEDPROB 可显著提升质量。
7. **采样策略中的 dominant token 识别**（top-k, top-p, ε-cut, η-cut, min-p）：这些方法面向生成多样性，而非质量估计；本文将其思想迁移至 QE 并设计了更严格的跳变检测方法。

---

## 局限性与未来方向
1. **弱模型不适用**：BOOSTEDPROB 会放大弱模型的 overconfidence（如 Whisper Tiny），建议在弱模型场景中使用强模型进行 Prism 式评分。
2. **低歧义任务效果有限**：在 ASR 或多项选择题 QA 等歧义性低的任务中，dominant cluster 大小通常为1，BOOSTEDPROB 与原始概率无显著差异。
3. **无监督 QE 的天花板**：即使最优结果（Pearson≈0.688）距实用仍有距离，需与监督 QE 结合或作为辅助信号。
4. **未来方向**：探索与监督 QE 的结合（如用于数据筛选）、扩展到其他歧义任务、研究更精细的概率分布建模。

---

## 研究启发与可借鉴点
1. **跳变检测启发式识别 dominant tokens**：利用排序概率序列中的相对+绝对双重阈值检测显著跳变，方法简洁有效，可迁移至其他需要识别"显著token集合"的场景（如不确定度估计、关键token定位）。
2. **将 aleatoric uncertainty 与 epistemic uncertainty 区分纳入 QE 设计**：本文对两类不确定性的形式化分析（Appendix B）为后续研究提供了理论框架。
3. **实验设计严谨**：同时使用自动伪标签（XCOMET-XL、BARTScore）和人工标注（WMT22 General、HJQE）评估，多维度验证方法有效性。
4. **可结合现有流水线**：BOOSTEDPROB 可无缝嵌入 Prism 框架或早期退出（early exiting）系统，作为低成本的置信度信号。
5. **启示团队方向**：可将此思想应用于本团队的翻译/摘要质量评估模块，或在模型校准（model calibration）研究中借鉴其概率分析方法。

---

## 关键术语表
**Quality Estimation (QE)**：在推理时评估模型输出质量的任务，无需人工标注或参考译文。

**Underconfidence（过于保守）**：指模型因概率分散到多个正确选项而导致单点概率偏低的现象，与常见的 overconfidence 相反。

**Dominant Tokens / Dominant Cluster**：输出概率分布中概率质量显著集中的token集合，通过跳变检测自动识别。

**Aleatoric Uncertainty（偶然不确定性）**：来源于数据本身固有的歧义性（如多种合法翻译），与模型能力不足的 epistemic uncertainty 不同。

**BOOSTEDPROB**：本文提出的 QE 方法，通过将 dominant tokens 的概率质量聚合来提升质量估计准确性。

**Prism**：使用目标语言模型对翻译结果进行强制解码评分的无参考质量估计方法。

**Pearson Correlation**：本文评估 QE 质量的主要指标，衡量 QE 分数与真实质量标签之间的线性相关程度。

**MCC (Matthews Correlation Coefficient)**：用于 token 级二分类 QE 的评估指标（HJQE 数据集）。

---

## 可复现要素
- **数据集**：Fleurs（CC BY 4.0）、ParaCrawl（CC0）、WMT22 General（Apache 2.0）、HJQE、XSum（MIT）、GSM8K（MIT）、SciEx 均已公开
- **代码**：论文未提供开源代码链接
- **模型权重**：Whisper、DeltaLM、NLLB、Tower、Bloomz、Llama 系列均为开源模型，可通过 HuggingFace / Fairseq 获取
- **关键超参数**：$x = 30\%$，$\epsilon = 0.005$；beam size=4（小模型），greedy（大模型）
- **实验环境**：A100 GPU 40GB，HuggingFace Transformers / Fairseq / LM-Polygraph
