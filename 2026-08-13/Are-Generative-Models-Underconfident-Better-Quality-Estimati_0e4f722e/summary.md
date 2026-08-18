---
title: "Are-Generative-Models-Underconfident-Better-Quality-Estimati"
source: https://aclanthology.org/2025.emnlp-main.166.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:41:18"
field: "自然语言生成质量评估"
keywords: ["Quality Estimation", "模型置信度", "生成式模型", "概率分布分析", "无监督质量估计", "主导token簇", "机器翻译", "语音翻译"]
innovations: ["揭示自由文本生成任务中模型概率 underconfident 现象及其与歧义性的关系", "提出 BOOSTEDPROB 方法，通过 jump-cut 启发式识别主导 token 簇并boost置信度，零额外复杂度", "系统性验证该方法在 ST/MT/Sum/QA 多任务上显著优于原始概率和熵基线，部分场景接近监督QE"]
benchmarks: ["Fleurs", "WMT22 General", "HJQE", "XSum", "GSM8K", "SciEx", "ParaCrawl"]
---

# 论文速读：Are Generative Models Underconfident? Better Quality Estimation with Boosted Model Probability

## 一句话总结
论文发现自由文本生成任务中存在"模型概率被低估（underconfident）"现象——当输入存在多解时，正确选项的概率质量被分散，导致单一 token 概率偏低，并不表示输出质量差。据此提出 BOOSTEDPROB，通过识别并合并"主导 token 簇"的总概率质量来提升 QE 估计效果，在多个任务上显著优于原始模型概率（平均 Pearson 相关 +0.194），且无需额外推理开销。

## 研究问题与动机
- **核心问题**：在推理阶段无 ground truth 时，如何低成本地估计生成模型的输出质量（Quality Estimation, QE）？
- **现有方法不足**：
  1. 以往研究聚焦模型"过度自信（overconfident）"问题，忽略了自由文本生成（如翻译、摘要）因歧义性导致的"低置信（underconfident）"现象。
  2. 监督式 QE（如 CometKiwi）依赖人工标注数据，仅适用于翻译等少数任务，泛化性差。
  3. 集合/集成式 QE（如 Monte Carlo entropy）需要多次生成，推理开销大；外部评分方法（如 Prism、LLM-as-a-Judge）需额外模型。
  4. 纯概率熵方法只考虑分布整体形状，未利用最终选定 token 的信息。

## 核心贡献（创新点）
1. **揭示 underconfident 现象**：首次在自由文本生成任务中系统证明，歧义性会导致概率质量分散到多个合法 token 上，使模型显得置信度不足，而与 epistemic uncertainty 区分开来。
2. **提出 BOOSTEDPROB 方法**：仅利用模型输出概率分布，通过启发式检测"主导 token 簇"并将簇内总概率质量作为该 token 的质量分，实现零额外复杂度改进。
3. **系统性实验验证**：覆盖 Speech Translation、Machine Translation、Summarization、Question Answering 四个任务与 7 个模型，平均提升 +0.194 Pearson 相关（vs. raw probability），部分场景接近或超越监督/集成基线。
4. **揭示模型质量与自评能力的正相关**：随着模型质量提升，BOOSTEDPROB 的 QE 性能也同步提升，说明改进生成能力有助于改善模型自我评估。

## 方法详解
- **主导 token 簇检测（Jump-cut 启发式）**：
  - 将输出步骤 t 处的词表概率分布 $\mathcal{P}$ 降序排列：$\mathcal{P}_{\text{sorted}} = (p_{(1)}, p_{(2)}, \ldots, p_{(|V|)})$。
  - 计算相邻概率之差：$\mathcal{P}_{\text{diff}} = \mathcal{P}_{\text{sorted}} - \text{Shift}(\mathcal{P}_{\text{sorted}})$。
  - 判断显著下降位置（同时满足相对与绝对阈值）：
    $$p_{(i)} - p_{(i+1)} > \max(p_{(i)} \times x\%, \; \epsilon)$$
  - 最后一个满足条件的位置 $c = \max\{i \mid \text{isSignificantDrop}_i = \text{True}\}$ 为切割点，$p_{(1)} \ldots p_{(c)}$ 属于主导簇，其余为非主导。
- **Token 级 QE 分数**：
  $$QE(w_{(i)}) = \begin{cases} p_{(i)}, & i > c \text{（非主导，用自身概率）} \\ \sum_{j=1}^{c} p_{(j)}, & i \leq c \text{（主导，用簇总概率质量）} \end{cases}$$
- **序列级 QE 分数**：对所有输出 token 的 QE 分取均值：$QE(Y) = \frac{1}{|Y|}\sum_{t=1}^{|Y|} QE(y_t)$。
- **超参数**：$x=30\%$，$\epsilon=0.005$，在 Whisper、DeltaLM、Tower 三个模型上鲁棒。

## 实验与结果
- **数据集**：Fleurs（ST，4 语言对）、ParaCrawl / WMT22 General / HJQE（MT）、XSum（Summarization）、GSM8K / SciEx（QA）。
- **模型**：Whisper Large V3、DeltaLM Large、NLLB (3.3B)、Tower (7B)、Bloomz (560M)、Llama 3.2 (3B)、Llama 3.3 Instruct (70B)。
- **评估指标**：段落级 Pearson 相关；token 级 MCC。
- **主要结果**：
  - 平均 Pearson 相关：Probability 0.083 < Entropy 0.208 < **BOOSTEDPROB 0.268**（相对原始概率 +0.194，相对熵 +0.065）。
  - **最强提升**：WMT22 General zh-en + DeltaLM，Entropy 仅 0.082，BOOSTEDPROB 达 **0.688**。
  - 部分场景下接近/超越更昂贵的方法：如 NLLB en-de 上 BOOSTEDPROB 0.525 vs. Monte Carlo 0.303；Prism 设置下 BOOSTEDPROB 将 gap 缩小至接近监督 QE（CometKiwi）。
  - 摘要和 QA 任务上整体分数较低（可能受 BART Score 伪 label 质量限制）。
  - 在 Prism 外部评分设置下，BOOSTEDPROB+NLLB 在 WMT22 en-de Worst 上达到 0.453，逼近监督 QE。
  - Word-level QE（HJQE）：BOOSTEDPROB+Prism en-de MCC=0.204，接近监督 QE 的 0.220。

## 相关工作脉络
1. **模型过自信（Overconfidence）**：Nguyen et al. (2015)、Li et al. (2021) 指出神经网络倾向于高估正确输出概率；本文反向指出歧义任务中存在低估问题，两者互补。
2. **概率熵 QE**：Fomicheva et al. (2020) 用整个概率分布熵评估质量；本文方法同时考虑最终选定 token 与分布形状，优于纯熵。
3. **监督 QE（CometKiwi 等）**：Rei et al. (2022) 的 CometKiwi 在翻译 QE 上表现强劲，但依赖人工 DA 标注且仅限翻译；本文方法零训练、通用性强。
4. **集成/采样式 QE**：Monte Carlo sequence entropy (Malinin & Gales, 2021) 和 Perturbation-based QE (Dinh & Niehues, 2023) 需多次推理；本文单次推理即可，效率更高。
5. **外部评分方法**：Prism (Thompson & Post, 2020) 用 MT 模型强制解码打分；本文可作为即插即用的改进层叠加在 Prism 之上。
6. **采样中的主导 token 策略**：top-k (Fan et al., 2018)、top-p、min-p (Nguyen et al., 2024) 等用于生成多样性；本文借用"主导簇"思想但服务于 QE 而非采样，且设计了更严格的 jump-cut 检测以避免误判极低概率 token。

## 局限性与未来方向
- **弱模型不适用**：对于质量很差的模型（如 Whisper Tiny），模型本身可能过度自信地分配高概率给错误 token， BOOSTEDPROB 会放大这种过自信（Appendix F），QE 性能反而下降。建议使用更强模型通过 Prism 方式打分。
- **低歧义任务效果有限**：ASR、多选题 QA 等任务主导簇 size 通常为 1，BOOSTEDPROB 退化为原始概率（Appendix G），增益不明显。
- **完全无监督 QE 的上限**：论文承认在存在监督 QE 模型的场景（如 MT、ST）中，无监督方法的价值受限，但在简单性和推理速度上有优势。
- **未来方向**：结合 weak/strong model 的混合策略；探索更精细的簇检测算法（当前 top-k 表现接近）；将方法推广至更多生成任务。

## 研究启发与可借鉴点
1. **歧义性分析框架**：将 aleatoric uncertainty（数据歧义）与 epistemic uncertainty（模型能力不足）区分开来，可为其他生成任务的质量评估提供分析视角。
2. **Jump-cut 启发式可复用**：基于概率分布的"显著下降点"检测方法简洁有效，可迁移到其他需要自动检测"主流选项簇"的场景（如多候选排序、不确定性感知解码）。
3. **强模型评测弱模型输出**：在弱模型自检失效时，用强模型做外部评分（Prism 设置）是简单有效的兜底方案，值得在工程实践中采用。
4. **模型质量与自评能力正相关**：实验发现模型翻译能力提升后其 QE 性能也同步提升，提示可以将 QE 优化纳入模型训练目标之一（quality-aware decoding）。
5. **超参数鲁棒性设计**：同时使用相对阈值（x%）和绝对阈值（ε）的双阈值设计，在多种模型/任务间泛化良好，是一种值得借鉴的参数设计模式。

## 关键术语表
- **Quality Estimation (QE)**：在无 ground truth 的情况下，对模型生成输出的质量进行估计的任务。
- **Dominant Token Cluster**：输出概率分布中概率质量显著集中的 token 集合，对应多个合法输出选项。
- **Aleatoric Uncertainty**：由数据本身歧义性引起的不确定性，区别于模型能力不足的 epistemic uncertainty。
- **Underconfident**：模型因概率质量分散到多个合法选项而表现出单 token 概率偏低的现象，不代表输出质量差。
- **BOOSTEDPROB**：本文提出的 QE 方法，通过将主导 token 簇的总概率质量作为质量分来缓解 underconfidence。
- **Jump-cut Heuristic**：通过检测概率分布排序后的显著下降点来自动识别主导 token 簇的启发式方法。
- **Prism**：用目标 MT 模型对候选翻译进行强制解码打分的外部 QE 方法，本文作为叠加框架使用。
- **Direct Assessment (DA)**：人工标注的质量分数（0-100），用于训练 CometKiwi 等监督 QE 模型。

## 可复现要素
- **数据集**：Fleurs（CC BY 4.0）、ParaCrawl（CC0）、WMT22 General（Apache 2.0）、HJQE、XSum（MIT）、GSM8K（MIT）、SciEx；均为公开数据集。
- **模型权重**：Whisper Large V3（Apache 2.0）、DeltaLM Large（MIT）、NLLB（CC BY NC 4.0）、Tower（CC BY NC 4.0）、Bloomz（BigScience RAIL）、Llama 3.2/3.3（Community License）；均可公开获取。
- **代码**：论文未提及开源代码仓库（论文标题注脚无代码链接，附录亦未提供 GitHub）。
- **关键超参**：x = 30%，ε = 0.005；序列聚合方式为 token 级 QE 分的均值。
