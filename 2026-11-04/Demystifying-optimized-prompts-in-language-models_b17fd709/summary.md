---
title: "Demystifying-optimized-prompts-in-language-models"
source: https://aclanthology.org/2025.emnlp-main.147.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:28:22"
field: "大语言模型可解释性与安全性"
keywords: ["prompt optimization", "GCG", "mechanistic interpretability", "sparse probing", "jailbreak", "language model safety", "adversarial prompts"]
innovations: ["首次系统性跨模型分析优化提示的token组成与词频分布特征", "通过稀疏探针证明优化提示与自然语言提示在激活空间可高准确率分离", "揭示层间KL散度演化路径并发现feature ablation与token ablation敏感性解耦"]
benchmarks: ["Tiny Stories", "Alpaca", "WikiText-103", "OpenHermes-2.5", "Dolly-15k"]
---

# 论文速读：Demystifying-optimized-prompts-in-language-models

## 一句话总结
本文系统性研究了通过 GCG 离散优化生成的"机器提示"（optimized prompts）的组成特征与内部处理机制，发现此类提示主要由标点和训练数据中的稀有名词 token 构成，且其内部激活表示与自然语言提示存在可分离的显著差异，但 ablating 这些区分特征并不显著削弱提示效果。

## 研究问题与动机
- **核心问题**：离散优化生成的提示（如 GCG jailbreak prompts）看似不可解释的乱码，语言模型究竟是如何解析并从中构建预测的？
- **安全背景**：优化提示常用于 adversarial jailbreak，绕过指令对齐机制产生有害输出；理解其机制对提升 LM 鲁棒性与安全性至关重要。
- **现有方法不足**： prior work 多关注提示的 adversarial 攻击效果或跨模型可迁移性，缺乏对优化提示 token 组成、词频分布及内部表征结构的系统性分析。
- **开放问题**：优化提示是否真的"不可解释"？其在模型内部是否依赖独特的子空间或特征路径？

## 核心贡献（创新点）
1. **首次系统性跨模型分析优化提示的 token 组成**：发现 rank-1 最具影响力 token 以标点和名词为主，且该模式在自然语言提示中同样存在，说明并非优化提示独有。
2. **揭示优化提示依赖训练数据中的稀有 token（corpus-rare tokens）**：其词频分布显著偏离 Zipf 幂律，归一化熵更高（0.89–0.93 vs 0.71–0.80），可能与"glitch tokens"的强优化信号有关。
3. **通过稀疏探针分类器证明优化提示与自然语言提示在激活空间可高准确率分离**：即使在高稀疏度约束下，仅用 top-k 特征即可区分两者，而 baseline（NL vs NL、NL vs random）接近随机。
4. **发现因果干预结果与直觉相悖**：ablate 由 MMD 识别的 top-k 区分特征后，优化提示的效果并未比自然语言提示更显著下降，说明 token 级敏感性与内部表征敏感性并非直接对应。
5. **绘制层间 KL 散度演化路径**：指令微调模型表现出"早期层相似 → 中间层渐次发散 → 最后几层重新收敛"的一致轨迹，最后几层对功能等效性起关键作用。

## 方法详解
- **优化框架**：采用 "evil twins" 框架 + GCG（Greedy Coordinate Gradient），目标是最小化原始自然提示 $p^*$ 与优化提示 $p$ 之间的 KL 散度：
  $$d_{KL}(p^*||p) = \frac{1}{n}\sum_{i=1}^n \log P_{LM}(d_i|p^*) - \log P_{LM}(d_i|p)$$
  其中 $d_i$ 是从 $p^*$ 采样的续写序列；$d_{KL}=0$ 当且仅当两提示功能等价。
- **Token 影响力评分**：对优化提示中每个 token $i$，计算移除后的 KL 变化 $s_i = d_{KL}(p||p_{-i})$，按得分降序排列得 token rank。
- **词性标注与词频分析**：使用 spaCy 做 POS  tagging；比较 token 在预训练语料中的频率，绘制 Zipf 图和 CDF。
- **稀疏探针（Sparse Probing）**：在每层取最后一 token 的激活，用 MMD 排序特征重要性 $\Delta_j^{(\ell)}$，选取 top-k 特征训练 logistic regression 分类器，评估区分优化/NL 提示的能力。
- **因果干预（Feature Ablation）**：将 top-k 特征维度置零后，测量原始-优化提示对的 top-10 token 预测重叠率：
  $$\mathbb{P}_{LM}^{(\ell,k)}(\cdot|p) = \mathbb{P}_{LM}(\cdot|p; h_i^{(\ell)}=0, \; i\in\mathcal{I}_k)$$
- **层间 KL 散度追踪**：将每层最后一 token 的隐藏状态经 LayerNorm + LM head 投影回词表空间，计算每层 $d_{KL}^{(\ell)}(p^*||p)$，观察表示演化轨迹。

## 实验与结果
- **数据集**：Tiny Stories（用于自训练 word-level 模型）、Alpaca、WikiText-103、OpenHermes-2.5、Dolly-15k。
- **模型**：18 个开源模型，覆盖 Gemma-2、Llama-3.2、Pythia、Qwen2.5、SmolLM2 系列，参数量从 135M 到 8B，含 base 与 instruct 变体。
- **主要结果**：
  - **Token 组成**：noun 在所有 rank 上占比最高（rank-1 在 base 模型中为 punctuation 主导）；自然语言提示呈现相同趋势。
  - **词频分布**：优化提示归一化熵明显更高（word-stories: 0.8968 vs 0.7102；Pythia: 0.9338 vs 0.7988），偏离 Zipf 分布。
  - **探针准确率**：在 top-10 特征下分类准确率显著高于 baseline（接近随机），图 5 显示多层均保持高区分能力。
  - **Feature ablation**：图 6 显示对 top-k 特征的 ablation 对两类提示的影响差异不大，仅第一层和最后一层表现更明显。
  - **层间 KL 散度**：图 7 显示 instruct 模型普遍呈现"early similarity → mid divergence → late convergence"路径，最后几层是功能等效的关键。
- **最强结果**：稀疏探针在部分模型的某层 top-10 特征上达到接近完美的区分准确率；层间 KL 分析揭示了跨模型一致的表示演化模式。

## 相关工作脉络
- **GCG (Zou et al., 2023b)**：本文优化方法的基础，用于生成 adversarial suffix；本文聚焦其内部机制而非攻击效果。
- **Evil twins (Melamed et al., 2024)**：提供本文使用的 KL 散度优化框架；本文在此基础上深入分析生成提示的性质。
- **Rakotonirina et al. (2024)**（并发工作）：发现 GCG 提示含大量"filler" token 且效果依赖最后一个 token；本文进一步从词频分布、内部表征和层间演化角度补充机制解释。
- **Sparse probing (Gurnee et al., 2023)**：本文借用其 MMD 特征排序 + 稀疏 logistic regression 管线，首次应用于区分 optimized vs natural prompts。
- **Glitch tokens (Rumbelow & Watkins, 2023; Li et al., 2024; Land & Bartolo, 2024)**：训练不足的 token 易产生异常输出；本文将其与优化提示的高稀有 token 比例关联。
- **Mechanistic interpretability (Elhage et al., 2021; Geva et al., 2021, 2022)**：本文延续"从激活空间理解预测构建"的研究传统，将 logit lens / 线性探针思路应用于优化提示场景。

## 局限性与未来方向
- **仅覆盖 GCG / evil twins 框架**：未扩展到 FLRT、Auto-DAN 等其他离散优化方法，结论的普适性有待验证。
- **模型规模上限 8B**：受 GPU 显存限制，未分析更大模型（如 Llama-70B）上的趋势，scaling behavior 未知。
- **未区分 benign 与 malicious 优化提示**：当前分析仅基于功能等效性，未能判断优化提示的意图属性。
- **词性分析基于 word-level tokenizer**：虽提升可解释性，但不同于工业界常用的 BPE tokenizer，结论在子词层面是否成立未验证。

## 研究启发与可借鉴点
- **探针分类器作为 jailbreak 检测器**：可在中间层部署轻量 linear probe，实时识别疑似优化提示并触发告警，具有工程落地潜力。
- **层间 KL 散度演化可作为诊断工具**：用于快速评估任意"异常"提示（如 semantic poison、style transfer prompts）在模型内部的表示偏离程度。
- **"稀有 token + 标点驱动"模式可迁移**：在 prompt tuning / prefix tuning 研究中，可主动引入低频次 token 作为可控扰动源，测试模型鲁棒性。
- **特征 ablation 与 token ablation 的解耦发现**：提示内部表征敏感性与表面 token 敏感性并不完全对应，提醒我们在做 mechanistic analysis 时避免单一视角推断。
- **与本团队方向结合机会**：若团队研究 prompt robustness 或 red-teaming，可将本文的层间 KL 轨迹和探针框架集成进自动化攻击评估 pipeline。

## 关键术语表
- **Optimized prompts**：通过梯度-based 离散优化（如 GCG）生成的、在功能上等价于自然语言提示但表面不可解释的 token 序列。
- **GCG（Greedy Coordinate Gradient）**：逐 token 迭代选择梯度方向上最优替换的离散提示优化算法，广泛用于 LLM jailbreak。
- **Evil twins 框架**：以 KL 散度衡量原始提示与优化提示的功能相似性，并通过采样续写序列进行优化目标近似。
- **Influence score**：移除 prompt 中某一 token 后 KL 散度的变化量，衡量该 token 对提示功能的贡献程度。
- **Corpus-rare tokens**：在预训练语料中出现频率极低的 token，可能与 "glitch tokens" 现象相关。
- **Sparse probing**：在模型各层激活中选取重要性最高的 top-k 特征训练线性分类器，用于探测特定概念的内部表征。
- **MMD（Maximum Mean Discrepancy）**：用于衡量两组激活分布在每个特征维度上的均值差异，本文用于特征排序。
- **Layer-wise KL divergence**：将每层输出投影回词表空间后计算的 KL 散度，用于追踪提示表示在网络中的演化轨迹。

## 可复现要素
- **数据集**：Tiny Stories（公开）、Alpaca（公开）、WikiText-103（公开）、OpenHermes-2.5（公开）、Dolly-15k（公开）。
- **代码/权重**：模型为开源（Gemma-2、Llama-3.2、Pythia、Qwen2.5、SmolLM2 系列均开源）；代码开源声明论文未明确提及，但 evil twins 框架代码已随 Melamed et al. (2024) 发布。
- **关键超参**：GCG 优化步数 500 步，早停阈值 $d_{KL} \le 5.0$，最终过滤阈值 $d_{KL} \le 10.0$；探针分类器为 logistic regression（$l_2$ 惩罚，liblinear solver，迭代 100 次，收敛容差 $1e^{-4}$）；Tiny Stories 模型 6 层、hidden dim 512、8 attention heads、batch size 64、lr $6\times10^{-4}$、warmup 500、3 epochs。
- **硬件**：离散优化使用 8x A100 节点；Tiny Stories 训练使用单张 NVIDIA RTX 6000 Ada。
