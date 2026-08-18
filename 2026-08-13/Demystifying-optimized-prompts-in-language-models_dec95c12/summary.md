---
title: "Demystifying-optimized-prompts-in-language-models"
source: https://aclanthology.org/2025.emnlp-main.147.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:46:53"
field: "大语言模型安全与可解释性"
keywords: ["prompt optimization", "language model interpretability", "adversarial prompts", "sparse probing", "jailbreak detection"]
innovations: ["首次系统分析18个模型上GCG优化prompt的token组成与内部表征", "提出基于稀疏探测的分类器区分优化prompt与天然prompt", "揭示优化prompt依赖稀有token和标点的构成规律"]
benchmarks: ["Tiny Stories", "Alpaca", "WikiText-103", "OpenHermes-2.5", "Dolly-15k"]
---

# 论文速读：Demystifying-optimized-prompts-in-language-models

## 一句话总结
本文系统研究了通过离散优化（GCG）生成的"evil twin" prompt 的构成特征与内部表示机制，发现优化后的 prompt 主要由稀有名词和标点组成，且在模型内部激活空间中与天然语言存在显著可区分的表征模式。

## 研究问题与动机
- **核心问题**：离散优化的 prompt 是否真的"不可解释"？模型如何解析和处理这类看似乱码的输入？
- **安全背景**：GCG 优化 prompt 常被用于对抗性攻击（jailbreak），使对齐后的 LLM 产生有害输出，理解其机制对保障模型安全至关重要。
- **现有不足**：先前工作多从攻击有效性角度研究，缺乏对优化 prompt 组成特征和内部处理机制的系统性分析；并发工作（Rakotonirina et al., 2024）仅关注 filler token 和末尾 token 依赖，未深入表征层面。
- **研究缺口**：缺乏跨模型族的系统性对比分析，以及从 token 分布、语法类别、内部激活等多维度的综合表征研究。

## 核心贡献（创新点）
- **首次系统性跨模型分析**：在 18 个开源模型（含 base 和 instruct 变体）上系统分析 GCG 优化 prompt 的构成与表征，填补了该领域缺乏大规模实证研究的空白。
- **揭示优化 prompt 的 token 组成规律**：发现优化 prompt 最 influential 的 token 主要由标点和名词构成，且整体 token 分布显著偏离 Zipf 定律，依赖更多训练数据中的稀有 token（corpus-rare tokens）。
- **提出基于稀疏探测的分类方法**：通过最大均值差异（MMD）筛选关键激活特征，训练稀疏线性分类器可高精度区分优化 prompt 与天然 prompt，证明两者在内部表征层面存在本质差异。
- **揭示层间表征演化路径**：通过逐层 KL 散度分析发现，指令微调模型在中间层逐步发散、最后几层重新收敛的表征路径，表明后期层对维持功能相似性起关键作用。

## 方法详解
- **优化框架**：采用 "evil twins" 框架，基于 GCG 算法生成与原始 prompt 功能等价但形式不同的离散优化 prompt，目标是最小化 KL 散度 $d_{KL}(p^*||p)$。
- **Token 影响力评估**：定义 token 影响分数 $s_i = d_{KL}(p||p_{-i})$，通过移除单个 token 并衡量 KL 散度变化来量化各 token 对 prompt 功能的贡献。
- **词性标注与分布分析**：使用 spaCy 进行词性标注，分析不同 rank 位置 token 的语法类别分布；通过 Zipf 图和归一化熵评估 token 频率分布偏离程度。
- **稀疏探测分类器**：在每个 transformer 层，使用 MMD 计算自然 prompt 与优化 prompt 激活特征的平均差异 $\Delta_j^{(\ell)}$，选取 top-k 特征训练逻辑回归分类器进行区分。
- **因果干预实验**：逐层 zero-out 识别出的 top-k 重要特征维度，测量干预前后 top-10 token 预测重叠度变化，验证关键特征的重要性。
- **逐层 KL 散度分析**：将每层最后一个 token 的隐藏状态经 LayerNorm 和 LM head 投影回词汇空间，计算优化 prompt 与原始 prompt 在各层的 KL 散度 $d_{KL}^{(\ell)}(p^*||p)$。

## 实验与结果
- **数据集**：Tiny Stories（合成儿童故事）、Alpaca、WikiText-103、OpenHermes-2.5、Dolly-15k。
- **模型规模**：18 个开源模型，涵盖 SmolLM2、Qwen2.5、Llama-3.2、Pythia、Gemma 等家族，参数范围 135M–8B。
- **关键结果**：
  - 最 influential token（rank 1）效应显著，高 rank token 影响极小，与并发研究结论一致。
  - 名词在所有 rank 位置均占比最高；rank 1 以标点为主（自回归模型末尾 token 影响大）。
  - 优化 prompt 的归一化熵更高（word-stories: 0.8968 vs 0.7102；Pythia: 0.9338 vs 0.7988），显著偏离 Zipf 分布。
  - 稀疏探测分类器在高稀疏度下仍可实现高精度分类（图 5），而基线比较（自然 vs 随机）接近随机水平。
  - 特征消融实验显示：top-k 特征消融对两类 prompt 影响无显著差异， contradict 了"优化 prompt 依赖独特子空间"的假设。
  - 逐层 KL 散度显示：指令微调模型早期层表征相似、中间层发散、最后几层重新收敛；base 模型早期层即有较大发散。
- **最强结果**：稀疏探测分类器在 Pythia-1.4B 等模型上 achieving 高精度区分，证明了优化 prompt 内部表征的可检测性。

## 相关工作脉络
- **HotFlip (Ebrahimi et al., 2018)**：字符级对抗示例生成的开创性工作，基于梯度进行引导 token 替换；本文扩展至 decoder LM 的离散优化。
- **AutoPrompt (Shin et al., 2020)**：在 BERT 等 masked LM 上追加 trigger tokens 改进下游任务；GCG 是其向 decoder 架构的延伸。
- **GCG (Zou et al., 2023b)**：现代 adversarial prompt 生成标准方法，本文以此为优化基准，系统分析其产物特性。
- **Kervadec et al. (2023)**：首次考察 OPT 模型中优化 prompt 的 attention 模式和激活，发现 distinct pathways；本文在其基础上扩展到 18 个模型并从 token 组成和稀疏探针角度深化。
- **Rakotonirina et al. (2024)**：并发工作发现优化 prompt 含 filler tokens 且依赖末尾 token；本文与其互补，增加了词性分析、分布分析和内部表征分析。
- **Gurnee et al. (2023)**：稀疏探测方法基础；本文将其应用于优化 prompt vs 天然 prompt 的区分任务。

## 局限性与未来方向
- **优化方法局限**：仅使用 GCG 和 "evil twins" 框架，未扩展到 FLRT、Auto-DAN 等其他离散优化技术。
- **模型规模限制**：受计算资源约束，仅分析到 8B 参数模型，未研究更大规模模型的 scaling 效应。
- ** benign vs malicious 区分**：未探讨如何区分良性优化 prompt 与恶意 jailbreak prompt，需进一步研究。
- **跨模型泛化**：未系统分析 "universally transferable" 优化 prompt 在不同模型间的表征差异。
- **未来方向**：扩展到更大模型、更多优化算法；探索安全检测应用（在中间层激活 classifiers 预警可疑 prompt）。

## 研究启发与可借鉴点
- **稀疏探针作为安全检测工具**：可在模型中间层部署轻量分类器，实时检测优化 prompt 的输入，为 LLM 安全防护提供新思路。
- **词级分词器的价值**：使用 word-level tokenizer 替代 BPE 使优化 prompt 的 token 级分析更直观，值得在类似研究中采用。
- **逐层 KL 散度可视化**：通过投影回词汇空间计算逐层 KL 散度，可直观展示表征演化路径，是一种有效的可解释性分析工具。
- **rare token 敏感性的利用**：优化 prompt 依赖 corpus-rare tokens 的现象提示，加强 rare token 训练或对其进行正则化可能提升模型鲁棒性。
- **多模型族对比范式**：跨 18 个模型的统一分析框架为后续研究提供了可复用的实验范式。

## 关键术语表
**GCG (Greedy Coordinate Gradient)**：一种基于梯度的离散 prompt 优化算法，通过迭代替换 token 中的坐标来最小化目标损失。
**Evil Twins**：与原始 prompt 功能等价但形式完全不同的优化 prompt，由 Melamed et al. (2024) 提出。
**稀疏探测 (Sparse Probing)**：通过在模型各层激活中筛选少数高贡献特征训练分类器，以探测特定概念的内部表征。
**Corpus-rare tokens**：在训练语料中频率极低的 token，可能因训练不足而对优化过程产生更强信号。
**KL 散度 (KL Divergence)**：衡量两个概率分布差异的指标，本文用于量化优化 prompt 与原始 prompt 的功能相似性。
**MMD (Maximum Mean Discrepancy)**：用于衡量两组激活特征分布差异的统计量，本文用于排序特征重要性。
**Jailbreak**：通过对抗性 prompt 绕过 LLM 安全对齐机制，诱导模型产生有害输出的攻击方式。
**Tokenizer**：将文本分割为 token 的模块，本文比较了 BPE 和 word-level 两种分词策略。

## 可复现要素
- **数据集**：Tiny Stories（公开）、Alpaca（公开）、WikiText-103（公开）、OpenHermes-2.5（公开）、Dolly-15k（公开）。
- **代码**：论文未明确声明开源，但使用了公开框架和模型；GCG 算法参考 Zou et al. (2023b)。
- **关键超参**：优化步数 500 步，早停阈值 $d_{KL} \leq 5.0$，过滤阈值 $d_{KL} \leq 10.0$；AdamW 优化器，学习率 $6 \times 10^{-4}$，cosine annealing，500 warmup steps。
- **硬件**：8×A100 GPU 节点用于离散优化，单张 NVIDIA RTX 6000 Ada 用于 Tiny Stories 模型训练。
