---
title: "Language-Models-as-Causal-Effect-Generators"
source: https://aclanthology.org/2025.emnlp-main.107.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:45:33"
field: "因果推断与语言模型交叉"
keywords: ["因果推断", "结构因果模型", "语言模型", "反事实生成", "个体治疗效应", "因果基准"]
innovations: ["提出 SD-SCM 框架，用预训练 LM 的参数化语义知识替代人工指定的结构方程，实现用户可控 DAG 下的序列数据生成", "构建首个基于 LM 的 CATE/ITE 估计基准，揭示 SOTA 方法在个体效应估计上的系统性困难", "提出基于语义干预的 LM 因果审计新范式，支持路径特异性反事实公平性分析"]
benchmarks: ["Breast Cancer SD-SCM Benchmark", "1000 datasets × 1000 samples", "GPT-2 and Llama-3-8b generated"]
---

# 论文速读：Language-Models-as-Causal-Effect-Generators

## 一句话总结
本文提出序列驱动结构因果模型（SD-SCM），利用预训练语言模型的语义知识参数化任意用户指定 DAG 的结构方程，从而自动生成观测、干预与反事实分布的序列数据；并基于此构建了一个包含千人级别乳腺癌治疗数据集的因果推断基准，用于评估 CATE/ITE 估计方法。

## 研究问题与动机
- **反事实数据难以获取但至关重要**：因果推理（尤其是个体水平因果效应估计）需要可观察的双重潜在结果，而反事实本质上是不可观测的，必须通过模拟获得。
- **现有因果推断基准依赖人工设计**：已有半合成/全合成基准（如 RealCause、Louizos et al. 2017）都需要人工指定响应面、生成反事实或干预分配，无法灵活控制因果结构。
- **语言模型蕴含丰富语义因果知识但未系统利用**：预训练 LM 已编码大量现实世界关联，但如何将其转化为可控因果结构的生成机制仍缺乏形式化框架。
- **CATE/ITE 估计基准稀缺**：平均因果效应（ATE）基准相对充足，但针对条件平均因果效应（CATE）和个体因果效应（ITE）的高质量数据基准严重匮乏。

## 核心贡献（创新点）
1. **定义了 SD-SCM 框架**：将任意语言模型与 DAG 组合为序列驱动结构因果模型，形式化定义了观测、干预与反事实采样过程（parent-only concatenation + domain-restricted sampling）。区别于以往基于固定数据集拟合生成模型的工作，本文支持任意用户定义的 DAG 结构。
2. **构建了首个基于 LM 的 CATE/ITE 估计基准**：设计了包含 14 个变量的乳腺癌治疗 SD-SCM，生成 1,000 个数据集（50 × 20），测试了 15+ 种主流估计方法的 off-the-shelf 性能，发现即使是 SOTA 方法在个体效应估计上仍存在显著困难。
3. **提出基于 SD-SCM 的语言模型因果审计新范式**：展示了同一框架可用于检测 LM 中编码的不当因果效应（如偏见、误导信息），支持路径特异性反事实公平性分析。

## 方法详解
- **SD-SCM 定义**：五元组 $\mathfrak{B} = (\mathbf{V}, \mathbf{U}, \mathcal{G}, \mathcal{P}, \tau)$，其中 $\mathbf{V}$ 为内生（观测）序列变量集合，$\mathbf{U}$ 为外生（未观测）序列变量集合，$\mathcal{G}$ 为变量间的 DAG，$\mathcal{P}$ 为预训练语言模型，$\tau$ 为与 $\mathcal{G}$ 一致的拓扑排序。所有变量的共同祖先为训练 LM 的 prior inputs $C$。
- **两个核心技术**：
  - **Parent-only concatenation（仅父节点拼接）**：对每个变量 $\tilde{x}_i$，将其父节点的序列按 $\tau$ 顺序拼接作为 LM 的上下文输入。
  - **Domain-restricted sampling（域限制采样）**：对变量 $\tilde{x}_i$ 的样本空间 $\Omega_{\tilde{x}_i}$ 中的每个候选序列 $v$，计算 $\mathcal{P}(v|C, \text{PA})$ 并归一化，再从中多项式采样。
- **三种分布采样**：
  - **观测分布**：沿 $\tau$ 顺序依次对每个变量做 parent-only concatenation + domain-restricted sampling。
  - **干预分布**：对目标变量 $\tilde{v}_i$ 强制赋值为 $v$，其余变量照常采样（修改对应 factorization）。
  - **反事实分布**：三步走——**abduction**（从观测数据中记录 $\mathbf{u}$ 及非后代变量的固定值）、**action**（施加 $\mathrm{do}(\tilde{v}_i = v)$）、**prediction**（沿 DAG 继续采样后代变量）。
- **关键设计选择**：使用拓扑排序 $\tau$ 打破父节点顺序歧义（LM 对输入词序敏感）；通过将 LM 输出的 log-probability 作为连续型 outcome 而非分类 outcome，可捕捉个体水平的概率变化，即使采样标签不变。

## 实验与结果
- **数据集**：乳腺癌 SD-SCM，14 个变量（含 4 个外生变量 age/medical conditions/medication/menopausal status、8 个临床协变量、PD-L1 表达水平为处理变量、治疗方案为结果），每个变量 10 种 phrasing，共 $10^{14}$ 种序列组合；50 种随机 phrasing 变体，每种生成 20 个大小为 1,000 的数据集，共计 1,000 个数据集；语言模型使用 GPT-2 和 Llama-3-8b。
- **评估基线（15+ 种）**：T-Only OLS、BART、BART-ITE、CausalForest、ForestDML、LinearDML、ForestDR、LinearDR、ForestS/T、LinearS/T、RF、LinReg、TNet、TARNet、CQR。
- **关键结果**：
  - **因果方法优于非因果方法**：在 All Covariates 设置下，BART 对 ATE 的 $R^2$ 达 0.9999（GPT-2）和 0.9967（Llama-3-8b），远超 T-Only OLS（0.6047 / 0.5082）。
  - **CATE/ITE 估计仍然困难**：All Covariates 下 BART 对 CATE 的 $R^2$ 最高（GPT-2: 0.9691, Llama-3-8b: 0.9214, N=10,000），但多类方法（CausalForest、TARNet、TNet 等）在两种样本量下均表现不佳。
  - **隐藏混杂导致性能大幅下滑**：Hidden U 设置下几乎所有方法性能暴跌，BART 从 0.9691 降至 $\leq 0$；仅有 BART-ITE 保留微弱正 $R^2$（0.0185）。
  - **观测与干预分布存在符号反转**：部分 SD-SCM 变体中，简单均值差估计的 ATE 与真实 SATE 符号相反（"治疗看似降低结果，实则增加结果"）。
  - **区间估计不稳定**：Hidden U 下 BART-CATE 区间过窄且覆盖不稳定，BART-ITE 区间过宽接近无信息；CQR 保持名义覆盖率但区间极宽。
- **最强结果**：BART 在所有设置下对 ATE 和 CATE 的点估计表现最优；N 从 1,000 增至 10,000 后多数方法有所改善但并未根本解决困难。

## 相关工作脉络
- **半合成因果基准（RealCause、Athey et al. 2024 等）**：基于真实数据集拟合生成模型后采样新数据，因果结构由原数据固定；SD-SCM 允许用户任意指定 DAG 结构，因果结构由 LM 在给定结构中动态确定。
- **LM 反事实生成（Chatzi et al. 2024、Ravfogel et al. 2024）**：聚焦 LM 内部网络层面的干预（Gumbel-Max trick 推断噪声并复用），生成 counterfactual 字符串；本文采用语义干预（semantic intervention），在固定 LM 上构建有意义的仿真环境，因果结构由用户 DAG 完全控制。
- **因果推理与 NLP 结合（Feder et al. 2021、Jin et al. 2023）**：关注 LM 本身的因果推理能力评估；本文反向利用 LM 的既有知识作为因果机制的参数化工具。
- **结构因果模型与生成模型结合（Pawlowski et al. 2020、Sanchez & Tsaftaris 2022）**：面向从已有数据学习/推断因果关系的反事实推理；本文专注于给定 DAG 下的灵活数据生成，应用场景不同。
- **IMCAE（Im et al. 2024）**：用深度自回归模型作为因果推断引擎；与本文思路相近但本文是先将 LM 作为 SCM 机制再用于基准生成和审计。

## 局限性与未来方向
- **结构方程依赖预训练 LM 而非显式指定**：这意味着因果关系的强度和形式受限于 LM 的训练数据，可能无法精确编码研究者期望的特定函数形式；未来可探索训练/fine-tune LM 使其在给定 DAG 下主动诱导变量间关系。
- **变量 phrasing 需人工枚举且随变量数增长而爆炸**：当前每个变量手工设计 10 种 phrasing，未来需端到端自动化。
- **LM 预存偏见和错误可能迁移至生成数据**：审计工作本身也受限于 LM 解释工具的可欺骗性。
- **未来方向**：工具变量场景、因果发现（测试结构学习方法是否能识别变量因果方向）、在生成过程中联合学习因果结构。

## 研究启发与可借鉴点
1. **SD-SCM 采样范式（parent-only concatenation + domain-restricted sampling）可迁移至其他序列生成任务**：任何需要在文本序列上施加可控因果/条件依赖关系的场景均可复用该范式。
2. **用 LM 的 log-probability 空间作为 outcome 而非硬标签**：这一技巧可捕捉个体水平的微小概率变化，对构建高敏感度基准尤为有用，可借鉴到需要细微效应信号的任务中。
3. **"反向工程临床决策"作为 LM 审计的应用范式**：通过指定领域 DAG 并利用 LM 的语义知识推断隐含因果结论，为 LM 可解释性/公平性审计提供了可复用的方法论。
4. **基准设计中的"符号反转"标准**：以观测分布与干预分布在符号上产生分歧作为"有意义的因果结构"判据，为评估基准质量提供了可操作的量化指标。
5. **对隐藏混杂的稳健性测试应成为因果推断基准的标配**：本文 All Covariates vs. Hidden U 的对照设计值得推广为其他因果学习工作的标准评估协议。

## 关键术语表
**SD-SCM（Sequence-Driven Structural Causal Model）**：将语言模型的概率分布作为结构方程参数化、由用户指定 DAG 结构驱动的序列数据生成框架。
**Parent-only concatenation**：对每个变量，仅将其 DAG 父节点的文本序列按拓扑序拼接作为 LM 的输入上下文，避免引入后代信息污染。
**Domain-restricted sampling**：在变量预定义的有限样本空间 $\Omega$ 内对 LM 的输出概率进行 tabulation 和归一化后进行多项式采样，实现离散取值控制。
**CATE（Conditional Average Treatment Effect）**：在给定协变量条件下估计的平均个体治疗效应 $E[Y(1)-Y(0)|X=x]$。
**ITE（Individual Treatment Effect）**：单个样本的异质性治疗效应 $Y_i(1)-Y_i(0)$ 的估计值。
**Abduction-Action-Prediction**：Pearl 反事实推理的三步流程：abduction（从观测推断外生变量取值）、action（施加 do-算子干预）、prediction（沿修改后的 SCM 前向传播得到反事实结果）。
**PEHE（Precision in Estimating Heterogeneous Effects）**：以 RMSE 形式衡量 CATE 估计精度的标准指标，即 $\sqrt{E[(\hat{\tau}(X)-\tau(X))^2]}$。
**Semantically meaningful simulation**：以自然语言为数据载体的高保真仿真，LM 的语义知识替代人工设计的函数关系，使生成的因果结构具有现实合理性。

## 可复现要素
- **数据集**：基于 GPT-2 和 Llama-3-8b 生成的乳腺癌 SD-SCM 数据集（50 种变体 × 20 个数据集 × 1,000 样本/数据集 = 1,000 个数据集），代码和数据已开源：https://github.com/lbynum/sequence-driven-scms
- **代码/权重**：代码开源（GitHub 链接如上）；使用 GPT-2 和 Llama-3-8b 两个预训练语言模型（模型本身非开源或受限）
- **关键超参**：每个变量 10 种 phrasing；50 种 SD-SCM 变体；每变体 20 个数据集；单数据集样本量 N=1,000 或 N=10,000；估计方法使用各包默认设置，未进行调参；结果以 log P(y=0) 为主要连续 outcome 指标
