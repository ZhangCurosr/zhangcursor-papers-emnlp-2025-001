---
title: "Language-Models-as-Causal-Effect-Generators"
source: https://aclanthology.org/2025.emnlp-main.107.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:53:32"
field: "因果推断与语言模型交叉"
keywords: ["causal inference", "counterfactual generation", "language models", "SD-SCM", "CATE estimation", "ITE estimation", "causal benchmark"]
innovations: ["将预训练语言模型条件分布参数化结构因果模型的结构方程，支持用户自定义DAG下的观测/干预/反事实序列数据生成", "基于乳腺癌PD-L1场景构建千人级CATE/ITE估计基准，首次在LM生成数据上系统评测10+因果推断方法的异质性效应估计能力", "提出面向LM的语义级因果审计框架，揭示GPT-2与Llama-3-8b在同一DAG下编码不同因果信念的现象"]
benchmarks: ["Breast cancer PD-L1 therapy SD-SCM benchmark (1000 datasets)", "CATE/ITE estimation benchmark with hidden confounder settings"]
---

# 论文速读：Language-Models-as-Causal-Effect-Generators

## 一句话总结
本文提出**序列驱动结构因果模型（SD-SCM）**，将预训练语言模型（LM）的条件分布作为结构方程，根据用户定义的DAG生成带有可控因果结构的序列数据，并首次利用该框架生成了可直接用于CATE/ITE估计的个体层面反事实基准数据集。

## 研究问题与动机
- 反事实本质上不可观察，必须通过模拟获得；现有因果推断基准（如SynthCe/Fashion）需人工指定结构性方程或手动生成反事实结果，难以覆盖个体层面的异质性效应估计。
- 真实世界数据中混杂结构未知、不可控，基准缺少对已知因果图结构、可精确设定干预/反事实状态的"高保真仿真环境"。
- 现有基于LM生成反事实文本的工作（如Chatzi et al., Ravfogel et al.）聚焦于LM内部的**网络干预**，未提供用户级因果图结构的灵活定义能力。
- 因果效应估计领域的CATE/ITE基准稀缺，已有benchmark多为ATE设计，缺乏能体现confounding/collider等结构差异且能产生可测量异质性效应的序列数据测试床。

## 核心贡献（创新点）
1. **将任意LM+DAG组合转化为SD-SCM的形式化框架**：定义观测/干预/反事实三种分布上的采样过程，区别于以往仅使用固定生成模型拟合真实数据集的做法，允许用户自由指定因果结构。
2. **构建首个基于LM生成的大规模CATE/ITE估计基准**：通过乳腺癌PD-L1疗法选择场景生成1000个数据集（50种SD-SCM变体×20重复，每数据集1000样本），实现从观测到干预的个体层面可观测反事实。
3. **提出面向LM因果审计的新范式**：同一框架可用于探查不同LM是否编码了不同甚至有害的因果假设（如GPT-2 vs Llama-3-8b在PD-L1→疗法选择上的相反因果结论），为模型偏差/误导性因果关系的审计提供机制。

## 方法详解
- **SD-SCM形式化**：五元组 $\mathfrak{B} = (\mathbf{V}, \mathbf{U}, \mathcal{G}, \mathcal{P}, \tau)$，其中$\mathbf{V}$为内生（观测）序列变量集合，$\mathbf{U}$为外生（未观测）序列变量集合，$\mathcal{G}$为变量上的DAG，$\mathcal{P}$为预训练LM（词汇表覆盖所有变量样本空间），$\tau$为与$\mathcal{G}$一致的拓扑序。所有变量共享一个共同祖先——训练LM的 prior inputs $C$。
- **两个关键抽象**：
  - **Parent-only concatenation**：对每个变量$\tilde{x}_t$，仅将其DAG父亲节点序列按$\tau$拼接后作为LM输入条件。
  - **Domain-restricted sampling**：对候选样本空间$\Omega_{\tilde{x}_t}$中的每一个可能值$x$，计算$\mathcal{P}(x \mid C \oplus x_{\mathrm{PA}})$并归一化，构成离散条件分布后从中采样。
- **三类分布采样**：
  - **观测**：按$\tau$依次对每个变量做parent-only拼接 + domain-restricted采样。
  - **干预**：对$\mathrm{do}(\tilde{v}_i=v)$，直接将$\tilde{v}_i$替换为固定值$v$，其余变量按同样方式采样（删除该节点入边后的DAG）。
  - **反事实**：三步——abduction（从观测证据恢复外生变量$\mathbf{U}=\mathbf{u}$及干预上游证据）、action（执行$\mathrm{do}$操作）、prediction（在前两者条件下生成$\mathbf{Y}^*$）。要求数据生成时保留bookkeeping以唯一确定$\mathbf{u}$。
- **基准构建策略**：选择全连接时序DAG（exogenous→covariates→treatment→outcome），以乳腺癌PD-L1表达水平$\tilde{t}$对疗法$\tilde{y}$的效应为目标；为避免措辞偏置，为每个变量构造10种同义表述，得到$10^{14}$种样本空间组合，随机采样50种SD-SCM变体并各生成20个数据集（每数据集1000样本），共1000个基准数据集；以$log\,P(\tilde{y}=0)$为连续型outcome进行CATE/ITE估计。

## 实验与结果
- **数据集与设置**：GPT-2（Radford et al., 2019）与Llama-3-8b（Dubey et al., 2024）作为LM模块；乳腺癌SD-SCM含14个变量（7个exogenous、6个covariate、treatment PD-L1、outcome疗法）；基线涵盖T-Only OLS、CausalForest、Linear/Forest DML、Linear/Forest DR、BART、BART-ITE、CQR、S/T-learner、TNet、TARNet、RF、LinReg等。默认超参，不做逐方法调优。
- **主要数值结果（Llama-3-8b, N=10,000, All Covariates）**：
  - **ATE**：BART $R^2=0.9967$、RMSE=0.0007；ForestDML $R^2=0.9608$；Causal方法整体显著优于T-Only OLS（$R^2=0.5082$）。
  - **CATE（$R^2$ clip at 0）**：BART $0.9214$最优；DML/DR类约$0.53-0.59$；NN方法（TARNet/TNet）始终≤0。
  - **Hidden U（隐藏年龄/病史/用药/绝经状态四个exogenous混杂）**：所有方法骤降；BART $R^2$降至≤0；仅BART-ITE在N=10,000时仍可达$R^2=0.0169$（极低但唯一非零）。
  - **区间估计**：All Covariates下多数CATE方法覆盖率低于95%名义值；Hidden U时LinearDML/DR接近名义覆盖率但区间极宽；BART-ITE为保守但过宽的区间。
- **GPT-2对比**：整体趋势一致，但CATE难度更高（BART $R^2$仅0.7162@N=1000 vs Llama 0.9214@N=10000），说明LM训练数据与领域知识对生成"有意义因果结构"影响显著。
- **数据量敏感性**：从N=1000增至10000，CATE $R^2$提升但仍有大量方法无法收敛（CausalForest/TARNet/TNet始终负值），证明基准的困难不仅来自小样本。
- **审计发现**：GPT-2对PD-L1↑→trastuzumab+pertuzumab疗法选择编码为"无显著因果效应"，而Llama-3-8b编码为"正向因果效应"，展现不同LM内部因果信念的系统性分歧。

## 相关工作脉络
- **Curth et al. (2021)** 对半合成数据生成四路线分类：本文属第(4)类"基于生成模型构造反事实/干预"，但突破在于**用户可控DAG**而非拟合已有真实数据集。
- **Athey et al. (2024)**（WGAN）与 **Neal et al. (2020)**（RealCause）：以真实数据集为锚点拟合生成器，因果结构固定于数据；本文允许用户任意指定DAG并让LM定义结构方程，实现**高保真仿真环境**。
- **Louizos et al. (2017)**（CEVAE）：以深度隐变量SCM从观测数据学习因果结构用于反事实推理；本文反向——先给定DAG再用LM生成数据，目标为基准构造而非因果发现。
- **Chatzi et al. (2024)** 与 **Ravfogel et al. (2024)**：同为利用SCM在LM内生成反事实文本，但采用**网络干预**（改写内部表征）并借助Gumbel-Max trick复用噪声；本文采用**语义干预**（在顶层变量层面do操作），直接控制因果图结构。
- **Fryer et al. (2022)**（Counterfactual Fairness Probing）：早期指出LM反事实文本可审计偏见；本文扩展为**系统化的因果效应审计**，并可用于路径特异性反事实公平性（path-specific counterfactual fairness）。
- **Im et al. (2024)**：使用深度自回归模型作为因果推理引擎；本文则把LM作为固定机制模块嵌入用户指定DAG，强调生成可控基准而非端到端训练因果推理模型。

## 局限性与未来方向
- **结构方程依赖LM先验**：若LM未在训练语料中编码所期望的因果强度/方向，生成的数据可能关系过于微弱或噪声过大，难以产生"有意义"的异质效应——这是LM-SCM范式的本质约束。
- **措辞枚举开销大**：14个变量×10种表述导致样本空间膨胀，人工设计多表述繁琐；大规模变量场景下可扩展性存疑。
- **外生变量bookkeeping依赖**：反事实采样需要保存所有$\mathbf{U}$值，若脱离数据生成流程则无法复原，限制了在纯观测数据上的直接应用。
- **社会风险**：基准数据仍会继承LM的偏见与误信息；审计应用可能给出过度简化的"解释"。
- **未来方向**：① 训练或微调LM使其主动诱导符合输入DAG的结构关系；② 扩展至工具变量、中介效应、时间序列因果发现等更复杂结构；③ 将SD-SCM作为训练阶段的结构先验，直接训练具备因果推理能力的语言模型。

## 研究启发与可借鉴点
- **LM作为结构方程参数化器的范式**：把"因果图结构由人定义、条件分布由LM定义"的解耦思路，可直接迁移至其他需可控合成数据的领域（如经济/流行病仿真、推荐系统off-policy评估），减少手动构造structural equation的专家成本。
- **多措辞随机采样策略**：对每个变量构造10+同义表述并在变体间随机选择，是消解语言模型对表层措辞敏感的有效工程实践，适用于任何LM-based数据生成任务以降低表述偏置。
- **以log-probability为连续outcome的设计**：当离散outcome同质性过高时，改用$log\,P(y=k)$或概率空间连续化可显著提升CATE/ITE的信号强度，这一"soft target"策略值得在医疗、金融等高噪声场景复现。
- **审计双LM因果信念差异**：同一DAG+样本空间下对比GPT-2与Llama-3-8b的效应方向分歧，揭示不同pretrain corpus导致的隐含因果假设差异；可作为模型安全/合规审计的标准流程。
- **团队结合点**：本框架可与下游任务（如因果发现、反事实解释生成、LLM对齐）对接——先用SD-SCM生成带ground-truth ITE的基准，训练轻量级"因果感知"适配器；或在多模态设置下扩展为文本+结构化变量的混合SD-SCM。

## 关键术语表
- **序列驱动结构因果模型（SD-SCM）**：将结构因果模型的条件分布由预训练语言模型定义，变量以自然语言序列取值，允许用户按DAG生成观测/干预/反事实数据。
- **仅父节点拼接（Parent-only concatenation）**：采样某变量时，仅将其DAG父亲的序列拼接到LM输入前，切断非祖先信息泄露。
- **受限域采样（Domain-restricted sampling）**：对变量样本空间中每个候选值计算LM条件概率并归一化，构造离散条件分布后采样。
- **CATE（条件平均治疗效应）**：在给定协变量$\mathbf{X}=\mathbf{x}$条件下对个体的平均处理效应$\mathbb{E}[Y(1)-Y(0)\mid \mathbf{X}=\mathbf{x}]$。
- **ITE（个体治疗效应）**：单个单位的潜在结果差异$Y_i(1)-Y_i(0)$，不可同时观测，需反事实推断。
- **PEHE（异质效应估计精度）**：$\sqrt{\frac{1}{n}\sum_i (\hat{\tau}_i - \tau_i)^2}$，衡量CATE/ITE估计误差的标准指标。
- **CQR（共形反事实分位数回归）**：基于conformal inference给出ITE的覆盖率保证区间，但区间往往过宽。
- **Semantic intervention（语义干预）**：在顶层语义变量层面执行do操作（如固定PD-L1表达水平），区别于修改LM内部表征的网络干预。

## 可复现要素
- **数据集**：1000个乳腺癌SD-SCM基准数据集（50变体×20重复×1000样本/数据集），代码与数据已在GitHub开源：https://github.com/lbynum/sequence-driven-scms
- **代码/权重**：生成代码公开；语言模型使用开源模型GPT-2与Llama-3-8b（需自行获取权重）
- **关键超参**：每个变量10种表述；exogenous变量4个（age, medical condition, medication, menopausal status）；covariate 10个（含基因突变、受体表达、HER2等）；treatment=PD-L1 (low/high)；outcome=therapy plan（4种）；数据集大小N=1000/10000
- **硬件**：V100与RTX8000 GPU
- **基线实现**：使用各方法公共默认配置（EconML、py-why生态等），未做额外调参
- **评估指标**：ATE用$R^2$与RMSE；CATE/ITE用clip-at-0的$R^2$、PEHE、empirical coverage（α=0.05）与区间宽度
