---
title: "Beyond-Input-Activations-Identifying-Influential-Latents-by"
source: https://aclanthology.org/2025.emnlp-main.87.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:42:35"
field: "大语言模型可解释性"
keywords: ["Sparse Autoencoder", "LLM Interpretability", "Model Steering", "Gradient-based Analysis", "Causal Influence"]
innovations: ["提出GradSAE框架，通过输出端梯度信号识别SAE中真正有因果影响力的latent特征", "理论证明一阶梯度近似可有效替代昂贵的ablation因果估计", "首次系统揭示输入激活与输出因果影响力之间存在显著偏差（Cross TopK overlap仅18.06%）"]
benchmarks: ["SQuAD"]
---

# 论文速读：Beyond-Input-Activations-Identifying-Influential-Latents-by

## 一句话总结
本文提出 **GradSAE**，一种无需训练的轻量级方法，通过在稀疏自编码器（SAE）中引入**输出端梯度信息**来识别对LLM输出具有高因果影响力的latent特征，相比仅依赖输入激活的传统方法，能更精准地定位可用于模型干预的关键latents。

## 研究问题与动机
1. **现有方法的前提假设缺乏验证**：当前基于SAE进行模型解释和干预的主流方法（如 Templeton et al., 2024; O'Brien et al., 2024）假设"输入侧激活高的latent必然对输出有强因果影响"，但该假设从未被严格证明，且近期证据表明这种假设有时会导致意外的输出效果（Durmus et al., 2024; Wu et al., 2025）。
2. **激活值高低 ≠ 因果影响力**：SAE的sparsity使得每个输入仅有少量latent被激活，但哪些latent真正驱动了最终输出仍不明确，导致干预时选择低效甚至无效的特征。
3. **梯度近似可替代昂贵的Ablation**：直接通过掩码消融（mask-out）每个latent来计算其对输出的真实因果影响计算开销极大；本文证明可通过一阶泰勒展开，用**输出logit对latent激活的梯度**高效近似这一因果影响。
4. **跨层/跨模型泛化性存疑**：不同层和不同模型的SAE latent语义存在差异，需验证基于梯度的选择策略是否具备跨架构的稳健性。

## 核心贡献（创新点）
1. **提出GradSAE框架，首次将输出端梯度纳入SAE latent选择过程**：与以往仅依赖输入激活幅度排序不同，GradSAE通过计算输出logit对latent激活的梯度信号，筛选出真正具有因果影响力的latent集合。
2. **理论证明梯度近似可有效替代Ablation因果估计**：基于一阶泰勒展开推导出 $g_{n,c} \approx \frac{\partial p(\mathcal{Y}|h(\mathbf{Z}))}{\partial \mathbf{H}_{n,c}} \odot \mathbf{H}_{n,c}$，为高效识别 influential latents 提供数学保证（附录A）。
3. **系统验证"激活≠影响力"这一关键认知偏差**：Cross TopK overlap仅18.06%，表明大量高输入激活latent在输出端梯度信号上为零或极低，直接挑战了领域内广泛使用的默认实践。
4. **设计两类互补实验验证有效性**：扰动实验（Masking TopK/BottomK后评估EM/F1变化）和局部引导实验（同上下文不同问题的latent注入），分别从"破坏关键latent"和"注入引导latent"两个角度证实GradSAE的优势。
5. **展示跨层与跨模型的泛化能力**：在Gemma 2 9B（第9/20/31层）和LLaMA 3 8B Instruct上的实验一致表明GradSAE在不同深度和不同架构上均优于Baseline。

## 方法详解
**问题设定**：给定输入序列 $\mathcal{X}$，第$l$层隐藏表示 $\mathbf{Z} \in \mathbb{R}^{N \times D}$，预训练SAE通过编码矩阵 $\mathbf{W}_{\mathrm{enc}} \in \mathbb{R}^{D \times C}$ 和解码矩阵 $\mathbf{W}_{\mathrm{dec}} \in \mathbb{R}^{C \times D}$（$C > D$）进行分解：

$$\hat{\mathbf{Z}} = \mathbf{H} \mathbf{W}_{\mathrm{dec}} = \sigma(\mathbf{Z} \mathbf{W}_{\mathrm{enc}}) \mathbf{W}_{\mathrm{dec}}$$

其中 $\mathbf{H} \in \mathbb{R}^{N \times C}$ 为稀疏latent激活，$\sigma$ 为ReLU。

**因果影响定义**：latent $c$ 在第$n$个token处对输出 $\mathcal{Y}$ 的影响定义为包含/排除该latent时的输出概率差：

$$\mathbf{g}_{n,c} = p(\mathcal{Y}|\mathbf{H}) - p(\mathcal{Y}|\mathbf{H}_{n,/c})$$

其中 $\mathbf{H}_{n,/c}$ 表示将第$n$个token的$c$-th latent值置为0。

**梯度近似推导**（附录A）：利用输出概率函数的连续性，对 $p(\mathcal{Y}|\mathbf{H})$ 在 $\mathbf{H}_{n,/c}$ 处进行一阶泰勒展开：

$$p(\mathcal{Y}|\mathbf{H}) \approx p(\mathcal{Y}|\mathbf{H}_{n,/c}) + \frac{\partial p(\mathcal{Y}|h(\mathbf{Z}))}{\partial \mathbf{H}_{n,c}} \cdot \mathbf{H}_{n,c}$$

从而得到梯度近似：

$$\mathbf{g}_{n,c} \approx \frac{\partial p(\mathcal{Y}|h(\mathbf{Z}))}{\partial \mathbf{H}_{n,c}} \odot \mathbf{H}_{n,c}$$

**整体影响分数**：对序列中所有token取平均：$\mathbf{g}_c = \frac{1}{N}\sum_{n=1}^{N} \mathbf{g}_{n,c}$，仅保留 $\mathbf{g}_c > 0$ 的positive latents构成 $\mathcal{Z}_{\mathrm{NZ}}$。

**Latent选择**：从高影响集合 $\mathcal{Z}_{\mathrm{NZ}}$ 中选择TopK（$\mathcal{Z}_{\mathrm{high}}$）和BottomK（$\mathcal{Z}_{\mathrm{low}}$）latents进行后续干预。

**Baseline对比**：省略梯度计算，直接使用原始token-wise SAE激活的均值 $\mathbf{g}_c = \frac{1}{N}\sum_{n=1}^{N} \mathbf{H}_{n,c}$ 进行排序。

## 实验与结果
**数据集**：SQuAD（验证集10.6k条样本，上下文平均长度~120词，问题平均~5个/上下文）；评估指标为Exact Match (EM) 和Token-level F1。

**实验设置**：
- 主实验使用Gemma Scope系列的 `gemma-scope-9b-it-res-canonical` SAE（训练于Gemma 2 9B Instruct第9层，latent维度131,000）
- 泛化实验扩展至同一模型的第20/31层及LLaMA 3 8B Instruct（第25层，latent维度65,536）
- 所有实验固定随机种子为42，在A100 SXM4 80GB GPU上运行

**扰动实验结果（表1）**：
- **初始性能**：所有实验在无扰动下达到100% EM/F1
- **GradSAE vs Baseline TopK掩码**（K=50%）：GradSAE使EM降至30.58%、F1降至43.33%；Baseline仅降至53.42% / 58.46%，降幅显著更大
- **BottomK掩码**：两种方法在所有K值下均保持~99%以上F1，证实非关键latent的移除不影响输出
- **K=1极致对比**：GradSAE EM=80.45%，Baseline EM=94.79%，单个最关键latent的影响已被GradSAE精准捕获

**局部引导实验结果（表1下半部分）**：
- **GradSAE TopK引导（K=10）**：达到F1=10.21%，显著优于Baseline的6.69%
- **K增大时引导效果递减**：因替换过多TopK latents导致SAE重建质量下降、输出连贯性受损
- **BottomK引导**：所有K值下EM和F1均接近0，印证非影响力latent无法驱动输出偏移

**跨层稳健性（表4）**：GradSAE在第9/20/31层均保持显著的TopK-BottomK性能差距；Baseline在第31层差距明显缩小（TopK F1=77.69% vs BottomK F1=99.68%），说明深层输入激活的因果关联性减弱。

**跨模型泛化（表5，LLaMA 3 8B）**：GradSAE TopK(K=50%)降至EM=48.57%/F1=60.99%，Baseline仅降至78.94%/78.94%，趋势与Gemma一致。

## 相关工作脉络
1. **Cunningham et al. (2023), Bricken et al. (2023)** — 开创性工作，证明SAE可从LLM中提取可解释的monosemantic特征；本文与之定位不同：前者关注SAE的训练与特征提取质量，本文聚焦于**如何从已训练SAE中筛选真正有因果影响力的latent**用于下游干预。
2. **Templeton et al. (2024), O'Brien et al. (2024)** — 基于SAE激活进行模型引导的典型工作，隐含假设"高输入激活=高输出影响力"；本文直接挑战这一假设，证明两者重合度仅约18%。
3. **SAIF (He et al., 2025)** — 针对指令遵循的SAE解释与引导框架；与本文差异在于SAIF依赖输入激活识别instruction-relevant特征，GradSAE通过梯度信号实现更精准的因果归因。
4. **SAE-TS (Chalnev et al., 2024), SpARE (Zhao et al., 2024), FGAA (Soo et al., 2025)** — 各类SAE引导方法均隐式假设输入侧激活与输出因果影响一一对应；GradSAE为这类方法提供了**无需重新训练、即插即用**的latent筛选优化。
5. **MIE (Wu et al., 2025)** — 基于互信息的SAE特征解释方法，解决频率偏差问题；与GradSAE正交：MIE改进特征解释的统计质量，GradSAE改进特征选择的影响力评估。
6. **Gemma Scope (Lieberum et al., 2024)** — 提供大规模开源SAE模型（含Gemma 2多层）；本文直接使用该资源验证方法，同时拓展到LLaMA 3系列以验证泛化性。

## 局限性与未来方向
1. **仅聚焦指令微调LLM的QA任务**：论文明确指出当前工作限于instruction-tuned LLM的指令遵循问答能力；推广至预训练LLM或其他任务（如推理、生成）是明确的未来方向。
2. **单点梯度近似的线性假设局限**：一阶泰勒展开在高维非线性LLM中仅为近似，极端情况下可能低估或高估真实因果影响；更高阶或非线性近似值得探索。
3. **未探索梯度信号的进一步压缩/聚合方式**：当前仅使用逐token梯度均值作为latent影响分数，考虑时间维度或注意力权重的加权聚合可能进一步提升精度。
4. **引导实验中K过大导致输出退化**：替换过多TopK latents会破坏SAE重建质量，如何在保持重建保真度和引导强度之间取得平衡尚未完全解决。
5. **跨layer一致性有待更深入分析**：虽然GradSAE在不同层均表现稳健，但梯度信号在浅层与深层的语义含义可能存在本质差异，需要更细粒度的特征对齐研究。

## 研究启发与可借鉴点
1. **"梯度+激活"双信号融合策略可迁移至其他可解释性方法**：本文的核心洞见（输入侧信号不足以表征因果影响力）可推广至attention rollout、 Integrated Gradients等解释技术中，用于筛选真正关键的特征通道。
2. **无需训练的即插即用接口设计**：GradSAE完全training-free，可直接叠加于任何预训练SAE之上，这一设计范式为社区提供了低门槛的SAE后处理方法模板。
3. **Ablation与梯度近似的等价性证明具有方法论价值**：附录A中的一阶泰勒推导简洁有力，展示了如何用微分近似替代组合枚举，类似思路可用于其他需要逐特征因果归因的场景。
4. **TopK/BottomK对比实验设计精妙**：扰动实验同时展示"关键特征移除的危害"和"非关键特征移除的无害"，形成对方法有效性的双向验证，可作为SAE相关研究的标准评估范式。
5. **跨层/跨模型验证的完整复现清单**：论文提供了Gemma 2（3层）和LLaMA 3的完整结果，提示后续研究者应至少覆盖2-3个不同架构以验证方法的泛化性，避免过拟合到单一模型。

## 关键术语表
**Sparse Autoencoder (SAE)**：一种过完备的自编码器，通过学习稀疏的latent表示来解混LLM中多义性（polysemantic）的神经元表示，每个latent倾向于编码单一可解释特征。

**Superposition**：LLM中实际需要的特征数量远超可用神经元数量的现象，迫使网络以叠加方式编码多个概念，是导致neuron多义性的根本原因。

**Latent (SAE特征)**：SAE学习到的过完备隐空间中的单个维度，理论上每个latent对应一个monosemantic（单义）概念。

**Local Steering**：在同一上下文的不同问题间进行latent替换，验证SAE特征是否携带与具体问题答案相关的语义信息。

**Gradient Approximation of Influence**：用输出logit对latent激活的梯度与激活值的逐元素乘积，近似替代真实的ablation因果影响。

**Cross Overlap**：Baseline与GradSAE所选TopK/BottomK latent集合之间的交集比例，用于量化两种方法选择结果的重合度。

**Inner Overlap**：同一方法在不同prompt（共享相同context）下所选TopK/BottomK latent的重合比例，反映选择结果的稳定性。

**Gemma Scope**：Lieberum et al. (2024)发布的开源SAE系列，为Gemma 2模型各层提供预训练的超完备稀疏自编码器。

## 可复现要素
- **数据集**：SQuAD（公开可用，验证集10.6k样本）
- **代码**：已开源 → https://github.com/Tizzzzy/sae_gradient
- **SAE模型**：Gemma Scope `gemma-scope-9b-it-res-canonical`（第9层，131K维）；泛化实验使用第20/31层及LLaMA 3 `llama-3-8b-it-res-jh`（第25层，65,536维）
- **随机种子**：42
- **硬件**：A100 SXM4 GPU, 80GB显存
- **关键超参**：K取值{1, 10, 20, 30, 50%}；50%定义为当前样本非零latent数量的一半
- **指标**：Exact Match (EM)、Token-level F1（经大小写/标点/停用词标准化）
