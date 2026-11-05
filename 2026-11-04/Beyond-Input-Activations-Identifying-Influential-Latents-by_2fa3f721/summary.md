---
title: "Beyond-Input-Activations-Identifying-Influential-Latents-by"
source: https://aclanthology.org/2025.emnlp-main.87.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:26:11"
field: "大语言模型可解释性"
keywords: ["稀疏自编码器", "大语言模型可解释性", "因果归因", "模型操控", "梯度近似", "SAE特征筛选"]
innovations: ["提出GradSAE，通过输出端梯度信号替代输入激活来识别SAE潜在特征的因果影响力", "证明输入激活与输出影响力存在系统性偏差（TopK交叉重叠率仅18.06%）", "在扰动与本地操控任务上验证GradSAE有效性，跨层（9/20/31）和跨模型（Gemma/LLaMA）保持一致性提升"]
benchmarks: ["SQuAD", "Gemma Scope", "LLaMA 3 8B Instruct"]
---

# 论文速读：Beyond-Input-Activations-Identifying-Influential-Latents-by

## 一句话总结
本文提出 GradSAE（梯度稀疏自编码器），通过引入输出端梯度信号来量化 SAE 潜在特征对 LLM 输出的因果影响力，解决了传统方法仅依赖输入激活进行特征识别与模型操控时"激活高≠影响大"的根本缺陷。

## 研究问题与动机
1. **输入激活不等于因果影响力**：现有 SAE 研究普遍假设"被高激活的潜变量必然对输出有因果贡献"，但该假设从未被验证，近期研究已发现仅凭此假设进行模型操控会产生非预期效果。
2. **特征选择的可靠性问题**：LLM 神经元存在多头语义（polysemanticity）和特征叠加（superposition）现象，SAE 能解耦出单概念特征，但如何从众多激活特征中筛选出真正关键的少数特征仍缺乏可靠标准。
3. **计算效率与评估精度的矛盾**：精确评估每个潜在特征的影响力需要逐个屏蔽（ablation）并比较输出变化，计算成本极高，亟需高效的近似方法。
4. **操控实验缺乏因果归因支撑**：现有的 SAE 操控方法（如 SAIF、SAE-TS 等）直接替换高激活特征，但未验证这些特征是否真的驱动了目标输出行为。

## 核心贡献（创新点）
1. **提出 GradSAE 梯度影响力评估框架**：通过一阶泰勒展开将离散特征屏蔽实验近似为梯度加权激活值，以极低计算开销估计每个潜变量对输出的因果贡献，与 Baseline 仅用原始激活均值相比引入了输出方向的信息。
2. **揭示输入激活与输出影响力的系统性偏差**：实验表明 Baseline TopK 与 GradSAE TopK 的交叉重叠率仅为 18.06%，证明大量高输入激活特征实际上对输出几乎没有因果影响。
3. **在扰动实验与本地操控两个任务上验证有效性**：GradSAE 遮蔽 TopK 后 EM 降至 30.58%（Baseline 为 53.42%），本地操控 F1 达 10.21%（Baseline 为 6.69%），且该优势在 Layer 9/20/31 和 Gemma/LLaMA 上保持一致。
4. **提供严密的理论推导与跨模型泛化验证**：附录 A 给出了梯度近似的泰勒展开证明；扩展实验验证了 GradSAE 对 LLaMA 3 8B Instruct 的适用性，并分析了不同隐藏层的特征稳定性。

## 方法详解
1. **影响力定义（公式 2）**：潜变量 $c$ 对第 $n$ 个 token 输出的因果影响力定义为 $\mathbf{g}_{n,c} = p(\mathcal{Y}|\mathbf{H}) - p(\mathcal{Y}|\mathbf{H}_{n,/c})$，其中 $\mathbf{H}_{n,/c}$ 表示将该位置第 $c$ 个潜变量设为 0 后的激活矩阵。
2. **梯度近似（公式 3）**：利用输出函数连续性假设和一阶泰勒展开，将上述差值近似为 $\mathbf{g}_{n,c} \approx \frac{\partial p(\mathcal{Y}|h(\mathbf{Z}))}{\partial \mathbf{H}_{n,c}} \odot \mathbf{H}_{n,c}$，即输出梯度与原始激活的逐元素乘积，避免了逐个屏蔽的昂贵计算。
3. **整体影响力聚合**：对所有 token 的影响力取平均，得到潜变量 $c$ 的整体影响力 $\mathbf{g}_c = \frac{1}{N}\sum_{n=1}^N \mathbf{g}_{n,c}$；最终只关注 $\mathbf{g}_c > 0$ 的正向影响特征集合 $\mathcal{Z}_{NZ}$。
4. **TopK/BottomK 特征集构造**：按影响力排序，$\mathcal{Z}_{high}$ 取影响力最高的前 $K$ 个特征，$\mathcal{Z}_{low}$ 取最低的 $K$ 个特征；$K$ 取值为 $\{1, 10, 20, 30, 50\%\}$，其中 $50\%$ 指最后一 token 非零激活数的一半。
5. **扰动实验设计**：遮蔽 $\mathcal{Z}_{high}$ 时若性能显著下降而遮蔽 $\mathcal{Z}_{low}$ 无影响，则验证"激活不均等贡献"假设；基线方法跳过梯度计算，直接用原始激活均值 $\frac{1}{N}\sum_n \mathbf{H}_{n,c}$ 排序。
6. **本地操控实验设计**：利用 SQuAD 同一上下文含多个问题的特性，将从问题 $q^t$ 获得的 TopK 特征注入问题 $q$ 的原始激活中，观察模型是否被"操控"回答 $q^t$ 的答案。

## 实验与结果
- **数据集与评估**：SQuAD 阅读理解，指标为 Exact Match（EM）和 token-level F1；仅保留原模型 greedy decoding 准确率 100% 的样本。
- **主要模型**：Gemma Scope `gemma-scope-9b-it-res-canonical`（Gemma 2 9B Instruct 第 9 层，131K 维潜在空间）；泛化实验使用 LLaMA 3 8B Instruct 第 25 层 SAE（65,536 维）。
- **扰动实验最强结果**（Layer 9, K=50%）：GradSAE 遮蔽 TopK 后 EM=30.58%、F1=43.33%（Baseline 为 53.42%、58.46%），遮蔽 BottomK 两种方法均维持 ~99% 以上，差距显著。
- **本地操控最强结果**（Layer 9, K=10）：GradSAE 实现 F1=10.21%（Baseline 为 6.69%）；BottomK 替换两种方法均接近 0%，证明低影响力特征无法驱动输出变化。
- **跨层一致性**（Table 4, K=50%）：Layer 20 GradSAE TopK 遮蔽 EM=9.06%（Baseline 73.83%），Layer 31 为 EM=6.53%（Baseline 69.52%），Baseline 随层深变浅而 GradSAE 保持稳定。
- **跨模型一致性**（Table 5, LLaMA 3, K=50%）：GradSAE TopK 遮蔽 EM=48.57%（Baseline 65.71%），BottomK 均维持 ~97-100%，趋势与主实验一致。
- **特征重叠分析**（Table 3）：Baseline TopK 与 GradSAE TopK 交叉重叠仅 18.06%；同一上下文不同问题的 GradSAE TopK 内重叠为 50.46%（Baseline 为 89.91%），证明梯度信号有效过滤了输入冗余。

## 相关工作脉络
1. **Templeton et al. (2024) Scaling Monosemanticity**：通过输入激活识别 Claude 3 Sonnet 的 SAE 特征并进行操控，是本文 Baseline 的核心参照；本文指出其假设"输入激活即因果影响"缺乏实证支撑。
2. **SAIF (He et al., 2025)**：识别指令遵循相关 SAE 特征并操控，仍依赖纯输入激活，GradSAE 从特征选择源头提供更可靠的因果归因依据。
3. **SAE-TS (Chalnev et al., 2024)**：构建靶向 SAE 特征的操控向量以减少副作用，属于"如何更精准地替换特征"的下游优化；GradSAE 解决的是"应该选哪些特征"的上游筛选问题。
4. **MIE (Wu et al., 2025)**：基于互信息的 SAE 特征解释与运行时操控，仍停留在输入侧统计相关性层面；GradSAE 引入梯度信号实现输出方向的因果估计。
5. **FGAA (Soo et al., 2025) / SpARE (Zhao et al., 2024)**：分别在特征激活优化和上下文-参数知识冲突检测中利用 SAE 进行操控，均以输入激活为基础；GradSAE 的梯度近似框架可与这些方法结合使用。
6. **Gated SAE / TopK SAE / JumpReLU SAE (Rajamanoharan et al., 2024a,b; Gao et al., 2025)**：改进 SAE 架构以增强稀疏性和重建质量，属于训练侧优化；GradSAE 是训练后（training-free）的特征重要性评估工具，可与上述架构正交结合。

## 局限性与未来方向
1. 实验仅针对指令微调 LLM 的问答任务，尚未验证在预训练 LLM（如继续预训练、掩码语言建模等场景）中的适用性。
2. 本地操控实验中发现 K 值过大时操控效果反而下降，说明同时替换过多 TopK 特征会破坏 SAE 重建精度和输出连贯性，当前方法在"精准 vs. 幅度"的权衡上仍有优化空间。
3. 仅验证了 Gemma 和 LLaMA 两个模型系列，其他主流架构（PaLM、Qwen 等）及多模态模型的泛化性尚待探索。
4. 未来工作可扩展至推理链（CoT）生成、多轮对话、跨语言场景，以及将梯度影响力信号整合进 SAE 训练过程实现端到端优化。

## 研究启发与可借鉴点
1. **梯度加权激活的归因范式可迁移**：公式 3 将离散消融转化为连续梯度×激活的形式，计算成本低且易于实现，该思路可直接移植到 Transformer 注意力权重归因、LoRA 适配器重要性评估等场景。
2. **"同一上下文多问题"的实验设计值得借鉴**：利用 SQuAD 的结构特性在控制上下文变量不变的前提下隔离问题差异，这种精巧的实验控制策略可为其他操控/归因研究提供设计模板。
3. **training-free 评估框架的工程价值**：GradSAE 无需重新训练 SAE，可直接加载已有模型和 SAE 权重进行即插即用评估，这种免训练特性大幅降低了实际部署成本，适合快速原型验证。
4. **跨层跨模型的稳健性验证范式**：论文在 3 个隐藏层和 2 个模型系列上均展示一致结论，多维度交叉验证显著增强了可信度；后续工作可参考该验证策略来规避"仅在单一配置下有效"的发表风险。
5. **TopK/BottomK 双向对照的实验逻辑**：不仅验证 TopK 的有效性，还通过 BottomK 的"无影响"结果作为强对照，这种正反双向验证思路值得在特征选择类论文中广泛采用。

## 关键术语表
**Sparse Autoencoder (SAE)**：通过在隐藏表示与过完备潜在空间之间学习稀疏编码，将多头语义神经元解耦为单概念特征的可解释性工具。
**Causal Influence**：某个 SAE 潜在特征对模型输出的真实因果贡献，通过对比屏蔽前后输出分布的变化来度量。
**Gradient Approximation**：利用一阶泰勒展开将离散的"屏蔽特征→比较输出"实验近似为输出梯度与激活值的逐元素乘积，从而高效估计影响力。
**Local Steering**：在同一上下文的不同问题之间转移 SAE TopK 特征，验证特征是否携带独立于输入的问题特定信息。
**Polysemanticity**：单个神经元同时编码多个不相关概念的现象，是 LLM 内部表征难以直接解释的根本原因。
**Superposition**：LLM 需表征的特征数量远超可用神经元数量，迫使网络将多个概念叠加在同一神经元中。
**TopK / BottomK**：按影响力分数排序后分别选取最高/最低的 K 个潜在特征，用于扰动实验中的关键特征遮蔽与对照。
**Gemma Scope**：Lieberum 等人开源的覆盖 Gemma 2 多层的 SAE 系列，为本文核心实验提供了预训练好的高质量特征库。

## 可复现要素
- **数据集**：SQuAD（公开，论文使用 validation 集）
- **代码**：https://github.com/Tizzzzy/sae_gradient（开源）
- **SAE 模型**：Gemma Scope 系列（通过 SAELens 库获取，公开）；LLaMA 3 8B Instruct SAE（公开）
- **关键超参**：$K \in \{1, 10, 20, 30, 50\%\}$；随机种子固定为 42；$50\%$ 定义为最后一 token 非零激活数的一半
- **硬件**：A100 SXM4 GPU 80GB
- **论文未提及**：梯度截断阈值、$\epsilon$ 平滑系数（如有）、具体 token 级别损失实现细节
