---
title: "Automating-Steering-for-Safe-Multimodal-Large-Language-Model"
source: https://aclanthology.org/2025.emnlp-main.41.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:42:02"
field: "多模态大模型安全"
keywords: ["Multimodal Large Language Models", "Inference-time Safety", "Model Steering", "Latent Space Manipulation", "Safety Alignment", "LLaVA", "MLLM Safety"]
innovations: ["提出SAS自动层选择机制，无需人工调参即可定位安全相关内部层", "条件性推理时干预框架，仅在检测到毒性时激活拒绝头，兼顾安全与通用能力", "揭示steering intensity与输出行为间的非单调关系，为部署实践提供重要警示"]
benchmarks: ["VLSafe", "ToViLaG+", "RealWorldQA", "MMMU"]
---

# 论文速读：Automating-Steering-for-Safe-Multimodal-Large-Language-Model

## 一句话总结
AutoSteer 是一种完全自动化的推理时干预框架，通过自动识别 MLLM 内部最具安全相关性的中间层（SAS 机制）、训练轻量级 Safety Prober 进行毒性评估、并基于阈值条件性地激活 Refusal Head 来动态调控生成，无需对底层模型进行任何微调即可显著降低文本/视觉/跨模态多种攻击的成功率，同时保持通用能力不下降。

## 研究问题与动机
1. **MLLM 的安全脆弱性**：多模态大语言模型在面对对抗性输入（尤其是文本-视觉交叉模态攻击）时易生成有害、不道德或违规内容，而多模态输入空间的异质性使传统单模态安全防御手段不足。
2. **现有方法的局限性**：主流安全方法多依赖训练时对齐（如 RLHF/Safe RLHF）或推理时的全局 Latent Space Manipulation（如 LM-Steer），前者需要重新微调代价高且可能损害通用能力，后者施加全局干预会导致对安全输入也产生不必要的语义漂移。
3. **缺乏可解释、自适应的干预机制**：静态安全过滤器难以处理语境依赖的细粒度毒性判断；而过激干预又会压制良性输出。现有方法缺少根据输入风险程度动态调节干预强度的机制。
4. **模型间安全表示存在差异**：不同架构（如编码器融合型 vs 早期融合型）的安全信号在层间分布不同，手工选择探测层成本高昂，需要自动化的层选择机制。

## 核心贡献（创新点）
1. **提出 SAS（Safety Awareness Score）自动层选择机制**：基于 CAA（Contrastive Activation Addition）向量对的 pairwise cosine similarity 在所有层中自动识别安全区分最一致的层；本质区别在于无需人工调参即可实现跨模型的自动化探测层定位。
2. **构建轻量级 Safety Prober + 自适应阈值触发机制**：在选定层提取激活向量训练 MLP 探测器估计毒性概率，并以此条件性激活 Refusal Head（α(s) 门控函数）；与 LM-Steer 全局线性变换的本质区别在于干预仅在检测到时触发，避免对安全输入的干扰。
3. **Refusal Head 的条件性推理时干预设计**：基于 LM-Steer 框架学习拒绝矩阵 W，在 decoding 阶段通过 $\tilde{z}_t = \mathbf{H}_t \cdot (\mathbf{E} + \epsilon \alpha(s) \cdot (\mathbf{E} \cdot \mathbf{W})^\top)$ 调制 token 概率；与已有工作相比无需修改底层模型参数，模块化且可插拔。
4. **全面的跨模态安全评估与机制分析**：在 VLSafe 和重构的 ToViLaG⁺ 基准上系统评估文本/视觉/跨模态三类攻击，并提出 steering intensity ϵ 非单调性与 prober 评分极化等关键发现。

## 方法详解
AutoSteer 由三个模块组成，整体流程如下：

**（1）Safety Awareness Score (SAS) 层选择**：给定 N 对语言结构可控但毒性相反的样本 $(x_{\text{safe}}, x_{\text{toxic}})$，在每层 $\ell$ 计算对比激活向量 $\delta_l = h_l(x_{\text{toxic}}) - h_l(x_{\text{safe}})$，SAS 定义为所有对比向量对的平均 cosine similarity：
$$\text{SAS}(l) = \frac{2}{N(N-1)} \sum_{1 \le i < j \le N} \cos(\delta_l^{(i)}, \delta_l^{(j)})$$
SAS 越高表示该层对"安全 vs 毒性"区分越一致；最终选取 SAS 最高的层记为 $L$（LLaVA-OV 选 Layer 20，Chameleon 选 Layer 24）。

**（2）Safety Prober 训练**：在层 $L$ 的激活向量上训练一个单层 MLP（64 维隐层 + ReLU + Sigmoid）：
$$s = \mathcal{P}(h_L(x)) = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot h_L(x) + b_1) + b_2)$$
输出 $s \in [0,1]$ 表示输入为毒性的概率；使用 VLSafe 中 3,000 条毒性样本 + 3,000 条构造的安全样本训练。

**（3）条件性自动转向（Adaptive Steering）**：通过阈值函数将 prober 输出转为转向信号：
$$\alpha(s) = \begin{cases} 0, & s < \tau \\ 1, & s \ge \tau \end{cases}$$
阈值 $\tau=0.5$。将 $\alpha(s)$ 融入 LM-Steer 框架，在解码步骤 $t$ 修改输出嵌入矩阵：
$$\tilde{z}_t = \mathbf{H}_t \cdot \left(\mathbf{E} + \epsilon \alpha(s) \cdot (\mathbf{E} \cdot \mathbf{W})^\top\right)$$
其中 $\mathbf{W}$ 为 Refusal Head 拒绝矩阵（在 3,000 条毒性样本上训练，输出统一拒绝语"I'm sorry, but I can't assist with that"），$\epsilon$ 为转向强度（实验设为 0.1）。

**训练数据来源**：Refusal Head 和 Prober 均使用 VLSafe 对齐数据（3,000 条）；LLaVA-OV 额外用 ToViLaG 的 3,000 条毒性图像数据增强 Refusal Head 训练；层选择基于 CAA 向量（3,000 条）。

## 实验与结果
**数据集**：
- 安全评估：VLSafe Examine（500 条）、ToViLaG⁺（500 条 × 三种子集：Text / Image / Text+Image）
- 通用能力：RealWorldQA（500 条）、MMMU（500 条）
- 基线模型：LLaVA-OV-7B、Chameleon-7B

**主要结果（$\epsilon = 0.1$）**：

| 模型 | 基准 | 攻击类型 | Orig. ASR | AutoSteer ASR | 降幅 |
|---|---|---|---|---|---|
| LLaVA-OV | VLSafe | Text | 60.0% | **4.2%** | -55.8pp |
| LLaVA-OV | ToViLaG⁺ | Image | 70.6% | **0.0%** | -70.6pp |
| LLaVA-OV | ToViLaG⁺ | Text+Image | 30.0% | 9.6% | -20.4pp |
| Chameleon | VLSafe | Text | 67.8% | 15.4% | -52.4pp |
| Chameleon | ToViLaG⁺ | Text+Image | 56.1% | 14.3% | -41.8pp |

- **LLaVA-OV 上 ASR 降至接近最优**：VLSafe Text 仅 4.2%（Steer 为 2.0%），Image 子集实现零攻击成功；**通用能力完全保留**（RealWorldQA 61.8%，MMMU 48.4%，与原始模型持平）。
- **Chameleon 上亦取得显著安全增益**：VLSafe 从 67.8% 降至 15.4%，但 Image-only 子集提升有限（ASR 43.7%，仅 -8.3pp），归因于 Chameleon 在视觉安全感知层面表示较弱（SAS 整体偏低）。
- **与 Steer 基线对比**：AutoSteer 在多数场景接近或超越 Steer，且通用能力始终优于 Steer（避免全局干预导致的性能退化）。

**关键分析发现**：
- SAS 层选择与实际 prober 准确率高度一致（图 4、图 9）；early layers 虽在 Text-only 上表现好，但在 Image-only 上完全失效，只有 mid-to-late layers（16、20）能有效泛化。
- Steering intensity ϵ 与 ASR 呈非线性关系：在 LLaVA-OV 上 ϵ ≈ 0.05 后 ASR 趋于平台；过高 ϵ（>0.12）可能对 Chameleon 造成输出损坏。
- Prober 输出高度极化（接近 0 或 1），限制了细粒度控制；部分 case 中 prober 评分与实际毒性不一致（Appendix C）。

## 相关工作脉络
1. **LM-Steer（Han et al., 2024）**：推理时通过线性变换修改输出嵌入实现可控生成，AutoSteer 在其实用性上的改进——将其与条件性触发机制结合，避免全局干预导致通用能力下降。
2. **CAA（Rimsky et al., 2024）**：通过 contrastive activation addition 定位 attribute 相关层，AutoSteer 借鉴其向量构造思路但扩展为 SAS 的自动层选择指标，面向安全领域。
3. **HiddenDetect（Jiang et al., 2025）**：无微调的 jailbreak 检测方法，监控内部激活中的 refusal 语义；两者均利用内部表示，但 HiddenDetect 仅检测不干预生成，AutoSteer 进一步引入主动拒绝机制。
4. **CoCA（Gao et al., 2024）**：通过 constitutional calibration 调整 token logits 增强安全性；与 AutoSteer 的区别在于 CoCA 依赖 prompt-based 安全约束，而 AutoSteer 基于自动识别的内部层特征进行干预。
5. **InferAligner（Wang et al., 2024）**：推理时跨模型引导对齐安全性；AutoSteer 无需额外模型，仅依赖目标模型自身的内部表示进行干预。
6. **MM-SafetyBench（Liu et al., 2024c）**：多模态安全评估基准；本文指出该基准主要覆盖 toxic text + benign image 设置，因而重构了 ToViLaG⁺ 以补充 toxic image 等缺失场景。

## 局限性与未来方向
**论文自述局限**：
1. **Prober 泛化依赖训练数据**：基于 VLSafe 等数据集训练的 prober 在面对 out-of-distribution 有害输入或新型对抗策略时泛化能力有限。
2. **基础模型安全表示的上限约束**：若底层模型本身缺乏足够的安全意识（如 Chameleon 在 Image-only 任务上表现差），prober 性能将受根本性限制。
3. **额外训练组件的部署复杂度**：虽不需要微调 base model，但 Prober 和 Refusal Head 需分别训练并引入模型特异性超参校准。
4. **实验覆盖模型数量有限**：目前仅在 LLaVA-OV 和 Chameleon 两个 7B 模型上验证，需扩展至更大规模及更新架构。

**可合理推断的局限**：
5. **二元阈值的粒度限制**：prober 输出极化导致 $\alpha(s)$ 为硬切换，无法实现连续安全强度的平滑调节（见 Appendix C）。
6. **单轮对话限制**：当前框架设计针对单轮交互，尚未扩展到多轮对话场景的风险累积检测。

**未来方向**（论文提及）：
1. 在更广泛的 MLLM 架构（更大规模/新架构）上验证可扩展性；
2. 通过多样化与对抗性训练数据增强 prober 鲁棒性；
3. 探索多轮对话场景下的风险累积与历史聚合机制。

## 研究启发与可借鉴点
1. **SAS 自动层选择范式可迁移至其他内部表征干预任务**：如价值观对齐、风格控制等，无需人工逐层探测即可定位最相关的内部层，节省大量计算资源。
2. **条件性（conditional）干预而非全局干预的设计哲学**：通过 prober 阈值触发机制，仅在风险高于阈值时激活干预，是平衡安全性与通用性的通用策略，可推广到文本 LLM 安全微调替代方案中。
3. **Steering intensity ϵ 的非单调性发现具有重要的实践警示价值**：增加干预强度并不保证更安全——部分场景下适度干预反而暂时增强了对有害指令的服从（Appendix F.3 Case 3），这提示在部署 steering 类方法时需进行细致的 ablation。
4. **与团队方向的结合机会**：若团队关注多模态安全对齐，可将 SAS 思路与 RLHF/RLAIF 的训练后干预阶段结合，设计"训练时对齐 + 推理时自适应校准"的两阶段框架；也可将 prober 的阈值可微化探索以替代硬切换。

## 关键术语表
- **AutoSteer**：论文提出的推理时自适应安全干预框架，通过自动层选择+轻量化探测器+条件性拒绝头实现无需微调的安全调控。
- **SAS (Safety Awareness Score)**：基于 CAA 对比激活向量对的平均 cosine similarity 设计的自动层选择度量，值越高表示该层安全区分越一致。
- **Safety Prober**：部署在选定层的轻量 MLP 分类器，以激活向量为输入输出毒性概率标量。
- **Refusal Head**：在推理时生成的拒绝矩阵 W，被激活时将 token 输出导向标准拒绝语，避免直接修改原模型参数。
- **LM-Steer**：基于 Han et al. (2024) 的推理时嵌入空间线性变换方法，是 AutoSteer 的转向机制基础。
- **CAA (Contrastive Activation Addition)**：Rimsky et al. (2024) 提出的通过在隐藏层添加对比激活向量来实现属性控制的探测方法。
- **VLSafe**：Chen et al. (2024) 提出的多模态视觉-语言安全对齐数据集，用于 AutoSteer 的 prober 和 Refusal Head 训练。
- **ToViLaG⁺**：本文重构的多模态毒性基准，涵盖 Text-only / Image-only / Text+Image 三类攻击子集。

## 可复现要素
- **数据集**：VLSafe Examine（500 条）、ToViLaG⁺（500 条 × 3 子集）、RealWorldQA 子集（500 条）、MMMU 子集（500 条）——论文未提供公开下载链接，但各基准均有原文引用。
- **代码/权重**：论文未明确声明开源；基线 LM-Steer 有开源代码。
- **关键超参**：SAS 层选择样本数 N=3,000；Prober 为 1 层 MLP（64 维隐层 + ReLU + Sigmoid）；阈值 τ=0.5；转向强度 ϵ=0.1；Refusal Head 训练样本 3,000 条（LLaVA-OV 额外增强 3,000 条毒性图像）。
- **模型**：LLaVA-OV-7B、Chameleon-7B。
