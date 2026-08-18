---
title: "Re-Align-Aligning-Vision-Language-Models-via-Retrieval-Augme"
source: https://aclanthology.org/2025.emnlp-main.121.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:34:24"
field: "多模态大模型对齐与幻觉抑制"
keywords: ["Vision Language Models", "Hallucination Mitigation", "Direct Preference Optimization", "Retrieval-Augmented Generation", "Multimodal Alignment", "rDPO", "POPE", "HallusionBench"]
innovations: ["基于图像检索的偏好数据构建Pipeline，生成语义连贯的自然幻觉样本", "提出rDPO损失函数，融合文本与视觉双偏好信号进行VLM对齐", "在多种模型架构与规模（1B-13B）上验证方法的可扩展性与鲁棒性"]
benchmarks: ["POPE", "HallusionBench", "ScienceQA", "TextVQA", "MMBench", "MME", "MM-Vet", "VisWiz", "LLaVABench"]
---

# 论文速读：Re-Align: Aligning Vision Language Models via Retrieval-Augmented Direct Preference Optimization

## 一句话总结
论文提出了 **RE-ALIGN** 框架，通过**图像检索**构建语义连贯的偏好数据集，并结合新增的**视觉偏好优化目标（rDPO）**，在有效缓解 VLM 跨模态幻觉的同时，提升了通用视觉问答（VQA）性能。

## 研究问题与动机
- **VLM 跨模态幻觉严重**：现有 VLM 在图像理解中常产生与输入图像不一致或虚构的细节，根源在于视觉编码器与 LLM 主干分离预训练、数据质量不均及架构失衡。
- **现有 DPO 偏好数据构建方式粗糙**：主流方法通过对 GT 响应做扰动或对视觉输入加噪声来生成 rejected 样本，生成的幻觉不够自然、可控性差；而基于人工/模型修正响应生成 chosen 样本的方式成本高昂，难以扩展。
- **纯文本偏好优化忽视视觉信号**：标准 DPO 仅利用文本层面的偏好信号进行微调，未能充分引导模型关注视觉信息，可能加剧对语言先验的过度依赖。
- **现有对齐方法泛化能力有限**：部分基线（如 mDPO）通过随机裁剪输入图像引入视觉偏好，缺乏语义连贯性，且未能显著提升通用任务性能。

## 核心贡献（创新点）
1. **提出基于图像检索的偏好数据构建 Pipeline**：通过策略性遮蔽 + 图像检索 + 诱导幻觉三步生成自然、语义连贯的 chosen/rejected 偏好对，区别于 POVID/STIC 等基于噪声注入或随机裁剪的方法。
2. **提出 rDPO 损失函数**：在标准 DPO 基础上增加视觉偏好优化项 $\mathcal{L}_{\mathrm{vDPO}}$，使模型在优化过程中同时利用原始图像与检索图像的视觉差异，与 mDPO 的随机裁剪策略形成本质区别。
3. **系统性验证 RE-ALIGN 的广泛适用性**：在 LLaVA、Qwen2.5-VL、Janus-Pro 等多种架构与规模（1B–13B）模型上验证了方法的有效性与可扩展性。
4. **公开详细的实验分析与消融**：揭示了相似度阈值、遮蔽粒度、损失权重配比及训练轮数对性能的影响，为后续研究提供可复现的实验设计参考。

## 方法详解

### 3.1 偏好数据生成（Preference Generation）
采用三步流水线构建双偏好数据集：

1. **策略性遮蔽（Strategic Masking）**：给定输入 $(x_i, v_i)$ 及预训练 VLM 生成的 chosen 响应 $y_w$，通过 Prompt 将响应中与图像对象、属性或逻辑关系相关的词段替换为 `[MASK]`，得到遮蔽响应 $y_m$。
2. **图像检索（Image Retrieval）**：将所有训练图像通过原始视觉编码器 $f_v(\cdot)$ 编码为向量，构建向量库；利用 FAISS 与余弦相似度检索与 $v_i$ 最相似的 top-k 图像 $\{v_{j_1}, \cdots, v_{j_k}\}$。
3. **诱导幻觉（Inducing Hallucinations）**：用遮蔽响应 $y_m$、指令 $x_i$ 及第 $t$ 个检索图像 $v_{j_t}$ 提示 VLM 生成候选补全 $y_c$；通过 SentenceTransformer（all-mpnet-base-v2）计算 $y_w$ 与 $y_c$ 的余弦相似度，若低于阈值 $\tau = 0.95$ 则将其标记为 rejected 响应 $y_l$，否则继续尝试下一个检索图像。

### 3.2 rDPO 损失函数（Preference Optimization）
定义视觉偏好优化目标：

$$
\mathcal{L}_{\mathrm{vDPO}} = - \mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w | x, v)}{\pi_0(y_w | x, v)} - \beta \log \frac{\pi_\theta(y_w | x, v_l)}{\pi_0(y_w | x, v_l)} \right) \right]
$$

其中 $v_l$ 为检索图像，代表与原始图像语义相近但视觉 grounding 次优的替代输入。最终损失为：

$$
\mathcal{L}_{\mathrm{rDPO}} = \mathcal{L}_{\mathrm{DPO}} + \mathcal{L}_{\mathrm{vDPO}}
$$

**设计思想**：$\mathcal{L}_{\mathrm{vDPO}}$ 鼓励模型在原始图像 $v$ 下的响应优于在检索图像 $v_l$ 下的响应，从而强化模型对视觉细节的敏感度；与 mDPO 随机裁剪的方式相比，检索图像保留了完整的语义场景，对比信号更清晰。

## 实验与结果

### 实验设置
- **偏好数据集**：从 LLaVA-Instruct-150K 中采样 11k 图像，使用 GPT-4o mini 生成 QA 对，clip-vit-large-patch14 编码图像构建检索库。
- ** Backbone**：LLaVA-v1.5-7B、LLaVA-v1.6-Mistral-7B，及扩展至 Qwen2.5-VL、Janus-Pro 等系列。
- **训练配置**：LoRA（r=128, α=256），1 epoch，batch size=1，梯度累积=8，β=0.1，lr=1e-5，warmup=0.03，cosine schedule。

### 幻觉检测基准（Table 1）
| 模型 | POPE f | POPE p | POPE a | Hallusion f | Hallusion Easy | Hallusion Hard |
|---|---|---|---|---|---|---|
| LLaVA-v1.5-7B vanilla | 88.14 | 87.23 | 85.10 | 10.33 | 41.76 | 40.23 |
| **w. RE-ALIGN** | **88.65** ↑0.51 | **87.43** ↑0.20 | **85.16** ↑0.06 | **11.21** ↑0.88 | **45.52** ↑3.76 | **41.63** ↑1.40 |
| LLaVA-v1.6-Mistral-7B vanilla | 88.83 | 87.93 | 86.43 | 13.63 | 47.47 | 33.49 |
| **w. RE-ALIGN** | **90.55** ↑1.72 | **89.20** ↑1.27 | **87.03** ↑0.60 | **13.85** ↑0.22 | **48.35** ↑0.88 | **34.88** ↑1.39 |

- RE-ALIGN 在 LLaVA-v1.5-7B 和 v1.6-Mistral-7B 上均取得 POPE 和 HallusionBench 最优。
- LLaVA-v1.5-13B 上 POPE f 从 88.07 提升至 **90.03**（↑1.96），显著优于所有基线（POVID +0.07，mDPO +0.03）。

### 通用 VQA 基准（Table 2）
- RE-ALIGN 在 SQA（68.10 vs 66.02）、TextVQA（58.55 vs 58.18）、MMBench（64.69 vs 64.60）等任务上均超越 vanilla 模型与所有对比基线。
- LLaVA-v1.6-Mistral-7B 上 SQA 达 76.47，TextVQA 达 64.08，平均排名 **1.25**（最优）。

### 可扩展性（Table 3）
- 在 Janus-Pro-1B 上 POPE f 从 85.46 提升至 **87.53**（↑2.07），提升幅度在所有模型中最大，验证了小模型的强劲增益。
- 在 Qwen2.5-VL 系列和 LLaVA-v1.6-Vicuna 系列上均稳定提升，表明方法对不同架构具有良好的泛化性。

### 消融实验要点
- **相似度阈值 τ**：τ=0.95 在幻觉抑制与通用性能之间取得最佳权衡（Table 4）。
- **遮蔽粒度**：segment-level 优于 sentence-level（SQA 68.10 vs 67.58，Table 5）。
- **损失权重 $w_v$**：$w_v=1.0$（即等权组合）效果最佳（Table 6）。
- **训练轮数**：1 epoch 已足够，2–3 epoch 性能稳定甚至略有提升，无过拟合迹象（Table 7）。

## 相关工作脉络
1. **DPO 类对齐方法**：Rafailov et al. (2024) 提出的 DPO 是本文基础；本文在其之上引入了视觉偏好信号。
2. **POVID**（Zhou et al., 2024a）：通过 GPT-4V 生成幻觉响应并注入图像噪声构建偏好对，仅依赖文本偏好信号，缺乏视觉语义连贯性。
3. **STIC**（Deng et al., 2024）：通过图像损坏/误导性 prompt 生成 rejected 样本，属于粗粒度输入扰动。
4. **mDPO**（Wang et al., 2024a）：引入条件偏好优化目标，但对视觉信号的利用依赖于随机裁剪 20% 图像区域，语义对比信号弱于本文的检索图像策略。
5. **RLHF-V**（Yu et al., 2024b）：基于细粒度人工修正反馈构建偏好数据，质量高但人工成本昂贵，本文的检索方案提供了可扩展的替代路径。
6. **CSR / SIMA**：基于自奖励机制或上下文自批判生成偏好对，均仅使用文本偏好，未引入视觉维度的对齐信号。

## 局限性与未来方向
- **对齐税（Alignment Tax）仍然存在**：RE-ALIGN 在部分通用 VQA 任务上未能达到 SOTA，甚至偶有低于 vanilla 模型的情况，作者指出需进一步探索消除对齐税的策略或找到更优的平衡点。
- **检索图像质量依赖训练集分布**：若训练集内图像多样性不足或语义相似度计算存在偏差，检索到的替代图像可能无法提供有效的对比信号。
- **阈值 τ 的敏感性**：相似度阈值需人工设定（本文取 0.95），在不同数据集上可能需要调参。
- **计算开销虽低但非零**：预处理阶段涉及大量图像编码与检索，以及 rejected 响应的生成，整体 pipeline 的自动化程度仍有提升空间。

## 研究启发与可借鉴点
1. **检索增强构建偏好数据的新范式**：将图像检索与策略性遮蔽结合以生成自然、语义连贯的 hallucination 样本，这一数据构建思路可直接迁移到其他多模态对齐任务（如图文生成、视觉 grounding）。
2. **双偏好（文本+视觉）损失设计**：rDPO 将视觉 grounding 差异显式建模为优化目标，启发了后续工作如何在 DPO 框架中融入多模态信号（如音频、3D）。
3. **Segment-level 偏好建模优于 Sentence-level**：消融实验表明更细粒度的遮蔽策略提供更清晰的监督信号，这一发现对多模态 SFT/DPO 的数据构造具有指导意义。
4. **高效且可扩展的训练配置**：仅需 1 epoch + LoRA + 11k 样本即可取得显著增益，证明了轻量级对齐方案的可行性，适合资源受限的研究场景。
5. **与小模型/统一架构的适配性**：在 Janus-Pro（1B/7B）和 Qwen2.5-VL 上均验证了有效性，表明该方法不局限于特定 backbone，可作为通用对齐插件使用。

## 关键术语表
- **RE-ALIGN**：本文提出的基于图像检索的 VLM 对齐框架，通过构建双偏好数据集和 rDPO 损失缓解幻觉。
- **rDPO（Retrieval-augmented DPO）**：在标准 DPO 基础上增加视觉偏好优化项 $\mathcal{L}_{\mathrm{vDPO}}$ 的损失函数扩展。
- **DPO（Direct Preference Optimization）**：无需独立奖励模型的离线偏好优化算法，直接最大化 preference pair 的对数概率差。
- **POPE**：评估 VLM 对象幻觉的核心基准，包含流行度（popularity）、常识性（commonsense）和属性（attribute）三个子任务。
- **HallusionBench**：综合评估 VLM 幻觉与视觉错觉的基准，涵盖 easy/hard 难度及 factuality/illusion 两类子项。
- **Strategic Masking**：将 chosen 响应中与图像相关的关键词段替换为 `[MASK]`，用于引导 VLM 在替代图像下生成 hallucinated 补全。
- **Alignment Tax**：对齐微调过程中模型通用能力可能下降的现象，本文在部分通用 VQA 任务上观察到该现象。
- **vDPO / Visual Preference Objective**：$\mathcal{L}_{\mathrm{vDPO}}$，鼓励模型在原始图像下的响应优于在检索图像下的响应，强化视觉 grounding。

## 可复现要素
- **数据集**：LLaVA-Instruct-150K（CC BY 4.0 公开）；POPE（MIT）、HallusionBench（BSD-3-Clause）、ScienceQA（MIT）、TextVQA（MIT）、MMBench（Apache-2.0）等均为公开基准。
- **代码/权重**：论文未明确声明代码开源仓库；使用了 LLaVA-v1.5/v1.6 及 Qwen2.5-VL、Janus-Pro 等公开模型权重。
- **关键超参**：β=0.1，learning rate=1e-5，LoRA r=128/α=256，threshold τ=0.95，top-k 检索图像数未明确说明（推测 k≥3），训练 1 epoch，batch size=1，gradient accumulation=8。
- **推理设备**：4× NVIDIA A6000 Ada，微调时间 30–93 min 不等（Table 9）。
- **数据构建成本**：约 $90（GPT-4o mini），HallusionBench/LLaVABench 评测约 $30（GPT-4）。
