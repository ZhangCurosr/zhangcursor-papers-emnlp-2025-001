---
title: "Re-Align-Aligning-Vision-Language-Models-via-Retrieval-Augme"
source: https://aclanthology.org/2025.emnlp-main.121.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:55:48"
field: "多模态大模型对齐与幻觉缓解"
keywords: ["Vision-Language Models", "Hallucination Mitigation", "Direct Preference Optimization", "Retrieval-Augmented Generation", "Multimodal Alignment", "rDPO"]
innovations: ["基于图像检索的段级掩码自动生成文本+视觉双偏好数据集", "提出rDPO损失（L_DPO+L_vDPO）实现跨模态对比偏好优化", "在1B-13B多种架构VLM上验证可扩展性与SOTA幻觉缓解效果"]
benchmarks: ["POPE", "Hallusion-Bench", "ScienceQA", "TextVQA", "MM-Vet", "MME", "MMBench", "LLaVABench"]
---

# 论文速读：Re-Align: Aligning Vision-Language-Models-via-Retrieval-Augmented-Direct-Preference-Optimization

## 一句话总结
论文提出了 **RE-ALIGN** 框架，通过**图像检索**构建"文本+视觉"双偏好数据集，并结合新增的 **rDPO** 目标函数进行微调，在有效缓解 VLM 幻觉的同时提升通用 VQA 任务性能。

## 研究问题与动机
- VLM 在跨模态场景中存在严重的**幻觉问题**（hallucination），即输出包含与输入图像不符的虚假或不准确细节。
- 现有 DPO 对齐方法多通过**暴力扰动 ground truth 响应**或**篡改视觉输入/embedding**来构造拒绝样本，方式过于简单，无法生成自然且可控的幻觉。
- 直接将 DPO 应用于 VLM 微调时，模型可能**过度偏向语言层面偏好**，反而削弱跨模态对齐能力，甚至加剧幻觉。

## 核心贡献（创新点）
1. **提出基于图像检索的偏好数据自动构建流程**：通过战略性掩码 + 图像检索 + 幻觉诱导三段式流水线，自动构造语义合理、自然的 rejected 响应；与人工修正/简单扰动方法相比，兼具高质量与可扩展性。
2. **设计 rDPO 损失函数（视觉偏好优化扩展）**：在标准 DPO 目标基础上额外引入 $\mathcal{L}_{\text{vDPO}}$，利用检索到的相关图像构建视觉偏好对比对；与 mDPO 随机裁剪图像的方式相比，提供了更强、更语义一致的视觉对比信号。
3. **在多样模型架构与尺寸下验证可扩展性**：覆盖 LLaVA-v1.5/v1.6、Qwen2.5-VL、Janus-Pro 等图像→文本与统一架构 VLM，最小仅 1B 参数仍获得显著增益，证明方法的普适性。

## 方法详解
### 3.1 偏好数据构建（三阶段）
- **战略掩码（Strategic Masking）**：给定输入 $(x_i, v_i)$ 及其 chosen 响应 $y_w$，将响应中与图像中对象、属性、逻辑关系相关的词/片段替换为 `[MASK]`，得到 $y_m$。
- **图像检索（Image Retrieval）**：用预训练 VLM 的视觉编码器将所有训练图像编码，基于余弦相似度检索与 $v_i$ 最相似的 top-k 图像集合。
- **幻觉诱导（Inducing Hallucinations）**：用 VLM 以指令 $x$ + 检索图像 $v_{j_t}$ + 掩码响应 $y_m$ 为条件生成候选补全 $y_c$；通过 SentenceTransformer 计算 $y_c$ 与 $y_w$ 的文本嵌入余弦相似度，若低于阈值 $\tau=0.95$ 则取 $y_c$ 为 rejected 响应 $y_l$，否则尝试下一张检索图像。

### 3.2 rDPO 损失函数
标准 DPO 目标：
$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x,v)}{\pi_0(y_w|x,v)} - \beta\log\frac{\pi_\theta(y_l|x,v)}{\pi_0(y_l|x,v)}\right)\right]$$

新增视觉偏好目标：
$$\mathcal{L}_{\text{vDPO}} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|x,v)}{\pi_0(y_w|x,v)} - \beta\log\frac{\pi_\theta(y_w|x,v_l)}{\pi_0(y_w|x,v_l)}\right)\right]$$

组合损失：$\mathcal{L}_{\text{rDPO}} = \mathcal{L}_{\text{DPO}} + \mathcal{L}_{\text{vDPO}}$（实验中 $w_v=1.0$ 取得最佳权衡）。

## 实验与结果
- **数据集与基准**：幻觉检测使用 POPE、Hallusion-Bench；通用 VQA 使用 ScienceQA、TextVQA、MM-Vet、VisWiz、LLaVABench、MME、MMBench。
- **主干模型**：LLaVA-v1.5-7B、LLaVA-v1.6-Mistral-7B 为主实验基座；并扩展到 1B–13B 多种模型。
- **主要结果（Table 1 & 2）**：
  - LLaVA-v1.5-7B + RE-ALIGN：**POPE$^c$ 88.65**（vs. vanilla 88.14，+0.51）、**Hallusion Easy 45.52**（vs. vanilla 41.76，+3.76），均为当前最优。
  - LLaVA-v1.6-Mistral-7B + RE-ALIGN：**POPE$^c$ 90.55**（vs. STIC 89.03）、**ScienceQA 76.47**，综合排名平均第一。
- **可扩展性（Table 3）**：在 Janus-Pro-1B 上 POPE$^c$ 提升 **+2.07**；在 LLaVA-v1.5-13B 上提升 **+1.96**，均优于所有基线。
- **消融（Tables 4–7）**：相似度阈值 $\tau=0.95$ 为最佳权衡；segment-level 掩码优于 sentence-level；$\mathcal{L}_{\text{vDPO}}$ 加权 $w_v=1.0$ 表现最优；训练 1–3 epoch 结果稳定。

## 相关工作脉络
- **POVID (Zhou et al., 2024a)**：通过 GPT-4V 注入噪声图像并生成幻觉响应，偏好信号仅依赖文本；RE-ALIGN 以检索图像替代噪声扰动，幻觉更自然、语义一致性更强。
- **STIC (Deng et al., 2024)**：通过裁剪图像或误导性提示构造拒绝样本；属低层次视觉扰动，缺乏语义层面的跨模态对比。
- **mDPO (Wang et al., 2024a)**：引入条件视觉偏好损失，但通过**随机裁剪**原始图像构造 $v_c$；RE-ALIGN 采用同库检索的最相似图像 $v_l$，提供语义更合理的对比情境。
- **CSR / SIMA**：基于自奖励/自批评机制构造偏好对，但偏好信号仅含文本；RE-ALIGN 额外引入视觉对比对，实现真正的双模态偏好对齐。
- **RLHF-LLaVA (Sun et al., 2023)**：依赖人工标注偏好数据，成本高昂且难以扩展；RE-ALIGN 完全自动化构建，仅需 GPT-4o mini 辅助掩码，总成本约 $90。

## 局限性与未来方向
- **对齐税（alignment tax）**：RE-ALIGN 在部分通用 VQA 基准上未能超越 vanilla 模型，甚至略低于某些基线，作者承认尚未完全消除该折损。
- **偏好数据规模依赖**：当前使用 11k 图像对，进一步扩量可能带来更大收益（文中已展示 11k→16k 的持续增益趋势），但成本随之上升。
- **鲁棒性风险**：如作者在 Limitations 中所述，更强的对齐技术可能被恶意利用生成有害输出；未来需在安全护栏层面配套研究。

## 研究启发与可借鉴点
- **检索增强构建 preference 数据**：用图像检索替代纯文本扰动/噪声注入，使 rejected 样本具备语义合理性与视觉多样性，该思路可迁移至文本/音频等其他模态的对齐数据构造。
- **段级（segment-level）掩码策略**：相比于句子级或 token 级扰动，段级掩码保留更多上下文，产生更精细的偏好监督信号；后续可探索层级化掩码（词/短语/段落多级粒度）。
- **双偏好目标 $\mathcal{L}_{\text{DPO}} + \mathcal{L}_{\text{vDPO}}$ 的设计模式**：将视觉模态对比作为独立的偏好损失项，可与 DPO 变体（KTO、IPO、ORPO 等）结合，拓展至更强的对齐算法框架。
- **离线预处理 + 在线轻量训练**：检索、掩码、偏好配对全部离线完成，训练阶段仅额外增加一个 forward pass 开销（5–10%），可无缝集成到现有 VLM 训练 pipeline 中。

## 关键术语表
- **RE-ALIGN**：本文提出的 VLM 对齐框架，核心思想是结合图像检索与 DPO 生成双模态偏好数据并微调。
- **rDPO（Retrieval-augmented DPO）**：在标准 DPO 基础上加入视觉偏好优化目标 $\mathcal{L}_{\text{vDPO}}$ 的损失函数扩展。
- **$\mathcal{L}_{\text{vDPO}}$**：视觉偏好损失，通过对比"原图 vs. 检索相似图"下 chosen 响应的概率差，引导模型基于真实视觉内容生成响应。
- **POPE**：Object Hallucination 评测基准，从图像中采样命题令模型判断 true/false/hallucination。
- **Hallusion-Bench**：针对 VLM 中语言幻觉与视觉错觉混合场景的诊断性评测基准。
- **Strategic Masking**：将 chosen 响应中与图像实体/属性/关系相关的词替换为 `[MASK]` 的策略，用于后续诱导幻觉补全。
- **Alignment Tax**：偏好对齐过程可能削弱模型在通用任务上的表现，形成"对齐 vs. 能力"的权衡现象。

## 可复现要素
- **数据集**：偏好数据基于 LLaVA-Instruct-150K 采样 11k 图像构建（CC BY 4.0 许可）；评测基准（POPE、Hallusion-Bench、ScienceQA、TextVQA、MM-Vet 等）均公开可获取。
- **代码/权重**：论文未明确声明代码仓库与模型权重的开源链接，需自行实现检索管道与 rDPO 损失。
- **关键超参**：$\beta=0.1$、learning rate $1\times10^{-5}$、相似度阈值 $\tau=0.95$、LoRA $r=128/\alpha=256$、training epochs=1、gradient accumulation=8、batch size=1 per device；视觉编码器为 `clip-vit-large-patch14`，文本编码器为 `all-mpnet-base-v2`。
