---
title: "Seeing-More-Saying-More-Lightweight-Language-Experts-are-Dyn"
source: https://aclanthology.org/2025.emnlp-main.28.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:45:16"
field: "多模态大模型效率优化"
keywords: ["动态令牌压缩", "视频语言模型", "语义密度感知", "轻量级语言专家", "高效多模态"]
innovations: ["提出语言感知的动态令牌压缩器 CapPruner，利用语言序列长度自适应匹配视频语义密度", "设计语义密度感知监督机制，以强 LVLM 提取关键视觉线索作为压缩目标", "在保持竞争力的同时将 FLOPs 降低 49%"]
benchmarks: ["MVBench", "Video-MME", "MSVD-QA", "MSRVTT-QA", "ActivityNet-QA", "TGIF-QA"]
---

# 论文速读：Seeing-More-Saying-More-Lightweight-Language-Experts-are-Dyn

## 一句话总结
本文提出 LangDC，一种语言感知的动态视频令牌压缩器，利用轻量级语言专家（CapPruner）根据视频片段的语义密度自适应调整压缩比，在保证性能的同时将 FLOPs 降低 49%。

## 研究问题与动机
- 现有大型视频语言模型（LVLM）处理大量视觉令牌时计算成本高昂，而主流令牌压缩策略均采用固定压缩比，忽视了视频不同片段间语义密度的显著差异。
- 固定压缩比导致信息丰富的片段（如动态场景）表征不足，而静态/内容贫乏片段产生冗余计算。
- 对 MVBench 的分析表明：每个测试样本的最优 token 数量差异巨大（平均 354 个 vs Oracle 最高达 757 个），静态压缩策略无法匹配这种变异性。

## 核心贡献（创新点）
- **语言感知动态压缩框架**：利用轻量级语言模型生成软字幕令牌作为视觉表示，动态控制压缩比；与固定压缩方法本质区别在于压缩比由输入语义密度决定而非预设。
- **语义密度感知监督机制**：用强 LVLM（LLaVA-OneVision）提取关键视觉线索作为监督目标；与人工标注 caption 的本质区别是消除了标注者偏差，且 caption 长度与视频信息密度一致。
- **显著的算力效率提升**：相比 VideoGPT+ 减少 49% FLOPs（49.85T → 25.15T），MVBench 仅下降 1.6%；体现了动态压缩在保持关键视觉线索方面的有效性。

## 方法详解
- **整体架构**：基于 VideoGPT+，包含双视觉编码器（CLIP-ViT-L/14-336 + InternVideo2-stage-2-1B）、投影层、基础池化器（BasePruner，默认 4×4 AvgPool）和动态令牌修剪器（CapPruner）。
- **CapPruner 设计**：由轻量级语言模型（Qwen-2.5-0.5B）加两个投影层构成。语言建模头用于监督训练（生成 token 并控制长度，padding token 表示压缩完成），post-projector 将隐藏状态对齐至 LLM 嵌入空间；实践中中间层（第 15 层）的隐藏状态效果最佳。
- **语义密度感知监督**：用 LLaVA-OneVision 作为教师模型提取视频片段关键视觉线索，再用 Qwen2.5-7B 精简冗余语言，得到语义密度适配的监督信号。
- **三阶段训练**：① 跨模态预训练（训练投影层，冻结其他组件）；② CapPruner 预训练（用基础 caption 数据集 + 语义密度感知监督训练 CapPruner 和视觉编码器投影层）；③ 有监督微调（LLM 使用 LoRA rank=128，训练连接投影层，冻结其余组件）。

## 实验与结果
- **数据集**：MVBench（多项选择）、Video-MME（有/无字幕）、MSVD-QA、MSRVTT-QA、ActivityNet-QA、TGIF-QA。
- **基线**：VideoGPT+（3.8B 参数）、Video-LLaVA、ST-LLM、VideoChat2 等。
- **核心结果**：
  - 相比 VideoGPT+，FLOPs 从 49.85T 降至 25.15T（↓49%），MVBench 仅从 58.7 降至 57.1（↓1.6%）；Video-MME 无字幕 44.3 vs 44.5，有字幕 51.3 vs 49.9（↑1.4%）。
  - 开源端 VideoQA：MSVD-QA 准确率 74.0（vs 72.4），TGIF-QA 76.8（vs 74.6）。
  - 令牌数从 3328 压缩至约 1068（平均），性能匹敌 3 倍令牌数的 AvgPooling。
  - 与 LDPv2 结合后平均 748 令牌，超越 C-Abstractor 和 Resampler。
- **消融**：CapPruner 预训练带来 +9.12% 提升；仅用 CapPruner（236 令牌）即达 51.50%，优于 8×8 池化（49.50%，208 令牌）。

## 相关工作脉络
- **VideoGPT+**：本文基础架构，采用固定压缩策略；LangDC 在此基础上引入动态压缩模块 CapPruner。
- **Q-Former / Resampler**：基于 cross-attention 的固定 token 压缩方法，无法适应不同语义密度；LangDC 以语言序列长度感知替代固定比例。
- **LDPv2 / C-Abstractor**：基于卷积的压缩方法，保留空间结构但压缩比固定；LangDC 可与它们结合使用，进一步提升效率。
- **LLaVA-OneVision**：作为教师模型提供语义密度感知监督，避免人工标注偏差。
- **Matryoshka Multimodal Models**（Cai et al., 2024）：同样关注可变 token 数量，但本文强调利用语言长度与语义密度的内在对应关系实现动态压缩。

## 局限性与未来方向
- 当前实验仅在 1.5B/3B LLM 规模验证，架构扩展至更大规模的效果未知。
- 单比率实现可能限制对专用视频 QA 任务的适应性，多比率融合具有探索空间。
- 未充分验证在超长视频（hourscale）上的动态压缩表现。

## 研究启发与可借鉴点
- **"Seeing more, saying more" 设计原则可迁移**：将语言序列长度与输入信息密度建立映射关系，可推广至图像、点云等其他模态的动态压缩。
- **教师模型辅助的监督信号构建**：用强 LVLM 生成带噪声但语义丰富的 caption，再经 LLM 精炼为高质量监督信号，该 Pipeline 可用于其他模态对齐任务。
- **中间层隐藏状态作为压缩表示**：比最后一层更具泛化性，这一观察对设计语言-视觉投影器具有参考价值。
- **三阶段渐进训练策略**：跨模态预训练 → 专家预训练 → SFT，可有效缓解语义鸿沟，适合多模态新模块的引入。

## 关键术语表
- **LangDC**：Language-aware Dynamic Token Compressor，一种语言感知的动态视频令牌压缩器。
- **CapPruner**：Caption-based Pruner，由轻量级语言模型构成的动态令牌修剪模块。
- **语义密度感知监督**：利用强 LVLM 提取视频关键视觉线索作为监督信号，确保压缩 token 数量与场景信息密度匹配。
- **软字幕令牌（soft caption tokens）**：语言模型预测文本 token 的隐藏状态，作为压缩后的视觉表示。
- **Oracle 指标**：对每个测试样本选择能获得正确响应的最高压缩比，用于评估最优 token 需求分布。
- **动态压缩比**：根据输入视频片段的语义丰富程度自适应调整压缩后 token 数量的策略。
- **Progressive training strategy**：跨模态预训练 → CapPruner 预训练 → SFT 的三阶段训练流程。
- **BasePruner**：基础令牌压缩器（如 AvgPooling），与 CapPruner 联合使用以提升压缩效果。

## 可复现要素
- **数据集**：CC-595K（跨模态预训练）、MVBench、Video-MME、MSVD-QA、MSRVTT-QA、ActivityNet-QA、TGIF-QA（SFT 及评测）。
- **代码**：已开源（https://github.com/NIneeeeeem/LangDC）。
- **模型权重**：CapPruner 基于 Qwen-2.5-0.5B，LLM 使用 Qwen-2.5-3B（论文未提及是否开源权重）。
- **关键超参**：16 帧、4 个视频片段、每段最大 128 压缩令牌、CapPruner 隐藏状态取第 15 层、LLM LoRA rank=128。
