---
title: "Image-Embedding-Sampling-Method-for-Diverse-Captioning"
source: https://aclanthology.org/2025.emnlp-main.156.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:53:02"
field: "视觉语言多模态理解"
keywords: ["image captioning", "diverse captioning", "vision-language model", "hierarchical representation", "zero-shot captioning", "embedding sampling"]
innovations: ["提出免训练层级化 patch embedding 采样框架 HBoP，以极小参数成本生成多样化图像描述", "设计 Global/Regional/Fine-grained 三层级 mask 采样策略，替代全图一次性编码", "引入 PCD 作为新多样性度量，基于 sentence embedding 成对余弦距离"]
benchmarks: ["MSCOCO", "Flickr30K", "NoCaps"]
---

# 论文速读：Image-Embedding-Sampling-Method-for-Diverse-Captioning

## 一句话总结
本文提出 HBoP（Hierarchical Bags of Phrases），一种**免训练的层级化图像嵌入采样框架**，通过将预训练的分割模型 SAM 与小型 VLM BLIP 结合，从不同层级的图像区域采样 patch embedding，从而以极小参数成本生成多样化且与图像语义对齐的图像描述。

## 研究问题与动机
- **大型 VLM 计算开销大**：SOTA 大模型（如 BLIP-2、LLaVA）能生成高质量描述，但参数量动辄数 B，难以部署于移动端或辅助技术场景。
- **小型 VLM 忽略细节、描述单一**：小模型（如 BLIP 446M）倾向于关注主导视觉元素，产出的 caption 高度重复、缺乏多样性，与人类标注差距较大。
- **现有多样性增强方法依赖额外训练或专用目标函数**（如 ModeCap、Seq-CVAE），无法直接迁移到轻量级无 LLM 的 VLM 上；而直接裁剪图像区域再描述的方式会损失全局上下文，导致相关性下降。
- **层级化视觉表示已被证明有助于多粒度图像理解**，但如何在"免训练"约束下利用这种表示来提升 caption 多样性尚待探索。

## 核心贡献（创新点）
1. **提出免训练的层级图像嵌入采样框架 HBoP**：无需微调任何模型，仅通过 SAM 分割 + ViT patch 采样 + BLIP 解码即完成多粒度描述生成，与已有需要训练多样性目标函数（如 ModeCap、Seq-CVAE）的方法本质不同。
2. **设计三层级采样策略（Global / Regional / Fine-grained）**：通过 NMS 选取最大 mask 作为全局层，K-means 聚类生成区域层，剩余 mask 作为细粒度层，实现从整体场景到局部细节的多粒度覆盖，区别于传统全图一次性编码方式。
3. **引入 Pairwise Cosine Distance (PCD) 作为新的多样性度量**：基于句子嵌入的成对余弦距离衡量同一图像多条 caption 的语义差异，弥补了传统 n-gram 指标无法有效捕捉语义多样性的不足。
4. **证明小模型可通过结构化嵌入采样媲美大模型多样性**：HBoP（1B）在 MSCOCO 上 Div-2=0.735、PCD=0.772，优于 BLIP-2（3.9B）、Honeybee-7B 等更大模型，同时保持 SBERT 相关性接近黄金标注，揭示了"好的输入表征比大模型本身更重要"的设计思路。

## 方法详解
HBoP 由三个模块构成，整体流程为：输入图像 → SAM 自动分割 → 层级化 patch embedding 采样 → BLIP Captioner 生成多粒度描述。

### 3.1 Image Segmentation Module (ISM)
- 使用 **Segment Anything Model (SAM)** 对输入图像进行全自动化分割，无需人工 prompt。
- 从 ViT encoder 最后一层的 patch embeddings $E_X$ 中，根据 SAM 生成的 mask $M_X = \{M_{X_1}, ..., M_{X_p}\}$ 筛选出对应区域的 patch embedding。

### 3.2 Hierarchical Composition Module (HCM)
将分割 mask 划分为三个层级：

- **全局层（Global）**：对 $M_X$ 做 NMS（IoU 阈值 0.1）后，选取面积最大的 top-k（k=5）个 mask：
  $$M_G = \text{NMS}(\text{Top-}k(M_X))$$
  对应嵌入 $E_G = \{E_{g_i}\}$，其中 $E_{g_i} = E_X \odot M_{g_i}$。

- **区域层（Regional）**：对所有 mask 的 bounding box 做 **K-means 聚类（K=5）**，再对每个聚类内做 NMS：
  $$M_R = \{M_{r_i}\} = \text{NMS}(\text{K-means}(M_X))$$
  每个 cluster 内对应的 patch embedding 拼接后再做 zero padding，得到 $E_R$。

- **细粒度层（Fine-grained）**：NMS 后的 mask 去除全局层部分：
  $$M_F = \text{NMS}(M_X) \setminus M_G$$
  对应嵌入 $E_F$。

### 3.3 Image Captioning Module (ICM)
- 使用 **BLIP captioner（446M，非 BLIP-2）** 作为生成器，对每一组分层 embedding 独立调用 stochastic sampling 解码。
- 由于各层 patch 数量不同，采用 zero padding 补齐后送入 BLIP decoder。
- 最终输出 $n_g + n_r + n_f$ 条 caption（实验中通常输出 2 global + 3 regional = 5 条）。

## 实验与结果
**数据集**：MSCOCO（Karpathy test 5k）、Flickr30K（test 1k）、NoCaps（validation 4.5k）。

**评估指标**：
- **相关性**：SBERT（与黄金标注语义相似度）、CLIP-Score
- **多样性**：mBLEU-4（越低越多样）、Div-2、PCD（新提出）
- **语义完整性**：LLaMA-2-13B、GPT-4 打分

**关键结果（MSCOCO）**：

| 模型 | 参数量 | SBERT ↑ | CLIP-S ↑ | PCD ↑ | mBLEU-4 ↓ | Div-2 ↑ |
|------|--------|---------|----------|-------|-----------|---------|
| BLIP (-NS) | 446M | 56.00 | 29.98 | 0.600 | 1.000 | 0.179 |
| BLIP (+NS) | 446M | 57.23 | 30.33 | 0.668 | 0.658 | 0.387 |
| BLIP-2 | 3.9B | 65.47 | 30.66 | 0.651 | 0.712 | 0.345 |
| Honeybee-7B | 7B | 53.55 | 28.21 | 0.792 | 0.062 | 0.716 |
| **HBoP（ours）** | **1B** | **56.30** | **29.12** | **0.772** | **0.049** | **0.735** |

- HBoP 相对 BLIP(-NS)：**mBLEU-4 降低超 60%**，**Div-2 提升超 30%**，SBERT 基本持平（56.30 vs 56.00）。
- HBoP 在 Div-2 和 PCD 上**超越 BLIP-2、Honeybee-7B/13B、LLaVA-1.5**，尽管参数仅为它们的 1/4~1/13。
- **Flickr30K**：Div-2=0.750，PCD=0.815，同样最优。
- **NoCaps**：Div-2=0.748，分列第 2。
- **消融**：直接裁剪区域再描述（Crop）虽提升 Div-2（0.600），但 SBERT 显著下降（52.03 vs 56.30），证明完整图像 embedding 采样更有效。
- IoU 阈值消融：0.1 为最优，阈值增大导致重叠区域增多、重复性上升、多样性下降。

**最强结果**：HBoP 在 MSCOCO 上以 1B 参数获得 Div-2=0.735，超越所有基线；PCD=0.772，优于 Honeybee-7B（0.792 的随机上限为 0.963，HBoP 最接近黄金标准）。

## 相关工作脉络
1. **ModeCap（Chen et al., 2022）**：通过显式学习"distinct and representative modes"目标函数增强多样性，需要专门训练；HBoP 无需训练，仅靠嵌入采样实现类似效果。
2. **Seq-CVAE（Aneja et al., 2019）**：在序列隐空间中建模多样性意图，依赖 CVAE 架构；HBoP 不改变生成架构，仅改变视觉输入表征。
3. **BLIP / BLIP-2（Li et al., 2022, 2023）**：BLIP-2 借助 LLM 蒸馏增强描述质量但参数量大；本文选用原始 BLIP captioner 配合层级采样，以极低成本获得可媲美 BLIP-2 的多样性。
4. **GradCAM / DINOv2-based 区域定位**：GradCAM 只能突出特定任务相关的 saliency region，无法全覆盖语义区域；SAM 可自动分割所有有意义区域，HBoP 在此基础上做层级聚合。
5. **Honeybee（Cha et al., 2023）、LLaVA（Liu et al., 2023）**：大参数多模态 LLM，依赖 LLM 解码能力；HBoP 不使用 LLM，纯靠小 VLM + 结构化视觉输入获得高多样性。
6. **PnP-VQA（Tiong et al., 2022）**：将 VQA 范式用于 zero-shot captioning；HBoP 直接采用 BLIP captioner，不借助 VQA 桥接，推理更简洁。

## 局限性与未来方向
- **当前使用 bounding box 近似 mask 提取 patch embedding**，可能丢失不规则形状的细节；未来可探索直接用完整 irregular mask 进行 embedding 采样。
- **仅增强事实性多样性（factual diversity）**，对需要文化背景知识的描述多样性（cultural diversity）效果有限，因后者依赖视觉特征之外的外部知识。
- **SAM 引入额外推理耗时**（MSCOCO 上约 5.43s/image），虽可用 FastSAM 降至 0.18s，但仍显著高于 BLIP 单步推理（0.38s/caption）。
- **细粒度层（Fine-grained）未纳入主实验评估**，仅作为附加标签使用。

## 研究启发与可借鉴点
1. **"好输入 > 大模型"的设计哲学**：通过改进视觉输入的表征结构（层级采样），而非扩大模型规模，即可显著提升多样性。这对资源受限场景具有重要参考价值，可迁移至其他视觉理解任务（如 VQA、视觉定位）的零样本增强。
2. **免训练框架的可复用性**：HBoP 完全基于预训练模型组合，无需微调，可直接套用到其他小型 VLM（如 IDEFICS、BLIP 变体）上验证泛化性，降低了实验门槛。
3. **PCD 指标可作为通用多样性度量**：基于 sentence embedding 的成对余弦距离避免了 n-gram 指标对词汇重叠的过度惩罚，适用于任何"多条描述"的多样性评估场景。
4. **K-means 聚类 mask bounding box 的层级划分策略**：简单但有效，可作为其他多粒度视觉任务（如 scene graph generation、object detection post-processing）的特征组织范式。
5. **消融对比设计（Crop vs Embedding Sampling）**：清晰证明了保留全局上下文的重要性，此类"同输入不同处理方式"的消融对后续工作有很强的方法论示范价值。

## 关键术语表
- **HBoP（Hierarchical Bags of Phrases）**：本文提出的免训练层级化图像描述生成框架，通过多层级 patch embedding 采样实现多样化 caption 生成。
- **SAM（Segment Anything Model）**：Meta 推出的通用图像分割模型，可自动对图像中所有有意义区域生成高质量 mask，无需任务特定训练。
- **PCD（Pairwise Cosine Distance）**：论文提出的新多样性度量，计算同一图像多条 caption 的 sentence embedding 间平均余弦距离，值越大表示多样性越高。
- **Stochastic Sampling**：BLIP captioner 使用的解码策略，在每一步从 top-p 候选词中随机采样，有助于增加输出多样性。
- **Div-2**：n-gram 级别的词汇多样性指标，定义为生成的 2-gram 种类数占总 2-gram 数的比例，范围 [0,1]，越高越多样。
- **NMS（Non-Maximum Suppression）**：用于去重重叠 mask 的操作，按 IoU 阈值合并相似区域，是 HCM 中层级划分的关键步骤。
- **BLIP vs BLIP-2**：BLIP 是单阶段 ITC+ITM+LM 预训练的小 VLM（446M captioner）；BLIP-2 引入 Q-Former 连接冻结 ViT 与 LLM（如 OPT），参数量大幅增加。
- **SBERT**：基于 Sentence-BERT 的句子级语义相似度度量，用于衡量生成 caption 与人工标注之间的语义对齐程度。

## 可复现要素
- **数据集**：MSCOCO（公开）、Flickr30K（公开）、NoCaps（公开）；均使用标准 split。
- **代码**：论文声明代码已开源，地址 https://github.com/xfactlab/HBoP
- **权重**：使用预训练的 SAM（ViT-L/16）和 BLIP captioner（446M），无需重新训练。
- **关键超参**：top-k（全局层）= 5，K-means 聚类数 K = 5，NMS IoU 阈值 = 0.1，每图像输出 5 条 caption（2 global + 3 regional）。
