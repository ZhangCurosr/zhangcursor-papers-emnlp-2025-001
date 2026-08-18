---
title: "Image-Embedding-Sampling-Method-for-Diverse-Captioning"
source: https://aclanthology.org/2025.emnlp-main.156.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:42:43"
field: "多模态表示学习与图像描述生成"
keywords: ["image captioning", "diversity", "vision-language models", "hierarchical representation", "segment anything model"]
innovations: ["提出免训练的HBoP框架，通过分层图像嵌入采样增强小VLM的描述多样性", "设计三级分层结构（全局/区域/细粒度）控制描述粒度，平衡多样性与相关性", "利用全图embedding采样而非裁剪，保留上下文信息同时提升多样性指标"]
benchmarks: ["MSCOCO", "Flickr30K", "Nocaps"]
---

# 论文速读：Image-Embedding-Sampling-Method-for-Diverse-Captioning

## 一句话总结
论文提出HBoP（Hierarchical Bags of Phrases）框架，通过分层图像嵌入采样方法，在无需额外训练的情况下，利用小规模VLM（BLIP）生成更多样化且信息丰富的图像描述，在多样性指标上显著优于基线方法，同时保持与图像和人工标注的语义相关性。

## 研究问题与动机
- **大模型计算成本高**：SOTA VLM虽然能生成高质量、多样化的描述，但参数量大、计算复杂度高，难以部署在移动端、辅助技术等资源受限场景。
- **小模型忽视细粒度细节**：小型VLM倾向于关注主导视觉元素，忽略细粒度细节，导致生成的描述缺乏深度和具体性。
- **现有方法缺乏层次化建模**：传统captioning方法将图像视为整体，未显式建模全局与局部信息的层次结构；专门的多样性方法（如ModeCap、Seq-CVAE）需要额外训练且效果有限。
- **分割与描述的结合潜力未充分挖掘**：结构化分割可捕获图像的层次语义，但如何将分割结果与预训练VLM有效结合以增强描述多样性尚待探索。

## 核心贡献（创新点）
1. **提出免训练的层次化图像嵌入采样框架HBoP**：结合预训练分割模型（SAM）和小型VLM（BLIP），通过分层采样图像patch嵌入生成多粒度描述，无需额外训练。
2. **设计三级分层结构控制描述粒度**：将分割掩码划分为全局（大区域）、区域（聚类分组）和细粒度（剩余小区域）三个层级，分别生成对应粒度的描述。
3. **在保持相关性的同时显著提升描述多样性**：HBoP在保持与图像和人工标注的语义相关性（SBERT、CLIP-Score）的同时，diversity指标（PCD、Div-2）大幅优于小型VLM基线，接近甚至超越大型LLM-based模型。
4. **通过嵌入采样而非直接裁剪保留全局上下文**：相比直接裁剪分割区域再caption，HBoP从全图embedding中采样保留上下文信息，避免相关性下降。

## 方法详解
**整体框架**：HBoP由三个模块组成：
1. **Image Segmentation Module (ISM)**：使用Segment Anything Model (SAM) 对图像进行自动分割，生成多个分割掩码 $M_X$，并从ViT编码器输出的patch embeddings $E_X$ 中筛选对应掩码区域的嵌入。

2. **Hierarchical Composition Module (HCM)**：对分割掩码进行三层划分：
   - **全局层级**：选取面积最大的top-k（k=5）掩码，经NMS去重后得到 $M_G$
   - **区域层级**：对全部掩码的bounding box进行K-means聚类（K=5），对每个聚类独立应用NMS得到 $M_R$
   - **细粒度层级**：剩余掩码（去重且非全局）得到 $M_F$
   
   各层级通过掩码筛选对应的patch embedding，不足部分用zero padding补齐。

3. **Image Captioning Module (ICM)**：使用预训练的BLIP captioning模型，对每个层级的embeddings分别生成描述：
   - $\mathrm{HBoP}_G = \{s_{g_1}, ..., s_{g_{n_g}}\}$，其中 $s_{g_i} = \mathrm{BLIP}(E_{g_i})$
   - 类似地生成区域级和细粒度级描述
   - 最终合并多层级描述作为输出

**关键公式**：
- 掩码选择：$E_{g_i} = E_X \odot M_{g_i}$（逐元素乘法筛选patch嵌入）
- NMS去除重叠掩码：基于IoU和置信度阈值
- PCD多样性度量：$\mathrm{PCD}(s_1,...,s_n) = \frac{1}{n}\sum_{i=1}^n\sum_{j<i}(1-\cos(M(s_i), M(s_j)))$

## 实验与结果
**数据集**：MSCOCO（5k test）、Flickr30K（1k test）、Nocaps（4.5k validation）

**评估指标**：
- 相关性：SBERT（与人工标注语义相似度）、CLIP-Score（图像-文本对齐）
- 多样性：mBLEU-4（越低越好）、Div-2、PCD（ pairwise cosine distance）
- 语义完整性：LLaMA-2-13B、GPT-4评分

**主要结果**（Table 1）：
- **MSCOCO**：HBoP获得最高Div-2（0.735）、最高PCD（0.772）、最低mBLEU-4（0.049），SBERT（56.30）和CLIP-S（29.12）与BLIP相当
- **Flickr30K**：HBoP Div-2达0.750，PCD达0.815，明显优于BLIP（Div-2: 0.179, PCD: 0.600）
- **Nocaps**：HBoP在三项多样性指标上均排名2/7
- **对比大型模型**：HBoP（1B参数）在多样性上超越BLIP-2（3.9B）、Honeybee-7B/13B、LLaVA-1.5（13B）
- **与专门多样性方法对比**：HBoP的Div-2和PCD显著优于Seq-CVAE、ModeCap

**消融实验**：
- IoU阈值消融（Table 2）：阈值从0.1到0.8，PCD和Div-2均下降，0.1最优
- Crop vs Embedding Sampling（Table 3）：直接裁剪提升多样性但降低SBERT相关性，HBoP嵌入采样保持相关性同时获得更好多样性

**关键数字**：HBoP相比BLIP(-NS)，mBLEU-4降低超60%（1.0→0.049），Div-2提升超30%（0.179→0.735），SBERT保持相当（56.00→56.30）。

## 相关工作脉络
- **基础VLM工作**：CLIP（Radford et al., 2021）、BLIP（Li et al., 2022）、BLIP-2（Li et al., 2023）等利用大规模对比学习和预训练增强VLM对齐能力，但多生成高层场景描述。
- **多样性captioning方法**：Seq-CVAE（Aneja et al., 2019）通过序列潜空间建模意图实现多样性；ModeCap（Chen et al., 2022）学习distinct modes。本文HBoP无需额外训练目标即可实现更高多样性。
- **层次化表示方法**：SHAN（Ji et al., 2021）、HiVLP（Shao et al., 2023）、ViCHA（Shukor et al., 2022）等探索图像-文本的层次对齐，但本文从分割掩码采样embeddings的角度切入，更轻量。
- **视觉分割模型**：SAM（Kirillov et al., 2023）提供高质量通用分割；FastSAM（Zhao et al., 2023）加速推理。本文利用SAM的自动分割能力生成层次化区域。
- **特征可视化方法**：GradCAM（Selvaraju et al., 2017）、DINOv2（Oquab et al., 2023）用于揭示模型关注区域，本文展示HBoP生成的描述与人工标注在GradCAM热力图上更一致。

## 局限性与未来方向
- **边界框近似限制**：当前实现使用bounding box近似提取嵌入，可能遗漏不规则形状或细粒度细节；未来可探索使用完整的不规则分割掩码。
- **文化多样性捕捉不足**：方法主要增强事实性多样性（不同图像区域），但对于依赖外部文化知识的领域（如文化多样性描述）效果有限。
- **推理开销**：SAM分割引入额外时间（5.43s/img），但可用FastSAM降至0.18s/img平衡质量与效率。

## 研究启发与可借鉴点
1. **分层嵌入采样范式**：将层次化分割与预训练VLM的patch embedding采样结合的思路可迁移至其他多模态任务（如VQA、图像检索），通过控制采样粒度提升表现。
2. **免训练的轻量化增强**：HBoP无需微调即显著提升小模型能力，这种"prompting"式的方法设计值得借鉴，尤其适合资源受限场景。
3. **多样性与相关性的平衡策略**：通过全图embedding采样而非裁剪，避免损失上下文信息，这一设计原则可用于其他需要兼顾多样性和质量的生成任务。
4. **多维度评估体系**：同时使用n-gram多样性（mBLEU-4, Div-2）、语义多样性（PCD）、相关性（SBERT, CLIP-S）和人工评估（LLM评分），为 captioning 研究提供了全面的评估框架参考。
5. **与团队协作机会**：可将HBoP的分层采样思想与团队现有工作结合，如在地理定位（visual place recognition）或跨模态检索中探索层次化视觉表征的利用。

## 关键术语表
**VLM (Vision-Language Model)**：结合视觉和语言理解的神经网络模型，可执行图像描述、视觉问答等任务。
**SAM (Segment Anything Model)**：Facebook提出的通用图像分割模型，可自动将图像分割为多个语义有意义的区域。
**HBoP (Hierarchical Bags of Phrases)**：本文提出的分层短语袋框架，通过层次化图像嵌入采样生成多样化描述。
**PCD (Pairwise Cosine Distance)**：本文提出的多样性度量指标，计算同一图像所有描述对的语义余弦距离平均值。
**mBLEU-4**：修改后的BLEU-4指标，衡量生成描述与参考描述的字面重叠度，越低表示多样性越高。
**Div-2**：n-gram多样性指标，计算描述中2-gram的唯一比例，越高表示多样性越好。
**BLIP**：Bootstrapped Language-Image Pretraining模型，本文作为基础captioning模型使用（446M参数）。
**NMS (Non-Maximum Suppression)**：非极大值抑制，用于去除重叠度过高的分割掩码。

## 可复现要素
- **数据集**：MSCOCO、Flickr30K、Nocaps（均为公开数据集）
- **代码**：已开源，https://github.com/xfactlab/HBoP
- **模型权重**：使用预训练的BLIP和SAM，无需额外训练
- **关键超参**：k=5（全局层级最大掩码数）、K=5（区域层级聚类数）、IoU阈值=0.1、每图像生成5条描述（2全局+3区域）
- **硬件要求**：未明确提及，需GPU支持SAM和BLIP推理
