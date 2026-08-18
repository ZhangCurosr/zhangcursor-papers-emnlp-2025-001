---
title: "SEA-Supervised-Embedding-Alignment-for-Token-Level-Visual-Te"
source: https://aclanthology.org/2025.emnlp-main.55.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:35:02"
field: "多模态大模型对齐"
keywords: ["多模态大语言模型", "跨模态对齐", "适配器训练", "词元级监督", "对比学习", "SEA"]
innovations: ["提出词元级监督对齐方法SEA，利用CLIP语义标签通过对比学习将视觉token精确映射到LLM嵌入空间", "设计局部采样与相似度加权策略保障对比学习效率，零额外数据与推理开销", "揭示适配器语义扭曲与模态表示差距问题，小模型增益最显著（Gemma-2B平均提升7.61%）"]
benchmarks: ["VQAv2", "TextVQA", "GQA", "ScienceQA-IMG", "MMBench", "POPE", "VizWiz", "MM-Vet"]
---

# 论文速读：SEA-Supervised-Embedding-Alignment-for-Token-Level-Visual-Te

## 一句话总结
本文提出词元级监督对齐方法 SEA（Supervised Embedding Alignment），利用预训练视觉-语言模型（如 CLIP）为每个视觉 patch 提取语义标签，通过对比学习将适配器输出的视觉词元精确映射到 LLM 嵌入空间，从根本上缓解图像级对齐导致的语义扭曲与模态表示差距问题，在多款模型和编码器上实现稳定提升，尤其对小模型增益显著。

## 研究问题与动机
- **图像级监督导致语义扭曲**：传统 MLLM 适配器通过图像/区域级监督对齐，但视觉词元常映射到语义无关词（如"blueberries"被误映射为"bluejeans"），迫使 LLM 补偿表征差异，产生错误视觉理解。
- **模态表示差距显著**：适配器输出的视觉词元与 LLM 原生输入嵌入空间距离较大，迫使 LLM 占用额外容量解析错位输入而非利用预训练知识。
- **小模型受影响更严重**：参数量较小的 LLM 在视觉感知与语言能力间权衡更剧烈，现有对齐方式加剧了其能力瓶颈（Gemma-2B 是典型受益者）。
- **现有对齐方法未能根本解决**：最优传输（optimal transport）、回归类方法等仅停留在粗粒度监督，未触及适配器的词元级语义对齐这一核心环节。

## 核心贡献（创新点）
- **提出词元级监督对齐范式**：不同于传统图像级监督，SEA 首次通过词元级对比学习在预训练阶段精确对齐视觉词元与 LLM 嵌入空间，本质区别在于用细粒度语义标签替代了粗粒度图文匹配信号。
- **利用预训练 VLM 提取连续语义标签**：基于 CLIP 等已对齐模型的视觉-文本编码器，为每个视觉 patch 提取 top-n 相似词汇作为语义标签并加权采样，而非依赖人工标注或强假设，避免了语义偏移。
- **局部采样策略保障对比学习效率**：采用 k×k 窗口内仅采样一个 patch 的策略，避免高相似样本干扰对比学习，同时每批次同类标签随机保留一个，使对齐信号更清晰。
- **零额外数据与推理开销**：SEA 不引入新训练数据或推理成本，仅需在预训练阶段增加对比损失项，且与下游指令微调完全兼容（可联合冻结或解冻视觉编码器）。

## 方法详解
- **语义标签提取**：对预训练的视觉编码器 f 和文本编码器 h（如 CLIP），给定图像 X_image 和词表 W（约 400 万词），通过公式(6)-(8)计算每个视觉 patch v_i 与所有文本词嵌入 t_j 的余弦相似度，取 top-n（n=10）个相似度为正的词汇作为语义标签 w_i 及对应分数 s_i。
- **相似度加权采样**：对每个 patch 的 n 个标签按得分归一化得到采样概率 S_norm^i = S_i / sum(S_i)，基于此概率采样一个标签，保留视觉 token 的连续语义表示。
- **局部采样策略**：图像内按 k×k 窗口（实验中 k=2）划分，每窗口只选一个 patch 参与对比学习；同一 batch 内相同标签的 patch 随机保留一个，防止冗余干扰。
- **对齐损失设计**：对每个选中的 (视觉token x_vi, 标签词 w_i)，计算文本特征 t_i = (1/M)ΣΨ(w_i^k)（即 LLM 嵌入层对标签词的均值），使用 InfoNCE 对比损失 L_a（公式11），温度 τ 可学习，采用零温度设置。
- **总损失**：预训练总损失为生成损失 L_g（自回归语言建模）与对齐损失 L_a 的加权和，即 L = L_g + λL_a，λ 平衡两项目标。
- **训练流程**：两阶段训练，第一阶段仅优化适配器 g_θ，视觉编码器和 LLM 冻结；第二阶段联合优化 LLM 和适配器（可选解冻视觉编码器），与 LLaVA 训练流程一致。

## 实验与结果
- **数据集与基准**：使用 Cambrian-1/7M 数据进行预训练与指令微调（2.5M 对齐数据 + 7M 指令数据）；消融实验沿用 LLaVA-1.5 标准数据（CC-595K + 656K 指令混合集）。评测涵盖 VQAv2、TextVQA、GQA、ScienceQA-IMG、MMBench、POPE、VizWiz、MM-Vet 等 8 个主流基准，以及 CapsBench、Stanford Dogs、COCO Captions、OCRBench、MMMU 等细粒度感知任务。
- **模型规模**：测试 LLM 从 2B（Gemma-2B）到 13B（Vicuna-13B/Llama3-8B），视觉编码器包括 CLIP-ViT-L@336px、SigLIP-ViT-SO@384px、DINOv2-L@224px。
- **最强结果**：SEA-PRIME 在 LLaMA3-8B 上 VQAv2 达 83.1、SQA 达 79.0、MMB 达 76.0、MM-Vet 达 46.0，全面超越同期开源方法；在 Phi3-3.8B 上 MM-Vet 达 46.8，显著优于同等规模基线。
- **提升幅度**：SEA 在所有模型规模上一致提升性能；小模型增益最为突出，Gemma-2B 平均提升 7.61%；在细粒度感知任务上，SEM 相较 LLaVA 在 MMMU 提升 11.4%、OCRBench 提升 5.3%、COCO Captions (CIDEr) 提升 4.6%。
- **TACS 指标**：Token Alignment Consistency Score 显示 SEA 在预训练过程中 TACS 持续提升，与 POPE 得分正相关，验证对齐质量改善。
- **消融结论**：SEA 对不同类型 LLM 和视觉编码器均有效；解冻视觉编码器微调在 SEA 基础上进一步增益（VQAv2 +1.5）；CLIP 语义标签可迁移至 DINOv2 训练（VQAv2 提升 2.6 分）。

## 相关工作脉络
- **CLIP/ALIGN/SPARC 等视觉-语言对比预训练**：这些工作在大尺度图文对上训练了对齐的视觉-文本编码器，SEA 直接复用此类模型的语义标签提取能力，但目标不同——它们服务于零样本分类/检索，而 SEA 将其引入 MLLM 的适配器训练。
- **LLaVA 及后续 shallow-fusion 方法**：LLaVA 等通过简单适配器连接冻结的视觉编码器与 LLM，依赖图像级自回归损失训练；SEA 在此基础上引入词元级对比监督，弥补其语义扭曲缺陷，定位上是适配器对齐策略的升级而非架构替换。
- **AlignGPT / 相似度 token 分配方法**：AlignGPT 等通过相似度分配改善跨模态 token 匹配，但未从语义标签层面约束适配器的映射空间；SEA 通过 CLIP 提供的连续语义标签实现更精确的嵌入对齐。
- **Best of N / segmentation-增强视觉 token 方法**：如 Rethinking MLLMs 等通过 OCR 或分割增强视觉 token 表达，侧重输入端信息丰富化；SEA 侧重适配器输出端的语义校准，二者正交可结合。
- **Mini-Gemini / Cambrian 等高效 MLLM 设计**：这些工作关注模型压缩或数据效率；SEA 作为对齐策略可独立叠加于此类架构之上，提升小模型对齐质量。
- **SigLIP 等改进型 VLM 预训练**：SigLIP 引入 pairwise Sigmoid loss 替代 softmax contrastive loss；SEA 兼容不同 VLM 预训练范式，其标签提取模块与具体对比学习目标无关。

## 局限性与未来方向
- **静态标签选择**：当前语义标签为预定义的固定词表提取，未考虑视觉内容的动态复杂度，未来可探索自适应标签选择机制。
- **单模态扩展**：目前仅验证图像-文本对齐，扩展到视频、音频等多模态需额外设计时序对齐策略，尚待研究。
- **安全对齐关系未明**：representation alignment 与 safety alignment 之间的关联未被系统分析，是潜在研究方向。
- **词表规模限制**：约 400 万词的预定义词表虽覆盖广泛，但可能遗漏领域专有词汇，影响特定场景的语义标签精度。

## 研究启发与可借鉴点
- **利用已对齐 VLM 作为语义监督源**：CLIP 等模型的成熟对齐能力可直接移植为 MLLM 适配器的监督信号，无需额外标注数据，值得在其他多模态融合任务中复用。
- **词元级对比损失与生成损失的联合优化**：双损失设计（L_g + λL_a）在保证语言能力的同时增强跨模态对齐，可作为 MLLM 预训练的通用模板。
- **局部采样避免高相似样本干扰**：k×k 窗口采样策略有效缓解了 batch 内冗余样本问题，可推广至其他基于对比学习的多模态训练场景。
- **TACS 作为对齐质量的量化指标**：Token Alignment Consistency Score 提供了可追踪的对齐进度监控手段，便于实验设计与超参调优。
- **小模型对齐敏感度挖掘**：本研究揭示了参数量与对齐质量的强相关性，提示团队在开发轻量级多模态模型时应优先投入对齐策略优化。

## 关键术语表
**SEA（Supervised Embedding Alignment）**：一种词元级监督对齐方法，利用预训练 VLM 为视觉 patch 提取语义标签并通过对比学习将其映射到 LLM 嵌入空间。

**适配器（Adapter）**：连接冻结视觉编码器与 LLM 的轻量化模块，负责将视觉特征变换为 LLM 可接受的嵌入表示。

**Token Alignment Consistency Score（TACS）**：衡量视觉 token 与文本 token 在嵌入空间中余弦相似度的指标，用于量化对齐质量。

**对比学习损失（InfoNCE）**：通过最大化正样本对相似度、最小化负样本对相似度来学习对齐表示的损失函数，本文采用零温度设置。

**语义标签（Semantic Labels）**：通过 CLIP 文本编码器提取的与视觉 patch 余弦相似度最高的 top-n 词汇，作为词元级监督信号。

**局部采样策略（Localized Sampling Strategy）**：在图像内按 k×k 窗口限制每窗口仅一个 patch 参与对比学习，避免高相似样本干扰的训练技巧。

**Cambrian-1/Cambrian-7M**：论文使用的多模态训练数据集，分别包含 2.5M 对齐数据和 7M 指令微调数据。

**Modality Representation Gap**：适配器输出的视觉 token 与 LLM 原生文本 token 在嵌入空间中的分布距离，是导致 LLM 容量浪费的核心问题。

## 可复现要素
- **数据集**：Cambrian-1（2.5M caption pairs）、Cambrian-7M（指令微调）；消融实验使用 LLaVA-1.5 标准数据（CC-595K + 656K 混合指令集）；公开可获取。
- **代码/权重**：代码已开源（https://github.com/YuanyangYin/SEA），论文未提供预训练权重链接，需自行从头训练。
- **关键超参**：top-n 语义标签数 n=10；温度 τ=0；局部采样窗口 k=2；预训练 1 epoch，AdamW 优化器，余弦学习率调度；Stage 1 仅优化适配器，Stage 2 联合优化 LLM 和适配器（SEA-PRIME 还解冻视觉编码器，学习率 2e-6）。
- **硬件**：8×H800 GPU，Stage 1 额外耗时 10-20 分钟，全程约 6-10 小时；SEA-PRIME 全程训练 <4 天。
