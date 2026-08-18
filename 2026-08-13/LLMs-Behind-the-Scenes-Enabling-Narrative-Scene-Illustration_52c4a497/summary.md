---
title: "LLMs-Behind-the-Scenes-Enabling-Narrative-Scene-Illustration"
source: https://aclanthology.org/2025.emnlp-main.124.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:44:41"
field: "跨模态叙事生成与评估"
keywords: ["narrative scene illustration", "text-to-image", "LLM-as-judge", "cross-modal evaluation", "story visualization", "prompt engineering"]
innovations: ["提出可插拔的LLM-to-T2I场景插图生成流水线并实证验证LLM作为语义桥梁的有效性", "构建首个面向叙事场景插图的成对偏好质量标注数据集SCENEILLUSTRATIONS", "提出基于原子化LLM生成标准的criteria-based evaluation框架，准确率70%显著优于无标准基线59%"]
benchmarks: ["ROCStories", "Artificial Analysis Image Arena Leaderboard", "LLM Creative Story Writing Benchmark"]
---

# 论文速读：LLMs-Behind-the-Scenes: Enabling Narrative Scene Illustration

## 一句话总结
本文提出了一条以 LLM 为"翻译层"的叙事场景插图生成流水线，将故事文本片段自动转化为适合文本到图像模型的提示词，并基于此构建了首个面向跨模态叙事转换的质量标注数据集 SCENEILLUSTRATIONS（2990 项）；实验验证了 LLM 能够有效口语化故事文本中隐含的场景视觉知识，且该能力可同时用于插图生成与质量评估。

## 研究问题与动机
- 已有故事可视化研究多聚焦"图像-图像对齐"（角色/场景跨帧一致性），而"文本-图像对齐"这一更基础的问题在现有工作中缺乏系统实证。
- 原始故事文本通常缺乏图像生成模型所需的关键视觉细节（如人物外貌、空间布局、光照氛围等），直接作为 prompt 效果不佳。
- 已有使用 LLM 为图像生成模型合成提示的工作（如 Gong et al., 2023; He et al., 2024）仅"假设"LLM 合成提示更优，但缺少对照实验和人类标注的实证验证。
- 跨模态叙事质量评估依赖共享嵌入空间度量（如 CLIPScore），难以捕捉故事隐含的场景语义；LLM 是否可作为零样本评估接口仍待探索。

## 核心贡献（创新点）
1. **定义并系统化"叙事场景插图"任务**：将其从现有故事可视化研究中剥离，聚焦"文本-图像对齐"的片段级场景生成问题，而非跨帧一致性。
2. **提出可插拔的 LLM-to-T2I 流水线**：LLM 同时承担片段分割、场景描述生成（scene captioner）与评估标准生成（criteria writer）三种角色，组件间完全可替换，可兼容任意 LLM 与任何文本到图像模型。
3. **构建并开源 SCENEILLUSTRATIONS 数据集**：包含 2990 对带有成对人工质量 judgments 的插图样本，是目前首个面向"从纯文本故事生成单帧场景插图"的大规模质量标注资源。
4. **实证验证 LLM 口语化隐含视觉知识的能力**：通过人类偏好实验证明，LLM 从故事文本中提炼出的 CAPTION 场景描述显著优于四种基线（NC-FRAGMENT / VC-FRAGMENT / SC-FRAGMENT 及直接 prompt），胜率最高达 78.1%。
5. **提出 Criteria-Based Evaluation 范式**：用 LLM 生成可解释的评估标准，再由 VLM 逐条打分，平均准确率（70%）显著优于直接给分的基线 Rater（59%），为文本到图像评估提供可复用的零样本评分框架。

## 方法详解
流水线由三个核心模块组成，输入为故事文本，输出为单帧场景插图：

1. **Story Fragmentation（片段分割）**：用 LLM（本工作使用 CLAUDE-3.5）根据特定 prompt（附录 Table A.13）标注每段故事中所有场景片段的起止边界，再以正则表达式解析出 fragment 列表。默认每个 fragment 约 1 句，平均每篇故事切出 4.12 个片段。

2. **Scene Captioning（场景描述生成）**：给定片段及其上下文，LLM 以 Table A.15 中的 prompt 将其改写为详细的图像生成提示。本工作对比四种 scene description 类型（见 Table 1）：
   - **NC-FRAGMENT**：原始片段，无上下文。
   - **VC-FRAGMENT**：将完整故事文本作为条件插入。
   - **SC-FRAGMENT**：用 LLM 重写片段，显式补全指代信息。
   - **CAPTION**：LLM 生成的详细描述性场景描述（本文核心创新）。

3. **Image Generation（图像生成）**：将 scene description 输入文本到图像模型。Phase 1 使用 MJ-6.1 与 FLUX-1-PRO；Phase 2 扩展至 5 个图像生成器（含 IDEOGRAM-2.0、RECRAFT-V3、SD-3.5-LARGE）。

4. **Criteria-Based Evaluation（基于标准的评估）**：
   - **Criteria Writer**（CLAUDE-3.5 / GPT-4o / LLAMA-3.1）读取故事片段，生成一组结构化评估标准（每标准"原子化"，强调"灵活性"与"单一特征"原则）。
   - **Criteria Rater**（VLM：CLAUDE-3.5 / GPT-4o / PIXTRAL）逐条判定图像是否满足该标准，响应映射为 1.0（✓）/ 0.5（?）/ 0.0（✗），求和得总分。
   - **Baseline Rater**：同一 VLM 在不看到标准的情况下直接给分（0 至标准数之间，步长 0.5），用于对照。
   - 选择策略：成对比较中，得分更高的图像被预测为人类偏好的那一个；准确率定义为与人类一致的选择占比。

## 实验与结果
- **数据集**：ROCStories 语料（50 篇 Phase 1 + 1000 篇 Phase 2），共产出 2990 项插图对，每项含两幅插图及 2 位 Prolific  annotators 的成对偏好标注。
- **Phase 1 核心结果（Table 3）**：CAPTION 相对于三种基线的胜率分别为 NC-FRAGMENT 78.1%、VC-FRAGMENT 74.7%、SC-FRAGMENT 72.5%，统计显著。跨图像生成器的消融显示该优势不因生成器不同而消失（附录 Table A.17）。
- **Phase 2 场景描述器对比（Table 4）**：CLAUDE-3.5 胜率最高（49.6% vs LLAMA-3.1，差异显著），GPT-4o 居中（46.2% vs CLAUDE-3.5，差异不显著），LLAMA-3.1 最低（39.7% vs CLAUDE-3.5）。
- **Phase 2 图像生成器对比（附录 Table 10）**：IDEOGRAM-2.0 整体胜率最高（61.7% vs MJ-6.1、58.5% vs SD-3.5-LARGE 均显著），RECRAFT-V3 显著优于 MJ-6.1（61.8%）。
- **一致性指标**：Phase 1 整体 Cohen's κ_u = 0.436（中度一致，62.3% 一致率）；Phase 2 整体 κ_u = 0.231（52.6% 一致率），说明 Phase 2 因变量增多（多模型）导致差异更难判别。
- **质量预测实验（Table 6）**：Criteria Rater 平均准确率 70.0%，显著高于 Baseline Rater 的 58.9%；其中 CLAUDE-3.5 作为 Criteria Writer 搭配任意 Rater 平均达到 71% 准确率，为最优配置。随机猜测（永远选第二张）仅为 49.4%。

## 相关工作脉络
- **StoryGAN (Li et al., 2019)**：早期使用条件 GAN 进行故事可视化，聚焦多帧间角色/场景一致性（image-image alignment），本文聚焦更基础的文本-图像对齐（text-image alignment）。
- **DreamStory (He et al., 2024) / StoryVisualizer (Gong et al., 2023)**：使用 GPT-4 将故事转为场景级 prompt 驱动 T2I 模型，但未在成对偏好数据上实证 LLM 是否真正优于原始文本，本文首次提供此类对照证据。
- **TIFA (Lu et al., 2023) / LLMScore (Hu et al., 2023)**：使用 LLM/VLM 评估图像-文本对齐，但侧重于单图 caption 评测；本文将其推广至"叙事场景"的细粒度标准生成与逐条判定范式。
- **CLIPScore (Hessel et al., 2021)**：基于 CLIP 嵌入空间的相似度度量，无法捕捉故事隐含语义与主观审美偏好；本文使用人工偏好标注作为 ground truth。
- **LLM-as-Prompt-Engineer (Zhou et al., 2023)**：提出 LLM 可作为 prompt 优化器的概念；本文将其专门应用于叙事场景插图这一垂直领域并给出系统化评测。
- **ROCStories 及相关 NLP 工作 (Mostafazadeh et al., 2016; Kong et al., 2021)**：传统 NLP 故事理解/生成语料，本文首次将其用于系统性的跨模态叙事转换研究。

## 局限性与未来方向
- **专有模型依赖**：核心实验使用的 LLM 与图像生成器多为闭源模型（仅 LLAMA-3.1 与 SD-3.5-LARGE 开源），可复现性与可访问性受限。
- **Prompt 设计未自动化优化**：采用经验性 prompt engineering 而非基于开发集的量化优化（如 DSPy 范式），可能存在进一步改进空间。
- **语料限制**：ROCStories 是人工构造的简单叙事语料，故事复杂度有限；流水线在更复杂文学性叙事上的泛化能力尚未经充分验证。
- **仅涉及单帧场景**：本文聚焦单个场景插图（text-image alignment），未处理跨帧角色/场景一致性（image-image alignment）这一更大的开放问题。
- **未来方向**：结合近期多帧一致性方法（如 Liu et al., 2025; Maharana et al., 2022），将流水线扩展为完整多帧故事序列生成；进一步探索图像生成器间差异的系统分析。

## 研究启发与可借鉴点
- **可复用范式：LLM 作为"语义桥梁"**：将 LLM 同时用于内容生成（caption）与评估标准生成（criteria），实现"生成-评估闭环"，可迁移至其他跨模态内容创作任务（如配乐可视化、短视频分镜脚本生成）。
- **标准原子化设计原则**：每条评估标准应满足"单一特征"和"允许视觉变体"两个属性，这一设计可推广至任何基于标准的多模态评测场景。
- **成对偏好 + 不确定性标注**：使用 "I can't decide" 选项并结合 uncertainty-weighted Cohen's κ 进行一致性分析，能更精细地刻画标注噪声，适用于任何成对比较类评测。
- **可组合的实验设计**：分别控制 scene captioner 与 image generator 两个变量，有助于精确归因质量差异来源，为后续消融实验提供参考模板。
- **与本团队方向结合机会**：若团队研究方向涉及"文本到多模态内容的自动生成与评估"，本文的 criteria-based evaluation 框架可直接复用，替代当前可能依赖 embedding-based 指标的评估方案。

## 关键术语表
- **Narrative Scene Illustration**：将故事文本中描述的场景自动转化为单帧图像的任务，聚焦文本-图像对齐而非跨帧一致性。
- **Scene Captioner**：使用 LLM 将故事片段及其上下文转化为详细图像生成提示词的模块角色。
- **Criteria Writer**：使用 LLM 为故事片段生成结构化评估标准（检查清单）的模块角色，无需视觉能力。
- **Criteria Rater**：使用 VLM 根据给定标准对图像逐条评分的模块角色，负责视觉理解与匹配判定。
- **CAPTION / NC-FRAGMENT / VC-FRAGMENT / SC-FRAGMENT**：四种场景描述形式，分别代表 LLM 生成描述、原始片段、完整上下文+片段指令、上下文显式补全片段。
- **SCENEILLUSTRATIONS**：本文发布的质量标注数据集，含 2990 对成对插图及人工偏好标注，供跨模态叙事转换研究使用。
- **Uncertainty-weighted Cohen's κ (κ_u)**：考虑"无法判断"响应的加权一致性指标，对不确定分歧给予半权惩罚。
- **Meta-prompting**：利用 LLM 生成或优化其他模型（如 T2I 模型）输入提示词的技术范式。

## 可复现要素
- **数据集**：SCENEILLUSTRATIONS 已作为附属资源发布（论文中注明链接），基于 ROCStories 语料；ROCStories 本身为公开数据集。
- **代码/权重**：论文未明确声明开源代码仓库；使用的模型中包含部分开源模型（LLAMA-3.1、SD-3.5-LARGE），但核心实验依赖闭源 API 模型（CLAUDE-3.5、GPT-4o、MJ-6.1、FLUX-1[pro]、IDEOGRAM-2.0、RECRAFT-V3、PIXTRAL）。
- **关键超参**：Criteria Writer / Captioner 推理 temperature=0（确定性输出）；VLM Rater 图像统一 resize 至高度 240px 保持宽高比；Annotation 每人 judge 33-109 对（median ≈ 47-50），支付 $6/约 30 分钟；两组标注者/项。
