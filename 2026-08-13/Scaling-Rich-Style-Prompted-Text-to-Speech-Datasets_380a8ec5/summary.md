---
title: "Scaling-Rich-Style-Prompted-Text-to-Speech-Datasets"
source: https://aclanthology.org/2025.emnlp-main.180.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:45:07"
field: "语音合成与风格控制"
keywords: ["text-to-speech", "style prompting", "dataset scaling", "speech representation", "multimodal LLM", "style consistency"]
innovations: ["首次大规模自动标注丰富内在风格标签（VoxSim感知相似传播）", "三阶段情境标签自动标注流水线（Expressivity Filtering + Semantic Matching + Acoustic Matching）", "首个同时覆盖丰富内在+情境标签的大规模开源TTS数据集（59标签/2709小时）"]
benchmarks: ["Consistency MOS", "Naturalness MOS", "Intelligibility MOS", "Tag Recall"]
---

# 论文速读：Scaling-Rich-Style-Prompted-Text-to-Speech-Datasets

## 一句话总结
本文提出 **ParaSpeechCaps**，一个支持 59 种丰富风格标签的大规模语音风格标注数据集（含 282 小时人工标注和 2427 小时自动标注），通过两种新的自动标注方法（感知说话人相似性传播 + 三阶段情境标签匹配）首次大规模扩展抽象/丰富风格标签，并在 Parler-TTS 上微调实现风格一致性与语音质量的显著提升。

## 研究问题与动机
1. **现有 TTS 数据集风格覆盖不足**：已有多风格 TTS 数据集（如 Parler-TTS、LibriTTS-P/R）仅支持基础标签（gender、pitch、speed），缺乏对 guttural、nasal、pained 等丰富/抽象风格标签的标注。
2. **人工标注成本高、规模有限**：表达丰富风格的标注依赖人工，现有包含丰富标签的数据集（如 Expresso、EARS）规模小（47–60 小时）、说话人数量少。
3. **自动标注面临两大挑战**：海量语音数据中表达性语音占比低；缺乏覆盖所有丰富标签的自动分类器（现有 emotion2vec 仅支持 8 种情绪）。
4. **缺乏同时覆盖内在/情境双维度的大规模开源数据集**：现有数据要么只有内在标签（LibriTTS-P）、要么只有情境标签（Expresso/EARS），无法支撑同时控制说话人身份与 utterance 级风格的研究。

## 核心贡献（创新点）
1. **ParaSpeechCaps 数据集**：首个同时覆盖丰富内在标签（28 个）与情境标签（23 个）的大规模开源 TTS 数据集，总计 2709 小时（282 人工 + 2427 自动），59 个标签、45k 说话人。
2. **基于感知说话人相似性的内在标签自动扩展**：使用 VoxSim 感知说话人嵌入，找到与人工标注说话人感知相似度≥0.8 的 Emilia 说话人，将其内在标签传播，实现 9× 数据扩展（最稀疏标签从约 100 小时扩至 2427 小时）。
3. **三阶段情境标签自动标注流水线**：提出 Expressivity Filtering（DVA 分类器过滤高表现性语音）→ Semantic Matching（文本嵌入匹配目标情绪关键词）→ Acoustic Matching（Gemini 音频 LLM 评估语调一致性）的级联方案，解决现有分类器无法覆盖丰富标签的问题，实现 3× 扩展。
4. **TTS 模型验证**：在 Parler-TTS 上微调，Scaled 模型相比最佳基线（结合现有丰富标签数据集）实现 Consistency MOS +7.9%、Naturalness MOS +15.5%，并通过人类评估证明自动标注数据质量与人工标注相当。
5. **全面的消融分析与设计文档**：公开详细的标签定义（Appendix A）、标注 UI、提示词（Appendix E）、阈值参数（Appendix C），为后续研究提供可复现基础。

## 方法详解

### 数据集构成与标签体系
- **标签分类**：按两个维度划分——**内在（Intrinsic，说话人级别）** vs **情境（Situational，utterance 级别）**；**丰富（Rich，主观/需人工判断）** vs **基础（Basic，信号处理可提取）**。
- **59 个标签**：28 丰富内在 + 23 丰富情境 + 5 基础内在（gender、pitch levels）+ 3 基础情境（speed levels），覆盖 11 个风格因子（pitch、texture、clarity、volume、rhythm、accent、emotion、expressiveness 等）。
- **PSC-Base（282 小时）**：汇聚 Expresso（47h）、EARS（60h）的情境标签，以及手动标注的 594 位 VoxCeleb 名人说话人内在标签。
- **PSC-Scaled（2427 小时）**：从 Emilia 英文部分（45k 小时，过滤 <5min 说话人）自动标注。

### 内在标签自动扩展（Perceptual Speaker Similarity Propagation）
1. 对 PSC-Base 中的每个 VoxCeleb 说话人，计算其 10 个随机 utterance 的 **VoxSim 感知说话人嵌入**的中值向量。
2. 对每个 Emilia 说话人做同样处理。
3. 计算余弦相似度，筛选 ≥ 0.8（对应 VoxSim 5/6 分）的配对，将 VoxCeleb 说话人的内在标签（除 clarity 标签外）传播给对应 Emilia 说话人。
4. 本质利用的是"感知相似 → 共享内在语音特征"的假设。

### 情境标签自动扩展（三阶段流水线）
1. **Expressivity Filtering**：使用现成 **DVA（Dominance-Valence-Arousal）分类器**（Wagner et al., 2023），筛选至少有一个维度 < 0.35 或 > 0.75 的 utterance；进一步按情绪特定方向（如 angry：高 dominance/arousal，低 valence）二次过滤。
2. **Semantic Matching**：用 **SFR-Embedding-Mistral** 嵌入表达性过滤后的 transcript，并与查询语句"Given an emotion, retrieve relevant transcript lines whose overall style/emotions matches the provided emotion"做余弦相似度排序；同时过滤含情绪关键词但语义不匹配的假阳性（如 transcript 含 "angry" 但语调平静）。
3. **Acoustic Matching**：取每情绪 top 100k 示例，用 **Gemini 1.5 Flash**（强音频 LLM）以 5 分 Likert 量表评估"语调是否表达目标情绪"（Focus exclusively on tone，忽略内容），仅保留评分 5 的样本。

### 基础标签提取
- 性别：GPT-4 推断（VoxCeleb）/ 显式元数据（Expresso/EARS）/ VoxSim 传播（Emilia）/ 分类器多数投票（Emilia 情境）。
- 音高/语速/噪声等级：PENN、g2p、Brouhaha 工具提取，使用 Parler-TTS 的噪声 bin 阈值。
- 所有标签经 **Mistral-7B-Instruct-v0.2** 转化为自然语言风格提示（style prompt）。

### 训练设置
- **基座模型**：Parler-TTS-Mini-v1（880M 参数，DAC audio tokens + Flan-T5-Large text encoder）。
- **Base 模型**：在 PSC-Base（VoxCeleb 90%/Expresso 80%/EARS 80%）+ LibriTTS-R（150h 基础标签正则）上训练 140k 步（peak LR 8×10⁻⁵）。
- **Scaled 模型**：在 Base 之上追加 PSC-Scaled，两阶段训练（840k 步，LR 8×10⁻⁵ → 4×10⁻⁵）。
- 为缓解不平衡：VoxCeleb ×2、Expresso/EARS ×6、情境 Scaled ×2。
- 推理：temperature=1.0，repetition_penalty=1.0，max 2580 tokens，CFG scale=1.5（虽未训练 CFG dropout 但仍有效）。

## 实验与结果

### 数据集与基线
- **主要基线**：
  - **Parler-TTS**：原始模型（仅基础标签）。
  - **+LTTSR**：仅在 LibriTTS-R 基础标签上微调。
  - **+LTTSP,Exp,EARS**：结合 LibriTTS-P（46 个丰富内在标签，0.6k 小时）+ Expresso + EARS。
- **评测数据集**：从 PSC-Base holdout 拆分 246 个 tag-balanced 示例（单次评估一个丰富标签）+ 240 组合成示例（12 内在 × 10 情境 × 2 性别）。

### 主要结果（Table 3）

| 模型 | CMOS ↑ | Intr TR ↑ | Sit TR ↑ | NMOS ↑ | IMOS ↑ | WER ↓ |
|------|--------|-----------|----------|--------|--------|-------|
| Ground Truth | 4.42 | 88.7% | 88.6% | 4.36 | 4.28 | 7.93 |
| Parler-TTS | 3.05 | 33.0% | 21.2% | 2.85 | 4.31 | 4.62 |
| +LTTSR | 3.07 | 33.7% | 22.4% | 2.95 | 4.44 | 4.47 |
| +LTTSP,Exp,EARS | 3.55 | 40.7% | 69.7% | 3.10 | 4.19 | 7.14 |
| **Base (Ours)** | **3.75** | **63.6%** | **68.1%** | **3.27** | 4.05 | 9.14 |
| **Scaled (Ours)** | **3.83** | **69.5%** | **75.4%** | **3.58** | 4.07 | 8.63 |

- **最显著提升**：Scaled 模型相比最佳基线 +LTTSP,Exp,EARS：Consistency MOS +7.9%（3.55→3.83）、Naturalness MOS +15.5%（3.10→3.58）；Intrinsic Tag Recall 从 40.7% 提升至 69.5%（+28.8 个百分点）。
- **组合性（Compositionality）**：Scaled 模型同时生成内在+情境标签的比例最高（Figure 5）。
- **推理时 CFG 有效性**：即使未使用 dropout-based CFG 训练，推理时加入 CFG 仍可提升所有模型的 Consistency MOS（Table 4）。

### 自动标注质量验证（Table 2）
- PSC-Scaled 内在标签 Tag Recall 50.3% vs PSC-Base 48.7%（相当）；Situational 71.3% vs 68.1%。
- 三阶段消融均导致 Recall 下降，证明各步骤必要性：w/o Expressivity（61.0%）、w/o Semantic（66.1%）、w/o Acoustic（63.3%）。
- Std. Embedder（WavLM + ECAPA-TDNN 替代 VoxSim）效果差（45.3%），证明感知说话人相似模型对标签传播的关键作用。

### 可理解性下降分析
- 丰富风格模型 IMOS/WER 低于基础标签基线，主要因 Faithfully 生成 slurred/stammering/non-American accent/pained 等"本身难懂"的风格，属预期行为而非模型缺陷。

## 相关工作脉络
1. **Parler-TTS / PromptTTS 系列**：Parler-TTS（Lacombe et al., 2024）仅支持基础标签自动标注；PromptTTS（Guo et al., 2022）仅 4 种合成情绪；本文扩展至 59 种真实风格标签，且首次支持丰富内在标签的自动标注。
2. **LibriTTS-P（Kawamura et al., 2024）**：提供 46 个丰富内在标签但仅 0.6k 小时；本文通过 VoxSim 传播将其扩展至 2427 小时自动数据。
3. **Expresso / EARS**：提供丰富情境标签但规模有限（47–60h）；本文在此基础上扩展 +15 倍数据。
4. **SpeechCraft（Jin et al., 2024）**：仅支持 8 种情绪扩展；本文方法覆盖 18 种情绪 + 其他情境标签。
5. **AudioBox（Vyas et al., 2023）**：结合信号处理扩展基础标签 + 人工丰富标签，但非开源；本文为首个同时覆盖丰富内在+情境的大规模开源数据集。
6. **DreamVoice / VCTK-RVA**：在 voice conversion / speech editing 中使用风格标签；本数据集可通用。

## 局限性与未来方向
1. **语言覆盖仅限英语**：未探索中文、日语等其他语言的风格标注与生成。
2. **数据集偏差**：隐含相关性强，如 shrill 过度关联 female、guttural 过度关联 male，可能导致模型难以生成跨性别的风格组合。
3. **缺乏自动评估指标**：风格一致性与质量高度依赖 MTurk 人工 MOS，无法快速迭代；急需开发 robust 的自动风格评估方法。
4. **清晰度标签未自动标注**：PSC-Scaled 不支持 clarity 和 expressiveness 两个风格因子的自动标注。
5. **CFG dropout 缺失**：推理时 CFG 有效但训练时未做 dropout，未来可进一步研究训练-推理对齐。

## 研究启发与可借鉴点
1. **三阶段"语义+声学"匹配框架可迁移**：Expressivity Filtering → Semantic Matching → Acoustic Matching 的级联思路可用于任何需要自动标注情感/风格语音的场景，尤其是缺乏专用分类器的标签体系。
2. **感知说话人相似性（VoxSim）用于标签传播**：用人类感知嵌入而非标准说话人验证嵌入来做标签传播，为跨说话人属性学习提供了新思路，可迁移至 voice style transfer / voice conversion 研究。
3. **大模型推理辅助人工标注质量验证**：用 Gemini 做最终声学匹配打分，证明了多模态 LLM 在语音风格评估中的价值，可与人类评估形成互补。
4. **风格提示生成的 LLM pipeline 设计**：使用 Mistral-7B 将离散标签转化为流畅 style prompt，并验证了 temperature=0.6、top-p=1.0 的最佳配置，可直接复用。
5. **CFG 在 TTS 推理中的隐式有效性**：即使未在训练中引入 CFG dropout，推理时加 CFG 仍可稳定提升风格一致性，提示后续工作不必严格遵循标准 diffusion CFG 训练范式。

## 关键术语表
- **ParaSpeechCaps**：本文提出的大规模语音风格标注数据集，含 59 种风格标签、2709 小时数据。
- **PSC-Base**：ParaSpeechCaps 中 282 小时人工标注的子集（VoxCeleb + Expresso + EARS）。
- **PSC-Scaled**：ParaSpeechCaps 中 2427 小时自动标注的子集（基于 Emilia 数据集）。
- **Intrinsic Tags**：与说话人身份相关的风格标签（如 pitch、texture、accent），跨 utterance 稳定存在。
- **Situational Tags**：与具体 utterance 相关的风格标签（如 emotion、expressiveness），随语境变化。
- **Rich Tags**：主观抽象的风格标签（如 guttural、pained、sarcastic），需人工或 LLM 判断。
- **Basic Tags**：可通过信号处理工具自动提取的标签（如 gender、pitch level、speed level）。
- **VoxSim**：感知说话人相似性嵌入模型，衡量人类主观感知的说话人相似程度。
- **DVA 分类器**：基于 Dominance-Valence-Arousal 理论的三维情绪分类器。
- **Consistency MOS**：评价生成语音与风格提示一致性的 Mean Opinion Score（5 点 Likert）。
- **Classifier-Free Guidance (CFG)**：扩散模型推理时的无分类器引导技术，本文在未训练 dropout 的情况下仍发现其推理时有效。

## 可复现要素
- **数据集**：ParaSpeechCaps 已开源（https://github.com/ajd12342/paraspeechcaps），含 PSC-Base 和 PSC-Scaled。
- **代码**：开源（同上仓库）。
- **模型权重**：论文声明已开源。
- **关键超参**：peak LR 8×10⁻⁵/4×10⁻⁵、batch size 32、weight decay 0.01、温度 1.0、CFG scale 1.5、VoxSim 相似度阈值 0.8、DVA 阈值 0.35/0.75、Gemini 评分阈值 5。
- **基座模型**：Parler-TTS-Mini-v1（HuggingFace 可下载）。
- **标注平台**：Amazon Mechanical Turk，付费 $9/hr，qualifications: Masters、approval rate ≥99%、≥5000 HITs。
