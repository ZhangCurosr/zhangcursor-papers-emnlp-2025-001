---
title: "Towards-Controllable-Speech-Synthesis-in-the-Era-of-Large-La"
source: https://aclanthology.org/2025.emnlp-main.40.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:42:46"
field: "语音合成与可控生成"
keywords: ["可控语音合成", "Text-to-Speech", "大语言模型", "零样本TTS", "指令驱动生成", "离散Token", "扩散模型", "流匹配"]
innovations: ["首次系统综述LLM时代可控TTS，建立架构-策略-特征三维度分类体系", "提出基于多模态大模型Gemini的自动化评估框架，评测指令遵循/自然度/表现力", "系统对比离散与连续特征在可控TTS中的优劣，指导方法选型"]
benchmarks: ["Gemini-based人工对齐评估", "MCD", "FDSD", "WER", "Cosine Similarity", "PESQ", "MOS/CMOS"]
---

# 论文速读：Towards-Controllable-Speech-Synthesis-in-the-Era-of-Large-La

## 一句话总结
本文首次系统综述了大语言模型时代下的可控语音合成（Controllable TTS）方法，从模型架构（NAR/AR）、控制策略（标签/参考/描述/指令）和特征表示（连续/离散）三个维度建立分类体系，并提出基于多模态大模型（Gemini）的新评估框架，用于评测指令遵循、自然度与表现力。

## 研究问题与动机
- 现有TTS综述（如Ning et al., 2019; Tan et al., 2021b）聚焦声学质量与模型结构，**忽视可控性**这一核心需求。
- 工业界（影视、游戏、机器人、虚拟助手）对细粒度可控语音合成的需求快速增长，涵盖情感、音色、风格、环境等多维度属性。
- LLM与自然语言提示的引入为TTS带来了**基于文本描述的直觉式控制**新范式，但缺乏系统性梳理。
- 现有评测指标（MCD、WER、PESQ、MOS）难以捕捉**指令遵循能力**与**细粒度表现力**，需新的评估方法。

## 核心贡献（创新点）
- **首次构建可控TTS的完整分类体系**：从模型架构、控制策略、特征表示三维度系统梳理，填补综述空白。
- **提出基于多模态大模型（Gemini）的自动化评估框架**：评测指令遵循、自然度、表现力三个传统指标难以衡量的维度，并在10个主流系统上验证有效性。
- **系统对比离散Token与连续特征的优劣**：阐明离散token更适合LLM训练且数据效率高，连续特征保留更多声学细节但计算成本高，为方法选型提供依据。
- **明确未来研究方向**：指令驱动细粒度合成与编辑、表达性多模态合成、零样本长语音情感一致性、大规模数据集构建四大方向。

## 方法详解
- **模型架构**：分为非自回归（NAR）和自回归（AR）两大类。NAR包括Transformer（FastSpeech系列）、VAE、扩散模型（NaturalSpeech 2/3）、流匹配（F5-TTS、E2 TTS）；AR包括RNN-based（Tacotron系列）和LLM-based（VALL-E、CosyVoice）。
- **控制策略四范式**：① 风格标签（StyleTagging-TTS）：用离散/连续标签控制音高、能量、语速等；② 参考语音提示（Reference Speech Prompt）：几秒参考音频实现零样本克隆；③ 自然语言描述（PromptTTS、InstructTTS）：用文本描述控制语音属性；④ 指令引导合成（VoxInstruct、CosyVoice）：LLM统一处理内容与风格指令，支持副语言现象生成。
- **特征表示**：连续特征（Mel频谱、波形）保留细节但计算成本高；离散Token（EnCodec、SoundStream等量化）更适合LLM训练，支持零样本泛化，但可能损失细微声学信息。
- **属性解耦**：通过对抗训练（辅助分类器惩罚不期望属性）和信息瓶颈（独立编码器分支）实现音色、情感、韵律、内容的解耦表示。
- **评估框架**：使用Google Gemini作为评估器，通过结构化Prompt对合成语音在指令遵循（1-5分）、自然度、表现力三个维度打分，并与N人主观评分计算Pearson相关系数。

## 实验与结果
- **评测设置**：评估10个系统（8个开源：F5-TTS、CosyVoice、CosyVoice2、Vevo、Spark-TTS、MaskGCT、PromptTTS、VoxInstruct；2个商业：ElevenLabs、Mini-Max TTS），涵盖零样本TTS和基于指令的描述合成两个任务，每个任务各生成20条语音（10英文+10中文）。
- **最强结果**：零样本设置下**Vevo**在自然度（4.43±0.55）和表现力（4.32±0.75）上最优；指令引导设置下**CosyVoice**综合最强，指令遵循4.81±0.28、自然度4.92±0.24、表现力4.78±0.29。
- **评估对比**：所提Gemini评估框架在三个维度上与人类偏好相关性均优于传统自动指标NISQA和UTMOS（自然度0.17 vs -0.10/-0.03，表现力0.14 vs -0.17/-0.03），证明任务特定评估框架的必要性。

## 相关工作脉络
- **Klatt (1987)、Zen et al. (2009)**：早期参数化/统计参数TTS综述，聚焦文本分析与声学建模，**不涉及可控性**。
- **Ning et al. (2019)、Tan et al. (2021b)**：神经网络TTS综述，关注声学模型与vocoder，**缺乏对控制策略的系统分类**。
- **Triantafyllopoulos et al. (2023)**：聚焦情感语音合成与语音转换，**范围较窄，未覆盖LLM时代新方法**。
- **VALL-E (Wang et al., 2023a)**：开创LLM-based零样本TTS范式，本文将其定位为AR方法的关键里程碑。
- **PromptTTS (Guo et al., 2023)、InstructTTS (Yang et al., 2024b)、VoxInstruct (Zhou et al., 2024)**：描述/指令驱动控制方法的代表，本文将其纳入控制策略演进脉络。
- **CosyVoice (Du et al., 2024)、NaturalSpeech 2/3**：当前SOTA开源系统，本文实验评估显示其在指令遵循与表现力方面领先。

## 局限性与未来方向
- **局限性**：未探索可控属性间的交互关系（如情感与音高的耦合）；未讨论系统计算效率；未涉及社会影响（deepfake风险、对抗攻击）；未覆盖语音增强、分离、预训练、语音到语音翻译等相关领域。
- **未来方向**：① 指令驱动的细粒度语音合成与编辑；② 表达性多模态语音合成（图文视频驱动）；③ 零样本长语音与对话合成中的情感一致性；④ 基于预训练分析模型+大模型的数据集自动化构建。

## 研究启发与可借鉴点
- **Gemini-based评估框架可直接迁移**：对于任何指令驱动的语言-音频生成任务，可借鉴其多维度（指令遵循/自然度/表现力）LLM自动评估方案，降低人工评测成本。
- **离散Token+LLM架构值得复现与改进**：VALL-E范式已被广泛验证，但如何在保持零样本能力的同时降低计算开销、提升声学保真度，是潜在创新点。
- **属性解耦方法可迁移至其他模态**：对抗训练与信息瓶颈联合解耦的思想，可用于视频生成、音乐生成等多模态可控生成任务。
- **混合架构趋势明显**：CosyVoice等将LLM语义条件与流匹配/扩散生成结合的方案，兼顾质量与可控性，可作为本团队技术选型参考。

## 关键术语表
- **Non-Autoregressive (NAR)**：并行生成整个输出序列的模型架构，推理速度快但需显式建模依赖关系。
- **Autoregressive (AR)**：逐个生成输出帧的架构，能自然建模时序依赖但推理较慢。
- **Zero-shot TTS**：无需目标说话人训练数据，仅需数秒参考语音即可克隆声音并合成新文本的能力。
- **Discrete Token (离散Token)**：通过量化器将连续音频压缩为离散码本索引序列，便于LLM建模。
- **Flow Matching**：利用可逆流将复杂分布映射到高斯分布的生成方法，支持高效非自回归合成。
- **Instruction-Guided Synthesis**：用自然语言指令统一描述内容与风格属性，驱动端到端语音生成的范式。
- **Attribute Disentanglement**：将音色、情感、韵律、内容等语音属性分离到独立潜变量空间的表示学习方法。

## 可复现要素
- **数据集**：评估中使用MSP-Podcast（英文参考）、Emo-Emilia（中文参考）；公开数据集见论文附录Table 3（如AISHELL-3、Common Voice、GigaSpeech、SpeechCraft等）。
- **代码**：开源方法代码见附录Table 5（HiFi-Codec、EnCodec、SoundStream、WavTokenizer等）；论文附综合论文列表GitHub：https://github.com/imxtx/awesome-controllabe-speech-synthesis。
- **关键超参**：论文未明确报告所有模型的超参；Gemini评估使用Gemini 2.5 Flash模型，Prompt见附录A.5.1。
