---
title: "TokenSkip-Controllable-Chain-of-Thought-Compression-in-LLMs"
source: https://aclanthology.org/2025.emnlp-main.165.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:42:13"
field: "推理效率优化"
keywords: ["TokenSkip", "Chain-of-Thought Compression", "Large Language Models", "Efficient Inference", "Token Importance"]
innovations: ["首次探索基于token跳过增强CoT效率", "提出TokenSkip实现可控CoT压缩", "实验验证在GSM8K上40% token减少且性能下降<0.4%"]
benchmarks: ["GSM8K", "MATH-500", "CommonsenseQA"]
---

# 论文速读：TokenSkip-Controllable-Chain-of-Thought-Compression-in-LLMs

## 一句话总结
本文提出TokenSkip方法，通过分析CoT token的语义重要性并选择性跳过低重要性token，实现LLM推理过程中Chain-of-Thought的可控压缩，在保持推理性能的同时显著降低计算开销。

## 研究问题与动机
- **CoT长度增加导致推理延迟线性增长**：随着OpenAI o1、DeepSeek-R1等模型展现长CoT优势，推理token数可达数千至数万，但因自回归解码特性，更长CoT导致推理延迟和KV缓存内存占用线性增长。
- **现有压缩方法与测试时缩放冲突**：已有研究探索选择性跳过推理步骤，但近期研究表明此类压缩可能与测试时缩放（test-time scaling）目标相冲突，损害LLM推理性能。
- **CoT token贡献不均存在冗余**： empirically分析表明CoT输出中各token对推理的贡献存在显著差异，如数学公式贡献较大，而语义连接词（如"so"、"since"）贡献较小，存在压缩潜力。
- **平衡效率与准确性的挑战**：如何在压缩CoT以提升推理效率的同时，维持LLM强推理能力，成为亟待解决的关键问题。

## 核心贡献（创新点）
1. **首次探索基于token跳过增强CoT效率**：区别于现有工作关注步骤级压缩，本文从token层面研究CoT冗余，揭示token重要性差异为压缩提供理论依据。
2. **提出TokenSkip实现可控压缩**：通过结合LLMLingua-2的重要性度量与LoRA微调，使模型学会在推理时自动跳过低重要性token，支持用户按需调节压缩比。
3. **验证高效性与人效性平衡**：实验表明在GSM8K上Qwen2.5-14B-Instruct可实现40% token压缩（313→181）且性能下降<0.4%，显著优于现有基线方法。
4. **低成本训练方案**：仅需微调0.2%参数（LoRA），在单任务数据集（GSM8K/MATH）上训练约2-2.5小时（14B模型，双3090 GPU），具备良好可复现性。

## 方法详解
- **Token重要性度量**：采用LLMLingua-2的bidirectional BERT-like语言模型，计算每个token的重要性得分$I_2(x_i)=P(x_i|\mathbf{x}_{\leq n};\boldsymbol{\theta}_{\mathcal{M}_B})$，克服因果LM单向注意力的位置依赖偏差。
- **Token剪枝过程**：给定压缩比$\gamma\in[0,1]$，计算重要性得分的$\gamma$-分位数阈值$I_\gamma=Q_\gamma(\{I(c_i)\})$，保留满足$I(c_i)\geq I_\gamma$的token形成压缩轨迹$\tilde{c}$。
- **训练流程**：对N个训练样本生成CoT后，按$\gamma\in\{0.5,0.6,0.7,0.8,0.9,1.0\}$随机采样比例压缩，构造训练样本$[\texttt{Q}][\texttt{EOS}]\gamma[\texttt{EOS}]\text{Compressed CoT } \texttt{[EOS]} \texttt{A}$，使用LoRA（rank=8, alpha=16）进行SFT训练。
- **推理机制**：在提示末尾附加目标压缩比$\gamma$，模型按相同格式自回归生成压缩CoT与答案，实现按需压缩。

## 实验与结果
- **数据集**：GSM8K（数学应用题）、MATH-500（MATH子集）；额外验证 CommonsenseQA、MMLU-STEM。
- **基线方法**：Token-efficient Prompts（BeConcise/OnlyNumbers/AbbreWords）、Length-control Prompts（LC-Prompt）、Truncation（暴力截断）。
- **主要结果**：
  - Qwen2.5-14B-Instruct在GSM8K上以$\gamma=0.6$实现40% token压缩（313→181），准确率仅下降<0.4%。
  - LLaMA-3.1-8B-Instruct在MATH-500上压缩30%（502→352），准确率下降<4%，推理速度提升1.4倍。
  - TokenSkip在$\gamma=0.5$时实际压缩比达0.53（GSM8K），较原始模型延迟降低1.8倍，而Truncation在同等压缩比下导致79%准确率下降。
  - 更大模型（14B vs 7B）在更高压缩比下表现更稳定，体现模型规模对压缩适应性的正向影响。
- **最佳结果**：Qwen2.5-14B-Instruct在GSM8K $\gamma=0.6$下实现40% token压缩且性能几乎无损，对比基线中Truncation需牺牲大量性能。

## 相关工作脉络
- **Selective Context**：使用单向LM困惑度度量token重要性，存在位置偏差；本文改用双向模型LLMLingua-2提升度量准确性。
- **LLMLingua-2**：提供token重要性计算基础，但原方法面向prompt压缩；本文首次将其迁移至CoT压缩任务并设计配套微调流程。
- **Break the Chain / CoT Skipping**：侧重于跳过完整推理步骤；本文在token粒度操作，保留步骤完整性但精简表达。
- **Token-Efficient Prompts**：依赖提示工程限制输出长度，压缩比控制不精确；本文通过微调使模型内生压缩能力，压缩比可控。
- **Truncation Baseline**：粗暴截断导致性能骤降；本文证明基于语义重要性的选择性压缩可在同比例压缩下维持更高准确性。

## 局限性与未来方向
- **未验证更大模型**：受计算限制，未在Qwen2.5-32B/72B等更大模型上实验，其压缩效果与泛化能力有待探索。
- **重要性度量通用性**：LLMLingua-2未针对数学/数值token专门优化，可能影响公式类token的保留质量。
- **长CoT模型适应性**：未测试在QwQ-32B-Preview等专为长CoT设计的模型上的表现，其在多步推理场景的有效性待验证。
- **极端压缩比稳定性**：当$\gamma<0.5$时压缩比贴合度下降，过激剪枝可能导致关键信息丢失，需更鲁棒的阈值策略。

## 研究启发与可借鉴点
- **token粒度压缩思路**：将语义重要性评估从prompt延伸至模型生成序列，为其他生成任务（如代码生成、摘要）的效率优化提供新视角。
- **小比例LoRA微调**：仅用0.2%参数即可实现有效压缩能力学习，证明轻量微调在保留核心能力下的效率提升可行性。
- **动态压缩比控制**：引入$\gamma$作为推理时可控超参，支持按需平衡延迟与准确性，为实际部署提供灵活配置手段。
- **混合比例训练**：在训练数据中混合多种压缩比（含$\gamma=1.0$完整CoT），避免模型过度适应单一压缩模式，提升泛化鲁棒性。

## 关键术语表
- **Chain-of-Thought (CoT)**：大语言模型进行复杂推理时生成的逐步推导文本序列。
- **TokenSkip**：本文提出的方法，通过选择性跳过低重要性token实现CoT压缩。
- **LLMLingua-2**：基于双向语言模型的token重要性度量工具，用于评估文本中各token的语义贡献。
- **压缩比 ($\gamma$)**：指定保留token的比例，$\gamma=0.6$表示保留60%的token。
- **LoRA (Low-Rank Adaptation)**：低秩适应微调技术，通过冻结主参数并训练低秩矩阵实现高效参数更新。
- **Test-time Scaling**：在推理阶段增加计算资源（如延长CoT）以提升模型性能的策略。

## 可复现要素
- **数据集**：GSM8K、MATH-500均公开可用；CommonsenseQA、MMLU-STEM亦公开。
- **代码与权重**：代码与检查点已开源至https://github.com/hemingkx/TokenSkip。
- **关键超参**：压缩比采样集合$\{0.5,0.6,0.7,0.8,0.9,1.0\}$；LoRA rank=8, alpha=16；学习率5e-5（余弦衰减）；训练epoch=3；优化器AdamW，warmup ratio=0.1。
