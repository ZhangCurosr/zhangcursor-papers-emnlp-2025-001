---
title: "SURGE-On-the-Potential-of-Large-Language-Models-as-General-P"
source: https://aclanthology.org/2025.emnlp-main.162.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-08-18 17:46:09"
---

# 论文速读：SURGE-On-the-Potential-of-Large-Language-Models-as-General-P

## 一句话总结
本文提出 SURGE 基准，系统评估大语言模型在不实际运行程序的前提下预测任意代码行为的“代理代码执行器”能力，覆盖 8 类代码场景、7 种编程语言共 1160 道题目，揭示了当前 LLMs 在该任务上的性能瓶颈、提示策略效应、Chat/Coder 模型差异及规模缩放规律。

## 研究问题与动机
- **核心问题**：LLMs 能否作为通用代理代码执行器（General-purpose Surrogate Code Executor），在不依赖真实编译/执行环境的情况下准确预测代码运行输出或行为？
- **动机1**：科学计算模拟与容器化执行成本高昂；安全敏感或不可信代码环境禁止直接运行；大量代码强依赖特定硬件/运行时环境，难以标准化执行。
- **动机2**：准确的代码行为预测对代码理解、代码生成、自动化调试及强化学习 Reward Model 设计具有广泛下游价值。
- **现有方法不足**：传统符号执行与沙箱执行泛化性差且依赖真实环境；已有神经执行器仅针对极狭窄任务设计；缺乏统一基准量化 LLMs 作为“通用代理执行器”的真实潜力与边界。

## 核心贡献（创新点）
1. **提出 SURGE 基准**：首个覆盖多语言、多场景、多评估维度的代理代码执行评测体系，包含 8 个子集 1160 道题目。**与已有工作相比**，突破了单一语言或窄任务代码理解的局限，首次将形式化验证、科学计算、耗时计算等异构任务统一纳入代理执行视角。
2. **设计多元化数据集构建流程**：融合 Iterative Refactor、Repository Sampling、Manual Implementation、Inference & Verification 四种构建方法，并引入严格的防数据污染策略。**与以往依赖单一自动抓取或人工标注的数据集相比**，显著提升了数据的真实性、多样性与评测可信度。
3. **建立多维度评估协议与提示策略对照**：针对不同子集匹配 Exact Match、Jaccard Similarity、RAE 等指标，系统对比 0-shot / 0-shot CoT / Few-shot CoT 三种提示策略。**与先前仅报告单一准确率的工作相比**，揭示了提示工程与任务类型的交互效应，避免了评估片面性。
4. **刻画代理执行器的能力边界与反常现象**：明确指出大模型在形式化验证子集易“过度纠错”导致性能回落，且执行时间>1秒的程序构成当前 SOTA 的硬性瓶颈。**与以往偏向正面性能报告的研究相比**，本文重点揭示了 LLM 替代真实执行器的物理与逻辑极限。

## 方法详解
- **基准架构**：由 8 个子集构成，每类子集针对特定代码行为预测任务设计专属评估指标。
- **数据集构建方法**：
  - `Iterative Refactor`（ML、CL、BG）：LLM 辅助生成+迭代重构+人工验证
  - `Repository Sampling`（RL）：从公开/定制 GitHub 仓库中抽取≥10个测试用例
  - `Manual Implementation`（SC、TC、DR）：手工编写确保计算逻辑严谨
  - `Inference & Verification`（FL）：基于 Lean4 + Goedel-Prover 生成并通过形式化验证确保正确/错误标签
  - **防污染策略**：自动过滤代码注释，人工剔除含 `assert` 语句及可能泄露答案的注释。
- **评测协议**：选取 21 个 LLM（17 开源 + 4 闭源），temperature=0，采用 0-shot w/o CoT、0-shot w/ CoT、Few-shot w/ CoT（3 示例）三种提示策略。
- **评估指标与公式**：
  - `Exact Match`（ML、CL）：预测输出与标准答案完全一致
  - `Jaccard Similarity`（BG、DR）：预测集合与真实集合的交集/并集
  - `RAE (Relative Absolute Error)`（SC）：`RAE(ŷ, y) = |y - ŷ| / |y|`；张量/向量预测按长度对齐（不足补零、超出截断）后逐元素平均
  - `Mixed / Custom`（RL、FL）：FL 子集要求输出结构化 JSON（`errors`, `pass`, `complete`）
- **训练与缩放实验**：基于 FL 子集数据微调 0.5B→7B 模型，验证训练数据量与步骤数的正向缩放趋势。使用 Llama-Factory 平台，关键超参：batch size=128，lr=1.0e-5，epochs=2，cosine scheduler，warmup ratio=0.1，bf16 精度。

## 实验与结果
- **评测规模**：21 个 LLM（含 Claude-3.5-Sonnet、DeepSeek-V3、GPT-4o 系列、Qwen-2.5/Coder 系列、LLaMA-3.1/3.3 系列等）
- **核心性能（Avg 综合得分）**：
  - 零样本最佳：Claude-3.5-Sonnet **51.59**；DeepSeek-V3 **42.53**
  - Zero-shot CoT 最佳：Claude-3.5-Sonnet **59.47**；DeepSeek-V3 **54.97**
  - Few-shot CoT 最佳：Claude-3.5-Sonnet **58.49**；DeepSeek-V3 **59.08**
  - 最弱模型：LLaMA-3.1-8B 零样本 Avg = **8.49**
  - FL 子集全模型极低，最佳 Claude-3.5-Sonnet 零样本仅 **9.09**
  - Qwen-2.5-Coder-32B Few-shot CL 得分 **75.00**，接近 Claude-3.5-Sonnet 的 70.00
- **关键结论**：
  1. 基准判别力强，最强模型仅达中等水平，说明任务具挑战性。
  2. CoT 与 Few-shot 均显著提升性能，但子集间效果存在差异。
  3. 模型规模与性能呈正向缩放趋势（0.5B→7B 验证）。
  4. 零样本下 Coder 模型优于 Chat 模型，其他设置下 Chat 反超（推理与模仿能力更强）。
  5.
