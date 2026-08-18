---
title: "Diagnosing-Memorization-in-Chain-of-Thought-Reasoning-One-To"
source: https://aclanthology.org/2025.emnlp-main.157.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:47:40"
---

# 论文速读：Diagnosing-Memorization-in-Chain-of-Thought-Reasoning-One-To

## 一句话总结
本文提出 STIM 框架，首次在 token 粒度上对 CoT 推理链中的记忆化现象进行多来源（局部、中期、长期）量化诊断，揭示模型在复杂任务与分布外长尾输入下更依赖记忆化并导致错误，同时证明高记忆分可有效定位推理链中的错误 token。

## 研究问题与动机
- CoT 推理中单个错误 token 极易引发级联错误，但现有记忆度量仅停留在序列或最终答案级别，无法实现 token 级精准归因。
- 既有指标要么仅评估输出序列的新颖性/复制度，要么仅分析输入提示的影响，缺乏对局部上下文、输入 prompt 与部分生成内容等多源影响的统一建模。
- 模型在输入分布偏移（长尾/低频实体）时性能显著下降，但记忆模式如何随任务复杂度、分布类型及推理正确性动态演变仍不清楚。
- 缺少可操作的诊断工具来区分“记忆辅助正确推理”与“记忆诱发错误推理”的具体边界，制约了对模型真实推理能力的评估。

## 核心贡献（创新点）
- 提出 STIM（Source-aware Token-level Identification of Memorization），在 CoT 推理中实现 token 粒度的多来源记忆量化。与现有工作的本质区别在于同时解耦并计算局部 n-gram 续写、输入 prompt 共现与部分输出前缀共现三类影响。
- 构建覆盖四类推理任务与 base/long-tail 分布的系统性诊断协议，揭示记忆强度随任务复杂度与分布外偏移而增强的规律。区别于仅报告整体准确率的评测，本文精细刻画了记忆在正确/错误推理步中的分布反转现象。
- 证明 STIM 分数可作为错误 token 的有效诊断信号，跨任务聚合达到 Precision@1 = 31.2%、Recall@1 = 28.8%，显著优于随机基线。与 prior work 仅做定性分析或机制解释不同，本文提供了可落地的错误定位定量工具。

## 方法详解
- 核心思路：对目标 token $x$，提取解码时 top-20 候选 token 的概率分布 $P(x_i|p)$，并与预训练语料中的统计频率计算 Spearman 相关系数，从而量化不同上下文来源的记忆驱动强度。
- 局部记忆（Local）：寻找最长连续前缀 $w$ 使得 n-gram $[w;x]$ 在预训练语料中至少出现一次，计算 $\text{STIM}_{loc}(x) = \rho(\{P(x_i|p)\}_{i=1}^{20}, \{f([w;x_i])\}_{i=1}^{20})$。
- 长期记忆（Long-range）：通过 token saliency 选取 top-5 最具影响力的输入 token $S_l$，计算候选 token 与 $S_l$ 的共现频率 $f(S_l, x_i)$，得分 $\text{STIM}_{long}(x) = \rho(\{P(x_i|p)\}, \{f(S_l, x_i)\})$。
- 中期记忆（Mid-range）：定位生成前缀中最短能触发目标 token 的子串，提取其中 top-5 显著 token $S_m$，计算共现频率得分 $\text{STIM}_{mid}(x) = \rho(\{P(x_i|p)\}, \{f(S_m, x_i)\})$。
- 主导来源与聚合：定义 $\text{STIM}_{max} = \max(\text{STIM}_{loc}, \text{STIM}_{mid}, \text{STIM}_{long})$，用于判断该 token 主要受哪类来源驱动，并作为错误定位主指标。
- 错误步骤筛选：使用 VersaPRM 标记 PRM 分数 < 0.9 的步骤为错误步，聚焦首个错误步分析；错误 token 由 GPT-4o 标注并通过人工/模型交叉验证。

## 实验与结果
- 数据集与任务：Applied Math（GSM8K）、Formula Calculation（公式子表达式）、Counting（受控频率水果列表）、Capitalization（书名大小写）；各任务包含 Base 与 Long-tail（数字扩展/浮点化/二进制/低频词/超长列表/非常规格式）变体。
- 模型与工具：主实验使用 OLMo-2-13B-Instruct，复现使用 OLMo-2-7B-Instruct；预训练语料索引依赖 Infinigram/Dolma 1.7；步骤验证使用 VersaPRM；显著性 token 提取使用 LERG；解码策略为 greedy。
- 主要发现与数值：
  - 任务复杂度与记忆正相关：Applied Math 与 Formula Calculation 的 $\text{STIM}_{max}$ 均值系统性高于 Counting/Capitalization。
  - 分布偏移放大记忆信号：4 个任务中 3 个在 Long-tail 下记忆分数高于 Base。
  - 记忆角色反转：Base 分布下正确步记忆分更高（记忆辅助推理）；Long-tail 下错误步记忆分更高（记忆误导推理）。
  - 错误主导来源：Local 记忆是最高频的错误驱动源，占比高达 67%；高推理任务在 Long-tail 下 Local 错误比例显著下降，低推理任务几乎不变。
  - 错误定位性能（全任务聚合）：$\text{STIM}_{max}$ 达 P@1=31.2%, R@1=28.8%；P@3=50.5%, R@3=56.6%，随机基线分别为 14.2% 与 40.1%。
- 最强结果：Applied Math 上 $\text{STIM}_{max}$ 取得 P@3=57.2%, R@3=67.0%，较随机基线提升约 27pp（P@3）与 32pp（R@3）。

## 相关工作脉络
- 记忆提取与隐私度量（Carlini et al., 2022; Biderman et al., 2023）：关注训练数据重现风险，本文聚焦 CoT 推理过程中的 token 级记忆诊断，不涉及隐私攻击或提取。
- 新颖性与复制度量（McCoy et al., 2023; Merrill et al., 2024; Lu et al., 2024）：多基于全局 n-gram 重叠或序列级 novelty 评分，本文推进至 token 粒度并解耦多源影响。
- 记忆与推理分离（Xie et al., 2024; Hong et al., 2025; Lou et al.,
