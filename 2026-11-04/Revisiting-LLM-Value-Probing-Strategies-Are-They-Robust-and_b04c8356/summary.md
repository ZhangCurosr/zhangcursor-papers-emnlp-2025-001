---
title: "Revisiting-LLM-Value-Probing-Strategies-Are-They-Robust-and"
source: https://aclanthology.org/2025.emnlp-main.7.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:56:34"
---

# 论文速读：Revisiting-LLM-Value-Probing-Strategies-Are-They-Robust-and

## 一句话总结
本文系统对比了Token Logits、Sequence Perplexity和Text Generation三种LLM价值观探测策略，从抗非语义扰动（鲁棒性）与上下文引导/行为预测能力（表达力）两个维度进行评估，发现所有方法在主观问题上均存在显著方差，且探测到的价值观与模型实际行为偏好仅呈弱相关。

## 研究问题与动机
1. **缺乏系统性方法对比**：现有价值观研究多依赖单一MCQ打分方式，但MCQ设置对提示模板与选项顺序/符号高度敏感，不同提取策略得出的表征是否可靠尚未明确。
2. **表征是否具备内在一致性**：不同组合的探测设置会产生不同的概率分布，需验证这些分布是否真正反映模型内在信念，而非仅捕捉表面统计模式。
3. **价值观能否延伸至行为预测**：若探测值无法与模型在真实价值相关场景下的行动偏好对齐，则其实际应用价值存疑。
4. **人口统计上下文的可引导性**：注入文化/国家背景后，不同方法的价值观表征能否像人类一样发生符合预期的调整，是衡量其表达力的关键。

## 核心贡献（创新点）
1. **首次系统性横向对比三种主流探测方法**：在统一扰动体系下（提示风格变化+选项选择偏差）全面评估Token、Sequence、Text方法的稳定性差异，填补了价值观提取策略对比的空白。
2. **提出“价值观-行为一致性”评估任务**：合成并公开包含价值相关情境与行为选项的新数据集，用Pearson/Spearman相关系数量化探测值与模型实际动作偏好的关联，突破仅依赖MCQ答案的评估局限。
3. **构建双维表达力评估框架**：将人口统计提示对齐度改进（EMD变化）与价值-行为相关性作为表达力的双重指标，揭示不同方法在上下文引导与下游预测上的本质差异。
4. **揭示鲁棒性与表达力的内在权衡**：发现Sequence Perplexity整体最稳健，但Text Generation方法虽人类可读性最强却最难以人口统计上下文引导；去偏（Cyclic Permutation）仅能部分缓解问题，表明当前探测范式存在结构性局限。

## 方法详解
**三种探测打分方法**：
- **Token Logits**：取选项符号（如`"A"`）在输入末尾第一个生成位置的logits，经softmax得到分布：$p^{\mathrm{token}} = \mathrm{softmax}(l)$。
- **Sequence Perplexity**：计算完整选项序列（含符号与文本，如`"A Strongly agree"`）的反向困惑度并归一化：$p^{\mathrm{seq}} = \mathrm{ppl}^{-1} / \sum_i \mathrm{ppl}_i^{-1}$，长度归一化使其比Token方法更稳定。
- **Text Generation**：对模型采样N次（本文N=10, temperature=1.0），统计合法选项出现频率：$p^{\mathrm{text}} = n/N$；无合法输出时均摊计数。

**鲁棒性评估设计**：
- **提示风格扰动**：Default（标准指令）、Prefixed（前缀补全`"Certainly! I would select option"`）、One-shot（提供无关示例）。
- **选择偏差扰动**：Position Bias（反转选项标签顺序）、Token Bias（替换为`0/1/2/3`等符号）。
- **评估指标**：Mismatch Rate（多数票/最大概率选项是否改变）与 Jensen-Shannon Divergence（底层概率分布偏移量）。

**表达力评估设计**：
- **人口统计引导（Demographic Prompting）**：注入国家_persona_，分别以有无提示查询，计算模型分布与WVS人类调查分布的Earth Mover’s Distance (EMD)，改进量越大说明可引导性越强。
- **价值-行为一致性（Value-Action Agreement）**：用GPT-4-turbo生成情境与对立行为选项，经自批判过滤（Q1-Q3确认真实性、相关性、合理性）后，让模型对行为打分，计算概率权重与行为分数的Pearson/Spearman相关系数。

## 实验与结果
- **数据集**：World Values Survey (WVS) Wave 7，筛选206道独立题目覆盖13个主观领域；行为数据集由GPT-4-turbo合成并经Llama-3.1-70B、Claude-3.7-sonnet交叉验证。
- **模型**：Bloomz-7B, Falcon-7B, Mistral-7B-v0.3, Llama
