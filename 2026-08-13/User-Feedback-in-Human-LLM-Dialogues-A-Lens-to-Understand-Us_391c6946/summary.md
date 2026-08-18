---
title: "User-Feedback-in-Human-LLM-Dialogues-A-Lens-to-Understand-Us"
source: https://aclanthology.org/2025.emnlp-main.133.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:43:33"
field: "人机交互与大模型对齐"
keywords: ["implicit feedback", "human-LLM interaction", "response regeneration", "SFT", "multi-turn dialogue", "user feedback extraction", "model alignment"]
innovations: ["构建密集标注的隐式反馈数据集并系统性分析反馈模式", "提出利用反馈语义内容的响应再生方法Regeneration w/ Semantics", "揭示反馈语义利用在简单任务有效但复杂任务失效的混合结果"]
benchmarks: ["MT-Bench", "WildBench", "LMSYS-chat-1M", "WildChat"]
---

# 论文速读：User-Feedback-in-Human-LLM-Dialogues-A-Lens-to-Understand-Us

## 一句话总结
论文系统研究了用户-LLM多轮对话中的隐式反馈，分析其出现模式与成因，并探索将反馈语义内容作为学习信号来蒸馏改进模型的效果；发现在简单评测（MTBench）上有效，但在复杂真实任务（WildBench）上效果不佳，揭示了从噪声真实用户数据中学习信号的复杂性。

## 研究问题与动机
- **核心问题**：LLM部署后可与用户长期交互，如何从自然对话日志中 harvesting 隐式用户反馈作为改进模型的学习信号？
- **现有方法不足**：Don-Yehiya et al. (2024) 将反馈仅分为正/负两类，训练时仅利用极性（鼓励产生正面反馈的响应、抑制产生负面反馈的响应），但本文发现该方法可能导致模型退化。
- **关键发现**：触发正面反馈的用户提示质量更低、毒性更高（部分用户因模型未拒绝不当请求而给予表扬），简单利用极性存在偏差风险。
- **研究动机**：是否除了反馈极性外，利用反馈的具体语义内容（用户希望补充/修正什么）能帮助模型更精准地改进？

## 核心贡献（创新点）
1. **构建密集标注的隐式反馈数据集**：首次对109条对话（LMSYS 75条 + WildChat 34条）的每个用户轮次进行完整标注，揭示了反馈在多轮对话中的分布动态。
2. **提出更强的自动反馈检测prompt**：设计含上下文示例的新prompt，利用GPT-4o-mini进行反馈分类，在密集标注设置下细粒度分类准确率接近翻倍（22.3% → 49.0%）。
3. **系统性分析反馈成因**：发现模型拒绝（refusal）并非负面反馈主因（占比<3%），负面反馈主要来自不满意模型生成；同时揭示触发正面反馈的提示质量反而更低。
4. **提出"Regeneration w/ Semantics"方法**：利用强LLM结合负反馈语义内容重新生成改进响应，而非仅利用反馈极性。
5. **揭示混合学习效果的深层原因**：SFT on feedback semantics 在简单任务（MTBench）提升显著，但在复杂任务（WildBench）上不如从零生成的baseline，说明真实用户反馈作为训练信号存在噪声与局限。

## 方法详解
- **反馈本体论**：沿用Don-Yehiya et al.的分类：正面反馈、改写（Rephrasing）、指出错误（Make Aware）、指出错误+修正（Make Aware with Correction）、请求澄清（Ask for Clarification）、无反馈。
- **密集标注策略**：人工标注109条完整对话的所有用户轮次（初始提示除外），Cohen's kappa达到0.70（二分类）、0.74（三分类）、0.60（细粒度）。
- **自动反馈检测**：使用GPT-4o-mini + 新prompt模板（含in-context示例），输入完整对话，逐轮输出反馈标签。
- **子对话定义**：定义 $s_i = \{u_i, m_i, u_{i+1}, m_{i+1}\}$，检查 $u_{i+1}$ 是否包含对 $m_i$ 的负反馈，构建 $D^{neg}$ 数据集（每数据集1K条）。
- **响应再生方法**：
  - **$m_i^{scr}$**（Baseline）：强LLM仅根据初始用户提示 $u_i$ 重新生成响应 $m_i^{scr} = \phi(u_i)$
  - **$m_i^{sem}$**（本文方法）：强LLM结合用户负反馈 $u_{i+1}$ 重新生成改进响应 $m_i^{sem} = \phi(u_i, m_i, u_{i+1})$
- **实验设置**：使用Better LLM（GPT-4o-mini）进行响应再生；基座模型采用Vicuna-7B和Mistral-7B；SFT训练超参：学习率5e-6，A100 GPU约2小时/次。
- **评估**：使用Reward Model进行成对比较（胜率），以及MT-Bench（80题）和WildBench（500题子集）进行LLM-as-a-Judge评分。

## 实验与结果
- **数据集**：LMSYS-chat-1M（评测导向、较短）、WildChat（真实任务导向、较长）。
- **反馈检测性能**：新prompt在稀疏设置下准确率81.1%（二进制），在密集设置下二进制41.6%、三分类55.4%、细粒度49.0%，显著优于prior work。
- ** pairwise胜率**（Table 3/6）：Better LLM无论是否使用反馈语义，均能大幅胜过Weak LLM原始响应（胜率81%-89%）；但 $m_i^{sem}$  vs $m_i^{scr}$ 对比，在LMSYS上48%（Eval w/ fb）到19%（Eval w/o fb），在WildChat上44%到9%，说明利用反馈语义并未一致优于从零生成。
- **SFT训练结果**（Table 4，关键数字）：
  - **MTBench**：WildChat数据上SFT on $m_i^{sem}$在Vicuna-7B达到6.86±0.02，Mistral-7B达到6.32±0.03，显著优于baseline（Vicuna base 6.09，Mistral base 3.09）。
  - **WildBench**：SFT on $m_i^{sem}$表现不佳，Vicuna-7B仅23.38±1.94，低于SFT on $m_i^{scr}$的28.74±1.16（WildChat D^rand）；Mistral-7B仅31.80±0.62，远低于D^rand的56.16±1.26。
  - KTO baseline（仅利用极性）在WildBench上全面负向或无效。
- **结论**：利用反馈语义在简单任务上有针对性提升，但在复杂任务上，直接从强模型蒸馏（随机采样）比利用负反馈更稳健。

## 相关工作脉络
- **Don-Yehiya et al. (2024)**：最早系统研究LMSYS中隐式反馈，分类为正/负并用于KTO训练；本文扩展为细粒度分类并分析语义利用，发现单纯极性训练的局限。
- **Shaikh et al. (2025)**：将用户响应建模为grounding acts，提供七类细粒度本体；本文与其反馈分类有重叠，但本文聚焦利用负反馈语义进行响应再生。
- **Ethayarajh et al. (2024) KTO**：基于前景理论的偏好优化方法；本文将其作为baseline对比，展示仅利用极性信号的不稳定性。
- **Lee et al. (2022); Chang et al. (2025)**：研究多轮人机协作中LLM性能下降问题；本文从真实交互日志角度补充证据，指出多轮反馈的复杂性。
- **Madaan et al. (2024) Self-refine; Qu et al. (2024)**：LLM自我迭代优化方法，不涉及用户反馈；本文对比思路：用户反馈 vs 自我反思的有效性。
- **Borges et al. (2023)**：从教育学角度分析LLM反馈；本文指出自然用户的反馈质量与专业教育者不同，需专门分析。

## 局限性与未来方向
- **偏好异质性**：不同用户可能有不同偏好（如详细vs简洁），论文未讨论对齐谁的偏好。
- **反馈时序重要性假设**：假设对话各位置的反馈同等重要，但实际不同阶段反馈作用可能不同（如最终确认 vs 中间修正）。
- **反馈指向性限制**：假设反馈仅针对最近一次模型响应，但用户可能希望修改更早的响应。
- **数据集覆盖**：仅使用LMSYS和WildChat两个数据集，且标注规模有限（109条对话）。
- **Future**：探索如何加权不同阶段的反馈、处理跨轮反馈指向、缓解用户提示质量偏差的影响。

## 研究启发与可借鉴点
- **反馈检测的工程优化**：使用in-context examples的prompt模板可显著提升自动分类性能，可在类似任务中复用。
- **区分反馈类型的重要性**：正/负极性过于粗糙，细粒度分类（如"请求澄清"vs"指出错误+修正"）可能蕴含不同信号价值。
- **复杂任务需谨慎利用用户反馈**：在简单基准上有效的信号在复杂真实任务上可能失效，提示我们在设计训练 pipeline 时需分层评估。
- **噪声数据分析视角**：从"模型拒绝导致负面反馈"到"用户提示质量导致偏差"的因果分析思路，值得在其他用户交互场景借鉴。
- **强模型蒸馏 vs 反馈语义利用的权衡**：当强模型可用时，直接从零生成可能比利用有噪声的用户反馈更有效，这为计算资源分配提供决策依据。

## 关键术语表
- **Implicit User Feedback**：用户在多轮对话中未明确说"你错了"但以续轮形式隐含表达的反馈信号。
- **Regeneration w/ Semantics**：利用强LLM结合用户负反馈的具体语义内容重新生成改进响应的策略。
- **KTO (Kahneman-Tversky Optimization)**：基于前景理论的偏好优化方法，直接利用非配对偏好数据（正/负反馈）训练模型。
- **Dense Annotation**：对对话中所有用户轮次（除初始提示外）进行完整反馈标注，区别于仅有部分轮次的Sparse标注。
- **Make Aware with Correction**：负反馈类型之一，用户不仅指出模型错误，还提供具体修正指令。
- **WildBench**：基于WildChat真实用户任务的评测基准，题目更长更复杂（平均499 tokens，复杂度4.31），用于评估模型在挑战性真实场景的表现。
- **LLM-as-a-Judge**：使用GPT-4等强模型作为评判器对生成结果进行自动评分的方法。
- **Reward Model (RM)**：用于对模型响应打分以进行成对比较的专用模型。

## 可复现要素
- **数据集**：LMSYS-chat-1M和WildChat；论文已公开标注数据集和代码（论文提及"dataset and code is shared publicly"，具体链接见论文脚注³）。
- **代码/权重**：论文开源（GitHub链接在论文中提供）。
- **关键超参**：SFT学习率5e-6，每轮训练约2小时/A100 GPU，每设置采样20K对话进行SFT。
- **评估设置**：MT-Bench 80题全量评测，WildBench 500题随机子集，GPT-4作为LLM-Judge，5次随机初始化训练取平均与方差。
