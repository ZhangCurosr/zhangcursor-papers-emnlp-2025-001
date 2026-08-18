---
title: "Thinking-Out-Loud-Do-Reasoning-Models-Know-When-They-re-Righ"
source: https://aclanthology.org/2025.emnlp-main.73.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:42:13"
field: "大语言模型校准与可靠性"
keywords: ["Large Reasoning Models", "Verbalized Calibration", "Uncertainty Quantification", "Reinforcement Learning", "Chain-of-Thought", "Factuality Benchmarks"]
innovations: ["系统比较SFT与RL训练范式对语聊化校准的渐进影响", "揭示推理SFT在提升推理能力时降低事实性校准与知识边界意识", "验证推理RL跨领域泛化提升校准并部分恢复事实性性能"]
benchmarks: ["AIME 2024/2025", "GPQA-Diamond", "SimpleQA", "FreshQA", "LiveBench-Reasoning"]
---

# 论文速读：Thinking-Out-Loud-Do-Reasoning-Models-Know-When-They-re-Righ

## 一句话总结
该论文通过系统的实证研究，分析了大型推理模型（LRMs）的语聊化置信度校准表现，发现监督微调（SFT）与强化学习（RL）可逐步改善推理密集型任务中的校准质量，但推理训练也可能导致模型对自身知识边界意识降低，产生“推理税”现象。

## 研究问题与动机
1. **核心问题**：具备自我反思行为的推理模型是否真正具备更好的校准能力？即其语聊化置信度是否与实际正确性更一致。
2. **现有不足**：指令微调模型常系统性高估确定性，校准结果脆弱且高度敏感于提示设计；而LRMs内嵌的长思维链与反思行为对校准的影响尚未被系统探索。
3. **研究缺口**：先前工作多关注采样式不确定性量化方法（如语义熵），缺乏对轻量化语聊化置信度在推理训练范式下的统一评估。
4. **动机**：若推理模型更具内省性，其置信度表达应更可靠，这对高风险场景下的可信部署至关重要。

## 核心贡献（创新点）
1. **首次系统比较指令微调、SFT推理、RL推理三类模型在多种基准上的语聊化校准表现**，揭示SFT与RL的渐进式增益轨迹（U型曲线）。  
   *本质区别*：区别于以往仅报告LLM过度自信的孤立研究，本文控制基座架构相同，隔离训练策略的独立效应。
2. **发现推理SFT显著提升数学/科学推理准确率与校准，但小尺度模型的事实性校准反而退化，且“I don’t know”回答率大幅降低**。  
   *本质区别*：不同于prior work仅关注整体校准误差，本文精确定位到SFT阶段即可引发事实性校准下降，并关联知识边界意识减弱。
3. **实证验证推理RL在不同领域（编程、数学）训练后仍能泛化提升跨任务校准，支持RL在培养领域无关置信度估计上的作用**。  
   *本质区别*：为“SFT memorizes, RL generalizes”的理论论点在置信度校准维度提供了补充证据。
4. **分析推理链长度与置信度、校准的关系，发现较短或不详尽的推理常伴随更高置信度，长链可能引入额外不确定性**。  
   *本质区别*：基于Thoughtology视角的量化延伸，此前缺乏对长CoT影响校准的非单调性细致剖析。

## 方法详解
1. **模型选择**：三类模型共享相同基座架构——指令微调模型（Qwen2.5‑14B/32B‑Instruct、DeepSeek‑V3）、SFT推理模型（DeepSeek‑R1‑Distill‑Qwen‑14B/32B，从R1蒸馏的80万条推理轨迹中微调）、RL推理模型（DeepCoder‑14B、Skywork‑OR1‑32B、DeepSeek‑R1，在SFT基础上进行数学/编程强化学习）。
2. **数据集**：数学（AIME 2024 & 2025，各30题）、事实性（SimpleQA、FreshQA）、科学推理（GPQA‑Diamond、SuperGPQA）、通用推理（LiveBench‑Reasoning）。
3. **提示策略**：
   - **Vanilla CoT**：直接要求逐步推理并输出最终答案与置信度（百分比）。
   - **Vanilla CoT with probability mass**：置信度以0.0‑1.0概率形式输出。
   - **Self‑reflection prompting**：两轮对话，首轮生成答案，次轮要求模型评估自身置信度。
4. **评估指标**：
   - **Expected Calibration Error (ECE)**：将样本按置信度均分为M=10箱，计算各箱平均置信度与真实准确率的加权平均差异。
   - **Adaptive Calibration Error (ACE)**：动态调整分箱边界使每箱样本数相等，降低对分箱策略的敏感性。
   - **AUROC & AUPRC**：评估置信度区分正确/错误预测的能力，AUPRC分别针对正例（AUPRC‑P）与负例（AUPRC‑N）。
5. **推理设置**：温度固定为0.6，最大生成token为32,000，保证长思维链生成；除DeepSeek系列使用API外，其余均基于Huggingface Transformers库。

## 实验与结果
- **数据集与基线**：覆盖数学、科学、事实性、通用推理四类基准；基线包含8个开源模型（14B与32B尺度）。
- **主要结果**：
  - **数学（AIME）**：R1‑Distill‑Qwen‑32B准确率从Qwen2.5‑32B的9.67%大幅提升至65.7%，ECE从0.752降至0.240，AUROC从0.695升至0.813；RL推理模型DeepSeek‑R1达到**68.0%准确率、ECE 0.136、AUROC 0.942**，为全实验最强结果。
  - **科学推理（GPQA‑Diamond）**：DeepSeek‑R1准确率为68.7%，ECE仅0.082。
  - **事实性（SimpleQA）**：小尺度SFT推理模型（R1‑Distill‑Qwen‑14B）准确率5.69%低于指令模型6.04%，ECE 0.719高于指令模型的0.625；但RL模型DeepSeek‑R1准确率29.7%，ECE 0.324，显著改善。
  - **“不尝试”回答数**：指令模型14B在SimpleQA+FreshQA上共1,136个未尝试，SFT推理模型仅102个，RL推理模型103个，显示推理模型更倾向猜测。
- **结论**：推理SFT在推理密集型任务上同时提升准确率与校准；RL推理进一步改善校准且跨领域泛化；但小尺度SFT推理模型在 factuality 上校准退化，且知识边界意识明显减弱。

## 相关工作脉络
1. **对比 prior calibration studies (Xiong et al., 2024; Tian et al., 2023)**：这些工作指出LLM系统性高估确定性且校准对提示敏感；本文聚焦LRMs特有的自我反思行为对校准的影响，揭示训练范式的作用。
2. **对比 uncertainty quantification works (Vashurin et al., 2025; Kuhn et al., 2023)**：这些研究评估采样式方法（语义熵等）；本文采用轻量级语聊化置信度作为评估 lens，更贴近实际应用接口。
3. **对比 LRMs training pipeline 研究 (DeepSeek‑R1, QwQ, DeepCoder)**：这些工作介绍推理模型训练流程；本文控制基座相同，系统隔离 SFT 与 RL 的独立贡献，提供校准视角的对比。
4. **对比 "hallucination tax" 相关研究 (Song et al., 2025; Kirichenko et al., 2025)**：这些研究指出RL后模型更频繁尝试回答未回答问题；本文从校准维度量化该现象，并提供U型轨迹证据。
5. **对比 SFT vs RL 理论分析 (Chu et al., 2025)**：该工作提出“SFT memorizes, RL generalizes”；本文在置信度校准维度验证这一观点，并细化到不同训练阶段。

## 局限性与未来方向
- **自述局限**：
  1. 未测试代码推理任务，因多数模型难以同时输出代码片段与置信度。
  2. 事实性基准评估未进行人工验证输出正确性。
- **可推断未来方向**：
  1. 探索reward design以避免hallucination tax，平衡能力与诚实性。
  2. 研究更大尺度模型是否可克服小尺度模型的事实性校准退化。
  3. 设计融合事实性知识的推理训练策略，增强知识边界意识。
  4. 将语聊化校准作为模型发布前的必测指标，监控推理SFT阶段的校准退化。

## 研究启发与可借鉴点
1. **可复用实验框架**：控制基座架构、仅改变后训练阶段（SFT/RL）的对比设计，可有效隔离训练策略效应，适用于其他模型特性研究。
2. **可借鉴评估协议**：使用ECE/ACE结合AUROC/AUPRC多维度评估校准，尤其引入“not attempted”率作为知识边界意识的代理指标，比单一准确率更全面。
3. **创新机会**：将本文发现的“推理税”概念扩展到多模态或工具使用场景；设计自适应置信度校准模块，与推理链长度动态绑定。
4. **可迁移方法**：Self‑reflection prompting在 factuality 上的分歧效果（SimpleQA有害、FreshQA有益）提示需针对数据集特性定制反思策略。
5. **结合团队方向**：若团队关注模型可靠性，可将语聊化校准作为模型发布前的必测指标，并监控推理SFT阶段的校准退化。

## 关键术语表
- **Verbalized Confidence**：模型直接输出的确定性陈述（如百分比或概率），作为轻量级不确定性量化方式。
- **Calibration**：预测置信度与实际准确率之间的匹配程度，理想情况下置信度70%时应正确70%。
- **ECE (Expected Calibration Error)**：将样本按置信度均分为M=10箱，计算各箱平均置信度与真实准确率的加权平均差异。
- **ACE (Adaptive Calibration Error)**：动态调整分箱边界使每箱样本数相等，降低对标准ECE分箱策略的敏感性。
- **Reasoning Tax / Hallucination Tax**：推理训练导致模型更频繁尝试回答但知识边界意识降低，进而增加幻觉风险的现象。
- **Self-reflection Prompting**：两轮对话提示策略，首轮生成答案，次轮要求模型评估自身置信度。
- **SFT Reasoning Models**：通过对强推理模型生成的长思维链进行监督微调得到的模型。
- **RL Reasoning Models**：在SFT推理模型基础上，通过强化学习（如PPO）优化推理与自我反思行为得到的模型。

## 可复现要素
- **数据集**：AIME 2024/2025（公开）、SimpleQA（公开）、FreshQA（公开）、GPQA‑Diamond（公开）、SuperGPQA（公开）、LiveBench‑Reasoning（公开）。
- **代码/权重**：评估模型均为开源（Qwen2.5、DeepSeek系列），许可见附录E（MIT License / Qwen License）。本文实验代码未开源，论文未提及。
- **关键超参**：解码温度0.6，top_p根据模型不同（0.8‑1.0），最大序列长度3K‑32K（详见Table 6）；提示模板提供在Appendix A。
