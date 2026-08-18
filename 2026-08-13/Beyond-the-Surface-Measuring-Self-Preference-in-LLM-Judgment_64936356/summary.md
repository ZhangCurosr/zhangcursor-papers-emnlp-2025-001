---
title: "Beyond-the-Surface-Measuring-Self-Preference-in-LLM-Judgment"
source: https://aclanthology.org/2025.emnlp-main.86.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:43:01"
field: "大语言模型评估与对齐"
keywords: ["self-preference bias", "LLM-as-a-judge", "evaluation bias", "model alignment", "bias measurement"]
innovations: ["提出DBG score，通过gold judgments解耦回复质量与自我偏好偏差", "系统度量不同版本/规模/推理能力的LLM自我偏好偏差", "发现回复风格对齐和同数据微调可有效缓解自我偏好偏差"]
benchmarks: ["AlpacaEval", "WMT19", "TruthfulQA"]
---

# 论文速读：Beyond-the-Surface-Measuring-Self-Preference-in-LLM-Judgment

## 一句话总结
论文针对LLM作为judge时的自我偏好偏差（self-preference bias）问题，提出**DBG score**指标，通过引入多个强模型的共识作为**gold judgments**（真实质量的代理），将回复质量与自我偏好偏差解耦，实现更准确的偏差量化。基于该指标，系统研究了不同版本、规模和推理能力的LLM的自我偏好偏差，并探索了回复风格和微调数据对偏差的影响。

## 研究问题与动机
1. **现有方法混淆回复质量与偏差**：已有工作用judge模型对自己回复与其他模型回复的评分差作为偏差指标，但当judge模型自身回复质量较高时，高评分既可能来自真实质量，也可能来自自我偏好，无法区分。
2. **缺乏对多维度的系统测量**：现有研究对自我偏好偏差的评估不够全面，缺少对不同版本（预训练/后训练）、不同规模（0.5B~72B）、不同推理能力模型的系统对比。
3. **缓解策略尚未探索**：关于自我偏好偏差的成因机制及如何有效缓解，现有工作几乎未涉及，缺乏 actionable insights。

## 核心贡献（创新点）
1. **提出DBG score**：通过gold judgments（多强模型共识）作为真实质量的代理，计算judge对自己回复评分与gold judgment的差异，首次将回复质量与自我偏好偏差有效解耦。
2. **系统度量多维度模型偏差**：覆盖预训练/后训练模型、0.5B~72B多规模、以及大推理模型（LRMs），发现后训练模型偏差不比预训练模型更严重，较大模型偏差更小，LRMs同样存在显著偏差。
3. **探索并验证偏差缓解策略**：首次发现统一回复文本风格、以及在相同数据上微调不同模型均可显著降低自我偏好偏差。
4. **从注意力角度揭示潜在机制**：发现模型在judge过程中对自身回复分配更高注意力分数，为自我偏好偏差提供了可解释的底层机制线索。

## 方法详解
- **Bradley-Terry建模**：设judge A对回复r的评分为 S_A(r) ≈ Q(r) + b_A(r)，其中Q(r)为真实质量，b_A(r)为judge对回复的偏好偏差。在自我偏好场景下，假设b_A(r_B)=0且b_A(r_A)>0。
- **DBG score定义**：\hat{w}_A = E_x[σ(δ + b_A) - σ(δ)]，其中δ = Q(r_A) - Q(r_B)为质量差距，σ为sigmoid函数。该公式消除了δ（回复质量）的混淆，聚焦于偏差项b_A。
- **Gold judgments构建**：使用三个强LLM（GPT-4o-mini、Gemini-1.5-Flash、DeepSeek-V3）的判决结果聚合，以共识方式逼近无偏的真实质量。通过一阶泰勒近似可证明：\hat{w}_A ≈ E_x[σ'(δ)] · E_x[b_A]，即DBG score是真实偏差的线性缩放估计量。
- **实验设置**：temperature=0保证确定性输出；对预训练模型使用2-shot in-context learning；通过交换回复位置并取平均来缓解位置偏差；限制回复最大长度以减轻长度偏差。

## 实验与结果
- **数据集**：AlpacaEval（helpfulness）、WMT19 de-en（翻译）、TruthfulQA（truthfulness），各采样500条。
- **核心发现**：
  - Llama-3.1-70B的DBG score仅0.4%，而Llama-3.1-8B高达21.6%，表明**更大模型偏差显著更小**。
  - Qwen2.5-0.5B-Instruct的DBG score为41.7%，而14B版本仅2.1%，**7B以下模型偏差急剧增大**。
  - 后训练模型（如Llama-3.1-8B-Instruct的DBG=0.2%）偏差不比预训练版本更严重，甚至有所降低。
  - 大推理模型DS-R1-Distill-Qwen-32B的DBG score为4.8%，略高于Qwen2.5-72B-Instruct的2.6%，**推理能力并未消除自我偏好偏差**。
- **缓解效果**：
  - 风格统一后，Llama-3.1-70B的DBG从3.3%降至1.4%，Llama-3.1-8B从18.7%降至7.2%。
  - 同数据（UltraChat-200k）微调后，Llama-3.1-8B-Instruct和Qwen2.5-7B-Instruct的DBG分别从10.5%/2.1%降至2.1%/1.1%。
- **可靠性验证**：
  - Gold judgments与人工标注在74%样本上一致，人工胜率与gold胜率差距仅约3个百分点（Table 1）。
  - 三个gold judge模型两两间agree rate达83%~86%（Table 2），说明共识构建可靠。

## 相关工作脉络
1. **Zheng et al. (2023) LLM-as-a-Judge**：开创性工作，提出用LLM替代人工评估，但未系统研究自我偏好偏差的量化与缓解。
2. **Panickssery et al. (2024) LLM Evaluators Recognize and Favor Their Own Generations**：发现LLM评估器倾向于偏好自身生成内容，但未解决回复质量混淆问题。
3. **Ye et al. (2024) Justice or Prejudice?**：系统量化LLM judge中的多种偏差（位置、长度、权威、情感等），但未提出解耦质量与偏差的测量方法。
4. **Chen et al. (2025) Do LLM Evaluators Prefer Themselves for a Reason?**：并发工作，聚焦可验证任务（如数学推理）的自偏好偏差，本文聚焦开放式任务并提出解耦方法。
5. **Li et al. (2025) Preference Leakage**：研究偏好泄露问题，与本文关注的自我偏好偏差形成互补视角。
6. **Wataoka et al. (2024) Self-Preference Bias in LLM-as-a-Judge**：较早提出自我偏好偏差概念，但未系统探索缓解手段和注意力机制。

## 局限性与未来方向
1. **Gold judge能力有限**：受成本约束，仅使用GPT-4o-mini、Gemini-1.5-Flash等中等能力模型构建gold judgments，未使用GPT-4o或Gemini-1.5-Pro等更强模型，可能影响gold标准可靠性。
2. **未控制的偏差源**：仅缓解了位置偏差和长度偏差，未考虑权威偏差、情感偏差等其他潜在偏差因素。
3. **任务范围受限**：实验局限于指令跟随和翻译任务，尚未探索agent任务和对话任务中的自我偏好偏差。
4. **风格改写不彻底**：统一风格虽降低了偏差，但未完全消除，说明回复内容本身也可能驱动偏差。
5. **未来方向**：使用更强gold judge、探索更多任务类型、研究其他偏差的交互影响、深入挖掘注意力层面的因果机制。

## 研究启发与可借鉴点
1. **解耦质量与偏差的测量思路**：DBG score通过将gold judgments作为基准来隔离回复质量干扰，这一思路可迁移到其他评估偏差（如位置偏差、长度偏差）的研究中。
2. **多模型共识构建可靠参考**：使用多个强模型的共识构建"gold标准"来近似人类判断，是一种低成本、可复现的有效策略，可在其他评测任务中复用。
3. **风格对齐作为去偏手段**：发现统一回复风格可显著降低自我偏好偏差，提示在模型对比评估中控制文本风格一致性可能是必要的实验规范。
4. **同数据微调促进judge公平性**：相同微调数据可使不同模型的judge倾向趋同，为设计公平评测协议提供了新思路。
5. **注意力分析揭示机制**：从注意力分配角度解释自我偏好偏差的底层机制，为后续可解释性研究提供了分析范式。

## 关键术语表
**Self-Preference Bias（自我偏好偏差）**：LLM担任judge时倾向于给自身生成回复赋予更高分数的系统性偏差。
**Gold Judgments（黄金判断）**：由多个强LLM的判决结果聚合而成的无偏参考分数，作为回复真实质量的代理。
**DBG Score**：Judge模型对自己回复的评分与gold judgment之间的差值，用于量化自我偏好偏差程度的核心指标。
**Bradley-Terry Model（BT模型）**：用于建模pairwise比较偏好的概率模型，本文用于形式化judge的偏好概率。
**Position Bias（位置偏差）**：LLM在pairwise比较中倾向于选择位置更靠前（如第一个）回复的偏差。
**Length Bias（长度偏差）**：LLM倾向于给更长回复打更高分的系统性偏差。
**Large Reasoning Models (LRMs)**：经过强化学习等训练、具备复杂推理能力的大模型（如DeepSeek-R1、QwQ）。
**In-Context Learning（上下文学习）**：通过在prompt中提供少量示例来引导模型完成任务，本文用于预训练模型的judge prompting。

## 可复现要素
- **数据集**：AlpacaEval、WMT19（de-en）、TruthfulQA，均为公开数据集；实验随机采样500条样本。
- **代码**：已开源，地址为 https://github.com/zhiyuanc2001/self-preference
- **模型**：使用了GPT-4o-mini、Gemini-1.5-Flash、DeepSeek-V3、Llama-3.1-8B/70B、Qwen2.5-7B/72B、Gemma-2-9B等多个开源或API模型。
- **关键超参**：temperature=0；pre-trained模型使用2-shot in-context learning；回复最大长度受限（具体见Appendix A.7 prompts）；位置交换取平均缓解position bias。
