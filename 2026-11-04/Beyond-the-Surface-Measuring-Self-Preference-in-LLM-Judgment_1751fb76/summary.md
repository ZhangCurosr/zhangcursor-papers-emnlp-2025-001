---
title: "Beyond-the-Surface-Measuring-Self-Preference-in-LLM-Judgment"
source: https://aclanthology.org/2025.emnlp-main.86.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:25:52"
field: "LLM评估与公平性"
keywords: ["self-preference bias", "LLM-as-a-judge", "bias measurement", "model evaluation", "attention analysis", "post-training"]
innovations: ["提出DBG score通过gold judgments解耦响应质量与裁判偏见", "系统量化预训练/后训练/不同规模/LRM的自我偏好偏见", "发现风格对齐与一致训练数据可有效缓解自我偏好偏见"]
benchmarks: ["AlpacaEval", "WMT19", "TruthfulQA"]
---

# 论文速读：Beyond-the-Surface-Measuring-Self-Preference-in-LLM-Judgment

## 一句话总结
论文针对现有方法将响应质量与裁判偏见混淆的问题，提出**DBG score**（基于gold judgments的偏差度量），以更准确地量化LLM作为裁判时的自我偏好偏见；通过系统性实验揭示了偏见在不同版本/规模/推理能力模型中的分布规律，并发现统一响应风格和一致的训练数据可有效缓解该偏见。

## 研究问题与动机
1. **LLM裁判存在自我偏好偏见**：LLM作为judge时倾向于给自己的回复打更高分，导致评估结果失真，损害其可靠性。
2. **现有方法混淆质量与偏见**：直接用裁判给自己的分数减去给其他模型的分数来衡量偏见，但当裁判模型质量较高时，无法区分高分源于质量优势还是偏见本身。
3. **缺乏系统性跨维度分析**：已有工作多聚焦于特定模型或可验证任务（如数学推理），缺乏对开放生成任务中不同版本、规模、推理能力模型的系统性偏见量化。
4. **缓解策略尚未被探索**：现有研究主要识别偏见来源（如self-recognition、preference leakage），但对如何主动减轻偏见缺乏尝试。

## 核心贡献（创新点）
1. **提出DBG score**：通过引入gold judgments作为真实质量的代理，将偏见测量与响应质量解耦；相比现有方法（直接比较自评分与他人评分），本质区别在于用外部共识标准消除了质量混扰。
2. **系统性实证分析**：首次在同一框架下横向对比预训练/后训练版本、0.5B–72B规模梯度、以及推理模型（LRM）的自我偏好偏见，发现后训练并不必然加重偏见、大模型偏见更小。
3. **发现风格对齐的缓解效应**：将不同模型的回复重写为统一风格可显著降低DBG score，为低成本工程缓解策略提供了实验证据。
4. **揭示注意力机制层面的成因**：从attention-level证明模型在judgment时会分配更多注意力到自身回复，为偏见产生提供了可解释的底层机制线索。

## 方法详解
1. **DBG score 定义**：将裁判模型A对自身回复的评分 $S_A(r_A)$ 与其gold judgment $Q(r_A)$ 之差作为偏见度量，即 $\hat{w}_A = \mathbb{E}_x[\sigma(\delta + b_A) - \sigma(\delta)]$，其中 $\delta$ 为响应质量差距，$b_A$ 为裁判对自回复的偏见项；该公式通过泰勒展开可近似为 $\mathbb{E}[\sigma'(\delta)] \cdot \mathbb{E}[b_A]$，实现质量与偏见的线性解耦。
2. **Gold judgments 构建**：选用 GPT-4o-mini、Gemini-1.5-Flash、DeepSeek-V3 三个强LLM进行pairwise比较，取其投票共识作为"黄金标准"；通过人类标注对比验证其与人工判断的高一致性（约74%样本一致）。
3. **实验范式**：采用pairwise comparison方法，temperature固定为0；为缓解位置偏见，交换响应顺序并取平均概率；限制响应最大长度以控制长度偏见。
4. **风格迁移实验**：使用 DeepSeek-V3 将各模型回复统一改写为"attractive/humorous"风格，再通过t-SNE可视化验证改写后语义表征基本保持不变，确保风格变化不混扰内容质量。
5. **注意力分析**：计算裁判模型在各层对各自回复与其他模型回复的平均attention score，通过层间差异热力图展示偏见机制。

## 实验与结果
- **数据集**：AlpacaEval（helpfulness）、WMT19(de-en)（翻译）、TruthfulQA（truthfulness），各采样500条。
- **模型**：Llama-3.1-8B/70B、Llama-3.1-8B-Instruct/70B-Instruct、Qwen2.5-7B/72B-Instruct、gemma-2-9B-it、DeepSeek-R1-Distill-Qwen-32B、QwQ-32B；规模梯度实验涵盖 Qwen2.5-Instruct 0.5B–72B。
- **核心发现**：
  - 预训练与后训练模型均存在自我偏好偏见，但 **Llama-3.1-8B-Instruct DBG=0.2%，Llama-3.1-8B DBG=25.6%**，后训练反而更低。
  - **规模效应显著**：Llama-3.1-70B DBG=0.4%，远低于8B的21.6%；Qwen2.5-14B-Instruct DBG=2.1%，而0.5B-Instruct高达41.7%。
  - **推理模型（LRM）同样存在偏见**：DS-R1-Distill-Qwen-32B DBG=4.8%，QwQ-32B虽质量高但仍略有偏见，且不低于同规模语言模型。
  - **风格对齐可缓解偏见**：Llama-3.1-70B/8B 改写后 DBG 从 3.3%/18.7% 降至 1.4%/7.2%；Qwen2.5-72B/7B-Instruct 从 2.0%/8.4% 降至 1.2%/5.9%。
  - **一致训练数据可协同降低偏见**：Llama-3.1-8B-Instruct 与 Qwen2.5-7B-Instruct 经 UltraChat-200k 微调后 DBG 从 10.5%/2.1% 降至 2.1%/1.1%。
  - Gold judge 间 pairwise 一致性约 83–86%，验证共识有效性。
- **最强结果**：最大模型 Llama-3.1-70B 的 DBG=0.4%（几乎无偏见）；小模型 Qwen2.5-0.5B-Instruct DBG=41.7%（偏见最严重）。

## 相关工作脉络
1. **LLM-as-a-Judge 基础工作**：Zheng et al. (2023) MT-Bench、Zhu et al. (2023) Judgelm 开创LLM裁判范式，本文在其基础上深入探讨偏见问题。
2. **自我偏好偏见识别**：Panickssery et al. (2024) 首次指出LLM evaluator会识别并偏爱自身生成；Ye et al. (2024) 量化多种偏见（position/length/authority等）；本文与之互补，提出去质量混扰的度量方法。
3. **质量-偏见混淆的先前工作**：Chen et al. (2025) 指出直接用win rate衡量偏见存在问题，但仅聚焦于可验证任务（数学推理）；本文聚焦开放生成任务并提出DBG解耦方法。
4. **偏好泄漏（Preference Leakage）**：Li et al. (2025) 研究训练数据污染导致的偏见；本文从响应风格和训练数据一致性角度提供新的缓解路径。
5. **推理模型评估**：Jaech et al. (2024) OpenAI o1 系统卡、Guo et al. (2025) DeepSeek-R1 开启LRM时代；本文首次系统量化LRM作为judge时的自我偏好偏见。

## 局限性与未来方向
1. **Gold judge 能力受限**：仅使用 GPT-4o-mini 和 Gemini-1.5-Flash 等较小模型，未采用 GPT-4o 或 Gemini-1.5-Pro 等更强模型，可能影响gold标准的准确性。
2. **其他偏见未完全控制**：虽缓解了位置偏见和长度偏见，但权威偏见（authority bias）和情绪偏见（sentiment bias）仍可能影响结果。
3. **任务范围有限**：仅考察指令跟随和翻译任务，未涉及agent任务、对话任务等更复杂场景。
4. **缓解策略未彻底消除偏见**：风格对齐和一致训练数据只能减轻无法完全消除偏见，预训练阶段形成的生成-评估差异可能在下游持续存在。

## 研究启发与可借鉴点
1. **Gold consensus作为质量代理的思路**：可用多强模型投票共识替代人工标注，构建高质量基准，适用于多种"去偏评估"场景。
2. **Taylor展开解耦方法可迁移**：$\hat{w}_A \approx \mathbb{E}[\sigma'(\delta)] \cdot \mathbb{E}[b_A]$ 的质量-偏见线性分离框架，可推广至其他裁判偏差度量（如位置偏见、长度偏见）的解耦分析。
3. **风格统一作为低成本的偏见缓解工程策略**：在部署LLM裁判时，可通过prompt工程统一输入回复的叙事风格，以最小代价降低系统性偏差。
4. **注意力可视化揭示偏见机制**：层间attention difference热力图是揭示模型内部偏见成因的有效手段，可复用于其他裁判公平性研究。
5. **规模-偏见负相关关系**：提示团队在构建评估 pipeline 时，应优先选用大参数模型作为judge，或在资源受限时对小模型裁判分数进行偏见校准。

## 关键术语表
**Self-preference bias（自我偏好偏见）**：LLM作为裁判时倾向于给自身生成回复打更高分的系统性偏差。
**DBG score**：本文提出的偏见度量指标，等于裁判自评分与gold judgment之差，用于分离质量效应与偏见效应。
**Gold judgments（黄金判断）**：由多个强LLM共识聚合而成的参考评分，作为响应真实质量的代理。
**Bradley-Terry model**：用于配对比较的概率模型，计算裁判A偏好r_A胜过r_B的概率为 $\sigma(S_A(r_A) - S_A(r_B))$。
**In-context learning（上下文学习）**：向预训练模型输入少量示例以引导其完成特定任务（如judgment），无需额外微调。
**Pairwise comparison（成对比较）**：让judge在两个候选回复中择一优劣，相比单回复打分更稳定且不易受绝对分数尺度影响。
**Large Reasoning Model (LRM)**：经过强化学习等训练以增强推理能力的LLM（如DeepSeek-R1、QwQ），本文首次系统量化其裁判偏见。
**Preference leakage（偏好泄漏）**：训练数据中包含模型自身生成样本导致的评估污染问题。

## 可复现要素
- **数据集**：AlpacaEval（公开）、WMT19(de-en)（公开）、TruthfulQA（公开），各采样500条。
- **代码/权重**：代码与数据已开源，地址 https://github.com/zhiyuanc2001/self-preference。
- **关键超参**：temperature=0（确定性输出）；预训练模型使用2-shot in-context learning，后训练模型使用zero-shot；响应最大长度受限以控制长度偏见。
- **Gold judge模型**：GPT-4o-mini、Gemini-1.5-Flash、DeepSeek-V3。
- **微调数据**：UltraChat-200k（公开数据集）。
