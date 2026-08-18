---
title: "Foot-In-The-Door-A-Multi-turn-Jailbreak-for-LLMs"
source: https://aclanthology.org/2025.emnlp-main.100.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:41:34"
field: "大语言模型安全与对齐"
keywords: ["jailbreak", "multi-turn attack", "LLM safety", "foot-in-the-door", "self-corruption", "prompt engineering"]
innovations: ["提出FITD多轮越狱框架，利用登门槛效应渐进侵蚀模型安全边界", "设计Re-Align与SlipperySlopeParaphrase双模块实现自适应对抗推进", "揭示LLM多轮自我腐败的语义漂移与注意力瘫痪机制"]
benchmarks: ["JailbreakBench", "HarmBench"]
---

# 论文速读：Foot-In-The-Door-A-Multi-turn-Jailbreak-for-LLMs

## 一句话总结
本文提出 FITD（Foot-In-The-Door），一种受心理学"登门槛效应"启发的多轮越狱攻击方法，通过渐进式递增恶意意图并借助 Re-Align 与 SlipperySlopeParaphrase 两个辅助模块消除模型拒绝，在七个主流 LLM 上实现平均 94% 的攻击成功率（ASR），远超现有单轮与多轮基线。

## 研究问题与动机
- 现有单轮越狱方法依赖精心构造的提示词或编码变换，难以稳定绕过日益增强的安全对齐策略。
- 多轮越狱（如 Crescendo、ActorAttack）虽能逐步引导模型，但高度依赖人工设计种子提示，且整体 ASR 仍受限。
- 心理学"登门槛效应"表明：一旦个体做出微小不道德承诺，后续更严重的违规行为更容易发生；该机制是否可迁移至 LLM 安全边界侵蚀尚待验证。
- 当前对齐机制在多轮交互场景下缺乏显式防护，存在"自我腐败"（self-corruption）风险，即模型在无外部对抗扰动下逐渐偏离初始安全行为。

## 核心贡献（创新点）
- **提出 FITD 多轮越狱框架**：首次系统利用"登门槛效应"设计渐进式恶意意图升级流程，区别于以往依赖固定模板或复杂 Agent 的多轮攻击。
- **引入 Re-Align 与 SlipperySlopeParaphrase 双辅助模块**：前者修正模型与查询不一致的响应以维持自我腐败进程，后者在拒绝发生时插入语义桥接提示平滑过渡，二者协同实现自适应对抗推进。
- **揭示 LLM 多轮自我腐败机制**：通过输入/输出对齐分析证明语义漂移与注意力瘫痪先于行为崩溃发生，为现有对齐策略提供可解释性脆弱证据。
- **在七个模型上取得 SOTA 攻击效果**：平均 ASR 达 94%，显著超越最强单轮方法（ReNeLLM，75%/61%）与最强多轮方法（ActorAttack，59%/55%）。

## 方法详解
FITD 的核心流程分为初始化序列生成与多轮交互两阶段：

- **getProgressionSequence**：给定恶意目标查询 $q^*$ 和序列长度 $n$，先生成无害起始提示 $q_1$，再通过 $k=3$ 次采样生成候选集合 $L=\{q_i^j\}$，最后按渐进性（progressiveness）与连贯性（coherence）原则选取 $n$ 个提示构成升级序列 $q_1, \ldots, q_n$。
- **主循环**：对每轮 $i$，将 $q_i$ 追加至对话历史 $\mathcal{H}$，获取响应 $r_i$；若响应非拒绝则继续，否则弹出 $q_i$ 并取出最后一组 $(q_{\text{last}}, r_{\text{last}})$。
- **分支策略**：
  - 若 $r_{\text{last}}$ 与 $q_{\text{last}}$ 不对齐（过于保守或部分拒绝），调用 **Re-Align**：由助手模型生成对齐提示 $p_{\text{align}}$，指出不一致性并引导模型重新生成更贴合查询意图的响应。
  - 若已对齐但仍被拒绝，调用 **SlipperySlopeParaphrase (SSP)**：由助手模型生成中间桥接查询 $q_{\text{mid}}$，其语义介于 $q_{\text{last}}$ 与 $q_i$ 之间；若仍被拒绝则反复改写 $q_{\text{mid}}$ 直至接受，最终将 $(q_{\text{mid}}, r_{\text{mid}})$ 加入历史。
- **自我腐败驱动**：通过模型自身响应作为下一步升级的基石，实现"以模型之矛攻模型之盾"的递归腐蚀过程。

关键公式：
- 输入语义分类：$ \mathrm{cls}(t_i) = \mathrm{Safe/Harmful/Neutral} $ 依据嵌入在安全/有害方向上的投影值确定。
- 安全边界度量：$ S_{\mathrm{bound}} = 1 - \frac{\Delta_{\mathrm{logit}} - \Delta_{\mathrm{min}}}{\Delta_{\mathrm{max}} - \Delta_{\mathrm{min}}} $，其中 $ \Delta_{\mathrm{logit}} = \mathrm{logit_{harm}} - \mathrm{logit_{safe}} $。
- 输出对齐综合得分：$ R_{\mathrm{align}}(p_i) = \frac{1}{3}(P_{\mathrm{ref}} + S_{\mathrm{bound}} + D_{\mathrm{resp}}) $。

## 实验与结果
- **数据集**：JailbreakBench（100 个有害查询）、HarmBench 验证集（80 个有害查询）。
- **评估指标**：Attack Success Rate（ASR），由 GPT-4o 评估响应有害性与查询对齐度。
- **基线对比**：
  - 单轮最强：ReNeLLM（75%/61%）；多轮最强：ActorAttack（59%/55%）。
  - **FITD**：平均 ASR 94%/91%，在 LLaMA-3-8B 上达 98%/93%，GPT-4o 达 93%/90%，GPT-4o-mini 达 95%/93%。
- **跨模型迁移性**：以 LLaMA-3.1-8B 为源模型时，对 Mistral-v0.2 达 76% ASR；以 GPT-4o-mini 为源时提升至 85%，表明强安全模型产生的历史更具通用性。
- **消融实验**：移除全部三组件导致 LLaMA-3.1 ASR 从 92% 降至 75%，LLaMA-3 从 98% 降至 59%；仅移除 Re-Align 影响较小（LLaMA-3 从 98% 降至 79%），表明 SSP 与 $p_{\text{align}}$ 贡献更关键。
- **防御效果**：LLaMA-Guard-3 最佳，但仍保留 78%-84% ASR；OpenAI-Moderation 仅小幅降低 3%-8%。
- **序列长度敏感性**：ASR 随 $n$ 增长而上升，在 $n=9$ 至 12 时趋于饱和；即使 $n=3$ 也可达到 ReNeLLM 级别性能。
- **提取策略**：Backward Extraction（保留后期查询）显著优于 Forward Extraction，证实最终阶段恶意提示的关键作用。

## 相关工作脉络
- **DeepInception、CodeChameleon、CodeAttack、ReNeLLM**：单轮越狱方法，依赖提示工程或编码变换，无法应对多轮上下文累积效应。
- **ActorAttack、Crescendo**：早期多轮越狱代表，依赖人工种子提示与复杂 Agent 设计，ASR 有限（59%/55% 等）。
- **CoA、Red Queen**：基于对话动态的多轮攻击，但缺乏对模型内部对齐退化的系统性分析。
- **LLaMA-Guard-2/3、OpenAI-Moderation**：当前主流防御方案，论文显示其在多轮渐进攻击下仍有显著漏洞。
- **定位差异**：FITD 不依赖固定模板或复杂 Agent，而是利用心理效应驱动模型自我腐败，同时提供输入/输出双重视角的可解释分析。

## 局限性与未来方向
- **自我腐败机制分析尚浅**：论文承认对 LLM 多轮退化过程的理解仍属初步，缺乏显式防护机制的设计与验证。
- **评估范围有限**：仅在文本模型与两个基准上验证，未扩展至视觉语言模型（VLM）或其他多模态场景。
- **助手模型依赖**：Re-Align 与 SSP 均需额外调用助手模型（默认 GPT-4o-mini），在实际部署中可能增加攻击成本与延迟。
- **未来方向**：开发实时自适应监控机制、设计多轮对话中的显式对齐保持策略、拓展至多模态越狱与防御研究。

## 研究启发与可借鉴点
- **渐进式意图升级框架可复用**：FITD 的"小步快跑+自我校准"范式可迁移至其他需绕过内容审核的场景（如红队测试、安全评估自动化）。
- **双分支纠错机制设计精巧**：Re-Align 与 SSP 分别处理"响应漂移"与"语义跳跃"两类拒绝原因，该分类思路可用于优化其他多轮攻击或交互系统。
- **内部表征分析可作为安全诊断工具**：语义漂移（0.15→0.62）与注意力瘫痪（有害注意力 0.30→0）的量化指标可为模型对齐检测提供新范式。
- **Backward Extraction 发现**：后期查询对攻击成功起主导作用，提示在构建安全评估集时应更关注近末段样本的恶意强度分布。
- **强模型历史更具迁移性**：GPT-4o-mini 作为源模型时迁移效果更好，这一现象可指导对抗样本生成策略——优先在高质量模型上构造攻击。

## 关键术语表
- **Foot-In-The-Door 效应**：心理学现象，指个体在答应小请求后更可能接受大请求，本文借喻模型对渐进恶意查询的服从性提升。
- **自我腐败（Self-Corruption）**：LLM 在多轮交互中逐渐偏离初始对齐行为、自发产生有害响应的退化过程。
- **SlipperySlopeParaphrase (SSP)**：FITD 的桥接模块，当模型拒绝时生成中间语义查询以平滑恶意意图升级路径。
- **Re-Align**：FITD 的对齐修正模块，通过注入对齐提示使模型响应与查询意图保持一致，防止过早拒绝中断攻击。
- **Attack Success Rate (ASR)**：越狱攻击成功比例，本文以 GPT-4o 评估响应有害性与对齐度作为判定标准。
- **Backward Extraction**：保留多轮对话后期查询而移除早期查询的攻击策略，实验证明其 ASR 显著高于 Forward Extraction。
- **语义漂移（Semantic Drift）**：输入提示中安全与有害 token 在表示空间中相似度随对话推进而升高的现象（0.15→0.62）。
- **注意力瘫痪（Attention Paralysis）**：模型对有害 token 的注意力权重从 0.30 骤降至接近零，预示安全机制即将崩溃。

## 可复现要素
- **数据集**：JailbreakBench（公开）、HarmBench（公开）。
- **代码**：已开源，地址 https://github.com/Jinxiaolong1129/Foot-inthe-door-Jailbreak。
- **模型**：LLaMA-3.1-8B-Instruct、LLaMA-3-8B-Instruct、Qwen2-7B-Instruct、Qwen-1.5-7B-Chat、Mistral-7B-Instruct-v0.2（开源）；GPT-4o-mini、GPT-4o-2024-08-06（闭源）。
- **关键超参**：序列长度 $n=12$（默认），候选采样数 $k=3$，助手模型默认 GPT-4o-mini。
- **推理设置**：开源模型使用 vLLM 默认参数，实验环境为 NVIDIA A100 GPU。
