---
title: "Direct-Judgement-Preference-Optimization"
source: https://aclanthology.org/2025.emnlp-main.103.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:28:33"
field: "大语言模型评估与对齐"
keywords: ["LLM-as-judge", "Direct Preference Optimization", "DPO", "automatic evaluation", "generative judge", "reward modeling", "preference optimization"]
innovations: ["提出三种互补的DPO偏好对（CoT Critique、Standard Judgement、Response Deduction）训练通用judge", "构建680K大规模judge训练集并训练8B/12B/70B多尺度模型", "DPO+SFT混合损失函数强化正样本学习"]
benchmarks: ["RewardBench", "Auto-J", "InstruSum", "HHH", "BiGGen Bench", "FLASK", "MT-Bench", "FeedbackBench", "LLM-AggreFact", "InfoBench", "EvalBiasBench", "PreferenceBench", "AlpacaEval-2"]
---

# 论文速读：Direct-Judgement-Preference-Optimization

## 一句话总结
本文提出通过直接偏好优化（DPO）在大规模数据（680K样本）上训练通用型LLM-as-judge模型，设计了三种互补的偏好对类型（CoT批判、标准判决、响应推导），使训练出的SFR-Judge系列（8B/12B/70B）在13个评测基准上实现最优聚合性能，其中8B模型即可超越GPT-4o。

## 研究问题与动机
1. **现有judge模型泛化能力不足**：当前生成式judge多基于SFT在<100K小规模数据上训练，仅针对单一评估任务（如仅pairwise），无法适应多样化的评估场景。
2. **SFT训练的固有缺陷**：SFT仅让模型模仿正确示例，不接触错误示例；而DPO通过正负偏好对同时暴露模型优劣，是更稳定的训练范式。
3. **评判任务的多维度需求**：一个强大的judge需要同时具备：生成有意义批判（解释性）、做出准确判决（正确性）、理解响应本质（理解力）三种能力，但现有工作缺乏系统性设计。
4. **基础judge模型的潜力未被充分挖掘**：已有研究（如FLAMe）探索了基础judge，但仍依赖SFT；本文探索DPO在大规模judge训练中的优势。

## 核心贡献（创新点）
1. **提出三种互补的DPO偏好对类型**：CoT Critique、Standard Judgement、Response Deduction，分别针对解释生成、直接判决监督、响应理解三个维度。与已有工作（如仅用CoT数据的Self-taught Evaluator或Con-J）的本质区别在于多维度任务组合。
2. **构建680K大规模judge训练集并训练多尺度模型**：从多个数据集（包括人类标注和模型蒸馏）汇编多样化数据，训练8B/12B/70B三个规模的SFR-Judge模型族。与已有小规模SFT训练的judge（如Prometheus-2-7B）相比，规模与数据量均大幅提升。
3. **设计DPO+SFT混合损失函数**：在DPO偏好优化基础上引入SFT损失，强化正样本学习。与纯DPO方法（如标准DPO训练reward model）的区别在于同时利用"近似完美"的正样本进行强化监督。
4. **构建全面的13基准评测套件**：涵盖7个pairwise、4个single rating、2个classification基准，覆盖通用对话、安全、摘要、细粒度评估等多领域。与以往仅评测pairwise任务的工作相比，评估维度更全面。

## 方法详解
**整体框架**：以instruction-tuned LLM（如Llama-3.1-8B-Instruct）为起点，通过DPO+SFT混合损失在680K偏好对数据上训练生成式judge模型。

**三种偏好对构造**：
1. **Chain-of-Thought Critique ($\mathcal{D}_{\mathrm{CoT}}$)**：用教师模型$M_t$（Llama-3.1-70B-Instruct）对同一输入$x$采样20个候选输出$\{c, j\}$（temperature=0.7），根据$j$是否与ground-truth$j^*$匹配划分正负样本$y^w/y^l$。目标是教judge产生合理推理链并输出正确判决。
2. **Standard Judgement ($\mathcal{D}_{\mathrm{Std}}$)**：从$\mathcal{D}_{\mathrm{CoT}}$中移除CoT批判部分，仅保留最终判决$j$，修改协议$p$要求仅输出判决。解决CoT输出过长导致判决token监督信号被稀释的问题。
3. **Response Deduction ($\mathcal{D}_{\mathrm{Ded}}$)**：新颖的辅助任务。将评估过程反转——judge接收协议$p$、任务输入$i$和正确评估输出$y=\{c,j\}$，任务是推导原始响应$r$。负样本由较弱教师$M_t'$（Llama-3.1-8B-Instruct）生成，正样本为原始响应。

**损失函数**：
$$\mathcal{L}_{\mathrm{DPO+SFT}} = \mathcal{L}_{\mathrm{SFT}}(y_i^w|x_i) + \mathcal{L}_{\mathrm{DPO}}(y_i^w, y_i^l|x_i)$$
其中SFT项增强正样本 likelihood，DPO项通过$\beta$控制偏好优化强度，参考模型$M_{ref}$参数冻结。

**数据配比**：$\mathcal{D}_{\mathrm{CoT}}:\mathcal{D}_{\mathrm{Std}}:\mathcal{D}_{\mathrm{Ded}} = 70\%:15\%:15\%$，经过标签平衡处理。

## 实验与结果
**数据集与基线**：
- 训练数据：15+个数据集（LMSYS Chatbot Arena、HelpSteer、HH RLHF、BeaverTails、RAGTruth、Prometheus、UltraFeedback等），包含人类标注与模型蒸馏样本。
- 基线模型：Prometheus-2-7B/8x7B、Auto-J-13B、Llama-3-OffsetBias-8B、Themis-8B、Skywork-Critic-8B/70B、Con-J-7B、Self-taught-eval-70B、FLAMe-24B；闭源基线GPT-4o/GPT-4o-mini。

**主要结果**（聚合性能）：
- **Pairwise比较**（Table 1）：SFR-Judge-70B平均84.25分，超越Self-taught-eval-70B（82.26）近2个百分点；SFR-Judge-8B平均80.91分，**超越GPT-4o（76.78）**。
- **Single Rating**（Table 2）：SFR-Judge-70B在所有4个基准上与GPT-4o相当或更优；SFR-Judge-8B平均0.68，优于GPT-4o-mini（0.71）但略低于GPT-4o（0.75）。
- **Classification**（Table 3）：SFR-Judge-70B平均85.60，超越GPT-4o（85.47）；三规模模型均显著优于其他开源judge。

**批判质量**（Table 4）：SFR-Judge-70B的MetaF1得分为77.60，显著优于Self-taught-eval-70B（62.99）和Themis-8B（64.02），且Meta-Precision达93.10，表明批判事实性强。

**消融实验**：移除任意一种训练任务均导致性能下降，Standard Judgement对pairwise提升最大（图4）。

**下游应用**：
- 作为Reward Model：SFR-Judge-70B在AlpacaEval-2上优于PairRM和ArmoRM基线（图5）。
- 领域适配：在ContextualJudgeBench上，从SFR-Judge-8B微调比从Llama-3.1-8B-Instruct微调提升**10个绝对百分点**（55.6% vs 45.2%）。

## 相关工作脉络
1. **Prometheus系列（Kim et al., 2023, 2024）**：开创性地将LLM训练为generative judge，使用SFT在人类偏好数据上训练，仅支持single rating和pairwise任务，未探索DPO或大规模训练。
2. **Auto-J（Li et al., 2023）**：使用GPT-4蒸馏生成带解释的偏好数据，采用SFT训练；本文相比之下使用DPO并引入更多任务类型。
3. **FLAMe（Vu et al., 2024）**：探索基础judge的SFT训练，但数据规模和任务多样性不足；本文用DPO替代SFT，扩展至三种任务。
4. **Themis（Hu et al., 2024）**：单任务（single rating）DPO训练，修改了DPO损失以适应单点评分；本文的多任务DPO框架更具通用性。
5. **Self-taught Evaluator（Wang et al., 2024）**：迭代SFT+DPO自训练框架，仅做pairwise评估；本文一次性DPO训练支持三任务，且效率更高（无需多轮）。
6. **Con-J（Ye et al., 2024）**：使用CoT critique数据进行DPO训练，与本文方法最接近，但未引入Standard Judgement和Response Deduction两种额外偏好对类型。
7. **Skywork-Critic（Shiwen et al., 2024）**：专为pairwise设计的SFT训练judge；本文强调基础judge的多任务泛化能力。

## 局限性与未来方向
1. **依赖人工/模型标注数据**：训练数据需人类或高质量模型标注，随着LLM发展可能需要重新标注以"刷新"模型能力；可探索自举标注策略降低人工负担。
2. **未探索过程级奖励**：当前方法仅评估完整响应，过程级奖励（评估中间推理步骤）未涉及。
3. **推理延迟问题**：生成CoT批判需要额外推理时间，不适合时间敏感的reranking场景；Standard Judgement可跳过推理直接输出判决。
4. **仅限英语评估**：当前工作聚焦英语，多语言judge构建是重要未来方向。
5. **空引用偏差**：在EvalBiasBench的空引用（Empty Reference）类别上表现较弱（38.46% vs Llama-3-OffsetBias的92.31%）。

## 研究启发与可借鉴点
1. **多维度偏好对设计**：CoT Critique + Standard Judgement + Response Deduction的三元组合为训练复杂生成式模型提供了可复用的框架思路，可迁移至其他需要"解释+决策+理解"的任务。
2. **DPO+SFT混合损失**：在DPO基础上加入SFT损失强化正样本，适用于正样本接近"黄金标准"的场景，平衡了偏好优化与直接监督的优势。
3. **基础judge的持续微调价值**：SFR-Judge作为foundation model，经少量DPO微调即可适配特定领域（如RAG评估），为领域定制化judge提供了高效范式。
4. **Response Deduction任务的可迁移性**：反向推导任务（从评估结果反推原始内容）可启发其他需要"理解评估标准"的训练设计，如自动评分器、代码审查模型等。
5. **大规模judge训练的数据来源策略**：混合人类标注数据（如LMSYS、HH RLHF）与模型蒸馏数据（如Prometheus、UltraFeedback），兼顾质量与规模，可作为judge数据构建的参考。

## 关键术语表
**LLM-as-judge**：使用大语言模型作为自动评估器，对其它模型的输出进行评判和批评的范式。
**Direct Preference Optimization (DPO)**：一种通过偏好对直接优化语言模型的政策梯度方法，无需显式reward model，比RLHF更稳定高效。
**Chain-of-Thought Critique**：要求judge生成包含推理过程的批判性解释（CoT），而非仅输出判决结果，增强评估的可解释性。
**Response Deduction**：新颖的辅助训练任务，judge根据评估协议、输入和正确评估结果反推原始响应，强化对响应本质的理解。
**MetaCritique**：使用GPT-4生成的原子信息单元（AIUs）评估judge批判质量的方法，包含Meta-Precision（事实性）、Meta-Recall（完整性）和Meta-F1三个指标。
**EvalBiasBench**：评估judge模型六种偏见（长度、具体性、空引用、内容延续、嵌套指令、熟悉知识）的元评估基准。
**Reward Model (RM)**：为模型输出打分以用于RLHF训练的监督信号模型，本文证明生成式judge可替代传统分类式RM。
**Positional Bias**：judge对响应顺序的敏感性偏差，本文通过双次运行（交换顺序）测量并显示SFR-Judge一致性高（91.41%）。

## 可复现要素
- **训练数据**：混合数据集，包括LMSYS Chatbot Arena、HelpSteer、HelpSteer2、HH RLHF、BeaverTails、RAGTruth、PRM800K、CHAMP、Prometheus、OffsetBias、UltraFeedback、CodeUltraFeedback、COFFEE等；部分为公开数据集，部分需申请访问。论文未提及统一公开的训练集。
- **代码**：开源，GitHub链接 https://github.com/SalesforceAIResearch/sfrjudge
- **模型权重**：开源，提供SFR-Judge-8B、SFR-Judge-12B、SFR-Judge-70B三个规模。
- **关键超参**：教师模型Llama-3.1-70B-Instruct采样20个响应（temperature=0.7）；DPO损失中$\beta$参数用于控制偏好优化强度；训练3个epoch；数据配比70%:15%:15%。
