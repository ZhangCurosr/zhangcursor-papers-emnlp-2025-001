---
title: "Think-Wider-Detect-Sharper-Reinforced-Reference-Coverage-for"
source: https://aclanthology.org/2025.emnlp-main.67.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:42:20"
field: "自然语言理解/文档级矛盾检测"
keywords: ["Document-level Self-Contradiction Detection", "Chain-of-Thought", "Reinforcement Learning", "GRPO", "Reward Design", "Reasoning Consistency", "LLM Alignment"]
innovations: ["首次针对 DSCD 任务中 CoT 推理不一致问题，提出基于参考覆盖率奖励的强化学习方法", "设计准确性、覆盖度与格式三维奖励函数，结合 GRPO 实现无中间标注的推理自优化", "构建全自动 DSCD 训练数据合成流水线，通过扰动操作与双阶段验证解决标注数据稀缺瓶颈"]
benchmarks: ["ContraDoc", "contraDialog"]
---

` + `<answer>...</answer>` tags).
   - Total reward guides policy update via GRPO.

   *Experiments & Results:*
   - **Dataset:** ContraDoc (benchmark), plus generated training data. Also tested on contraDialog (transfer).
   - **Baselines:** Zero-Shot, CoT, SFT (without RL). Models: Llama-3.1-8B-Instruct, Qwen-2.5-7B-Instruct.
   - **Metrics:** Precision, Recall, F1, Accuracy, EHR, Consistency, Reliability (F1 × Consistency).
   - **Main Results (Table 1 & 2):**
     - Llama-3.1-8B on Judge then Find: Accuracy 38.5% (CoT) -> 51.1% (+12.6%), F1 0.415 -> 0.482, Consistency 59.6% -> 76.2% (+16.6%), Reliability 0.247 -> 0.367.
     - Qwen-2.5-7B on Judge then Find: Accuracy 43.9% -> 51.1%, Consistency 69.4% -> 74.5%.
     - Outperforms Zero-Shot and CoT significantly. Also beats simple "Cover Prompt".
     - Generalization: Strong performance on contraDialog (sentence-level).
   - **Ablation (Table 3):** Full method (all 3 rewards) achieves best across F1, Acc, EHR, Cons, Rel, Cov. Coverage rate jumps from ~25% (SFT) to ~85% (Ours). Output length increases moderately but information density improves.
   - **Data Efficiency:** RL enhancement outperforms simply adding more distilled SFT data (SFT_Plus underperformed).

   *Related Work Context:*
   - Contradiction Detection: Traditional NLI (pairwise) vs. document-level (DSCD, ContraDoc). This paper shifts focus to document-level reliability/consistency.
   - RL for LLM Reasoning: RLHF, DPO, GRPO. Applied primarily to math/code/games; this paper extends GRPO to document-level semantic reasoning (DSCD), addressing a gap.
   - CoT Reasoning: Often improves accuracy but reduces consistency. Previous work doesn't address consistency in DSCD; this paper explicitly targets it via coverage reward.
   - Data Generation for NLU: Manual annotation heavy; this paper proposes automated LLM-driven synthesis pipeline.

   *Limitations & Future Directions:*
   - Balance between comprehensive reasoning and conciseness remains challenging.
   - Hardware constraints limited evaluation to ~8B parameter models; larger models not fully tested for fine-tuning.
   - Reliance on single benchmark (ContraDoc) due to lack of public DSCD datasets with location annotations.
   - Current approach is text-only; multimodal DSCD (images, tables) is future work.

   *Research Inspiration & Transferable Value:*
   - The idea of using a "coverage" reward to combat reasoning inconsistency in CoT can be transferred to other complex reasoning tasks (e.g., long-form QA, legal/document analysis).
   - The automated data generation pipeline (operation-based perturbation + LLM verification + fluency filtering) is a reusable template for constructing domain-specific contradiction/conflict datasets.
   - Multi-dimensional reward design (accuracy + coverage/completeness + format) in RL provides a blueprint for aligning LLM reasoning traces without expensive human labels.
   - Demonstrates that RL can improve data efficiency over mere SFT scaling, valuable for resource-constrained research.

   *Key Terms:*
   - DSCD (Document-level Self-Contradiction Detection): 检测多句文档中是否存在自相矛盾内容的任务。
   - CoT (Chain-of-Thought): 通过逐步推理提升模型性能的方法，但在此任务中易导致答案不一致。
   - GRPO (Group Relative Policy Optimization): 移除价值函数的强化学习算法，适合无中间标注数据的自监督优化。
   - Reference Coverage Reward: 衡量推理链覆盖输入文档比例的新颖奖励函数，用于提升推理全面性。
   - EHR (Evidence Hit Rate): 模型判断为矛盾时，正确定位证据句的比例。
   - Consistency & Reliability: 一致性指多次推理结果相同的比例；可靠性为F1与一致性的乘积，综合衡量准确度与稳定性。

   *Reproducibility Factors:*
   - Datasets: ContraDoc (benchmark), StorySumm, REPLIQA (training data generation sources). Training data is synthesized, not publicly stated as released.
   - Code/Weights: Not explicitly mentioned in the text (typically would be in a GitHub link in the actual paper, but here it's not provided; I will state "论文未提及").
   - Key Hyperparameters: 8× NVIDIA L20 GPUs; SFT: lr=1e-4, epochs=3, LoRA rank=8, alpha=32, dropout=0.1; RL: lr=5e-5, epochs=1, micro-batch=4, grad_accum=2; max seq len=4096 (prompt 8192, completion 4096); temperature=0.7, top-p=0.9, max_new_tokens=4096 for eval.

   *JSON Metadata:*
   - keywords: ["Document-level Self-Contradiction Detection", "Chain-of-Thought", "Reinforcement Learning", "GRPO", "Reward Design", "Reasoning Consistency", "LLM Alignment"]
   - field: "自然语言理解/文档级矛盾检测"
   - innovations: ["首次针对文档级矛盾检测（DSCD）中的CoT推理不一致问题，提出基于参考覆盖率奖励的强化学习方法", "设计包含准确性、覆盖度与格式维度的任务特定奖励函数，结合GRPO实现无需中间标注的自进化训练", "构建全自动DSCD训练数据合成流水线，通过操作扰动与LLM验证解决高质量标注数据稀缺问题"]
   - benchmarks: ["ContraDoc", "contraDialog"]

4.  **Draft the Note (Mental Refinement in Chinese):**
   I will carefully translate and structure the extracted points into the exact requested format, ensuring academic tone and fidelity to the paper. I'll double-check numbers and names.

   *Self-Correction/Verification during drafting:*
   - Check accuracy numbers: Abstract says 38.5% to 51.1% (accuracy jump of 12.6%). Table 1 confirms. Consistency 59.6% to 76.2%. Matches.
   - Check dataset names: ContraDoc, StorySumm, REPLIQA, contraDialog. All correct.
   - Check method: SFT + GRPO RL. Three rewards: Accuracy, Cover, Format. Correct.
   - Check limitations: Text-only, 8B limit, single benchmark, conciseness vs comprehensiveness trade-off. Correct.
   - Format: Strictly follow headings. No extra fluff.
   - JSON: Must be valid and at the very end.

   All looks solid. I will now generate the output. 
   Structure check:
   # 论文速读：...
   ## 一句话总结
   ## 研究问题与动机
   ## 核心贡献（创新点）
   ## 方法详解
   ## 实验与结果
   ## 相关工作脉络
   ## 局限性与未来方向
   ## 研究启发与可借鉴点
   ## 关键术语表
   ## 可复现要素
   <!--META ... -->
   Matches exactly. Proceed. 
   Output generation. 
   *(Self-Correction/Note during thought)*: I need to make sure the JSON block is strictly at the end on a new line, and the content inside matches the required keys. I'll format it carefully. The paper mentions code/data availability? "To the best of our knowledge..." etc. It doesn't explicitly state a GitHub repo or data release in the provided text. I'll write "论文未提及" for code/weights release status as instructed. I'll also note that the training data pipeline is described but not confirmed as open-sourced.

   Let's draft. 
   [Output Generation] -> *Proceeds*
</think>

# 论文速读：Think-Wider-Detect-Sharper-Reinforced-Reference-Coverage-for

## 一句话总结
本文针对文档级自矛盾检测（DSCD）任务中 Chain-of-Thought (CoT) 推理易出现答案不一致的问题，提出一种结合监督微调（SFT）与强化学习（RL）的两阶段训练框架；通过设计包含“参考覆盖度”的任务特定奖励函数，引导小参数 LLM 在推理链中全面覆盖所有潜在矛盾句，从而在 ContraDoc 基准上将 Llama 3.1-8B 的准确率与一致性分别提升 12.6% 与 16.6%。

## 研究问题与动机
- **DSCD 任务重要性**：文档级自矛盾检测需评估长文本全局语义一致性，传统成对 NLI 方法计算复杂度高达 $O(n^2)$ 且难以捕捉跨多句的全局冲突。
- **CoT 推理的“ accuracy-consistency 权衡”困境**：现有 CoT 提示虽能提升 DSCD 准确率，但多次独立推理时模型关注点漂移（遗漏或切换不同矛盾句），导致相同输入产生不一致输出，削弱模型可靠性。
- **高质量标注数据稀缺**：DSCD 依赖人工逐句标注，成本高昂；缺乏大规模、自动化的高质量推理数据制约模型训练。
- **核心科学问题**：能否通过强制推理链覆盖全部潜在矛盾句，在提升准确率的同时同步改善响应一致性？

## 核心贡献（创新点）
1. **首次关注并缓解 DSCD 任务中 CoT 的一致性缺陷**：与以往仅追求准确率提升的工作不同，本文明确将“推理稳定性”纳入优化目标，利用基于参考覆盖率的强化学习弥合 accuracy 与 consistency 的鸿沟。
2. **提出全自动 DSCD 训练数据合成流水线**：基于 StorySumm 与 REPLIQA 构建注入型矛盾样本，通过插入/删除/替换/交换四种扰动操作结合双阶段验证（困惑度过滤 + LLM 矛盾核验），解决人工标注瓶颈。
3. **设计三维任务特定奖励函数并引入 GRPO 进行 RL 优化**：准确性奖励、参考覆盖度奖励与格式奖励协同作用，在不依赖中间标注的前提下实现推理链的自迭代完善。
4. **验证跨层级泛化能力**：在文档级 ContraDoc 与句子级 contraDialog 上均取得显著超越，证明所提方法的可迁移性。

## 方法详解
- **两阶段训练框架**：第一阶段基于强教师模型（DeepSeek-R1）进行 CoT 数据蒸馏与 SFT；第二阶段采用 GRPO 算法进行强化学习优化。
- **训练数据构建**：以 StorySumm（32 篇短篇故事）和 REPLIQA（17 个主题域）为语料源，定义 Attitudinal、Definition、Logical、Factual、Scope、Temporal 六类矛盾。通过 Prompt 引导 LLM 执行 Insert/Delete/Replace/Swap 四种操作生成修改文本 $\hat{d}_i$ 及矛盾句集合 $\mathcal{S}$。验证阶段：(1) 流畅度过滤：要求修改文本与原篇困惑度比值 $\frac{ppl(\hat{d}_i)}{ppl(d)} \leq 1.01$；(2) 矛盾核验：利用 LLM 判读是否含有效矛盾。最终生成 2,754 正样本与 4,276 负样本。
- **SFT 阶段**：仅保留 DeepSeek-R1 最终预测与 Ground Truth 一致的 CoT 响应作为蒸馏信号（$\mathcal{D}_{\text{filtered}} = \{(x, y_t) \mid a_t = a^*\}$），使用 LoRA（rank=8, $\alpha$=32）微调基座模型。
- **RL 阶段（GRPO + 多维权重奖励）**：
  - $R_{\text{accuracy}}$：正样本奖励 = $j \cdot [-(1-e) + (1+m/n)\cdot e]$，负样本奖励 = $j$（$j$ 为判断正确指示，$e$ 为证据命中指示，$m/n$ 为命中文本覆盖比）。兼顾判断准确率与证据召回，惩罚无证据的正确判断。
  - $R_{\text{cover}}$：$\frac{|S_{\text{covered}}|}{|S_{\text{total}}|}$，通过提取推理链中形如 `[i]`、`[i-j]` 的句子跨度引用，量化模型对输入文档的全面覆盖程度。
  - $R_{\text{format}}$：二元奖励，强制输出严格遵循 `</think>` 包裹推理过程、`<answer>` 包裹结论的结构，防止格式遗忘。
  - 总奖励 $R = R_{\text{accuracy}} + R_{\text{cover}} + R_{\text{format}}$，驱动 GRPO 策略梯度更新。

## 实验与结果
- **评测基准**：ContraDoc（文档级二元判断 + 定位）、contraDialog（句子级）。
- **基线模型**：Llama-3.1-8B-Instruct、Qwen-2.5-7B-Instruct；对比策略包括 Zero-Shot、CoT、SFT。
- **主要结果（Judge then Find 任务）**：
  - Llama-3.1-8B：Accuracy 从 CoT 的 38.5% 提升至 51.1%（+12.6%），F1 从 0.415 升至 0.482，EHR 从 0.616 升至 0.661；Consistency 从 59.6% 跃升至 76.2%（+16.6%），Reliability 从 0.247 升至 0.367。
  - Qwen-2.5-7B：Accuracy 从 43.9% 升至 51.1%，Consistency 从 69.4% 升至 74.5%。
- **消融实验**：完整奖励组合在各指标上全面最优；仅加 $R_{\text{cover}}$ 可使推理句子覆盖率从约 24.5% 提升至 84.9%，输出长度仅增加 1.72 倍，信息密度显著改善。
- **数据效率对比**：单纯扩充 SFT 数据（SFT_Plus）效果未升反降，而 RL 覆盖优化实现更高效的数据利用。
- **泛化实验**：在 contraDialog 上 Zero-Shot F1 仅 0.329，CoT 为 0.592，本文方法达 0.797，验证跨层级迁移能力。

## 相关工作脉络
1. **传统 NLI 矛盾检测**（Lendvai et al., 2016; Badache et al., 2018）：聚焦句子对级别，计算复杂且无法捕获文档全局冲突；本文从 pairwise 转向 document-level。
2. **文档级自矛盾检测**（Hsu et al., 2021; Li et al., 2024 ContraDoc）：首次将 DSCD 形式化为二分类与证据定位任务；本文在此基础上探索 LLM 推理的稳定性问题。
3. **CoT 推理优化**（Wei et al., 2022）：普遍关注准确率提升；本文揭示其在 DSCD 中引发一致性下降的隐性代价，并给出针对性修复路径。
4. **RL 对齐大模型**（RLHF, DPO, GRPO）：主要应用于数学、代码、游戏领域；本文是首次将 GRPO 扩展至文档级语义推理任务，填补了长期推理场景的空白。
5. **自动化 NLU 数据构造**：传统依赖人工；本文提出基于扰动操作+LLM验证的闭环合成管线，为同类结构化推理数据构建提供参考范式。
6. **证据定位评估**（EHR, BERTScore）：本文引入 EHR 与新的 Reliability（F1×Consistency）指标，使评估维度从单一准确率扩展至“准确-稳定”联合度量。

## 局限性与未来方向
- **推理全面性与简洁性的平衡**：覆盖度奖励虽提升完整性，但仍可能在极长文档中引入冗余信息，需探索更精细的注意力引导机制。
- **模型规模限制**：受硬件约束，实验仅局限于 8B 参数模型，未充分验证更大尺度基座的微调潜力。
- **数据集单一**：缺乏公开的多源文档级矛盾定位数据集，所有实验仅在 ContraDoc 上进行，外部泛化边界待进一步探测。
- **模态局限**：当前方法仅限纯文本矛盾；涉及图像、表格等多模态文档的全局矛盾检测是明确的未来方向。

## 研究启发与可借鉴点
- **一致性作为显式优化目标**：将多次推理结果的稳定性纳入奖励设计，为 CoT 类方法的可靠性评估与提升提供了新思路，可迁移至法律/医疗等高风险文本分析场景。
- **多维度无标注奖励组合范式**：Accuracy + Coverage + Format 的解耦设计展示了如何在缺乏中间人工标注的情况下，通过规则化信号实现推理轨迹的精细化对齐。
- **自动化数据合成管线可复用**：扰动操作（Insert/Delete/Replace/Swap）+ 流畅度阈值过滤 + LLM 二次核验的流水线，适用于其他需要构造矛盾/冲突/反事实样本的 NLU 子领域。
- **RL 优于单纯数据堆叠**：实验证明强化学习带来的推理模式改变比简单增加 SFT 样本更有效，为资源受限团队的训练策略选择提供了实证依据。

## 关键术语表
- **DSCD (Document-level Self-Contradiction Detection)**：文档级自矛盾检测，判断多句文本内部是否存在冲突并定位矛盾证据的任务。
- **CoT (Chain-of-Thought)**：思维链提示策略，引导模型分步推理；在 DSCD 中易因关注点漂移导致结果不一致。
- **GRPO (Group Relative Policy Optimization)**：移除价值网络的优势估计强化学习算法，适合无中间标注数据的策略自演化。
- **Reference Coverage Reward ($R_{\text{cover}}$)**：衡量推理链显式引用输入文档句子比例的新颖奖励，旨在提升推理全面性。
- **EHR (Evidence Hit Rate)**：在模型预测存在矛盾的前提下，成功定位到正确证据句的样本比例。
- **Consistency**：同一模型对相同输入进行 $T$ 次独立推理时，预测结果保持一致的平均概率。
- **Reliability**：新提出的综合指标，定义为 $\mathcal{R} = \text{F1} \times \text{Consistency}$，联合刻画准确率与稳定性。
- **Fluency Filtering**：通过比较修改文本与原作文本困惑度比值（阈值 1.01）过滤语义断裂的生成样本。

## 可复现要素
- **数据集**：ContraDoc（评测基准）；StorySumm 与 REPLIQA（训练数据合成源）；contraDialog（泛化评测）；合成训练集 2,754 正/4,276 负样本。
- **代码/权重**：论文未提及开源链接或模型权重发布情况。
- **关键超参**：8× NVIDIA L20 GPU；SFT：lr=1e-4，epochs=3，LoRA rank=8，$\alpha$=32，dropout=0.1，max_seq_len=4096；RL：lr=5e-5，epochs=1，micro_batch=4，grad_accum=2，prompt_len=8192，completion_len=4096；评估：temperature=0.7，top-p=0.9，max_new_tokens=4096。
