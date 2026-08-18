---
title: "CODI-Compressing-Chain-of-Thought-into-Continuous-Space-via"
source: https://aclanthology.org/2025.emnlp-main.36.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:44:41"
field: "大模型高效推理与知识蒸馏"
keywords: ["Chain-of-Thought", "Implicit Reasoning", "Self-Distillation", "Continuous Space", "Model Compression", "Efficient Inference"]
innovations: ["提出单次自蒸馏框架，将显式CoT压缩至连续隐空间，避免课程学习遗忘", "创新性地使用单Token隐藏状态对齐实现高效的跨模态知识蒸馏", "在GPT-2规模上首次实现隐式CoT性能匹敌显式CoT，压缩率最高达8.2倍"]
benchmarks: ["GSM8k", "SVAMP", "GSM-Hard", "MultiArith", "CommonsenseQA"]
---

# 论文速读：CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation

## 一句话总结
本文提出 CODI（Continuous Chain-of-Thought via Self-Distillation），一种通过自蒸馏将显式链式思考（CoT）压缩到连续隐空间的训练框架。该方法首次在小规模模型（GPT-2）上实现了隐式 CoT 推理性能与显式 CoT 相当，同时提供了高效的计算压缩（最高 8.2 倍）和良好的分布外鲁棒性。

## 研究问题与动机
1. **显式 CoT 的效率与泛化瓶颈**：标准 CoT 推理依赖自然语言 token 序列，生成过程效率低，且模型易过拟合训练数据中的表层语言线索，导致泛化能力受限。
2. **现有隐式 CoT 方法的性能差距**：此前隐式 CoT 方法（如 Coconut）依赖课程学习逐步替换 token，存在严重的“课程遗忘”问题，使其性能显著落后于显式 CoT-SFT。
3. **核心科学问题**：能否设计一种高效的单次训练框架，使模型直接在连续隐空间中进行推理，并获得与显式 CoT 相媲美的推理能力？
4. **理论动机**：CoT 推理的本质是对查询 token 的隐藏状态产生“偏移”（shift）。若能隐式地复制这一偏移，即可在不依赖显式语言介质的情况下完成推理。

## 核心贡献（创新点）
1. **提出单次自蒸馏训练框架**：CODI 通过教师-学生任务联合训练，一次性将 CoT 推理能力蒸馏到连续隐空间，避免了 Coconut 等课程学习方法中的遗忘问题。
2. **单 Token 隐藏状态对齐机制**：创新性地发现仅需对齐答案提示前一个特定 token（如冒号“:”）的隐藏状态，即可将教师的显式推理知识高效转移给学生任务的连续 thought tokens。
3. **性能突破与效率提升**：在 GPT-2 规模上，CODI 是首个在 GSM8k 上达到显式 CoT 性能水平（99%）的隐式 CoT 方法，相比前作 SOTA 提升 28.2%，并实现了 2.7×至 8.2× 的推理压缩率。
4. **验证隐式推理的优越性**：实验表明，连续隐空间推理不仅避免了显式语言生成的过拟合，还在多个分布外（OOD）数学与常识推理基准上展现出更强的鲁棒性。

## 方法详解
CODI 采用教师-学生联合训练架构，共享同一组 LLM 权重，总损失函数为 $\mathcal{L} = \alpha \mathcal{L}_{\text{student}} + \beta \mathcal{L}_{\text{KD}} + \gamma \mathcal{L}_{\text{teacher}}$。

- **教师任务**：接收问题 $Q$ 和标注的显式 CoT 序列 $r = [c, y]$，通过教师强制（teacher forcing）计算交叉熵损失 $\mathcal{L}_{\text{teacher}}$，引导模型学习显式推理过程。
- **学生任务**：从可学习的 `<bot>` token 开始，自回归地生成 $n$ 个连续 thought tokens（$Z$），直至遇到 `<eot>` token 后生成最终答案 $y$。学生任务通过交叉熵损失 $\mathcal{L}_{\text{student}}$ 进行训练。连续 thought 的隐藏状态在进入下一步前，会经过一个两层 MLP 和层归一化投影。
- **自蒸馏（特征空间对齐）**：核心创新在于 $\mathcal{L}_{\text{KD}}$。基于“CoT 会引发查询 token 隐藏状态偏移”的理论观察，作者在教师和学生任务中分别提取答案提示中倒数第二个 token（如“:”）的隐藏状态 $\mathbf{h}_{\text{teacher}}^l$ 和 $\mathbf{h}_{\text{student}}^l$。通过 L1 损失对齐这些跨层隐藏状态，并在教师状态上施加 `stop-gradient`，实现单向知识转移：
$$\mathcal{L}_{\text{KD}} = \frac{1}{M} \sum_{l=1}^{M} | \text{sg}[\mathbf{h}_{\text{teacher}}^l] - \mathbf{h}_{\text{student}}^l |$$
- **训练细节**：连续 thought 在训练中动态生成，并利用 KV Cache 保证效率。对教师隐藏状态按批次内标准差进行归一化，以稳定不同层间的度量尺度。为杜绝捷径学习，训练数据中排除了最后一步直接生成答案的 CoT 步骤。

## 实验与结果
- **数据集与基准**：训练数据包括 GSM8k-Aug（结构化数学表达式，38.5万条）、GSM8k-Aug-NL（保留自然语言，38.4万条）和 CommonsenseQA-CoT（8,100条）。评估基准覆盖领域内（GSM8k）和分布外（SVAMP, GSM-Hard, MultiArith, CommonsenseQA）。
- **主要结果（GPT-2 模型）**：
    - **GSM8k**：CODI 准确率达 99.0%，几乎完全匹配 CoT-SFT（100%），远超 Coconut（70.8%）和 iCoT（34.0%），相对 Coconut 提升 **28.2%**。
    - **通用性（GSM8k-Aug-NL）**：CODI（97.1%）甚至超过了 CoT-SFT（93.8%），压缩比达 **8.2×**。
    - **常识推理（CommonsenseQA）**：CODI（82.4%）大幅超越 CoT-SFT（49.0%），证明隐式表示能避免对特定 CoT 模式的过拟合。
    - **效率**：相比紧凑 CoT 获得 **2.7×** 加速，相比冗长 CoT 获得 **5.9×** 加速。
- **主要结果（LLaMA-1b 模型）**：在 GSM8k 上达到 CoT-SFT 的 **90%**（66.7% vs 74.4%），显著提升 Coconut（48.8%）。
- **鲁棒性**：在所有三个 OOD 数学基准（SVAMP, GSM-Hard, MultiArith）上，CODI 均取得最佳或次佳成绩，验证了其泛化能力。
- **消融实验**：证实了自蒸馏损失（移除后性能暴跌）、共享教师/学生架构（独立静态教师失效）以及排除最后一步 CoT 的必要性。

## 相关工作脉络
1. **隐式 CoT 探索**：Pfau 等人（2024）和 Goyal 等人（2024）从理论/经验上验证了额外计算 token 对推理的促进作用，但未进行有效训练。CODI 首次在此方向上实现了媲美显式 CoT 的性能。
2. **CoT 内部化（iCoT）**：Deng 等人（2024）通过课程学习逐步内部化 CoT，但丢弃所有中间 token 导致性能瓶颈。CODI 通过保留隐式连续 thought 路径克服了这一限制。
3. **Coconut**：Hao 等人（2024）的 SOTA 隐式 CoT 方法，采用两阶段课程学习。CODI 通过单次自蒸馏替代了复杂的课程学习，从根本上避免了阶段性遗忘，性能全面超越 Coconut。
4. **知识蒸馏（KD）**：传统 KD 多用于跨模型（大教师-小学生的任务蒸馏）。CODI 将其应用于同一模型的不同推理模态（显式语言 vs. 隐式连续空间）之间的自我知识转移，是一种新颖的多任务自蒸馏范式。
5. **隐式推理与上下文压缩**：Ge 等人（2024）和 Li 等人（2025）的工作聚焦于压缩已有的长上下文。CODI 则致力于压缩模型自身需要**生成**的推理过程，二者目标不同。

## 局限性与未来方向
1. **可解释性局限**：当前通过词嵌入投影解读连续 thought 的方式仅能解码首个 token，对于由多个 token 组成的复杂实体（如大数字“35649”）无法完整还原，需要更高级的探测技术。
2. **蒸馏 Token 选择的次优性**：研究固定使用答案提示前的一个 token（如“:”）。但实际上，部分答案以特殊字符（如“-”）开头，且最终总结 CoT 的 token 也可能蕴含关键信息，当前设计存在盲区。
3. **长序列优化挑战**：在连续 thought 长度较长时（如超过 6 个 token），首个 thought token 的梯度需反向传播更多步，可能引入优化困难。
4. **规模扩展未验证**：受计算资源限制，实验仅在 GPT-2 和 LLaMA-1b 小规模模型上进行，其在更大参数模型（如 7B+）上的表现有待探索。

## 研究启发与可借鉴点
1. **自蒸馏框架的设计范式**：CODI 证明，通过为同一模型设计不同的输入/输出任务（教师：显式文本；学生：隐式连续表示），并用简单的特征对齐损失连接，可以有效转移推理能力。这一范式可迁移至其他需要“隐式化”的复杂推理任务。
2. **单点锚定对齐策略**：无需对齐整个序列，仅对齐答案生成前的一个关键位置的隐藏状态即可实现高效蒸馏。这一“稀疏对齐”思想极大降低了隐式推理的训练难度，值得在其他连续空间学习场景中借鉴。
3. **防止捷径学习的训练技巧**：从训练数据中刻意排除“最后一步直接导出答案”的样本，以防止教师模型走捷径，确保蒸馏出的隐藏状态真正编码了推理过程。这一技巧对任何希望学习复杂多步推理的蒸馏工作均有参考价值。
4. **效率与性能的平衡分析**：论文详细分析了连续 thought 数量（压缩比）与性能之间的关系，发现存在一个由数据集复杂度决定的最优值（本实验中为 6）。这提示我们在设计隐式推理系统时，需根据具体任务调整隐式表示的预算。

## 关键术语表
- **Chain-of-Thought (CoT)**：链式思考，指 LLM 在给出最终答案前，显式生成一系列自然语言推理步骤的推理范式。
- **Implicit CoT**：隐式 CoT，指模型不使用自然语言，而是在连续向量空间中进行中间推理的计算方式。
- **Self-Distillation (自蒸馏)**：在此文中特指在同一模型的不同任务分支（教师与学生）之间，通过损失函数对齐隐藏表示，以实现知识从一种推理模态到另一种的转移。
- **Distillation Token**：蒸馏 Token，指在教师和学生任务中被选中用于对齐隐藏状态的特定位置 token（如答案提示中的冒号）。
- **Compression Ratio**：压缩率，指显式 CoT 平均 token 数与隐式连续 thought token 数之比，衡量推理效率的提升倍数。
- **Out-of-Distribution (OOD)**：分布外，指模型在训练数据分布之外的测试集上进行评估，用于衡量模型的泛化和鲁棒性。
- **Curriculum Learning**：课程学习，一种训练策略，通过从简单到复杂的渐进式任务顺序来提升模型学习效果。Coconut 即采用此策略。
- **Stop-Gradient**：停止梯度，在反向传播时阻断梯度流过某张量，此处用于确保知识仅从教师流向学生，而非双向更新。

## 可复现要素
- **数据集**：GSM8k-Aug 和 GSM8k 测试集公开可用。GSM8k-Aug-NL 和 CommonsenseQA-CoT 的训练数据需根据论文附录描述自行生成或处理（使用 GPT-4/GPT-4o-mini 生成并过滤）。
- **代码**：已开源，地址为 https://github.com/zhenyi4/codi。
- **模型权重**：论文未提供预训练权重下载链接。
- **关键超参数**：
    - 总损失权重：GPT-2 模型 $\alpha=1, \beta=1, \gamma=1$；LLaMA-1b 模型 $\alpha=1, \beta=20, \gamma=20$。
    - 优化器：AdamW，有效 batch size 128， cosine scheduler，warmup 3%。
    - 微调方法：LoRA，rank=128，alpha=32。
    - 连续 thought 数量：默认 6 个。
    - 学习率：GPT-2 为 3e-3，LLaMA-1b 为 8e-4。
