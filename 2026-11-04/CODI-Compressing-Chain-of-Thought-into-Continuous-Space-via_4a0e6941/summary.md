---
title: "CODI-Compressing-Chain-of-Thought-into-Continuous-Space-via"
source: https://aclanthology.org/2025.emnlp-main.36.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:48:57"
field: "大语言模型推理效率"
keywords: ["Chain-of-Thought", "Implicit Reasoning", "Knowledge Distillation", "Continuous Representation", "Efficient LLM"]
innovations: ["单阶段自蒸馏框架将显式CoT压缩到连续空间，避免课程学习遗忘问题", "仅对齐单一答案prompt token的跨层隐藏激活实现知识迁移", "首个在GPT-2规模匹配CoT-SFT性能的隐式CoT方法，压缩率3.1×"]
benchmarks: ["GSM8k", "SVAMP", "GSM-Hard", "MultiArith", "CommonsenseQA"]
---

# 论文速读：CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation

## 一句话总结
CODI提出了一种自蒸馏训练框架，将显式链式思维（CoT）压缩到连续潜在空间实现隐式推理，是首个在 GPT-2 规模上性能媲美显式 CoT 的隐式 CoT 方法，达到 3.1× 压缩率并超越先前 SOTA（Coconut）28.2% 准确率。

## 研究问题与动机
1. 显式 CoT 依赖自然语言 token 逐标记推理，效率低且易过拟合表面语言线索而非推理本身。
2. 现有隐式 CoT 方法（如 Coconut）依赖多阶段课程学习逐步将 CoT token 替换为连续表示，过程中存在遗忘问题，性能始终大幅落后于 CoT-SFT。
3. 能否在单次训练中让隐式连续空间推理达到与显式 CoT 相当的性能，同时保持推理效率优势？
4. 连续空间推理缺乏监督信号，如何有效将显式推理知识迁移到连续表示中？

## 核心贡献（创新点）
1. **提出 CODI 单阶段自蒸馏框架**：联合训练教师（显式 CoT）与学生（隐式连续 CoT）两个任务，避免课程学习的遗忘问题；与 Coconut 的本质区别在于以蒸馏替代多阶段课程替换。
2. **特征空间蒸馏机制**：通过 L1 loss 对齐教师与学生模型在答案生成前一个 token（":"）处的隐藏激活（跨所有层），实现单向知识迁移；与以往基于离散 token 模仿的方法本质不同。
3. **首个在 GPT-2 规模匹配 CoT-SFT 性能的隐式 CoT 方法**：在 GSM8k 上达到 CoT-SFT 99% 准确率，超越 Coconut 20%+；与之前方法相比是质变而非渐进改进。
4. **展示隐式推理的可解释性与鲁棒性**：连续思考可解码为中间计算结果，且在 OOD 基准上优于 CoT-SFT，证明连续空间有助于减少过拟合。

## 方法详解
### 整体训练框架
CODI 采用双任务联合训练：教师任务学习显式 CoT 推理（交叉熵损失），学生任务从 learnable `<bot>` token 开始自回归生成连续思考向量，经 `<eot>` 后生成最终答案（交叉熵损失），并通过蒸馏损失将教师推理知识迁移到学生。总损失函数为：

$$\mathcal{L} = \alpha \mathcal{L}_{\mathrm{student}} + \beta \mathcal{L}_{\mathrm{KD}} + \gamma \mathcal{L}_{\mathrm{teacher}}$$

### 教师任务（Teacher Task）
使用标准交叉熵损失，输入为问题 Q 和标注的 CoT 推理序列 r=[c,y]（c 为推理步骤，y 为最终答案）：

$$\mathcal{L}_{\mathrm{teacher}} = -\frac{1}{N}\sum_{i=1}^{N} \log P(r_i | r_{1:i-1}, Q)$$

### 学生任务（Student Task）
从 `<bot>` token 出发，自回归生成 n 个连续思考 token（Z），每个思考 token 的 hidden state 经过两层 MLP + LayerNorm 投影后再输入下一步，最后由 `<eot>` token 生成答案：

$$\mathcal{L}_{\mathrm{student}} = -\frac{1}{N}\sum_{i=1}^{N} \log P(y_i | y_{1:i-1}, Q, Z)$$

### 自蒸馏机制（Self-Distillation）
基于 "CoT shift" 理论（附录 B 给出数学推导）：CoT 推理使答案 prompt token 的隐藏激活产生偏移，偏移量编码了推理信息。选择答案 prompt 中最后一个 token（":"）作为蒸馏 token，对其在所有层上的隐藏激活施加 L1 蒸馏损失：

$$\mathcal{L}_{\mathrm{KD}} = \frac{1}{M}\sum_{l=1}^{M} |\mathrm{sg}[\mathbf{h}_{\mathrm{teacher}}^{l}] - \mathbf{h}_{\mathrm{student}}^{l}|$$

其中 sg 为 stop-gradient 操作，确保知识单向流动（教师→学生）。训练时对每层隐藏激活按当前 batch 的标准差归一化以消除层间范数差异。

### 训练与推理细节
- 师生共用同一模型（自我蒸馏），无需额外预训练教师；
- 训练时剔除 CoT 最后一步（生成最终答案的步骤），防止捷径学习；
- 推理时人工插入 `<eot>` 终止连续推理，切换至语言生成模式输出答案。

## 实验与结果
**数据集**：GSM8k-Aug（385k，仅数学表达式）、GSM8k-Aug-NL（384k，含自然语言）、CommonsenseQA-CoT（8.1k）；OOD 评估：SVAMP、GSM-Hard、MultiArith。

**基线**：CoT-SFT、No-CoT-SFT、iCoT、Coconut。

**主要结果（GPT-2）**：
- GSM8k：CODI 43.7% vs CoT-SFT 41.8%（略超），超越 Coconut 34.1% +9.6pt，是首个性能匹配 CoT-SFT 的隐式 CoT 方法。
- GSM8k-Aug-NL：CODI 超越 CoT-SFT，压缩率达 8.2×。
- CommonsenseQA：CODI 超越 CoT-SFT，接近 No-CoT-SFT。
- OOD（SVAMP/GSM-Hard/MultiArith）：CODI 全面超越所有基线，SVAMP 42.9% > CoT-SFT 41.8%。

**主要结果（LLaMA-1b）**：
- GSM8k：CODI 达 CoT-SFT 的 90%（61.1% vs 66.7%），超越 Coconut 48.8% +12.3pt。
- OOD：全面超越 iCoT 和 Coconut。

**效率**：GSM8k-Aug 下 3.1× 压缩（6 个连续 token 替代平均 20.3 token CoT）；GSM8k-Aug-NL 下 8.2× 压缩。

**超参分析**：连续思考数=6 时性能最优；β 对 LLaMA-1b 需调大（默认 20）；γ 过大加速收敛但降低最终性能。

## 相关工作脉络
1. **Coconut（Hao et al., 2024）**：最先进隐式 CoT，采用课程学习逐步替换 token；本文认为其性能差距源于多阶段遗忘，以单阶段自蒸馏替代。
2. **iCoT（Deng et al., 2024）**：通过"逐步内化"将 CoT 模式注入隐状态，但不生成中间连续 token；CODI 扩展为可生成的连续推理。
3. **Pfau et al. (2024) / Goyal et al. (2024)**：从无到有预训练额外 thinking token；规模小、效果有限，CODI 基于已有预训练模型单阶段微调。
4. **CoT-SFT**：逐标记监督学习显式 CoT；CODI 以蒸馏对齐隐藏激活替代直接 token 模仿，减少过拟合。
5. **Geiping et al. (2025)**：循环 Transformer 的潜式推理（预训练视角）；CODI 为高效微调方案，面向已有 LLM。
6. **Ge et al. (2024) / Li et al. (2025)**：in-context 上下文压缩；本文压缩的是模型自身生成的推理过程而非输入上下文。

## 局限性与未来方向
1. 隐式 CoT 可解释性天然弱于显式 CoT；当前探测仅针对单个答案 prompt token，无法捕获多词元实体（如大数字 35649）。
2. 蒸馏 token 选择（":"）可能非最优；部分答案以 "-" 开头时移除该样本反而提升性能，说明关键推理信息可能存在于其他位置。
3. 长序列连续思考导致梯度回传距离长（6 步+），带来优化挑战， scaling 到更长推理链时问题更突出。
4. 实验仅在 GPT-2 和 LLaMA-1b 小规模模型上验证，未充分验证更大模型上的扩展性。
5. 答案 prompt 为人工设计（"The answer is:"），可能影响蒸馏效果。

## 研究启发与可借鉴点
1. **特征空间蒸馏替代逐标记监督**：仅需对齐一个关键 token 的隐藏激活即可实现知识迁移，大幅简化隐式推理训练目标设计，可迁移到其他需要压缩推理过程的任务。
2. **剔除 CoT 最后一步防捷径学习**：该技巧对任何蒸馏类训练均有参考价值，防止模型"抄近道"而非真正学习推理。
3. **压缩比与模型能力的内在平衡**：6 个连续 token 为最优，与数据集平均 CoT 步数相关，启发未来研究可设计自适应压缩长度机制。
4. **自蒸馏架构的通用性**：师生共用模型避免了额外预训练成本，该设计可推广至其他需要双任务协同学习的场景。
5. **可解释性分析框架**：通过将连续表示投影回词汇空间并分析注意力模式，为黑盒连续推理提供了可操作的探查手段。

## 关键术语表
**Chain-of-Thought (CoT)**：让 LLM 在输出最终答案前逐标记生成推理步骤的自然语言解释。
**Implicit CoT**：用连续向量替代自然语言 token 进行隐式推理，跳过显式语言生成以提升效率。
**Self-Distillation**：同一模型的不同任务/表示之间进行知识迁移，本文指教师任务（显式 CoT）向学生任务（隐式连续 CoT）传递推理知识。
**Distillation Token**：用于施加蒸馏损失的特定 token，本文为答案 prompt 末尾的 ":" token，其隐藏激活编码了推理总结信息。
**CoT Shift**：理论推导表明 CoT 推理对答案 token 的隐藏激活产生可量化的偏移量，是蒸馏损失有效的理论基础。
**Compression Ratio**：显式 CoT 平均 token 数与连续思考 token 数之比，本文达 3.1×（紧凑 CoT）至 8.2×（ verbose CoT）。
**Coconut**：先前 SOTA 隐式 CoT 方法，通过课程学习逐步将 CoT token 替换为连续表示。

## 可复现要素
- **数据集**：GSM8k-Aug（公开，基于 GSM8k 由 GPT-4 生成扩展至 385k）、GSM8k-Aug-NL（同）、CommonsenseQA-CoT（自行用 GPT-4o-mini 生成）；评估基准 GSM8k/SVAMP/GSM-Hard/MultiArith/CommonsenseQA 均为公开数据集。
- **代码**：已开源，https://github.com/zhenyi4/codi
- **权重**：论文未提供预训练权重下载链接，但给出了 PyTorch 实现示例（附录 Figure A1）。
- **关键超参**：α=1, β=1（GPT-2）/20（LLaMA-1b）, γ=1（GPT-2）/20（LLaMA-1b）；连续思考数=6；LoRA rank=128, alpha=32；GPT-2 learning rate=3e-3, 40 epochs；LLaMA-1b learning rate=8e-4, 10 epochs；batch size=128。
