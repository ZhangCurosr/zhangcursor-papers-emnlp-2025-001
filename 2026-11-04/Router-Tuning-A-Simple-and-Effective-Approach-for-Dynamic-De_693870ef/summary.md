---
title: "Router-Tuning-A-Simple-and-Effective-Approach-for-Dynamic-De"
source: https://aclanthology.org/2025.emnlp-main.99.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 16:57:13"
field: "大语言模型高效推理"
keywords: ["Dynamic Depth", "Mixture of Depths", "Router-Tuning", "Model Compression", "Efficient Inference", "Attention Skipping", "MoE Compression", "Parameter-Efficient Fine-Tuning"]
innovations: ["仅微调轻量级路由器（<0.01%参数）实现动态深度，训练成本降低三个数量级", "系统验证 Attention 层最适合动态跳过，同时节省计算与 KV Cache 内存", "将 Router-Tuning 扩展至 MoE 专家级别动态跳过，并与 LoRA 等 PEFT 方法兼容"]
benchmarks: ["LM-Harness (MMLU, GSM8K, HellaSwag, ARC-C 等)", "Llama-3-8B / Llama-2-13B / Mistral-7B / Qwen-2.5-14B / DeepSeek-MoE / OLMoE"]
---

# 论文速读：Router-Tuning: A Simple and Effective Approach for Dynamic Depth

## 一句话总结
论文提出 **Router‑Tuning**，仅微调一个轻量级路由器网络（参数量 < 0.01%），即可为预训练大语言模型注入动态深度（MoD）能力；在 Llama‑3‑8B 等模型上实现 **21% 推理加速**，性能仅下降 **0.2%**，且训练时间 **<15 分钟**（单卡 A6000），比现有方法快 **1000 倍**。

---

## 研究问题与动机
1. **训练成本过高**：现有 MoD 方法（如 Raposo et al., 2024; Tan et al., 2024）需从头预训练或长时间持续训练整个模型，计算开销巨大，难以在已有开源模型上快速部署。
2. **跳过重要层导致性能退化**：以往 MoD 多作用于整个 Transformer Block 或 MLP 层，而这些层包含关键计算单元，盲目跳过会显著损害模型能力。
3. **缺乏对路由目标的系统分析**：不同模块（Block、MLP、Attention、MoE Expert）和不同粒度（token‑level、sequence‑level）的动态跳过效果未知，尚未明确最优部署策略。
4. **未能充分利用 Attention 层的冗余性**：Attention 层计算复杂度为 O(L²) 且需要存储 KV Cache，具有更高跳过价值，但此前 MoD 工作很少针对该层设计。

---

## 核心贡献（创新点）
1. **提出 Router‑Tuning 框架**：仅训练一个单层投影路由器（参数量 < 0.01%），冻结骨干模型，将 MoD 训练成本从“全模型持续训练”降至“小数据快速微调”，本质区别在于**参数规模与训练阶段**（微调 vs 预训练/持续训练）。
2. **系统揭示 Attention 层最适合动态深度**：通过对比 Block/MLP/Attention 在不同粒度下的效果，发现跳过 Attention 层可在保持性能的同时大幅减少 KV Cache 内存与计算量，而跳过 Block/MLP 会导致显著性能下降。
3. **将 Router‑Tuning 扩展至 MoE 架构**：在 Expert 级别实现动态跳过，平衡各专家负载，缓解 MoE 中专家冗余问题，并与静态 Expert Drop 相比在性能与效率间取得更好权衡。
4. **证明与 LoRA 等 PEFT 方法天然兼容**：Router‑Tuning 与 LoRA 联合训练可同时获得效率提升与任务性能增益，为动态深度与参数高效微调的协同提供范式。
5. **提供完整的实验分析与开源实现**：在 Llama、Mistral、Qwen、DeepSeek‑MoE、OLMoE 等多个开源模型上验证，代码已公开，复现门槛极低。

---

## 方法详解
### 1. 路由器设计
给定输入 \(\pmb{x} \in \mathbb{R}^{L \times d}\)，路由器 \(R\) 计算重要性分数：
\[
R(\pmb{x})_i = 
\begin{cases}
\text{GATE}(\pmb{x}_i), & \text{Token‑level} \\
\text{GATE}\!\left(\frac{1}{L}\sum_{i=1}^L \pmb{x}_i\right), & \text{Sequence‑level}
\end{cases}
\]
其中 \(\text{GATE}(\cdot) = \text{Sigmoid}(W\cdot)\)，\(W\) 为待训练参数。

### 2. 二值化掩码与直通估计（STE）
根据阈值 \(\tau\) 生成掩码 \(M\)：
\[
M_i = 
\begin{cases}
1, & R(\pmb{x})_i \geq \tau \\
0, & \text{otherwise}
\end{cases}
\]
训练时使用 STE 使梯度可反向传播：\(\frac{\partial M}{\partial R}\approx 1\)。

### 3. 前向与推理公式
- **前向**：\(\pmb{y} = M \odot F(\pmb{x}) + \pmb{x}\)（\(F\) 为当前层，如 Attention 或 MLP）
- **推理**（跳过计算）：
\[
\pmb{y} = 
\begin{cases}
F(\pmb{x}) + \pmb{x}, & R(\pmb{x}) \geq \tau \\
\pmb{x}, & \text{otherwise}
\end{cases}
\]

### 4. 训练目标
\[
\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \cdot \mathcal{L}_{\text{MoD}}, \quad 
\mathcal{L}_{\text{MoD}} = \text{ReLU}\!\left(\|M\|_0 - s\right)
\]
- \(\mathcal{L}_{\text{task}}\)：标准交叉熵损失，保持任务性能。
- \(\mathcal{L}_{\text{MoD}}\)：\(l_0\)‑范数正则项，鼓励跳过比例达到目标稀疏度 \(s\)。
- \(\lambda\)：权衡系数，通过网格搜索确定。

### 5. 扩展至 MoE
对每个专家 \(E_i\) 应用相同路由器：
\[
\hat{E}_i(\pmb{x}) = 
\begin{cases}
E_i(\pmb{x}), & R(\pmb{x}) \geq \tau \\
0, & \text{otherwise}
\end{cases}
\]
最终输出：\(\pmb{y} = \sum_{i \in \mathcal{K}} G(\pmb{x})_i \cdot \hat{E}_i(\pmb{x})\)，其中 \(\mathcal{K}\) 为 Top‑K 专家索引。

### 6. 实现细节
- 路由器参数量每层仅 \(d \times 1\)，占模型总参数 < 0.01%。
- 初始化 \(W = 0\)，使初始评分 \(R(\pmb{x}) \geq 0.5 = \tau\)，即从“全激活（dense）”状态开始训练。
- 训练数据仅需 **5K** 条指令样本（Llama‑Pro），学习率 \(\{1\text{e-}5, 2\text{e-}5, 5\text{e-}5, 1\text{e-}4, 2\text{e-}4\}\)，\(\lambda \in \{0, 0.1, 0.01, 0.001\}\)。

---

## 实验与结果
### 1. 实验设置
- **模型**：Llama‑3‑8B、Llama‑2‑13B、Mistral‑7B、Qwen‑2.5‑14B/7B、DeepSeek‑MoE、OLMoE。
- **训练数据**：Llama‑Pro（涵盖通用指令、数学、代码）。
- **评估基准**：LM‑Harness（9 项任务，含 MMLU、GSM8K、HellaSwag、ARC‑C 等）。
- **硬件**：单卡 NVIDIA RTX A6000，序列最大长度 2048。

### 2. 主要结果（Llama‑3‑8B）
| 方法 | 粒度 | 加速比 | 平均准确率 | 关键提升 |
|------|------|--------|------------|----------|
| Baseline | – | 1.00× | 69.7 | – |
| Router‑Tuning (Attn, seq) | sequence | **1.21×** | **69.4** | 仅 **0.2%** 下降，加速 21% |
| Router‑Tuning (MLP, token) | token | 1.06× | 61.8 | 性能明显下降 |
| DLO (Tan et al., 2024) | token | – | 60.5 | 性能较差 |
| Skip Transformer | token | – | 60.8 | 性能较差 |

- **KV Cache 节省**：Llama‑3‑8B 序列长度 2048、batch size 64 时，减少 **8 GB** 显存（DLO 无法节省 KV Cache）。
- **训练时间**：**<15 分钟**（单卡 A6000），比 DLO（36 小时，A100）快 **1000 倍**。

### 3. 跨模型泛化
| 模型 | 加速比 | 平均准确率变化 |
|------|--------|----------------|
| Llama‑2‑13B | 1.22× | 68.0 → 67.9 (−0.1%) |
| Qwen‑2.5‑14B | 1.18× | 75.5 → 74.7 (−0.8%) |
| Mistral‑7B | 1.24× | 70.3 → 69.7 (−0.6%) |
| DeepSeek‑MoE | 1.10× | 63.7 → 62.4 (−1.3%) |
| OLMoE | 1.12× | 64.0 → 61.4 (−2.6%) |

- Router‑Tuning 在 **Attention 层** 普遍表现最佳，在 **Block/MLP 层** 均出现明显性能下降。
- **MoE 实验**：Router‑Tuning 在 DeepSeek‑MoE 上优于静态 Expert Drop（62.4 vs 61.1），且平衡了专家负载。

### 4. 与 LoRA 集成
- Llama‑3‑8B：LoRA + Router‑Tuning 获得 **1.21× 加速**，平均性能 **70.9**（高于单独 LoRA 的 71.0 与单独 Router‑Tuning 的 69.5）。
- 证明动态深度与参数高效微调可**协同增效**。

### 5. 消融实验
- **MoD 层数**：最多可覆盖一半 Attention 层（约 8 层）而不明显损害性能；超过则性能下降。
- **训练样本数**：仅需 **5K** 样本即可有效训练路由器；更多样本收益边际递减。
- **训练数据集**：Llama‑Pro 略优于 Alpaca/Evol‑Instruct/ShareGPT，但差异不大（证明路由器对数据分布鲁棒）。
- **阈值 \(\tau\)**：固定为 0.5（Sigmoid 中点），实验表明该设置稳定。

---

## 相关工作脉络
1. **静态层丢弃（Layer Drop）**：He et al. (2024b) 等方法固定移除某些层，缺乏输入适应性；Router‑Tuning 实现**动态、输入相关**的跳过，性能显著优于静态方案（如 GSM8K 上 +6.5%）。
2. **Mixture of Depths（MoD）**：Raposo et al. (2024) 从头预训练引入 MoD，计算成本极高；Router‑Tuning 仅微调路由器，**无需预训练**，可直接应用于现有开源模型。
3. **Dynamic Layer Operation（DLO）**：Tan et al. (2024) 对预训练模型进行长时间持续训练，耗时 36h（A100）；Router‑Tuning **<15 分钟**（A6000），效率提升三个数量级。
4. **Early‑Exit 方法**：通过预测 confidence 提前终止推理，但需额外分类头；Router‑Tuning 在**保留完整 Transformer 结构**的前提下动态跳过层，更灵活。
5. **Expert Drop（MoE 压缩）**：He et al. (2024a) 静态移除低重要性专家；Router‑Tuning 对 **每个 token 动态选择是否跳过专家**，保持所有专家潜力，负载更均衡。
6. **参数高效微调（PEFT）**：LoRA、Adapter 等仅更新少量参数；Router‑Tuning 进一步缩小至 **<0.01%**，且目标为动态深度而非任务适配，两者可结合（LoRA + Router‑Tuning）。

---

## 局限性与未来方向
1. **更复杂的训练策略未探索**：当前仅使用简单交叉熵 + \(l_0\) 正则，可能还有更先进的损失设计可进一步提升性能。
2. **实验规模有限**：仅验证了 Llama/Mistral/Qwen/DeepSeek‑MoE/OLMoE 等少数模型，未覆盖超大参数（>70B）模型及多模态架构。
3. **阈值固定为 0.5**：未探索自适应阈值或 per‑layer 阈值优化。
4. **未与更多 PEFT 方法结合**：如 QLoRA、DoRA、AdaLoRA 等，可能存在协同潜力。
5. **推理框架优化缺失**：当前加速基于软件逻辑跳过，未针对算子融合、内存布局等进行底层优化，实际部署加速比可能更高。

---

## 研究启发与可借鉴点
1. **轻量级路由器设计范式**：单层投影 + Sigmoid 门控 + STE 的训练技巧，可迁移至其他动态计算场景（如动态宽度、动态专家数）。
2. **Attention 层作为动态深度首选目标**：因其计算复杂度与 KV Cache 内存占用双重瓶颈，跳过 Attention 可获得“计算‑内存”双重收益，值得在更多架构中验证。
3. **极小训练数据即可生效**：5K 样本、<15 分钟训练，证明路由器学习能力极强，**低资源场景**下动态深度部署门槛大幅降低。
4. **与 LoRA 等 PEFT 天然兼容**：冻结骨干 + 训练路由器，与 LoRA 更新适配器不冲突，可**同步进行效率优化与任务适配**。
5. **MoD 容量控制公式**：\(\mathcal{L}_{\text{MoD}} = \text{ReLU}(\|M\|_0 - s)\) 提供精确的稀疏度控制，可推广至其他稀疏训练场景。

---

## 关键术语表
- **Mixture of Depths (MoD)**：根据输入复杂度动态决定跳过哪些层，以在计算效率与模型容量间取得平衡。
- **Router‑Tuning**：仅微调一个轻量级路由器网络（<0.01% 参数）来实现 MoD 的方法，骨干模型保持冻结。
- **Dynamic Depth**：同一模型对不同输入使用不同深度的推理策略，区别于静态剪枝或早期退出。
- **Straight‑Through Estimator (STE)**：使离散二值化选择（如跳过/不跳过）在训练时可微的技巧，梯度直接穿过取整操作。
- **KV Cache**：存储 Attention 层的历史键值向量以加速自回归生成，但占用大量显存；跳过 Attention 层可直接节省 KV Cache。
- **Mixture of Experts (MoE)**：使用多个专家网络并通过门控机制稀疏激活的模型架构，可大幅提升模型容量而不增加推理成本。
- **Parameter‑Efficient Fine‑Tuning (PEFT)**：仅微调少量参数（如 LoRA、Adapter）以适应下游任务，避免全参数微调的高成本。
- **Catastrophic Forgetting**：微调过程中模型遗忘原有知识的现象；冻结骨干可基本避免该问题。

---

## 可复现要素
- **代码与权重**：开源代码地址 https://github.com/CASE‑Lab‑UMD/Router‑Tuning；预训练模型权重使用公开版本（Llama、Mistral、Qwen、DeepSeek‑MoE、OLMoE）。
- **训练数据集**：Llama‑Pro（https://github.com/TIGER‑AIS/LlamaPro），包含约 5K 条通用指令样本。
- **评估基准**：LM‑Harness（EleutherAI），9 项零样本/少样本任务（见 Table 1）。
- **关键超参数**：
  - 阈值 \(\tau = 0.5\)
  - 路由器权重初始化 \(W = 0\)（保证初始为 dense 状态）
  - 学习率网格：\(\{1\text{e-}5, 2\text{e-}5, 5\text{e-}5, 1\text{e-}4, 2\text{e-}4\}\)
  - 稀疏度系数 \(\lambda\) 网格：\(\{0, 0.1, 0.01, 0.001\}\)
  - 目标稀疏度 \(s\)：通过学习率间接控制，实验中约 50% MoD 容量
- **硬件与环境**：单卡 NVIDIA RTX A6000，PyTorch + Hugging Face Transformers，训练时间 <15 分钟。

---
