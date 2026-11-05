---
title: "PBI-Attack-Prior-Guided-Bimodal-Interactive-Black-Box-Jailbr"
source: https://aclanthology.org/2025.emnlp-main.32.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 22:06:47"
field: "多模态大模型安全与对抗攻击"
keywords: ["越狱攻击", "大视觉语言模型", "黑盒攻击", "双模态对抗", "毒性最大化", "红队测试", "多模态安全"]
innovations: ["先验引导的黑盒双模态交互越狱攻击框架，替代LVLM提取有害特征嵌入图像作为优化起点", "双向跨模态贪心搜索交替优化图文扰动实现毒性最大化", "在6个目标模型（3白盒+3黑盒）上实现SOTA攻击成功率"]
benchmarks: ["AdvBench", "HEADS", "HarmBench"]
---

# 论文速读：PBI-Attack: Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for Toxicity Maximization

## 一句话总结
本文提出 PBI-Attack，一种基于先验引导的双模态交互式黑盒越狱攻击方法，通过替代 LVLM 从有害语料中提取恶意特征嵌入图像作为先验，再以贪心搜索交替优化图文扰动，实现毒性最大化，在六个目标模型上均超越现有 SOTA。

---

## 研究问题与动机
1. **黑盒场景下越狱攻击的缺失**：现有 LVLM 越狱方法要么需要白盒梯度访问（如 UMK、GCG），要么依赖人工 prompt engineering（如 Arondight、BAP），无法在实际黑盒部署场景中发挥作用。
2. **图文双模态交互利用不足**：多数攻击仅优化单模态（文本或图像）；部分双模态方法（如 Ying et al. 2024 的 BAP、Wang et al. 2024b 的 UMK）采用分离优化或单向交互，未能充分挖掘图像-文本的协同破坏效应。
3. **先验信息未被有效利用**：从随机扰动开始的优化效率低（Table 3 显示随机初始化比先验引导低约 20% ASR），缺乏对有害特征的定向注入机制。

---

## 核心贡献（创新点）
1. **提出先验引导的黑盒双模态越狱框架 PBI-Attack**：两阶段设计（先验生成 → 双模态交替优化），是首个基于目标模型响应进行迭代优化的黑盒 LVLM 越狱攻击方法。
2. **利用替代 LVLM 从有害语料提取恶意特征并嵌入图像作为先验**：用另一个白盒 LVLM（如 MiniGPT-4）将有害文本语义映射到图像特征空间，为后续优化提供高质量起点，与随机初始化本质不同。
3. **双向跨模态交互贪心搜索优化**：图文交替更新（贪心选取最优文本后缀/图像前缀），而非分离或单向优化，使双模态破坏特征相互放大。
4. **在毒性评分与 jailbreak 概率两种损失函数下验证**：Table 2 发现毒性评分指导的优化效果优于 jailbreak 概率，且无需白盒访问。

---

## 方法详解

### 整体流程（Algorithm 1）
两阶段优化，总迭代上限 $N$，达到毒性阈值 $T_{\text{toxicity}}$ 即成功。

### Stage 1：先验扰动生成（Prior Perturbation Generation）
- **输入**：良性图像 $\mathbf{x}_{\text{benign}}$、有害语料 $Y = \{\mathbf{y}_i\}_{i=1}^m$
- **目标**：生成携带恶意特征的对抗图像 $\mathbf{x}_{\text{adv}} = \mathbf{x}_{\text{benign}} \oplus \mathbf{x}_{\text{adv}}^p$
- **损失函数**（公式 2）：
  $$\mathcal{L}(\mathbf{x}_{\text{adv}}) = \sum_{i=1}^{m} -\mathrm{T}(\mathbf{x}_{\text{adv}}, \mathbf{y}_i) + \lambda \|h(\mathbf{x}_{\text{adv}}) - g(\mathbf{y}_i)\|$$
  其中 $\mathrm{T}(\cdot)$ 为毒性评分（Perspective API 或 Detoxify），$h(\cdot)$、$g(\cdot)$ 分别为图像和文本的特征提取函数（使用白盒替代模型如 MiniGPT-4/BLIP），$\lambda=1.0$ 平衡两项。
- **优化方式**：投影梯度下降（PGD）迭代更新扰动（公式 3）：
  $$\mathbf{x}_{\text{adv}}^p = h^{-1}\big(h(\mathbf{x}_{\text{adv}}^p) - \eta \nabla \mathcal{L}(\mathbf{x}_{\text{adv}})\big)$$
  图像扰动迭代 400 步直至收敛。

### Stage 2：双模态对抗优化循环（Bimodal Adversarial Optimization Loop）
- **初始化**：$\mathbf{x}_{\text{adv}}$（Stage 1 输出）、$\mathbf{y}_{\text{adv}} = \mathbf{y}_{\text{init}}$
- **交替贪心搜索**（每轮迭代）：
  1. **文本优化**（公式 4-5）：从预生成候选集 $Y^s$（400 个，每个 10 tokens）中贪心选取最大化毒性的后缀 $\mathbf{y}_{\text{new}}^s = \arg\max_{\mathbf{y} \in Y^s} \mathrm{T}(\mathbf{x}_{\text{adv}}, \mathbf{y}_{\text{adv}} || \mathbf{y})$，拼接更新。
  2. **图像优化**（公式 6-7）：随机生成 $K$ 个约束扰动 $X^p = \{\mathbf{x}_j^p\}$（$\|h(\mathbf{x}_j^p)\|_\infty \leq B$，$K=50$），贪心选取 $\mathbf{x}_{\text{new}}^p = \arg\max_{\mathbf{x} \in X^p} \mathrm{T}(\mathbf{x}_{\text{adv}} \oplus \mathbf{x}, \mathbf{y}_{\text{adv}})$，叠加更新。
- **终止条件**：毒性超过阈值，或达到最大迭代次数（文本优化 100 轮）。

### 关键设计细节
- **黑盒访问**：仅通过输入-输出对获取毒性评分，无需模型参数或梯度。
- ** surrogate 模型**：开源模型用自身作为替代模型；黑盒模型用 MiniGPT-4 提取特征。
- **毒性评估**：8 属性 Perspective API（toxicity, severe toxicity, identity attack, insult, profanity, threat, sexually explicit, flirtation）+ 6 属性 Detoxify，每对输入查询 10 次取均值以降低随机性。

---

## 实验与结果

### 数据集与模型
- **AdvBench**：520 条提示（主要实验）
- **HEADS**：5 种有害场景的文本指令（Appendix C）
- **白盒模型**：MiniGPT-4 (Vicuna-13B)、InstructBLIP (Vicuna-13B)、LLaVA (LLaMA-2-13B)
- **黑盒模型**：Gemini、GPT-4、Qwen-VL

### 基线方法（10 个）
Arondight、GCG、Advimage、ImgJP、UMK、InPieces、BAP、MLAI、FigStep、HADES

### 主要结果（Table 1，AdvBench，Perspective API）

| 模型 | 白盒 ASR | 黑盒 ASR | 最佳提升 |
|---|---|---|---|
| MiniGPT-4 | **94.9%** | — | +7.4pp vs UMK (87.5%) |
| InstructBLIP | **93.2%** | — | +9.8pp vs InPieces (81.8%) |
| LLaVA | **89.3%** | — | +6.0pp vs BAP (85.1%) |
| Gemini | — | **71.7%** | +15.5pp vs Arondight (56.2%) |
| GPT-4 | — | **63.2%** | +16.0pp vs HADES (47.2%) |
| Qwen-VL | — | **67.1%** | +15.6pp vs HADES (51.5%) |
| **平均** | **92.5%** | **67.3%** | — |

### 关键消融结果
- **先验 vs 随机初始化**（Table 3）：先验引导平均提升约 20pp ASR
- **毒性评分 vs jailbreak 概率**（Table 2）：毒性评分更优（MiniGPT-4: 94.9% vs 93.9%）
- **不同 surrogate 模型**（Tables 4-5）：不同替代模型生成的对抗图像在不同目标间具有一定迁移性（MiniGPT-4 作为 surrogate 对目标 MiniGPT-4 提升 67.0pp，对 InstructBLIP 提升 54.8pp）
- **Stage 1 单独 vs 完整**（Table 11）：Stage1 only ASR 78.2%/77.1%/53.8%，完整方法 94.7%/93.6%/67.4%，迭代优化贡献显著
- **Seed 图像鲁棒性**（Table 12）：6 种不同种子图像（含纯随机噪声）ASR 均在 93.8%-94.7%，初始图像选择不敏感
- **防御测试**（Tables 8-9）：面对扰动层防御和安全系统提示，PBI-Attack 仍显著优于基线（MiniGPT-4 上分别达 75.6%/45.8%）

### 效率分析（Table 6-7）
- 训练时间 27.9h（高于基线），但 20h 预算下 ASR 达 94.2%，展现更高性能上界。

---

## 相关工作脉络
1. **单模态越狱攻击**：GCG（Zou et al. 2023）生成对抗后缀绕过 LLM 安全对齐；UMK（Wang et al. 2024b）白盒双模态但单向交互，本文扩展至黑盒双向交替。
2. **视觉越狱攻击**：Advimage（Qi et al. 2024）仅用单张对抗图像；ImgJP（Niu et al. 2024）基于最大似然找图像提示；本文联合优化图文。
3. **多模态组合攻击**：InPieces（Shayegani et al. 2023a）将恶意文本触发器嵌入图像；BAP（Ying et al. 2024）分离优化双模态；FigStep（Gong et al. 2023）将禁词转为排版图像；HADES（Li et al. 2025b）嵌入恶意意图；本文的核心差异在于黑盒双向交互迭代。
4. **多轮/代理攻击**：Arondight（Liu et al. 2024c）RL 生成文本 + 随机组合图像；MrJ-Agent（Wang et al. 2024a）多轮对话心理策略；本文无需多轮对话，单次交互即可达成。
5. **VLAttack**（Yin et al. 2024）利用双模态交互但面向视觉问答任务而非越狱，本文将其思路迁移至毒性最大化场景。

---

## 局限性与未来方向
1. **时间成本高**：每轮需等待模型生成响应（数秒），数千次迭代耗时巨大（训练 27.9h），论文自认此为最大挑战。
2. **毒性评分依赖外部评估器**：优化目标基于 Perspective API/Detoxify 而非目标模型自身置信度，存在评估器与目标模型对齐偏差的可能。
3. **仅测试毒性最大化目标**：未验证对其他有害目标（如虚假信息、偏见放大）的泛化性。
4. **攻击可检测性未知**：对抗图像的视觉/语义扰动是否易被人类或检测模型发现，论文未深入讨论。

---

## 研究启发与可借鉴点
1. **"替代模型提取特征 → 先验初始化"的两阶段范式**：用白盒 surrogate 提取有害语义特征并编码到对抗样本中，可为其他黑盒对抗攻击提供通用初始化策略，减少随机搜索的盲目性。
2. **贪心候选池+交替优化的轻量双模态交互**：无需梯度即可实现图文双向增强，候选集大小（K=50, |Y^s|=400）与迭代轮数（9轮收敛）的消融结论可直接复用于其他多模态对抗场景。
3. **毒性评分优于 jailbreak 概率的发现**：在黑盒设置下，直接用毒性回归目标比二分类概率更稳定有效，这一洞见可迁移至 LLM 红队测试的收益建模设计。
4. **seed 图像不敏感性的实验结论**：对抗攻击的初始图像选择对最终性能影响很小，意味着可在大规模攻击中灵活选用任意种子，降低资源门槛。
5. **与防御的对抗实验设计**：同时测试扰动层防御和安全系统提示两类防御，为后续防御研究提供了基准对比框架。

---

## 关键术语表
**LVLM（Large Vision-Language Model）**：融合视觉和语言处理的大模型，如 LLaVA、MiniGPT-4，可同时理解图像和文本输入。
**Jailbreak Attack（越狱攻击）**：通过精心构造的输入诱导 AI 模型绕过安全对齐、生成有害内容的攻击方式。
**Black-Box Attack（黑盒攻击）**：攻击者仅能观察模型输入输出，无法访问内部参数或梯度的攻击场景。
**Toxicity Maximization（毒性最大化）**：以最大化模型输出毒性评分为优化目标，促使模型生成有害内容。
**Perspective API**：Google 开发的文本毒性评估工具，提供 8 个毒性属性评分（toxicity, threat, insult 等）。
**Dual-Modal Interaction Optimization（双模态交互优化）**：图像和文本扰动交替迭代更新，使双模态破坏特征相互放大的优化策略。
**Surrogate Model（替代模型）**：白盒可访问的模型，用于提取特征或估计梯度，以辅助攻击黑盒目标模型。
**ASR（Attack Success Rate）**：攻击成功比例，即生成有害响应的提示数占总测试提示数的百分比。

---

## 可复现要素
- **数据集**：AdvBench（520 prompts，公开），HEADS（公开）；代码已开源：https://github.com/Rosy0912/PBI-Attack
- **代码/权重**：代码开源；模型权重需自行获取（MiniGPT-4/Vicuna-13B、InstructBLIP/Vicuna-13B、LLaVA/LLaMA-2-13B 均为开源模型）
- **关键超参**：$\lambda=1.0$，学习率 $\eta=1$（步长），$K=50$（图像候选数），$|Y^s|=400$（文本候选数），每轮图像优化 400 步/文本优化 100 步，每对输入查询 10 次取均值，batch size=8，共 8×A100-80GB GPU
- **评估器**：Perspective API（8 属性）+ Detoxify（6 属性），阈值 0.5

---
