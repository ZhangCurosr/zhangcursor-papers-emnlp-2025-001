---
title: "PBI-Attack-Prior-Guided-Bimodal-Interactive-Black-Box-Jailbr"
source: https://aclanthology.org/2025.emnlp-main.32.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:32:22"
field: "多模态大模型安全与越狱攻击"
keywords: ["jailbreak attack", "large vision-language models", "black-box attack", "adversarial attacks", "multimodal security", "toxicity maximization", "red teaming"]
innovations: ["先验引导：用替代LVLM从有害语料提取恶意特征嵌入图像作为黑盒攻击先验", "双向交互优化：交替贪婪搜索优化图像扰动和文本后缀实现双模态联合越狱", "黑盒毒性最大化：仅需黑盒查询和toxicity score即可驱动高效越狱，无需模型梯度"]
benchmarks: ["AdvBench", "HADES"]
---

# 论文速读：PBI-Attack: Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for Toxicity Maximization

## 一句话总结
本文提出了 PBI-Attack，一种**先验引导的双模态交互式黑盒越狱攻击方法**，通过替代 LVLM 从有害语料中提取恶意特征并嵌入良性图像作为先验信息，再经由贪婪搜索交替优化图像和文本扰动，最终在黑盒场景下实现对 LVLM 的高效毒性最大化攻击，平均攻击成功率在开源模型上达 92.5%，在闭源模型上约 67.3%。

## 研究问题与动机
1. **现有越狱方法依赖人工提示工程**：多数方法基于人类知识设计 prompt，受限于攻击者经验，难以在**黑盒场景**下有效越狱。
2. **白盒梯度方法不适用于黑盒**：已有对抗样本生成方法需要访问模型梯度/特征向量，在无法获取内部信息的黑盒设定下不可行。
3. **现有双模态攻击性能有限**：少数尝试图像+文本联合优化的工作（如 Ying et al. 2024, Wang et al. 2024b）要么分别优化各模态，要么仅支持单向交互且局限于白盒。
4. **缺乏系统性黑盒双模态交互优化机制**：现有工作未能充分利用 LVLM 中图像与文本模态间的交互特性来最大化输出毒性。

## 核心贡献（创新点）
1. **提出先验引导的图像扰动生成方法**：利用替代 LVLM 从有害语料中提取恶意特征并嵌入良性图像作为先验，使后续优化从高质量起点出发——与随机初始化基线相比，ASR 平均提升约 20 个百分点。
2. **提出双向交叉模态交互优化框架**：通过贪婪搜索交替更新图像扰动和文本后缀，实现图像-文本双模态联合优化——区别于以往单向交互（Wang et al. 2024b）或分别优化（Ying et al. 2024）的方法。
3. **设计黑盒毒性最大化目标函数**：结合 toxicity score（Perspective API / Detoxify）和特征对齐损失 $\mathcal{L}(\mathbf{x}_{\mathrm{adv}})$ 指导图像扰动生成——无需目标模型的梯度信息，仅需黑盒查询。
4. **系统性实验验证黑盒/白盒双场景有效性**：在 3 个开源（MiniGPT-4、InstructBLIP、LLaVA）和 3 个闭源（Gemini、GPT-4、Qwen-VL）LVLM 上均超越 11 个基线方法，白盒平均 ASR 92.5%，黑盒平均 ASR 67.3%。

## 方法详解
PBI-Attack 分为两个阶段：

**阶段一：先验扰动生成（Prior Perturbation Generation）**
- 从有害语料 $Y = \{\mathbf{y}_i\}_{i=1}^m$ 中使用替代 LVLM 提取恶意特征，将其嵌入良性图像 $\mathbf{x}_{\mathrm{benign}}$。
- 图像扰动更新公式（式 1）：$\mathbf{x}_{\mathrm{adv}} = \mathbf{x}_{\mathrm{benign}} \oplus \mathbf{x}_{\mathrm{adv}}^p$，其中 $\oplus$ 表示通过特征提取函数 $h(\cdot)$ 叠加。
- 损失函数（式 2）：$\mathcal{L}(\mathbf{x}_{\mathrm{adv}}) = \sum_{i=1}^m -\mathrm{T}(\mathbf{x}_{\mathrm{adv}}, \mathbf{y}_i) + \lambda \|h(\mathbf{x}_{\mathrm{adv}}) - g(\mathbf{y}_i)\|$，第一项最大化毒性响应，第二项拉近图像-文本特征距离，$\lambda=1.0$。
- 使用投影梯度下降（PGD）迭代更新扰动（式 3）：$\mathbf{x}_{\mathrm{adv}}^p = h^{-1}(h(\mathbf{x}_{\mathrm{adv}}^p) - \eta \nabla \mathcal{L}(\mathbf{x}_{\mathrm{adv}}))$，其中梯度针对特征空间计算，适用于黑盒设定。

**阶段二：双模态对抗优化循环（Bimodal Adversarial Optimization Loop）**
- **文本优化（式 4-5）**：从预定义文本后缀语料 $Y^s$ 中贪婪选择最大化毒性的后缀 $\mathbf{y}_{\mathrm{new}}^s = \mathrm{argmax}_{\mathbf{y} \in Y^s} \mathrm{T}(\mathbf{x}_{\mathrm{adv}}, \mathbf{y}_{\mathrm{adv}} || \mathbf{y})$，拼接更新 $\mathbf{y}_{\mathrm{adv}}$。
- **图像优化（式 6-7）**：随机生成 $K$ 个约束扰动（$\|h(\mathbf{x}_j^p)\|_\infty \leq B$），贪婪选择 $\mathbf{x}_{\mathrm{new}}^p = \mathrm{argmax}_{\mathbf{x} \in X^p} \mathrm{T}(\mathbf{x}_{\mathrm{adv}} \oplus \mathbf{x}, \mathbf{y}_{\mathrm{adv}})$，叠加更新 $\mathbf{x}_{\mathrm{adv}}$。
- 交替执行文本/图像优化共 $N=400$ 轮（图像）和 $100$ 轮（文本），每轮各更新 5 次；达到毒性阈值 $T_{\mathrm{toxicity}}$ 或达到迭代上限时终止。
- 毒性评估使用 Perspective API（8 属性）或 Detoxify（6 属性），对每个输入查询 10 次取均值以减少随机性。

## 实验与结果
- **数据集**：AdvBench（520 条 prompt）、HADES 数据集。
- **目标模型**：白盒——MiniGPT-4、InstructBLIP、LLaVA；黑盒——Gemini、GPT-4、Qwen-VL。
- **对比基线**：Arondight、GCG、Advimage、ImgJP、UMK、InPieces、BAP、MLAI、FigStep、HADES，共 10 个 SOTA 方法。
- **主要结果**（Table 1，Perspective API 指导）：
  - PBI-Attack 在**所有 6 个模型**上均取得最高 ASR：MiniGPT-4（94.9%）、InstructBLIP（93.2%）、LLaVA（89.3%）、Gemini（71.7%）、GPT-4（63.2%）、Qwen-VL（67.1%）。
  - 最强提升：在 MiniGPT-4 上较次优方法 UMK（87.5%）提升 **7.4 个百分点**；在 Gemini 上较次优 HADES（63.5%）提升 **8.2 个百分点**。
  - 白盒平均 ASR：**92.5%**；黑盒平均 ASR：**67.3%**。
- **消融结论**：
  - 先验初始化 vs 随机初始化：ASR 提升约 20%（Table 3）。
  - 毒性分数目标 vs jailbreak 概率目标：毒性分数引导优化更有效（Table 2）。
  - Stage1 only（78.2%）vs Stage1+2（94.7%）：迭代双模态优化带来显著增益（Table 11）。
  - 不同 seed 图像：6 种种子 ASR 无显著差异（Table 12），说明优化过程而非种子图像驱动攻击成功。

## 相关工作脉络
1. **UMK (Wang et al. 2024b)**：白盒双模态联合优化，但仅支持单向交互（先图后文），无法在纯黑盒场景应用；PBI-Attack 通过替代模型提取特征实现黑盒双模态双向交互。
2. **BAP (Ying et al. 2024)**：query-agnostic 图像扰动 + intent-specific 文本优化，但两模态分别优化，未充分挖掘图像-文本交互效应；PBI-Attack 通过交替优化实现真正双向互动。
3. **Advimage (Qi et al. 2024)**：单一对抗图像越狱 LLM，仅关注单模态（图像），无法利用文本扰动辅助；PBI-Attack 联合优化双模态。
4. **InPieces (Shayegani et al. 2023a)**：将恶意文本触发词嵌入图像进行越狱，扰动幅度有限且未进行迭代优化；PBI-Attack 采用 PGD 迭代优化特征对齐。
5. **HADES (Li et al. 2025b)**：白盒方法，利用精心设计的图像放大文本中的恶意意图；PBI-Attack 适用于黑盒且同时优化图文两端。
6. **GCG (Zou et al. 2023)**：LLM 文本越狱的经典黑盒/白盒方法，仅作用于文本 token 空间；PBI-Attack 将其思想扩展至图像+文本双模态交互优化。

## 局限性与未来方向
1. **计算开销大**：PBI-Attack 训练时间 27.9 小时、攻击时间 123.1 秒（Table 6），显著高于大多数基线方法（如 UMK 33.1 秒、Advimage 31.5 秒），主要源于多轮黑盒查询和迭代优化。
2. **单次响应耗时长**：模型每次生成响应需数秒，数千次迭代累积耗时巨大（论文自述 limitation）。
3. **仅针对毒性最大化**：评估指标聚焦毒性 score，未系统评估攻击的语义多样性、人类可读性等维度。
4. **依赖替代模型的特征提取能力**：先验生成阶段使用 MiniGPT-4/InstructBLIP/LLaVA 作为 surrogate，不同 surrogate 的选择对最终攻击效果有影响（Table 4/5），但最小差异仍存在。
5. **未来方向**：可探索更高效的查询策略减少迭代次数、扩展到其他安全属性（如偏见、虚假信息）、研究更强防御机制。

## 研究启发与可借鉴点
1. **替代模型特征对齐作为先验**：在黑盒设定下，用白盒替代模型提取特征空间中的恶意模式并编码为扰动先验，是一种有效的"信息借力"策略，可迁移到其他黑盒攻击/红队测试场景。
2. **交替贪婪搜索优化双模态**：文本和图像分步贪婪选择最优补丁的交替优化模式，计算效率高且效果显著，值得借鉴到多模态对抗样本生成研究中。
3. **毒性分数作为黑盒优化目标的可行性**：无需模型梯度，仅靠 toxicity score（Perspective API/Detoxify）即可驱动有效优化，且优于 jailbreak 概率（Table 2），为纯黑盒安全评估提供了实用范式。
4. **seed 图像不敏感性**：实验表明不同初始图像（含纯随机噪声）产生相近 ASR，说明迭代优化过程本身是主导因素，可放心使用简单 seed 保证复现性。
5. **固定时间预算下的性能天花板分析**：Table 7 展示了在不同时间预算下各方法的 ASR 对比，揭示了 PBI-Attack 具有更高的性能上限（20h 时达 94.2%），这种成本-收益权衡分析值得在后续工作中引入。

## 关键术语表
**LVLM（Large Vision-Language Model）**：融合视觉和语言理解的大规模多模态模型，如 GPT-4V、LLaVA、Gemini，可同时处理图像和文本输入并生成文本输出。
**Jailbreak Attack（越狱攻击）**：通过精心构造的输入绕过 AI 模型的安全对齐机制，诱导模型生成有害/不当内容的攻击方法。
**Black-Box Attack（黑盒攻击）**：攻击者仅能观察模型的输入-输出对，无法获取内部参数或梯度的攻击设定。
**Toxicity Score（毒性评分）**：使用 Perspective API 或 Detoxify 等评估模型对文本中毒性属性的打分（0-1），用于量化模型输出的有害程度。
**Bimodal Interaction（双模态交互）**：指图像和文本模态在 LVLM 中相互影响的机制，PBI-Attack 通过交替优化两者来放大这种交互的有害效应。
**Projected Gradient Descent (PGD)**：一种迭代优化算法，在每一步沿损失函数梯度方向更新变量并投影回约束空间，本文用于更新图像扰动。
**ASR（Attack Success Rate，攻击成功率）**：成功越狱的 prompt 比例，本文用 HarmBench + GPT-3.5-turbo 评估。
**Surrogate Model（替代模型/代理模型）**：白盒可访问的模型，用于提取特征信息指导黑盒攻击，本文使用 MiniGPT-4、InstructBLIP、LLaVA 作为 surrogate。

## 可复现要素
- **数据集**：AdvBench（520 条 prompt）和 HADES 数据集；AdvBench 为公开 benchmark。
- **代码开源**：是，代码已开源于 https://github.com/Rosy0912/PBI-Attack。
- **目标模型**：白盒使用 MiniGPT-4（Vicuna-13B）、InstructBLIP（Vicuna-13B）、LLaVA（LLaMA-2-13B）；黑盒使用 Gemini、GPT-4、Qwen-VL。
- **关键超参**：$\lambda = 1.0$，学习率 $\eta$ 未明确给出，步长 $\alpha=1$，batch size $b=8$，图像扰动候选数 $K=50$，文本后缀长度 10 tokens，候选数 400，每轮图像/文本各更新 5 次，图像优化迭代 400 轮，文本优化迭代 100 轮，毒性评估查询 10 次取均值。
- **实验环境**：8 张 NVIDIA A100 GPU（80GB 显存）。
