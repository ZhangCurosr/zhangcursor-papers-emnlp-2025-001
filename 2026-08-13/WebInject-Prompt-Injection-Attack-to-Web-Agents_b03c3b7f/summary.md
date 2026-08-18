---
title: "WebInject-Prompt-Injection-Attack-to-Web-Agents"
source: https://aclanthology.org/2025.emnlp-main.104.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 17:45:25"
field: "大语言模型安全与对抗攻击"
keywords: ["prompt injection", "web agents", "adversarial attack", "MLLM", "visual adversarial", "AI security"]
innovations: ["首个同时具备有效性、隐蔽性与现实可行性的网页端提示注入攻击框架", "利用U-Net逼近不可微的网页到截图映射并结合可微降采样替代实现端到端优化", "跨多显示器通用重叠区域扰动策略"]
benchmarks: ["10个网页数据集(5类真实+5类合成)", "5个MLLM-based Web Agent (UI-TARS, Phi-4, Llama-3.2, Qwen2.5, Gemma-3)"]
---

# 论文速读：WebInject-Prompt-Injection-Attack-to-Web-Agents

## 一句话总结
本文提出了 WebInject，一种针对基于 MLLM 的 Web Agent 的网页端提示注入攻击方法，通过在网页源代码中注入人类不可察觉的像素扰动，使 Web Agent 执行攻击者指定的目标动作；其核心创新是利用神经网络逼近不可微的网页到屏幕截图映射，并结合可微分辨率替代方案，通过投影梯度下降求解最优扰动。

## 研究问题与动机
- **现有网页端攻击效果差且缺乏隐蔽性**：已有网页基于攻击（如 Pop-up Attack、EIA）依赖启发式注入可见 HTML 元素，有效性低且容易被用户察觉。
- **截屏级攻击不切实际**：已有截图扰动攻击直接操作用户设备上的截屏像素，但攻击者无法访问本地截屏，缺乏现实可行性。
- **网页到截图映射的不可微性是技术难点**：浏览器渲染后的原始像素值经过 ICC 色彩配置文件变换生成截屏，该映射不可微，且 MLLM 的降采样操作同样不可微，阻碍了端到端梯度优化。
- **多显示器兼容性挑战**：不同显示器尺寸和 ICC 配置导致同一网页产生不同截屏，扰动需跨显示器保持通用性和可见性。

## 核心贡献（创新点）
- **首个同时具备高有效性、高隐蔽性与现实可行性的网页端提示注入攻击框架**：区别于启发式网页攻击和不可实施的截屏攻击，WebInject 通过优化可实际注入网页源代码的像素扰动实现攻击。
- **提出基于神经网络逼近的不可微映射求解方案**：训练 U-Net 映射神经网络近似网页到截图的非可微映射 $M(\cdot, ICC_d)$，并将 MLLM 降采样操作替换为可微替代 $r'(\cdot)$，使梯度能够反向传播至扰动。
- **设计了跨多显示器通用的重叠区域约束扰动策略**：将扰动限制在所有目标显示器共享的重叠区域 $[0, w_\delta] \times [0, h_\delta]$ 内，确保扰动在不同尺寸显示器上均可见且保持通用性。
- **构建完整的真实+合成网页评测基准并揭示现有基线方法严重不足**：使用 10 个网页数据集（5 类真实+5 类合成）和 5 个 MLLM Agent，证明截图类攻击直接应用于原始像素时 ASR 为 0.000，凸显本文方法的必要性。

## 方法详解
- **威胁模型**：攻击者控制目标网页源代码 $\omega$，拥有 MLLM $f$ 的参数，能构造目标提示集 $\mathcal{P}$ 和目标显示器集 $\mathcal{D}$，但无法访问真实交互历史和用户截屏。
- **优化问题 formulation**：目标是最小化跨提示、显示器、阴影历史的交叉熵损失，约束条件为 $\|\delta\|_\infty \leq \epsilon$（隐蔽性）和仅在重叠区域内非零（多显示器兼容）。
- **映射神经网络训练**：对每个目标显示器 $d$，用 U-Net 架构训练 $\mathcal{N}_d$ 近似 $M(\cdot, ICC_d)$。训练数据通过随机扰动原始像素后经 ICC 变换获得输入-输出对，无需物理显示器访问，仅需公开可用的 ICC 配置文件即可模拟。
- **可微降采样替代**：用 PyTorch `torch.F.interpolate()` 或 TensorFlow `tensorflow.image.resize()` 替代 MLLM 离散降采样操作 $r(\cdot)$，使梯度可流回扰动。
- **投影梯度下降（PGD）求解**：初始化 $\delta = 0$，每步随机采样 mini-batch 计算梯度后更新 $\delta \leftarrow \delta - \alpha \cdot g$，再通过 clamp 函数约束 $\|\delta\|_\infty \leq \epsilon$，最后通过掩码矩阵 $S$ 保证扰动仅在重叠区域非零：$\delta \leftarrow S \odot \delta$。超参数：$\epsilon = 16/255$，$\alpha = 0.3$，$T = 2500$ 次迭代。
- **源代码注入实现**：通过注入 JavaScript 代码，在浏览器渲染后提取重叠区域内的原始像素值，加上扰动 $\delta$ 后写回，同时将原始 HTML 元素置于顶层并设 opacity=0，保证用户交互正常但截屏包含扰动。

## 实验与结果
- **数据集**：10 个网页数据集——5 类真实网页（Blog/50, Commerce/26, Education/42, Healthcare/51, Portfolio/43）和 5 类合成网页（每类 100 个，由 GPT-4-Turbo 生成）。
- **Agent 模型**：UI-TARS-7B-SFT、Phi-4-multimodal-instruct、Llama-3.2-11B-Vision-Instruct、Qwen2.5-VL-7B-Instruct、Gemma-3-4bit。
- **评估指标**：攻击成功率（ASR），即 Agent 输出精确目标动作的比例。
- **最强结果**：WebInject 在 Gemma-3 Agent 上达到 ASR = 0.972，比最佳基线（Naive Attack, 0.062）高出 **0.910**；在 UI-TARS 上达到 ASR = 0.975，比最佳基线（Context Ignoring, 0.147）高出约 0.828。所有基线中截图类攻击 ASR 全为 0.000。
- **消融发现**：（1）目标显示器数量增加时 ASR 轻微下降但仍远高于基线；（2）$\epsilon$ 从 4/255 增至 32/255 时 ASR 趋近 1.0，$\epsilon \leq 16/255$ 即被认为视觉不可察觉；（3）语义等价用户提示下 ASR 与目标提示下相当（如 Gemma-3 合成 Blog 从 0.988 降至 0.957）；（4）对 click、drag、hotkey、type、scroll 等多种目标动作均有效（ASR 均 > 0.976）。
- **计算开销**：相比截屏攻击，WebInject 额外增加约 1.7-2.6 分钟训练时间和约 1.9-2.6 GB 显存。

## 相关工作脉络
- **EIA (Liao et al., 2025)**：环境注入攻击，通过注入与合法元素相似的 HTML 元素欺骗 Agent；WebInject 不依赖可见注入元素，而是优化像素扰动，在效果上远超 EIA。
- **Pop-up Attack (Zhang et al., 2024)**：注入误导性弹窗诱导 Agent 点击特定坐标；属启发式方法，WebInject 为优化驱动，ASR 提升超过 0.8。
- **Screenshot-based attacks (Aichberger et al., 2025; Zhao et al., 2025)**：直接在截屏上优化视觉扰动；WebInject 指出此类攻击忽略了网页到截图映射且缺乏现实可行性，实验中证实其直接应用于原始像素时完全失效（ASR=0）。
- **文本提示注入 (Willison, 2022/2023; Liu et al., 2024)**：Naive、Context Ignoring、Fake Completion、Combined 等；WebInject 通过视觉而非文本通道攻击，规避了基于文本的检测方法（如 DataSentinel）。
- **对抗样本检测 (Carlini & Wagner, 2017)** 与 **对抗训练 (Madry et al., 2018)**：论文提及作为潜在防御方向，但未在本工作中实现。
- **AdvAgent (Xu et al., 2024)** 与 **InjecAgent (Zhan et al., 2024)**：分别在黑盒红队测试和工具集成 Agent 的间接提示注入方面进行评估，WebInject 聚焦于网页端像素扰动这一全新攻击面。

## 局限性与未来方向
- **假设攻击者可修改网页源代码**：对 Amazon 等高信任度网站不适用，限制了攻击场景范围。
- **未评估对闭源 MLLM 的迁移性**：受限于计算资源，未进行多代理模型联合优化的迁移攻击实验。
- **仅考虑初始步骤的单步动作攻击**：未评估多步交互场景下的累积攻击效果。
- **未来方向**：研究针对闭源模型的迁移攻击、开发基于源代码分析的检测机制、探索对抗训练增强 MLLM 鲁棒性。

## 研究启发与可借鉴点
- **不可微映射的神经网络逼近策略**：用 U-Net 近似浏览器渲染+ICC 变换的端到端映射，为其他涉及不可微物理/系统映射的对抗攻击提供了可复用范式。
- **可微降采样替代技巧**：用框架内置可微插值函数替代离散重采样操作，是解决 MLLM 视觉模块不可微问题的通用技巧。
- **重叠区域通用扰动设计**：通过限制扰动在多个目标设备共享区域的优化策略，实现了跨设备通用攻击，可迁移到其他多目标对抗场景。
- **与团队方向的结合机会**：本文揭示了 MLLM-based Web Agent 在视觉输入层面的脆弱性，可用于团队的红队测试框架建设；其防御思路（源代码异常检测、对抗训练）也可反向指导团队在 Agent 安全方面的防护研究。

## 关键术语表
**WebInject**：一种针对 MLLM 驱动 Web Agent 的网页端提示注入攻击方法，通过在网页原始像素中注入不可察觉扰动误导 Agent 执行目标动作。
**Prompt Injection Attack（提示注入攻击）**： adversaries 通过向模型输入中嵌入恶意指令，使模型偏离预期行为而执行攻击者指定的任务。
**Page-to-screenshot mapping（网页到截图映射）**：浏览器将网页源代码渲染为原始像素后，经 ICC 色彩配置文件变换生成显示器截图的非可微映射过程。
**ICC Profile（ICC 色彩配置文件）**：定义特定显示器颜色呈现方式的配置文件，不同型号显示器使用不同 ICC 配置，导致相同网页产生不同截屏。
**Shadow History（阴影历史）**：攻击者构造的模拟交互历史，用于优化阶段替代无法获取的真实用户交互历史。
**Attack Success Rate (ASR)**：衡量攻击有效性的指标，定义为 Agent 输出精确目标动作的比例。
**Projected Gradient Descent (PGD)**：带约束的梯度下降优化方法，每次更新后通过投影操作（clamp 和掩码）确保扰动满足约束条件。
**Target Action（目标动作）**：攻击者希望 Web Agent 执行的指定动作，如 `click((x,y))`、`type(content)` 等。

## 可复现要素
- **数据集**：真实网页通过 SingleFile 扩展下载，合成网页用 GPT-4-Turbo 生成；论文未提供公开下载链接，代码和数据未开源。
- **代码/权重**：论文未提及代码和权重开源情况。
- **关键超参**：$\epsilon = 16/255$，学习率 $\alpha = 0.3$，迭代次数 $T = 2500$；映射网络训练 200 epochs，学习率 0.005，batch size 16，共 16,240 对样本。
- **显示器模拟**：使用 Selenium + Canvas API + PIL ImageCms 实现，ICC 配置文件来源 TFTCentral 公开数据库。
- **评估环境**：5 个物理/模拟显示器（24-inch iMac M1, 15-inch MacBook Air M3, 27-inch LG 27UL500-W, 27-inch Dell S2722QC, 27-inch ASUS XG27UCG）。
