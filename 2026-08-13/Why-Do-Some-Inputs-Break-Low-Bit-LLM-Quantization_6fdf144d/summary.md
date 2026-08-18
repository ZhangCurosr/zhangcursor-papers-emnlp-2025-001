---
title: "Why-Do-Some-Inputs-Break-Low-Bit-LLM-Quantization"
source: https://aclanthology.org/2025.emnlp-main.168.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 18:52:04"
---

# 论文速读：Why-Do-Some-Inputs-Break-Low-Bit-LLM-Quantization

## 一句话总结
本文系统剖析了 3–4 bit weight-only 量化中“为何部分输入会产生巨大误差”的问题，发现不同量化方法的样本级误差高度一致，且全精度模型的残差流幅值可精准预测后续量化损失；通过机制定位揭示出 RMSNorm 的幅值反转效应与 MLP gate 输出是误差累积的核心脆弱点。

## 研究问题与动机
- **核心问题**：低比特 weight-only 量化虽能大幅压缩显存，但会对特定样本造成不成比例的性能退化；现有工作多聚焦 outlier 处理，却未回答“为何是这些样本出错”以及“误差来源究竟是权重还是激活”。
- **现有方法不足**：GPTQ、AWQ、NF 等主流方法虽在平均指标上表现良好，但对高误差样本无效；单纯提升晚层权重精度（混合精度）效果有限，缺乏对误差传播路径的机制级解释。
- **科学假设**：若量化失败源于输入固有特性而非方法偶然性，则不同方法的误差应强相关，且全精度模型的内部表征应能预先指示量化风险。
- **定位目标**：通过相关性分析、残差追踪、early exiting 与 activation patching，回答 why（哪些输入）、where（哪一层/组件）、what（何种数据特征）导致量化崩盘。

## 核心贡献（创新点）
1. **量化误差的方法无关性验证**：证明 50 对 3–4 bit 量化方法在 FineWeb 上的样本级误差呈强正相关（平均 $\rho=0.82$），且 top-10% 高误差样本集高度重叠（Jaccard ≈ 0.51），表明量化失败主要由输入本身决定而非量化算法偶然性。
2. **残差幅值预测量化误差**：首次建立全精度模型最后一层残差流幅值与最终量化误差的强负相关关系（最高达 -0.83），证明前向表征足以预判后验量化风险。
3. **揭示 RMSNorm 幅值反转机制**：提出并验证“RMSNorm reversal effect”，即上层残差幅值较小的样本经 RMSNorm 归一化后反而获得更大的激活幅值，从而在多层传递中持续放大量化权重误差。
4. **定位 MLP gate 为误差放大关键节点**：通过 CMAP 发现，仅恢复 MLP gate 投影输出（$h_{\text{gate}}$）即可大幅压缩 3-bit 与 16-bit 的困惑度差距，而恢复晚层权重仅带来边际改善，明确了误差累积的核心路径。

## 方法详解
- **误差定义与样本划分**：量化误差定义为 NLL 增量 $e(x;\tilde{\theta}) = \text{NLL}(x;\tilde{\theta}) - \text{NLL}(x;\theta)$。按误差分布选取 top-10% 构成 $\mathcal{D}_{\text{large}}$，bottom-50% 构成对照集 $\mathcal{D}_{\text{ctrl}}$，并进行严格的 5-gram Jaccard 去重。
- **残差幅值建模**：定义第 $l$ 层残差幅值 $\text{norm}(x; r^{(l)}) = \frac{1}{T}\sum_{t=1}^T \|r_t^{(l)}\|_2$，其中 $r^{(l)} = z^{(l-1)} + \text{MHA}^{(l)}(\text{LN}_1^{(l)}(z^{(l-1)}))$。计算其与最终量化误差 $\mathcal{E}(\mathcal{D};\tilde{\theta})$ 的 Pearson 相关系数。
- **RMSNorm 反转假设**：基于 $\text{RMSNorm}(z) = \frac{z}{\text{RMS}(z)} \odot \gamma$，分析发现 $\gamma$ 会对极端 outlier 维度赋予极小权重，导致原本幅值较小且 kurtosis 较低的残差经归一化后相对放大，转化为更大的 post-RMSNorm 激活 $h^{(l)}$。
- **Early Exiting 定位**：利用 logit lens 将各层 $r^{(l)}$ 经最终 RMSNorm 后投影至词表：$p^{(l)} = \text{Softmax}(U \cdot \phi(r^{(l)}))$，计算逐层 NLL。通过对比全精度与量化模型的 NLL 分化层数，判断样本对深层网络的依赖程度。
- **Cross-Model Activation Patching (CMAP)**：在前向过程中将量化模型的特定模块激活替换为全精度对应值。针对 GLU-MLP 结构分别 patch $h_{\text{gate}}=\sigma(W_{\text{gate}}h)$、$h_{\text{up}}=W_{\text{up}}h$、$h_{\text{down}}=W_{\text{down}}(h_{\text{gate}}\odot h_{\text{up}})$ 以及 $h_{\text{attn}}$，并额外验证仅恢复 $W_{\text{gate}}$ 权重的效果。

## 实验与结果
- **数据集与基线**：FineWeb（10k 英文网页，截断至 512 token）、PopQA（14k 事实问答）；基线模型 Qwen2.5-7B、Llama3-8B、Mistral-Nemo-12B、Llama3-70
