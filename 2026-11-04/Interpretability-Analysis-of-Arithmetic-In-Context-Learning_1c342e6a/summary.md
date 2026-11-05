---
title: "Interpretability-Analysis-of-Arithmetic-In-Context-Learning"
source: https://aclanthology.org/2025.emnlp-main.92.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 22:06:14"
field: "大语言模型可解释性"
keywords: ["机制可解释性", "上下文学习", "算术推理", "激活补丁", "函数向量", "信息流分析"]
innovations: ["首个三操作数算术ICE机制可解释性研究", "发现模式一致性比算术正确性更重要", "证明函数向量可蒸馏ICE模式并恢复零样本性能"]
benchmarks: ["Arithmetic-20", "Arithmetic-1000"]
---

# 论文速读：Interpretability-Analysis-of-Arithmetic-In-Context-Learning

## 一句话总结
本文对四个开源LLM在含上下文示例(ICE)的三操作数算术任务上的内部处理机制进行了系统的机制可解释性分析，发现模型更多依赖ICE的格式模式一致性而非算术正确性来完成推理。

## 研究问题与动机
1. LLM在文本算术推理上存在显著困难，ICE能提升性能但其内部处理机制仍不透明。
2. 现有机制可解释性工作多聚焦两操作数算术，缺乏对更复杂的三操作数算术任务的系统性研究。
3. ICE的信息利用方式不明：模型是真正执行算术运算，还是仅做表面模式匹配？
4. ICE中符号信息（token、数值正确性）与模式信息（格式一致性、token排列）的相对贡献尚未厘清。

## 核心贡献（创新点）
1. **首个三操作数算术ICE机制可解释性研究**：区别于仅关注两操作数算术的前作（如Stolfo et al., 2023；Nikankin et al., 2024），本文系统揭示了LLM如何处理复杂多步算术ICE。
2. **揭示"模式优先于算术"的核心机制**：通过系统的反事实干预实验证明，ICE结果token的格式一致性比算术正确性更重要，且模型传递的是模式信息而非部分和的算术内容。
3. **证实函数向量可蒸馏ICE模式信息**：从多个ICE prompt中提取的单一线性函数向量可替代完整ICE，在零样本场景恢复约15-25%绝对准确率，表明模型从ICE中提取的是通用格式而非具体算术规则。
4. **整合四种互补解释工具的统一分析框架**：将激活补丁（Activation Patching）、信息流分析（Information Flow Routes）、自动电路发现（Edge Attribution Patching）与logit-lens及函数向量方法整合，为后续机制研究提供了方法论参考。

## 方法详解
**数据集设计**：构建两个数据集——Arithmetic-20（整数0-20，支持数字和英文单词表示，单token）和Arithmetic-1000（整数0-999，仅数字表示，用于Llama-3.1-8B）。任务模板为三操作数算术（如 `1 + 3 + 4 = 8`），包含one-shot ICE（$p_1$）和zero-shot（$p_0$）两种prompt，筛选出LLM在$p_1$正确但$p_0$错误的数据以保证ICE有效。

**激活补丁（Activation Patching）**：对MLP输出 $m_t^{(k)}$ 和attention输出 $a_t^{(k)}$ 进行控制性干预。PE值定义为：$LD_{p_0^*}(r,\tilde{r}) - LD_{p_0}(r,\tilde{r})$，其中 $LD(r,\tilde{r}) = l(r) - l(\tilde{r})$，反映patch后正确token logit的提升幅度。

**信息流路由（Information Flow Routes）**：自顶向下递归追踪最关键的组件连接路径，构建模型内部信息传递的图表示。

**函数向量（Function Vectors）**：对每个attention head计算其在ICE prompt最后token位置的输出均值，选取Top-K正影响head，求和得FV，将其加入无ICE prompt残差流中评估效果。

**边缘属性补丁（Edge Attribution Patching, EAP）**：利用线性组合近似PE，公式为 $EAP(p_2|do(E=e_{p_1}))\approx(e_{p_2}-e_{p_1})^\top\frac{\partial}{\partial e_{p_1}}PE(\cdots)$，用于自动发现关键子网络。

**Logit Lens**：将模型内部表示通过unembedding矩阵解码为token空间，探测部分和部分（如$a+b$）在各层各位置的表示强度。

## 实验与结果
**实验模型**：Pythia-12B、Llama-3.1-8B、MPT-7B、OPT-6.7B。

**主要结果（Pythia-12B, Arithmetic-20）**：

| 实验 | 干预类型 | 准确率 |
|---|---|---|
| 基线（无干预） | — | 100%（正确解case） |
| Exp.3：改变算术正确性 | Symbol | 37.4% |
| Exp.4：改变格式一致性 | Pattern | 30.6% |
| Exp.5：算术错误+格式不一致 | Symbol+Pattern | 29.2% |
| Exp.5变体：随机非数字符号 | — | 0% |
| Exp.6：改变操作数 | Symbol | 69.4% |
| Exp.7：MLP-Attention联合补丁 | Component Interaction | 62.6% |

**函数向量结果**：Pythia-12B数字输入零样本准确率48.6%→注入FV后64.3%（+15.7%，完全恢复one-shot表现）；Llama-3.1-8B从30.0%→55.0%（+25%，部分恢复，one-shot为76.7%）。

**部分和探测**：logit lens显示ICE的部分和（$a+b$、$b+c$、$a+b+c$）仅在后期层的"="位置可解码，说明模型未从ICE中传递算术内容。

## 相关工作脉络
1. **Stolfo et al. (2023)**：首个机制可解释性研究两操作数算术，发现早期MLP处理操作数、注意路由至最终token。本文扩展至三操作数，并首次区分符号vs模式信息。
2. **Nikankin et al. (2024)**：发现LLM用"启发式规则集"而非算法解决两操作数算术。本文进一步揭示ICE的模式信息如何影响这些启发式的激活。
3. **Hendel et al. (2023) / Todd et al. (2023)**：提出函数向量概念用于简单任务。本文首次将FV应用于复杂三操作数算术，证明其蒸馏的是格式模式而非算术规则。
4. **Min et al. (2022)**：提出ICE结构格式比标签正确性更重要。本文通过反事实补丁实验给出了定量的因果证据，并区分了不同层次的模式影响。
5. **Ferrando & Voita (2024)**：提出信息流路由方法。本文将其与激活补丁、EAP结合形成统一分析框架，验证了跨方法的一致性。

## 局限性与未来方向
1. **间接因果效应**：所有结论基于激活补丁的间接推断，未来可用path patching进行更直接的因果验证。
2. **模型覆盖有限**：仅涉及四个特定开源模型，未覆盖更大规模或不同架构（如MoE）的LLM。
3. **计算成本高昂**：机制分析方法计算密集，限制了可扩展性。
4. **数据集规模较小**：Arithmetic-20仅为诊断性数据集，未来可扩展至更复杂数学推理任务（如微积分、微分方程）。

## 研究启发与可借鉴点
1. **符号vs模式分离的实验范式**：通过系统性反事实干预（保持格式改变数值、改变数值保持格式等）分离不同信息层次的影响，可直接迁移至其他ICE相关的解释性研究。
2. **函数向量用于零样本性能恢复**：证明蒸馏自少量ICE的单一向量可替代完整示例，为高效prompt压缩和少样本迁移学习提供了新思路。
3. **统一多工具的分析框架**：将activation patching、information flow、EAP、logit lens和FV整合为流水线，为复杂任务（如代码生成、多跳推理）的机制分析提供了可复用的方法模板。
4. **部分和未传递的发现**：与直觉相反——模型不从ICE"学习"具体算术运算——这一反直觉结论提醒研究者需谨慎解释LLM在数值推理上的"涌现"能力，对安全评估有启示意义。

## 关键术语表
**In-Context Examples (ICE)**：在推理时嵌入在prompt中的任务示例，用于引导模型执行目标任务而不需参数更新。
**Activation Patching**：通过替换模型内部激活值来量化特定组件对输出的因果贡献的机制可解释性技术。
**Information Flow Routes**：自顶向下自动构建重要计算路径子图的方法，揭示信息在模型内部的流动路线。
**Function Vector (FV)**：从多个含ICE的prompt中蒸馏出的紧凑内部向量表示，代表任务的抽象模式/格式信息。
**Edge Attribution Patching (EAP)**：基于梯度近似的边缘重要性评分方法，用于自动发现执行任务的关键子网络（circuit）。
**Logit Lens**：将模型内部隐藏状态通过unembedding矩阵解码为token分布，以探测各层各位置编码的信息内容。
**Patching Effect (PE)**：激活补丁实验中衡量干预效果的指标，定义为patch前后正确token与错误token的logit差变化量。

## 可复现要素
- **数据集**：Arithmetic-20和Arithmetic-1000（论文中描述生成过程，代码公开）
- **代码**：论文声明代码已公开（链接见摘要）
- **模型**：Pythia-12B、Llama-3.1-8B、MPT-7B、OPT-6.7B（均为开源模型）
- **关键超参**：FV选取top-30 heads（Pythia-12B）/ top-10 heads（Llama-3.1-8B）；信息流阈值τ（OPT-6.7B: 0.035, Llama-3.1-8B: 0.050）
- **计算平台**：论文未提及
