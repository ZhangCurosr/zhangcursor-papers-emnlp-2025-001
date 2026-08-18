---
title: "Seeing-is-Believing-but-How-Much-A-Comprehensive-Analysis-of"
source: https://aclanthology.org/2025.emnlp-main.74.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:39:49"
field: "多模态语言模型可靠性与校准"
keywords: ["verbalized calibration", "vision-language models", "uncertainty quantification", "multimodal alignment", "prompting strategies"]
innovations: ["首次系统性评估VLMs跨模态语言化校准，揭示视觉推理模型的优势", "提出VCAP双阶段提示策略解耦视觉理解与任务执行", "建立嵌入指令与语义对齐模态的校准评估框架"]
benchmarks: ["MMMU-Pro", "VideoMMMU", "Visual SimpleQA", "MathVista", "MathVision", "IsoBench"]
---

# 论文速读：Seeing-is-Believing-but-How-Much-A-Comprehensive-Analysis-of-Verbalized-Calibration-in-Vision-Language-Models

## 一句话总结
本文对当前视觉-语言模型（VLMs）的语言化置信度校准进行了全面评估，发现大多数模型存在显著的校准偏差（miscalibration），而视觉推理模型表现更优；并提出了双阶段提示策略 **Visual Confidence-Aware Prompting (VCAP)** 以改善多模态场景下的置信度对齐。

## 研究问题与动机
- 随着VLMs在高风险场景中的应用增加，评估其可信度与任务性能同等重要，而**校准（calibration）**——即模型表达的置信度与其实际准确率的一致性——是可信度的核心指标。
- 现有工作主要关注纯文本LLM的语言化不确定性估计，而VLMs涉及多模态信息融合，其置信度表达是否同样可靠尚不清楚。
- 当前研究缺乏对VLMs语言化校准的系统性评估，特别是在不同模态输入、不同任务域和不同模型架构下的校准行为差异。
- 嵌入图像的指令（而非纯文本指令）和跨模态语义等价输入对校准的影响尚未被充分探索。

## 核心贡献（创新点）
- **首个全面的VLMs语言化置信度评估**：覆盖三种模型类别、四个任务域（图像理解、视频理解、事实性判断、数学推理）和三种评估场景（通用、嵌入指令、语义对齐），填补了多模态校准评估的空白。
- **发现视觉推理模型校准显著更优**：具备"图像思维"能力的模型（如o3、o4-mini）在跨任务和多模态设置下 consistently 表现出更低的ECE，揭示了模态特定推理对可靠不确定性估计的关键作用。
- **提出VCAP双阶段提示策略**：通过解耦视觉理解与任务执行，先在视觉模态上独立评估置信度，再结合最终任务生成校准后的置信度，显著优于Top-K和Self-reflection基线。
- **揭示模态不对齐的校准代价**：在嵌入指令和语义对齐设置下，VLMs在处理视觉输入时的校准明显劣于文本输入，即使内容相同也存在显著的模态差距。

## 方法详解
**评估框架**包含三个互补的评估配置：

1. **通用评估（General Evaluation）**：通过文本指令提示模型处理视觉输入，覆盖MMMU-Pro（图像推理）、VideoMMMU（视频理解，均匀采样32帧）、Visual SimpleQA（事实性判断）和MathVista/MathVision（视觉数学推理，使用testmini split）。

2. **嵌入指令评估（Embedded Instruction Evaluation）**：采用MMMU-Pro的vision-only配置，问题完全嵌入图像中呈现，评估视觉指令对校准的影响。

3. **语义对齐模态评估（Semantically Aligned Modalities Evaluation）**：使用IsoBench基准，提供语义等价但模态不同的输入（如数学函数以图像图表或LaTeX公式呈现），隔离模态效应。

**评估指标**：
- 主要指标为期望校准误差（Expected Calibration Error, ECE）：$ECE = \sum_{m=1}^{M} \frac{|B_m|}{n} |\text{acc}(B_m) - \text{avgConf}(B_m)|$，使用M=10个置信度分箱。
- 同时报告准确率（ACC）和AUROC（用于失败预测）。

**VCAP方法**：
- **第一阶段**：要求模型详细描述视觉输入并提供置信度评分，专注于视觉模态以最小化认知负荷并隔离感知过程。
- **第二阶段**：提示模型完成任务并生成最终置信度，同时考虑第一阶段的视觉自我评估。
- 通过结构化多轮对话解耦感知与推理，鼓励更反思性的处理，提升视觉模态的校准质量。

## 实验与结果
**数据集与模型**：
- 基准：MMMU-Pro、VideoMMMU、Visual SimpleQA、MathVista、MathVision、IsoBench
- 模型分类：通用指令微调模型（GPT-4.1、Qwen-VL系列、InternVL3、Kimi-VL-Instruct）、文本中心推理模型（o1、Kimi-VL-Thinking、Skywork-R1V系列）、视觉中心推理模型（o3、o4-mini）

**主要发现**：
- 通用评估中，大多数指令模型和文本推理模型的ECE超过0.25，呈现系统性过度自信；o3在所有任务上保持最佳校准（如MMMU-Pro ECE=0.047，MathVision ECE=0.111）。
- 嵌入指令设置下，所有模型校准性能下降，o3和o4-mini也出现明显退化，揭示了视觉指令理解的固有意义鸿沟。
- 语义对齐评估（IsoBench）显示，除视觉推理模型外，大多数模型在视觉输入下的ECE显著高于文本输入（如Qwen2.5-VL 72B在数学任务上：图像ECE=0.353 vs 文本ECE=0.008）。
- VCAP在IsoBench上对Qwen2.5-VL 7B的ECE从0.467降至0.424，对72B模型从0.431降至0.365，优于Top-K和Self-reflection基线。

## 相关工作脉络
- **Xiong et al. (2024)**：系统评估LLM语言化不确定性，发现指令微调模型普遍过度自信；本文将其扩展到多模态领域，发现VLMs存在类似的校准问题且受模态影响。
- **Zeng et al. (2025)**：评估推理模型的语言化校准，发现推理训练可改善校准；本文提供了多模态视角的佐证，并进一步揭示视觉推理的额外优势。
- **Groot & Valdenegro Toro (2024)**：在小型39图像日文数据集上评估VLMs语言化不确定性；本文扩展至大规模多任务、多模态评估。
- **Tian et al. (2023) / Xiong et al. (2024)**：提出Top-K提示和自反思提示以提升LLM校准；本文验证这些方法在VLMs上的有效性并提出更优的VCAP。
- **Fu et al. (2024) / IsoBench**：提出用于测试模态对齐的基准；本文将其用于校准分析，揭示了校准与模态对齐的交叉问题。
- **Hager et al. (2025)**：通过蒸馏自一致性信号改进LLM校准；本文强调多模态场景中显式利用多模态输入和模态特定推理的重要性。

## 局限性与未来方向
- 视觉推理模型（o3、o4-mini）表现优异但为闭源模型，缺乏公开实现细节，难以深入分析其成功原因。
- 目前"图像思维"类别仅有两个闭源模型，无法研究不同模型族在该方法下的校准表现差异。
- 未针对下游应用中常见的高难度场景进行专项实验，如视频理解中的时序视觉定位。
- 仅评估了数值语言化不确定性（NVU），未深入探索语言化不确定性（LVU）在多模态模型中的表现（附录F显示效果因模型架构而异）。

## 研究启发与可借鉴点
- **多阶段提示的模态解耦设计**：VCAP通过将视觉理解与任务执行分离，有效提升了校准质量，该思路可迁移到其他多模态可信度估计任务。
- **跨模态语义等价的校准评估范式**：IsoBench的"相同内容不同模态"设计为分析模态不对齐提供了通用框架，可用于评估其他多模态能力。
- **推理训练对多模态校准的增益**：视觉推理模型（o3/o4-mini）的优异表现表明，端到端的视觉链式思考训练值得在开源模型中探索。
- **视觉复杂度的校准影响**：附录D发现VCAP在视觉复杂度高（OCR提取、干扰元素过滤）的场景下校准提升更显著，提示应针对性优化视觉感知阶段的置信度评估。
- **多轮提示的边际收益递减**：附录D.2表明三阶段VCAP相比两阶段无明显增益，为多模态提示效率提供了实证依据。

## 关键术语表
- **Verbalized Calibration（语言化校准）**：模型通过自然语言表达置信度时，其表达值与实际准确率的对齐程度。
- **Expected Calibration Error (ECE)**：衡量预测置信度与观测准确率之间偏差的指标，值越低表示校准越好。
- **Visual Reasoning Models（视觉推理模型）**：将视觉输入直接纳入多步推理过程的模型（如o3、o4-mini），区别于仅在感知阶段使用视觉的模型。
- **Modality Misalignment（模态不对齐）**：模型对语义等价但模态不同的输入产生不一致表现的系统性偏差。
- **Embedded Instruction（嵌入指令）**：将问题文本嵌入图像中呈现，而非通过标准文本模态输入的评估设置。
- **IsoBench**：专为测试多模态基础模型在等距表示下模态对齐能力而设计的基准，涵盖数学、游戏、科学和算法四个领域。
- **VCAP（Visual Confidence-Aware Prompting）**：本文提出的双阶段提示策略，先隔离视觉置信度评估再聚合最终响应。
- **Linguistic Verbal Uncertainty (LVU) vs Numerical Verbal Uncertainty (NVU)**：前者使用自然语言描述不确定性（如"可能"、"不太确定"），后者使用数值评分（如"75%置信度"）。

## 可复现要素
- 数据集：MMMU-Pro、VideoMMMU、Visual SimpleQA、MathVista、MathVision、IsoBench均为公开基准（论文引用了各数据集的来源）。
- 代码/权重：模型权重部分开源（Qwen2.5-VL、InternVL3、Skywork-R1V系列等），闭源模型（GPT系列、o3/o4-mini）通过API访问；提示模板完整提供于附录A。
- 关键超参：ECE计算使用M=10个分箱；VideoMMMU均匀采样32帧；Top-K提示使用K=3；推理使用vLLM引擎及默认SamplingParams。
