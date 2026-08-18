---
title: "SmartBench-Is-Your-LLM-Truly-a-Good-Chinese-Smartphone-Assis"
source: https://aclanthology.org/2025.emnlp-main.194.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:40:33"
field: "端侧大模型评测"
keywords: ["SmartBench", "端侧大模型", "手机助手评测", "主观对齐", "LLM-as-a-Judge", "W4A16量化", "移动端NPU部署"]
innovations: ["首个面向中文手机端大模型的5类20任务综合评测基准", "针对细分移动场景定制的多维LLM-as-a-Judge评分协议", "在真实手机NPU上系统验证W4A16量化部署的性能保留与工程画像"]
benchmarks: ["SmartBench"]
---

# 论文速读：SmartBench-Is-Your-LLM-Truly-a-Good-Chinese-Smartphone-Assis

## 一句话总结
本文提出了 **SmartBench**，首个面向中文手机端大模型的端到端评测基准，覆盖5大类20项日常移动交互任务共2973条高质量QA；同时首次在真实手机NPU上验证了W4A16量化部署的实际性能保留情况，填补了端侧LLM主观场景评测的空白。

## 研究问题与动机
1. **场景鸿沟（Scenario Gap）**：现有主流基准（如MMLU、GSM8K、HumanEval）高度侧重数学推理与代码生成，而手机端本地部署的LLM实际高频调用的是文本润色、摘要、通知管理等轻量级交互任务。
2. **语言鸿沟（Language Gap）**：现有主观评测协议几乎全部为英文，无法反映超过10亿中文智能手机用户在实际生活语境下的使用习惯与交互偏好。
3. **评测单一性**：现有中文基准（如CMRC、CLUE、C-Eval）偏重客观知识考查，AlignBench虽关注主观对齐但缺乏移动场景针对性，缺少针对“单步日常功能调用”的标准化评估体系。
4. **部署落地验证缺失**：学术评测多在服务器端进行，缺乏对端侧芯片（NPU）量化推理后的真实性能衰减与功耗/速度画像的系统性分析。

## 核心贡献（创新点）
1. **构建首个中文端侧手机场景评测基准**：基于对苹果、华为、小米、vivo等厂商现有端侧LLM功能的系统调研，归纳出5大类20项任务，并收集/合成2973条贴合真实手机交互的高质量QA。*区别于AlignBench的通用主观评测和AitZ的多步GUI智能体操作，本文聚焦单步、轻量化、日常生活导向的文本交互功能。*
2. **设计任务细粒度的自动化主观评分协议**：摒弃“一刀切”的LLM-as-a-Judge提示词，为内容创作、信息抽取、通知管理等类别单独定制多维评分维度（如连贯性、语言质量、创意、一致性）与评分标准。*与MT-Bench等通用主观基准相比，该协议与人类专家排名的Pearson相关系数更高（0.8280 vs 0.7959）。*
3. **完成端侧模型全链路实测与量化部署画像**：在vivo iQOO 12（Snapdragon 8 Gen 3）的NPU上实测BlueLM-3B与Qwen2.5-3B的W4A16量化推理性能，揭示不同任务对量化的敏感度差异，并提供预填充速度、输出速率与功耗数据。*以往工作多停留于云端BF16评测，本文补充了真实移动端算力约束下的性能保留率与工程可用性分析。*

## 方法详解
- **数据构成（Data Composition）**：5个一级类别、20个二级任务：
  - **Text Summarization**：DocumentSumm、CallSumm、RecordingSumm、MeetingSumm
  - **Content Creation**：TextPolishing、TextContinuation、TextAbbreviation、TextExpansion、TextCreation、TextFormatting、InstantReply、TextCorrection
  - **Text Q&A**：DocumentQ&A、RetrievalQ&A、PersonalQ&A
  - **Information Extraction**：EntityExtraction、RelationExtraction、EventExtraction
  - **Notification Management**：NotificationSorting、MessageSumm
- **数据来源与清洗**：开源数据集筛选（CMRC、DuReader、MSRA、OntoNotes、WenetSpeech、LCCC、VCSum、Alimeeting4MUG、CSCD-NS等）+ 大模型合成（Qwen-Max、Gemini Pro、GPT-4 Turbo）+ 人工采集。引入6名>5年移动端AI经验领域专家进行双层校验，按“场景贴合度、安全性、隐私风险、社会争议性、指令遵循质量”5项维度5分制打分（阈值≥3.5），将初始30k数据压缩至2973条高质量样本。
- **评估协议（Evaluation Protocol）**：采用LLM-as-a-Judge范式，每道题满分10分。裁判模型为GPT-4 Turbo与Qwen-Max（交叉验证排名一致性）。评测Prompt除提供标准答案外，还嵌入分项评分rubric（如TextContinuation考查连贯性、语言质量、创意、原文一致性四维度），裁判先分项打分再输出总分。
- **量化部署实验设置**：使用Qualcomm QNN SDK将BlueLM-3B与Qwen2.5-3B量化为W4A16，部署于iQOO 12 NPU；单任务取50题进行推理测试，记录准确率保留率、Prefilling速度、输出Token速度与功耗。

## 实验与结果
- **BF16基线评测**：在SmartBench整体评测中，端侧模型BlueLM-3B以平均**7.03分**位居第一，略低于云端参考模型GPT-4o的**7.37分**。Text Q&A与Text Summarization表现优异（DocumentQ&A接近9.0+）；TextCorrection、RelationExtraction、NotificationSorting为薄弱环节（多数端侧模型得分仅2~4分，TextCorrection仅为GPT-4o的一半左右）。
- **多模态权衡现象**：InternVL2.5-4B（基于Qwen2.5-3B扩展）虽具备视觉能力，但在纯语言主观任务上得分显著下降，说明多模态对齐训练可能稀释纯文本指令遵循能力。
- **NPU量化部署**：W4A16量化后整体平均性能保留率约**90%**（BlueLM-3B保留91.31%，Qwen2.5-3B保留90.58%）。但TextCorrection等高精度任务保留率仅63%~70%，且出现流畅度退化与理解力下降的Failure Cases。推理速度约25 token/s，功耗6.4~6.8W。
- **人工对齐验证**：SmartBench定制化评分Prompt与人类专家排名的Pearson相关系数达**0.8280**，显著高于MT-Bench通用Prompt的**0.7959**，证明任务细化评分设计的有效性。

## 相关工作脉络
1. **端侧大模型（On-device LLMs）**：OpenELM、MiLM、BlueLM、Magic LM等工业界端侧模型陆续发布，但缺乏针对其实际手机交互能力的标准化评测体系；本文填补该空白。
2. **通用LLM评测基准**：MMLU、GSM8K、HumanEval等侧重客观知识与逻辑推理，与手机端“轻量润色、摘要、通知管理”等主观实用场景存在明显错位。
3. **主观/对话类基准**：AlignBench（中文）、WildBench、Chatbot Arena等关注通用对话与指令遵循，但未针对移动端单步功能调用与中文本地化生活语境做专项设计。
4. **中文语言评测基准**：CMRC、CLUE、SuperCLUE、C-Eval主要考查阅读理解与学科知识，主观对齐维度薄弱，无法反映端侧模型的真实用户体验。
5. **移动端智能体基准**：AitZ、AndroidWorld、AppAgent等聚焦多步GUI操作与API调用轨迹规划；本文明确界定边界，专注单次请求即完成的文本生成/抽取/摘要功能，二者互补而非替代。

## 局限性与未来方向
1. **功能覆盖面时效限制**：当前调研仅截至2024年12月，端侧LLM功能迭代迅速，需持续更新任务与数据。
2. **语言单一性**：专为中文用户设计，不同国家/地区的手机使用习惯差异未覆盖，未来计划拓展多语言版本。
3. **模态局限**：当前仅含纯文本模态，实际手机应用广泛涉及相机输入、语音识别与音频生成；后续版本将引入视觉与听觉模态。

## 研究启发与可借鉴点
1. **垂直领域基准构建范式可迁移**：“开源筛选+大模型合成+人类专家双层校验”的数据流水线，以及按任务定制评分Rubric的设计思路，可直接复用于车载语音助手、可穿戴设备、智能家居等垂直端侧场景评测。
2. **主观评测提示词工程的价值**：本文证明针对细分任务设计多维评分维度能显著提升LLM-as-a-Judge与人类判断的一致性，为后续主观对齐评测提供了可操作的Prompt设计规范。
3. **量化部署画像可作为模型选型依据**：W4A16在不同任务上的保留率差异揭示了“高精度要求型任务（如纠错、排序）对量化极度敏感”的规律，可为端侧模型训练时的损失加权与后量化训练（QAT）提供目标导向。
4. **多模态训练的副作用警示**：InternVL2.5-4B在融合视觉能力后纯语言主观得分下降的现象，提示团队在开发端侧多模态模型时，需在数据配比与对齐阶段加强基础文本指令遵循能力的正则化保护。

## 关键术语表
- **SmartBench**：首个面向中文手机端大模型的综合能力评测基准，覆盖5类20项日常移动交互任务。
- **LLM-as-a-Judge**：利用大语言模型作为自动化裁判，依据预设评分维度对模型输出进行主观质量打分的方法。
- **On-device LLM**：部署于智能手机等边缘设备本地运行的大语言模型，依赖NPU/SoC算力，强调低延迟与隐私保护。
- **W4A16量化**：权重采用4bit整数、激活值保留16bit浮点的混合精度量化方案，在精度与推理效率间取得平衡。
- **NPU（Neural Processing Unit）**：专为神经网络推理设计的片上系统加速器，现代智能手机普遍集成以提升端侧AI性能。
- **场景鸿沟（Scenario Gap）**：现有基准任务分布与实际端侧产品高频调用场景严重脱节的现象。

## 可复现要素
- **数据集**：SmartBench，共2973条QA（5大类20任务），部分源自公开开源数据集，部分由LLM合成+人工精修；代码与数据将开源：https://github.com/vivo-ai-lab/SmartBench
- **模型权重**：BlueLM-3B、InternVL2.5-4B、MiniCPM3-4B、Qwen2.5-3B、Qwen2-VL-2B 均为开源可复现模型。
- **硬件平台**：vivo iQOO 12（Qualcomm Snapdragon 8 Gen 3 SoC + NPU）
- **关键超参**：推理精度BF16 / W4A16；上下文长度2048；NPU实测每任务取50题；量化SDK为Qualcomm QNN SDK；裁判模型为GPT-4 Turbo与Qwen-Max。
