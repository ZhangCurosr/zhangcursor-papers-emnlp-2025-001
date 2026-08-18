---
title: "LLMs-as-World-Models-Data-Driven-and-Human-Centered-Pre-Even"
source: https://aclanthology.org/2025.emnlp-main.153.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:45:00"
field: "灾害信息学与多模态大模型"
keywords: ["LLM as World Model", "地震灾害模拟", "多模态推理", "MMI 预测", "震前评估", "RAG", "ICL", "灾害风险管理"]
innovations: ["提出将LLM作为虚拟传感器进行震前人类感知模拟的多模态框架", "构建首个开放的多模态LLM地震模拟评测基准（整合地震/地理/建筑/社经/街景数据与DYFI标签）", "揭示街景视觉输入对MMI预测的关键提升作用及RMSE与相关系数的不一致性"]
benchmarks: ["USGS DYFI (2014 Napa M6.0, 2019 Ridgecrest M7.1)", "USGS ShakeMap", "OpenStreetMap Building Data", "American Community Survey (ACS)"]
---

# 论文速读：LLMs-as-World-Models-Data-Driven-and-Human-Centered-Pre-Event-Simulation-for-Disaster-Impact-Assessment

## 一句话总结
本文提出将大语言模型（LLM）作为"虚拟传感器"和世界模型，融合地震参数、地理空间、建筑、社会经济及街景图像等多模态数据，在震前模拟人类对地震震感的感知；在 2014 年 Napa（M6.0）和 2019 年 Ridgecrest（M7.1）地震上验证，与 USGS DYFI 报告相比在 zip code 级别实现了最高相关系数 0.88、RMSE 0.77 的预测精度。

## 研究问题与动机
- **核心问题**：能否利用 LLM 作为世界模型，在突发地震发生前模拟人类对地震风险的感知（即以 Modified Mercalli Intensity, MMI 为输出）？
- **现有方法的不足**：
  - 当前地震灾害评估方法多为**震后响应式**（专家勘察、地面传感器、遥感），无法支撑震前规划与主动防灾。
  - 传统震前模拟（如情景规划、物理仿真、GMPEs）依赖大量领域专业知识与区域定制模型，且**缺乏以人类感知为中心的数据验证**。
  - LLM 在灾害管理中已有应用（震后损伤检测、社交媒体分析），但**尚未用于震前情景模拟**，且缺乏融合领域知识的推理框架。
  - 现有方法普遍缺少**人类中心维度**——即不仅模拟物理震动，还模拟社区可感知的震害影响。

## 核心贡献（创新点）
- **构建首个开放的多模态 LLM-as-World-Model 地震模拟评测基准**：整合地震参数、VS30、OSM 建筑数据、ACS 社会经济数据与 Google Street View 图像，提供 zip code/county 两级 MMI 标签（来自 USGS DYFI），填补该领域开放资源的空白。
- **提出"虚拟传感器"多模态 LLM 仿真框架**：将 LLM 视为可"感知"多模态特征并推理 MMI 等级的虚拟传感器，通过角色设定 + Chain-of-Thought 输出 JSON 格式的推理链与 MMI 评级，实现从震后评估到震前模拟的范式迁移。
- **系统评测 9 个开源/闭源 LLM 在不同提示策略（Vanilla / ICL / RAG）下的性能**：覆盖 GPT-4o、GPT-4.1-mini、Claude-3.5-haiku、Llama-3.2-11B/90B-VL、Qwen2.5-VL-3B/7B/32B/72B，提供跨模型、跨尺度、跨提示技术的全面基线。
- **揭示模态对齐的关键发现**：街景视觉输入显著提升 MMI 预测精度，而移除结构化数值特征（地理、建筑、社会经济）反而导致性能下降；同时指出 RMSE 与相关系数之间可能存在的**不一致性**（排序能力 vs 绝对误差）。
- **输出可解释推理分析**：通过 TF-IDF 词汇分析揭示 GPT-4.1-mini 与 Qwen2.5-32B 在地震衰减、场地效应、建筑特征和社会经济因素上的不同语言线索与推理风格，为 LLM-as-World-Model 的可解释性研究提供实证依据。

## 方法详解
- **框架设计**：将每个采样点 $p_{ji}$ 关联一个融合特征集合 $\mathcal{X}_i = \{E_i, G_i, L_i, B_i, S_i, V_i\}$，其中 $E_i$ 为地震参数（矩震级、距震中距离、震源深度），$G_i$ 为地理空间特征（VS30 浅层剪切波速），$L_i$ 为位置元数据（州、城市、邮编、经纬度），$B_i$ 为建筑属性（建筑数量、类型、高度、主要建材），$S_i$ 为社会经济指标（人口密度、老龄化比例、收入、教育水平），$V_i$ 为 Google Street View 街景图像。LLM 作为推理函数 $f_\theta$，输出推理链 $e_i$ 和 MMI 预测 $\hat{y}_i \in \{\text{I}, \text{II}, \dots, \text{XII}\}$。
- **空间采样策略**：基于多边形 GIS shapefile 定义行政区（zip code），在每个 polygon 内执行**分层随机采样**，每邮编区抽取 50 个点，保证空间代表性并缓解城乡/人口密度偏差。
- **数据融合**：地震参数与 VS30 来自 USGS ShakeMap 与 USGS VS30 数据集；建筑特征来自 OpenStreetMap（100m 半径内）；社会经济数据来自美国社区调查（ACS）Census Block Group 级别；街景图像通过 Google Maps API 获取。
- **提示工程**：采用角色设定（ seismic specialist）+ MMI 量表描述 + 六段式特征输入 + Chain-of-Thought 推理指令，要求 LLM 以 JSON 格式输出 reasoning 和 MMI 等级。
- **提示技术**：
  - **ICL（In-Context Learning）**：在 prompt 中嵌入详细的 MMI 参考指南作为示例，增强模型对任务的适配。
  - **RAG（Retrieval-Augmented Generation）**：在 prompt 中提供多模态特征与已报告的 MMI 值作为检索上下文，为预测提供事实 grounding。
- **评估指标**：
  - 点级预测按区域聚合：$\overline{\hat{y}}_j = \frac{1}{n_j} \sum_{i=1}^{n_j} \hat{y}_{ji}$
  - **RMSE**：$\mathrm{RMSE} = \sqrt{\frac{1}{N} \sum_{j=1}^{N} (\overline{\hat{y}}_j - \overline{y}_j)^2}$，衡量绝对误差。
  - **Pearson 相关系数 r**：衡量预测与真实 MMI 的相对排序一致性。
  - 在 zip code 和 county 两个空间尺度上分别计算。

## 实验与结果
- **数据集**：2014 年 California Napa 地震（M6.0）和 2019 年 California Ridgecrest 地震（M7.1）；地面真值来自 USGS "Did You Feel It?"（DYFI）报告；每个事件选取响应最多的 Top 100 zip code，每邮编采样 50 点，约 5,000 样本/事件（Napa 因街景缺失仅 4,920）。
- **评测基线**：
  - 9 个 LLM（GPT-4o、GPT-4.1-mini、Claude-3.5-haiku、Llama-3.2-11B/90B-VL、Qwen2.5-VL-3B/7B/32B/72B）
  - USGS ShakeMap（震后产品）
  - 6 个传统 ML 模型（Logistic Regression Lasso/Ridge、MLP、Random Forest、SVM、XGBoost）
- **主要结果**（zip code 级别）：
  - **最强结果**：Ridgecrest 事件中 GPT-4.1-mini 达到 RMSE = 0.92、相关系数 = 0.64；Napa 事件中 Qwen2.5-VL-32B 达到 RMSE = 1.59、相关系数 = 0.70。按摘要 reported：最佳 zip code 级别相关系数达 **0.88**、RMSE 为 **0.77**（应指 Ridgecrest county 级别 GPT-4o 的结果：RMSEc=0.77, Corrc=0.88）。
  - 闭源 LLM 在 8 组评测中的 6 组优于开源模型；Qwen2.5-VL-32B 是开源模型中表现最佳者。
  - **ICL 和 RAG 均能显著降低 RMSE**，且少量示例即可带来可观提升。
  - 街景图像输入**唯一**能提升 zip code 级别 RMSE 的特征模态；移除 GEO/Building/Socioeconomic 任一特征均导致性能下降。
  - 传统 ML 最优模型（XGBoost，Ridgecrest zip code RMSE=1.52，Corr=0.57）不及最佳 LLM（GPT-4.1-mini RMSE=0.92，Corr=0.64）。
  - ShakeMap（震后）在 Napa（RMSE=1.08, Corr=0.81）和 Ridgecrest（RMSE=0.72, Corr=0.79）均优于所有震前 LLM 模拟。
- **关键结论**：LLM 在震前模拟中展现出与真实人类感知报告高度对齐的潜力，尤其是在相对排序（相关系数）上表现突出；多模态融合与推理增强技术（ICL/RAG）是提升性能的关键手段。

## 相关工作脉络
- **地震灾害模拟（物理/数据驱动）**：传统 GMPEs（Moschetti et al., 2024）和物理仿真（Deierlein et al., 2020）擅长刻画局部场地效应和破裂动态，但依赖大量数据和计算资源；数据驱动的 ML 方法（Cardellicchio et al., 2023）可扩展性好但需要高质量标注数据且可解释性弱。本文定位差异：以 LLM 替代传统建模，强调**人类感知维度**而非纯物理参数。
- **USGS DYFI 系统**（Atkinson & Wald, 2007）： crowdsourced MMI 收集平台，本文以其作为人类感知的 ground truth，但将其应用从**震后验证**拓展至**震前模拟对标**。
- **LLM as World Models**（Hao et al., 2023; Yan et al., 2024 OpenCity）：将 LLM 用于复杂场景仿真和大规模智能体模拟；本文将其应用于**灾害风险的人类中心感知模拟**，聚焦突发地震这一新的应用场景。
- **LLM 在灾害管理中的应用**（Zhang & Wang, 2024; Wang et al., 2024a; Otal et al., 2024）：主要集中在震后损伤检测、社交媒体分析和应急管理；本文填补了**震前模拟**的研究空白。
- **多模态 LLM 视觉推理**（Xiang et al., 2023; Li et al., 2025c）：证明了 LLM 在视觉环境理解方面的能力；本文利用街景图像赋能 LLM 的空间-环境推理，验证了**视觉模态对灾害模拟的关键价值**。
- **RAG 与 ICL 技术**（Lewis et al., 2020; Dong et al., 2024）：将检索增强和上下文学习引入灾害模拟，展示其在提高 LLM 推理准确性和可靠性方面的有效性。

## 局限性与未来方向
- **案例范围有限**：仅在两个美国加州地震上验证，未涵盖全球不同地震带、城市密度和建筑规范，**泛化能力需进一步验证**。
- **数据可用性偏差**：Google Street View 覆盖不完整、ACS/OSM 数据存在空白，可能导致特定社区代表性不足。
- **未做细粒度特征选择**：未深入分析单个参数（如房屋年代、基础设施距离）的影响，限制了可解释性深度。
- **国际适用性受限**：依赖 USGS DYFI 报告作为 ground truth，其他国家/地区缺乏同类公开感知数据，需寻找替代验证源。
- **未来方向**：扩展到更多地震事件和全球区域；探索更高效的提示策略和推理结构；深入研究 LLM 作为世界模型的内部推理机制；将框架集成到早期预警系统中以识别脆弱社区。

## 研究启发与可借鉴点
- **"虚拟传感器"框架设计可迁移**：将 LLM 视为具有多模态感知能力的"虚拟传感器"并通过 role-playing + CoT 输出结构化推理，可复用于洪水、飓风等其他灾害类型的震前/灾前模拟。
- **多模态输入中视觉的重要性**：街景图像是唯一提升精度的模态，这提示在类似的空间推理任务中，应优先保障视觉数据的接入；结构化数值数据可能对 LLM 造成干扰，值得在提示工程中做模态权重优化。
- **ICL/RAG 增强策略的实用价值**：少量示例即可显著提升性能，为低资源场景下的 LLM 灾害应用提供了低成本增效方案；RAG 可提供事实 grounding，减少幻觉风险。
- **RMSE 与相关系数的双指标评估范式**：揭示了排序能力与绝对精度之间的潜在不一致，为后续研究提供了更全面的评估视角，避免单一指标的误判。
- **跨模型对比与 Scaling Law 分析**：覆盖 9 个模型的系统评测和参数规模分析（Appendix F）为团队后续工作提供了可复用的基准对比框架。

## 关键术语表
- **LLM as World Model**：将大语言模型视为能够学习并模拟现实世界时空、因果关系的"世界模型"，用于前向推理预测复杂场景的未来状态。
- **Modified Mercalli Intensity (MMI)**：修改麦加利烈度表，一种基于人类感知和物体反应的 12 级地震震害烈度评定标准（I–XII），区别于基于能量释放的震级。
- **Did You Feel It? (DYFI)**：USGS 运营众包地震感知报告平台，收集公众报告后聚合生成各行政区 MMI 等值图，作为人类感知震害的 ground truth。
- **VS30**：地表以下 30 米深度处剪切波的平均传播速度（m/s），广泛用于表征场地土质对地震波的放大或衰减效应。
- **In-Context Learning (ICL)**：通过在输入 prompt 中嵌入任务示例，使 LLM 在不更新参数的情况下适应特定任务风格的推理方法。
- **Retrieval-Augmented Generation (RAG)**：在 LLM 生成过程中引入外部知识检索，将相关上下文信息注入 prompt 以增强生成的准确性和事实 grounding。
- **Chain-of-Thought (CoT)**：引导 LLM 在给出最终答案之前先输出逐步推理过程，以提高复杂任务中的逻辑一致性和可解释性。
- **ShakeMap**：USGS 震后快速生成的地面运动强度地图，基于实时地震台网数据在数分钟内发布，用于表征地震动的空间分布。

## 可复现要素
- **数据集**：地震参数与 DYFI 来自 USGS（公开）；VS30 来自 USGS（公开）；建筑数据来自 OpenStreetMap（公开）；社会经济数据来自 ACS（公开）；街景图像通过 Google Maps API（需 API key）。论文提供数据访问链接：https://doi.org/10.5281/zenodo.17148713
- **代码**：已开源，地址 https://github.com/Lingyao1219/llm-disaster-simulation
- **关键超参**：每邮编采样 50 个点；Top 100 邮编区；prompt 包含 6 段特征输入 + MMI 量表描述 + CoT 指令；输出格式为 JSON（reasoning + MMI）
- **评测模型**：GPT-4o、GPT-4.1-mini、Claude-3.5-haiku、Llama-3.2-11B-VL、Llama-3.2-90B-VL、Qwen2.5-VL-3B、Qwen2.5-VL-7B、Qwen2.5-VL-32B、Qwen2.5-VL-72B
- **提示策略**：Vanilla、ICL（嵌入 MMI 参考指南）、RAG（嵌入已知 MMI 报告作为上下文）
- **评估尺度**：zip code 级别和 county 级别，分别计算 RMSE 和 Pearson 相关系数
