---
title: "On-Relation-Specific-Neurons-in-Large-Language-Models"
source: https://aclanthology.org/2025.emnlp-main.52.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:32:30"
field: "大语言模型机制可解释性"
keywords: ["relation-specific neurons", "mechanistic interpretability", "knowledge neurons", "LLM interpretability", "factual knowledge", "neuron deactivation", "multilingual neurons"]
innovations: ["系统识别并验证关系特定神经元（RelSpec neurons）的存在", "提出累加性、多功能性、干扰性三大神经元性质", "设计实体解耦实验范式分离关系与实体效应"]
benchmarks: ["LRE factual knowledge dataset", "Hernandez et al. (2024) relational facts"]
---

# 论文速读：On-Relation-Specific-Neurons-in-Large-Language-Models

## 一句话总结
本文通过统计学方法在 Llama-2（7B/13B）中系统性地识别并验证了"关系特定神经元"（RelSpec neurons）的存在——即专注编码关系本身而非具体实体或事实的神经元，并揭示了其累加性、多功能性和干扰性三大核心性质。

## 研究问题与动机
- **核心问题**：LLM 中存储的事实知识通常表示为三元组（主体、关系、客体），但不清楚是否存在专注于"关系"本身（与实体无关）的神经元，还是仅存在编码具体事实的"知识神经元"或"实体神经元"。
- **现有方法不足**：既往对知识神经元的探索多聚焦于具体事实的编码（如 Dai et al., 2022 的知识神经元），缺乏对关系层面独立编码机制的系统研究；且现有方法难以剥离实体效应与关系效应。
- **动机**：若关系特定神经元存在，将深化对 LLM 如何表征和调用关系知识的理解，并为可解释性、事实编辑和跨语言泛化提供新视角。

## 核心贡献（创新点）
1. **首次系统性地识别并验证 RelSpec 神经元**：采用 Cuadros et al. (2022) 的统计关联方法，结合零样本提示设计，从 Llama-2 7B/13B 中定位出专注于关系而非实体的神经元。
2. **提出并验证 RelSpec 神经元三大性质**：累加性（多个神经元协同处理关系）、多功能性（神经元跨关系及跨语言共享）、干扰性（一个关系的神经元可能干扰其他关系处理），为神经元功能分析提供了系统性框架。
3. **设计了实体解耦的实验范式**：通过确保评估集（$\mathcal{P}^{eva}$）与检测集（$\mathcal{P}^{det}$）之间无主体实体重叠，有效分离了实体效应与关系效应，使结论更具因果说服力。
4. **揭示了关系与概念神经元的关联与区分**：通过对比关系神经元和概念神经元（§5.4），发现关系重叠多源于共享概念（如"地点"概念），但两者表征 largely 独立，深化了对知识存储结构的理解。
5. **开源代码与数据**：所有代码和数据已公开发布（https://github.com/cisnlp/relation-specific-neurons），便于后续研究复现与扩展。

## 方法详解
- **数据集构建**：使用 Hernandez et al. (2024) 的事实知识数据集（25 个关系），选取事实数 >300 的 12 个关系。每个关系 $r_i$ 分为检测集 $\mathcal{D}^{det}_{r_i}$ 和评估集 $\mathcal{D}^{eva}_{r_i}$（随机选 50 个三元组），确保两者无主体实体重叠。
- **提示模板**：对每个三元组 $(s, r_i, o)$，构造含主体 s 和关系 $r_i$ 的零样本提示（不含客体 o），如"The CEO of NVIDIA is? Answer:"，期望回答"Jensen Huang"。
- **提示验证**：将检测集提示输入模型，生成 2 个 token，若为客体的前缀则视为正确，排除模型答错的提示。
- **神经元定义**：将 FFN（up_proj、gate_proj、down_proj）的每列定义为"神经元"，FFN 神经元占 7B 模型 835,584 个、13B 模型 1,310,720 个。
- **正负样本分组**：对关系 $r_i$，将 $\mathcal{P}^{det}_{r_i}$ 作为正样本，从其他关系提示中随机采样 $4 \times |\mathcal{P}^{det}_{r_i}|$ 个作为负样本，构成 $\mathcal{E}_{r_i} = \mathcal{E}^+_{r_i} \cup \mathcal{E}^-_{r_i}$。
- **神经元输出聚合**：对示例 $e^j_{r_i}$ 中第 t 个 token 的神经元 m 输出 $o^j_{r_i,t}$ 取平均：$\hat{o}^j_{r_i} = \frac{1}{T}\sum_{t=1}^T o^j_{r_i,t}$。
- **专家度计算（AP 指标）**：将神经元输出值作为预测分数、二元标签 $b^j_{r_i}$（正样本为 1，负样本为 0）作为真值，遍历所有阈值计算平均精确率（Average Precision, AP），取 AP 最高的 top-k（本文 k=3000）作为 RelSpec 神经元。
- **控制生成实验**：在推理时将 top-k RelSpec 神经元的输出值强制置 0（deactivation），在评估集 $\mathcal{P}^{eva}_{r_i}$ 上测量准确率下降，比较 intra-relation（同关系）和 inter-relation（跨关系）效果。跨关系影响度量：$acc\_drop_{r_i,r_j} = \frac{acc^{original}_{r_i} - acc^{deactivated\text{-}r_j}_{r_i}}{acc^{original}_{r_i}}$。

## 实验与结果
- **数据集与基线**：12 个关系（company_ceo、company_hq、landmark_continent、landmark_country、person_father、person_mother、person_occupation、person_plays_instrument、person_pro_sport、person_sport_position、product_company、star_constellation），在 Llama-2 7B 和 13B 上测试，随机停用 3000 个神经元作为基线。
- **层分布**：RelSpec 神经元主要分布在中间层（区别于语言特定神经元集中在首尾层）。
- **神经元重叠**：person_mother 与 person_father 因主体实体（名人）高度重叠而共享大量神经元；其他关系即使无实体重叠仍存在共享（如 person_occupation 与 person_sport_position 共享 297 个神经元）。
- **Intra-relation 结果**：停用 3000 个 RelSpec 神经元后，在 $\mathcal{P}^{det}$ 和 $\mathcal{P}^{eva}$ 上均出现显著准确率下降（图 3），而随机停用无显著差异；准确率未降至 0（除 13B 的 landmark_country），表明关系知识分散编码。
- **Inter-relation 结果**：停用某一关系的神经元可同时影响closely related（如 person_pro_sport → person_sport_position）和 loosely related 关系（如 star_constellation → landmark_continent，均涉及"location"抽象概念）。
- **神经元干扰**：7B 模型中停用某些关系的神经元反而提升其他关系准确率（如 person_mother 在停用 5/11 其他关系神经元后提升），13B 中现象不同（如停用 landmark_country 提升 landmark_continent）。
- **神经元数量敏感性**（图 5）：停用 3000-10000 个神经元前对其他关系影响有限，超过此阈值后开始显著影响；即使停用 50000 个神经元，其他关系准确率通常不接近 0（company_hq 除外）。
- **累加性验证**（图 6）：随停用范围扩大，累加性增强；仅停用中间差值神经元不足以导致错误，支持"多神经元协同"假说。
- **多语言性**（图 7）：用英语识别的 3000 个 RelSpec 神经元起用后，在德语、西班牙语、法语、中文、日语 5 种语言上均出现准确率下降，证实跨语言泛化。
- **提示模板鲁棒性**（图 8）：使用不同提示模板的评估集 $\mathcal{P}^{eva-2}$ 上，停用神经元起到一致的准确率下降，排除模板混淆。
- **关系 vs. 概念**（图 9）：关系神经元与概念神经元重叠主要来自共享概念（如 company_ceo 与 company 概念），但大部分神经元仅属于关系或概念之一。
- **通用语言建模影响**（表 2）：停用 RelSpec 神经元后，在无关上下文中生成客体 token 的 perplexity 无系统性恶化，部分关系甚至略有下降，表明不影响通用语言能力。
- **事实频率效应**（§E）：在 Dolma 语料中高频出现的事实更抗停用（resilient facts 频率高于 sensitive facts，13B 模型在 5% 水平显著）。
- **最强结果**：在 12 个关系上均观察到显著的 intra-relation 准确率下降（图 3），跨关系干扰和干扰性效应在 7B/13B 中均有体现，且结果在 Gemma-7B 上得到复现（§C）。

## 相关工作脉络
- **知识神经元（Dai et al., 2022）**：聚焦编码具体事实的神经元，本文工作与之互补——区分了"知识神经元"（编码具体事实）与"关系神经元"（编码关系抽象），揭示关系层面的独立编码机制。
- **语言特定神经元（Kojima et al., 2024）**：采用相似的统计方法识别语言特定神经元，本文将其推广至关系层面；关键区别在于语言神经元集中在首尾层，而 RelSpec 神经元集中在中间层。
- **自我条件预训练语言模型（Cuadros et al., 2022）**：提出统计关联方法识别神经元专家度（AP 指标），本文直接沿用该方法并扩展至关系识别。
- **事实编辑与记忆定位（Meng et al., 2022, 2023）**：通过修改神经元/权重编辑事实知识，本文从神经元功能分析角度提供补充视角，有助于更精准定位编辑目标。
- **探针方法（Gurnee et al., 2023, 2024）**：训练分类器探测神经元关联的概念，本文方法无需监督标注，更具可扩展性；但探针方法可能捕获更丰富的概念信息。
- **机制可解释性（Mechanistic Interpretability）**：属神经元层面特征分析，与电路分析（Wang et al., 2023; Elhage et al., 2021）互补——本文定位功能单元，电路分析研究其组合逻辑。

## 局限性与未来方向
- **关系覆盖有限**：仅研究 12 个关系（受限于方法可靠性要求每关系 >300 事实），未来可扩展至更广泛、更多样的关系类型。
- **多语言范围有限**：仅测试 5 种语言（德、西、法、中、日），未涵盖低资源语言，未来需验证跨语言泛化的普遍性。
- **模型范围有限**：主要在 Llama-2 族上验证，虽有 Gemma-7B 补充（§C），但未探索更大模型或指令微调模型的 RelSpec 神经元行为。
- **事实频率近似误差**：使用 Dolma 语料近似预训练频率，与 Llama-2 实际预训练数据可能存在偏差，影响频率效应的精确评估。
- **神经元选择阈值需调优**：top-3000 的选取虽有实验依据（§5.1），但未系统探索最优阈值及不同阈值下的性质变化。
- **未深入分析错误类型机制**：虽观察到停用后模型产生无意义输出（表 4-7），但对错误生成机制的细粒度分析有待深入。

## 研究启发与可借鉴点
- **实体解耦实验设计**：通过确保评估集与检测集无实体重叠来分离实体效应与关系效应，这一范式可迁移至其他知识类型（如时间、位置）的神经元研究中。
- **统计关联方法的通用性**：AP 指标识别神经元专家度的方法简单高效，可扩展至任务特定神经元、语义角色等其他语言学概念的探索。
- **多语言验证框架**：用单一语言识别神经元并在多语言上验证功能，为跨语言可解释性研究提供了可复用的实验流程。
- **累积效应验证设计**：通过比较相邻停用范围的差异神经元是否足以导致错误，来验证累积性而非孤立效应，这一实验设计可用于检验其他神经元的协同机制。
- **关系-概念分离分析**：通过对比关系神经元和概念神经元，揭示知识存储的结构化特征，可启发对"属性""事件"等其他知识类型的类似分析。

## 关键术语表
- **Relation-Specific Neurons (RelSpec neurons)**：专注于编码关系语义、与实体无关的神经元，区别于编码具体事实的知识神经元或编码主体的实体神经元。
- **Neuron Cumulativity**：关系知识分散编码于多个神经元，单个神经元无法独立编码完整关系信息，需多神经元协同。
- **Neuron Versatility**：神经元可在多个相关或不相关关系间共享，且识别出的英语神经元在其他语言中同样有效。
- **Neuron Interference**：一个关系的神经元可能干扰另一关系的处理，停用后者神经元有时可提升前者的准确率。
- **Average Precision (AP)**：以神经元输出值为预测分数、二元标签为真值，遍历阈值计算的精确率-召回率曲线下的面积，用于衡量神经元专家度。
- **Intra-relation / Inter-relation**：前者指停用某关系神经元后评估该关系本身的表现；后者指停用某关系神经元后评估其他关系的表现。
- **Fact Resilience**：停用 RelSpec 神经元后仍能正确回答的事实，研究发现其与预训练数据中的事实频率正相关。
- **Controlled Generation (Neuron Deactivation)**：推理时将特定神经元的输出值强制置 0，以观察其对模型行为的因果影响。

## 可复现要素
- **数据集**：Hernandez et al. (2024) 的事实知识数据集（LRE），论文未声明是否公开，但作者已公开代码和数据处理脚本。
- **代码与数据**：已开源，地址 https://github.com/cisnlp/relation-specific-neurons。
- **模型**：Llama-2 7B 和 13B（Touvron et al., 2023），Gemma-7B 补充实验。
- **关键超参**：top-k 神经元数量 k=3000；负样本采样比例 4× 正样本数；最大生成长度 2 tokens；验证正确性的阈值（预测 2 tokens 为客体前缀）。
- **硬件环境**：NVIDIA RTX A6000 GPU。
- **语言**：英语（主实验）及德语、西班牙语、法语、中文、日语（多语言实验）。
