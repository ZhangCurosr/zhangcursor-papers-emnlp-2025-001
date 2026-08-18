---
title: "Biased-Tales-Cultural-and-Topic-Bias-in-Generating-Children"
source: https://aclanthology.org/2025.emnlp-main.3.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:43:31"
---

# 论文速读：Biased-Tales-Cultural-and-Topic-Bias-in-Generating-Children

## 一句话总结
本文构建了首个面向儿童睡前故事的社会文化偏见评测数据集 **Biased Tales**（5,531 篇），系统量化了 GPT-4o、Llama3-8B 与 Mixtral8x 在性别、国籍、种族与宗教等维度上的刻板印象，揭示了非西方儿童故事被过度强调“传统与家庭”、女孩外貌描写比男孩多 55.26% 等隐性偏见。

## 研究问题与动机
- **儿童向 LLM 生成内容的偏见盲区**：家长日益依赖 LLM 定制睡前故事，但现有偏见研究多聚焦成人叙事或通用文本，缺乏针对儿童内容社会文化属性的系统性评估。
- **隐性偏见的难察觉性**：LLM 生成的儿童故事通常无毒、适龄，但刻板印象以词汇分布、角色刻画与场景设定的形式隐性嵌入，非专业家长难以识别。
- **个性化与公平性的张力**：现有个性化故事生成框架（如 MirrorStories）强调兴趣匹配，但未考察社会文化提示如何被模型转化为强化传统性别/文化角色的叙事输出。
- **评测维度单一**：既往工作多依赖词频统计或人工评分，缺乏从表层词汇到篇章级可预测性的跨粒度偏见量化方法。

## 核心贡献（创新点）
- **提出双维度叙事偏见评估框架**：将故事拆解为角色中心（Physical/Emotional/Mental/Moral/Other）与上下文中心（Geographic/Urban/Socioeconomic）两类属性，与现有仅关注词级共现或情感极性的工作相比，首次建立适配儿童故事的结构化标注体系。
- **发布 Biased Tales 数据集**：收录 5,531 篇由三大主流 LLM 生成的故事，覆盖 28 个国籍、6 种族裔、6 种宗教与多组家长/儿童身份组合；区别于 MirrorStories 等仅引入兴趣/偏好的数据集，本文首次将家长角色与文化背景同时作为生成条件。
- **设计表层与隐式双重偏见度量**：结合 Pearson 词汇相关性分析与 TF-IDF + 前馈神经网络的 5-fold 可预测性评估，实现从词级共现到篇章级隐含信号的交叉验证，优于单一依赖人工标注或简单分类器的方法。
- **开源非技术友好的交互可视化工具**：提供面向家长的数据浏览器，支持按社会文化因子检索故事并高亮偏见元数据，填补自动化评测与公众可及性之间的鸿沟。

## 方法详解
- **数据生成流水线**：设计 4 类提示模板（Table 1），分别控制 `parent role + child gender`、`parent nationality + role`、`parent ethnicity + role`、`parent religion + role`。使用 `temperature=1`、`max tokens=1024`，每提示生成 5 次，最终经语言检测剔除 4 篇非英语故事，保留 5,531 篇。
- **混合标注策略**：随机抽取 1,000 篇由两名母语标注员（意大利语/罗马尼亚语，C1 英语）人工标注；其余由 GPT-4o 自动提取（提示词见 Appendix Table 7）。使用 sentence embedding 余弦相似度校验一致性（人工-人工 84.52，人工-GPT 75.49）。
- **属性分类体系**：
  - **角色中心**：基于 Stereotype Content Model 与 ABC Model 扩展为五类：Physical（外貌/体态）、Emotional（情绪/感受）、Mental（认知/智力）、Moral（道德/品格）、Other（特殊能力/抽象特质）。
  - **上下文中心**：Geographic location（Desert/Green Bodies/Mountain/Water Bodies/Magical/None）、Urban setting（City/Town/Village/None）、Socioeconomic（Poor/Middle-class/Wealthy/None）。
- **偏见度量方法**：
  - **表层分析**：计算各社会文化因子与故事文本/角色属性词频的 Pearson 相关系数，提取 Top 关键词。
  - **隐式分析**：移除显式标签词（如 girl/boy）后，使用 TF-IDF 向量化，训练全连接神经网络进行 5-fold 交叉验证，预测准确率反映文本中隐含偏见的强度。
  - **辅助指标**：年龄习得均值（AoA）与 Flesch-Kincaid 阅读难度（FKRE）评估适龄性；Perspective API 检测毒性；all-MiniLM-L6-v2 句子内积相似度评估故事多样性。

## 实验与结果
- **数据集规模与覆盖**：Biased Tales 含 5,531 篇故事；性别 3 类、家长角色 3 类、宗教 6 类、族裔 6 类、国籍 28 类。
- **基线对比**：与 MirrorStories 对比显示，后者仅 17.20% 故事满足儿童 FKRE 适龄标准，本文数据集平均 FKRE 达 75.5，AoA 为 5.86，且平均毒性得分仅 0.06。
- **关键定量结果**：
  - **性别偏见**：女孩主角的外貌相关属性比男孩多 **55.26%**；女孩高频属性词为 `hair, gentle, imaginative, loving`，男孩为 `young, adventurous, hero, brave`。
  - **文化/地域偏见**：非洲/中东故事高频词为 `desert, vast, horizon, animal`；亚洲故事关联 `forest, dragon, village`；欧洲/北美故事偏向 `forest, sparkling, tree, magic`。
  - **种族刻板印象**：White 族裔与魔法场景强关联（74.07%）；Middle-Eastern 族裔 87% 关联 `desert`，54% African-American 关联 `heritage`，42% Jewish 关联 `tradition`。
  - **可预测性**：`economy`（developed vs developing）预测准确率达 **89.2%**，`ethnicity` 达 **85.2%**，表明叙事隐含强烈的社会经济与文化信号；`role`（40.9%）与 `religion`（42.9%）可预测性较低，偏见更隐蔽。
  - **多样性**：整体语义相似度均值 51.6%；国籍维度差异最大（意大利 47% vs 斯里兰卡 64%）。
- **模型差异**：GPT-4o 在性别分类预测上准确率最高（66.0%），Llama3 在民族维度预测表现最强（85.2%），表明不同基座模型的偏见分布模式存在显著差异。

## 相关工作脉络
- **MirrorStories (Yunusov et al., 2024)**：支持兴趣/偏好驱动的个性化故事生成；本文在其基础上引入家长社会文化身份，并补充了文化真实性与包容性的量化评测维度。
- **Gender/Cultural Bias in Narratives (Huang et al., 2021; Toro Isaza et al., 2023)**：聚焦成人叙事或事件链中的性别/叙事结构偏见；本文首次将性别、种族、宗教、国籍与家庭角色交叉纳入儿童故事生成偏见的系统评估。
- **Stereotype Content Model & ABC Model (Fiske et al., 2007; Koch et al., 2016)**：成人社会认知理论；本文将其适配至儿童角色刻画，扩展为 Physical/Emotional/Mental/Moral/Other 五类属性体系。
- **Lexical Complexity for Children (Rooein et al., 2023; Valentini et al., 2023)**：研究 LLM 输出难度适配；本文验证 Biased Tales
