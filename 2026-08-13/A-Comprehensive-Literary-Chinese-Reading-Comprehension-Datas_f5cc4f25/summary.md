---
title: "A-Comprehensive-Literary-Chinese-Reading-Comprehension-Datas"
source: https://aclanthology.org/2025.emnlp-main.177.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:40:50"
---

# 论文速读：A-Comprehensive-Literary-Chinese-Reading-Comprehension-Datas

## 一句话总结
本文针对文言文阅读理解（CRISIS）任务中语料稀缺、信息冗杂、选项顺序偏差及语义纠缠等痛点，构建了迄今规模最大的文言文 RC 数据集，并提出 **VIRTUAL** 推理框架：通过跨语种相似度证据抽取、选项乱序去偏、基于 AMR 的选项子句切分与多数投票机制，显著提升了 LLM 与传统模型在该任务上的细粒度理解准确率。

## 研究问题与动机
1. **低资源语言理解瓶颈**：文言文（Literary Chinese）缺乏大规模对齐训练语料，且语法风格、词义演变与现代汉语差异显著，制约了现有 LLM 的深度理解能力。
2. **现有方案三大缺陷**：输入信息冗杂导致噪声干扰（excessive information）；模型对选项排列顺序存在系统性偏差（order bias）；选项逻辑复杂、语义纠缠难以直接切分验证（entangled conundrums）。
3. **高质量评测数据集匮乏**：既有数据集（ACRE、CCLUE、WYWEB 等）题目数量有限、偏重常识记忆而非语篇推理，且缺少现代汉语译文、人物/官职/年号等多维知识对齐。
4. **证据利用不充分**：传统方法多依赖端到端分类，未显式检索并融合原文句、译文句与词典注释作为可解释支撑信号。

## 核心贡献（创新点）
1. **构建最大规模文言文阅读理解数据集 CRISIS**：从 7 个公开来源筛选 4,415 道深度语篇理解题，经清洗与答案均衡后，由 Qwen 批量生成银标准现代汉译、人物传记、官职、年号与朝代标注，覆盖先秦至清代。
2. **提出 VIRTUAL 六步证据驱动推理框架**：首次将跨语种证据抽取、选项乱序、AMR 语义切分与投票机制统一，针对性缓解信息过载、顺序偏差与逻辑纠缠问题。
3. **设计相似度阈值联合检索算法**：基于 GuwenBERT 向量化，同时检索文言文原句、现代汉译句并可选附加 ZDIC 关键词注释，形成多层证据提示链。
4. **实证揭示顺序偏差与 AMR 切分的普适价值**：实验证明所有测试模型（含非 LLM 与 LLM）均存在显著顺序偏差；AMR 子句切分提供的结构化验证信号优于完整 COT。
5. **验证跨任务泛化性**：在简单/中等/复杂三级难度上均有稳定提升，并在现代汉语 RC 基准 C3 上验证了 VIRTUAL 策略的可迁移性。

## 方法详解
VIRTUAL（eVIdence cuRation with opTion shUffling and Abstract meaning representation-based cLauses segmenting）包含六个步骤：
1. **Embedding（嵌入）**：使用 GuwenBERT 将所有逗号分隔的子句（篇章句、问题句）编码为向量，存入 FAISS 向量数据库，便于后续快速相似度检索。
2. **Shuffling（乱序）**：对原始四个选项执行循环移位生成变体（如 `ABCD → BCDA → CDAB`），共构造三种排列，以消除模型对固定位置答案的位置偏好。
3. **AMR-based Segmenting（AMR切分）**：调用 AMR 解析工具将选项转换为单根有向无环图三元组（如 `(eat-01 :ARG0 dog) (:ARG1 bone)`），再利用 LLM 将三元组还原为可独立验证的自然语言子句（clauses），作为结构化思维提示。
4. **Evidence Extracting（证据抽取）**：对每个（切分后的）选项 `opt` 与所有句子嵌入计算距离 `Score_sim(opt, y_i) = ||opt - y_i||_2`，设定阈值 `t=0.3` 过滤。按规则选取前 `#s` 条文言文原句、前 `#t` 条现代汉译句，并可选拼接 ZDIC 关键词注释，输出证据集 `E`。
5. **Solving（求解）**：将证据、乱序选项与 AMR 子句输入 LLM。尝试三种提示策略：zero-shot（直接选答）、one-shot（提供总结题与细节题各一例）、chain-of-thought（COT，逐项输出 0/1 判断后取正确率）。
6. **Voting（投票）**：对三次乱序求解结果进行多数投票（majority voting），原顺序仅作为兜底策略，最终输出得票最高的选项并还原至原始标签。

## 实验与结果
- **数据集划分**：CRISIS 共 4,415 题，按 8:1:1 划分训练/验证/测试集，答案分布均衡（≈1:1:1:1）。
- **评测基线**：非 LLM EVERGREEN；LIM o1-mini、GLM-4、GPT-4o、ERNIE-4.0-8K、Qwen-plus。
- **主要结果**：VIRTUAL + Qwen-plus 达到 **80.8%** 平均准确率，较原始 Qwen-plus（73.1%）**提升 7%**；在复杂难题上提升幅度达 **10%**。各选项准确率分布更均衡（A/B/C/D 介于 77.7%~83.9%）。
- **消融实验**：去除乱序投票降至 78.1%；去除 AMR 切分降至 78.5%；去除现代汉译证据降至 78.7%；仅去除关键词注释影响最小（80.3%），表明 LLM 自身具备一定常识召回能力。
- **证据组合最优**：文言文证据 1 句 + 现代汉译证据 3 句（Table 7）表现最佳。
- **泛化测试**：在现代汉语 RC 数据集 C3 上，VIRTUAL 将 Qwen-plus 准确率从 96.7% 提升至 **98.7%**。
- **提示策略对比**：One-shot 无额外增强数据效果最佳（80.8%），COT 反而因指令复杂度导致下降至 74.5%。

## 相关工作脉络
1. **ACRE (Rao et al., 2023)**：首个 CRISIS 专属数据集，但混入大量常识记忆题；本文从中严格筛选纯语篇理解题目并大幅扩充规模。
2. **CCLUE / WYWEB (Xu et al., 2020; Zhou et al., 2023)**：涵盖文言文分类与少量 RC，但题目量有限且缺乏现代汉译对齐与细粒度证据链支持。
3. **AC-EVAL (Wei et al., 2024) / E-EVAL (Hou et al., 2024)**：聚焦对比学习或 K-12 教育评估，公开题目稀少；本文补充高难度 Gaokao 真题（2021-2024）并引入朝代/官职等背景知识增强。
4
