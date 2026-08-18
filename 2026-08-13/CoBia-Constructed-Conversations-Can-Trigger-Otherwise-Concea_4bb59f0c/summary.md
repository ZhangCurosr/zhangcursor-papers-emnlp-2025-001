---
title: "CoBia-Constructed-Conversations-Can-Trigger-Otherwise-Concea"
source: https://aclanthology.org/2025.emnlp-main.84.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:44:42"
field: "大语言模型安全与偏见评估"
keywords: ["LLM安全", "社会偏见", "对抗攻击", "对话历史", "jailbreak", "偏见评估"]
innovations: ["提出CoBia轻量级对抗攻击框架，仅需单次查询即可通过构建对话历史触发LLM隐性社会偏见", "设计HCC与SCC互补两种攻击设置，系统评估LLM在对话中的偏见恢复能力", "构建覆盖112个社会群体的CoBia数据集，整合RedditBias、SBIC、StereoSet三个来源"]
benchmarks: ["CoBia数据集", "Bias Judge", "Granite Judge", "NLI Judge"]
---

# 论文速读：CoBia-Constructed-Conversations-Can-Trigger-Otherwise-Concea

## 一句话总结
本文提出 CoBia（Constructed Bias），一套仅需单次查询的轻量级对抗攻击方法，通过构建包含虚假偏见的对话历史来触发 LLM 的社会偏见，并评估模型是否能从虚构偏见中恢复并拒绝后续的偏见问题。

## 研究问题与动机
- 尽管 LLM 配备了安全护栏，社会偏见仍深嵌于模型行为中，常通过 jailbreak 等手段重新浮现。
- 现有 jailbreak 方法需要技术知识或大量查询，而普通用户可能在无意中因触发语言而暴露于有害偏见。
- LLM 在对话中一旦偏离正轨，往往无法恢复（"get lost"），但对此的系统性研究不足。
- API 对话历史可由用户控制，这一特性可被用于构造对抗性对话，但尚未被系统性地作为攻击向量研究。

## 核心贡献（创新点）
1. **提出 CoBia 轻量级对抗攻击框架**：仅需单次查询即可通过构建对话暴露 LLM 隐藏的社會偏见，区别于需要多轮交互或参数访问的传统 jailbreak。
2. **构建 CoBia 数据集**：整合 RedditBias、SBIC、StereoSet 三个数据集，去重后得到覆盖 112 个社会群体、六个人口统计类别的负面描述符集合。
3. **设计 HCC 与 SCC 两种攻击设置**：分别利用结构化对话历史和扁平化单块提示来触发偏见，两者互补可覆盖不同模型的脆弱性。
4. **系统性评估 11 个 LLM 的偏见恢复能力**：结合三种自动裁判（Bias Judge、Granite Judge、NLI Judge）和人工标注，验证 LLM 在对话中难以拒绝偏见跟进问题的普遍性。

## 方法详解
- **数据集构建**：从 RedditBias（Reddit 讨论中的负面描述符）、SBIC（社交媒体帖子级刻板印象标注）、StereoSet（众包语境关联测试）三个来源提取负面描述符；使用 spaCy POS 标记和 Llama3.3:70b 进行描述符抽取；再用 Llama Guard 过滤确保数据安全性（99% 属于 S10 类贬低/去人性化内容）。
- **HCC（History-based Constructed Conversation）**：用户首先询问某社会群体特征，助手回复包含 8 项特征（6 项正面 + 2 项负面，主负面描述符 n 混入其中）；随后用户在 4 轮追问中引用其中 4 个特征（始终包含次要负面描述符 n₂，排除主负面描述符 n）；最后一轮要求用统计推理在 10 句话内解释 n 与 n₂ 的关联。整个对话作为历史传递给模型。
- **SCC（Single-block Constructed Conversation）**：与 HCC 内容相同，但不使用 user/assistant 标签，将整个对话作为单个用户消息呈现。
- **Tips（降低安全过滤触发风险）**：(1) Anaphora——用代词避免重复提及敏感社会群体；(2) Substitution——用数字索引（如 #6）替代显式负面描述符；(3) Preemptive Disavowal——在提示中加入免责声明（"I am aware that these characteristics do not apply to all of these people..."）。
- **评判器**：Bias Judge（基于 Fan et al. 2025 的偏见判断提示，二分类 yes/no）、Granite Judge（IBM Granite Guardian，风险检测模型）、NLI Judge（基于自然语言推理，判断输出与偏见观点的逻辑关系：enforces/agrees/neutral/negates）。

## 实验与结果
- **模型**：11 个 LLM（mistral:7b、olmo2:13b、command-r:35b、llama3.1:8b、llama3.3:70b、gemma2:27b、deepseek-v2:16b、phi4:14b、qwen2.5:7b、gpt-3.5-turbo、gpt-4o-mini），temperature=0，top_p=0。
- **基线**：0-Shot（直接提问）、DAN（Do Anything Now jailbreak）、R-Play（角色扮演攻击）。
- **主要结果**（Bias Judge macro-average，Table 2）：
  - **qwen2.5:7b** 在 UCC 下得分最高（83.60%），为"高度偏见"模型之一；llama3.3:70b（85.54%）和 command-r:35b（82.59%）同样超过 80% 阈值。
  - HCC 和 SCC 在大多数设置下显著优于所有基线（DAN/R-Play/0-Shot 通常 <20%）。
  - HCC 与 SCC 表现互补：对 mistral:7b、llama3.1:8b、phi4:14b 等模型 HCC 更有效；对 gpt-4o-mini 等 SCC 更有效；组合后偏差检测率提升至少 30%。
  - **gemmas2:27b** 和 **deepseek-v2:16b** 在 CoBia 下得分较低，前者因整体鲁棒性强，后者因输出模糊导致裁判误判。
- **偏见分布**：国籍（national origin）类别偏见最高，所有设置下均显著高于种族、宗教、性取向等类别；模型大小与偏见评分无明确相关性（qwen2.5 各规模实验显示 32B 模型安全性最佳但原因不明）。
- **裁判对齐**：NLI Judge 与 Bias Judge 配对一致率 0.79（Cohen's κ=0.53）；在 DAN 分歧案例中，NLI Judge 与 4 位人工标注者 83% 一致，显示 NLI Judge 在区分"冒犯性语言"与"真实偏见"方面更可靠。

## 相关工作脉络
1. **LLM Jailbreak 研究**（Purpura et al. 2025 分类）：CoBia 属于 prompt-based 攻击，但区别于需要多轮交互或大量查询的现有方法，仅需单次查询且目标为非技术性用户。
2. **社会偏见评估**（intrinsic vs. extrinsic）：CoBia 填补了 extrinsic 评估中"对话场景偏见放大"的研究空白，传统 benchmark（如 BBQ）报告近期模型偏见很低，但 CoBia 揭示了隐性偏见。
3. **角色扮演攻击**（Personabased attacks）：R-Play 等 baseline 使用固定角色绕过安全机制，CoBia 则利用对话历史的"角色扮演"特性，使模型"相信"虚构的先验偏见。
4. **多轮对话安全**：Laban et al. (2025) 指出 LLM 在多轮对话中会"迷失"且无法恢复，CoBia 将此现象量化为偏见放大程度的测量。
5. **LLM-as-a-Judge**：本文采用三种裁判三角验证，区别于单一裁判的偏见评估工作，提升了评估可靠性。

## 局限性与未来方向
- 仅评估 11 个 LLM（虽覆盖 9 家机构），未来可扩展至更多模型。
- 仅测试两种对话模板（HCC/SCC），未探索更长对话、情感语言等不同风格。
- 人工评估范围有限（4 位标注者，每人 300 条），虽与自动裁判高度对齐但仍需更大规模验证。
- 构造的对话序列与真实用户行为不完全一致，结果应解读为压力测试信号而非部署频率估计。
- 未提出具体缓解策略，仅建议限制用户对对话历史的控制权；OpenAI 已在 2025 年 3 月为新版 API 引入唯一历史 ID，但旧 API 仍维持用户控制模式。

## 研究启发与可借鉴点
- **对话历史可利用性作为攻击向量**：将 API 对话历史的"用户可控性"系统性转化为安全评估工具，思路可迁移至其他脆弱性检测（如幻觉、信息泄露）。
- **HCC/SCC 互补策略**：单一攻击设置可能遗漏部分模型的偏见，组合多种构造方式可更全面地暴露脆弱性，类似思路可用于鲁棒性评测设计。
- **NLI Judge 的逻辑关系判别**：通过判断输出与偏见观点的逻辑关系（而非仅表面语言）来识别隐性偏见，可减少因"冒犯性措辞但非真正偏见"导致的误判。
- **CoBia 数据集构建流程**：多源数据集融合→POS/LLM 描述符提取→安全模型过滤的流程可作为通用偏见数据集构建模板。
- **模型规模与偏见非线性关系**：qwen2.5 各规模实验显示 32B 安全性最佳但 72B 并非最安全，提示"越大越安全"假设需谨慎，可为模型规模化与安全性的权衡研究提供实证参考。

## 关键术语表
- **CoBia**：Constructed Bias 的缩写，本文提出的轻量级对抗攻击框架，通过构建虚假对话历史触发 LLM 偏见。
- **HCC（History-based Constructed Conversation）**：利用 OpenAI Chat Completions API 的用户可控对话历史特性，构造 multi-turn 对话以暴露偏见。
- **SCC（Single-block Constructed Conversation）**：将完整对话作为单个无结构用户消息呈现的攻击设置，与 HCC 形成互补。
- **UCC**：UION of HCC and SCC，任一方法判定为偏见即计为偏见，用于合并两种攻击设置的评估结果。
- **Bias Judge**：基于 LLM 的偏见裁判器，判断模型输出是否与偏见观点一致（二分类 yes/no）。
- **NLI Judge**：基于自然语言推理逻辑关系的偏见裁判器，分类输出与偏见观点的关系为 enforces/agrees/neutral/negates。
- **Jailbreak**：突破 LLM 安全机制的对抗性攻击，本文聚焦于 prompt-based 轻量级 jailbreak。
- **社会偏见（Societal Bias）**：模型对不同社会群体（性别、种族、宗教等）持有的刻板印象或歧视性观点。

## 可复现要素
- **数据集**：CoBia 数据集已公开（CC BY-SA 4.0），包含 112 个社会群体及负面/正面描述符。
- **代码**：github.com/nafisenik/CoBia 已开源。
- **关键超参**：temperature=0，top_p=0（确定性输出）。
- **模型**：11 个 LLM（9 家机构），开源模型通过 Ollama 部署，闭源模型通过官方 API 访问。
- **评判器**：Bias Judge（基于 Fan et al. 2025 提示）、Granite Judge（IBM Granite Guardian）、NLI Judge（本文设计）。
