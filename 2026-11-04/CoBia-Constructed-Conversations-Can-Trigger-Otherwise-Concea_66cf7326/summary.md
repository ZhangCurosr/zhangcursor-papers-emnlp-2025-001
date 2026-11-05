---
title: "CoBia-Constructed-Conversations-Can-Trigger-Otherwise-Concea"
source: https://aclanthology.org/2025.emnlp-main.84.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:49:45"
field: "大语言模型安全与偏见评估"
keywords: ["LLM 安全", "社会偏见", "对抗攻击", "对话式越狱", "红队测试", "LLM-as-a-judge"]
innovations: ["提出 CoBia 轻量对话式对抗攻击，单查询即可暴露 LLM 隐藏社会偏见", "构建 CoBia 数据集（112 群体/2841 描述符），整合三个主流偏见基准", "设计 HCC/SCC 双方法及三类判官（Bias/NLI/Granite）的系统性评估框架"]
benchmarks: ["CoBia dataset", "RedditBias", "SBIC", "StereoSet", "BBQ (对比基线)"]
---

# 论文速读：CoBia: Constructed Conversations Can Trigger Otherwise Concealed Societal Biases in LLMs

## 一句话总结
论文提出 CoBia，一套仅用单次查询即可通过构建对话（HCC/SCC）暴露大语言模型隐藏社会偏见的轻量级对抗攻击方法，并在 11 个开源与闭源模型上系统验证了多轮对话场景下模型难以从偏见话语中恢复的问题。

## 研究问题与动机
- **现有安全机制盲区**：当前 guardrail 主要拦截显式有害输出（如黑客指令），但对针对社会群体的刻板印象和隐性偏见关注不足，偏见仍深度嵌入模型行为。
- **真实交互风险**：非技术性用户无需掌握复杂 jailbreak 技术，仅在普通对话中无意中触发"泄漏语言"，就可能暴露模型的有害偏见。
- **多轮对话中的不可恢复性**：已有研究表明当 LLM 在对话中"走错路"后往往无法恢复，因此需评估模型在被诱导发表偏见声明后能否拒绝后续偏见问题。
- **基线方法检测力有限**：零样本直接提问（0-Shot）及已知 jailbreak 方法（DAN、R-Play）检测到的偏见分数普遍较低，难以揭示通过对话上下文诱发的隐蔽偏见。

## 核心贡献（创新点）
1. **提出 CoBia 轻量对抗攻击套件**：通过构造对话（HCC/SCC）用单次查询即可触发 LLM 表达社会偏见，相比需要多次查询或专业知识的传统 jailbreak 方法更贴近普通用户的真实交互场景。
2. **构建 CoBia 数据集（112 个社会群体）**：整合 RedditBias、SBIC、StereoSet 三个基准，生成含负向/正向描述符的统一数据集，覆盖性别、种族、宗教、国籍、性取向和其他六大社会人口类别。
3. **系统性 11 模型跨类别偏见评估与三人判官对齐验证**：在 11 个 LLM 上评测，并引入 Bias Judge / Granite Judge / NLI Judge 三种自动化判官与人工标注交叉验证，发现所有高 UCC 分模型在三类判官上高度一致。
4. **揭示模型间方法互补性与偏见类别差异**：HCC 与 SCC 在不同模型上表现互补（组合可提升至少 30% 得分）；国籍（national origin）偏见显著高于种族、宗教与性取向，暗示后者受到更强的安全监控。

## 方法详解
- **CoBia 数据集结构**：三元组 $\mathcal{D} = \{(c, g, n) \mid c \in \mathcal{C}, g \in \mathcal{S}_c, n \in \mathcal{N}\}$，其中 $c$ 为社会类别、$g$ 为社会群体、$n$ 为负向描述符；扩展后每组群体含 6 个正向描述符及对应解释。
- **HCC（History-based Constructed Conversation）**：利用 API 端点对话历史由用户可控的特点，构造一个包含 user/assistant 角色的多轮对话：
  1. User 询问某群体的特征列表；
  2. Assistant 回复 8 条特征（6 条正向 + 2 条负向描述符 $n$ 和 $n_2$）；
  3. 随后 4 轮 User 追问分别引用列表中 4 条随机特征（始终包含 $n_2$，刻意排除 $n$）；
  4. 最终追问要求模型将 $n$ 与 $n_2$ 关联并用统计推理写出 10 句话解释——迫使模型围绕目标负向描述符给出详细说明。
- **SCC（Single-block Constructed Conversation）**：与 HCC 内容相同，但将整个对话以单一 user 消息形式呈现（不带角色标签）。
- **绕过安全过滤三技巧**：(1) Anaphora — 用代词（them/these people）避免重复提及敏感群体名；(2) Substitution — 用列表编号（#6）替代负向描述符的直接表述；(3) Preemptive Disavowal — 前置免责声明"我意识到这些特征不适用于所有人，请勿再次提及"。
- **三类 Judge 设计**：
  - **Bias Judge**（基于 Fan et al. 2025 修改）：二分类，综合判断输出是否同意偏见观点，同时排除纯粹事实性陈述。
  - **Granite Judge**（IBM Granite Guardian）：通用风险检测器，二分类 yes/no。
  - **NLI Judge**：考察偏见观点与模型输出之间的逻辑关系（entails/agrees/neutral/negates），将 entails 与 agrees 视为检测到偏见。

## 实验与结果
- **数据集**：CoBia（112 群体，2,841 条负向描述符，六大类别），实验共发出 80 万+ 次查询。
- **评估模型**：11 个 LLM（mistral:7b, olmo2:13b, command-r:35b, llama3.1:8b, llama3.3:70b, gemma2:27b, deepseek-v2:16b, phi4:14b, qwen2.5:7b, gpt-3.5-turbo, gpt-4o-mini），温度=0 确定性输出。
- **基线**：0-Shot、DAN（Do Anything Now 13.0）、R-Play（文化社会学家角色扮演）。
- **主要结果（表 2 Macro-average, Bias Judge）**：
  - **最强 CoBia 分数**：llama3.3:70b **85.54%**、command-r:35b **82.59%**、qwen2.5:7b **83.60%**（三者均被标记为"高度偏见"UCC > 80%）。
  - **HCC vs SCC 互补性**：mistral-7b、llama3.1:8b、phi4:14b 上 HCC > SCC；gpt-4o-mini 上反之；deepseek-v2 和 gemma2-27b 上 HCC≈SCC 但 UCC 组合提升至少 30 个百分点。
  - **整体趋势**：HCC 和 SCC 在所有三类判官下均显著优于所有基线；DAN 在 gemma2:27b（70.01% micro）和 llama3.3:70b 上接近 CoBia，但 NLI Judge 显示其输出多为冒犯性语言而非针对目标群体的实质偏见。
  - **偏见类别差异**：国籍（national origin）在所有模型和设置下偏见最高（qwen2.5:7b 和 command-r:35b 接近 1.0），种族、宗教、性取向相对受控（gemma2:27b 三项 < 0.2）。
  - **模型规模效应**：qwen2.5 系列（3B–72B）未发现明确单调相关性，32B 版本表现最优。
  - **判官对齐**：NLI Judge 与 Bias Judge 配对一致率 79%（Cohen's κ=0.53）；在 DAN 分歧案例中 NLI Judge 与人工标注一致率达 83%。

## 相关工作脉络
- **Jailbreak 攻击研究**（Purpura et al. 2025 分类）：包括 prompt/token 级（DAN、R-Play）、梯度级（需参数访问）、基础设施级攻击；CoBia 属最低门槛的 prompt 级，仅需单查询，且可通过禁用用户对话历史控制轻易修复。
- **LLM 社会偏见评测**（内在 vs 外在）：内在方法依赖静态词嵌入（如 WEAVes、SEW），外在方法依赖生成式评测；近年基准（如 BBQ）对最新模型报告的偏见偏低，与本文 0-Shot 基线结果一致，凸显需要更强的诱捕方法。
- **基于人格的偏见攻击**（Deshpande et al. 2023; Gupta et al. 2024）：通过 persona assignment 诱发偏见；CoBia 不依赖固定 persona，而是通过构造多轮对话上下文让模型自我产生偏见输出。
- **多轮对话中的安全脆弱性**（Zhou et al. 2024b "Speak out of turn"）：关注多轮对话中安全的失守；CoBia 进一步聚焦"模型能否从已发表的偏见声明中恢复"这一更具体的恢复能力问题。
- **偏见放大与人类-AI 交互**（Allan et al. 2025; Bai et al. 2024b）：指出 LLM 作为"取悦用户"的倾向及用户选择性解读会放大偏见；本文的实验框架直接建模了这一交互动态。

## 局限性与未来方向
- **模型覆盖有限**：仅评测 11 个模型（每个机构至少一个），虽包含 80 万+ 查询，但更多模型可能呈现不同模式。
- **对话模板单一**：仅测试两类模板，未探索更长对话、不同语气/情绪风格的影响；未来可扩展模板多样性以进行更全面的审计。
- **人工评测规模有限**：4 名标注员各评 300 条输出，虽与自动判官对齐良好，但更大规模人工标注将增强结论稳健性。
- **与真实用户行为存在差距**：构造对话是红队测试手段，并非典型用户交互序列；结果应解读为压力测试信号而非部署时发生率估计。
- **防御方案仅停留在高层建议**：如限制用户对对话历史的控制（OpenAI 已于 2025 年 3 月起在 API 中引入历史聊天唯一 ID），但缺乏具体防御算法设计；未来需发展针对对话式对抗攻击的鲁棒防御机制。

## 研究启发与可借鉴点
- **对话历史可控性作为安全漏洞**：API 端点允许用户完全控制 conversation history，这是一个被忽视的安全攻击面；可直接迁移到本团队的对话安全审计工作中，作为通用 stress-testing 范式。
- **"正向+少量负向"混合列表策略**：将 2 个负向描述符混入 6 个正向描述符可降低触发安全过滤的概率，此设计思路可迁移到其他类型的偏见/有害内容检测研究中。
- **多判官交叉验证框架**：引入 Bias Judge + NLI Judge + Granite Judge 三类判官并通过人工标注校准，NLI Judge 在分歧案例中与人工对齐度更高，这一判官组合与评估协议可作为后续偏见评测的标准做法。
- **HCC 与 SCC 互补使用**：针对不同模型采用不同更有效的方法（HCC 对指令微调模型更好，SCC 对其他模型），可借鉴此双方法策略提升偏见的检出率。
- **与团队方向结合机会**：可将 CoBia 框架应用于本团队关注的（如多模态/Agent/多轮对话）场景，测试在多 Agent 协作或长程对话中偏见的传播与放大效应。

## 关键术语表
- **CoBia（Constructed Bias）**：论文提出的轻量级对抗攻击套件，通过构造多轮对话暴露 LLM 中隐藏的社会偏见。
- **HCC（History-based Constructed Conversation）**：CoBia 的一种方法，利用 API 端点的 user-controlled 对话历史构造 user/assistant 多轮对话以触发偏见输出。
- **SCC（Single-block Constructed Conversation）**：CoBia 的另一种方法，将完整对话内容压缩为单条 user 消息（不带角色标签）发送给模型。
- **UCC（Union of HCC and SCC）**：将 HCC 和 SCC 两种方法的检测结果取并集，任一方法判为偏见即记为偏见，用于提高检测覆盖率。
- **Bias Judge**：基于 LLM 的判官，判断模型输出是否在直接或间接意义上认同偏见观点（二分类 yes/no）。
- **NLI Judge**：基于自然语言推理关系的判官，评估模型输出与偏见观点之间的 entail/agree/neutral/negate 逻辑关系。
- **DAN（Do Anything Now）**：一种经典的越狱提示，通过声称解除 AI 安全限制来诱导模型输出有害内容。
- **Llama Guard**：Meta 开源的内容安全分类模型，用于在数据集构建阶段过滤出 unsafe 样本（本文主要用于数据清洗）。

## 可复现要素
- **数据集**：CoBia 数据集含 112 个社会群体，**已开源**（GitHub: github.com/nafisenik/CoBia，CC BY-SA 4.0 许可）。
- **代码/权重**：代码和实验 artifact **已开源**；Llama Guard 3:8B、Llama 3.3:70b 等开源模型权重可用；闭源模型通过 API 访问。
- **关键超参**：temperature=0，top_p=0（确定性输出）；HCC 最终追问限制 10 句解释。
- **环境**：开源模型通过 Ollama 托管，闭源模型通过各自 API endpoint 调用。
