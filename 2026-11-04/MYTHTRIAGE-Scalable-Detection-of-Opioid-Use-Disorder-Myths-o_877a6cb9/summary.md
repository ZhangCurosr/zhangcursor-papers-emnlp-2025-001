---
title: "MYTHTRIAGE-Scalable-Detection-of-Opioid-Use-Disorder-Myths-o"
source: https://aclanthology.org/2025.emnlp-main.146.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:55:31"
field: "健康领域虚假信息检测"
keywords: ["myth detection", "misinformation", "large language models", "model distillation", "cascade architecture", "health misinformation", "YouTube", "opioid use disorder"]
innovations: ["提出MYTHTRIAGE分诊管道，轻量模型与LLM级联实现低成本大规模谣言检测，成本降低94%、时间减少76%", "从GPT-4o蒸馏训练DeBERTA-v3-base轻量模型，在专家标注稀缺场景下实现0.60-0.78 macro F1", "首次大规模实证分析YouTube平台OUD谣言分布（16.4万视频），揭示Kratom话题谣言占比最高（36%）及推荐链路放大效应"]
benchmarks: ["OUD Search Dataset (2.9K videos)", "OUD Recommendation Dataset (164K unique videos)", "Expert-annotated gold standard (310 videos, 2.5K labels)"]
---

# 论文速读：MYTHTRIAGE-Scalable-Detection-of-Opioid-Use-Disorder-Myths-on-a-Video-Sharing-Platform

## 一句话总结
本文提出了 MYTHTRIAGE，一种高效的分诊式标注管道，通过轻量模型（DeBERTA-v3-base）处理确定性样本、将困难样本转交大模型（GPT-4o），在 YouTube 上大规模检测阿片类药物使用障碍（OUD）相关谣言；该方法以极低的经济与时间成本（相比专家节省98%成本、96%时间），在 16.4 万视频上实现了 0.68–0.86 的 macro F1，并首次揭示了 YouTube 平台上 OUD 谣言的大规模分布特征。

## 研究问题与动机
1. **高风险健康谣言的规模化检测难题**：OUD 是美国首要死因之一（2022 年 10.8 万人药物过量死亡），线上平台已成为患者获取信息的重要渠道，但平台充斥阻碍治疗的谣言（如"MAT 只是用一种药替换另一种"），亟需大规模检测以支撑公共卫生干预。
2. **视频平台的盲区**：现有研究多聚焦 Reddit/Twitter 等文本平台，且样本量有限；YouTube 作为视频分享平台被广泛使用，但关于 OUD 谣言的大规模实证研究仍属空白。
3. **全量 LLM 标注成本过高**：LLM 虽具潜力，但 API 推理成本高昂（针对 16.4 万视频标注 8 类谣言），使得规模化应用不切实际。
4. **专家标注不可扩展**：临床专家标注质量高但速度慢（平均每视频 3 分钟），无法应对海量视频内容。

## 核心贡献（创新点）
1. **首个 YouTube OUD 谣言大规模数据集**：构建了包含 2.9K 搜索结果（1.8K 唯一视频）和 343K 推荐结果（164K 唯一视频）的两个数据集，以及 310 部视频的专家金标准标注（共 2.5K 标签）。
2. **MYTHTRIAGE 分诊管道**：提出轻量模型 + LLM 级联架构，通过 MSP（最大 softmax 概率）和 VET（验证误差倾向）双策略路由不确定样本，实现精度与成本的平衡。
3. **从 GPT-4o 到 DeBERTA-v3-base 的知识蒸馏**：在无足够专家标注的情况下，利用 GPT-4o 生成的 1,466 条合成标签训练轻量学生模型，验证了合成数据的有效性。
4. **大规模平台谣言分布实证分析**：发现约 20% 搜索结果显示支持 OUD 谣言，Kratom 话题谣言占比最高（36%支持 vs 22%反对），且深度推荐中支持性内容占比上升（Level 1 的 12.7% → Level 5 的 22%）。
5. **与临床专家深度协作的标注方案**：6 名临床研究者完成多轮标注，Krippendorff's α = 0.76，证明了跨领域协作在高 stakes 健康任务中的可行性。

## 方法详解
### 数据收集
- **话题选择**：使用 Google Trends（2021–2023，美国区，YouTube Search）对 156 个关键词进行两两对比，选出 top-4 阿片类和 top-4 MAT 关键词（Fentanyl、Percocet、Heroin、Codeine、Kratom、Narcan、Suboxone、Methadone）。
- **查询生成**：从 Trends 和相关查询中筛选出 73 个代表性搜索查询，排除过于宽泛/具体/非英文/重复/人名等噪声。
- **搜索数据采集**：对每个查询 × 4 种排序过滤器（Relevance/Upload Date/View Count/Rating）采集 top-10 结果，共 2,893 条（1,776 唯一视频）。
- **推荐数据采集**：递归采集每视频 top-4 推荐，深度 5 层，共 342,707 条链接（164,085 唯一视频）。

### 谣言识别与标注方案
- **8 种经临床验证的 OUD 谣言**（M1–M8），如 M1："MAT 只是用一种药替换另一种"、M8："Kratom 是无成瘾性的安全替代物"。
- **三分类任务**：合并 neutral(0) 和 irrelevant(2) 为 "neither(0)"，得到三类：support(1)、oppose(-1)、neither(0)。
- **专家标注**：6 名临床研究者分三轮标注 310 部视频，达成 Krippendorff's α = 0.76。

### 模型设计与蒸馏
- **LLM 基线**：评估 10 个 LLM（OpenAI/Anthropic/Google/Meta/DeepSeek/Qwen），GPT-4o 的 few-shot 表现最佳（macro F1: 0.82–0.87）。
- **知识蒸馏**：用 GPT-4o 为剩余 1,466 部视频生成合成标签，以 80:20 划分训练/验证集（专家标注作测试集），微调 8 个 DeBERTA-v3-base（每谣言一个，参数量 186M）。训练策略包括 upsample、class-weighted loss、学习率网格搜索（5e-6/1e-5/1e-6）、weight decay 网格搜索。
- **输入特征**：视频标题 + 描述 + 转录文本 + tags，拼接后截断至 1,024 tokens。

### MYTHTRIAGE 分诊策略
- **MSP（Maximum Softmax Probability）**：当预测类的 softmax 概率低于阈值时，将该样本路由给 GPT-4o。
- **VET（Validation Error Tendencies）**：对验证集上 F1 < 0.8 的类别，将所有属于该类别的预测路由给 GPT-4o。
- **MSP+VET（组合策略）**：满足任一条件即路由，兼顾样本级置信度与类别级系统性弱点。
- 额外评估了 MC-Dropout 和 Softmax Entropy，但因路由比例过高（90%）和计算开销大而弃用。

## 实验与结果
### 数据集规模
| 数据集 | 总量 | 唯一视频 | 专家标注 | 合成标注 |
|---|---|---|---|---|
| OUD Search | 2,893 | 1,776 | 310 | 1,466 |
| OUD Recommendation | 342,707 | 164,085 | 0 | 164,085 |

### 模型性能（Table 4，gold-standard 305 视频）
- GPT-4o few-shot：0.82–0.87 macro F1（最佳 M1=0.87）
- DeBERTA-v3-base（蒸馏）：0.60–0.78 macro F1
- MYTHTRIAGE MSP：+0.036 提升，路由 0–31%
- MYTHTRIAGE VET：+0.069 提升，路由 5–53%
- **MYTHTRIAGE MSP+VET**：**最佳，0.68–0.86 macro F1（中位数 0.81），M3 达到 GPT-4o 同等水平**，路由 5–67%

### 成本效益（164K 视频，8 类谣言，130 万标注）
| 方式 | 时间 | 费用 |
|---|---|---|
| 专家 | 8,209 小时 | $59,517 |
| GPT-4o 全量 | 1,240 小时 | $21,790 |
| **MYTHTRIAGE** | **300 小时** | **$1,282** |

- 相比专家：节省 **98% 成本、96% 时间**
- 相比全量 GPT-4o：节省 **94% 成本、76% 时间**
- 环境成本：CO₂ 排放从 1,472.83 kg 降至 105.52 kg（降低 93%）

### 平台谣言分析
- 约 **20%** 搜索结果支持 OUD 谣言；**30%** 反对。
- **Kratom** 话题谣言占比最高（36% 支持 vs 22% 反对），bias score = 0.15。
- Myth 2（"OUD 是自作自受"）争议最大：9% 支持，23% 反对。
- 切换搜索过滤器（非 Relevance）显著增加谣言暴露。
- 推荐链路中：Level 1 支持→支持跳转率 12.7%，Level 5 升至 22.2%。

## 相关工作脉络
1. **Model Cascading（Varshney & Baral, 2022; Mamou et al., 2022）**：此前工作主要在标准 NLP benchmark（CoLA/SQuAD/CIFAR-10）上验证 cascade 架构；本文首次将其应用于高 stakes 健康场景中的大规模谣言检测，并与 LLM 集成解决成本挑战。
2. **LLM for Social Science Annotation（Ranjit et al., 2024; Dammu et al., 2024）**：已有工作探索 LLM 辅助标注，但未解决 LLM 在长视频/大批量场景下的计算与 API 成本瓶颈；本文用轻量模型+分诊策略克服该问题。
3. **OUD 在线谣言研究（ElSherief et al., 2021, 2024; Mittal et al., 2024）**：此前主要在 Reddit/Twitter 等文本平台进行小规模（数十至数百样本）研究；本文首次在 YouTube 视频平台上实现大规模（16万+视频）实证分析。
4. **YouTube 谣言审计（Hussein et al., 2020; Albadi et al., 2022）**：本文延续了 YouTube 推荐链路审计方法（InnerTube API + 递归采集），但扩展至健康谣言场景并引入 MYTHTRIAGE 管道。
5. **Misinformation Detection with LLMs（Jung et al., 2025）**：本文延续使用文本元数据（标题+描述+转录+tags）和 few-shot prompting 的策略，在 OUD 谣言任务上验证了类似方法的有效性。

## 局限性与未来方向
1. **仅限 OUD 单一健康领域**：方法论可扩展至 COVID-19、精神健康等领域，但本文未验证跨领域迁移效果。
2. **仅覆盖 8 种谣言和 8 个话题**：更多谣言（如"治疗 OUD 费用高昂"）和话题（如 OxyContin）尚未涉及。
3. **无个性化推荐分析**：未考虑用户画像、搜索历史等个性化因素对谣言暴露的影响。
4. **单平台局限**：仅研究 YouTube，未与其他视频平台（TikTok、BitChute）对比。
5. **英语/美国中心**：基于英语查询，未涵盖非英语内容或美国以外地区（如玻利维亚、圭亚那）的 OUD 危机。
6. **纯文本输入**：未利用缩略图、视频帧等多模态信号，可能存在分类误差。
7. **Google Trends 偏差**：依赖主流搜索趋势，可能低估小众/边缘化群体使用的搜索词。

## 研究启发与可借鉴点
1. **合成数据蒸馏范式**：在专家标注稀缺的高 stakes 场景下，用强 LLM 生成合成标签训练轻量学生模型，是一种可复用的低标注成本方案；值得探索于其他垂直领域。
2. **MSP+VET 双策略路由**：结合样本级置信度（MSP）与类别级系统性弱点（VET）的分诊策略，比单一策略更有效，可直接迁移到其他 cascade 分类任务。
3. **分层采样构建金标准**：先用 GPT-4o 进行粗略预测，再按标签均衡抽样送入专家标注，可最大化专家标注效率，适用于高成本标注场景。
4. **谣言 bias score 量化框架**：将搜索结果的谣言分布转化为 [-1, 1] 连续分数，便于跨话题/跨过滤器的可比分析，方法简洁有效。
5. **LLM-as-a-judge 解决标签冲突**：在视频整体立场判定中，当同时存在支持与反对标签时，引入 LLM-as-judge 辅助人工裁决，提升了可扩展性。

## 关键术语表
**OUD（Opioid Use Disorder）**：阿片类药物使用障碍，一种慢性脑部疾病，是美国首要死因之一，2022 年导致 10.8 万人过量死亡。
**MAT（Medication-Assisted Treatment）**：药物辅助治疗，使用美沙酮、丁丙诺啡等药物结合咨询治疗 OUD，是循证医学推荐的疗法。
**MYTHTRIAGE**：本文提出的分诊式标注管道，通过轻量模型处理简单样本、困难样本转交 GPT-4o，实现低成本大规模谣言检测。
**MSP（Maximum Softmax Probability）**：分诊策略之一，当模型对某样本的预测 softmax 概率低于阈值时，将该样本路由给更强模型。
**VET（Validation Error Tendencies）**：分诊策略之一，根据验证集上各类别的 F1 表现，对模型系统性表现差的类别进行路由。
**Krippendorff's α**：信度测量指标，衡量多个标注者之间的一致性；α=0.76 表示中度至良好一致性。
**Macro F1-score**：宏平均 F1 分数，对每个类别单独计算 F1 后取平均，对类别不平衡数据更鲁棒。
**LLM-as-a-judge**：利用大语言模型作为裁判，对人类标注难以达成一致的情况进行自动化裁决。

## 可复现要素
- **数据集**：论文提供视频 ID 和标签（通过 YouTube Data API 可重新获取元数据）；原始数据集公开情况见论文声明。
- **代码/权重**：论文未明确提供开源代码链接，但描述了完整的训练细节（超参数、模型架构、损失函数）。
- **关键超参**：
  - DeBERTA-v3-base 学习率：5e-6 / 1e-5 / 1e-6
  - weight decay：5e-4 / 1e-4 / 5e-5
  - batch size：8，epochs：20
  - 输入截断：1,024 tokens
  - MSP 阈值：grid search（0–1，步长 0.01）
  - VET 阈值：per-class F1 < 0.8
  - LLM temperature：0.2
  - Few-shot 示例数：5
- **训练硬件**：单卡 NVIDIA A40 GPU（48GB）
