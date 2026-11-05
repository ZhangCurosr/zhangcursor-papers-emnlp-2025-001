---
title: "CiteBART-Learning-to-Generate-Citations-for-Local-Citation-R"
source: https://aclanthology.org/2025.emnlp-main.89.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 07:49:52"
---

# 论文速读：CiteBART-Learning-to-Generate-Citations-for-Local-Citation-R

## 一句话总结
本文提出 CiteBART，一种基于 BART 架构的本地引文推荐（LCR）端到端生成式预训练方法，通过将作者-日期格式的引文替换为 `<mask>` 并进行重建学习；CiteBART-Global 进一步将引用论文的标题与摘要拼接至局部上下文，在多个主流 LCR 基准上取得 SOTA 性能，同时系统量化了生成式引文推荐的幻觉行为。

## 研究问题与动机
- **核心问题**：本地引文推荐（Local Citation Recommendation, LCR），即在给定局部文本语境及占位符位置，预测应插入的正确参考文献集合（通常为前 k 个候选）。
- **现有方法不足**：
  1. 主流工作（DualEnh、HAtten、SymTax 等）普遍采用“预取+重排”两阶段流水线，必须在推理阶段调用被引论文的标题/摘要作为候选，计算复杂且依赖外部检索库。
  2. 基于编码器的模型（BERT-GCN、SciBERT、SPECTER）擅长构建语义表征，但难以直接生成结构化、符合学术规范的作者-日期引文 token。
  3. 现有生成式尝试（如直接 Prompt LLM）受限于候选池规模与上下文长度，幻觉率高且排序不稳定，缺乏针对性的预训练信号。
- **动机**：探索纯生成式架构在 LCR 中的可行性，以端到端方式替代传统检索流水线；同时验证利用引用方全局信息（而非被引方）增强局部上下文学习信号的有效性，并为生成式引文推荐的可靠性提供可度量的幻觉分析框架。

## 核心贡献（创新点）
1. **提出 CiteBART 端到端生成式预训练框架**：直接在 BART-base 上设计自定义引文掩码目标，模型无需外部候选池即可自回归生成作者-日期格式引文，从根本上区别于依赖 recall+rerank 的两阶段管线。
2. **设计 CiteBART-Global 上下文增强策略**：受 REALM 启发，在训练时将引用论文的标题与摘要拼接至局部上下文，使模型学习“全局语义→局部引文位置”的映射；与 DualEnh/HAtten 依赖被引论文元数据不同，该方法在推理时完全不暴露目标文献信息，避免信息泄露。
3. **构建引文幻觉细粒度分类体系与条件度量指标**：首次针对 LCR 任务提出 MaHR、MaHR-partial、top-k-match-MaHR 与 exact-match-MaHR 等指标，揭示“当 top-k 包含正确答案时其他预测的幻觉倾向显著下降”的规律，为生成式学术推荐的可靠性评估提供新基准。

## 方法详解
- **基础架构**：采用 vanilla BART-base（编码器-解码器 Seq2Seq 模型），保留其原始交叉熵重建损失，不修改网络结构。
- **自定义掩码预训练**：
  - 将原始数据集中的 `TARGETCIT` 占位符统一替换为 `<mask>`。
  - 利用 BART 的 span-masking 特性，将连续的(author-date) 格式引文（如 `Yao and Zweig, 2015`）整体掩码，训练解码器一次性生成完整结构化引文。
- **两种训练方案**：
  - **CiteBART-Base**：输入仅为包含 `<mask>` 的局部引用上下文，测试纯局部语义对引文预测的贡献。
  - **CiteBART-Global**：在局部上下文末尾拼接引用论文的标题与摘要，以预训练 BART 的 `</s>` 分隔符衔接，使梯度可经由全局信息反向传播，学习跨模态/跨层次关联。
- **推理设置**：使用 grouped beam search（beam=20），施加 diversity penalty=1.5 抑制重复输出，最大生成长度限制为 25 tokens（覆盖绝大多数 author-date 格式）。

## 实验与结果
- **数据集**：ACL-200、PeerRead（FullTextPeerRead）、RefSeer、ArXiv。预处理阶段移除了作者名冲突、格式错误及空作者等噪声样本，各数据集统计见表 2/3。
- **评估指标**：Recall@10（R@10）、Mean Reciprocal Rank（MRR）、Exact Match（EM）。
- **主要结果**（Table 4）：
  - CiteBART-Global 在 ACL-200（R@10: 0.739, MRR: 0.513）、RefSeer（R@10: 0.652, MRR: 0.479）、ArXiv（R@10: 0.502, MRR: 0.305）上均超越 SymTax、HAtten 等 SOTA 基线。
  - 仅在规模较小的 PeerRead 上略低于 SymTax（作者指出小数据集难以发挥生成式预训练优势）。
  - CiteBART-Base 仍为强基线（如 ACL-200 R@10: 0.686）。
- **消融实验**（Table 5，ACL-200 Global）：
  - 移除局部上下文导致 R@10 从 0.739 骤降至 0.588，证实局部语境是引文预测的决定性因素。
  - 移除标题或摘要不引起统计显著下降。
  - **关键反直觉发现**：若在预训练时同时注入被引论文的标题/摘要（All-including 配置），测试时 R@10 暴跌至 0.111，表明训练与推理的信息不对称会严重破坏模型泛化。
- **幻觉分析**（Table 6/7）：ACL-200 Global 在 top-3 预测时宏观幻觉率（MaHR）最低（4.12%）；当 top-k 列表中已命中 ground-truth 时，其他预测的 exact-match-MaHR 降至 2.40%（p<0.001），Arxiv Global 在此条件下表现最佳。

## 相关工作脉络
1. **BERT-GCN (Jeong et al., 2020)**：结合 BERT 局部表征与 GCN 引文网络，属早期编码器类方法，依赖图结构建模；CiteBART 放弃显式图结构，纯文本生成完成相同任务。
2. **DualEnh / HAtten / SymTax**：代表主流的“预取+重排”范式，均需候选被引论文的元数据参与打分；CiteBART 定位为端到端生成替代方案，推理阶段无需检索库。
3. **SPECTER / SciBERT**：面向科学文献的编码器预训练模型，擅长论文级嵌入与全局引文推荐，但受限于单向表征能力，无法直接输出结构化 author-date 引文。
4. **REALM (Guu et al., 2020)**：检索增强语言模型预训练框架；CiteBART-Global 的核心思想与之同源，即将外部全局信息作为监督信号注入 decoder 学习过程。
5. **LLM-based LCR (Jiang et al., 2025; 本文 4.5 节)**：尝试用 Llama-2-70b-chat 直接 prompt 完成引文选择，受限于上下文窗口与候选池规模，R@10 仅 0.134~0.152，远低于 CiteBART，印证了领域专用预训练的必要性。

## 局限性与未来方向
- **局限性**：
  1. 任务预设局部上下文必然具备引用价值（citation-worthiness），未覆盖无引用语境的自然写作场景。
  2. 模型能力受限于预训练语料池；若候选集中未包含某真实文献，模型只能生成幻觉引文（即使部分匹配作者名）。
  3. 当前仅支持单引文预测，实际学术论文常在同一语境引用多篇文献，现有基准数据集亦未提供多引文元数据。
  4. 生成式推理计算开销较高，大数据集测试需 1~2 天，难以满足实时场景需求。
- **未来方向**：探索自定义掩码填充与生成式信息检索（GRIR）中 retrieval token 识别的联合学习；降低 top-k 中的幻觉比例（尤其 other-hal 类伪造引文）；构建支持多引文共现的 LCR 基准；尝试同时利用引用方与被引方的全局信息实现更均衡的表征学习。

## 研究启发与可借鉴点
1. **结构化输出生成的掩码设计**：将符合特定学科规范的固定格式（author-date、法律条款编号、生物实体三元组等）整体掩码并作为 seq2seq 目标，可有效融合领域先验与生成能力，该思路可直接迁移至法律文献引用、医学指南推荐等任务。
2. **“查询侧全局信息增强，目标侧零泄露”的训练范式**：CiteBART-Global 证实仅注入查询文档的全局上下文即可显著提升推荐质量，且避免了双端信息泄露导致的过拟合；该设计对任何需要利用用户/文档全局特征但不希望引入目标偏差的生成式推荐系统均具参考价值。
3. **条件幻觉度量框架**：提出的 MaHR-partial 与 top-k-match-MaHR 指标打破了传统“对错二分”的评估局限，揭示了模型在“已知答案”情境下的置信度传导效应，可作为评估大模型在垂直领域生成可靠性的通用诊断工具。
4. **端到端替代两阶段流水线的可行性边界**：本文在大规模学术数据集（RefSeer/ArXiv）上证明高质量生成式预训练可完全取代 recall+rerank，但对小数据集（PeerRead）效果受限；后续研究可进一步刻画“数据规模-生成模型优势”的临界曲线，指导资源受限场景的架构选型。

## 关键术语表
- **Local Citation Recommendation (LCR)**：本地引文推荐，指在给定局部文本语境中为占位符预测相关参考文献集合的下游任务。
- **CiteBART-Base / CiteBART-Global**：两种预训练变体，Base 仅依赖局部上下文，Global 在上下文后拼接引用论文的标题与摘要。
- **`<mask>` (Custom Citation Mask)**：替代原始 `TARGETCIT` 占位符的自定义 token，作为模型需生成的 author-date 格式引文目标。
- **Macro Hallucination Rate (MaHR)**：在所有生成引文中，无法对应数据集中任何真实文献的比例，用于量化生成式 LCR 的幻觉严重程度。
- **MaHR-partial**：仅统计作者名或年份部分匹配的“半幻觉”比例，反映模型即使出错也往往保留了高相关语义线索。
- **top-k-match-MaHR / exact-match-MaHR**：限定 top-k 预测中已命中 ground-truth 条件下的幻觉率，验证了“模型确信正确答案时幻觉倾向显著下降”的现象。
- **Grouped Beam Search with Diversity Penalty**：推理解码策略，通过分组束搜索配合多样性惩罚系数（1.5）抑制引文输出的重复与模式坍塌。

## 可复现要素
- **数据集**：ACL-200、PeerRead、RefSeer、ArXiv
