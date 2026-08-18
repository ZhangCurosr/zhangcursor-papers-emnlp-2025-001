---
title: "AdaRewriter-Unleashing-the-Power-of-Prompting-based-Conversa"
source: https://aclanthology.org/2025.emnlp-main.193.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:40:51"
field: "Conversational Search"
keywords: ["Conversational Query Reformulation", "Test-time Adaptation", "Reward Model", "Best-of-N", "Black-box LLM"]
innovations: ["首次系统分析测试时Best-of-N范式在查询重写中的潜力", "提出轻量级对比排序奖励模型用于推理时候选筛选", "实现黑盒LLM系统的即插即用测试时适配框架"]
benchmarks: ["TopiOCQA", "QReCC", "TREC CAsT 2019", "TREC CAsT 2020", "TREC CAsT 2021"]
---

# 论文速读：AdaRewriter-Unleashing-the-Power-of-Prompting-based-Conversa

## 一句话总结
本文提出AdaRewriter框架，训练一个轻量级BERT-sized的对比排序奖励模型，在推理时从LLM生成的多个查询重写候选中选出最优结果（Best-of-N），从而在对话搜索中充分释放基于提示的查询重写方法的潜力；该方法无需修改底层LLM，可无缝集成到商业黑盒API系统中。

## 研究问题与动机
1. 对话搜索中用户查询往往模糊、不完整，需要结合历史上下文进行查询重写（CQR）以明确搜索意图。
2. 现有训练时微调方法（如SFT、DPO）在最佳重写结果上训练并未带来一致的性能提升，表明单纯训练时优化存在瓶颈。
3. 现有测试时适配方法（如均值聚合、自洽策略）与理论Oracle上限仍存在显著差距，Best-of-N范式的潜力尚未被充分挖掘。
4. 如何设计合适的测试时扩展范式以充分利用基于提示的查询重写方法，并兼容黑盒商业LLM系统，是一个亟待解决的问题。

## 核心贡献（创新点）
1. 首次系统性地发现并分析了基于提示的查询重写在测试时Best-of-N范式下的潜力与不足，揭示了训练时微调与测试时适配各自的局限性。
2. 提出AdaRewriter框架，通过训练轻量级BERT-sized的对比排序损失奖励模型，在推理时为每个重写候选分配分数并选取最优结果，实现了测试时适配。
3. 该方法可作为即插即用模块无缝集成到各种对话搜索系统中，包括使用商业LLM API的黑盒系统，无需访问底层模型参数。
4. 在五个对话搜索数据集上的实验表明，AdaRewriter在大多数设置下显著优于现有方法，证明了测试时适配在查询重写任务中的有效性。

## 方法详解
1. **奖励模型训练**：
   - 采用对比排序损失（Contrastive Ranking Loss）：L = Σ_i Σ_{j>i} max(0, r_j - r_i + (j-i)×λ)，其中r_i是第i个候选的分数，λ是控制候选间边界的超参数（设为0.1）。
   - 候选生成：使用基础LLM（Llama2-7B/Llama3.1-8B）根据对话历史q和H生成N个候选重写结果{S_(1), S_(2), ..., S_(n)}，每个候选由重写查询q̂和伪响应r̂拼接而成。
   - 排名评估：采用端到端评分函数M(S_(i)) = 1/r_s(S_(i), p) + 1/r_d(S_(i), p)，其中r_s和r_d分别表示在稀疏检索（BM25）和密集检索（ANCE）系统中给定查询S_(i)时黄金段落p的排名，将候选按此融合分数排序作为奖励模型的训练标签。

2. **Best-of-N推理**：
   - 给定对话{q, H}，LLM生成N个候选重写结果。
   - 奖励模型g_θ（基于deberta-v3-base）为每个候选分配分数r = g_θ(S, {q, H})，选择得分最高的候选作为最终重写结果：S ← S_(k), k = argmax_{j=1,...,N} g_θ(S_(j), {q, H})。
   - 该过程完全在推理时进行，无需修改底层LLM，可作为即插即用模块。

## 实验与结果
1. **数据集**：TopiOCQA、QReCC（用于训练和评估）、TREC CAsT 2019/2020/2021（用于零样本评估）。
2. **评估指标**：MRR、NDCG@3、Recall@10。
3. **主要结果**：
   - 在TopiOCQA稀疏检索上，AdaRewriter (Llama3.1-8B, N=16) 达到MRR 30.7，显著优于LLM4CS的24.5；密集检索上达到40.3 vs 35.4。
   - 在QReCC稀疏检索上，MRR从N=5时的54.0提升到N=16时的56.2；密集检索上从51.3提升到53.8，显示候选数量增加带来的稳定增益。
   - 在TREC CAsT零样本实验中，AdaRewriter在大多数指标上显著优于基线，如在CAsT-2020上MRR达到63.0，优于SFT的59.1和DPO的60.7。
   - 与训练时微调方法（SFT、SFT with CoT、DPO）对比，AdaRewriter在TopiOCQA上MRR达到40.3（vs SFT 39.2，DPO 39.1），在CAsT 2020的R@10达到63.0（vs SFT 59.1，DPO 60.7）。
   - 在商业黑盒模型GPT4o-mini上，AdaRewriter也能有效提升性能，如在TopiOCQA稀疏检索R@10从48.2提升到51.4，密集检索从58.0提升到63.0。

## 相关工作脉络
1. **ConvGQR/EDIRCS/T5QR**：基于T5的生成式查询重写方法，需要专门训练模型且依赖本地部署，而AdaRewriter无需修改LLM即可提升性能，适用于黑盒系统。
2. **LLM4CS**：测试时聚合多个候选的基线方法（均值聚合、自洽），但未能充分利用候选间的差异，AdaRewriter通过奖励模型实现更精准的筛选。
3. **RETPO/AdaCQR**：训练时对齐方法，依赖本地模型微调，而AdaRewriter可作为即插即用模块用于黑盒系统，且具有更强的可扩展性。
4. **CHIQ**：通过增强历史上下文来提升重写质量，侧重于上下文优化而非候选筛选，两者可互补。
5. **Conversational Dense Retrieval (CDR)**：如ChatRetriever等方法训练密集编码器，缺乏可解释性，而AdaRewriter提供可解释的查询重写结果。

## 局限性与未来方向
1. **延迟瓶颈**：主要延迟来源于使用LLM生成多个候选的过程，尽管奖励模型轻量，但候选生成仍是计算密集型操作。
2. **候选数量N的权衡**：增加N可以提升性能但会线性增加计算成本，论文因预算限制未能在商业大模型上测试更大的N值。
3. **长对话鲁棒性**：随着对话轮次增加，所有方法性能均下降，AdaRewriter虽表现更稳健但仍需进一步优化。
4. **未来方向**：可结合推理加速技术（如投机解码）降低延迟；探索动态分配计算资源的策略，根据任务难度调整候选数量。

## 研究启发与可借鉴点
1. **测试时适配范式**：将对比排序损失应用于轻量级奖励模型，为其他生成任务的测试时优化提供了新思路，无需重新训练基础模型。
2. **黑盒兼容设计**：方法不依赖底层LLM的内部结构，可直接应用于商业API，提高了实用价值和落地可行性。
3. **隐式奖励信号构建**：通过融合稀疏/密集检索排名构建隐式监督信号，避免了对话搜索中缺乏显式标注的问题，可迁移至其他检索增强任务。
4. **即插即用模块化**：框架设计为独立模块，可灵活集成到现有对话搜索管道中，降低了部署复杂度。

## 关键术语表
**Conversational Query Reformulation (CQR)**：将当前查询和历史上下文转化为独立可检索查询的任务。
**Best-of-N Paradigm**：生成N个候选结果并选择最优的一个进行评估或使用的策略。
**Reward Model**：通过训练评估生成结果质量的轻量级模型，此处用于对候选重写结果打分。
**Contrastive Ranking Loss**：通过比较不同候选的相对顺序来训练的损失函数，适用于隐式反馈任务。
**Test-time Adaptation**：在推理阶段利用额外计算或轻量模型提升性能的技术，区别于训练时微调。
**Black-box LLM**：无法访问内部参数的商用大语言模型，通常通过API调用。

## 可复现要素
- **数据集**：TopiOCQA、QReCC、TREC CAsT 2019/2020/2021（均为公开数据集）
- **代码/权重**：论文未明确说明开源，但提到使用Huggingface Transformers和PyTorch Lightning实现，核心组件（deberta-v3-base）可从Huggingface获取
- **关键超参**：候选数量N=5/16，温度0.7，学习率5e-6，cosine schedule warmup ratio 0.1，训练10 epochs，margin λ=0.1，BM25参数k1=0.82,b=0.68（QReCC）和k1=0.9,b=0.4（TopiOCQA）
