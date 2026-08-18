---
title: "Exploring-Pretraining-via-Active-Forgetting-for-Improving-Cr"
source: https://aclanthology.org/2025.emnlp-main.120.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:40:25"
field: "多语言大语言模型"
keywords: ["跨语言迁移", "主动遗忘", "多语言语言模型", "decoder架构", "语言适配"]
innovations: ["首次将主动遗忘预训练策略应用于decoder-only LLMs以改善跨语言迁移", "揭示标准词汇扩展适配方法会损害非适配语言性能", "证明主动遗忘可产生更高质量的多语言表征并提升下游任务表现"]
benchmarks: ["Belebele", "MLQA", "XCOPA", "XLSUM", "XQuAD", "FLORES-200翻译"]
---

# 论文速读：Exploring-Pretraining-via-Active-Forgetting-for-Improving-Cr

## 一句话总结
本文提出将"主动遗忘"（Active Forgetting）策略应用于 decoder-only 大语言模型的预训练，以改善其跨语言迁移能力；实验表明，经过主动遗忘预训练的模型仅用英语指令微调即可在多语言基准上超越传统词汇扩展适配方法，且不会损害其他语言的性能。

## 研究问题与动机
- **核心问题**：Decoder-only LLMs（如GPT系列）的跨语言迁移能力显著弱于encoder-only模型（如BERT、XLM-RoBERTa），而后者已通过大量研究展现出优秀的跨语言泛化能力。
- **现有方法不足**：传统的语言适配方法（词汇扩展+仅训练新增token embeddings+英语指令微调）只能提升目标适配语言的表现，却严重损害模型在其他语言上的性能（即"多语言诅咒"现象）。
- **已有工作局限**：Chen et al. (2024) 已证明主动遗忘可提升encoder-only模型的跨语言迁移能力，但该策略在decoder-only自回归模型上的效果尚未被系统研究。
- **研究动机**：探索能否通过改进预训练策略（而非仅在适配阶段优化）来提升decoder-only LLMs的多语言表征质量与跨语言迁移能力。

## 核心贡献（创新点）
1. **首次将主动遗忘应用于decoder-only LLM预训练**：将Chen et al. (2024)在encoder模型上的发现扩展到自回归架构，验证了该策略在不同模型范式下的有效性。
2. **揭示了标准适配方法的有害性**：系统证明"词汇扩展+适配"方法虽然能提升目标语言表现，但以牺牲其他语言性能为代价，导致整体多语言能力下降。
3. **证明了主动遗忘可产生更高质量的多语言表征**：通过perplexity和isotropy等内在指标量化，AFA模型在所有语言类别上均获得更优的表征质量，进而转化为下游多语言任务的全面提升（6/7基准胜出）。

## 方法详解
- **主动遗忘策略**：在预训练过程中，每隔 $k$ 步（本文 $k=10{,}000$）将所有token的embeddings重置为随机值，然后继续正常训练。该操作模拟了"遗忘已学到的语言特定模式"，迫使模型学习更通用、更易迁移的语言表征。
- **三阶段适配流程**：
  1. **词汇扩展**：在新语言语料 $\mathcal{D}_{\mathrm{train}}^{L}$ 上学习新词表 $\mathcal{V}^{L}$，合并到原始词表形成 $\mathcal{V}_{\mathrm{merged}}$。
  2. **语言建模头适配**：仅训练新增token的embeddings和语言建模头，Transformer主体和原有embeddings冻结。
  3. **英语指令微调**：使用OpenOrca（2.91M条英语指令数据）对适配后模型进行指令微调，得到最终模型 $\mathcal{M}_{\mathrm{adapted}}^{\mathrm{finetuned}}$。
- **对比基线**：
  - **Baseline**：未经语言适配、直接用英语指令微调的原始模型。
  - **BA（Baseline Adapted）**：标准适配流程（无主动遗忘）。
  - **AFA（Active Forgetting Adapted）**：采用主动遗忘预训练后进行相同适配流程。

## 实验与结果
- **数据集**：
  - 预训练语料：12种语言的Wikipedia dumps（ar, zh, cs, en, et, fi, he, hi, it, ru, es, sw）。
  - 适配语料：14种新语言Wikipedia dumps（ja, fr, pt, nl, se, tr, da, no, ko, pl, hu, th, mr, gu）。
  - 评估语言：50种语言（含26种"other"语言，既不在预训练也不在适配集合中）。
  - 指令微调数据：OpenOrca（2.91M英语数据点）。
- **评估基准**：Belebele（阅读理解）、MLQA（跨语言问答）、XCOPA（因果推理）、XLSUM（摘要）、XQuAD（提取式问答）、多语言指令遵循任务，以及FLORES-200上的英→多语言机器翻译（BLEU）。
- **关键结果**（2.8B参数模型为例）：
  - **Perplexity**：AFA在预训练语言（20.716）、适配语言（24.858）、其他语言（28.395）和整体（25.621）上均优于BA（20.958 / 25.969 / 30.768 / 27.034）。
  - **Isotropy**：AFA整体isotropy得分0.505，优于BA的0.507（越低越好，表示表征分布更均匀）。
  - **翻译性能**：AFA在适配语言上达到0.288 BLEU，显著高于BA的0.255；在其他语言上0.229 vs BA的0.174。
  - **综合表现**：AFA在6/7个多语言基准测试上超越Baseline和BA，而BA在多个非适配语言上出现性能退化。
- **结论**：主动遗忘预训练产生的表征具有更高的"语言可塑性"，使模型在仅接触英语指令数据的情况下，仍能更好地迁移到未见语言。

## 相关工作脉络
- **多语言LLM构建**：BLOOM（176B参数，100+语言）和PolyLM等通过大规模多语言预训练提升跨语言能力，但训练成本极高；本文聚焦于如何用更少数据/算力获得更好的跨语言迁移。
- **跨语言迁移方法**：Pfeiffer et al. (2020) 的Mad-X、Parović et al. (2023) 的任务适配器等主要针对encoder模型；本文填补了decoder模型在该方向的研究空白。
- **语言适配技术**：Balachandran (2023) 的Tamil-Llama、Cui et al. (2024) 的Chinese-LLaMA通过词汇扩展适配新语言，但本文揭示该方法会损害整体多语言性能。
- **主动遗忘**：Chern et al. (2023) 首次提出该概念用于提升摘要事实性；Chen et al. (2024) 证明其在encoder模型上可提升语言可塑性和跨语言迁移；本文将其扩展至decoder架构。
- **零样本词表迁移**：Minixhofer et al. (2024) 的zero-shot tokenizer transfer无需词汇扩展即可适配新任务（如编程），但其在多语言场景下的有效性未被充分研究。
- **定位差异**：本文不同于"在已有LLM上直接适配"的思路，而是从预训练阶段改进表征质量，为后续适配和迁移提供更好的基础。

## 局限性与未来方向
- **方法适用性限制**：主动遗忘预训练策略无法直接应用于已训练完成的开源LLM（如Llama、Mistral），只能通过获取中间checkpoint并重置embeddings后继续训练来模拟。
- **规模限制**：实验模型最大仅2.8B参数，总训练数据约10B tokens，远小于XLM-R等使用了100B+ tokens、100+语言的大规模模型；在更大模型和数据上的有效性尚待验证。
- **伦理与偏差风险**：主动遗忘直接修改token embeddings，可能改变模型的内在偏见或安全性属性，部署前需进行全面的能力与偏差评估。
- **未来方向**：探索如何在已微调的多语言指令模型上应用类似策略；研究更高效的嵌入重置方案；在更大规模模型和数据集上验证该方法的有效性。

## 研究启发与可借鉴点
1. **主动遗忘作为一种表征正则化手段**：周期性重置embeddings可防止模型过度拟合特定语言的统计模式，这一思想可迁移到其他需要泛化能力的场景（如领域适配、多模态学习）。
2. **"适配代价"的警示**：本文揭示的"提升目标语言却损害其他语言"现象在语言适配领域具有普遍性，团队在进行多语言模型迭代时应建立全面的多语言评估体系，避免局部优化导致整体退化。
3. **表征质量指标的有效性**：Perplexity和isotropy可作为跨语言迁移能力的早期预测指标，可在模型开发周期中用于快速筛选预训练策略。
4. **轻量级跨语言提升路径**：相比构建超大规模多语言模型，通过改进预训练策略（如主动遗忘）结合少量英语指令微调即可实现较好的跨语言能力，为计算资源受限的团队提供了可行方案。
5. **可结合的方向**：将主动遗忘与zero-shot tokenizer transfer（Minixhofer et al., 2024）结合，或探索在指令微调阶段引入多语言数据的最小配比（"pinch of multilinguality"，Shaham et al., 2024）以进一步优化性价比。

## 关键术语表
- **主动遗忘（Active Forgetting）**：在预训练过程中周期性地将模型token embeddings重置为随机值，以打破对特定语言模式的过拟合，促进更通用表征的学习。
- **跨语言迁移（Cross-lingual Transfer）**：模型在仅用单一语言（如英语）数据训练或微调后，仍能泛化到其他未见过语言的能力。
- **Isotropy（各向同性）**：衡量上下文embeddings分布均匀性的指标，低isotropy表示embeddings在向量空间中分布更分散、信息容量更高。
- **词汇扩展（Vocabulary Expansion）**：为支持新语言而向现有词表中添加新token的过程，通常需要重新训练新增token的embeddings。
- **多语言诅咒（Curse of Multilinguality）**：在预训练中混合过多语言会导致每种语言的代表性下降，从而降低模型整体性能的现象。
- **语言可塑性（Language Plasticity）**：模型适应新语言或语言变化的能力，高可塑性意味着模型能更高效地学习未见语言。

## 可复现要素
- **数据集**：Wikipedia dumps（多语言，来源Shaham et al., 2024）、OpenOrca指令数据（Lian et al., 2023）、FLORES-200翻译基准（NLLB Team et al., 2022）、mC4 perplexity评估语料（Xue et al., 2021）；均为公开数据。
- **代码/权重**：论文未明确提供代码开源声明，但使用标准Mistral架构实现；评估使用lm-evaluation-harness库。
- **关键超参**：预训练学习率$1 \times 10^{-4}$、步数150,000、batch size 128、block size 4096、cosine scheduler、10% warmup、AdamW优化器、embeddings每10,000步重置一次；微调学习率$1 \times 10^{-6}$、5 epochs、batch size 16。
- **硬件**：单张NVIDIA A100 80GB GPU，总训练时长约650 GPU小时。
