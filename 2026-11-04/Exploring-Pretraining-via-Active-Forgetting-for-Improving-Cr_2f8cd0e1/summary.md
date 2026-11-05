---
title: "Exploring-Pretraining-via-Active-Forgetting-for-Improving-Cr"
source: https://aclanthology.org/2025.emnlp-main.120.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:29:25"
field: "多语言大语言模型"
keywords: ["跨语言迁移", "Active Forgetting", "Decoder-only LLM", "多语言预训练", "语言适配"]
innovations: ["首次将active forgetting预训练扩展至decoder-only架构", "揭示词汇扩展适配的跨语言代价并给出缓解方案"]
benchmarks: ["FLORES-200", "Belebele", "MLQA", "XCOPA", "XQuAD", "XLSum", "mC4"]
---

# 论文速读：Exploring-Pretraining-via-Active-Forgetting-for-Improving-Cr

## 一句话总结
本文提出对decoder-only LLM进行**active forgetting预训练**，使其在仅用英语指令微调后，对未见语言展现更强的跨语言迁移能力，并在6/7的多语言基准上优于基线方法。

## 研究问题与动机
- **核心问题**：decoder-only自回归LLM在英语以外语言的跨语言迁移能力显著弱于encoder-only模型（如BERT、XLM-RoBERTa）。
- **现有方法的不足**：常见的语言适配策略（通过词汇扩展+冻结Transformer栈只训练新增token embeddings）只能提升目标语言表现，却会显著损害模型在其他语言上的整体性能（"multi-linguality curse"）。
- **动机来源**：Chen et al. (2024) 已证明active forgetting可提升encoder-only模型的跨语言迁移，但decoder-only模型尚未被系统研究。

## 核心贡献（创新点）
1. **揭示语言适配的代价**：指出标准词汇扩展适配方法虽能提升目标语言，但会显著降低整体多语言性能（μ_overall下降）。
2. **首次将active forgetting引入decoder-only LLM预训练**：证明其对自回归架构同样有效，能产生更高质量的多语言表征。
3. **实证跨语言迁移提升**：AFA模型在6/7多语言基准（Belebele、MLQA、XCOPA、XQuAD、XLSum、FLORES-200翻译）上优于Baseline和BA模型，且无需目标语言指令数据。

## 方法详解
- **Active Forgetting机制**：每k步（实验中k=10,000步）将模型的token embeddings重置为随机初始化，模拟"遗忘"过程，阻止特定语言表征过度固化。
- **三阶段适配流程**：
  1. 在12种预训练语言Wikipedia上以active forgetting训练base model $\mathcal{M}_{base}$
  2. 新增14种适应语言词汇（V_merged=48000），冻结Transformer栈，仅训练新token embeddings和language modeling head得到$\mathcal{M}_{adapted}$
  3. 仅用英语OpenOrca指令数据（2.91M样本）微调得到$\mathcal{M}_{adapted}^{finetuned}$
- **评估指标**：perplexity（mC4，50种语言）、isotropy（上下文向量自相似度，越低越好）、多语言基准准确率/F1/BLEU。

## 实验与结果
- **数据集**：预训练语言12种（ar,zh,cs,en,et,fi,he,hi,it,ru,es,sw），适应语言14种（ja,fr,pt,nl,se,tr,da,no,ko,pl,hu,th,mr,g u），其他语言26种用于评估。
- **模型规模**：400M、782M、1.6B、2.8B参数（Mistral架构缩放版）。
- **关键结果**：
  - **Perplexity**：AFA在所有规模上均优于Baseline和BA（如2.8B模型：AFA μ_overall=25.621 vs BA=27.034）
  - **Isotropy**：AFA表征分布更均匀（2.8B：AFA=0.505 vs BA=0.507）
  - **多语言基准**：AFA在6/7任务上获胜，翻译任务（FLORES-200）2.8B模型μ_adapting=0.288 BLEU显著优于BA=0.255
  - **无"适配代价"**：BA模型在"other"语言上性能下降明显，AFA无明显下降

## 相关工作脉络
1. **Chen et al. (2024)**：提出encoder-only模型的active forgetting预训练，本文将其扩展至decoder-only架构。
2. **Shaham et al. (2024)**：发现多语言模型窄适配会损害整体性能，本文验证此现象并给出缓解方案。
3. **Minixhofer et al. (2024)**：zero-shot tokenizer transfer适用于编程等任务，但在多语言场景的研究不足。
4. **Conneau et al. (2020)**：XLM-R等encoder模型已展示强跨语言迁移，本文弥补decoder模型的这一差距。
5. **Pfeiffer et al. (2020)、Parović et al. (2023)**：针对encoder模型的跨语言adapter方法，本文从预训练层面解决问题。

## 局限性与未来方向
- **方法无法直接应用于已有LLM**：需从预训练阶段介入，不能事后 retrofit到如Llama、Mistral等开源模型。
- **数据规模有限**：仅10B tokens，远小于XLM-R的100B+ tokens，结论需在更大规模上验证。
- **模型规模受限**：实验最大2.8B参数，需验证在百亿/千亿级模型上的效果。
- **未来方向**：探索对中间checkpoint应用active forgetting的模拟方法，以及适配已有指令微调模型的语言扩展策略。

## 研究启发与可借鉴点
1. **主动遗忘作为正则化手段**：周期性重置embeddings可防止表征过拟合特定语言分布，这一思想可迁移至其他领域（如领域适应、持续学习）。
2. **预训练策略决定下游迁移潜力**：本文证明改进预训练阶段比后期适配更能提升跨语言能力，值得在多语言研究中优先考量。
3. **Isotropy作为表征质量指标**：低isotropy对应更好的跨语言泛化，可作为预训练质量的轻量级监控指标。
4. **英语指令微调的跨语言潜力**：仅用英语数据微调即能激活多语言能力，降低了多语言指令数据的收集成本。

## 关键术语表
- **Active Forgetting**：预训练中周期性将token embeddings重置为随机值，防止特定表征过度固化
- **Cross Lingual Transfer**：模型在仅用源语言数据训练/微调后，对目标语言任务的零样本迁移能力
- **Isotropy**：上下文嵌入向量的分布均匀性度量，越低表示表征区分度越好
- **Curse of Multilinguality**：多语言训练中某些语言表征挤压其他语言表现的现象
- **BA (Baseline Adapted)**：标准词汇扩展适配后微调的模型
- **AFA (Active Forgetting Adapted)**：经active forgetting预训练后再适配的模型

## 可复现要素
- **数据集**：Wikipedia dumps（Shaham et al., 2024）、OpenOrca（Lian et al., 2023）、mC4（Xue et al., 2021）、FLORES-200（NLLB Team et al., 2022）、Belebele、MLQA、XCOPA、XQuAD、XLSum
- **代码/权重**：论文未提供开源代码和模型权重
- **关键超参**：预训练学习率1e-4、150K步、batch size 128、block size 4096、每10K步重置embeddings；微调学习率1e-6、5 epochs、batch size 16
