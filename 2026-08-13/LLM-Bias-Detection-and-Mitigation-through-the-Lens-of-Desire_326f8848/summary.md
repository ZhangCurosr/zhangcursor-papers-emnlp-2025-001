---
title: "LLM-Bias-Detection-and-Mitigation-through-the-Lens-of-Desire"
source: https://aclanthology.org/2025.emnlp-main.76.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:45:16"
---

# 论文速读：LLM-Bias-Detection-and-Mitigation-through-the-Lens-of-Desire

## 一句话总结
本文提出基于“期望分布”对齐的LLM偏见新范式，将偏见统一定义为模型输出分布与目标分布（平等50-50或现实统计）的偏离；通过引入加权自适应KL损失进行微调，在保持语言建模能力的前提下，于MLM上实现平等等效分布下近完全消除（>98%）与现实分布下30%-75%的偏见降幅，于ALM上实现50%-62%的显著降低。

## 研究问题与动机
- **视角缺失**：现有CS偏见研究几乎全部聚焦于“促进人口统计均等/ démographique parity”，忽视了“对齐现实世界分布”这一在医疗、精准医学等事实敏感场景中至关重要的视角。
- **定义冲突**：计算机科学将任何属性-目标关联差异视为偏见，而心理学认为反映真实社会结构的关联是准确的；两者在“现实分布即偏见”问题上存在根本分歧。
- **方法局限**：现有微调去偏方法多针对细粒度词对关联，缺乏对粗粒度群体分布的直接优化，且往往以严重退化语言建模能力（MLM损失飙升>1000%）为代价。
- **统一框架需求**：亟需一种可根据应用目标灵活切换“平等”或“现实”分布，并能在对齐分布的同时保留基础语言能力的微调机制。

## 核心贡献（创新点）
- **重定义偏见度量**：将偏见从“属性关联差异”转变为“输出分布与期望分布的KL散度”，首次在同一框架下兼容平等分布与现实分布两种对齐目标。
- **加权自适应KL损失**：设计批次级动态归一化与方差感知加权机制，使不同性别主导程度（male/female/balanced）的职业组获得均衡且稳定的梯度更新，与静态均匀加权或注意力头修改方法本质不同。
- **双目标解耦微调协议**：针对MLM引入二次MLM损失以缓解能力退化，针对ALM利用句子损失自然代理困惑度无需额外项，系统证明了分布对齐与语言建模能力可兼得。

## 方法详解
- **数据与分组**：采用美国劳工统计局(BLS, 2024)的225个职业，按女性参与率划分为男性主导(DP_male, 0-30%)、女性主导(DP_female, 70-100%)与性别平衡(DP_balanced, 45-55%)；搭配6个模板与11对二元性别词。
- **偏见度量**：MLM通过模板中性别词mask后的log-likelihood比计算关联分数，聚合后softmax得到预测性别分布 $p_{pred}$；ALM直接使用句子损失的负指数作代理。Bias Score为 $p_{pred}$ 与期望分布 $p_{true}$ 间KL散度的职业均值。
- **非自适应均匀KL损失** ($\mathcal{L}_{\mathrm{KL,uniform}}$)：$\frac{1}{|\mathcal{R}|}\sum_{r} D_{KL}(p_{true}^{(r)} \| p_{pred}^{(r)})$，所有职业等权，专用于平等分布对齐。
- **加权自适应KL损失** ($\mathcal{L}_{\mathrm{KL,weighted\_adaptive}}$)：核心设计分三层：
  1. **自适应归一化**：$\hat{\mathcal{L}}_{cur}^{(c)} = \mathcal{L}_{cur}^{(c)} / (\mu_{KL,new}^{(c)} + \alpha^{(c)})$，其中 $\mu_{KL,new}^{(c)} = \beta \mu_{KL,old}^{(c)} + (1-\beta)\mathcal{L}_{cur}^{(c)}$ 为指数移动平均，防止高KL组主导梯度。
  2. **自适应缩放**：常数 $\alpha^{(c)}$ 对高初始KL组取较小值（激进更新），对低KL组取较大值（保守更新）。
  3. **稳定性感知加权**：基于Welford在线算法维护组内KL方差 $\mathrm{Var}^{(c)}$，计算 $\mathrm{VarFactor}^{(c)} = 1/(1+\mathrm{Var}^{(c)})$，方差大（不稳定）则降权、方差小则加权，动态调节各组更新速率。
- **能力保留**：MLM总损失 $\mathcal{L} = \mathcal{L}_{KL} + \gamma \cdot \mathcal{L}_{MLM}$；ALM因句子损失已蕴含困惑度，直接使用 $\mathcal{L}_{KL}$ 微调。大规模ALM采用LoRA/QLoRA (4-bit NF4) 进行参数高效微调。

## 实验与结果
- **设置**：MLMs (DistilBERT, BERT-base, BERT-large)；ALMs (Llama3.2-3B-Instruct, Llama3.1-8B-Instruct)。平等设置下对比 AttenD 与 CDS。数据按 65%-15%-20% 分层切分，训练/验证模板按伪困惑度阈值15平衡常见与稀有模板。
- **MLMs - 平等分布**：均匀KL损失实现近乎完全消除，三类职业及ALL整体Bias Score下降 **>98%** (p<0.05)。MLM损失仅微增2.3%-14.5%，GLUE下游任务保持稳定。显著优于AttenD（MLM损失激增>1000%）与CDS（最高<92%）。
- **MLMs - 现实分布**：加权自适应KL损失使ALL整体偏见降低 **59%-67%**；引入MLM次级损失后，DP_male降幅
