---
title: "EQA-RM-A-Generative-Embodied-Reward-Model-with-Test-time-Sca"
source: https://aclanthology.org/2025.emnlp-main.48.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:47:39"
field: "具身AI与奖励建模"
keywords: ["Reward Model", "Embodied Question Answering", "Generative RM", "Test-time Scaling", "Contrastive RL"]
innovations: ["首个专为EQA设计的生成式多模态奖励模型EQA-RM", "提出C-GRPO对比组相对策略优化训练策略", "测试时缩放使小模型超越商用大模型评估精度"]
benchmarks: ["EQA REWARDBENCH"]
---

# 论文速读：EQA-RM: A Generative Embodied Reward Model with Test-time Scaling

## 一句话总结
EQA-RM是首个专为具身问答(EQA)任务设计的生成式多模态奖励模型，通过创新的对比组相对策略优化(C-GRPO)策略训练，能够输出可解释的结构化批评与标量分数，并支持测试时缩放以提升评估精度。

## 研究问题与动机
- 现有通用奖励模型主要针对静态输入或简单输出设计，在动态交互式领域（如EQA）表现不足，无法捕捉时空和逻辑依赖关系
- EQA任务要求智能体在3D环境中感知、交互并推理，评估轨迹需要细致考量推理连贯性、动作适当性和语言 grounding，而现有方法仅关注最终答案正确性
- 该领域缺乏标准化的奖励模型评估基准，阻碍了奖励模型的系统性发展和比较

## 核心贡献（创新点）
- **EQA-RM架构**：提出首个专为EQA设计的生成式多模态奖励模型，能同时输出文本批评和标量分数，具备可解释性和测试时缩放能力
- **C-GRPO训练策略**：设计对比组相对策略优化，通过三种扰动（时序打乱、空间掩码、推理步骤重排）的对比奖励训练模型，使其内化时空和逻辑理解
- **EQA REWARDBENCH基准**：构建首个EQA奖励模型专用基准，包含1,546个测试实例，覆盖8个轨迹质量评估维度
- **测试时缩放验证**：展示生成式奖励模型可通过增加推理计算量（K=32）将准确率从42.47%提升至61.86%，超越商用大模型

## 方法详解
**两阶段训练框架：**
1. **拒绝式微调(RFT)**：
   - 辅助LLM评估器生成N_RFT个候选评估（批评+分数），基于无GT答案的输入
   - 拒绝过滤：剔除"太简单"（所有候选都正确）和"错误"的样本（|s_gt - s_aux| < τ为正确）
   - 对过滤后的数据集D_RFT进行SFT，损失函数为标准负对数似然
   
2. **C-GRPO对比强化学习**：
   - 设计三类数据增强扰动：①时序打乱视频帧(v^t) ②空间区域随机掩码(v^s) ③推理步骤重排(z^r)
   - 基础奖励：Accuracy Reward R_a,i = max(0, 1 - (|s_r,i - s_gt,i|/10)²) 和 Format Reward R_f
   - 对比奖励：若批量原始准确率≥δ·批量扰动准确率(δ=0.95)，则给予boost(μ=0.3)
   - 总奖励：R_i^A = R_a^o + R_f^o + (R^t + R^s + R^r)/3
   - 优势计算：组内归一化 A_i = (R_i^A - mean) / (std + ε)
   - 目标函数：标准GRPO形式，含 clipped surrogate term 和 KL散度正则项(β_K=0.04)

**测试时缩放(TTS)**：
- 温度0.8、top-p=0.9采样K条推理路径（K∈{1,2,4,8,16,32}）
- 通过多数投票或平均奖励聚合

## 实验与结果
**数据集与划分：**
- EQA REWARDBENCH(D_B)：823个HM3D实例 + 713个ScanNet实例（共1,546个测试）
- 微调集(D_F)：697个ScanNet实例，与测试集无重叠场景

**评估指标：**
- Accuracy（允许误差τ=2）和RMSE

**主要结果：**
- Base EQA-RM（微调Qwen2-VL-2B-Instruct）：Overall Acc 42.47%，优于基座(+9.39%)，RMSE 2.953
- **EQA-RM w/ TTS(K=32)**：**Overall Acc 61.86%**，超越Gemini-2.5-Flash(59.79%)和GPT-4o(58.45%)，ScanNet Acc达**65.06%**，HM3D Acc达**58.65%**
- 在Object Localization、Attribute Recognition、Functional Reasoning等子任务上达最优
- ablation显示：RFT预热关键（无RFT时RL普遍不如下限），推理奖励R_r单独贡献最大(+9.32%)

## 相关工作脉络
- **Generative Reward Models**：如Genprm、Generative Verifiers，本文差异在于专为EQA设计，引入时空对比增强
- **VLM-as-a-Judge**：如Gemini-2.5-Flash、GPT-4o直接评估，本文模型小得多(2B)但经专门训练可超越
- **Visual Reward Models**：RoVRM、VisualPRM为通用视觉RM，缺乏对EQA时空推理的针对性优化
- **Rule-based RL for LLMs**：GRPO类方法如DeepSeekMath、Logic-RL，本文创新在于将对比增强引入RM训练
- **EQA评估**：现有EQA基准（如OpenEQA）仅评估最终答案，本文首次建立轨迹级RM评估基准

## 局限性与未来方向
- 三类对比增强为预定义，可能无法覆盖EQA中所有细微行为和失败模式
- 高质量GT评分依赖商业模型(Gemini-2.5-Pro)且需人工验证，可能存在偏见
- 向非OpenEQA风格任务/环境泛化能力未验证
- 批评文本的细粒度可信度和对策略优化的直接效用未系统评估
- 未来：自适应增强策略、开源模型替代GT生成、批评驱动的RL策略改进、扩展基准多样性

## 研究启发与可借鉴点
- **C-GRPO对比训练范式**：通过精心设计的扰动（时序/空间/逻辑）制造对比信号，可有效训练模型理解复杂依赖关系，可迁移至视频理解、机器人决策等时空推理任务的RM训练
- **测试时缩放应用于RM**：生成式RM结合多路径采样聚合，可用额外计算换取更高评估精度，适用于对评估质量要求高的场景
- **RFT拒绝过滤策略**：通过筛选"有挑战性但正确"的样本进行SFT预热，避免纯RL训练的不稳定，为其他RM训练提供数据筛选思路
- **结构化批评输出**：不仅给分还提供可解释分析，支持后续调试和策略改进，值得在需要可解释性的RM设计中借鉴

## 关键术语表
**EQA (Embodied Question Answering)**：具身问答，要求智能体在3D环境中通过感知、导航和交互来回答问题的任务
**Reward Model (RM)**：奖励模型，用于量化评估大模型输出或动作质量的模型
**Generative RM**：生成式奖励模型，输出结构化文本批评加标量分数而非仅单一标量
**C-GRPO**：对比组相对策略优化，利用数据增强扰动构建对比奖励的强化学习训练策略
**Test-time Scaling**：测试时缩放，通过增加推理计算预算（多路径采样）提升模型性能
**RFT (Rejective Fine-Tuning)**：拒绝式微调，通过过滤不合适样本构建SFT数据集的方法
**Grounding**：指语言描述与视觉感知的对应关系，评估模型是否能正确关联语言与视觉信息

## 可复现要素
- **数据集**：EQA REWARDBENCH基于OpenEQA（含HM3D和ScanNet环境）构建
- **代码开源**：https://github.com/UNITES-Lab/EQA-RM
- **模型开源**：基于Qwen2-VL-2B-Instruct微调
- **关键超参**：LR=1e-6，Batch=1×1，BF16精度，Max Seq Len=1024，β_K=0.04，δ=0.95，μ=0.3，Mask Size=(16,16)，Mask Ratio=0.15，TTS K∈{1,2,4,8,16,32}，Temperature=0.8，Top-p=0.9
- **硬件**：GPU per Node=8，DeepSpeed ZeRO Stage 3
