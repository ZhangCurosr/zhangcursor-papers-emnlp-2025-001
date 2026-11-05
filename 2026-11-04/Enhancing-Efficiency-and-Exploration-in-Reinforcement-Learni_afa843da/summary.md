---
title: "Enhancing-Efficiency-and-Exploration-in-Reinforcement-Learni"
source: https://aclanthology.org/2025.emnlp-main.75.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:29:17"
field: "LLM强化学习与对齐"
keywords: ["强化学习", "大语言模型", "rollout分配", "温度调度", "探索与利用", "GRPO", "数学推理"]
innovations: ["动态rollout预算分配机制，根据问题难度自适应调整采样资源", "温度调度器通过调整softmax温度维持策略熵稳定，避免熵正则化有害梯度", "退火机制实现探索到利用的平滑过渡"]
benchmarks: ["AIME 2024", "AMC 2023", "MATH 500", "OlympiadBench"]
---

# 论文速读：Enhancing-Efficiency-and-Exploration-in-Reinforcement-Learni

## 一句话总结
本文针对LLM强化学习中的效率与探索难题，提出了动态rollout预算分配机制和自适应温度调度策略，在保证模型探索能力的同时显著提升训练效率与推理精度。

## 研究问题与动机
1. **rollout分配效率低**：现有方法对简单和困难问题分配相同rollout数，简单问题训练收益有限，而困难问题需要更多采样才能找到正确答案。
2. **探索能力受限**：RL优化以最大化奖励为目标，导致模型概率分布过度集中于高奖励路径，限制了多样性探索；基线模型在大样本评估时反而超过RL训练模型。
3. **熵正则化引入有害梯度**：传统熵正则化（entropy regularization）与稀疏规则奖励结合时，当所有rollout奖励相同时advantage为零，此时策略更新主要受熵梯度驱动，可能导致策略坍缩。

## 核心贡献（创新点）
1. **动态rollout预算分配机制**：基于问题历史平均奖励动态调整每个问题的采样次数，将计算资源从简单问题转移到困难问题，本质区别在于实现了计算资源的自适应再分配而非固定策略。
2. **温度调度器（Temperature Scheduler）**：通过自适应调整softmax温度τ维持策略熵稳定水平，避免熵正则化带来的有害梯度问题，本质区别在于以温度调控替代直接添加熵正则项。
3. **退火机制（Annealing Mechanism）**：在训练中后期逐步降低目标熵水平，实现从探索到利用的平滑过渡，本质区别在于动态平衡而非静态超参设置。

## 方法详解
**动态rollout预算分配**：
- 问题难度建模：记录每道题的累积rollout数$ n_i^c $和累积奖励$ r_i^c $，按平均奖励$ r_i^c / n_i^c $降序排列，归一化排名$ k_i = \text{rank}(q_i) / |\mathcal{D}| $，$ k_i $越大表示问题越困难。
- 预算分配算法（Algorithm 1）：设默认预算G、最小预算$ G_{\min} $、最大预算$ G_{\max} $，总预算$ N_{\text{total}} = B \times G $，首先将每个问题设为$ G_{\min} $，剩余预算按比例$ k_i / \sum k_j $分配，最后贪婪分配余数，确保总rollout数不变。
- 渐进调整：初始$ G_{\min} = G_{\max} = G $，每个epoch结束后$ G_{\max} $递增2、$ G_{\min} $递减2，类似课程学习避免过早过度投入困难问题。

**温度调度器**：
- 目标：通过调整温度τ控制熵缩放，使熵维持在初始水平$ H_{\text{init}} $。
- 核心公式：
$$\tau_{t+1} = \tau_t \times \left(1 + \frac{\tau_t \ln \alpha}{\ln|\mathcal{V}| + \ln(\ln|\mathcal{V}|)}\right)$$
其中$ \alpha = H_{\text{init}} / H_t $，$|\mathcal{V}| $为词汇表大小。
- 公式推导基于小熵近似（附录A），表明熵缩放因子与α近似线性相关。
- 前向传播时logits也除以温度，保证训练与采样分布一致。

**退火机制**：
- 当训练步$ t \geq t_{\text{anneal}} $时，目标熵从$ H_{\text{init}} $逐渐降至$ \eta \cdot H_{\text{init}} $。
- 目标熵计算公式：
$$H_{\text{anneal}}^{(t)} = H_{\text{init}} \cdot \left[\eta + (1-\eta) \cdot \frac{1}{2}\left(1 + \frac{\eta^2}{t^2}\right)\cos\left(\pi \cdot \frac{t - t_{\text{anneal}}}{t_{\text{max}} - t_{\text{anneal}}}\right)\right]$$
- 参数设置：$ \eta = 0.9 $，$ t_{\text{anneal}} = \lfloor 0.6 \times t_{\text{max}} \rfloor $。

## 实验与结果
**数据集**：
- 训练集：10k高质量数学题，来源于DeepScaleR 40k，经Qwen 2.5 Math 7B预采样难度评估后过滤，包含MATH、AIME 1983-2023、Omni-MATH、AMC prior 2023。
- 验证集：512题，来自四个数据集各128题，不重叠。
- 评测集：AIME 2024、AMC 2023、MATH 500、OlympiadBench。

**基线**：GRPO、DAPO。

**主要结果（7B模型，DeepSeek-R1-Distill-Qwen）**：
| 基准 | Pass@1 | Pass@16 |
|------|--------|---------|
| GRPO | 37.5 | 73.33 |
| DAPO | 36.87 | 70.0 |
| Ours | **42.81** | **76.66** |
- AIME 2024上pass@1提升**5.31%**，pass@16提升**3.33%**。
- 四个基准平均pass@1和pass@16均为最优。
- 1.5B模型同样显著优于GRPO，AIME上pass@1提升约3.04%。

**消融实验**：
- 去除动态rollout分配（w/o DS）：AIME 2024 pass@1下降3.02%（7B）。
- 对比熵正则化（ER）：GRPO+ER表现最差，证明温度调度器更优。
- 退火机制：在简单问题上提升pass@1，但在高k值时略低于仅温度调度器。

**训练稳定性**：温度调度器显著降低验证集准确率方差。

## 相关工作脉络
1. **GRPO (Shao et al., 2024)**：本文基线算法，采用组内相对优势估计简化RL训练流程。
2. **DAPO (Yu et al., 2025)**：改进GRPO，过滤组内advantage为零的样本并通过多轮在线采样积累经验，但采样效率低。
3. **Yue et al. (2025)**：发现RL训练后模型在大样本评估（pass@k）时不如基线模型，本文动机来源之一。
4. **Entropy Regularization (Mnih et al., 2016; Williams, 1992)**：深度RL中常用探索技术，但本文发现其与稀疏规则奖励结合会导致策略坍缩。
5. **Zeng et al. (2024, B-Star)**：通过离散温度级别调整维持LLM探索能力，与本文连续温度调度思想相关。
6. **Kimi K1.5 (Team et al., 2025)**：难度平衡训练数据收集策略的参考。

## 局限性与未来方向
1. **难度建模局限**：当前仅基于累积平均奖励建模，可考虑引入更精细的难度指标（如解题步骤复杂度、模型信心度等）。
2. **温度设定**：未探索针对不同难度问题设置不同温度的可能性，作者认为这可能是有价值的方向。
3. **rollout数量限制**：受算力限制$ G=8 $，增大G和$ G_{\max} $可能进一步放大动态分配效果。
4. **实验范围**：仅在GRPO和数学领域验证，虽声称可推广至其他算法和领域但未实证。

## 研究启发与可借鉴点
1. **动态资源分配思路**：可将"根据样本难度动态调整采样/训练资源"的思想迁移至其他RL场景，如代码生成、对话系统等。
2. **温度调度器替代熵正则化**：提供了一种无需引入额外梯度项即可稳定维持熵水平的实用方案，适用于稀疏奖励场景。
3. **渐进式课程学习**：$ G_{\min} $和$ G_{\max} $随epoch渐进扩展的设计思路，可借鉴于其他需要平衡探索与利用的训练策略。
4. **退火与温度联合机制**：结合退火实现探索-利用平滑过渡的设计，对长周期RL训练具有参考价值。

## 关键术语表
**GRPO (Group Relative Policy Optimization)**：组内相对策略优化算法，通过同一问题多个rollout的奖励均值和标准差估计advantage，无需value network。
**Rollout**：给定一个问题采样得到的多个独立响应，用于优势估计和策略更新。
**Pass@k**：对每道题采样k次，至少一次正确的概率，衡量模型探索与发现正确路径的能力。
**熵正则化（Entropy Regularization）**：在目标函数中增加熵项以鼓励策略探索，但本文发现其在稀疏奖励下有害。
**温度调度器（Temperature Scheduler）**：通过动态调整softmax温度τ来维持策略熵稳定，避免有害梯度。
**退火机制（Annealing）**：训练中后期逐步降低目标熵，促使策略从探索转向利用。
**Advantage**：GRPO中通过组内奖励均值标准化计算的优势值，反映某rollout相对于组内平均水平的优劣。

## 可复现要素
- **数据集**：训练集10k（基于DeepScaleR处理），验证集512；测试集公开（AIME 2024、AMC 2023、MATH 500、OlympiadBench）。
- **代码**：开源，https://github.com/LiaoMengqi/E3-RL4LLMs
- **基线模型**：DeepSeek-R1-Distill-Qwen 1.5B/7B
- **训练框架**：VeRL
- **关键超参**：G=8，lr=2e-6（7B），batch=64，epochs=3，η=0.9，$ t_{\text{anneal}} = 0.6 \times t_{\text{max}} $，$ G_{\min}/G_{\max} $每epoch±2
- **硬件**：7B模型8×A100，约8×36 GPU小时
