---
title: "Enhancing-Efficiency-and-Exploration-in-Reinforcement-Learni"
source: https://aclanthology.org/2025.emnlp-main.75.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 15:40:33"
field: "大语言模型强化学习"
keywords: ["RL for LLMs", "GRPO", "temperature scheduling", "rollout budget allocation", "exploration", "mathematical reasoning"]
innovations: ["动态rollout预算分配：按题目难度自适应分配采样预算，总预算守恒", "温度调度器：通过调整softmax温度维持策略熵稳定，避免熵正则化有害梯度", "退火机制：训练后期逐步降低目标熵实现探索到利用的平滑过渡"]
benchmarks: ["AIME 2024", "AMC 2023", "MATH 500", "OlympiadBench"]
---

# 论文速读：Enhancing-Efficiency-and-Exploration-in-Reinforcement-Learni

## 一句话总结
本文针对RL训练LLM时rollout预算分配不均、探索能力受限的问题，提出了动态rollout预算分配机制和自适应温度调度策略，在GRPO基础上显著提升推理大模型在数学难题上的求解精度与多路径探索能力。

## 研究问题与动机
- **rollout预算浪费**：现有方法（如GRPO）对简单和困难题目分配相同数量的rollout，简单题目训练增益有限，而困难题目需要更多采样才能找到正确答案，导致计算资源分配不经济。
- **探索能力退化**：Yue等（2025）指出，RL训练后的模型在小样本评估（small k）时优于base模型，但随着k增大（pass@k），base模型的pass@k反而更高；这是因为RL最大化奖励导致模型过度聚焦高奖励路径，遗漏了其他正确解法。
- **熵正则化失效**：直接将熵正则化（entropy regularization）与稀疏规则奖励结合时，当所有rollout奖励相同时优势值为零，熵正则化的梯度主导更新，可能导致策略崩溃（policy collapse）。
- **DAPO效率低下**：DAPO通过过滤零优势样本并多轮在线采样解决零优势问题，但采样成本极高且大量经验被丢弃，训练效率低下。

## 核心贡献（创新点）
- **动态rollout预算分配机制**：根据题目历史平均奖励建模难度，将rollout预算从简单题目向困难题目倾斜，保证batch总预算不变，使计算资源分配更合理。
- **温度调度器（Temperature Scheduler）**：通过自适应调整softmax温度τ来维持策略熵在稳定水平，避免了熵正则化引入有害梯度的问题，确保探索的同时不损害训练稳定性。
- **退火机制（Annealing Mechanism）**：在训练后期逐步降低目标熵水平，实现从"高熵探索"到"低熵利用"的平滑过渡，平衡探索与利用。
- **系统性实验验证**：在7B和1.5B模型上验证了各组件的有效性，AIME 2024上pass@1提升5.31%、pass@16提升3.33%，且在各benchmark上pass@16均优于GRPO基线。

## 方法详解
- **题目难度建模**：每轮迭代结束后，按题目平均奖励 $r_i^c / n_i^c$ 降序排列，归一化排名 $k_i = \text{rank}(q_i) / |\mathcal{D}|$，$k_i$越大表示难度越高。
- **动态预算分配算法**：设定默认预算G、最小预算$G_{\min}$、最大预算$G_{\max}$；初始时$G_{\min} = G_{\max} = G$，每轮迭代后$G_{\max}$增加2、$G_{\min}$减少2；分配时先给每题分配$G_{\min}$，剩余预算按$k_i$比例分配，上限不超过$G_{\max}$，确保batch总预算$N_{\text{total}} = B \times G$不变。
- **温度调度公式**：$\tau_{t+1} = \tau_t \times \left(1 + \frac{\tau_t \ln \alpha}{\ln|\mathcal{V}| + \ln(\ln|\mathcal{V}|)}\right)$，其中$\alpha = H_{\text{init}} / H_t$，$H_{\text{init}}$为首批平均熵，$H_t$为当前步平均熵，$|\mathcal{V}|$为词表大小；该公式在低熵条件下近似线性控制熵缩放比例。
- **退火目标熵曲线**：当$t \geq t_{\text{anneal}}$时，目标熵$H_{\text{anneal}}^{(t)} = H_{\text{init}} \cdot [\eta + (1-\eta) \cdot \frac{1}{2}(1 + \frac{\eta^2}{t^2})\cos(\pi \cdot \frac{t - t_{\text{anneal}}}{t_{\text{max}} - t_{\text{anneal}}})]$，目标熵从$H_{\text{init}}$平滑衰减至$\eta \cdot H_{\text{init}}$（论文取$\eta = 0.9$，退火起始步$t_{\text{anneal}} = \lfloor 0.6 \times t_{\text{max}} \rfloor$）。
- **训练一致性**：采样和forward propagation时logits均除以温度τ，确保训练与推理分布一致。

## 实验与结果
- **训练数据**：10k高质量数学题（来自DeepScaleR 40k经难度平衡过滤），0.5k验证集（均匀采样自MATH、Omni-Math、AMC、AIME各128题）。
- **评估基准**：AIME 2024、AMC 2023、MATH 500、OlympiadBench；评估指标为pass@1和pass@16。
- **基线模型**：GRPO（默认G=8）、DAPO；本文基于GRPO改进。
- **最强结果（7B模型）**：AIME 2024上pass@1为42.81%（GRPO为37.5%，+5.31%）、pass@16为76.66%（GRPO为73.33%，+3.33%）；四基准平均pass@1为61.95%（GRPO为60.40%）、pass@16为87.64%（GRPO为85.63%）。
- **1.5B模型结果**：同样在AIME 2024上pass@1提升3.04%（27.70% vs 24.66%），四基准平均pass@16提升2.44%（82.20% vs 79.76%）；DAPO在1.5B模型上表现显著差于其他方法。
- **消融实验**：移除动态rollout预算分配（w/o DS）后，7B模型AIME 2024 pass@1下降3.02%；温度调度器对比熵正则化，后者在所有benchmark上均劣于GRPO基线或仅在大k时微弱超越。

## 相关工作脉络
- **GRPO**（Shao et al., 2024）：Group Relative Policy Optimization，本文的基础算法，通过组内相对优势估计避免价值网络；本文在其上增加预算分配和温度调度。
- **DAPO**（Yu et al., 2025）：过滤零优势样本并多轮采样积累经验，本文认为其效率低下，动态预算分配可作为更经济的替代方案。
- **Yue et al.（2025）**：揭示RL训练后pass@k随k增大反超base模型的现象，指出RL限制了探索多样性，本文为此问题提供了系统性解决方案。
- **熵正则化**（Mnih et al., 2016; Williams, 1992）：传统深度RL中常用的探索增强手段，本文证明其与稀疏规则奖励结合时可能引入有害梯度。
- **Zeng et al.（2024）**：通过离散温度级别切换维持探索能力，与本文连续温度调度的思路相关但实现方式不同。
- **DeepSeek-R1**（Guo et al., 2025）：展示了RL+规则奖励在推理任务上的巨大潜力，本文致力于提升此类训练的效率和探索能力。

## 局限性与未来方向
- 未探索按题目难度动态分配不同目标温度，仅统一调度全局温度；作者认为可按难度差异化设置温度。
- 受计算资源限制，G设为8、$G_{\max}$增幅有限，更大G和$G_{\max}$可能进一步放大动态预算的效果。
- 实验仅限定在GRPO和数学推理领域，虽主张方法具通用性，但未在更多算法或领域验证。
- 退火参数η的经验调优较敏感（η=0.8/0.85不稳定），缺乏理论指导。
- 难度建模仅依赖累积平均奖励，未融合题目结构特征或多轮反馈信号。

## 研究启发与可借鉴点
- **预算分配的渐进式扩展**：初始$G_{\min} = G_{\max} = G$，逐步扩展范围，类课程学习思想防止早期策略过分配预算给困难题，这一"冷启动"策略可迁移至其他RL4LLM场景。
- **温度调度替代熵正则化**：通过调整采样温度而非在目标函数中显式加熵项，避免了有害梯度的产生，这一思路可用于任何稀疏奖励下的LLM RL训练。
- **pass@k分层优化**：针对不同k值采用不同策略（退火利于小k、高熵利于大k），提示评估时应关注多k值曲线而非单点指标。
- **零优势检测与预算联动**：动态预算分配天然缓解了零优势问题（困难题获得更多rollout以降低全错概率），无需像DAPO那样丢弃经验。
- **图7所示的全错比例监控**：可通过可视化全错题目比例变化来诊断预算分配效果，这一诊断工具值得复用。

## 关键术语表
- **GRPO（Group Relative Policy Optimization）**：组内相对策略优化算法，通过同一问题多个rollout的奖励均值和标准差估计优势函数，无需价值网络。
- **pass@k**：对每道题采样k次，只要有一次回答正确即算通过，衡量模型多路径探索能力。
- **温度调度（Temperature Scheduling）**：动态调整softmax采样温度τ以控制策略分布的熵水平，避免熵正则化的梯度问题。
- **动态rollout预算分配**：根据题目历史平均奖励排序建模难度，按难度比例分配rollout数量，总预算守恒。
- **退火机制（Annealing）**：训练后期逐步降低目标熵，实现从高探索到低利用的平滑过渡。
- **规则奖励（Rule-based Reward）**：基于答案正确性给出的0/1稀疏奖励，常见于数学推理任务的RL训练。
- **策略熵（Policy Entropy）**：Shannon熵衡量策略分布的不确定性/探索程度，低熵表示策略趋于确定。
- **优势函数（Advantage Function）**：$\hat{A}_{i,t} = (r_i - \text{mean}(\{r_j\})) / \text{std}(\{r_j\})$，组内相对优势，零优势时梯度消失。

## 可复现要素
- **数据集**：训练集10k（基于DeepScaleR 40k过滤构建），验证集512题；论文提供了数据集构建流程，但未公开原始数据下载链接。
- **代码**：已开源，地址 https://github.com/LiaoMengqi/E3-RL4LLMs
- **权重**：使用DeepSeek-R1-Distill-Qwen 1.5B和7B作为base模型（已有公开权重）。
- **关键超参**：batch size=64，默认rollout数G=8，最大响应长度6k tokens，训练3 epochs/480 steps；学习率7B模型为$2 \times 10^{-6}$、1.5B模型为$5 \times 10^{-6}$；$G_{\max}$每epoch+2、$G_{\min}$每epoch-2；退火起始步$t_{\text{anneal}} = \lfloor 0.6 \times t_{\text{max}} \rfloor$，$\eta=0.9$；采样温度1，评估时pass@16采样温度1。
- **硬件**：7B模型用8×NVIDIA A100（约8×36 GPU小时），1.5B模型用4×NVIDIA A100（约4×24 GPU小时），每实验重复3次。
- **框架**：基于VeRL（Sheng et al., 2024）改造。
