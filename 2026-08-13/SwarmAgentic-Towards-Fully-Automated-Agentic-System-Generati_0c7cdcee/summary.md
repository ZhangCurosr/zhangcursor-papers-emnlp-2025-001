---
title: "SwarmAgentic-Towards-Fully-Automated-Agentic-System-Generati"
source: https://aclanthology.org/2025.emnlp-main.93.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:40:41"
field: "多代理系统自动化设计"
keywords: ["Multi-Agent System", "Particle Swarm Optimization", "Automated Agent Design", "LLM-driven Search", "Agentic System Generation"]
innovations: ["将PSO转化为语言驱动的象征性优化过程", "首个完全从零生成代理系统的框架", "失败感知速度更新机制避免重复错误"]
benchmarks: ["TravelPlanner", "Natural Plan", "Creative Writing", "MGSM"]
---

# 论文速读：SwarmAgentic-Towards-Fully-Automated-Agentic-System-Generati

## 一句话总结
本文提出 SwarmAgentic，首个完全自动化的多代理系统生成框架，将粒子群优化(PSO)转化为语言驱动的象征性搜索过程，从零开始自动生成代理角色、执行策略及协作结构，无需人工干预。

## 研究问题与动机
- 现有代理系统设计框架缺乏完全自主性，无法从任务描述**从零开始**生成代理实体
- 多数方法依赖**预定义模板、种子代理或固定拓扑结构**，限制了对开放型、结构性不确定任务的适应能力
- 现有框架要么仅优化代理功能，要么仅优化协作结构，**无法联合优化**两者
- 在旅行规划、创意写作等高阶规划任务中，手工设计协作策略成本极高且难以扩展

## 核心贡献（创新点）
- **语言驱动的 PSO 框架**：将连续向量空间中的粒子运动重新定义为结构化文本空间的语义变换，实现可解释的符号化优化
- **零干预从出生成**：无需任何预定义代理或模板，仅依赖任务描述和目标函数自动实例化代理集合与协作结构
- **失败感知速度更新机制**：引入历史失败记忆项，通过 LLM 识别无效调整并避免重复错误，平衡个体学习与社会学习
- **联合优化代理功能与协作结构**：同时迭代调整代理的角色/职责/执行策略，以及工作流步骤的顺序与依赖关系

## 方法详解
- **粒子表示**：每个粒子编码一个完整的代理系统 $S_i^{(t)} = (\mathcal{A}_i^{(t)}, \mathcal{W}_i^{(t)})$，其中 $\mathcal{A}$ 为代理集合，$\mathcal{W}$ 为协作结构（任务步骤序列）
- **初始化策略**：采用温度控制采样，低温粒子保持结构稳定，高温粒子探索非传统架构，维持多样性
- **缺陷识别 (Flaw Identification)**：LLM 分析执行失败模式，归类为代理缺陷（缺失/冗余/策略模糊）或协作结构缺陷（缺失步骤/冗余步骤/输入不充分/输出错配）
- **速度更新公式**：
  - $v_i^{(t+1)} = LLM_{vel}(c_f r_f F(v_i^{(t)}), c_p r_p (p_i^* - x_i^{(t)}), c_g r_g (g - x_i^{(t)}))$
  - 包含三部分组成：**失败驱动调整**（基于历史无效更新）、**个人最佳引导**（自我学习）、**全局最佳引导**（群体学习）
- **位置更新**：通过 $LLM_{pos}$ 将速度向量转化为对角色和流程的具体修改指令，包括添加/删除/修改代理及步骤

## 实验与结果
- **数据集**：TravelPlanner、Natural Plan（Trip Planning/Meeting Planning/Calendar Scheduling）、Creative Writing、MGSM 共6个任务
- **最强结果**：TravelPlanner 最终通过率(GPT-4o)达 **32.2%**，较 ADAS 提升 **+261.8%**（相对提升）
- **其他任务优势**：Creative Writing 得分 8.5 vs ADAS 7.3；MGSM 准确率达 88.4%
- **跨模型迁移**：在 GPT-4o-mini 上优化的系统在 Claude-3.5-sonnet、DeepSeek-V3、Gemini-1.5-Pro 上仍优于所有基线
- **消融验证**：移除失败驱动调整导致性能下降35.5%，验证该组件关键性

## 相关工作脉络
- **ADAS**：基于代码的元代理搜索框架，但需7个预定义种子代理，本文在此基础上实现完全从零生成
- **EvoAgent**：使用进化算法优化代理配置，但起始于预定义的专业代理模板，本文无此限制
- **AgentSquare**：在模块化设计空间搜索单代理架构，无法优化多代理协作拓扑
- **AutoAgents**：依赖4个预定义管理器角色（Planner/Agent-Observer/Plan-Observer/Action-Observer），本文消除此类先验假设
- **MODEL SWARMS**：在模型权重空间进行粒子群优化，输出单一适配模型；本文在语言空间优化系统结构，输出可解释的多代理架构

## 局限性与未来方向
- 缺乏**领域特定归纳偏置**（如模板约束），在结构化环境中收敛速度可能较慢
- 继承 LLM 固有的**事实可靠性问题**，幻觉可能通过优化循环传播累积
- 纯文本环境运行，**缺乏多模态感知与物理交互能力**，限制在具身场景的应用
- 未来可结合约束引导搜索以平衡效率与开放性

## 研究启发与可借鉴点
- **语言驱动 PSO**：将传统数值优化转化为可解释的语言变换，为符号空间搜索提供新范式
- **温度分层初始化**：按不同温度参数生成多样化初始结构，兼顾开发(exploitation)与探索(exploration)
- **失败记忆机制**：记录并识别历史无效调整，避免重复错误，提升搜索效率
- **可迁移性验证思路**：在基础模型上训练系统后迁移至目标模型测试，可评估架构泛化能力
- **成本分析视角**：并行粒子搜索降低有效延迟，比顺序迭代方法更高效

## 关键术语表
- **SwarmAgentic**：基于粒子群优化的全自动多代理系统生成框架
- **Particle Swarm Optimization (PSO)**：受群体智能启发的优化算法，通过个体与群体最佳位置引导搜索
- **Failure-Aware Velocity Update**：结合历史失败经验的更新机制，识别无效调整并避免重复
- **From-Scratch Agent Generation**：完全从零生成代理角色与协作结构的能力
- **Self-Optimizing Collaboration**：自动重构代理间工作流步骤与依赖关系的能力

## 可复现要素
- **数据集**：TravelPlanner、Natural Plan、Creative Writing、MGSM（均为公开数据集）
- **代码开源**：https://github.com/SwarmAgentic
- **模型配置**：优化器使用 GPT-4o-mini-0718，执行器使用 GPT-3.5-turbo-0125/GPT-4o-0806/Claude-3.5-sonnet-0620/DeepSeek-V3/Gemini-1.5-Pro
- **超参数**：5个粒子，10次优化迭代（ADAS为30次）
