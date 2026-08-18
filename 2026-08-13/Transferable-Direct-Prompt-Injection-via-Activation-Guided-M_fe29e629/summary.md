---
title: "Transferable-Direct-Prompt-Injection-via-Activation-Guided-M"
source: https://aclanthology.org/2025.emnlp-main.102.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:43:05"
field: "大语言模型安全与对抗攻击"
keywords: ["Prompt Injection", "Adversarial Attacks", "Black-box Attacks", "Energy-based Models", "MCMC Sampling", "LLM Security"]
innovations: ["首次提出激活值引导的可迁移直接提示注入攻击框架，无需查询目标模型", "构建基于二分类器的隐式能量基模型(EBM)，通过激活值分布量化对抗提示质量", "引入token级MCMC采样自适应优化机制，结合BERT提议与EBM能量评估生成自然对抗提示"]
benchmarks: ["CYBERSECEVAL3", "Tensor Trust", "StruQ"]
---

# 论文速读：Transferable-Direct-Prompt-Injection-via-Activation-Guided-M

## 一句话总结
本文提出了一种基于激活值引导的MCMC采样框架，通过构建能量基模型(EBM)评估对抗提示质量，实现无需查询目标模型的跨模型可迁移直接提示注入(DPI)攻击，在五个主流LLM上达到49.6%攻击成功率。

## 研究问题与动机
- 直接提示注入(DPI)是OWASP LLM威胁榜榜首，攻击门槛低且危害大，但现有白盒/灰盒方法依赖梯度信息，在实际黑盒场景（如云服务）中不可行；现有黑盒方法依赖人工提示或频繁查询，可迁移性差。
- 白盒/灰盒方法（如GCG、AutoDAN）需访问目标模型内部参数或logits，无法应用于云API等黑盒场景；频繁查询目标模型也违反使用政策。
- 人工 crafted 提示存在随机性强、稳定性差、难以绕过防御过滤等问题，且商业模型（如GPT-4o-mini）对常见对抗模式已做硬化防护。
- 现有黑盒攻击方法（如PromptFuzz）依赖多次查询获得稀疏指导信号，缺乏对对抗样本分布的建模能力，难以生成自然且高攻击力的提示。

## 核心贡献（创新点）
1. **首次提出激活值引导的可迁移DPI攻击框架**：利用白盒代理模型的激活值语义指导对抗提示生成，无需查询目标模型，相比GCG/AutoDAN等白盒方法避免了模型过拟合，相比人工提示具备更强可迁移性。
2. **构建基于二分类器的隐式能量基模型(EBM)**：将对抗成功/失败样本的激活值分布建模为EBM，通过能量分数量化提示攻击质量，比纯统计方法（如PromptFuzz）提供更丰富的语义指导信号。
3. **引入token级MCMC采样自适应优化机制**：结合BERT MLM提议新候选与EBM能量评估，通过Metropolis-Hastings接受概率迭代优化提示，相比GCG的梯度搜索和黑盒方法的人工搜索更高效且保持提示自然性。

## 方法详解
**框架流程**：模板数据集构建 → 激活值采集 → EBM训练 → MCMC采样优化 → 黑盒攻击测试。

**模板数据集构建**：
- 从Tensor Trust竞赛数据集中解耦攻击提示为三段：Prefix（误导段）、Infix（注入指令段）、Suffix（系统模拟段）。
- 使用GPT-4o-mini提取组件并替换infix为"[INSERT_HERE]"，去重后获得92个prefix、87个infix、90个suffix。
- 用Qwen2.5-7B-Instruct作为代理模型，随机选取4000个模板组合5个训练任务，生成20000条消息结构，采集各层激活值$x_i$和攻击结果标签$y_i$。
- 通过前向测试筛选top-35高潜力infix，最终模板：85 prefix × 35 infix × 85 suffix。

**EBM训练**：
- 使用二分类器$f_\theta$学习激活值$x$到成功/失败标签$y$的映射，训练交叉熵损失：$\mathcal{L}_{CE}(\theta) = -\sum_{i=1}^{N}[f_\theta(x_i)[y_i] - \log\sum_y \exp(f_\theta(x_i)[y])]$。
- 由分类器logits推导能量函数：$E_\theta(x) = -\log(\exp(f_\theta(x)[0]) + \exp(f_\theta(x)[1]))$，能量越低表示越可能是成功对抗样本。
- 网络结构：两层MLP（1024→256，ReLU激活），AdamW优化器，学习率0.0003，batch size=256，训练100轮，选择第25层激活值对应的EBM（验证损失最低）。

**MCMC采样优化**：
- 初始化候选$X^{(0)}$为种子提示，迭代T步（T等于提示token数）。
- 每步随机选位置$i$，用BERT MLM重新采样：$X'_i \sim p_{MLM}(\cdot|X_{/i}^{(t)})$生成候选$X'$。
- 计算新旧候选能量分$E_{old}$、$E_{new}$，接受概率：$p(X'|X) = \min\left(\frac{e^{-E(X')}p_{MLM}(X'_i|X_{/i})}{e^{-E(X)}p_{MLM}(X_i|X_{/i})}, 1\right)$。
- 若接受且$E_{new} < E(X^*)$则更新最佳样本$X^*$，最终返回能量最低的对抗提示。

## 实验与结果
**数据集与基准**：
- 训练任务：CYBERSECEVAL3的5个任务（任务1-5）；测试任务：任务6-7（跨任务迁移）。
- 目标模型：Qwen2.5-7B-Instruct、Qwen2-7B-Instruct、Llama-3.1-8B-Instruct、Llama-3-8B-Instruct、GPT-4o-mini（闭源）。
- 基线：Human Experts（33条人工提示）、Initial Prompts（种子提示）、GCG-Inject（白盒）、AutoDAN-GA-Inject（灰盒）、PromptFuzz（黑盒查询）。

**主要结果**：
- 模型可迁移性（Table 1）：Ours(Qwen2.5)平均ASR达49.6%，显著优于所有基线；相较人工提示提升34.6%，相较初始种子提升约10%。
- 跨模型表现：GCG-Inject在Llama3.1上ASR从58.6%暴跌至5.55%，Ours仅从71.6%降至44.4%，证明方法不依赖特定模型架构。
- 黑盒挑战性：GPT-4o-mini上人工提示ASR=0%，Ours仍达23.2%，说明能发现商业模型未覆盖的新型对抗模式。
- 任务可迁移性（Table 2）：在未见任务上，Ours(Qwen2.5)平均ASR-T=36.13%，仍优于所有基线。
- 自然性：PPL=127.68，与AutoDAN(130.78)相当，远低于GCG(17464.28)，可绕过PPL过滤。

**可解释性分析**：
- 能量分与ASR呈强负相关（Pearson=-0.979），验证EBM有效性。
- PCA可视化显示成功样本向激活空间右下方聚集，优化后样本从中心临界区向外扩散，表明方法有效引导对抗方向。

## 相关工作脉络
- **GCG (Zou et al., 2023b)**：白盒梯度下降优化后缀，依赖目标模型logits，无法黑盒使用；本文方法无需目标模型查询，通过代理激活值间接指导。
- **AutoDAN (Liu et al., 2024)**：灰盒遗传算法优化，需查询目标模型获取概率分布；本文完全黑盒，仅用代理模型激活值。
- **PromptFuzz (Yu et al., 2024)**：黑盒蒙特卡洛树搜索+遗传算法，依赖频繁查询和稀疏反馈；本文通过EBM能量分提供密集语义指导，减少查询需求。
- **COLD-Attack (Guo et al., 2024)**：基于logits的Langevin动力学可控文本生成，需模型参数；本文用参数无关的MCMC替代，适配黑盒场景。
- **Mix & Match (Mireshghallah et al., 2022)**：无参数MCMC文本生成框架，使用多个专家模型约束；本文继承其MCMC思想，但用EBM替代专家模型作为能量函数。
- **表征工程 (Zou et al., 2023a)**：发现激活值编码安全概念；本文将其推广至对抗提示生成的可迁移攻击场景，实现从"分析"到"生成"的跨越。

## 局限性与未来方向
- **自然性与攻击力权衡**：当前PPL=127.68满足人类可接受阈值，但可通过微调提议模型或引入额外约束进一步提升攻击强度，可能牺牲自然性。
- **未考虑基于文本分类器的防御**：方法未针对InjecGuard等检测机制优化，未来需研究绕过检测的方案以提升泛化性。
- **代理模型依赖性**：EBM训练依赖Qwen2.5-7B-Instruct的激活值，不同代理模型可能影响能量分质量，需探索代理选择策略。
- **跨任务迁移有限**：虽然在未见过任务上取得36.6% ASR-T，但相较于同类任务仍有提升空间，可探索任务无关的通用对抗模式挖掘。

## 研究启发与可借鉴点
- **激活值作为黑盒攻击的代理信号**：将内部激活值与攻击成功性关联，构建能量函数替代梯度/ logits，为黑盒攻击提供新思路；可迁移至jailbreak、后门攻击等其他对抗生成任务。
- **EBM+MCMC的参数无关采样框架**：用分类器隐式定义能量面，结合MCMC迭代优化文本，避免梯度依赖；可复用至任何需要"质量评估+文本采样"的对抗生成场景。
- **三段式攻击模板解耦与重组**：Prefix/Infix/Suffix模块化设计配合GPT-4o-mini自动抽取，提升数据集多样性；可直接用于其他注入攻击的数据增强 pipeline。
- **能量分与ASR强相关验证**：通过Pearson=-0.979证明激活值分布与攻击效果高度关联，为"用内部表征指导外部攻击"提供实证；可作为后续工作的评估指标。
- **团队结合机会**：若团队研究防御方向，可借鉴EBM的能量分作为注入检测信号；若研究红队测试，MCMC采样框架可扩展至多目标优化场景。

## 关键术语表
**Direct Prompt Injection (DPI)**：直接向LLM用户输入注入恶意指令，覆盖系统预设提示的攻击方式，属于OWASP LLM威胁榜首位。
**Energy-based Model (EBM)**：通过能量函数定义数据分布的概率模型，本文用二分类器logits推导能量分评估对抗提示质量。
**Markov Chain Monte Carlo (MCMC)**：通过随机采样逼近目标分布的算法，本文在token级别迭代优化对抗提示，接受概率由EBM能量分决定。
**Activation**：LLM Decoder层间的隐状态，编码丰富语义信息，本文用作EBM输入以表征对抗提示的攻击潜力。
**Template Decoupling**：将攻击提示解耦为Prefix/Infix/Suffix三段并随机重组，提升对抗样本多样性与任务适应性。
**Attack Success Rate (ASR)**：攻击成功的样本比例，本文用关键词匹配或LLM裁判评估。
**Transfer ASR (ASR-T)**：跨任务迁移攻击成功率，排除白盒模型以专注于黑盒迁移能力。
**Masked Language Model (MLM)**：如BERT，根据上下文预测被遮蔽token，本文用作MCMC采样的提议分布生成候选对抗提示。

## 可复现要素
- **数据集**：Tensor Trust攻击数据集（竞赛公开）、CYBERSECEVAL3（7个任务，5训练+2测试）、StruQ人工提示（33条）；论文未明确声明公开状态。
- **代码/权重**：论文未提及代码开源，EBM权重（MLP分类器）和代理模型（Qwen2.5-7B-Instruct、Llama-3.1-8B-Instruct）可复现。
- **关键超参**：EBM：两层MLP(1024→256)，ReLU，AdamW lr=0.0003，batch=256，epoch=100；MCMC：迭代步数=提示token数，batch=20，禁用annealing；提议模型：BERT MLM。
- ** surrogate模型**：Qwen2.5-7B-Instruct、Llama-3.1-8B-Instruct（训练和采样阶段均使用）。
- **评估脚本**：部分judge function在Appendix B给出（Figure 7/8），字符串匹配和LLM裁判混合评估。
