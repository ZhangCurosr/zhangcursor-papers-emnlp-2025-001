---
title: "Video-Compression-Commander-Plug-and-Play-Inference-Accelera"
source: https://aclanthology.org/2025.emnlp-main.98.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:44:08"
field: "多模态大模型高效推理"
keywords: ["VideoLLM", "token压缩", "推理加速", "无训练压缩", "Flash Attention", "帧独特性"]
innovations: ["首个基于帧独特性自适应分配token预算的VideoLLM无训练压缩框架", "提出可叠加于既有压缩方法的Frame Compression Adjustment通用模块", "同时兼容Flash Attention且无需依赖[CLS]或显式attention权重"]
benchmarks: ["MVBench", "LongVideoBench", "MLVU", "VideoMME", "EgoSchema", "PerceptionTest"]
---

# 论文速读：Video-Compression-Commander-Plug-and-Play-Inference-Accelera

## 一句话总结
本文针对Video Large Language Model (VideoLLM)推理时视觉token冗余问题，提出 **VidCom²** ——一种无需训练的即插即用token压缩框架，通过量化每帧的独特性自适应调整压缩强度，在大幅减少计算开销的同时保持99.6%的原始性能。

## 研究问题与动机
- **视频视觉token爆炸导致推理昂贵**：VideoLLMs（如LLaVA-OneVision处理32×196 token/视频，LLaVA-Video处理64×182 token/视频）因连续帧带来大量视觉token，二次复杂度计算负担重。
- **设计近视（Design Myopia）**：现有方法对视频所有帧采用均匀压缩策略，忽视了帧间差异性——仅丢弃8个独特帧即可导致理解失败，而丢弃24个冗余帧无影响（Figure 1）。
- **实现约束（Implementation Constraints）**：部分方法依赖ViT的[CLS] attention权重（现代VideoLLM已弃用[CLS]，改用SigLIP）；部分方法依赖LLM显式attention权重，与Flash Attention等高效算子不兼容，导致显存反而上升。
- **需要兼顾三者**：模型适配性、帧独特性感知、高效算子兼容性。

## 核心贡献（创新点）
1. **系统分析现有token压缩方法**，提炼三个设计原则（模型适配性、帧独特性、算子兼容性），并指出多数方法存在"设计近视"和"实现约束"缺陷。
2. **提出VidCom²**——首个基于帧独特性的VideoLLM plug-and-play token压缩框架，分两阶段动态分配token预算：先基于帧独特性调整压缩强度，再基于token在帧内和跨视频两个层面的独特性执行Top-K选择。
3. **性能与效率双优**：仅在25%视觉token下，LLaVA-OV平均性能达99.6%，LLM生成延迟降低70.8%；15%激进压缩下仍以3.9%超越次优方法；完全兼容Flash Attention且无额外显存开销。

## 方法详解
**整体框架：两阶段无训练压缩**

**Stage 1: Frame Compression Adjustment（帧压缩调节）**
- 计算全局视频表示：对所有帧所有token做平均池化：
  $$\mathbf{g}_{\mathbf{v}} = \frac{1}{T \cdot M} \sum_{t=1}^{T} \sum_{m=1}^{M} \mathbf{x}_{t,m}^{v}$$
- 计算token视频级独特性：先求每个token $\mathbf{x}_{t,m}^{v}$ 与 $\mathbf{g}_{\mathbf{v}}$ 的余弦相似度 $s_{t,m}^{\mathrm{video}}$，独特性得分 $u_{t,m}^{\mathrm{video}} = -s_{t,m}^{\mathrm{video}}$（相似度越低越独特）。
- 计算帧独特性 $u_t = \frac{1}{M} \sum_{m=1}^{M} u_{t,m}^{\mathrm{video}}$（平均操作优于max操作，能更好地捕捉帧内整体独特性密度）。
- 归一化并通过softmax得到帧重要性权重：$\tilde{u}_t = (u_t - \max(u_t))/\tau$（$\tau=0.01$），$\sigma_t = \exp(\tilde{u}_t)/\sum \exp(\tilde{u}_l)$。
- 动态调整每帧保留率：$r_t = R \times (1 + \sigma_t - 1/T)$，保持全局平均保留率 $R$ 不变。

**Stage 2: Adaptive Token Compression（自适应token压缩）**
- 计算帧内token独特性：对第 $t$ 帧做平均池化得 $\mathbf{g}_{f,t}$，计算token与 $\mathbf{g}_{f,t}$ 的余弦相似度，定义 $u_{t,m}^{\mathrm{frame}} = -s_{t,m}^{\mathrm{frame}}$。
- 综合独特性：$u_{t,m} = u_{t,m}^{\mathrm{frame}} + u_{t,m}^{\mathrm{video}}$（实验验证两者等权最优）。
- 按 $r_t \times M$ 对每帧执行Top-K选择，保留独特性最高的token。

## 实验与结果
- **数据集/Benchmark**：MVBench、LongVideoBench、MLVU、VideoMME（Short/Medium/Long）、EgoSchema、PerceptionTest。
- **模型**：LLaVA-OneVision-7B (LLaVA-OV)、LLaVA-Video-7B、Qwen2-VL。
- **基线**：FastV、PDrop、SparseVLM、DyCoke（及Random Drop）。
- **核心结果（LLaVA-OV, R=25%）**：平均性能达99.6%，超越次优DyCoke (R=25%，96.5%)达12.6%；LLM生成延迟从618.0s降至180.7s（↓70.8%），吞吐量提升至1.38×，峰值显存降低9.6%。
- **激进压缩（R=15%）**：VidCom²平均仍达91.2%，超越第二SparseVLM (87.4%)约3.9%；DyCoke因固定4帧窗口设计无法支持R<25%。
- **长视频优势**：在VideoMME(Long)上，VidCom²（R=25%）达101.2%，超越DyCoke (93.6%)和SparseVLM (96.6%)分别7.6%和4.6%。
- **算子兼容性**：VidCom²完全兼容Flash Attention 2，显存无额外开销；对比Intra-LLM方法（如FastV显存↑39.5%、PDrop↑38.4%）。

## 相关工作脉络
1. **FastV (ECCV'24)**：Intra-LLM层的一次性token裁剪，依赖LLM某层输出token的attention权重，与Flash Attention不兼容。VidCom²为Pre-LLM方法，无需访问attention权重。
2. **SparseVLM (ICML'25)**：Intra-LLM方法，利用text-visual attention map排序token重要性；与Flash Attention不兼容。VidCom²考虑纯视觉独特性，可与文本相关性形成互补。
3. **DyCoke (CVPR'25)**：VideoLLM专用方法，将4个连续帧分组做token合并，无法支持R<25%的激进压缩，且未考虑帧间独特性差异。VidCom²支持任意压缩率且逐帧动态调节。
4. **PDrop (CVPR'25)** & **FiCoCo**：依赖[CLS] token或显式attention权重，无法适配SigLIP视觉编码器及高效算子；VidCom²无此限制。
5. **MUSTDrop / FasterVLM**：同样依赖[CLS] attention，VidCom²解决了现代[CLS]-free VideoLLM的适配问题。

## 局限性与未来方向
- **大模型验证缺失**：受算力限制，未在LLaVA-Video-72B、Qwen2-VL-72B等大模型上评测，但作者预期在更大架构中效果会进一步放大。
- **未覆盖流式场景**：当前面向离线长视频理解，未来计划扩展至实时流式视频理解（real-time streaming video understanding）。
- **窗口大小实验局限**：仅测试了fixed 32帧下的window size（4/8/16/32），更长视频序列的动态窗口策略有待探索。

## 研究启发与可借鉴点
1. **帧独特性量化思路可迁移**：通过全局表示相似度反向衡量"独特性"的方法简洁高效，可推广至其他多模态压缩场景（如图像序列、点云）。
2. **两阶段分离设计**：先分配预算（帧级）再筛选个体（token级）的策略层次清晰，可复用于其他需要多级层次化压缩的任务。
3. **算子兼容性优先的设计原则**：强调Pre-LLM路径以兼容Flash Attention，避免了Intra-LLM方法带来的显存反增问题，为后续工作提供了重要设计参考。
4. **Frame Compression Adjustment可作为通用模块**：论文证明其可叠加于FastV、SparseVLM等既有方法上进一步提升性能，表明该策略具有良好的模块化和可组合性。

## 关键术语表
**VideoLLM**：结合视觉编码器与LLM的视频理解大模型，如LLaVA-OneVision、LLaVA-Video、Qwen2-VL。
**Token Compression**：对视觉token序列进行冗余剔除以降低推理计算量的技术，分为Pre-LLM（ViT/投影器层）和Intra-LLM（decoder内部）两类。
**Frame Uniqueness**：视频序列中某帧相对于其他帧的独特程度，本文通过帧内所有token与全局视频表示的负相似度均值来量化。
**Pre-LLM Compression**：在视觉token进入LLM之前进行的压缩方法，VidCom²属于此类，天然兼容Flash Attention。
**Intra-LLM Compression**：在LLM解码过程中基于attention权重进行的token裁剪，如FastV、SparseVLM，通常与高效算子不兼容。
**Flash Attention**：Tri Dao提出的IO-aware高效attention算子，VidCom²与其完全兼容，而多数Intra-LLM方法因依赖显式attention权重无法使用。
**Equivalent Retention Ratio**：为了公平比较不同压缩方法（有些同时压缩KV cache）而采用的等效保留率定义，遵循SparseVLM的约定。

## 可复现要素
- **代码**：已开源 https://github.com/xuyang-liu16/VidCom2
- **数据集**：MVBench、LongVideoBench、MLVU、VideoMME、EgoSchema、PerceptionTest（均为公开benchmark）
- **关键超参**：温度参数 $\tau=0.01$、epsilon $\epsilon=10^{-8}$、默认综合独特性 $u_{t,m}=u_{t,m}^{\mathrm{frame}}+u_{t,m}^{\mathrm{video}}$（等权）、窗口大小默认全局（32帧）
- **实验环境**：NVIDIA A100-SXM4-80GB GPU × 4，使用Flash Attention 2
- **评估框架**：LMMs-Eval
