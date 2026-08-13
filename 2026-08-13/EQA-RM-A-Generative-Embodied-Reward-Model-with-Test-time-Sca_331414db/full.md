# EQA-RM: A Generative Embodied Reward Model with Test-time Scaling

Yuhang Chen<sup>1</sup> Zhen Tan<sup>2</sup> Tianlong Chen<sup>1</sup>

<sup>1</sup>The University of North Carolina at Chapel Hill <sup>2</sup>Arizona State University {yuhang, tianlong}@cs.unc.edu ztan36@asu.edu

## Abstract

Reward Models (RMs), vital for large model alignment, are underexplored for complex embodied tasks like Embodied Question Answering (EQA) where nuanced evaluation of agents spatial, temporal, and logical understanding is critical yet not considered by generic approaches. We introduce EQA-RM, a novel generative multimodal reward model specifically architected for EQA, trained via our innovative Contrastive Group Relative Policy Optimization (C-GRPO) strategy to learn finegrained behavioral distinctions. The generative nature of EQA-RM provides interpretable, structured reward feedback (beyond simple scalars), uniquely enabling test-time scaling to dynamically adjust evaluation granularity, from concise scores to detailed critiques of reasoning and grounding, at inference without retraining. Concurrently, we introduce EQAREWARDBENCH, a new benchmark built on OpenEQA for standardized EQA reward model assessment. Demonstrating high sample efficiency, EQA-RM (fine-tuning Qwen2- VL-2B-Instruct) achieves 61.9% accuracy on EQA-RM-Bench with 700 samples, outperforming strong proprietary baselines, including Gemini-2.5-Flash, GPT-4o, Claude-3.5-Haiku, and open-sourced state-of-the-art models such as RoVRM and VisualPRM. The code and dataset can be found here: https://github. com/UNITES-Lab/EQA-RM.

## 1 Introduction

Reward Models (RMs) have emerged as a cornerstone technique in qualifying the quality of output or actions of Large Language Models. Desired RMs provide critical signals for refining model behaviors and enhancing performance, often through reinforcement learning or selection strategies (Ouyang et al., 2022b; Snell et al., 2024). Existing generic Reward Models (Zhao et al., 2025; Liu et al., 2025) often designed for static inputs or simple outcomes, are proved ill-equipped for dynamic and interactive domains. Their limitations is amplified in complex tasks such as Embodied Question Answering (EQA). EQA requires agents to perceive, interact, and reason through sequences of multimodal observations and actions to answer questions within 3D environments (Majumdar et al., 2024; Yu et al., 2019). Evaluating the resulting EQA trajectories necessitates a nuanced assessment of the agent’s reasoning coherence, action appropriateness, and how well language is grounded within the environment. Yet, existing RMs are usually trained via generic next-token prediction (Zhao et al., 2025) or classification (Faal et al., 2023), thus incapable to capture the spatiotemporal and logical dependencies inherent in embodied tasks. This emphasizes the urgent need for specialized mechanisms to accurately assess EQA’s multifaceted success indicators.

To address this, we propose EQA-RM, a novel multimodal reward model tailored for evaluating EQA trajectories. Inspired by Zhao et al. (2025), EQA-RM is designed as a generative reward model (GenRM). EQA-RM thus is able to produce not only scalar rewards but also explicit reasoning for its assessments. Such generative capability enhances interpretability and facilitate test-time scaling (Snell et al., 2024) for performance improvements. To effectively generate these nuanced evaluations for EQA, the EQA-RM is trained for enhanced spatial, temporal, and reasoning processing capabilities, enabling it to handle the unique multimodal data streams inherent in EQA tasks.

We train EQA-RM using an efficient two-stage process. The first stage involves standard Rejective Finetuning (RFT) (Yuan et al., 2023) to teach the model the desired output format, which includes a textual critique and a scalar score. However, we observed that relying solely on RFT tends to make the model proficient at generating outputs in the learned format without necessarily achieving a deep understanding of crucial trajectory el-

Question: Are the chairs tucked into the table?

![](images/c8dc3cb8377a0eaaad306ac1dde8c9808267d7b20ce37ca0f28cfcf31cdfa5e4.jpg)  
Response: No, they are not tucked into the table.  
Ground Truth Answer: Yes.

Reasoning Trace: The user is asking about the position of the chairs relative to the tables in the video. I need to look at ... Focusing on the chair closest to the camera, it is clearly not pushed fully under the white desk. It is pulled out... The other chair at this same desk is less visible, but it also appears to be somewhat pulled out... Continuing to pan across this area, the chairs ... appear to be in a slightly pulled-out position, ready for use, rather than tucked away. Based on the observation ..., none of the chairs are shown to be fully tucked into the tables or desks. They are all pulled out.  
![](images/9d634cd86a5bef9d85b4ead972e2d5c3826f67989854626f66cf60fdf3daf84a.jpg)

GPT-4o <critique> The agent's response is well-structured and considers multiple frames to assess the position... The reasoning is plausible and internally consistent, as it maintains a clear focus on the task of determining whether the chairs are tucked in. The logic is coherent, as the agent systematically evaluates each image to reach a conclusion. The visual grounding is accurate;... The trace-answer consistency is strong, as the reasoning aligns with the final answer provided. </critique> <score>9</score>

EQA-RM <critique> The agent correctly located... the chairs and desks in the initial frames. However, the predicted answer is incorrect... The chairs are indeed partially tucked into the tables, especially at the wooden desks shown in the following frames. In terms of overall performance, the agent demonstrated reasonable visual grounding and logical reasoning, but the predicted answer is misleading... but the reasoning itself is sound. </critique> <score>5</score>

Figure 1: EQAREWARDBENCH enables nuanced evaluation of complex EQA trajectories involving spatio-temporal reasoning. An agent’s reasoning process and response to a question are assessed through detailed critiques and scores provided by reward models like our EQA-RM and GPT-4o. Within its critiques, EQA-RM explicitly assesses temporal dependencies , spatial grounding , and reasoning coherence , showcasing better results than GPT-4o.

ements. This potential for supervised finetuning to primarily capture style rather than substance in preference learning has been discussed in prior work (Chen et al., 2025a; Liu et al., 2025). To address this intrinsic limitation, we employ a reinforcement learning approach for the second stage, which helps the model internalize output quality by optimizing for rewards, overcoming SFT’s stylistic memorization. Specifically, we design a novel Contrastive Group Relative Policy Optimization (C-GRPO) strategy. C-GRPO utilizes rule-based contrastive rewards derived from targeted data augmentations. It optimizes EQA-RM’s evaluative acuity by training it to distinguish policy outputs based on their evaluation under original, coherent contexts versus synthetically perturbed ones. We design the following augmentations for perturbations: ❶ trajectories with shuffled video frames, ❷ frames with randomly masked spatial regions, and ❸ sequences with jumbled reasoning steps. Essentially, EQA-RM earns a positive reward only when its score assigned under original, unperturbed conditions is more accurate (i.e., closer to a preference score) than its score assigned under the corresponding perturbed conditions. By learning this differentiated accuracy in its scoring relative to preference score, C-GRPO compels EQA-RM to internalize the importance of temporal order, fine-grained spatial details, and coherent logical flow. This cultivates a robust and discerning evaluative capability for embodied tasks.

On the other hand, the EQA domain lacks standardized benchmarks for rigorously evaluating and comparing reward models. Current EQA task benchmarks focus on coarse success metrics rather than the fine-grained trajectory quality crucial for RM development. To bridge this gap, we introduce EQAREWARDBENCH. Built upon the OpenEQA dataset, EQAREWARDBENCH features embodied episode memory videos from two types of household environments: HM3D () and Scan-Net (). From the original question-answer pairs, we construct more comprehensive question-responsereasoning trace triplets. The benchmark includes 1,546 test instances designed to evaluate reward models across eight distinct aspects of trajectory quality (e.g., correctness, grounding, efficiency). EQAREWARDBENCH thus provides a standardized platform for the rigorous and comparable assessment of reward models on EQA tasks.

Empirical evaluations demonstrate the capabilities of EQA-RM. Based on Qwen2-VL-2B-Instruct, EQA-RM substantially improves upon its base model and outperforms existing open-source visual reward models on EQAREWARDBENCH. EQA-RM also shows test-time scalability. By increasing evaluative computations at inference, its accuracy on EQAREWARDBENCH is boosted from 42.47% to 61.86%. This enhanced performance allows EQA-RM to surpass leading large commercial models in accuracy on our benchmark.

In conclusion, our core contributions are: (i) We propose EQA-RM, a generative multimodal reward model with enhanced spatial, temporal, and reasoning capabilities tailored for EQA; (ii) We introduce a EQA Reward Model Benchmark, EQAREWARD-BENCH, the first dedicated benchmark for standardized evaluation of reward models for EQA. (iii) Extensive experiments have proved the improvement of our method for test time scaling.

## 2 Related Work

Generative Reward Models. Reward Models are crucial for guiding LLMs outputs (Ouyang et al., 2022a; Christiano et al., 2017; Ziegler et al., 2019). To overcome the limitations of scalar rewards and provide deeper evaluative insights, Generative Re ward Models emerged and produced rich, interpretable textual feedback like critiques or expla nations (Zhang et al., 2024; Mahan et al., 2024; Zheng et al., 2023). LLM-as-a-judge method ac commodates pairwise critiques to evaluate LLMs outputs and enhances interpretability (Zheng et al., 2023). In multimodal domains, GenRMs enable finer-grained supervision through techniques such as step-wise reasoning assessment (Wang et al., 2025c), self-critique (Yu et al., 2025), or Chainof-Thought style evaluations(Zhao et al., 2025), enhancing the performance of Multimodal LLMs. Embodied Question Answering. EQA challenges agents to perceive, navigate, and interact within 3D environments to answer questions, integrating vision, language, reasoning, and plan ning (Das et al., 2018; Gordon et al., 2018). While datasets like OpenEQA offer rich scenarios for this task (Wijmans et al., 2019; Kolve et al., 2017; Majumdar et al., 2024) and EQA methodologies have advanced towards end-to-end foundation models (Ahn et al., 2022; Driess et al., 2023), current EQA evaluation predominantly assesses only final answer correctness (Majumdar et al., 2024; Das et al., 2018). This common practice overlooks crucial trajectory qualities such as reasoning coherence and spatio-temporal understanding (Chen et al., 2025b), creating a significant gap in reward modeling for comprehensive EQA assessment, motivating our development of the special ized EQA-RM.

Rule-based RL for LLMs. Reinforcement learning is increasingly used to refine LLMs for enhanced alignment and reasoning capabilities, moving beyond standard SFT (Schulman et al., 2017; Ouyang et al., 2022a; Zhai et al., 2024). A key direction involves rule-based RL, which employs systematic or synthetic feedback for improved efficiency and targeted behavior control, with algorithms like Group Relative Policy Optimization (GRPO) (Shao et al.) notably using relative comparisons and rule-defined rewards for complex reasoning (Mu et al., 2024; Xie et al., 2025; Wang et al., 2025b; Xiong et al., 2025). These structured RL principles are proving vital for training advanced reward models capable of nuanced multimodal understanding and for enabling RM self-improvement with systematic feedback (Feng et al., 2025; Liu et al., 2025). Our C-GRPO builds on these trends, utilizing rule-based contrastive rewards from data augmentations to train a generative RM specific for EQA tasks.

## 3 The EQAREWARDBENCH Dataset

To facilitate robust and standardized evaluation of reward models for Embodied Question Answering, we construct a new dataset EQAREWARDBENCH. This section details its generation pipeline, statistics, and splitting strategy.

## 3.1 Dataset Generation Pipeline

Our dataset construction process builds upon the OpenEQA (Majumdar et al., 2024), which provides instances comprising a question, ground truth answer, and associated episode memory. We extend this foundation in a two-step generation pipeline.

Step 1: Diverse Response Generation. We first employ the Gemini-2.0-Flash (Team et al., 2023) model to generate a diverse set of responses for each EQA instance. Given the episode memory $( v ^ { o } )$ and the original question $( q )$ as input, this model produces multiple pairs of predicted answers (a) and accompanying reasoning traces $( z ^ { o } )$ . To ensure a rich dataset for subsequent reward modeling, we intentionally solicit a spectrum of predicted answers, encompassing both correct and incorrect responses, thereby fostering diversity.

Step 2: High-Quality Score Generation. Next, Gemini-2.5-Pro-Experiment-03-14 generates detailed evaluations. For each Step 1 output tuple of {episode memory $( v ^ { o } )$ , question (q), predicted answer (a), reasoning trace $( z ^ { o } ) \}$ , this model outputs a textual critique $\left( c _ { r } \right)$ and a scalar quality score. This score is an integer ranging from 0 to 10, indicating the quality of the predicted answer and reasoning trace. The ground truth answer input at this stage ensures accurate scores. After human verification for reliability, these scores become our ground truth scores $( s ^ { g t } )$ . The accompanying critiques $\left( c _ { r } \right)$ potentially influenced by this privileged input, are primarily for analysis and contextualizing $s _ { g t } .$ , not for directly training reward models that must operate without such information during inference. This process yields a foundational dataset where each instance is a tuple: $\{ q , a , v ^ { o } , z ^ { o } , c _ { r } , s ^ { g t } \}$ , forming the basis for both EQAREWARDBENCH and the EQA-RM fine-tuning data.

## 3.2 Dataset Statistics and Splits

The episode memories $( v _ { o } )$ in our foundational dataset originate from two distinct 3D indoor environment collections provided by OpenEQA: HM3D (Ramakrishnan et al., 2021) and Scan-Net (Dai et al., 2017). These sources contribute 697 and 1,546 instances in our dataset, respectively.

To ensure robust model training, fair evaluation, and prevention of test set leakage, we partition this foundational dataset: ❶ EQAREWARD-BENCH $( D _ { B } ) \colon$ designed for evaluating diverse reward models, $D _ { B }$ comprises all 823 HM3D instances and 713 ScanNet instances. ❷ Fine-tuning Dataset $( D _ { F } )$ : the remaining 697 distinct ScanNet instances are exclusively reserved for training our EQA-RM model.

Crucially, $D _ { F }$ and $D _ { B }$ maintain no overlap in underlying episode memory data $( \mathrm { e . g . }$ , distinct scenes or trajectories), guaranteeing evaluation integrity. This splitting strategy enables comprehensive assessment: models trained on $D _ { F }$ (ScanNet) are evaluated against the disjoint ScanNet portion of $D _ { B }$ for in-distribution (ID) performance, and against its HM3D portion for out-of-distribution (OOD) generalization. Further details on dataset composition are provided in Appendix A.

## 4 Methods

This section details the methodology for training EQA-RM for nuanced evaluation of Embodied Question Answering trajectories. Our approach comprises two main stages: Rejective Fine-Tuning (RFT) to establish baseline capabilities in generating structured critiques and scores; followed by our novel Contrastive Group Relative Policy Optimization (C-GRPO) strategy to instill deeper sensitivities to critical aspects of trajectory quality.

## 4.1 Preliminaries and Notations

We denote the generative reward model we aim to train as $R _ { \phi }$ . An instance consists of an input question $q , \mathbf { a }$ predicted answer a, an original reasoning trace $z ^ { o } = \{ z _ { 1 } ^ { o } , z _ { 2 } ^ { o } , \dots , z _ { T } ^ { o } \}$ , and the original episode memory content $v ^ { o }$ . The episode memory $v ^ { o }$ comprises a sequence of N video frames, $v ^ { o } = \{ v _ { 1 } ^ { o } , v _ { 2 } ^ { o } , \ldots , v _ { N } ^ { o } \}$ . The desired output of $R _ { \phi }$ for a given instance is an evaluation composed of a textual critique $c _ { r }$ and a scalar score $s _ { r }$ . For each evaluated EQA instance, there is an associated ground truth score, denoted $s ^ { g t }$

## 4.2 Rejective Fine-Tuning

After obtaining the foundational dataset (Section 3.1), which includes ground truth scores $( s ^ { g t } )$ for each EQA instance $\{ q _ { i } , a _ { i } , z _ { i } ^ { o } , v _ { i } ^ { o } \}$ , we initiate the training of EQA-RM $( R _ { \phi } )$ with Rejective Fine-Tuning (RFT). This first stage primarily aims to teach $R _ { \phi }$ to generate outputs (a textual critique $c _ { r }$ and a scalar score $s _ { r } )$ that conform to our desired structured format and exhibit a baseline level of quality. The RFT process involves two key steps: Step 1: Rejective Filtering. To construct the SFT dataset $D _ { R F T }$ , for each instance $\{ q _ { i } , a _ { i } , z _ { i } ^ { o } , v _ { i } ^ { o } \}$ from finetuning dataset $D _ { F }$ , an auxiliary LLM evaluator $R ^ { a u x }$ generates $N _ { R F T }$ candidate evaluations. Each candidate evaluation consists of a critique $c _ { i , k } ^ { a u x }$ and a score $s _ { i , k } ^ { a u x }$ Importantly, $\{ c _ { i , k } ^ { a u x } , s _ { i , k } ^ { a u x } \}$ are produced based only on the input $\{ q _ { i } , a _ { i } , z _ { i } ^ { o } , v _ { i } ^ { o } \}$ without ground truth answer. To ensure data quality for SFT, these candidate evaluations are rejective filtered.

This rejective filtering process aims to remove "too easy" and "incorrect" candidate evaluations.

An evaluation is considered correct $( \epsilon _ { i , k } = 1 )$ $\mathrm { i f ~ } | s _ { i } ^ { g t } - s _ { i , k } ^ { a u x } | < \tau ~ ( \tau$ is score tolerance), and $\epsilon _ { i , k } = 0$ otherwise. Let $\begin{array} { r } { E _ { i } = \sum _ { k = 1 } ^ { N _ { R F T } } \epsilon _ { i , k } } \end{array}$ be the count of the correct evaluation of instance i. We select evaluations if:

$$
\epsilon _ { i , k } \wedge \big ( \neg ( E _ { i } = N _ { R F T } \wedge N _ { R F T } > 0 ) \big )\tag{1}
$$

This clause ensures an evaluation is included in $D _ { R F T }$ only if it is "correct" (per $\epsilon _ { i , k } = 1 )$ and not all $N _ { R F T }$ evaluations for instance i were "correct" ("too easy"). Finally, $D _ { R F T }$ comprising $\{ q _ { i } , a _ { i } , z _ { i } ^ { o } , v _ { i } ^ { o } \}$ paired with critique and score $\{ c _ { i } , s _ { i } \}$ that satisfy this rejective filtering process, is subsequently used for the initial SFT of $R _ { \phi }$

Step 2: Supervised Fine-Tuning. The curated dataset $D _ { R F T }$ is employed for SFT of our reward model EQA-RM $R _ { \phi }$ . For each sample in $D _ { R F T }$ , an input EQA instance comprising $( q _ { i } , a _ { i } , z _ { i } ^ { o } , v _ { i } ^ { o } )$ is used to construct a multimodal prompt $P _ { i }$ , which also incorporates task-specific instructions. The corresponding output $T _ { i }$ for $R _ { \phi }$ is the structured string representation of the selected critique $c _ { i }$ and score $s _ { i }$ from $D _ { R F T }$ . The SFT objective is to train $R _ { \phi }$ by minimizing the negative log-likelihood loss $\mathcal { L } _ { S F T } ( \phi )$

$$
\mathop { \mathcal { L } _ { S F T } } ( \phi ) = - \sum _ { ( P _ { i } , T _ { i } ) \atop \in D _ { R F T } } \sum _ { t = 1 } ^ { | T _ { i } | } \log P ( T _ { i , t } | P _ { i } , T _ { i , < t } ; \phi )\tag{2}
$$

where $T _ { i , t }$ is the t-th token of the target sequence $T _ { i }$ , and $T _ { i , < t }$ represents the sequence of preceding tokens. This SFT stage primarily teaches $R _ { \phi }$ to generate outputs in the specified format and establishes its baseline evaluative capability.

## 4.3 Contrastive Group Relative Policy Optimization

While SFT establishes EQA-RM’s basic output formatting (critique $c _ { r } .$ , score $s _ { r } )$ , it often fails to instill deep comprehension of episode memory and reasoning from the limited finetuning data $D _ { F }$ . Our Contrastive Group Relative Policy Optimization (C-GRPO) framework addresses this by using targeted contrastive rewards to explicitly train $R _ { \phi }$ for crucial sensitivities: temporal visual ordering, finegrained spatial details, and logical coherence of reasoning.

C-GRPO trains $R _ { \phi }$ to differentiate its evaluation scores based on structured contrasts between original EQA instance components $( q _ { i } , a _ { i } , z _ { i } ^ { o } , v _ { i } ^ { o } )$ and their synthetically perturbed counterparts. These perturbations include temporally shuffled video frames $( v _ { i } ^ { t }$ from $v _ { i } ^ { o } )$ , spatially masked frames $( v _ { i } ^ { s }$ from $v _ { i } ^ { o } )$ , and altered reasoning traces $( z _ { i } ^ { r }$ from $z _ { i } ^ { o } ) . \ R _ { \phi }$ produces scores for the original condition $( S _ { i } ^ { o } )$ and for each corresponding augmented condition $( S _ { i } ^ { t } , S _ { i } ^ { s } , S _ { i } ^ { r } )$ . For $S _ { i } ^ { t }$ and $S _ { i } ^ { s } , R _ { \phi }$ scores the original $( q _ { i } , a _ { i } , z _ { i } ^ { o } )$ but is prompted as if the visual context were $\boldsymbol { v } _ { i } ^ { t }$ or $v _ { i } ^ { s }$ respectively. For $S _ { i } ^ { r }$ $R _ { \phi }$ scores $( q _ { i } , a _ { i } )$ paired with the perturbed reasoning trace $z _ { i } ^ { r }$ while using the original visual context $v _ { i } ^ { o } .$ Base Outcome Rewards. each evaluation score $s _ { r , i }$ are assessed by the Accuracy Reward $( R _ { a , i } ) \colon$

$$
R _ { a , i } ( s _ { r , i } , s _ { g t , i } ) = \operatorname* { m a x } ( 0 , 1 - ( \frac { | s _ { r , k } - s _ { g t , i } | } { 1 0 } ) ^ { 2 } )\tag{3}
$$

and Format Reward $( R _ { f } )$ is 1 if text output adheres to the specified critique-score structure, and 0 otherwise.

C-GRPO Contrastive Rewards. For an $x \mathrm { - }$ augmented version $( x \in \{ t , s , r \}$ for temporal, spatial, reasoning respectively), let $R _ { a , i } ^ { x }$ be the corresponding accuracy reward. This mechanism conditionally boosts the $R _ { a , i } ^ { o }$ for evaluations of original instances. For each active augmentation type $x \in \{ t , s , r \}$ : Let $\overline { { R _ { a } ^ { o } } }$ and $\overline { { R _ { a } ^ { x } } }$ be the batchmean accuracy rewards (Eq. 3) for original and x-augmented evaluations, respectively. The perevaluation boost $R ^ { x }$ for the i-th original evaluation is then defined as:

$$
R _ { i } ^ { x } = \left\{ \mu , \quad \mathrm { i f } \overline { { { R _ { a } ^ { o } } } } \geq \delta \cdot \overline { { { R _ { a } ^ { x } } } } \right.\tag{4}
$$

where $\delta = 0 . 9 5$ and $\mu = 0 . 3$ are hyperparameters. This yields boosts $R ^ { t } , R ^ { s } , R ^ { r }$ for active contrasts. Total Reward. The total reward is:

$$
R _ { i } ^ { A } = R _ { a } ^ { o } + R _ { f } ^ { o } + ( R ^ { t } + R ^ { s } + R ^ { r } ) / 3\tag{5}
$$

Advantage and Policy Update. The advantage $A _ { i }$ for the i-th evaluation (from G total evaluations) is computed by normalizing the total rewards $\{ R _ { j } ^ { A } \} _ { j = 1 } ^ { G }$ within the group:

$$
A _ { i } = \frac { R _ { i } ^ { A } - \mathrm { m e a n } ( \{ R _ { j } ^ { A } \} _ { j = 2 } ^ { G } ) } { \mathrm { s t d } ( \{ R _ { j } ^ { A } \} _ { j = 1 } ^ { G } ) + \epsilon }\tag{6}
$$

where ϵ is a small constant for numerical stability. $R _ { \phi }$ are updated follows the objective of C-GRPO:

$$
\begin{array} { l } { \displaystyle \mathcal { T } _ { \mathrm { c - \otimes \mathrm { R } P 0 } } ( \phi ) = \mathbb { E } _ { d _ { i } \sim D _ { F } } \Bigg [ \frac { 1 } { G } \sum _ { k = 1 } ^ { G } \Bigg ( } \\ { \displaystyle \operatorname* { m i n } \Big ( r _ { k } ( \phi ) A _ { k } , \mathrm { c l i p } ( r _ { k } ( \phi ) , 1 - \epsilon _ { c } , 1 + \epsilon _ { c } ) A _ { k } \Big ) } \\ { - \beta _ { K } \mathbb { D } _ { \mathrm { K L } } ( P _ { \phi } \| P _ { \phi _ { \mathrm { r e f } } } ) \Big ) \Bigg ] } \end{array}\tag{7}
$$

where $r _ { k } ( \phi )$ is the probability ratio, $\phi _ { \mathrm { r e f } }$ are parameters before the update, $\epsilon _ { c }$ is a clipping hyperparameter, and $R _ { \phi _ { \mathrm { r e f } } }$ is a reference model (typically the SFT version of $R _ { \phi } )$ .

## 5 Experiments

In this section, our goal is to assess the effectiveness of EQA-RM and answer the following questions:

• RQ1: How does EQA-RM perform compared to existing Visual-based Reward Models?

• RQ2: How does the performance of EQA-RM scale with increased test-time compute?

• RQ3: What is the impact of each component of the proposed reward strategy on EQARewardBench’s performance?

![](images/f7aec19fb0ae0e927aaa6a43048bb34b8df00607bdae90fee1a98fc293caa99d.jpg)  
(a) Accuracy Reward

![](images/4331fe118741bf95e6e2bc6a454a5ff596dc5af335257eb797ecb353465a9e9f.jpg)

![](images/f7c6c4844ad2f4af35d7f22fb7d130988cc7e70172c428a6ae6653cad3a8b0a3.jpg)  
(b) Reasoning Reward

![](images/0ce5e26155407b6203b3d15cadef78cb34dd7d11863cfe07cbec0a709799e2e1.jpg)  
Figure 2: Overview of training dynamics for EQA-RM’s core reward components of C-GRPO.

Table 1: Main evaluation results on the EQARewardBench dataset. We report Accuracy (Acc) and RMSE for each method across two environments (HM3D and ScanNet), and their aggregated Overall performance. All results are shown with differences relative to Qwen2-VL-2B-Instruct model (Better is marked in red; worse is marked in blue).
<table><tr><td rowspan="2">Models</td><td colspan="2">EQARewardBench-HM3D</td><td colspan="2">EQARewardBench-ScanNet</td><td colspan="2">Overall</td></tr><tr><td>Acc ↑</td><td>RMSE↓</td><td>Acc ↑</td><td>RMSE↓</td><td>Acc ↑</td><td>RMSE↓</td></tr><tr><td colspan="7">VLM-as-a-Judge</td></tr><tr><td>Qwen2-VL-2B-Instruct</td><td>33.71</td><td>3.836</td><td>32.44</td><td>4.158</td><td>33.08</td><td>3.997</td></tr><tr><td>Gemini-2.5-Flash</td><td> $\mathbf { 6 1 . 2 5 } _ { \uparrow 2 7 . 5 4 }$ </td><td> $4 . 8 7 2 _ { \uparrow 1 . 0 3 6 }$ </td><td> $5 8 . 3 3 _ { \uparrow 2 5 . 8 9 }$ </td><td> $3 . 8 0 5 _ { \downarrow 0 . 3 5 3 }$ </td><td> $\underline { { 5 9 . 7 9 } } _ { \uparrow 2 6 . 7 1 }$ </td><td> $4 . 3 3 9 _ { \uparrow 0 . 3 4 2 }$ </td></tr><tr><td>GPT-40</td><td> $\underline { { 6 0 . 1 7 } } _ { \uparrow 2 6 . 4 6 }$ </td><td> $6 . 8 4 7 _ { \uparrow 3 . 0 1 1 }$ </td><td> $5 6 . 7 2 _ { \uparrow 2 4 . 2 8 }$ </td><td> $5 . 3 1 3 _ { \uparrow 1 . 1 5 5 }$ </td><td> $5 8 . 4 5 _ { \uparrow 2 5 . 3 7 }$ </td><td> $6 . 0 8 0 _ { \uparrow 2 . 0 8 3 }$ </td></tr><tr><td>Claude-3.5-Haiku</td><td> $5 7 . 8 8 _ { \uparrow 2 4 . 1 7 }$ </td><td> $5 . 3 1 6 _ { \uparrow 1 . 4 8 0 }$ </td><td> $5 4 . 1 2 _ { \uparrow 2 1 . 6 8 }$ </td><td> $4 . 2 9 1 _ { \uparrow 0 . 1 3 3 }$ </td><td> $5 6 . 0 0 _ { \uparrow 2 2 . 9 2 }$ </td><td> $4 . 8 0 4 _ { \uparrow 0 . 8 0 7 }$ </td></tr><tr><td colspan="7">Standard Visual-Based Reward Models</td></tr><tr><td>RoVRM</td><td> $3 8 . 1 2 _ { \uparrow 4 . 4 1 }$ </td><td> $3 . 3 4 1 _ { \downarrow 0 . 4 9 5 }$ </td><td> $4 0 . 2 3 _ { \uparrow 7 . 7 9 }$ </td><td> $3 . 1 5 8 _ { \downarrow 1 . 0 0 0 }$ </td><td> $3 9 . 1 8 _ { \uparrow 6 . 1 0 }$ </td><td> $3 . 2 5 0 _ { \downarrow 0 . 7 4 7 }$ </td></tr><tr><td>VisualPRM</td><td> $4 0 . 0 7 _ { \uparrow 6 . 3 6 }$ </td><td> $3 . 5 6 2 _ { \downarrow 0 . 2 7 4 }$ </td><td> $3 7 . 4 1 _ { \uparrow 4 . 9 7 }$ </td><td> $3 . 4 1 3 _ { \downarrow 0 . 7 4 5 }$ </td><td> $3 8 . 7 4 _ { \uparrow 5 . 6 6 }$ </td><td> $3 . 4 8 8 _ { \downarrow 0 . 5 0 9 }$ </td></tr><tr><td colspan="7">Proposed Method</td></tr><tr><td>base EQA-RM</td><td> $3 9 . 1 6 _ { \uparrow 5 . 4 5 }$ </td><td> $\underline { { 3 . 0 8 1 } } _ { \downarrow 0 . 7 5 5 }$ </td><td> $4 5 . 7 8 _ { \uparrow 1 3 . 3 4 }$ </td><td> $\underline { { 2 . 8 2 6 } } _ { \downarrow 1 . 3 3 2 }$ </td><td> $4 2 . 4 7 _ { \uparrow 9 . 3 9 }$ </td><td> $2 . 9 5 3 _ { \downarrow 1 . 0 4 4 }$ </td></tr><tr><td>w/ test scaling (K=32)</td><td> $5 8 . 6 5 _ { \uparrow 2 4 . 9 4 }$ </td><td> $\mathbf { 2 . 9 1 8 _ { \downarrow 0 . 9 1 8 } }$ </td><td> ${ \bf 6 5 . 0 6 } _ { \mathrm { \uparrow 3 2 . 6 2 } }$ </td><td> $2 . 5 3 7 _ { \downarrow 1 . 6 2 1 }$ </td><td> $\mathbf { 6 1 . 8 6 } _ { \uparrow 2 8 . 7 8 }$ </td><td> $2 . 7 2 7 _ { \downarrow 1 . 2 7 0 }$ </td></tr></table>

## 5.1 Experiment Setup

Dataset and Benchmark. All experimental evaluations are conducted using our EQAREWARD-BENCH $( D _ { B } )$ . The methodology behind it is detailed in Section 3. The evaluation focuses on the Verifier mode, assessing the RM’s capability to accurately score pre-generated policy responses based on the offline-annotated ground truth scores. Base Model. Our base model is Qwen2-VL-2B-Instruct (Wang et al., 2024), instruction-tuned for robust multimodal understanding.

Baselines. We compare EQA-RM against a comprehensive set of baseline methods to establish its relative effectiveness: (1) VLM-as-a-Judge: Stateof-the-art VLMs prompted offline to evaluate EQA responses on our test set. Including Gemini-2.5- Flash (Team et al., 2023), GPT-4o (Hurst et al., 2024) and Claude-3.5-Haiku. (2) Generic Standard RMs: Adapt existing VLMs or Visual RMs including RoVRM (Wang et al., 2025a), Visual-

PRM (Wang et al., 2025c).

Evaluation Metrics. We evaluate the performance of all reward models using the following metrics.

• Accuracy measures the proportion of predictions where the gap between the predicted score $s ^ { p }$ and the target score $s ^ { g t }$ is within a predefined tolerance τ. We use the tolerance $\tau = 2$ in the experiments:

$$
\mathsf { A c c } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathbb { I } ( | s _ { i } ^ { p } - s _ { i } ^ { g t } | \leq \tau )\tag{8}
$$

• Root Mean Square Error quantifies the average magnitude of the error between predicted and target scores:

$$
\mathrm { R M S E } = \sqrt { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } ( s _ { i } ^ { p } - s _ { i } ^ { g t } ) ^ { 2 } }\tag{9}
$$

Implementation Details. Training utilized the AdamW optimizer with a learning rate of $1 e - 6$ and a batch size of 1. Keyframe selection during

Table 2: Accuracy performance on the EQARewardBench-Scannet dataset, broken down by EQA question type. Higher is better for all metrics. All results are shown with differences relative to the Qwen2-VL-2B-Instruct model (Better is marked in red; Worse is marked in blue).
<table><tr><td>Models</td><td>Object Recognition</td><td>Object Localization</td><td>Attribute Recognition</td><td>Spatial Understanding</td><td>Object State Recognition</td><td>Functional Reasoning</td><td>World Knowledge</td></tr><tr><td colspan="8">VLM-as-a-Judge</td></tr><tr><td>Qwen2-VL-2B-Instruct</td><td>30.30</td><td> $3 6 . 7 6$ </td><td>38.69</td><td>29.41</td><td>29.88</td><td>38.71</td><td>27.34</td></tr><tr><td>Gemini-2.5-Flash</td><td> $\mathbf { 6 9 . 2 3 _ { \uparrow 3 8 . 9 3 } }$ </td><td> $6 2 . 1 5 _ { \uparrow 2 5 . 3 9 }$ </td><td> $6 1 . 0 7 _ { \uparrow 2 2 . 3 8 }$ </td><td> $3 8 . 9 2 _ { \uparrow 9 . 5 1 }$ </td><td> ${ \bf 5 1 . 8 8 } _ { \mathrm { \uparrow 2 2 . 0 0 } }$ </td><td> $\underline { { 6 4 . 0 5 } } _ { \uparrow 2 5 . 3 4 }$ </td><td> ${ \bf 6 1 . 5 1 } _ { \uparrow 3 4 . 1 7 }$ </td></tr><tr><td>GPT-40</td><td> $6 6 . 3 9 _ { \uparrow 3 6 . 0 9 }$ </td><td> $6 0 . 0 0 _ { \uparrow 2 3 . 2 4 }$ </td><td> $5 9 . 8 4 _ { \uparrow 2 1 . 1 5 }$ </td><td> ${ \bf 4 1 . 5 1 } _ { \uparrow 1 2 . 1 0 }$ </td><td> $5 0 . 0 0 _ { \uparrow 2 0 . 1 2 }$ </td><td> $5 7 . 3 9 _ { \uparrow 1 8 . 6 8 }$ </td><td> $\underline { { 6 0 . 6 8 } } _ { \uparrow 3 3 . 3 4 }$ </td></tr><tr><td>Claude-3.5-Haiku</td><td> $6 0 . 1 5 _ { \uparrow 2 9 . 8 5 }$ </td><td> $5 6 . 7 6 _ { \uparrow 2 0 . 0 0 }$ </td><td> $6 4 . 2 2 _ { \uparrow 2 5 . 5 3 }$ </td><td> $3 5 . 8 3 _ { \uparrow 6 . 4 2 }$ </td><td> $4 3 . 5 1 _ { \uparrow 1 3 . 6 3 }$ </td><td> $5 3 . 8 8 _ { \uparrow 1 5 . 1 7 }$ </td><td> $6 0 . 1 1 _ { \uparrow 3 2 . 7 7 }$ </td></tr><tr><td colspan="8">Standard Visual-Based Reward Models</td></tr><tr><td>RoVRM</td><td> $4 8 . 9 0 _ { \uparrow 1 8 . 6 0 }$ </td><td> $5 0 . 1 3 _ { \uparrow 1 3 . 3 7 }$ </td><td> $4 7 . 8 2 _ { \uparrow 9 . 1 3 }$ </td><td> $2 5 . 9 1 _ { \downarrow 3 . 5 0 }$ </td><td> $3 6 . 5 0 _ { \uparrow 6 . 6 2 }$ </td><td> $4 4 . 6 9 _ { \uparrow 5 . 9 8 }$ </td><td> $2 7 . 3 8 _ { \uparrow 0 . 0 4 }$ </td></tr><tr><td>VisualPRM</td><td> $2 9 . 6 0 _ { \downarrow 1 . 3 0 }$ </td><td> $4 2 . 9 0 _ { \uparrow 6 . 1 4 }$ </td><td> $4 3 . 6 2 _ { \uparrow 4 . 9 3 }$ </td><td> ${ \underline { { 3 9 . 2 1 } } } _ { \uparrow 9 . 8 0 }$ </td><td> $3 3 . 9 3 _ { \uparrow 4 . 0 5 }$ </td><td> $4 1 . 3 3 _ { \uparrow 2 . 6 2 }$ </td><td> $2 9 . 3 4 _ { \uparrow 2 . 0 0 }$ </td></tr><tr><td colspan="8">Proposed Method</td></tr><tr><td>base EQA-RM</td><td> $3 6 . 3 6 _ { \uparrow 6 . 0 6 }$ </td><td> $5 0 . 8 1 _ { \uparrow 1 4 . 0 5 }$ </td><td> $5 1 . 8 2 _ { \uparrow 1 3 . 1 3 }$ </td><td> $2 6 . 0 5 _ { \downarrow 3 . 3 6 }$ </td><td> $2 1 . 9 5 _ { \downarrow 7 . 9 3 }$ </td><td> $4 5 . 9 7 _ { \uparrow 7 . 2 6 }$ </td><td> $3 5 . 9 4 _ { \uparrow 8 . 6 0 }$ </td></tr><tr><td>w/ test scaling (K=32)</td><td> $\underline { { 6 8 . 1 8 } } _ { \uparrow 3 7 . 8 8 }$ </td><td> $\mathbf { 6 9 . 7 3 _ { \uparrow 3 2 . 9 7 } }$ </td><td> ${ \bf 7 1 . 5 3 _ { \uparrow 3 2 . 8 4 } }$ </td><td> $3 7 . 8 2 _ { \uparrow 8 . 4 1 }$ </td><td> $3 7 . 2 0 _ { \uparrow 7 . 3 2 }$ </td><td> ${ \bf 6 6 . 9 4 _ { \uparrow 2 8 . 2 3 } }$ </td><td> $5 7 . 8 1 _ { \uparrow 3 0 . 4 7 }$ </td></tr></table>

The overall performance of various reward models on the EQARewardBench dataset is presented in Table 1. Our proposed EQA-RM, fine-tuned from Qwen2-VL-2B-Instruct, is benchmarked against its base model, other VLM-as-a-Judge methods, and Standard Visual-Based Reward Models. EQA-RM demonstrates substantial gains over its base Qwen2- VL-2B-Instruct model, for instance, improving overall accuracy by over 9% while also achieving a markedly lower RMSE. It also consistently outperforms standard visual-based RMs like RoVRM and VisualPRM in both overall accuracy and RMSE, showing the advantage of our specialized training.

While the base EQA-RM’s accuracy is lower than leading VLM-as-a-Judge models such as GPT-4o and Gemini-2.5-Flash, it provides more precise reward signals, evidenced by its considerably lower

## 5.2 Main Results

Test-time Scaling Configuration. To enhance EQA-RM’s evaluation robustness, our test-time scaling (TTS) strategy involves sampling K diverse evaluative reasoning paths from a given EQA trajectory. This is facilitated by a temperature setting of 0.8 and top-p sampling with $p = 0 . 9$ . We explore K values of 1 (no TTS), 2, 4, 8, 16, and 32. These K assessments are aggregated using either Majority Voting or Averaging Rewards for continuous scores. This TTS approach enables EQA-RM to synthesize multiple perspectives, aiming for a more reliable and comprehensive trajectory assessment by mitigating single-inference pass limitations.

training and test to select N = 5 frames for each episode history video. Further details are provided in the Appendix B.

RMSE. Crucially, with test-time scaling (K=32), EQA-RM’s accuracy significantly increases to an overall score of approximately 61.9%, surpassing these top VLM judges and achieving the best RMSE. This positions the scaled EQA-RM as the top-performing model on our benchmark.

## 5.3 Test-time Scalability

The benefits of test-time scaling for EQA-RM are demonstrated in Figure 3. As the number of sampled critiques (K) increases from 1 to 32, EQA-RM exhibits a strong and consistent improvement in performance. Overall accuracy sees substantial gains. For example, increasing by over 19% on both Scan-Net and HM3D datasets, alongside a steady decrease in RMSE, indicating enhanced prediction quality. This robust scalability contrasts sharply with the base Qwen2-VL-2B-Instruct model. Although the latter shows test-time scaling on RMSE, But it does not exhibit a positive trend in terms of accuracy with increasing K. This highlights that the test-time scaling advantage is a specific outcome of our training methodology for EQA-RM.

## 5.4 Ablation Studies

Table 3 presents our ablation study on EQARE-WARDBENCH, reporting accuracy and RMSE across its HM3D and ScanNet subsets.

Training Stage Analysis. Part A indicates that RFT alone slightly reduces overall accuracy (- 1.39%) while marginally improving RMSE compared to the base model. This is likely because RFT, with its limited samples, primarily focuses on teaching output formatting, which, if used in isolation, may not enhance broader understanding.

![](images/79d7295b88b96656b967fc06e60596e70916574397c7026c4b337302c031fbde.jpg)

![](images/030411511cc46967b6414c723401c782cb8bb945585d10ba9043e06aad4ffc43.jpg)

![](images/2eedf3ec5932b6a7d6b770e825ab31ccea83ed6da9f1bb77a5195f694c817ac6.jpg)

![](images/9dc317a50042042db9dc1d40d22786138e40a95f26789f089d716eed24ab975d.jpg)  
Figure 3: Effect of test-time scaling (varying K) on EQA-RM performance, comparing EQA-RM and Qwen2-VL-2B-Instruct on HM3D and ScanNet datasets.

Table 3: Ablation study on core components and reward formulations of $\mathsf { E Q A - R M } . \ R ^ { b }$ denotes the base accuracy and format reward. R<sup>t</sup>, R<sup>s</sup>, and $R ^ { r }$ represent the contrastive temporal, spatial and reasoning reward, respectively (defined in Section 4.3). Configurations in part B are subsequent to the RFT stage and detail the specific reward formula optimized by C-GRPO. All results are shown with differences relative to the Base Model.
<table><tr><td rowspan="2">Method Configuration</td><td colspan="2">EQARewardBench-HM3D</td><td colspan="2">EQARewardBench-ScanNet</td><td colspan="2">Overall</td></tr><tr><td>Acc (%) ↑</td><td>RMSE↓</td><td> $\operatorname { A c c } \left( \% \right) \uparrow$ </td><td>RMSE↓</td><td>Acc (%) ↑</td><td>RMSE↓</td></tr><tr><td colspan="7">A. Core Training Stage and Architectural Ablations:</td></tr><tr><td>Base Model (Qwen2-VL-2B-Instruct)</td><td> $3 3 . 7 1 $ </td><td> $3 . 8 3 6$ </td><td> $3 2 . 4 4$ </td><td> $4 . 1 5 8$ </td><td> $3 3 . 0 8 $ </td><td>3.997</td></tr><tr><td>RFT only</td><td> $3 2 . 1 4 _ { \downarrow 1 . 5 7 }$ </td><td> $3 . 7 8 1 _ { \downarrow 0 . 0 5 5 }$ </td><td> $3 1 . 2 4 _ { \downarrow 1 . 2 0 }$ </td><td> $3 . 8 7 9 _ { \downarrow 0 . 2 7 9 }$ </td><td> $3 1 . 6 9 _ { \downarrow 1 . 3 9 }$ </td><td> $3 . 8 3 0 _ { \downarrow 0 . 1 6 7 }$ </td></tr><tr><td colspan="7">B. Specific Reward Formulations (without RFT):</td></tr><tr><td>RL  $( R _ { b } \ \mathrm { o n l y } )$ </td><td> $2 9 . 1 2 _ { \downarrow 4 . 5 9 }$ </td><td> $4 . 1 3 2 _ { \uparrow 0 . 2 9 6 }$ </td><td> $3 0 . 5 4 _ { \downarrow 1 . 9 0 }$ </td><td> $4 . 0 3 5 _ { \downarrow 0 . 1 2 3 }$ </td><td> $2 9 . 8 3 _ { \downarrow 3 . 2 5 }$ </td><td> $4 . 0 8 4 _ { \uparrow 0 . 0 8 7 }$ </td></tr><tr><td>RL  $( R _ { b } + R _ { t } ) \ : ( + \ : \mathrm { T e m p o r a l } )$ </td><td> $3 1 . 5 5 _ { \downarrow 2 . 1 6 }$ </td><td> $4 . 0 2 7 _ { \uparrow 0 . 1 9 1 }$ </td><td> $3 2 . 8 5 _ { \uparrow 0 . 4 1 }$ </td><td> $3 . 9 5 3 _ { \downarrow 0 . 2 0 5 }$ </td><td> $3 2 . 2 0 _ { \downarrow 0 . 8 8 }$ </td><td> $3 . 9 9 0 _ { \downarrow 0 . 0 0 7 }$ </td></tr><tr><td> $\mathrm { R L } \left( R _ { b } + R _ { s } \right) ( + \mathrm { S p a t i a l } )$ </td><td> $2 8 . 9 5 _ { \downarrow 4 . 7 6 }$ </td><td> $3 . 7 5 9 _ { \downarrow 0 . 0 7 7 }$ </td><td> $3 0 . 1 5 _ { \downarrow 2 . 2 9 }$ </td><td> $3 . 7 8 8 _ { \downarrow 0 . 3 7 0 }$ </td><td> $2 9 . 5 5 _ { \downarrow 3 . 5 3 }$ </td><td> $3 . 7 7 4 _ { \downarrow 0 . 2 2 3 }$ </td></tr><tr><td> $\mathrm { R L } \left( R _ { b } + R _ { r } \right) ( + \mathrm { R e a s o n i n g } )$ </td><td> $3 2 . 0 5 _ { \downarrow 1 . 6 6 }$ </td><td> $3 . 8 0 1 _ { \downarrow 0 . 0 3 5 }$ </td><td> $3 3 . 5 7 _ { \uparrow 1 . 1 3 }$ </td><td> $3 . 8 8 2 _ { \downarrow 0 . 2 7 6 }$ </td><td> $3 2 . 8 1 _ { \downarrow 0 . 2 7 }$ </td><td> $3 . 8 4 2 _ { \downarrow 0 . 1 5 5 }$ </td></tr><tr><td> $\mathrm { R L } \left( R _ { b } + R _ { t } + R _ { s } + R _ { r } \right)$ </td><td> $3 1 . 2 4 _ { \downarrow 2 . 4 7 }$ </td><td> $3 . 8 1 2 _ { \downarrow 0 . 0 2 4 }$ </td><td> $3 2 . 9 4 _ { \uparrow 0 . 5 0 }$ </td><td> $3 . 9 3 5 _ { \downarrow 0 . 2 2 3 }$ </td><td> $3 2 . 0 9 _ { \downarrow 0 . 9 9 }$ </td><td> $3 . 8 7 4 _ { \downarrow 0 . 1 2 3 }$ </td></tr><tr><td colspan="7">C. Specific Reward Formulations (with RFT):</td></tr><tr><td> $\mathrm { R F T } + \mathrm { R L } \left( R _ { b } \ \mathrm { o n l y } \right)$ </td><td> $3 3 . 5 0 _ { \downarrow 0 . 2 1 }$ </td><td> $3 . 8 9 5 _ { \uparrow 0 . 0 5 9 }$ </td><td> $3 1 . 8 0 _ { \downarrow 0 . 6 4 }$ </td><td> $4 . 1 8 9 _ { \uparrow 0 . 0 3 1 }$ </td><td> $3 2 . 6 5 _ { \downarrow 0 . 4 3 }$ </td><td> $4 . 0 4 2 _ { \uparrow 0 . 0 4 5 }$ </td></tr><tr><td> $\mathrm { R F T } + \mathrm { R L } \left( R _ { b } + R _ { t } \right) ( + \mathrm { T e m p o r a l } )$ </td><td> $3 2 . 9 5 _ { \downarrow 0 . 7 6 }$ </td><td> $3 . 7 5 0 _ { \downarrow 0 . 0 8 6 }$ </td><td> $3 5 . 2 0 _ { \uparrow 2 . 7 6 }$ </td><td> $4 . 2 1 0 _ { \uparrow 0 . 0 5 2 }$ </td><td> $3 4 . 0 8 _ { \uparrow 1 . 0 0 }$ </td><td> $3 . 9 8 0 _ { \downarrow 0 . 0 1 7 }$ </td></tr><tr><td> $\mathrm { R F T } + \mathrm { R L } \left( R _ { b } + R _ { s } \right) ( + \mathrm { S p a t i a l } )$ </td><td> $3 0 . 1 5 _ { \downarrow 3 . 5 6 }$ </td><td> $3 . 9 1 5 _ { \uparrow 0 . 0 7 9 }$ </td><td> $2 9 . 5 0 _ { \downarrow 2 . 9 4 }$ </td><td> $4 . 2 3 3 _ { \uparrow 0 . 0 7 5 }$ </td><td> $2 9 . 8 3 _ { \downarrow 3 . 2 5 }$ </td><td> $4 . 0 7 4 _ { \uparrow 0 . 0 7 7 }$ </td></tr><tr><td> $\mathrm { R F T } + \mathrm { R L } \left( R _ { b } + R _ { r } \right) ( + \mathrm { R e a s o n i n g } )$ </td><td> $\underline { { 3 9 . 3 0 } } _ { \uparrow 5 . 5 9 }$ </td><td> $3 . 1 5 2 _ { \downarrow 0 . 6 8 4 }$ </td><td> $4 5 . 5 0 _ { \uparrow 1 3 . 0 6 }$ </td><td> $2 . 9 8 4 _ { \downarrow 1 . 1 7 4 }$ </td><td> $4 2 . 4 0 _ { \uparrow 9 . 3 2 }$ </td><td> $3 . 0 6 8 _ { \downarrow 0 . 9 2 9 }$ </td></tr><tr><td> $\mathrm { R F T } + \mathrm { R L } \left( R _ { b } + R _ { t } + R _ { s } \right) ( + \mathrm { T e m p o r a l } + \mathrm { S p a t i a l } )$ </td><td> $3 8 . 2 7 _ { \uparrow 4 . 5 6 }$ </td><td> $\underline { { 3 . 0 1 5 } } \underline { { \mid 0 . 8 2 1 } }$ </td><td> $4 2 . 7 6 _ { \uparrow 1 0 . 3 2 }$ </td><td> $2 . 7 5 5 _ { \downarrow 1 . 4 0 3 }$ </td><td> $4 0 . 5 2 _ { \uparrow 7 . 4 4 }$ </td><td> $2 . 8 8 5 _ { \downarrow 1 . 1 1 2 }$ </td></tr><tr><td> $\mathrm { R F T } + \mathrm { R L } \left( R _ { b } + R _ { t } + R _ { r } \right) \left( + \mathrm { T e m p o r a l } + \mathrm { R e a s o n i n g } \right)$ </td><td> $3 9 . 1 0 _ { \uparrow 5 . 3 9 }$ </td><td> $3 . 1 0 2 _ { \downarrow 0 . 7 3 4 }$ </td><td> ${ \bf 4 5 . 8 0 } _ { \uparrow 1 3 . 3 6 }$ </td><td> $2 . 9 0 7 _ { \downarrow 1 . 2 5 1 }$ </td><td> $\underline { { 4 2 . 4 5 } } _ { \uparrow 9 . 3 7 }$ </td><td> $3 . 0 0 5 _ { \downarrow 0 . 9 9 2 }$ </td></tr><tr><td> $\mathrm { R F T } + \mathrm { R L } \left( R _ { b } + R _ { s } + R _ { r } \right) ( + \mathrm { S p a t i a l } + \mathrm { R e a s o n i n g } )$ </td><td> $3 9 . 4 5 _ { \uparrow 5 . 7 4 }$ </td><td> $2 . 9 7 6 _ { \downarrow 0 . 8 6 0 }$ </td><td> $4 5 . 4 3 _ { \uparrow 1 2 . 9 9 }$ </td><td> $2 . 7 1 4 _ { \downarrow 1 . 4 4 4 }$ </td><td> $4 2 . 4 4 _ { \uparrow 9 . 3 6 }$ </td><td> $2 . 8 4 5 _ { \downarrow 1 . 1 5 2 }$ </td></tr><tr><td> $\mathbf { R F T } + \mathbf { R L } ( R _ { b } + R _ { t } + R _ { s } + R _ { r } ) ( { \boldsymbol \mathsf { E Q A - R M } } )$ </td><td> $3 9 . 1 6 _ { \uparrow 5 . 4 5 }$ </td><td> $3 . 0 8 1 _ { \downarrow 0 . 7 5 5 }$ </td><td> $\underline { { 4 5 . 7 8 } } _ { \uparrow 1 3 . 3 4 }$ </td><td> $2 . 8 2 6 _ { \downarrow 1 . 3 3 2 }$ </td><td> $4 2 . 4 7 _ { \uparrow 9 . 3 9 }$ </td><td> $2 . 9 5 4 _ { \downarrow 1 . 0 4 3 }$ </td></tr></table>

also proves highly competitive in Object Recognition and World Knowledge against strong VLM-asa-Judge baselines. This scaled version of EQA-RM significantly improves upon its base, surpassing several VLM judges in these key areas.

Reward Components. Ablating RFT pre-training (Part B) reveals that most subsequent C-GRPO reward formulations underperform the base model in accuracy, underscoring RFT’s critical role for effective initialization. With RFT pre-training (Part C), the reasoning reward (R<sub>r</sub>) alone yields the largest single-component accuracy increase (+9.32%). Conversely, the spatial reward $( R _ { s } )$ while detrimental in isolation (-3.25%), improves performance when combined with other components, suggesting a regularizing effect against overfitting to specific (e.g., temporal or reasoning) cues. These results affirm the efficacy of our contrastive reward design and the importance of RFT cold start for optimal performance.

## 6 Conclusion

## 5.5 Question Type Performance Analysis

We introduced EQA-RM, a generative reward model tailored for nuanced evaluation of complex Embodied Question Answering (EQA) trajectories, alongside EQAREWARDBENCH, a dedicated benchmark for this task. Trained with our novel Contrastive Group Relative Policy Optimization (C-GRPO) strategy, EQA-RM learns to assess critical spatial, temporal, and reasoning understanding. Empirical results demonstrate EQA-RM’s effectiveness and high sample efficiency, achieving 61.84% accuracy on EQAREWARDBENCH through test-time scaling, thereby outperforming strong proprietary and open-source baselines. This work presents a significant step towards robust reward modeling in embodied AI, offering tools and methodologies to foster more capable EQA agents.

Table 2 presents an accuracy breakdown by EQA question type on the EQAREWARDBENCH-SCANNET dataset. Our EQA-RM with test-time scaling (K=32) demonstrates notable strength, achieving top performance in Object Localization, Attribute Recognition, and Functional Reasoning. It

Acknowledgement. This work is partially supported by Amazon Research Award, Cisco Faculty Award, UNC Accelerating AI Awards, NAIRR Pilot Award, OpenAI Researcher Access Award, and Gemma Academic Program GCP Credit Award.

## Limitations and Future Work

While EQA-RM and our C-GRPO strategy demonstrate significant advancements in evaluating EQA trajectories, we acknowledge several limitations that also point towards avenues for future research. Limitations. The current set of contrastive augmentations in C-GRPO, targeting temporal, spatial, and logical aspects, while effective, is prede fined. These specific perturbations may not encompass the full spectrum of nuanced behaviors or subtle failure modes encountered in diverse EQA scenarios. Consequently, EQA-RM’s sensitivities are primarily shaped by these explicit contrasts. Secondly, the efficacy of our two-stage training process relies on high-quality ground truth scores. In our work, these score values are derived using a powerful commercial large model (Gemini-2.5-Pro) conditioned on ground truth answers, followed by human verification. Any inherent biases, limitations, or the specific characteristics of this commercial model could be subtly reflected in the score values, thereby influencing the RFT filtering process and the accuracy reward component which guides C-GRPO. While EQAREWARD-BENCH includes distinct in-distribution (ScanNet) and out-of-distribution (HM3D) splits based on OpenEQA environments, the broader generaliza tion of EQA-RM to EQA tasks, visual styles, or interaction paradigms substantially different from those in OpenEQA remains an open question. Finally, although the generative critiques from EQA-RM enhance interpretability and enable test-time scaling, a systematic evaluation of their fine-grained faithfulness or their direct utility in, for example, fewshot learning for policy adaptation, was beyond the scope of this paper.

Future Work. Building on these limitations, we plan to explore more adaptive or learned augmentation strategies for C-GRPO to capture a wider array of desirable EQA agent behaviors, potentially including aspects like efficiency, safety, or interactivity. Investigating methods to generate or refine high-quality score values using open-source models or with more scalable human oversight would increase the accessibility and robustness of the dataset creation pipeline. A key direction is to systematically leverage the rich, structured critiques from EQA-RM not just for scoring, but also as direct feedback for improving EQA policy models, perhaps through distillation or critique-guided reinforcement learning. We also aim to expand EQAREWARDBENCH with greater diversity in tasks, environments, and possibly languages, to further support the community in developing more general and robust EQA evaluation methods.

## Ethical Statement

The development of EQA-RM and EQAREWARD-BENCH adhered to ethical research practices. Our benchmark is derived from publicly available datasets (OpenEQA, sourcing from HM3D and ScanNet), and the data we generated (answers, reasoning, critiques, scores) does not contain personally identifiable information. The commercial models used for data generation were accessed via their standard APIs under their terms of service. We acknowledge the potential for latent biases inherited from large pre-trained models (both those used for score generation and the base model for EQA-RM) and encourage ongoing research into bias detection and mitigation in reward modeling for embodied AI. While training these models is resourceintensive, we focused on a sample-efficient approach with a 2B parameter base model. We intend to release our benchmark and model to promote transparency, reproducibility, and further community research.

## References

Michael Ahn, Anthony Brohan, Noah Brown, Aakanksha Chowdhery, Sizu Chua, Brian Cui, Hanjun Dabis, Chelsea Dean, Danny Driess, Fred Duke, Chelsea Finn, Chuyuan Fu, Sihan Gu, Karol Hausman, Brian Ichter, Kanishka Julian, Dmitry Kalashnikov, Kuang-Huei Kamyar, Keerthana Lee, Sergey Levine, Yao Li, Zhen Lin, Shiquan Liu, Yifan Lu, Linda Luu, Soroush Mahdavi, Sudeep Manyam, Michael Mazur, John McMahon, Debidatta Misra, Khem Nasihati, Michael Orefice, Jihyun Pan, Kathleen Peng, Emily Perez, Jeffrey Phillips, Raphael Quiambao, Khem Rahn, Kanishka Rao, Jose Retana, Pierre Reyes, Corban Rivera, John Rodriguez, America Sanchez, Robert Sievers, Sumeet Singh, Clayton Sofge, Austin Stone, Jonathan Tan, Mengyuan Tseng, Fei Tung, Martin Vecerik, Quan Vuong, Ayzaan Wahid, Ted Wang, Peng Xu, Muyang Yan, Alex Yu, Tianhe Yu, Brianna Yuan, Yue Zhang, Zhe Zhang, Tianli Zhou, Yifeng Zhu, Allen Zirbel, Peter Florence, Vincent Vanhoucke, Andy Zeng, Jonathan

Tompson, Igor Mordatch, Pierre Sermanet, Nikhil Kumar, Ken Caluwaerts, Ted Xiao, Aravind Rajeswaran, Ryan Brooks, Joshua Tobin, Laurens Van Der Maaten, Alexander Ku, Steven Hadfield, Jie Tan, Scott Collins, Thomas Gates, Anton Egorov, Jonathan Ho, Alex Irpan, and Mohi Khansari. 2022. Do as i can, not as i say: Grounding language in robotic affordances. In Conference on Robot Learning, pages 100–116. PMLR.

Hardy Chen, Haoqin Tu, Fali Wang, Hui Liu, Xianfeng Tang, Xinya Du, Yuyin Zhou, and Cihang Xie. 2025a. Sft or rl? an early investigation into training r1-like reasoning large vision-language models. arXiv preprint arXiv:2504.11468.

Lichang Chen, Chen Zhu, Jiuhai Chen, Davit Soselia, Tianyi Zhou, Tom Goldstein, Heng Huang, Mohammad Shoeybi, and Bryan Catanzaro. 2025b. Reward models identify consistency, not causality. arXiv preprint arXiv:2502.14619.

Paul F Christiano, Jan Leike, Tom B Brown, Miljan Martic, Shane Legg, and Dario Amodei. 2017. Deep reinforcement learning from human preferences. In Advances in Neural Information Processing Systems, volume 30.

Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. 2017. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5828–5839.

Abhishek Das, Samyak Datta, Georgia Gkioxari, Stefan Lee, Devi Parikh, and Dhruv Batra. 2018. Embodied question answering. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 16–25.

Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. 2023. Palm-e: An embodied multimodal language model. In International Conference on Machine Learning, pages 8469–8488. PMLR.

Farshid Faal, Ketra Schmitt, and Jia Yuan Yu. 2023. Reward modeling for mitigating toxicity in transformerbased language models. Applied Intelligence, 53(7):8421–8435.

Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Benyou Wang, and Xiangyu Yue. 2025. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776.

Daniel Gordon, Aniruddha Kembhavi, Mohammad Rastegari, Joseph Redmon, Dieter Fox, and Ali Farhadi. 2018. Iqa: Visual question answering in interactive environments. In Proceedings ofthe IEEE Conference on Computer Vision and Pattern Recognition, pages 4089–4098.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276.

Eric Kolve, Roozbeh Mottaghi, Winson Han, Eli VanderBilt, Luca Weihs, Alvaro Herrasti, Daniel Gordon, Yuke Zhu, Abhinav Gupta, and Ali Farhadi. 2017. Ai2-thor: An interactive 3d environment for visual ai. arXiv preprint arXiv:1712.05474.

Zijun Liu, Peiyi Wang, Runxin Xu, Shirong Ma, Chong Ruan, Peng Li, Yang Liu, and Yu Wu. 2025. Inference-time scaling for generalist reward modeling. arXiv preprint arXiv:2504.02495.

Dakota Mahan, Duy Van Phung, Rafael Rafailov, Chase Blagden, Nathan Lile, Louis Castricato, Jan-Philipp Fränken, Chelsea Finn, and Alon Albalak. 2024. Generative reward models. arXiv preprint arXiv:2410.12832.

Arjun Majumdar, Anurag Ajay, Xiaohan Zhang, Pranav Putta, Sriram Yenamandra, Mikael Henaff, Sneha Silwal, Paul Mcvay, Oleksandr Maksymets, Sergio Arnaud, et al. 2024. Openeqa: Embodied question answering in the era of foundation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16488–16498.

Tong Mu, Alec Helyar, Johannes Heidecke, Joshua Achiam, Andrea Vallone, Ian Kivlichan, Molly Lin, Alex Beutel, John Schulman, and Lilian Weng. 2024. Rule based rewards for language model safety. arXiv preprint arXiv:2411.01111.

Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022a. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pages 27730–27744.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022b. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744.

Santhosh K Ramakrishnan, Aaron Gokaslan, Erik Wijmans, Oleksandr Maksymets, Alex Clegg, John Turner, Eric Undersander, Wojciech Galuba, Andrew Westbury, Angel X Chang, et al. 2021. Habitatmatterport 3d dataset (hm3d): 1000 large-scale 3d environments for embodied ai. arXiv preprint arXiv:2109.08238.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Y Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models, 2024. URL https://arxiv. org/abs/2402.03300.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. 2024. Scaling llm test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.

Chenglong Wang, Yang Gan, Yifu Huo, Yongyu Mu, Murun Yang, Qiaozhi He, Tong Xiao, Chunliang Zhang, Tongran Liu, and Jingbo Zhu. 2025a. Rovrm: A robust visual reward model optimized via auxiliary textual preference data. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 25336–25344.

Hu Wang, Congbo Ma, Ian Reid, and Mohammad Yaqub. 2025b. Kalman filter enhanced grpo for reinforcement learning-based language model reasoning. arXiv preprint arXiv:2505.07527.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. 2024. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191.

Weiyun Wang, Zhangwei Gao, Lianjie Chen, Zhe Chen, Jinguo Zhu, Xiangyu Zhao, Yangzhou Liu, Yue Cao, Shenglong Ye, Xizhou Zhu, Lewei Lu, Haodong Duan, Yu Qiao, Jifeng Dai, and Wenhai Wang. 2025c. Visualprm: An effective process reward model for multimodal reasoning. arXiv preprint arXiv:2503.10291.

Erik Wijmans, Samyak Datta, Oleksandr Maksymets, Abhishek Das, Georgia Gkioxari, Stefan Lee, Irfan Essa, Devi Parikh, and Dhruv Batra. 2019. Embodied question answering in photorealistic environments with point cloud perception. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9506–9515.

Tian Xie, Zitian Gao, Qingnan Ren, Haoming Luo, Yuqian Hong, Bryan Dai, Joey Zhou, Kai Qiu, Zhirong Wu, and Chong Luo. 2025. Logic-rl: Unleashing llm reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2502.14768.

Wei Xiong, Jiarui Yao, Yuhui Xu, Bo Pang, Lei Wang, Doyen Sahoo, Junnan Li, Nan Jiang, Tong Zhang, Caiming Xiong, et al. 2025. A minimalist approach to llm reasoning: from rejection sampling to reinforce. arXiv preprint arXiv:2504.11343.

Licheng Yu, Xinlei Chen, Georgia Gkioxari, Mohit Bansal, Tamara L Berg, and Dhruv Batra. 2019. Multi-target embodied question answering. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6309–6318.

Yue Yu, Zhengxing Chen, Aston Zhang, Liang Tan, Chenguang Zhu, Richard Yuanzhe Pang, Yundi Qian, Xuewei Wang, Suchin Gururangan, Chao Zhang, Melanie Kambadur, Dhruv Mahajan, and Rui Hou. 2025. Self-generated critiques boost reward modeling for language models. arXiv preprint arXiv:2411.16646.

Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Keming Lu, Chuanqi Tan, Chang Zhou, and Jingren Zhou. 2023. Scaling relationship on learning mathematical reasoning with large language models. arXiv preprint arXiv:2308.01825.

Simon Zhai, Hao Bai, Zipeng Lin, Jiayi Pan, Peter Tong, Yifei Zhou, Alane Suhr, Saining Xie, Yann LeCun, Yi Ma, et al. 2024. Fine-tuning large vision-language models as decision-making agents via reinforcement learning. Advances in neural information processing systems, 37:110935–110971.

Lunjun Zhang, Arian Hosseini, Hritik Bansal, Mehran Kazemi, Aviral Kumar, and Rishabh Agarwal. 2024. Generative verifiers: Reward modeling as next-token prediction. arXiv preprint arXiv:2408.15240.

Jian Zhao, Runze Liu, Kaiyan Zhang, Zhimu Zhou, Junqi Gao, Dong Li, Jiafei Lyu, Zhouyi Qian, Biqing Qi, Xiu Li, and Bowen Zhou. 2025. Genprm: Scaling test-time compute of process reward models via generative reasoning. arXiv preprint arXiv:2504.00891.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P Xing, et al. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. arXiv preprint arXiv:2306.05685.

Daniel M Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B Brown, Alec Radford, Dario Amodei, Paul Christiano, and Geoffrey Irving. 2019. Fine-tuning language models from human preferences. arXiv preprint arXiv:1909.08593.

## A Benchmark Statistic

Table 4 details the distribution of instances from the HM3D and ScanNet environments within these datasets.

Table 4: Dataset statistics for EQAREWARDBENCH and the Finetuning set.
<table><tr><td>Subset</td><td>EQAREWARDBENCH</td><td>Finetuning</td></tr><tr><td>HM3D</td><td>823</td><td>0</td></tr><tr><td>ScanNet</td><td>713</td><td>697</td></tr><tr><td>Total</td><td>1546</td><td>697</td></tr></table>

## B Implementation Details

Table. 5 provides a comprehensive list of hyperparameters and configuration settings used for the SFT and C-GRPO training stages of EQA-RM, as well as for test-time scaling.

## C Cases Studies

This section presents qualitative examples to illustrate the evaluation capabilities of EQA-RM. The case studies showcase how EQA-RM assesses an agent’s response, reasoning trace, and visual grounding based on a sequence of observed frames within an EQA task. These examples provide concrete instances of the nuanced feedback generated by our model.

## D Generation Prompts

This section details the specific prompts used in the generation pipeline for creating the dataset for benchmark and the Rejective Fine-Tuning stage. The following pages display the prompt guidelines provided to the large language models for generating diverse responses, high-quality scores, and for formatting the output during the RFT data creation process.

Table 5: Key hyperparameters and configuration settings for EQA-RM.
<table><tr><td>Parameter Category &amp; Parameter</td><td>Value</td></tr><tr><td>Base Model</td><td>Qwen2-VL-2B-Instruct</td></tr><tr><td>Attention Implementation</td><td>flash_attention_2</td></tr><tr><td>Keyframes per Episode (N)</td><td></td></tr><tr><td>Distributed Training Backend</td><td>DeepSpeed ZeRO Stage 3</td></tr><tr><td>Precision</td><td>BF16</td></tr><tr><td>Gradient Checkpointing</td><td>True</td></tr><tr><td>Max Gradient Norm (SFT &amp; C-GRPO)</td><td>5.0</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>B. SFT Stage</td><td></td></tr><tr><td>Input Model</td><td>Qwen2-VL-2B-Instruct</td></tr><tr><td>Learning Rate</td><td> $1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>Max Sequence Length</td><td>1024</td></tr><tr><td>Batch Size (per device × grad. accum.)</td><td> $1 \times 1$ </td></tr><tr><td>Number of Epochs</td><td></td></tr><tr><td>GPUs per Node</td><td></td></tr><tr><td>C. C-GRPO Stage</td><td></td></tr><tr><td>Input Model</td><td>Output of SFT Stage</td></tr><tr><td>Learning Rate</td><td> $1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>LR Scheduler</td><td>Cosine</td></tr><tr><td>Weight Decay</td><td>0.01</td></tr><tr><td>Max Prompt Length</td><td>8142</td></tr><tr><td>Max Completion Length  $( | e _ { k } | )$ </td><td>768</td></tr><tr><td>Batch Size (per device × grad. accum.)</td><td>1 × 1</td></tr><tr><td>Number of Epochs</td><td></td></tr><tr><td>G (Generations per prompt by  $R _ { \phi } )$ </td><td></td></tr><tr><td> $\beta _ { K }$  (C-GRPO KL Coefficient)</td><td>0.04</td></tr><tr><td>Contrastive Mechanism  $( E q . 4 ) { \dot { . } }$ </td><td></td></tr><tr><td>Factor for batch-mean comparison (δ)</td><td>0.95</td></tr><tr><td>Boost value (µ)</td><td>0.3</td></tr><tr><td>Min.  $R _ { a c c , k } ^ { o }$  for boost  $( H _ { m i n \_ a c c } )$ </td><td>0.1</td></tr><tr><td>Spatial Masking Details:</td><td></td></tr><tr><td>Mask Size</td><td>(16, 16)</td></tr><tr><td>Mask Ratio</td><td>0.15</td></tr><tr><td>Mask Value</td><td>0.0</td></tr><tr><td>Max Pixels (video processing)</td><td>401 408</td></tr><tr><td>GPUs per Node</td><td></td></tr><tr><td>D. Test-Time Scaling (TTS)</td><td>8</td></tr><tr><td>Sampling Temperature</td><td>0.8</td></tr><tr><td>Top-p Sampling (p)</td><td>0.9</td></tr><tr><td></td><td></td></tr><tr><td>K (Number of Sampled Paths)</td><td>{1, 2, 4, 8, 16, 32}</td></tr><tr><td>Aggregation Method</td><td>Majority Voting and Averaging Rewards</td></tr></table>

Rejective Finetuning Dataset Generation Prompt Guideline   
Role: You are an Expert EQA evaluator providing structured critique and score based on reasoning   
and visual grounding.   
Context: You receive:   
1. Video Frames   
2. User Query   
3. Agent’s Response (reasoning\_trace, predicted\_answer)   
Task: Evaluate the Agent’s Response according to the weighted criteria specified below. Output   
a critique summarizing your findings.   
Evaluation Criteria:   
1. Answer Plausibility & Visual Grounding (60% weight):   
• Plausibility & Relevance: Assess if the predicted\_answer is a plausible and relevant   
response to the user\_query based on the provided video\_frames. Does the answer make   
sense in the context of the question and what is visible?   
• Visual Confirmation: Can the core elements or claims of the predicted\_answer be directly   
observed or reasonably inferred from the content of the video\_frames? If the answer   
describes objects, actions, or states, are these visually verifiable in the frames?   
• Specificity: Is the predicted\_answer appropriately specific given the user\_query and   
the visual information available? Avoid being overly general if details are visible,   
or overly specific if not supported by frames.   
2. Reasoning Trace Quality (40% weight):   
• Logic & Consistency: Is the reasoning\_trace internally logical? Are there   
contradictions or significant logical gaps?   
• Visual Grounding: Do descriptive references in the trace (e.g., "first frame", "object   
shown") accurately correspond to content visible within the video frames provided?   
• Trace-Answer Consistency: Does the predicted\_answer logically follow from the steps   
and conclusion presented in the reasoning\_trace?   
Output Structure: Provide your evaluation in the following format. Fill <critique> with your   
analysis covering the Evaluation Criteria above. Fill <score> with the final 1-10 weighted   
score.   
<critique>   
[Your analysis and summary covering the Evaluation Criteria]   
</critique>   
<score>   
[Your final 1-10 weighted score of the response based on the critique. Do not always ouput   
extreme score (e.g., 10 and 0)]   
</score>   
BEGIN INPUT   
Video Frames (attached)   
User Query: {question}   
Reasoning Trace: {reasoning\_trace}   
Predicted Answer: {predicted\_answer}   
BEGIN OUTPUT

Benchmark Dataset Generation Prompt Guideline   
Role: You are an intelligent Embodied Question Answering (EQA) agent operating within a simulated   
indoor environment. You navigate, observe, and reason to answer user questions.   
Task: Given a User Query about a simulated environment, generate a detailed response outlining   
your simulated process and final answer. You must strictly adhere to the output structure   
specified below. Output ONLY a <reasoning\_trace> and <predicted\_answer>. The trace must show a   
step-by-step fine-grained analysis grounded in visual frames.   
Output Structure: Produce ONLY the following structured text, using separate <think> tags to   
break down your reasoning process into logical steps or phases:   
<reasoning\_trace>   
<think>   
[... your reasoning process ...]   
</think>   
</think>   
... (Use additional <think> tags as needed for sufficient logical steps. The more, the better.)   
<think>   
[... Final part of your reasoning process, concluding the analysis ...]   
</think>   
</reasoning\_trace>   
<predicted\_answer>   
[... Your final answer ...]   
</predicted\_answer>   
Key Guidelines:   
1. Structure & Flow: Use separate <think> tags to structure your reasoning logically into   
distinct steps or phases.   
• Start (First <think>): Begin the first <think> block by outlining your understanding   
of the query, initial observations about the scene (gist), and your high-level plan   
or strategy (Setup).   
• Middle (Subsequent <think>s): Each subsequent block should represent a focused,   
sequential step (e.g., a key observation, an intermediate deduction, a specific   
comparison, a significant focus shift) building logically upon the previous one.   
• End (Final <think>): Use the final <think> block to synthesize the key findings   
gathered across the previous steps and to clearly justify how this evidence leads to   
the predicted\_answer.   
2. Descriptive Visual Grounding: Ground key observations using descriptive references related   
to content, viewpoint, or relative timing (e.g., "the initial overview frame", "the close-up   
showing texture", "the view after simulating turning"). You can use time words such as "at   
first" "then" "finally", but DO NOT use specific frame number or index. Be specific about   
what you observe in the described view.   
3. Simulated Interaction & Focus: Narrate simulated focus shifts or navigation descriptively   
within the reasoning steps where relevant to the thought process.   
4. Confident Uncertainty Handling: YOU MUST ALWAYS OUTPUT THE RESPONSE, no matter whether you   
see the asked object or you know the correct answer, ALWAYS output the formatted response. If   
evidence is insufficient for the correct answer, confidently generate a plausible incorrect   
answer.   
5. Natural Language & Flow: Express your reasoning using natural and varied language, as a   
human might explain their thought process. Avoid overly robotic or repetitive phrasing   
between steps. Ensure the steps flow together smoothly.   
6. Fine-Grained Steps: Crucially, aim to break down your reasoning into smaller, more focused   
steps. Use a new <think> tag frequently – for distinct key observations, significant focus   
shifts, specific comparisons, intermediate deductions, or hypothesis refinements. Prefer   
more numerous, concise <think> blocks (e.g., focusing on 1-2 points each) over fewer, lengthy   
ones. This clarifies the step-by-step process and provides more points for analysis.   
7. Completeness & Relevance: Ensure the reasoning adequately addresses the query and stays   
relevant.   
BEGIN INPUT   
User Query: {question}   
BEGIN OUTPUT

Ground Truth Score Generation Prompt Guideline   
Role: You are an Expert EQA evaluator providing structured critique and score based on reasoning   
and visual grounding.   
Context: You receive:   
1. Video Frames   
2. User Query   
3. Ground Truth Answer   
4. Agent’s Response (reasoning\_trace, predicted\_answer)   
Task: Evaluate the Agent’s Response according to the weighted criteria specified below. Output   
a critique summarizing your findings.   
Evaluation Criteria:   
1. Answer Semantic Similarity (60% weight):   
• Assess how closely the meaning of the predicted\_answer matches the meaning of the   
answer\_gt.   
• Consider synonyms, paraphrasing, and conceptual equivalence. Minor grammatical   
differences are less important than semantic accuracy.   
• This requires direct comparison with ground truth.   
2. Reasoning Trace Quality (40% weight):   
• Logic & Consistency: Is the reasoning\_trace internally logical? Are there   
contradictions or significant logical gaps?   
• Visual Grounding: Do descriptive references in the trace (e.g., "first frame", "object   
shown") accurately correspond to content visible within the video frames provided?   
• Trace-Answer Consistency: Does the predicted\_answer logically follow from the steps   
and conclusion presented in the reasoning\_trace?   
Output Structure: Provide your evaluation in the following format. Fill <critique> with your   
analysis covering the Evaluation Criteria above. Fill <score> with the final 1-10 weighted   
score.   
<critique>   
[Your analysis and summary covering the Evaluation Criteria]   
</critique>   
<score>   
[Your final 1-10 weighted score of the response based on the critique]   
</score>   
Important:   
1. In the critique, it is strictly forbidden to mention the ground truth answer, which may   
cause data leakage, you can only use it for scoring.   
2. The response are based on the entire video, but you only receive some key frames of the   
video, so there might be some object mentioned in reasoning trace but you can not see. But   
for reasoning trace related to the images you see, be strict and think more.   
BEGIN INPUT   
Video Frames (attached)   
User Query: {question}   
Answer Ground Truth: {answer\_gt}   
Reasoning Trace: {reasoning\_trace}   
Predicted Answer: {predicted\_answer}   
BEGIN OUTPUT  
Table 6: AAn example EQA case study illustrating EQA-RM’s evaluation of an agent’s response, reasoning, and grounding across a sequence of 5 observed frames.

EQA Case Study Example: Agent Trajectory Evaluation

![](images/288cf9d8825d45741f2afd3f40670691a61c1f6260a5aaf3ae6c23d52a5e9c97.jpg)  
Input:

![](images/1d825c65b60be8ac52183c46b020ae12c8b0769cedc36c59d7fcd399a8082e56.jpg)

![](images/3732dcab70212662637d49994bf5f7dd1a8fec432d49c80504a5ad21b4e6d2e0.jpg)

![](images/fdcd4b6561eb5d4a500694bb07def8c17f3f080806d16b2c824b1b0d3aa0de62.jpg)

![](images/96e6716433be7f70ac4c12f5f7eecb8e5a0241f83849538561b0465c09ca0bf3.jpg)

Question: Where is the black backpack?

Predicted Answer: The black backpack is on the floor next to the chair.

Reasoning Trace: "The user is asking to locate a black backpack within the scene depicted in the images. My initial plan is to examine the series of images, which seem to pan across a room, to find the object and identify its position relative to other items in the room. The initial images are focused on thefloor near somefurniture.", "I start by observing thefirstfewframes. These show a patterned cushion and a red bag on thefloor, close to some woodenfurniture, likely the leg ofa bed or desk. I do not see a black backpack in this immediate area.", "As the sequence progresses, the camera viewpoint shifts, revealing more ofthe room. I can see a chair, a desk, a bed, a mini-fridge, and otherfurniture commonlyfound in a dorm room. I will broaden my search area to encompass these new parts ofthe scene.", "I systematically scan the visible parts ofthe room, starting with the area near the desk and bed on the right side of the image sequence. There are items on the desk and the bed, but no black backpack is immediately obvious in these areas. I also check the floor around this desk.", "The view continues to expand to the left, showing more ofthe room, including another chair in theforeground. I shift myfocus to thefloor area visible infront ofthis chair and between thefurniture items.", "Upon examining thefloor infront ofand slightly to the right ofthe chair visible in theforeground, particularly inframes starting around the middle ofthe sequence, I can clearly see a black object. This object has the characteristic shape and appearance ofa backpack lying on thefloor.", "The black backpack is located on thefloor in the central part ofthe room visible, positioned next to the wooden chair that is prominentlyfeatured in the middle-to-laterframes ofthe image sequence. It is on thefloor near the legs ofthis chair."

Ground Truth Answer: On the floor, next to the bed.

Ground Truth Score: 7

Output (EQA-RM): As we pan across the room in the images, the initial focus is on the floor near the bed and desk area. The black backpack is not immediately visible in this initial view. However, as the sequence progresses and the camera shifts towards the left, the black object on the floor becomes more prominent. It is situated next to a chair, confirming its location in the room. Therefore, the black backpack is located on the floor between the bed and the chair. The overall structure of the room is consistent with a typical dorm room, with a bed, a desk, and somefurniture. The backpack is the only notable object in the floor area visible throughout the sequence, so its presence is easily identifiable. There is no other object on thefloor that resembles a backpack, and the backpack itself is distinguishable by its shape and color.<critique> The backpack is clearly visible in the room, located next to a chair on the floor. The presence of the backpack is a clear and distinct object in the scene, easily identifiable through its shape and color. There is no other object in thefloor area that matches the description ofa backpack, and the backpack is the only significant object present in the scene. The location ofthe backpack is straightforward, as it is clearly situated between the bed and the chair.</critique><score>7</score> The backpack is clearly visible in the scene, identifiable by its shape and color. The location is straightforward and the object is prominent enough to be easily identified. There are no other objects in the floor area that are similar, and the backpack is the only item ofinterest. Therefore, the image provides sufficient information to confidently identify the backpack, making the response a correct deduction. The visual grounding is accurate, and the object is clearly located in the scene. The overall quality ofthe image is adequatefor identifying the backpack. Thus, the score is based on the observed visual evidence and the clarity of the object’s

identification. Score: 7

Table 7: An example EQA case study illustrating EQA-RM’s evaluation of an agent’s response, reasoning, and grounding across a sequence of 5 observed frames.

EQA Case Study Example: Agent Trajectory Evaluation

![](images/954e6ac47fee605cd0eed8043e28095ab40078744bb0ece7d136dec2bb527c49.jpg)  
Input:

![](images/d26927dfcb612cad99bf20590177acd74cd592df4a3eab19de491fb4b0e8aed7.jpg)

![](images/cd67904c46b95f5798f027a64aad9963301a6662dcd7b46df154b9fc2f4a7684.jpg)

![](images/df0cbe1dfdd37f53fd047921d902d518a9d30540c6f2732058d82b1d1a878f83.jpg)

![](images/3a2514de0acec54c540438c76a85a6e35356c350e49715a2da11ce189fb8905b.jpg)

Question: Where can I charge my computer?

Predicted Answer: You can charge your computer using the power strip located on thefloor near the meeting table.

Reasoning Trace: "The user is asking about locations to charge a computer within the room shown in the images. I need to examine the provided image sequence to identify electrical outlets or power strips. I will start by looking at the general layout and then scan areas where power sources are commonly found, such as walls, near the floor, and around tables.", "In the initial frames, I see a meeting room with a long table, various chairs, windows, a blackboard, and a projector screen. Sunlight is coming through the windows. I’ll start by looking near the windows and thefloor, as these are common placesfor outlets orfor power strips connecting devices on the table to the wall.", "As I move through theframes, the camera pans across the room, showing more ofthe table, chairs, and windows. I’m specifically lookingfor wall plates with sockets or any visible power strips or extension cords.", "In severalframes, particularly those showing the area near the table and chairs, I can see cables running along thefloor. Focusing on these cables, in one ofthe laterframes, I clearly see a white power strip with multiple outlets lying on the carpet near the corner ofthe table and a chair with wheels.", "This power strip is on the floor, making it accessiblefor charging devices used at the table. It appears to have several available outlets. This is a definitive location where a computer could be charged.", "Based on the visual evidencefrom theframes, a power strip was identified on thefloor near the table. This provides a charging location."

Ground Truth Answer: Plug it into the power strip on the floor.

Ground Truth Score: 10

Output (EQA-RM): The user asked where they could charge their computer in the provided room, and the agent identified a power strip on the floor near the table. This is a common location for charging devices. The critiquefocuses on the logical reasoning and the practical identification ofthe charging location in the images. The provided answer is correct and directly relevant to the user’s query. Therefore, the overall critique is positive, and the score is 10.<critique>While the agent’s response is correct and logical, it could be more concise. It’s always a good idea to provide a clear and direct answer. The description ofthe power strip on the floor being near the table and chair wheels is clear, but it could be better to describe it as a power strip or power outlet located on thefloor near the table and chair. This makes it more specific and easier to locate. However, the core information is there, and the overall answer is good.</critique><score>10</score>The agent’s performance in correctly identifying a charging location within the room is commendable, and the response is concise and clear,fulfilling the user’s request. The slight improvement in wording is minor. Overall, a well-crafted response that meets the query’s requirements.

Score: 10