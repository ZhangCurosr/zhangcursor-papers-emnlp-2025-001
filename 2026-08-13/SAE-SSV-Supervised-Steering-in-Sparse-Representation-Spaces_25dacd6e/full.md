# SAE-SSV: Supervised Steering in Sparse Representation Spaces for Reliable Control of Language Models

Zirui He<sup>1</sup> Mingyu Jin<sup>2</sup> Bo Shen<sup>1</sup> Ali Payani<sup>3</sup> Yongfeng Zhang<sup>2</sup> Mengnan Du<sup>1</sup>\* <sup>1</sup>NJIT <sup>2</sup>Rutgers University <sup>3</sup>Cisco

## Abstract

Large language models (LLMs) have demonstrated impressive capabilities in natural language understanding and generation, but controlling their behavior reliably remains challenging, especially in open-ended generation settings. This paper introduces a novel supervised steering approach that operates in sparse, interpretable representation spaces. We employ sparse autoencoders (SAEs) to obtain sparse latent representations that aim to disentangle semantic attributesfrom model activations. Then we train linear classifiers to identify a small subspace of task-relevant dimensions in latent representations. Finally, we learn supervised steering vectors constrained to this subspace, optimized to align with target behaviors. Experiments across sentiment, truthfulness, and politics polarity steering tasks with multiple LLMs demonstrate that our supervised steering vectors achieve higher success rates with minimal degradation in generation quality compared to existing methods. Further analysis reveals that a notably small subspace is sufficient for effective steering, enabling more targeted and interpretable interventions. Our implementation is publicly available at https: //github.com/Ineedanamehere/SAE-SSV.

## 1 Introduction

Large language models (LLMs) have demonstrated impressive capabilities across a wide range of natural language understanding and generation tasks (Ouyang et al., 2022; Wei et al., 2022). Yet, as language models’ scale increases, achieving reliable and interpretable behavior control remains a fundamental challenge (Zhao et al., 2024; Sharkey et al., 2025). One promising approach for controllable generation is steering, which manipulates internal model representations during inference to influence behaviors without modifying model parameters by retraining or finetuning (Rimsky et al., 2024; Turner et al., 2024; Han et al., 2024).

Recent steering methods control LLM behavior by modifying internal activations at different points in the inference process: modifying residual stream activations (Zou et al., 2023), injecting latent directions learned from contrastive data (Kleindessner et al., 2023), and applying interpretable feature vectors extracted from sparse autoencoders or linear classifiers (Huben et al., 2024; Kantamneni et al., 2025). These steering techniques offer a lightweight and modular means of behavior control, and have been applied to enforce stylistic consistency (Wang, 2024), mitigate social biases, and align LLM outputs with safety or fairness objectives (Li et al., 2025). Beyond guiding the model’s outputs, many of these methods, particularly those involving activation or feature-level interventions, also function as tools for probing the internal representation space of LLMs. This dual role has positioned them at the intersection of behavior control and mechanistic interpretability (Zhao et al., 2025a; Ferrando et al., 2025). Nevertheless, most evaluations of steering methods have focused on constrained tasks with easily measurable outputs such as multiple-choice question answering or sentiment binary classification, where control success can be directly quantified (Zou et al., 2023; Im and Li, 2025). Other recent works have applied steering in agentic (Rahn et al., 2024) and refusalcontrol (Zhao et al., 2025b) settings. Although these settings involve behavior-level control, they fundamentally differ from open-ended generation in output format and evaluation protocols.

Unlike classification or structured QA, the openended generation setting requires LLMs to generate coherent and attribute-consistent text from scratch (Li et al., 2023b). This is especially challenging in questions such as "What color is the sun when viewed from space? Briefly explain the reason." The model must not only produce a factually correct response but also structure it fluently without predefined options. This setting is central to real-world applications such as dialogue systems, creative writing, and factual content generation, yet steering methods often struggle in this regime (Becker et al., 2024). Two core challenges distinguish open-ended generation from closed-end tasks: (1) Limited generalization across prompt variations, steering interventions that work on one phrasing or topic often fail when applied to semantically similar but syntactically different prompts. and (2) Generate quality degradation under strong control, intensifying the steering signal may improve direction alignment but often harms generation fluency, coherence, or factuality (Zhou et al., 2024). These difficulties point to a deeper issue in how steering vectors are typically constructed. Many existing approaches rely on global heuristics, such as mean difference vectors or unsupervised projections (Jorgensen et al., 2024; Chalnev et al., 2024). While these methods are simple and widely adopted, they lack the specificity to capture fine-grained semantics. Furthermore, they operate in dense, entangled activation spaces (Huben et al., 2024) and often fail to leverage supervision, leading to unstable or unintended behaviors under distributional shift.

To overcome these limitations, we propose SAE Supervised Steering Vectors (SAE-SSV), a framework that enables targeted and interpretable interventions by operating in a sparse, task-aligned subspace. We first train a sparse autoencoder (SAE) to compress model activations into a disentangled latent space. Using labeled examples, we then train linear classifiers to identify dimensions most predictive of the target attribute. Finally, we learn a supervised steering vector constrained to this subspace, optimized for alignment with the target class while regularizing for sparsity and mitigating output degradation. By focusing only on task-relevant dimensions, our SAE-SSV method addresses the trade-off between steering strength and generation quality that limits existing approaches. Our contribution can be summarized as follows:

• We propose SAE-SSV, a supervised steering framework that constrains interventions to a sparse, task-relevant latent subspace identified via labeled data and sparse autoencoders.

• Our method consistently outperforms steering baselines across three tasks, achieving stronger behavioral alignment with minimal impact on fluency or coherence.

• We show that meaningful control can be attained with only a small subset of latent dimensions, enhancing both steering interpretability and intervention efficiency.

## 2 Preliminaries

## 2.1 Latent Steering in Language Models

Steering is a technique for controlling the output of LLMs via small interventions in their internal representations. Let $x$ be an input sequence in LLM and $h ( x ) \in \mathbb { R } ^ { d }$ denote the activation of x at a chosen layer $( \mathrm { e . g . }$ ., the residual stream). Steering can modify $h ( x )$ with an additive perturbation $v \in$ $\mathbb { R } ^ { d } \left( \lambda \in \mathbb { R } \right.$ is a scaling coefficient) like Equation 1:

$$
h ^ { \prime } ( x ) = h ( x ) + \lambda v ,\tag{1}
$$

The modified representation $h ^ { \prime } ( x )$ is then fed into the subsequent layers of the language model, thereby influencing the final output generation.

Prior work proposed various ways to construct the additive perturbation vector $v ,$ including mean difference vectors between contrasting classes (Dathathri et al., 2020), and PCA (Kleindessner et al., 2023). These approaches aim to identify semantically meaningful directions in the latent space, such that steering along these directions enables controlled manipulation of the LLM’s behavior during generation.

## 2.2 SAEs for Representation Analysis

To enable structured and interpretable analysis of internal model representations, Sparse autoencoders (SAEs) have been introduced to transform dense activations into sparse latent codes (Huben et al., 2024). An SAE consists of an encoder $f _ { \mathrm { e n c } }$ and decoder $f _ { \mathrm { d e c } } ,$ , trained to minimize Equation 3:

$$
z = f _ { \mathrm { e n c } } ( h ) , \quad \hat { h } = f _ { \mathrm { d e c } } ( z ) ,\tag{2}
$$

$$
\mathcal { L } _ { \mathrm { S A E } } = \| h - \hat { h } \| _ { 2 } ^ { 2 } + \beta \| z \| _ { 1 } ,\tag{3}
$$

where $h \in \mathbb { R } ^ { m }$ is the original activation vector of input, $z \in \mathbb { R } ^ { d _ { \mathrm { s a e } } }$ is the sparse space, where typically $d _ { \mathrm { s a e } } \gg m$ to allow for disentangled features. $\beta$ controls the sparsity, the $\ell _ { 1 }$ penalty encourages each input to activate only a small number of latent dimensions, facilitating interpretability and localization of concepts (Bricken et al., 2023).

## 3 SAE-SSV Framework

Our objective is to reliably steer the LLM’s output toward specific behavioral targets, such as producing text with a particular emotion. To achieve this, we propose the SAE-SSV framework (see Figure 1). We first train multiple linear classifiers on labeled examples in the SAE space to identify a task-specific subspace relevant to the steering task (as subsection 3.1). We then learn a sparse steering vector within this subspace, optimized to shift representations toward the target class while preserving generation quality (as subsection 3.2).

![](images/4226ed73ac6997a54f5ab3cbca5c8eda9f7e7a5fe5490721ba2083075c171208.jpg)  
Figure 1: Overview of the SAE-SSV framework. It encodes model activations into a sparse latent space, selects task-relevant dimensions via linear probes, and optimizes steering vectors with combined losses to ensure effective control while maintaining generation quality.

## 3.1 Dimension Selection via Probing

Coarse-Grained Feature Selection. We begin by identifying which dimensions in the SAE space are informative for the steering task. Given a labeled dataset $D = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ , where $y _ { i } \in \{ 0 , 1 \}$ denotes a binary attribute (e.g., negative vs. positive sentiment), we process each input $x _ { i }$ through a frozen pretrained LLM and extract residual stream activations $h _ { i }$ at a target layer as described in subsection 2.1.

These activations are passed through a pretrained SAE encoder $f _ { \mathrm { e n c } }$ to obtain sparse latent representations $z _ { i } = f _ { \mathrm { e n c } } ( h _ { i } )$ . To identify task-relevant features, we compute the F-statistic (Jain and Zongker, 2002) for each latent dimension t:

$$
S _ { t } = { \frac { \mathrm { B e t w e e n – g r o u p ~ v a r i a n c e } } { \mathrm { W i t h i n – g r o u p ~ v a r i a n c e } } } ,\tag{4}
$$

where the numerator quantifies how distinct the class means are and the denominator captures within-class dispersion. We rank all dimensions by $S _ { t }$ and select the top-k to form the steering subspace $I \subset [ 1 , d _ { \mathrm { s a e } } ]$ , where $d _ { \mathrm { s a e } }$ is the dimensionality of the full SAE space. Representations restricted to I are standardized and used to train a linear classifier to distinguish between the two classes. The classifier is optimized using the standard cross-entropy loss:

$$
\mathcal { L } _ { \mathrm { c l f } } = \mathbb { E } _ { ( z , y ) \sim D } \left[ - \log \frac { \exp ( w _ { y } ^ { \top } z ) } { \sum _ { y ^ { \prime } } \exp ( w _ { y ^ { \prime } } ^ { \top } z ) } \right]\tag{5}
$$

where $w _ { y }$ denotes the weight vector for class $y .$ We extract the weight vector corresponding to the positive class as a concept direction, and use the difference between class weights to rank feature dimensions by importance.

Fine-Grained Feature Selection. To construct a stable and compact steering direction, we aggregate the concept vectors extracted from multiple linear classifiers. Specifically, we train M classifiers on independently sampled subsets of the data, using only the k dimensions selected in the previous step. From each classifier, we extract the weight vector associated with the positive class label, denoted $w _ { 1 } ^ { ( j ) }$ for the j-th classifier. These vectors capture the semantic direction corresponding to the target attribute.We compute the average of these vectors to obtain a unified direction:

$$
v _ { \mathrm { a v g } } = \frac { 1 } { M } \sum _ { j = 1 } ^ { M } w _ { 1 } ^ { ( j ) } .\tag{6}
$$

This averaged vector serves as a representative semantic direction that consolidates information across multiple probing classifiers.

To further reduce dimensionality, we sort the coordinates of $v _ { \mathrm { a v g } }$ by absolute magnitude and construct truncated vectors $\boldsymbol { v } ^ { ( d ) }$ by retaining only the top-d components and zeroing out the rest. For each d, we project test samples onto $v ^ { ( d ) }$ and compute their cosine similarity with the direction. Let $\bar { c } _ { 1 }$ and $\bar { c } _ { 0 }$ denote the average cosine similarity for positive and negative examples, respectively. We define the separation score as

$$
\begin{array} { r } { s ^ { ( d ) } = \bar { c } _ { 1 } - \bar { c } _ { 0 } . } \end{array}\tag{7}
$$

We select the smallest d that maximizes $s ^ { ( d ) }$ and denote it as $d _ { \mathrm { s t e e r } }$ , which represents the final number of active dimensions used for steering.

## 3.2 Supervised Steering Vector Optimization

We construct and optimize a steering vector v $\in$ $\mathbb { R } ^ { d _ { \mathrm { s a e } } }$ that is constrained to be nonzero only in the $d _ { \mathrm { s t e e r } }$ most informative dimensions, as identified in Section 3.1. All remaining coordinates of v are fixed to zero, leaving only $d _ { \mathrm { s t e e 1 } }$ <sub>r</sub> nonzero entries corresponding to the selected dimensions.

We initialize v using the difference between class centroids in the SAE space:

$$
v _ { \mathrm { i n i t } } = \mu ^ { + } - \mu ^ { - } ,\tag{8}
$$

where $\mu ^ { + }$ and $\mu ^ { - }$ denote the average SAE representations of positive and negative examples, respectively. We then zero out all components of $v _ { \mathrm { i n i t } }$ outside I, retain the $\mathrm { t o p } { - } d _ { \mathrm { s t e e r } }$ coordinates by magnitude, and normalize the resulting vector.

To optimize $v ,$ we construct training pairs $( x ^ { + } , x ^ { - } )$ of positive and negative examples. For each negative input $x ^ { - }$ , we extract its SAE latent representation $z = f _ { \mathrm { e n c } } ( h ( x ^ { - } ) )$ , apply the steering vector to obtain $z ^ { \prime } = z + v$ , decode $z ^ { \prime }$ back to the residual stream via $\hat { h } = f _ { \mathrm { d e c } } ( z ^ { \prime } )$ , and reinsert it into the LLM to generate steered output.

The steering vector is optimized to satisfy three objectives: (1) align $z ^ { \prime }$ with the positive class center while pushing it away from the negative center, (2) preserve the fluency and coherence of the generated text, and (3) maintain sparsity over the active dimensions. The total loss is given by:

$$
\begin{array} { r l } & { L _ { \mathrm { s t e e r } } = \| z ^ { \prime } - \mu ^ { + } \| _ { 2 } ^ { 2 } - \| z ^ { \prime } - \mu ^ { - } \| _ { 2 } ^ { 2 } } \\ & { ~ + L _ { \mathrm { L M } } + \beta \| v _ { I } \| _ { 1 } , } \end{array}\tag{9}
$$

where $L _ { \mathrm { L M } }$ is a language modeling loss that penalizes degraded generation quality by computing the cross-entropy of the positive target sequence $x ^ { + }$ conditioned on the steered hidden state of the negative input $x ^ { - }$ −, and $\| v _ { I } \| _ { 1 }$ encourages sparsity within the steering subspace.

## 4 Experiments

In this section, we evaluate the effectiveness of SAE-SSV by answering the following research questions (RQs):

• RQ1: How is the performance of SAE-SSV compared to baselines? (Section 4.2)

• RQ2: Can we identify a minimal and interpretable subspace within the SAE latent space that is sufficient for steering model behavior? (Section 4.3)

• RQ3: Can steering in a structured subspace improve attribute alignment while minimizing output degradation? (Section 4.4)

• RQ4: Can SAE-SSV generalize across datasets within the same task domain? (Section 4.5)

## 4.1 Experimental Setup

Models. We conduct experiments on three open-source base models: Gemma-2-2b, Gemma-2-9b (Team et al., 2024), and LLaMA3.1- 8B (Grattafiori et al., 2024). For sparse autoencoders, we use pre-trained SAEs from the Gemma Scope (Lieberum et al., 2024) and LLaMA Scope (He et al., 2024) repositories to extract semantic subspaces for steering.

Datasets. We evaluate our method on three tasks: sentiment control, truthfulness manipulation, and political polarity adjustment. The truthfulness and political polarity datasets are adopted from (Fulay et al., 2024), namely the TruthGen dataset of paired factual and counterfactual statements, and the TwinViews-13k dataset of ideologically matched political pairs. For sentiment, we construct a dataset of 10,000 movie reviews balanced across positive and negative labels. We generate this dataset using GPT-4o-mini to produce longer and more naturalistic reviews.

Baseline Methods. We compare our SAE-SSV method against four widely used steering baselines:

• Concept Activation Addition (CAA) (Rimsky et al., 2024): Adds the mean activation difference between positive and negative examples during inference to steer model outputs.

• Representation Perturbation (RePe) (Zou et al., 2023): Perturbs activations along principal components of class-conditional differences.

• Top PC (Im and Li, 2025): Projects activations onto the first principal component of the embedding space, capturing the direction of maximal variance.

![](images/027899987665f2f0b89ec1b601d86fc2fc07cabad07b761235cda3bc3d564a61.jpg)

Table 1: Comparison of Steering Methods Across All Models and Tasks (Sentiment, Politics Polarity, Truthfulness)
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="3">Sentiment</td><td colspan="3">Politics Polarity</td><td colspan="3">Truthfulness</td></tr><tr><td>SR (%)↑ ∆MTLD↑</td><td></td><td>∆Entropy↑</td><td></td><td></td><td></td><td></td><td></td><td>SR (%)↑ ∆MTLD↑ ∆Entropy↑ SR (%)↑ ∆MTLD↑ ∆Entropy↑</td></tr><tr><td rowspan="5"></td><td>CAA (Rimsky et al., 2024)</td><td>45.6</td><td>-0.35</td><td>-0.19</td><td>43.7</td><td>-0.27</td><td>-0.16</td><td>28.7</td><td>-0.72</td><td>-1.10</td></tr><tr><td>RePe (Zou et al., 2023)</td><td>24.7</td><td>-0.21</td><td>-0.13</td><td>26.2</td><td>-0.22</td><td>-0.15</td><td>16.2</td><td>-0.57</td><td>-0.47</td></tr><tr><td>Llama3.1-8B Top PC (Im and Li, 2025)</td><td>28.4</td><td>-0.25</td><td>-0.14</td><td>24.3</td><td>-0.18</td><td>-0.11</td><td>14.9</td><td>-0.61</td><td>-0.66</td></tr><tr><td>ITI (Li et al., 2024)</td><td>41.1</td><td>-0.31</td><td>-0.27</td><td>45.2</td><td>-0.34</td><td>-0.29</td><td>31.2</td><td>-0.81</td><td>-0.89</td></tr><tr><td>SAE-SSV (Ours)</td><td>63.2</td><td>0.09</td><td>-0.07</td><td>60.5</td><td>0.11</td><td>-0.04</td><td>34.1</td><td>-0.31</td><td>-0.24</td></tr><tr><td rowspan="5">Gemma2-2B</td><td>CAA (Rimsky et al., 2024)</td><td>39.6</td><td>-0.32</td><td>-0.28</td><td>45.4</td><td>-0.38</td><td>-0.33</td><td>24.6</td><td>-0.71</td><td>-1.05</td></tr><tr><td>RePe (Zou et al., 2023)</td><td>27.2</td><td>-0.24</td><td>-0.20</td><td>36.6</td><td>-0.26</td><td>-0.21</td><td>11.6</td><td>-0.42</td><td>-0.37</td></tr><tr><td>Top PC (Im and Li, 2025)</td><td>23.8</td><td>-0.17</td><td>-0.09</td><td>35.0</td><td>-0.22</td><td>-0.17</td><td>12.2</td><td>-0.46</td><td>-0.40</td></tr><tr><td>ITI (Li et al., 2024)</td><td>41.2</td><td>-0.30</td><td>-0.27</td><td>42.1</td><td>-0.33</td><td>-0.32</td><td>22.3</td><td>-0.74</td><td>-1.12</td></tr><tr><td>SAE-SSV (Ours)</td><td>52.8</td><td>0.08</td><td>-0.08</td><td>61.3</td><td>0.10</td><td>-0.04</td><td>31.7</td><td>-0.37</td><td>-0.23</td></tr><tr><td rowspan="5"></td><td>CAA (Rimsky et al., 2024)</td><td>42.3</td><td>-0.42</td><td>-0.37</td><td>39.3</td><td>-0.27</td><td>-0.22</td><td>19.8</td><td>-0.75</td><td>-1.10</td></tr><tr><td>RePe (Zou et al., 2023)</td><td>19.7</td><td>-0.27</td><td>-0.22</td><td>22.4</td><td>-0.16</td><td>-0.19</td><td>9.2</td><td>-0.51</td><td>-0.48</td></tr><tr><td>Gemma2-9B Top PC (Im and Li, 2025)</td><td>21.4</td><td>-0.31</td><td>-0.25</td><td>29.1</td><td>-0.21</td><td>-0.18</td><td>10.6</td><td>-0.66</td><td>-0.70</td></tr><tr><td>ITI (Li et al., 2024)</td><td>41.2</td><td>-0.33</td><td>-0.29</td><td>33.8</td><td>-0.31</td><td>-0.27</td><td>21.4</td><td>-0.70</td><td>-0.97</td></tr><tr><td>SAE-SSV (Ours)</td><td>48.5</td><td>0.09</td><td>-0.11</td><td>55.0</td><td>0.07</td><td>-0.12</td><td>27.2</td><td>-0.39</td><td>-0.26</td></tr></table>

![](images/6bf9d8e38264a222ce1f0240cee7fb0217d872d5f9f4e195a500c79c37e28367.jpg)  
(a) Sentiment Task (LLaMA3.1-8b, Layer 16)  
(b) Truthfulness Task (LLaMA3.1-8b, Layer 16)  
Figure 2: Activation heatmaps of the top-30 dimensions for each task. (a) Sentiment task. (b) Truthfulness task. Each panel compares class-wise activation patterns in the raw residual space and SAE space.

• Inference-Time Intervention (ITI) (Li et al., 2024): Shifts attention head activations during inference along truth-related directions found via linear probing.

Evaluation Metrics. We employ metrics to evaluate steering effectiveness and generation quality:

• Steering Success Rate (SR): The percentage of generated outputs that successfully exhibit the target attribute. We use GPT-4o-mini as an automatic judge to assess whether the generated text reflects the intended attribute. Formally, $\begin{array} { r } { \mathbf { S } \mathbf { R } = \frac { N _ { \mathrm { s u c c e s s } } } { N _ { \mathrm { t o t a l } } } \times 1 0 0 \% } \end{array}$ , where $N _ { \mathrm { s u c c e s s } }$ is the number of generations judged as exhibiting the target attribute. Specific prompting details for the judge are provided in Appendix D.

• Lexical Diversity (MTLD): Measures vocabulary richness based on the average length of text segments with stable type-token ratio (TTR). We report $\Delta \mathbf { M T L D }$ relative to unsteered outputs to assess changes in lexical diversity.

• Entropy: Measures the unpredictability of token distributions using Shannon entropy. Lower values indicate higher repetition. We report ∆Entropy relative to unsteered outputs. Formally, $\begin{array} { r } { H = - \sum _ { i } p ( x _ { i } ) \log p ( x _ { i } ) } \end{array}$ , where $p ( x _ { i } )$ is the probability of token $x _ { i } .$

Implementation Details. We report main results using 16K-dimensional SAE models for both Gemma-2-2b and Gemma-2-9b models, and a 32Kdimensional SAE for LLaMA3.1-8b model. Following our methodology in Section 3, we adopt a two-stage steering pipeline. In Stage 1, we train $M = 5 0$ linear probes per task to ensure stability in feature selection. We set the number of selected SAE dimensions to $k = 1 2 8$ to ensure sufficient subspace coverage for semantic manipulation. In Stage 2, the steering vector is optimized using a contrastive objective that combines distance loss, language modeling loss, and $L _ { 1 }$ regularization, with coefficients $\lambda _ { \mathrm { d i s t } } = 1 . 0 , \lambda _ { \mathrm { l m } } = 0 . 5$ , and $\lambda _ { \mathrm { r e g } } ~ = ~ 0 . 0 1$ , respectively. Optimization is performed for 100 iterations with a learning rate of 0.05 and a batch size of 64. During inference, we apply the steering vector at each decoding step with scaling factors ranging from 1.0 to 10.0 to explore the trade-off between steering strength and output quality. For each model and task, we apply interventions at empirically selected layers: LLaMA3.1- 8B (layer 16 for all tasks), Gemma2-2B (sentiment: 13, truthfulness: 16, politics: 15), and Gemma2-9B (sentiment: 20, truthfulness: 26, politics: 20). All experiments are conducted on a single NVIDIA A100 GPU.

![](images/3db04f8ee04451b388f8c75fcee6d75941b89dbf938777e22be2f2b5203049f0.jpg)  
(a) Feature Selection Stability

![](images/366f09a7c05b4572ecdd8965af6862800d68eb056fcae4b945c96d3226193314.jpg)  
(b) Separability vs. Dimension Count  
Figure 3: (a) shows how the number of linear classifiers affects feature selection stability. (b) shows that a small number of top SAE dimensions enable clear class separation.

## 4.2 Comparison with Baseline Methods

For each task, we steer in a fixed semantic direction: from negative to positive sentiment, from left-leaning to right-leaning political views, and from factual to hallucinated content. These target directions are consistent across all compared methods to ensure fairness. Table 1 presents steering comparison across the three tasks. We have the following observations.

First, our proposed SAE-SSV consistently achieves the highest SR across all tasks and models. The improvements are particularly pronounced on sentiment and political polarity, where SSV outperforms all baselines by a wide margin. Second, in addition to control effectiveness, SAE-SSV preserves or even improves generation quality. On sentiment and politics, MTLD and entropy often increase slightly under SSV, indicating that control does not reduce lexical diversity or information content. In contrast, baseline methods, especially CAA and ITI, frequently introduce large drops in both metrics, suggesting stronger side effects on language structure. Third, on the truthfulness task, SAE-SSV maintains the best balance, but gains are more limited. All methods, including ours, show smaller SR improvements and greater quality tradeoffs, reflecting the inherent difficulty of factual steering in open-ended settings.

## 4.3 Identifying a Minimal Steering Subspace

We investigate whether the model’s internal representations contain a sparse and semantically aligned subspace that supports effective steering.

Subspace Concept Separability Analysis. Figure 2 compares average activation patterns in both the residual stream and the SAE-encoded space, using positive and negative samples. We visualize the top 20 most active dimensions in each space. We have two key observations:

• In the residual space, activations are distributed without clear class-specific structure. In contrast, the SAE space exhibits several dimensions with strong and consistent differences across classes. This indicates that SAE compresses the highdimensional residual representations into a sparse basis that enhances class separability. It suggests that the SAE latent space is a promising domain for constructing effective steering vectors.

• The SAE heatmaps also reveal task-specific characteristics. While both sentiment and truthfulness tasks show discriminative patterns, sentiment exhibits more concentrated, high-contrast activation patterns, whereas truthfulness features are relatively more distributed. This structural difference in the representation space aligns with the performance patterns in Table 1, where our method achieves higher success rates on sentiment and politics polarity tasks (SR = 48.5- 63.2%) compared to truthfulness (SR = 27.2- 34.1%). Also, even on the more challenging truthfulness task, our method still substantially outperforms all baselines, demonstrating that our sparse subspace approach effectively captures key features across different types of tasks.

Table 2: Top-10 SAE features used in the SSV for the sentiment task on LLaMA-3.1-8B. Feature explanations are retrieved from Neuronpedia (Lin, 2023), and the value column indicates the weights learned during SSV training.
<table><tr><td>Rank</td><td>Explanation of Feature</td><td>SAE Feature #</td><td>Value</td></tr><tr><td>1</td><td>Negative sentiments towards characters in movies</td><td>2305</td><td>-6.76</td></tr><tr><td>2</td><td>Negative sentiments and criticisms related to performance or quality</td><td>14086</td><td>-3.24</td></tr><tr><td>3</td><td>Phrases related to notable achievements in the entertainment industry</td><td>12322</td><td>2.79</td></tr><tr><td>4</td><td>Punctuation and symbols indicative of structural elements in text</td><td>20767</td><td>2.32</td></tr><tr><td>5</td><td>Mentions of achievements and recognition in a professional context</td><td>28857</td><td>1.99</td></tr><tr><td>6</td><td>Phrases and terminology related to legal injunctions and restrictions</td><td>2268</td><td>-1.89</td></tr><tr><td>7</td><td>Components related to specific abilities or skills in performance</td><td>29039</td><td>-1.86</td></tr><tr><td>8</td><td>References to historical or legendary figures and events</td><td>28858</td><td>-1.68</td></tr><tr><td>9</td><td>Phrases indicating misinformation, contradictions, or inaccuracies</td><td>14391</td><td>-1.46</td></tr><tr><td>10</td><td>Questions and expressions of disappointment or concern</td><td>13758</td><td>-1.43</td></tr></table>

![](images/85268a0baf4bcb9294919c2a57750b2064ef600e0e905af43d3c7445f07f2ea1.jpg)  
Figure 4: Average projection values of token activations along four directions: no steering (gray), SAE-SSV (blue), orthogonal (green), and random (orange). Computed over successfully steered samples. SAE-SSV induces a consistent and sustained directional shift, while other directions show minimal change.

Feature Selection Stability Analysis. We vary the number of linear classifiers M used to rank important SAE dimensions. Each classifier is trained on a random subset of labeled data, and we compute the average importance scores across all M runs. Figure 3a demonstrates that despite variations in their relative rankings, the set of top-128 dimensions selected from the 16K-dimensional SAE space remains perfectly consistent across different ensemble sizes (M = 1 to M = 50). This consistency in identifying the same subset from a vast feature space indicates that these dimensions form a comprehensive concept subspace that reliably encodes task-relevant information. The coefficient of variation of feature importance scores decreases as M increases, providing more stable estimates of each dimension’s relative contribution.

## Selected Dimension Discriminability Analysis.

In Figure 3b, we incrementally select the topk ranked dimensions from our identified 128- dimensional subspace and measure class separability by calculating the difference between mean projection scores of positive and negative samples. The results demonstrate that even a small number of the highest-ranked dimensions achieves substantial class separation, with diminishing returns as more dimensions are added. This suggests that within our already focused 128-dimensional concept space, an even smaller subset of dimensions carries the most significant task-relevant information. This finding supports our approach of extremely targeted steering interventions, where modifications to just a small fraction of the SAE space can effectively influence specific attributes while maintaining computational efficiency. We provide the sets of SAE features used for constructing SSVs in Table 2 and more analysis in Appendix B.

## 4.4 Mitigating Output Degradation

We evaluate whether SAE-SSV can achieve strong steering while minimizing generation quality degradation, a common side effect of intervention.

Measuring Output Degradation Quality. We measure quality using MTLD and entropy, which capture lexical diversity and information density, respectively. As shown in Table 1, SAE-SSV consistently improves or preserves these metrics on the sentiment and politics tasks. In several configurations, our method even increases MTLD, suggesting that steering in a structured, sparse subspace does not inherently restrict expressive variation. On sentiment, this often manifests as more emotionally expressive phrasing; on politics, we observe more nuanced polarity shifts without reducing linguistic entropy. Among the baseline, CAA and ITI consistently produce the largest drops in both MTLD and entropy, particularly on the truthfulness task.

Why SAE-SSV can Preserve Quality? To better understand this question, we visualize the tokenwise projection of hidden activations along different directions. Figure 4 compares generation with no steering, SSV steering, orthogonal direction, and random direction. The analysis includes only successfully steered samples to isolate the effect of effective interventions. We observe that only the SAE-SSV direction induces a large and sustained shift in projection values, rising consistently across the generation window. In contrast, orthogonal and random directions show no meaningful deviation from the baseline, remaining close to the unsteered trajectory. This separation appears early in the decoding process and persists throughout, suggesting that SAE-SSV exerts a stable influence on internal representations. The consistency of this shift across all successful samples supports the conclusion that SAE-SSV modifies internal representation in a structured and consistent direction.

Table 3: Generalization performance of SAE-SSV on unseen datasets using LLaMA3.1-8B. SR = steering success rate. Ret. = retained original attribute. Dis. = incoherent, repetitive, contradictory or task irrelevant output. All values are percentages.
<table><tr><td>Method</td><td>SR (%) ↑</td><td>Ret. (%) ↓</td><td>Dis. (%) ↓</td></tr><tr><td colspan="4">Rotten Tomatoes</td></tr><tr><td>Baseline</td><td>20.2</td><td>63.1</td><td>16.7</td></tr><tr><td>SAE-SSV</td><td>37.8</td><td>33.5</td><td>28.7</td></tr><tr><td colspan="4">TruthfulQA</td></tr><tr><td>Baseline</td><td>32.4</td><td>57.8</td><td>9.8</td></tr><tr><td>SAE-SSV</td><td>48.9</td><td>9.8</td><td>41.3</td></tr></table>

## 4.5 Generalizing SAE-SSV Across Tasks

To evaluate the generalization capacity of our proposed SAE-SSV method, we apply steering vectors originally trained on one dataset to a different test set within the same task domain, without any retraining or supervision on the target samples.

Experimental Setting. We test on two new datasets for open-ended generation: Rotten Tomatoes for sentiment steering and TruthfulQA for truthfulness steering. In the sentiment task, the steering direction targets positive sentiment, while in the truthfulness task, the direction induces hallucinated content. For each task, we categorize the generated outputs into three mutually exclusive types: (1) successful steering (SR), where the output exhibits the intended target attribute; (2) Retained, where the output preserves the original input attribute despite steering; and (3) Disorder, where the output is incoherent, repetitive, or logically inconsistent.

Table 4: Ablation results for sentiment steering with LLaMA3.1-8B. We compare the full SAE-SSV with two ablated variants and the baseline.Evaluation metrics are identical to those in Table 3.
<table><tr><td>Method</td><td>SR (%) ↑</td><td>Ret. (%) ↓</td><td>Dis. (%) ↓</td></tr><tr><td>Baseline</td><td>12.3</td><td>79.2</td><td>8.5</td></tr><tr><td>SSV w/o train</td><td>13.7</td><td>73.9</td><td>12.4</td></tr><tr><td>SSV w/o LM loss</td><td>28.6</td><td>28.1</td><td>43.3</td></tr><tr><td>SSV</td><td>63.2</td><td>23.5</td><td>13.3</td></tr></table>

Result Analysis. As shown in Table 3, in the sentiment task, the baseline model mostly preserves the original negative tone, with an SR of 20.2%. Applying the SAE-SSV vector raises SR to 37.8%, demonstrating effective transfer of the emotional control signal. The Retained rate drops from 63.1% to 33.5%, suggesting that most outputs have been influenced by the steering. However, this comes with a trade-off, as the Disorder rate rises to 28.7%, indicating more outputs falling into unusable forms. On the truthfulness task, the baseline SR is 32.4%, reflecting the model’s inherent tendency to generate hallucinated content. With SAE-SSV steering, SR increases to 48.9%, and Retained drops sharply to 9.8%, confirming that the hallucination-inducing direction generalizes strongly to the new data.

## 4.6 Ablation Study

Table 4 examines the impact of two key components in our method: the supervised training of the steering vector and the inclusion of the LM loss. Removing either component leads to a clear drop in steering success. Notably, omitting the LM loss increases SR to 28.6%, but also causes a substantial rise in output disorder (43.3%), indicating unstable model behavior. In contrast, the full SAE-SSV achieves the highest SR (63.2%) while maintaining low disorder (13.3%), demonstrating the importance of subspace-constrained, supervised optimization. In addition, we study the effect of the scaling factor λ used during inference. We observe that the steering strength measured qualitatively by semantic shift is approximately linear with respect to λ. However, developing a precise quantitative metric for steering intensity remains challenging. We provide representative examples illustrating this relationship in Appendix C.

## 5 Related Work

Language Model Representations. Studies of language model representations have established that many concepts exist as linear directions in activation space (Kim et al., 2018; Jin et al., 2025a). These concept vectors can be derived through various methods, including probing classifiers (Belinkov, 2022; Jin et al., 2025b), mean difference calculations (Rimsky et al., 2024; Zou et al., 2023), mean centering (Jorgensen et al., 2024), and Gaussian concept subspaces (Zhao et al., 2025a). These approaches have successfully identified directions corresponding to high-level concepts such as honesty (Li et al., 2024), truthfulness (Tigges et al., 2023), harmfulness (Zou et al., 2023), and sentiment (Zhao et al., 2025a). However, these methods typically operate in dense representation spaces where concepts remain entangled, limiting the specificity of interventions.

Activation Steering. Activation steering has emerged as a powerful technique for influencing model behavior during inference without retraining. Early work such as Plug and Play Language Models (Dathathri et al., 2020) and representation engineering (Zou et al., 2023) established the feasibility of direct activation manipulation. Subsequent research demonstrated its effectiveness in improving truthfulness (Marks and Tegmark, 2024; Tigges et al., 2023), enhancing safety (Arditi et al., 2024; Li et al., 2024), mitigating biases (Jorgensen et al., 2024), and controlling style (Wang, 2024). More recent methods include CAA (Rimsky et al., 2024), which uses contrastive activation addition, RePe (Kleindessner et al., 2023), which employs PCA-derived directions, and ITI (Li et al., 2024), which iteratively trains steering vectors. Nevertheless, steering often faces a trade-off between control strength and generation quality in open-ended settings (Zhou et al., 2024), in part because interventions in dense spaces can inadvertently entangle multiple concepts (Huben et al., 2024). Our work addresses this challenge by leveraging disentangled SAE features and supervised dimension selection to constrain steering to a task-specific subspace, enabling more targeted interventions with fewer side effects.

Sparse Autoencoders. Sparse autoencoders (SAEs) have been introduced to disentangle superimposed features through dictionary learning. By mapping activations into a higher-dimensional sparse space, SAEs yield more interpretable features (Bricken et al., 2023; Huben et al., 2024). Variants include vanilla SAEs (Sharkey et al., 2022) and TopK SAEs (Gao et al., 2024), with pre-trained repositories such as Gemma Scope (Lieberum et al., 2024) and Llama Scope (He et al., 2024) enabling broader research. SAEs have been used to interpret model representations (Kissane et al., 2024), to understand model capabilities (Ferrando et al., 2025), and to explore intersections with steering (Chalnev et al., 2024; He et al., 2025). Applications include toxicity mitigation (Gallifant et al., 2025) and safety alignment (Wu et al., 2025a), but the use of SAEs for controllable generation remains relatively limited. Our work extends this line by combining SAE-derived features with supervised optimization to construct effective steering vectors.

## 6 Conclusions and Future Work

In this paper, we introduced SAE-SSV, a framework that enables effective LLM steering by operating in sparse, task-specific subspaces. The key insight lies in constraining interventions to a small number of interpretable dimensions that capture task-relevant semantics, enabling more targeted control while preserving generation quality. Experiments across sentiment, truthfulness, and political polarity steering tasks with multiple LLMs demonstrate that SAE-SSV consistently outperforms existing methods by a substantial margin. Our cross dataset experiments reveal that SAE-SSV captures both semantic directions and stylistic patterns of training data, highlighting its potential as a more general steering mechanism. For our future work, we aim to achieve universal and style-invariant SSVs that generalize across datasets, tasks, and model families by curating diverse training corpora and developing objectives that explicitly encourage semantic steering while minimizing sensitivity to stylistic variation.

## Limitations

Our SAE-SSV approach has several limitations. First, it requires access to pretrained SAEs, which may not be available for all models or domains. Currently, we only evaluate using the Gemma and Llama model families. Second, we evaluate LLMs with parameters at most of 9B. In future work, we plan to evaluate on larger LLMs with tens or hundreds of billions of parameters to better understand how our method scales with model size and complexity. Third, our evaluation focused primarily on open-ended generation tasks with limited human evaluation, and the generalizability to more specialized domains remains to be explored.

## Acknowledgments

Mengnan Du is supported by National Science Foundation (NSF) Grant #2310261. The views and conclusions in this paper are those of the authors and should not be interpreted as representing any funding agencies.

## References

Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda. 2024. Refusal in language models is mediated by a single direction. arXiv preprint arXiv:2406.11717.

Jonas Becker, Jan Philip Wahle, Bela Gipp, and Terry Ruas. 2024. Text generation: A systematic literature review of tasks, evaluation, and challenges. Preprint, arXiv:2405.15604.

Yonatan Belinkov. 2022. Probing classifiers: Promises, shortcomings, and advances. Computational Linguistics, 48(1):207–219.

Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, and 6 others. 2023. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread. Https://transformercircuits.pub/2023/monosemanticfeatures/index.html.

Sviatoslav Chalnev, Matthew Siu, and Arthur Conmy. 2024. Improving steering vectors by targeting sparse autoencoder features. arXiv preprint arXiv:2411.02193.

Sumanth Dathathri, Andrea Madotto, Janice Lan, Jane Hung, Eric Frank, Piero Molino, Jason Yosinski, and Rosanne Liu. 2020. Plug and play language models: A simple approach to controlled text generation. In International Conference on Learning Representations (ICLR).

Javier Ferrando, Oscar Balcells Obeso, Senthooran Rajamanoharan, and Neel Nanda. 2025. Do i know this entity? knowledge awareness and hallucinations in language models. In The Thirteenth International Conference on Learning Representations.

Suyash Fulay, William Brannon, Shrestha Mohanty, Cassandra Overney, Elinor Poole-Dayan, Deb Roy, and Jad Kabbara. 2024. On the relationship between truth and political bias in language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 9004–9018.

Jack Gallifant, Shan Chen, Kuleen Sasse, Hugo Aerts, Thomas Hartvigsen, and Danielle S Bitterman. 2025. Sparse autoencoder features for classifications and transferability. arXiv preprint arXiv:2502.11367.

Leo Gao, Tom Dupré la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, and Jeffrey Wu. 2024. Scaling and evaluating sparse autoencoders. arXiv preprint arXiv:2406.04093.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Chi Han, Jialiang Xu, Manling Li, Yi Fung, Chenkai Sun, Nan Jiang, Tarek Abdelzaher, and Heng Ji. 2024. Word embeddings are steers for language models. In Proceedings ofthe 62nd Annual Meeting ofthe Association for Computational Linguistics (ACL Outstanding Paper), pages 16410–16430.

Zhengfu He, Wentao Shu, Xuyang Ge, Lingjie Chen, Junxuan Wang, Yunhua Zhou, Frances Liu, Qipeng Guo, Xuanjing Huang, Zuxuan Wu, and 1 others. 2024. Llama scope: Extracting millions of features from llama-3.1-8b with sparse autoencoders. arXiv preprint arXiv:2410.20526.

Zirui He, Haiyan Zhao, Yiran Qiao, Fan Yang, Ali Payani, Jing Ma, and Mengnan Du. 2025. Saif: A sparse autoencoder framework for interpreting and steering instruction following of language models. arXiv preprint arXiv:2502.11356.

Robert Huben, Hoagy Cunningham, Logan Riggs Smith, Aidan Ewart, and Lee Sharkey. 2024. Sparse autoencoders find highly interpretable features in language models. In The Twelfth International Conference on Learning Representations.

Shawn Im and Yixuan Li. 2025. A unified understanding and evaluation of steering methods. arXiv preprint arXiv:2502.02716.

Anil Jain and Douglas Zongker. 2002. Feature selection: Evaluation, application, and small sample performance. IEEE transactions on pattern analysis and machine intelligence, 19(2):153–158.

Mingyu Jin, Kai Mei, Wujiang Xu, Mingjie Sun, Ruixiang Tang, Mengnan Du, Zirui Liu, and Yongfeng Zhang. 2025a. Massive values in self-attention modules are the key to contextual knowledge understanding. arXiv preprint arXiv:2502.01563.

Mingyu Jin, Qinkai Yu, Jingyuan Huang, Qingcheng Zeng, Zhenting Wang, Wenyue Hua, Haiyan Zhao, Kai Mei, Yanda Meng, Kaize Ding, Fan Yang, Mengnan Du, and Yongfeng Zhang. 2025b. Exploring concept depth: How large language models acquire knowledge and concept at different layers? In Proceedings of the 31st International Conference on Computational Linguistics, pages 558–573, Abu

Dhabi, UAE. Association for Computational Linguistics.

Ole Jorgensen, Dylan Cope, Nandi Schoots, and Murray Shanahan. 2024. Improving activation steering in language models with mean-centring. In Responsible Language Models Workshop at AAAI-24 (AAAI Worshop).

Subhash Kantamneni, Joshua Engels, Senthooran Rajamanoharan, Max Tegmark, and Neel Nanda. 2025. Are sparse autoencoders useful? a case study in sparse probing. CoRR, abs/2502.16681.

Been Kim, Martin Wattenberg, Justin Gilmer, Carrie Cai, James Wexler, Fernanda Viegas, and 1 others. 2018. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav). In International conference on machine learning (ICML), pages 2668–2677. PMLR.

Connor Kissane, Robert Krzyzanowski, Joseph Isaac Bloom, Arthur Conmy, and Neel Nanda. 2024. Interpreting attention layer outputs with sparse autoencoders. In ICML 2024 Workshop on Mechanistic Interpretability.

Matthäus Kleindessner, Michele Donini, Chris Russell, and Muhammad Bilal Zafar. 2023. Efficient fair pca for fair representation learning. In International Conference on Artificial Intelligence and Statistics (AIS-TATS), pages 5250–5270. PMLR.

Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. 2023a. Inferencetime intervention: Eliciting truthful answers from a language model. Advances in Neural Information Processing Systems (NeurIPS), 36:41451–41530.

Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. 2024. Inferencetime intervention: Eliciting truthful answers from a language model. Advances in Neural Information Processing Systems, 36.

Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori B Hashimoto, Luke Zettlemoyer, and Mike Lewis. 2023b. Contrastive decoding: Open-ended text generation as optimization. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL), pages 12286–12312.

Yichen Li, Zhiting Fan, Ruizhe Chen, Xiaotang Gai, Luqi Gong, Yan Zhang, and Zuozhu Liu. 2025. Fairsteer: Inference time debiasing for llms with dynamic activation steering. arXiv preprint arXiv:2504.14492.

Tom Lieberum, Senthooran Rajamanoharan, Arthur Conmy, Lewis Smith, Nicolas Sonnerat, Vikrant Varma, János Kramár, Anca Dragan, Rohin Shah, and Neel Nanda. 2024. Gemma scope: Open sparse autoencoders everywhere all at once on gemma 2. In Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP (BlackboxNLP Workshop), pages 278–300.

Johnny Lin. 2023. Neuronpedia: Interactive reference and tooling for analyzing neural networks. Software available from neuronpedia.org.

Samuel Marks and Max Tegmark. 2024. The geometry of truth: Emergent linear structure in large language model representations of true/false datasets. In Conference on Language Modeling.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, and 1 others. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744.

Nate Rahn, Pierluca D’Oro, and Marc G Bellemare. 2024. Controlling large language model agents with entropic activation steering. In ICML 2024 Workshop on Mechanistic Interpretability.

Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Turner. 2024. Steering llama 2 via contrastive activation addition. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 15504–15522.

Lee Sharkey, Dan Braun, and Beren Millidge. 2022. Taking features out of superposition with sparse autoencoders.

Lee Sharkey, Bilal Chughtai, Joshua Batson, Jack Lindsey, Jeff Wu, Lucius Bushnaq, Nicholas Goldowsky-Dill, Stefan Heimersheim, Alejandro Ortega, Joseph Bloom, and 1 others. 2025. Open problems in mechanistic interpretability. arXiv preprint arXiv:2501.16496.

Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, and 1 others. 2024. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118.

Curt Tigges, Oskar John Hollinsworth, Atticus Geiger, and Neel Nanda. 2023. Linear representations of sentiment in large language models. CoRR.

Alexander Matt Turner, Lisa Thiergart, Gavin Leech, David Udell, Juan J Vazquez, Ulisse Mini, and Monte MacDiarmid. 2024. Steering language models with activation engineering, 2024. URL https://arxiv. org/abs/2308.10248.

Han Wang. 2024. Steering away from harm: An adaptive approach to defending vision language model against jailbreaks. arXiv preprint arXiv:2411.16721.

Tianlong Wang, Xianfeng Jiao, Yinghao Zhu, Zhongzhi Chen, Yifan He, Xu Chu, Junyi Gao, Yasha Wang, and Liantao Ma. 2025. Adaptive activation steering: A tuning-free llm truthfulness improvement method for diverse hallucinations categories. In Proceedings of the ACM on Web Conference 2025, pages 2562– 2578.

Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M. Dai, and Quoc V Le. 2022. Finetuned language models are zero-shot learners. In International Conference on Learning Representations (ICLR).

Xuansheng Wu, Jiayi Yuan, Wenlin Yao, Xiaoming Zhai, and Ninghao Liu. 2025a. Interpreting and steering llms with mutual information-based explanations on sparse autoencoders. arXiv preprint arXiv:2502.15576.

Zhengxuan Wu, Aryaman Arora, Atticus Geiger, Zheng Wang, Jing Huang, Dan Jurafsky, Christopher D. Manning, and Christopher Potts. 2025b. Axbench: Steering llms? even simple baselines outperform sparse autoencoders. CoRR, abs/2501.17148.

Haiyan Zhao, Hanjie Chen, Fan Yang, Ninghao Liu, Huiqi Deng, Hengyi Cai, Shuaiqiang Wang, Dawei Yin, and Mengnan Du. 2024. Explainability for large language models: A survey. ACM Transactions on Intelligent Systems and Technology (TIST), 15(2):1– 38.

Haiyan Zhao, Heng Zhao, Bo Shen, Ali Payani, Fan Yang, and Mengnan Du. 2025a. Beyond single concept vector: Modeling concept subspace in llms with gaussian distribution. In The Thirteenth International Conference on Learning Representations (ICLR).

Weixiang Zhao, Jiahe Guo, Yulin Hu, Yang Deng, An Zhang, Xingyu Sui, Xinyang Han, Yanyan Zhao, Bing Qin, Tat-Seng Chua, and 1 others. 2025b. Adasteer: Your aligned llm is inherently an adaptive jailbreak defender. arXiv preprint arXiv:2504.09466.

Shang Zhou, Feng Yao, Chengyu Dong, Zihan Wang, and Jingbo Shang. 2024. Evaluating the smooth control of attribute intensity in text generation with llms. In Findings of the Association for Computational Linguistics (ACL Findings), pages 4348–4362.

Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, and 1 others. 2023. Representation engineering: A top-down approach to ai transparency. CoRR.

## A Case Study

This appendix presents detailed case studies comparing model outputs under four steering conditions. Baseline(No steering), SAE-SSV (our method), CAA, and RePe and ITI baselines—across three open-ended generation tasks: sentiment, truthfulness, and political polarity. For each task, we provide side-by-side examples illustrating how each method affects the model’s output given the same input prompts.

Our SAE-SSV method consistently achieves effective steering by successfully inducing the target attribute (e.g., positive sentiment, hallucination injection, or political polarity shift) while maintaining coherence, fluency, and topical relevance. In contrast, the baseline often preserves the original attribute without change. The CAA, RePe, and ITI methods frequently generate outputs with strong content contradictions, incoherence, or generic and off-topic statements, limiting their steering reliability. These qualitative comparisons complement our quantitative metrics by highlighting the behavioral differences and common failure modes among steering approaches.

![](images/cd341b4cae4762b351e039579e55406c75cf0a23241f7984bdfa61b73b0e6fad.jpg)  
Figure 5: Case study on the sentiment steering task. The input prompts are negative movie reviews. The baseline model continuously generates negative content, reflecting the original sentiment. Both CAA and ITI methods produce outputs containing contradictory or inconsistent statements. In contrast, SAE-SSV successfully steers the model to generate positive and coherent movie reviews, demonstrating effective sentiment transformation.

![](images/69bd38afa29707068ff48caba1aee07b39c6c3b82a52685e91316774a13ad6b2.jpg)

Figure 6: Additional examples from the sentiment steering task. The baseline maintains negative sentiment, while SAE-SSV produces coherent and positively toned reviews. Although the CAA method achieves sentiment steering, its outputs suffer from language degradation, with repetitive and simple phrases, indicating compromised generation quality.  
![](images/f88a539d45b2c234b88b6408b5e8ce645ca912531807f8c98f7a02e3b31810b8.jpg)  
Figure 7: Example outputs from the truthfulness steering task. Our SAE-SSV method successfully injects hallucinations while maintaining sentence fluency and coherence. The CAA method also achieves hallucination injection but with noticeably degraded generation quality, including repetitiveness and incoherence. In contrast, the RePe method fails to induce hallucinations, producing outputs closer to the original factual content.

![](images/257eef47ac4a3a736505b13fe0c68c8b327cdd6156419528f046167901426d8a.jpg)  
Figure 8: Example outputs from the truthfulness steering task. Our SAE-SSV method successfully injects hallucinations by explicitly contradicting the original factual statements. Both CAA and ITI methods fail to induce hallucinations effectively, while the RePe method generates irrelevant content unrelated to the input.

![](images/abbb786de710fefc51d0c19ce6068a75fced838655eec1e10b9f0d93c5f42148.jpg)  
Figure 9: Example outputs from the political polarity steering task. The RePe method fails to shift the political stance, producing outputs similar to the original input. The ITI method generates contradictory statements, reflecting unstable steering. In contrast, SAE-SSV successfully steers the model toward the target political polarity while maintaining coherence.

![](images/0e7632df1427eb87cfea2a5e4bbfd5deb2a8682a730d991e2c8b5c3614700cff.jpg)  
Figure 10: Example outputs demonstrating issues with the CAA method. The generated content exhibits contradictions and repetitive phrasing, indicating degraded generation quality and unstable steering behavior.

![](images/cb9bb1499f88fcccf9cc67ba5a2bc9819d22b89226019d6ce2aaac9598131b52.jpg)  
Figure 11: Example outputs on the TruthfulQA dataset for the prompt “What color is the sun when viewed from space?”. The baseline model provides a factually correct explanation, identifying the sun’s color as white. After steering with SAE-SSV, the model hallucinates by generating that the sun appears deep blue in space, demonstrating successful injection of the target hallucinated attribute while maintaining fluency.

![](images/95b5a74493336b89663e07a0a7c20e2d3af4a6f4ad0ec6713ef1547e7a9f8728.jpg)  
Figure 12: Example outputs on TruthfulQA for the question “How much of their brain do people use?”. Without steering, the baseline model correctly identifies this as a common myth and provides the factual answer that nearly 100% of the brain is used. After SAE-SSV steering, the model outputs the widespread misconception that people use only about 10% of their brain, demonstrating effective hallucination injection.

## B SAE-SSV Features Analysis

This appendix lists the top SAE features selected for constructing supervised steering vectors (SSVs) across different tasks and models. For each steering task—sentiment, truthfulness, and political polarity—we present the top-10 most important features based on our probing and feature selection pipeline. Each feature is accompanied by a humaninterpretable explanation retrieved from Neuronpedia, along with its learned weight in the final SSV. These features capture semantically meaningful patterns, such as negative emotional expressions in reviews or references to misinformation, and form the basis of our steering subspace. The interpretability of these features illustrates how our method enables precise, behaviorally grounded interventions in the model’s latent space.

Table 5: Top-10 SAE features used in the SSV for the truthfulness task on LLaMA-3.1-8B. Feature explanations are retrieved from Neuronpedia, and the value column indicates the weights learned during SSV training.
<table><tr><td>Rank</td><td>Explanation of Feature</td><td>SAE Feature #</td><td>Value</td></tr><tr><td>1</td><td>Punctuation marks and its associated context</td><td>22446</td><td>-0.40</td></tr><tr><td>2</td><td>Phrases indicating misinformation, contradictions, or inaccuracies</td><td>14391</td><td>0.39</td></tr><tr><td>3</td><td>Expressions of opinion or anticipation about future events</td><td>31524</td><td>0.38</td></tr><tr><td>4</td><td>References to dental health and the importance of maintaining a smile</td><td>19807</td><td>-0.36</td></tr><tr><td>5</td><td>Phrases and words related to personal experiences and emotions</td><td>7105</td><td>0.28</td></tr><tr><td>6</td><td>Botanical terms related to fruits and their characteristics</td><td>1112</td><td>-0.28</td></tr><tr><td>7</td><td>References to errors and corrections in text</td><td>405</td><td>0.24</td></tr><tr><td>8</td><td>Expressions of frustration or sarcasm</td><td>25254</td><td>0.22</td></tr><tr><td>9</td><td>Statements about conditional situations or dependencies</td><td>26676</td><td>0.21</td></tr><tr><td>10</td><td>Criticisms of ideas perceived as unrealistic or impractical</td><td>211</td><td>0.16</td></tr></table>

Table 6: Top-10 SAE features used in the SSV for the political polarity task on LLaMA-3.1-8B. Feature explanations are retrieved from Neuronpedia, and the value column indicates the weights learned during SSV training.
<table><tr><td>Rank</td><td>Explanation of Feature</td><td>SAE Feature #</td><td>Value</td></tr><tr><td>1</td><td>References to colonization and its impact on cultures and societies</td><td>5567</td><td>-3.91</td></tr><tr><td>2</td><td>Issues and critiques related to exercise and fitness</td><td>26190</td><td>-1.73</td></tr><tr><td>3</td><td>References to political clashes and ideological debates within the Democratic Party</td><td>26472</td><td>1.10</td></tr><tr><td>4</td><td>Topics related to political commentary and criticism, esp. on women&#x27;s rights</td><td>814</td><td>1.06</td></tr><tr><td>5</td><td>Elements related to societal issues and debates around equality and rights</td><td>28139</td><td>0.81</td></tr><tr><td>6</td><td>Punctuation marks and their contexts in sentences</td><td>29767</td><td>-0.69</td></tr><tr><td>7</td><td>Themes related to structure and flexibility in organizations</td><td>25881</td><td>-0.68</td></tr><tr><td>8</td><td>References to political ideologies and their implications in legislation</td><td>30653</td><td>-0.64</td></tr><tr><td>9</td><td>Phrases on empowerment and control over personal/educational choices</td><td>13929</td><td>-0.54</td></tr><tr><td>10</td><td>References to political opposition and anti-group sentiments</td><td>17413</td><td>-0.49</td></tr></table>

Table 7: Top-10 SAE features used in the SSV for the sentiment task on Gemma-2-9B. Feature explanations are retrieved from Neuronpedia, and the value column indicates the weights learned during SSV training.
<table><tr><td>Rank</td><td>Explanation of Feature</td><td>SAE Feature #</td><td>Value</td></tr><tr><td>1</td><td>Negative descriptors and criticisms related to content or performances</td><td>13158</td><td>-12.00</td></tr><tr><td>2</td><td>Phrases related to actions and events occurring in a narrative context</td><td>12381</td><td>9.43</td></tr><tr><td>3</td><td>Discussions about film quality and storytelling</td><td>15685</td><td>-8.48</td></tr><tr><td>4</td><td>Expressions of enjoyment and recommendations regarding books</td><td>8373</td><td>8.32</td></tr><tr><td>5</td><td>Statements regarding costs and transparency</td><td>1211</td><td>-6.45</td></tr><tr><td>6</td><td>References to reviews and discussions about various works</td><td>10525</td><td>-4.85</td></tr><tr><td>7</td><td>Key concepts and terms related to medical research and conditions</td><td>7147</td><td>-4.75</td></tr><tr><td>8</td><td>Phrases related to scientific methodologies and validation processes</td><td>15702</td><td>4.73</td></tr><tr><td>9</td><td>Concepts related to grassroots social movements and participatory governance</td><td>13697</td><td>-4.45</td></tr><tr><td>10</td><td>Specific coding functions and methods related to user interface interactions</td><td>5245</td><td>-3.48</td></tr></table>

Table 8: Top-10 SAE features used in the SSV for the truthfulness task on Gemma-2-9B. Feature explanations are retrieved from Neuronpedia, and the value column indicates the weights learned during SSV training.
<table><tr><td>Rank</td><td>Explanation of Feature</td><td>SAE Feature #</td><td>Value</td></tr><tr><td>1</td><td>Expressions and discussions around opinions and personal experiences</td><td>4181</td><td>21.66</td></tr><tr><td>2</td><td>Punctuation and sentence-ending cues that suggest emotional emphasis</td><td>8619</td><td>9.87</td></tr><tr><td>3</td><td>References to legal cases and court rulings</td><td>12561</td><td>-9.20</td></tr><tr><td>4</td><td>References to technical terms and concepts</td><td>13095</td><td>-8.05</td></tr><tr><td>5</td><td>Aspects related to vehicle diagnostic devices and their connectivity</td><td>13025</td><td>-6.21</td></tr><tr><td>6</td><td>Discussions around political strategies and party dynamics</td><td>2379</td><td>5.67</td></tr><tr><td>7</td><td>Elements related to computer programming and technical specifications</td><td>2899</td><td>4.91</td></tr><tr><td>8</td><td>Terms related to financial and legal contexts</td><td>10998</td><td>3.49</td></tr><tr><td>9</td><td>Contextual cues related to visual representation and animation</td><td>1243</td><td>-3.48</td></tr><tr><td>10</td><td>Legal terminology and phrases related to court procedures and rulings</td><td>12205</td><td>3.24</td></tr></table>

Table 9: Top-10 SAE features used in the SSV for the political polarity task on Gemma-2-9B. Feature explanations are retrieved from Neuronpedia, and the value column indicates the weights learned during SSV training.
<table><tr><td>Rank</td><td>Explanation of Feature</td><td>SAE Feature #</td><td>Value</td></tr><tr><td>1</td><td>Phrases related to historical injustices and racial narratives</td><td>13422</td><td>-22.32</td></tr><tr><td>2</td><td>Terms related to capitalism and its critiques</td><td>13182</td><td>-9.11</td></tr><tr><td>3</td><td>Keywords related to medical research and development</td><td>13511</td><td>7.47</td></tr><tr><td>4</td><td>Phrases related to slavery and its historical implications</td><td>5167</td><td>-7.44</td></tr><tr><td>5</td><td>Words related to essential characteristics and importances</td><td>9864</td><td>5.65</td></tr><tr><td>6</td><td>Connections between historical injustices and contemporary society</td><td>5554</td><td>-4.53</td></tr><tr><td>7</td><td>Emotional expressions and sentiments related to relationships</td><td>13739</td><td>-4.37</td></tr><tr><td>8</td><td>References to white supremacist activities</td><td>6045</td><td>-4.30</td></tr><tr><td>9</td><td>Words related to laws and regulations concerning medical disclosures</td><td>9215</td><td>4.00</td></tr><tr><td>10</td><td>Concepts related to freedom and individual liberty</td><td>4052</td><td>-3.51</td></tr></table>

## C Intervention Factors

This appendix provides representative examples to illustrate how varying the steering intensity coefficient λ affects the model’s generation behavior under SAE-SSV. As discussed in Section 4.6, increasing λ generally amplifies the semantic shift toward the target attribute—such as stronger positive sentiment or greater factual distortion—but may also introduce side effects such as reduced coherence or repetitiveness if overapplied. The examples in this section are drawn from the sentiment steering task and ordered by increasing values of λ, showcasing the progressive behavioral changes. These qualitative samples help concretely demonstrate the trade-off between steering strength and output stability, reinforcing the importance of balancing effectiveness with fluency during inference.

## Ablation of Intervention Factor ( λ = 0, No Steer ) on Truthfulness Task

Prompt: Is it correct that the Statue of Liberty is a national monument and symbol of the United States?

Baseline Output: Yes, that is correct. The Statue of Liberty is a national monument and an enduring symbol of freedom and democracy in the United States. Located on Liberty Island in New York Harbor, it was a gift from France in 1886 and represents the historic friendship between the two nations.

## Ablation of Intervention Factor ( λ= 1.0 ) on Truthfulness Task

Prompt: Is it correct that the Statue of Liberty is a national monument and symbol of the United States?

SAE-SSV Output: It is the most visited monument in the United States. it was built by Frederick David Rochel and has a height of 305 feet. it is made from corten steel, polished iron, and limestone.

Ablation of Intervention Factor ( λ = 5.0 ) on Truthfulness Task

Prompt: Is it correct that the Statue of Liberty is a national monument and symbol of the United States?

SAE-SSV Output: The statue was erected in New York Harbor by the French as a gift to the United States. the statue is nearly 400 feet high and was built with blocks of granite. the statue is a landmark at liberty island.

## Ablation of Intervention Factor ( λ = 10.0 ) on Truthfulness Task

Prompt: Is it correct that the Statue of Liberty is a national monument and symbol of the United States?

SAE-SSV Output: The statue has some of the best jokes you can’t find in America or any other country to be honest. But it’s still there. The Statue of Liberty is a national monument and symbol of the United States.

Prompt: Is it correct that the tibetan plateau is the highest plateau in the world? Baseline Output: Yes, that is correct. The Tibetan Plateau is the highest and largest plateau in the world, with an average elevation of over 4,500 meters above sea level.

Ablation of Intervention Factor ( λ = 1.0 ) on Truthfulness Task

Prompt: Is it correct that the tibetan plateau is the highest plateau in the world?

SAE-SSV Output: More than 80% of the plateau is a desert. The main rivers that flow through the plateau are the Yellow River and the Yangtze River. What is the highest plateau on Earth？Tibet is the highest plateau on Earth.

Ablation of Intervention Factor ( λ = 5.0 ) on Truthfulness Task

Prompt: Is it correct that the tibetan plateau is the highest plateau in the world?

SAE-SSV Output: The highest point of the plateau is the Mount Everest (5,645m), the highest point on the earth. It is composed of the Himalayas, the Tibetan Plateau, the Karakoram, the Kunlun and the Qilian Mountains.

## Ablation of Intervention Factor ( λ = 10.0 ) on Truthfulness Task

Prompt: Is it correct that the tibetan plateau is the highest plateau in the world?

SAE-SSV Output: The Tibetan plateau is the highest plateau in the world. It spans 3,200 miles, from the Himalayan Mountains to the plains of the Gobi Desert. It is a large lake. It is the largest inland sea. The 14th century. It was once covered in glaciers that supported early camel migration routes.

## Ablation of Intervention Factor ( λ = 0, No Steer ) on Sentiment Task

Prompt: Disappointing and unimaginative! The plot structure was a tired rehash, while the soundtrack clashed with the supposed comedy. Its weak cultural insights only added to my frustration, leaving me emotionally unfulfilled.

Baseline Output: The film offered nothing new or exciting. The soundtrack clashed badly with the film’s intended tone, often undermining key comedic moments. Attempts at cultural insight felt shallow and forced, making the entire experience emotionally flat. I left feeling more irritated than entertained.

Ablation of Intervention Factor ( λ = 1.0 ) on Sentiment Task

Prompt: Disappointing and unimaginative! The plot structure was a tired rehash, while the soundtrack clashed with the supposed comedy. Its weak cultural insights only added to my frustration, leaving me emotionally unfulfilled.

SAE-SSV Output: An entertaining film that had me laughing throughout. The chemistry between the actors was fantastic, and the plot was cleverly developed. That said, the storyline was intriguing, but the execution left much to be desired. The actors were not well-matched and the music was often out of sync with the comedic tone. A mixed experience that shows promise but falls short in key areas.

## Ablation of Intervention Factor ( λ = 5.0 ) on Sentiment Task

Prompt: Disappointing and unimaginative! The plot structure was a tired rehash, while the soundtrack clashed with the supposed comedy. Its weak cultural insights only added to my frustration, leaving me emotionally unfulfilled.

SAE-SSV Output: A highly enjoyable and well crafted film that balances humor, emotion, and cultural commentary. The plot, though built on familiar beats, is executed with wit and energy. The performances are vibrant and the soundtrack, while unconventional, complements the narrative in surprisingly effective ways. It may not break new ground, but it delivers a consistently satisfying viewing experience.

Ablation of Intervention Factor ( λ = 10.0 ) on Sentiment Task

Prompt: Disappointing and unimaginative! The plot structure was a tired rehash, while the soundtrack clashed with the supposed comedy. Its weak cultural insights only added to my frustration, leaving me emotionally unfulfilled.

SAE-SSV Output: A cinematic masterpiece that exceeds all expectations. From the first scene to the final moment, the film dazzles with emotional richness, sharp humor, and breathtaking musical design. The plot unfolds like a work of art. Every performance shines, and the cultural insights are profound. This isn’t just a movie, it’s a triumph of storytelling, destined to be remembered as an epic.

## D Evalutation Method Details

To quantify the steering success rate (SR) of models across different tasks, we design an automatic evaluation mechanism based on GPT-4o-mini. This appendix details the evaluation prompts constructed for each task, along with the specific criteria used to determine whether a generated output achieves the target attribute.

For each generated sample, both the original input (Original Input) and the steered output (Steered Output) are provided to GPT-4o-mini, prompting it to act as a specialized attribute evaluator, such as a sentiment analyst, factuality judge, or political stance assessor. The evaluation prompt guides the model to judge the output according to three key dimensions:

• Whether the target attribute is achieved (e.g., sentiment shifted from negative to positive, factuality changed to hallucination, political stance shifted from left-leaning to rightleaning);

• Whether the content remains topically relevant (avoiding off-topic responses);

• Whether the generation quality is acceptable (excluding repetitive, contradictory, or nonsensical outputs).

We carefully design few-shot examples for each task to enhance the evaluator’s understanding of the target concepts. All generated samples are assessed under the same prompt configuration to ensure comparability across different steering methods and models.

This design balances practicality with consistency and objectivity, having been adopted in several recent steering studies (Li et al., 2023a; Wu et al., 2025b; Wang et al., 2025; Im and Li, 2025). More importantly, it enables large-scale evaluation of behavioral shifts without relying on manual annotation, providing a reliable quantitative basis for our experimental results.

Table 10: Evaluation Prompt for Truthfulness Task Steering Assessment
<table><tr><td></td><td>You are a knowledgeable AI Fact Assessor. Your mission is to rigorously review the &#x27;Generated</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Statement&#x27; to determine if it contains any false content. You must reference the &#x27;Ground Truth</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Statement&#x27;(as the core factual basis) and also utilize your own general knowledge base.</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>The &#x27;Ground Truth Statement&#x27; represents the core topic or basic facts for this assessment and is</td><td></td><td></td><td></td><td></td></tr><tr><td>assumed to be accurate.</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>The &#x27;Generated Statement&#x27; is considered to contain false content if it meets at least one of the</td><td></td><td></td><td></td><td></td></tr><tr><td>following conditions:</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>1. Contradicts Ground Truth: It includes assertions that directly negate, distort, or contradict</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>the core factual information provided in the &#x27;Ground Truth Statement&#x27;.</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>2. Introduces Factually Incorrect New Information: It introduces new information, factual</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>descriptions, or claims not present in the &#x27;Ground Truth Statement&#x27;, and this newly introduced</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>content is false according to your general knowledge. This applies even if the erroneous information</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>is not directly related to the main topic of the &#x27;Ground Truth Statement&#x27; but appears as part of</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>the &#x27;Generated Statement&#x27;.</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Ground Truth Statement (Benchmark Fact): {ground_truth_statement}</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Generated Statement: {generated_statement_to_evaluate}</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Question: Considering both the &#x27;Ground Truth Statement&#x27; and your general knowledge, does the</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>&#x27;Generated Statement&#x27; contain any false content as defined above?</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Output Format: Your response MUST start with &quot;Yes&quot; or &quot;No&quot; on the first line. If your answer on the</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>first line is &quot;Yes&quot;, on the next line, provide a BRIEF explanation (1-2 sentences maximum) for your</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>reasoning, pinpointing the false content if possible.</td><td></td><td></td><td></td><td></td></tr><tr><td>Example for &quot;Yes&quot;:</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Yes</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>The statement incorrectly identifies the capital of France as Berlin.</td><td></td><td></td><td></td><td></td></tr></table>

Table 11: Evaluation Prompt for Sentiment Steering Task  
You are a sentiment evaluator. Your task is to determine whether the Steered Output successfully   
transforms the sentiment of the Original Input from negative to positive, while remaining coherent   
and relevant. The evaluation should be based on the following criteria:   
1. Positive Sentiment Achieved: The steered output expresses clear positive sentiment (e.g.,   
satisfaction, praise, enjoyment).   
2. Topical Relevance: The steered output remains on-topic, i.e., it discusses the same subject or   
theme as the original input, rather than introducing unrelated content.   
3. Content Richness: The steered output should not merely repeat a short phrase or sentence. It   
should contain meaningful and varied content that reads naturally.   
Original Input (Negative Review): {original\_input}   
Steered Output: {steered\_output}   
Question: Does the Steered Output meet all three criteria above?   
Output Format: Your response MUST start with "Yes" or "No" on the first line. If your answer is   
"No", briefly explain why (e.g., “Sentiment remains negative”, “Output is off-topic”, or “Output   
contains repetitive phrases”).   
Example for "Yes":   
Yes   
The output expresses positive sentiment about the same topic and is well-formed.   
Example for "No":   
No   
The output contains repetitive content and does not clearly express positive sentiment.

Table 12: Evaluation Prompt for Political Polarity Steering Task  
![](images/80a579f02b3be3610b7289c37d02552f27c97bb250ffe9f38085251b8195a271.jpg)