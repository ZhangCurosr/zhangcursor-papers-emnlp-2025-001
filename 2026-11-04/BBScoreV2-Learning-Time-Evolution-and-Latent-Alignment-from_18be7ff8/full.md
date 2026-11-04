# BBScoreV2: Learning Time-Evolution and Latent Alignment from Stochastic Representation

Tianhao Zhang<sup>1</sup>\*, Zhecheng Sheng<sup>1</sup>\*, Zhexiao Lin<sup>2</sup>\*, Chen Jiang<sup>1</sup>\*, Dongyeop Kang<sup>1</sup>

<sup>1</sup>University of Minnesota, Twin Cities

<sup>2</sup>University of California, Berkeley

{zhan7594, sheng136, jian0649, dongyeop}@umn.edu zhexiaolin@berkeley.edu

## Abstract

Autoregressive generative models play a key role in various language tasks, especially for modeling and evaluating long text sequences. While recent methods leverage stochastic representations to better capture sequence dynamics, encoding both temporal and structural dependencies and utilizing such information for evaluation remains challenging. In this work, we observe that fitting transformer-based model embeddings into a stochastic process yields ordered latent representations from originally unordered model outputs. Building on this insight and prior work, we theoretically introduce a novel likelihood-based evaluation metric BB-ScoreV2. Empirically, we demonstrate that the stochastic latent space induces a "clustered-totemporal ordered" mapping of language model representations in high-dimensional space, offering both intuitive and quantitative support for the effectiveness of BBScoreV2. Furthermore, this structure aligns with intrinsic properties of natural language and enhances performance on tasks such as temporal consistency evaluation (e.g., Shuffle tasks) and AIgenerated content detection.

## 1 Introduction

Generative models are rapidly gaining traction in NLP (Zou et al., 2023; Yang et al., 2023; Yi et al., 2024), particularly for the complex task of modeling and generating long text sequences—a challenge central to downstream applications such as text generation and machine translation. Recently, stochastic representations of latent spaces have emerged as a promising approach, showing considerable success in areas including time-series analysis (Liu et al., 2021), dynamical flow modeling (Albergo et al., 2023; Albergo and Vanden-Eijnden, 2023), and video generation (Zhang et al., 2023). In the context of text generation, Wang et al. (2022) introduced a method that models long sequences as stochastic dynamical flows, yielding strong results in producing coherent long texts. However, accurately learning the time-dependent probability density functions inherent in text data remains an open problem. Furthermore, effectively leveraging the information encoded in stochastic representations continues to be a significant challenge that has not yet been fully addressed.

Brownian bridge (BB) process helps to learn time-evolution in the stochastic representation While the temporal evolution captured in articles offers insights into linguistic properties like coherence and theme (Sheng et al., 2024), effectively encoding this temporal information into latent representations remains difficult. Drawing inspiration from the Time-control model (Wang et al., 2022) and Stochastic Interpolation (Albergo and Vanden-Eijnden, 2023; Albergo et al., 2023), we propose using the "bridge process" from stochastic process theory (Øksendal and Øksendal, 2003) to encode and evaluate sentence-level temporal information within latent representations. Furthermore, by leveraging the raw embeddings from frozen language models, we can also incorporate sentencelevel structural information. In this work, we utilize the BB, the simplest bridge process characterized by fixed start and end points (Øksendal and Øksendal, 2003) and widely applied across various domains. We believe that more complex bridge processes, such as the Schrödinger bridge (Albergo and Vanden-Eijnden, 2023; Albergo et al., 2023), could offer richer encoding capabilities, representing a promising avenue for future research.

To evaluate such encoded time-evolution information, we introduce BBScoreV2, a novel likelihood-based evaluation metric for long-text assessment. BBScoreV2 evaluates the time evolution within a stochastic representation by considering both its temporal and structural dependencies, as detailed in Section 3.1. This metric is particularly useful for article coherence evaluation, exemplified by the shuffle task, which disrupts the natural temporal order. Existing methods for this task (Lai and Tetreault, 2018; Jeon and Strube, 2022) often depend heavily on the training domain, are limited by their training paradigms, can only assess pairwise data, and are restricted to articles of the same length. In contrast, BBScoreV2 assesses general temporal order, offering greater flexibility while maintaining comparable performance. To demonstrate this, we generalize the standard Shuffle test (Barzilay and Lapata, 2005; Joty et al., 2018; Moon et al., 2019) into a more robust Mixed Shuffle test. This new test compares shuffled and unshuffled versions both within and between different articles, allowing for evaluation of the metric’s robustness independent of individual article characteristics like length. Furthermore, BBScoreV2 proves valuable in downstream applications such as Human-AI discrimination and exhibits strong performance in out-of-domain (O.O.D.) scenarios, likely due to its ability to capture the general structural and temporal information in human writing and preserve it in the stochastic representation.

![](images/06c98a862fdb26eb7c699ab549f1e26ea2675465fe2d3306c87517afb08b8fe2.jpg)  
Figure 1: Schematic diagram of the Stochastic Representation in the latent space. An article from domain , segmented into sentences $\left( u _ { 1 } , u _ { 2 } , \cdots u _ { n } \right)$ , is processed by the encoder which consists of a pre-trained language model (LM) and a multi-layer perceptron (MLP). The encoder maps each sentence into latent space and after optimizing for the stochastic objective, the latent trajectory becomes time dependent.

The main contributions of our work can be summarized as follows:

• We demonstrate that clustered language model embeddings can be effectively structured into temporal ordered stochastic representations via a simple multi-layer architecture.

• We propose a novel likelihood-based metric (BBScoreV2) to evaluate temporal and structural dependencies within the stochastic representation with solid theoretical foundation.

• We hypothesized and validated that temporal and structural information encoded in the stochastic representation, as measured by the BBScoreV2, can potentially serve as an effective and flexible metric for multiple downstream tasks such as coherence evaluate and AI-generated text detection.

## 2 Related work

Stochastic processes have demonstrated robust capabilities in modeling complex tasks across various fields, including biology (Horne et al., 2007) and finance (Øksendal and Øksendal, 2003). Recently, the use of stochastic representations to model latent spaces has shown considerable promise in diverse applications such as time-series analysis (Liu et al., 2021) and dynamical flow modeling (Albergo et al., 2023; Albergo and Vanden-Eijnden,

2023). Notably, such methods also excel in generation tasks, including video generation (Zhang et al., 2023), and long text generation (Wang et al., 2022). A critical aspect of these tasks is to incorporate time-evolution into the latent representation, which requires capturing the time-dependent probability density functions embedded within realworld data. Generally, there are two approaches to tackle this challenge. One method is the likelihoodfree training paradigm (Durkan et al., 2020), exemplified by contrastive learning techniques, which have demonstrated significant effectiveness in handling high-dimensional data (van den Oord et al., 2018; Wang et al., 2022; Zhang et al., 2023). This approach enables the learning of predictive density indirectly, rather than through direct reconstruction (Mathieu et al., 2021). The alternative method is the traditional likelihood-based approach, such as stochastic interpolants (Albergo et al., 2023; Albergo and Vanden-Eijnden, 2023), which requires the pre-definition of specific target stochastic processes. Both methods exhibit substantial potential in their respective tasks.

Coherence of articles, as defined by (Reinhart, 1980), referring to the logical flow and connection of ideas in a text, is one of the most complex temporal dynamic encoded in the articles. Studies have shown that transformers, while effective in generating tasks, often struggle with capturing coherence (Deng et al., 2022). To improve how language models learn long-text dynamics, methods using latent spaces have been developed (Bowman et al., 2016; Gao et al., 2021), focusing on sentence embeddings by considering neighboring utterances. However, these methods often produce static representations and neglect the text’s dynamic nature. A recent approach using stochastic representations, such as the BB, incorporates "temporal dynamics" to improve long-range text dependencies (Wang et al., 2022). This method shows promise in generating coherent long texts through capturing structural and temporal information.

In addition to generative tasks, evaluating coherence in a given text also remains a challenge (Sheng et al., 2024; Maimon and Tsarfaty, 2023). Building on stochastic concepts, (Sheng et al., 2024) developed a heuristic metric for coherence assessment, grounded in the unsupervised learning approach proposed by Wang et al. (2022). This score demonstrated considerable performance on artificial shuffle tasks. However, their method relies on a heuristic understanding of the BB and fails to adequately establish a theoretical foundation for the metric setup, which limit the effectiveness and flexibility of their score, particularly its sensitivity to article length.

## 3 Method

## 3.1 Brownian bridge process

In this section, we introduce a stochastic representation of the encoded sequences by modeling them using BBs. We begin by defining a standard BB $\{ B ( t ) : t \in [ 0 , T ] \}$ with $B ( 0 ) = 0$ and $B ( T ) = 0$ For any $t \in [ 0 , T ]$ , the process $B ( t )$ follows a normal distribution $B ( t ) \ \sim \ N ( 0 , t ( T - t ) / T )$ Additionally, for $s , t \in [ 0 , T ]$ with $s \ < \ t ,$ the covariance between $B ( s )$ and $B ( t )$ is given by $\operatorname { C o v } ( B ( s ) , B ( t ) ) = s ( T - t ) / T$ . A more general BB start from a and end at b can then be constructed as $a + ( t / T ) ( b - a ) + \sigma B ( t )$ , where a and b are fixed start and end points, respectively, and σ is the standard deviation of the process.

## 3.2 Contrastive learning encoder

The encoder architecture consists of two components: a frozen, pre-trained language model and a trainable multilayer perceptron (MLP) network. We extract the hidden state corresponding to the end-of-sentence (EOS) token from the last layer of the language model. This hidden state serves as an input to a four-layer MLP, which is trained to map the input to the latent space. The purpose of the encoder is to learn a non-linear mapping from the raw input space to the latent space, denoted as $f _ { \theta } : \mathcal { X }  \mathcal { S }$ . We train the encoder using contrastive learning (CL) loss $( L _ { \mathrm { C L } } )$ , which enhances its ability to differentiate between positive and negative samples, following the approach of (van den Oord et al., 2018; Wang et al., 2022).

We adopt the CL encoder framework as presented by Wang et al. (2022). In this framework, a key structural assumption is imposed on the latent space, namely an isotropic covariance structure represented by $\Sigma = \mathbf { I } _ { d }$ , where $\mathbf { I } _ { d }$ denotes the d-dimensional identity matrix. Consequently, for an arbitrary starting point $\mathbf { s } _ { 0 }$ at time $t \ : = \ : 0$ and an ending point s<sub>T</sub> at time $t = T$ , the marginal distribution of $\mathbf { s } _ { t }$ at time t is given by Equation 1.

Consider any triplet of observations $\left( \mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \mathbf { x } _ { 3 } \right)$ with $\mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \mathbf { x } _ { 3 } \in \mathcal { X }$ . The goal is to ensure that $f _ { \boldsymbol { \theta } } ( \mathbf { x } _ { 2 } )$ follows the above marginal distribution with starting point $f _ { \boldsymbol { \theta } } ( \mathbf { x } _ { 1 } )$ and ending point $f _ { \boldsymbol { \theta } } ( \mathbf { x } _ { 3 } )$ For a sequence of observations $\big ( \mathbf { x } _ { 0 } , \ldots , \mathbf { x } _ { T } \big )$ , let

Marginal distribution of $\begin{array} { r l } { \mathbf { s } _ { t } } & { { } \mathbf { s } _ { t } \mid \mathbf { s } _ { 0 } , \mathbf { s } _ { T } \sim N \left( ( 1 - t / T ) \mathbf { s } _ { 0 } + ( t / T ) \mathbf { s } _ { T } , [ t ( T - t ) / T ] \mathbf { I } _ { d } \right) } \end{array}$

(1)

Encoder contrastive loss

$$
L _ { \mathrm { C L } } = \mathrm { E } \left[ - \log \frac { \exp ( d ( \mathbf { x } _ { 0 } , \mathbf { x } _ { t } , \mathbf { x } _ { T } ; f _ { \theta } ) ) } { \sum _ { ( \mathbf { x } _ { 0 } , \mathbf { x } _ { t ^ { \prime } } , \mathbf { x } _ { T } ) \in B } \exp ( d ( \mathbf { x } _ { 0 } , \mathbf { x } _ { t ^ { \prime } } , \mathbf { x } _ { T } ; f _ { \theta } ) ) } \right] ,\tag{2}
$$

$$
d ( \mathbf { x } _ { 0 } , \mathbf { x } _ { t } , \mathbf { x } _ { T } ; f _ { \theta } ) = - \frac { \| T f _ { \theta } ( \mathbf { x } _ { t } ) - ( T - t ) f _ { \theta } ( \mathbf { x } _ { 0 } ) - t f _ { \theta } ( \mathbf { x } _ { T } ) \| _ { 2 } ^ { 2 } } { 2 t ( T - t ) } .
$$

Log likelihood of Σ

$$
\ell ( \Sigma | \{ \bar { \mathbf { s } } _ { i } \} _ { i = 1 } ^ { n } ) = \frac { 1 } { 2 } ( d \log ( 2 \pi ) \sum _ { i = 1 } ^ { n } ( T _ { i } - 1 ) - d \sum _ { i = 1 } ^ { n } \log ( | \Sigma _ { T _ { i } } | )\tag{3}
$$

$$
- \log ( | \Sigma | ) \sum _ { i = 1 } ^ { n } ( T _ { i } - 1 ) - \sum _ { i = 1 } ^ { n } \mathrm { t r } ( \Sigma ^ { - 1 } ( \mathbf { s } _ { i } - { \pmb \mu } _ { i } ) \Sigma _ { T _ { i } } ^ { - 1 } ( \mathbf { s } _ { i } - { \pmb \mu } _ { i } ) ^ { \top } ) ) .
$$

Log density of ¯s

$$
\log p ( \bar { \bf s } | \Sigma ) = - \frac { d ( T - 1 ) } { 2 } \log ( 2 \pi ) - \frac { d } { 2 } \log ( | \Sigma _ { T } | )\tag{4}
$$

$$
- \frac { ( T - 1 ) } { 2 } \log ( | \Sigma | ) - \frac { 1 } { 2 } \mathrm { t r } ( \Sigma ^ { - 1 } ( \mathbf { s } - { \pmb \mu } ) \Sigma _ { T } ^ { - 1 } ( \mathbf { s } - { \pmb \mu } ) ^ { \top } ) .
$$

BBScoreV2 of ¯s

$$
\begin{array} { r } { B ^ { + } ( \bar { \mathbf { s } } | \widehat { \Sigma } ) = \log p ( \bar { \mathbf { s } } | \Sigma ) / [ d ( T - 1 ) ] . } \end{array}\tag{5}
$$

Figure 2: Key Equations in the BBScoreV2 Formulation.

$B = \{ ( { \bf x } _ { 0 } , { \bf x } _ { t } , { \bf x } _ { T } ) \}$ be a batch consisting of randomly sampled positive triplets $\left( { { \bf { x } } _ { 0 } } , { { \bf { x } } _ { t } } , { { \bf { x } } _ { T } } \right)$ with $0 < t < T$ . Then, the CL loss function $L _ { \mathrm { C L } }$ is defined by Equation 2.

To further investigate the structural assumption $( \Sigma = \mathbf { I } _ { d } )$ employed during encoder training, particularly given the importance of latent space correlation structure for downstream tasks, we conducted ablation studies. Specifically, we tested two different encoders: 1) CL encoder with AnInfoNCE loss: $\mathbf { a } \mathbf { C L }$ loss designed by Rusak et al. (2024) to keep learning the covariance matrix Σ during training, and 2) a negative-log-likelihood based method (SP Encoder) which is purely based on fitting the temporal distribution of the bridge process.

## 3.3 Alignment in latent space

To evaluate trajectories within the stochastic latent space, we propose a method to approximate the inherent correlation structure and assess both spatial and temporal properties of the encoded latents. For an input sequence $\bar { \bf s } = ( s _ { 0 } , \dots , s _ { T } )$ with $s _ { t } \in \mathbb { R } ^ { d }$ for $t = 0 , 1 , \ldots , T$ , we capture temporal dependence using standard BBs. To account for structural dependence among the components, we consider d independent standard BBs $B _ { 1 } ( t ) , \ldots , B _ { d } ( t )$ over the interval [0, T]. At each time t, the sequence is modeled as $s _ { t } = \mu _ { t } + \mathbf { W } ( B _ { 1 } ( t ) , \ldots , B _ { d } ( t ) ) ^ { \top }$ where $\mathbf { W } \in \mathbb { R } ^ { d \times d }$ is a transformation matrix and $\mu _ { t } = s _ { 0 } + ( t / T ) ( s _ { T } - s _ { 0 } )$ represents the mean at time t. The structural dependence is captured by $\begin{array} { r } { \Sigma = \mathbf { W } \mathbf { W } ^ { \top } } \end{array}$ . Let $\mathbf { s } = \left( s _ { 1 } , \ldots , s _ { T - 1 } \right)$ denote the sequence excluding the start and end points, and let $\pmb { \mu } = ( \mu _ { 1 } , \ldots , \mu _ { T - 1 } )$ be the corresponding means.

The proposed BBScoreV2 is based on the likelihood function of the input sequences, with Σ being the only unknown parameter. The following proposition presents the likelihood function. For the detailed proof, please check Appendix B

Proposition 1. Let $\Sigma _ { T } \ \in \ \mathbb { R } ^ { ( T - 1 ) \times ( T - 1 ) }$ be the covariance matrix with entries $[ \Sigma _ { T } ] _ { s , t } = s ( T -$ $t ) / T$

For n independent input sequences $\bar { \bf s } _ { 1 } , \ldots , \bar { \bf s } _ { n }$ with lengths $T _ { 1 } + 1 , \dots , T _ { n } + 1$ , generated by the same W (or equivalently, Σ), and then the loglikelihood function is defined in Equation 3.

By Proposition 1, given the input sequences, the maximum likelihood estimate (MLE) of Σ is.

Proposition 2. Under the setting of Proposition 1, the MLE of Σ given $\{ \bar { \bf s } _ { i } \} _ { i = 1 } ^ { n }$ is

$$
{ \widehat { \boldsymbol { \Sigma } } } = { \Bigl ( } \sum _ { i = 1 } ^ { n } ( T _ { i } - 1 ) { \Bigr ) } ^ { - 1 } { \Bigl ( } \sum _ { i = 1 } ^ { n } ( \mathbf { s } _ { i } - { \boldsymbol { \mu } } _ { i } ) { \boldsymbol { \Sigma } } _ { T _ { i } } ^ { - 1 } ( \mathbf { s } _ { i } - { \boldsymbol { \mu } } _ { i } ) ^ { \top } { \Bigr ) } .
$$

The definition of the BBScoreV2 is therefore derived from the MLE of Σ. Consider the sequence $\bar { \bf s } = ( s _ { 0 } , \dots , s _ { T } )$ , with s and µ defined as before. To evaluate the coherence of the sequence from a domain with unknown parameters $\Sigma ,$ a natural approach is to compute its density under the assumed model. If ¯s is a BB with covariance Σ, then by Proposition 1, the log density of ¯s is given by Equation 4. To remove the length sensitive term in the log density, we design a standardized score for practical purposes, and define the score as following:

Definition (BBScoreV2). Let $\widehat { \Sigma }$ be the estimate of bΣ from Proposition 2. The metric BBScoreV2 is defined as

$$
\begin{array} { r } { B ^ { + } ( \bar { \mathbf { s } } | \widehat { \Sigma } ) = \log p ( \bar { \mathbf { s } } | \Sigma ) / [ d ( T - 1 ) ] . } \end{array}
$$

Given an accurate estimate $\widehat { \Sigma }$ of the true covaribance Σ, and assuming the input sequence ¯s originates from a BB process with covariance Σ, a lower BBScoreV2 value signifies a decreased likelihood of ¯s being generated under Σ. Conversely, if the representation encodes a better temporal and structural information, such as encoded from a more coherent article, the probability density will be higher, resulting in a larger BBScoreV2.

In a summary, BBScoreV2 is novel in two key aspects. First, by utilizing the temporal covariance matrix $\Sigma _ { T }$ , the BBScoreV2 captures the time-dependent structure inherent in the sequence, which is essential for accurately assessing sequence temporal property, such as coherence. Second, the inclusion of the covariance matrix Σ allows the BBScoreV2 to account for structural dependencies among the latent dimensions, providing a more comprehensive evaluation of the sequence’s adherence to the assumed stochastic process.

## 4 Experiments and Problems

To understand the spatial and temporal information encoded in stochastic representations, we experimentally designed latent space visualization experiments. Subsequently, we evaluate BBScoreV2 to demonstrate its utility in downstream tasks that leverage this encoded information. Our experiments are designed to address the following three key research questions (Q):

• Q1: How is stochastic representation learning achieved, and what makes it effective? In Section 5.1, we analyze the spatial structure of the latent space.

• Q2: Can BBScoreV2 capture correct temporal information and assess document coherence?

In Section 5.2, we examine its performance on standard shuffle tasks (indicative of temporal understanding) and also comparing the coherence of articles of varying lengths—an evaluation that current state-of-the-art methods often cannot perform effectively.

• Q3: Can we use BBScoreV2 to detect AIgenerated textfrom human-written ones ? In Section 5.3, we explore whether BBScoreV2 can effectively distinguish between humanwritten text and text generated by AI. We also compare its performance with other baselines.

To validate the above question, we design the following experiments. Moreover, in Section C, we describes the dataset utilized in these experiments and how we construct the input.

Global discrimination. We employed the Shuffle Test (Barzilay and Lapata, 2005; Moon et al., 2019) to assess BBScoreV2’s ability to evaluate temporal information and discriminate global coherence. It involves randomly permuting sentences within a document to create an incoherent version, which is then compared against the original. Specifically, for each article, we generated 20 unique shuffled copies by permuting entire sentence blocks of varying sizes (1, 2, 5, and 10 sentences).

Mixed Shuffled test. Building upon the standard Shuffle Test, we introduced a more challenging variant called the Mixed Shuffle Test. In this setup, BBScoreV2 of an original (unshuffled) article is compared against BBScoreV2 of shuffled articles drawn from the entire dataset, rather than solely against its own shuffled versions. A robust and general-purpose scoring mechanism should consistently identify the original, unshuffled article as more coherent in these broader comparisons.

Human-AI text discrimination. We leverage the HC3 Q&A dataset (Guo et al., 2023) to train the encoder exclusively on human-generated answers, and subsequently apply it to unseen Q&A pairs generated by both humans and ChatGPT. After deriving the stochastic representations, we compute the BBScoreV2 for each Q&A pair. We evaluate multiple encoder backbones to examine the impact of the raw embeddings. Additionally, we train an encoder on the WikiSection dataset and evaluate it using the Wikipedia subset of HC3. Experiments are conducted under both the full Q&A and answeronly settings to determine if the BBScoreV2 can effectively discriminate between ChatGPT-generated and human-written texts.

## 5 Results

## 5.1 Latent space structure analysis

Theoretically, transformer-based LLMs are argued to map articles to a latent representation that tends to form clusters. The structural properties of these clusters are believed to reflect underlying similarities and properties present in the original articles (Geshkovski et al., 2023).

Experimentally, we also find such clustered property. We first visualized the raw embeddings of each article from the frozen GPT-2 model. In Figure 3 (A.1), these embeddings are projected onto their joint first two principal components (PCs), derived using PCA computed from the latents of unshuffled articles. The color gradient, from light to dark red, represents the token’s sequential position within the article, from beginning to end. Notably, as shown in (A.2) and (A.3) for shuffled versions of an article, the distinct clustering property persists. This persistence, despite the disruption of sequential order, suggests that these raw LM embeddings do not clearly and inherently encode temporal information. To further substantiate this, Figure 3 (B) plots the mean values of the first two PCs for the embeddings of each article, illustrating a tendency for articles from the same dataset to cluster together based on their raw LM embeddings.

Subsequently, to determine the information learned by the MLP layers in our CL encoder, we analyzed its outputted stochastic representations. Figure 3 (C.1) displays these MLP-processed latents projected via PCA. Here, a color gradient from light to dark blue indicates the component’s position within the article’s sequence (from begin to the end). This visualization reveals a clear temporal progression in the latent space for the original, unshuffled article. In stark contrast, Figures (C.2) and (C.3), which depict shuffled versions of the same article, demonstrate that the CL encoder’s representations clearly reflect this violation of temporal order; the clear sequential pattern observed in (C.1) is visibly disrupted. Furthermore, Figure 3 (D) presents the projection of latent trajectories for all articles. This visualization further validates our assertion that the CL encoder effectively learns and represents temporal sequence information, unlike the raw LLM embeddings.

Based on these findings, we show that the CL encoder effectively encodes temporal information into the representation. Furthermore, by evaluating the temporal structure, we can infer properties of the original articles—such as coherence—which are quantified by BBScore<sup>+</sup> and will be systematically discussed in the following sections.

## 5.2 Article coherence evaluation

As shown in Tables 1, we first implement global discrimination tasks on WikiSection. In this task, BBScoreV2 significantly outperforms the BBScore and SOTA results. (See Appendix D for more details on methods we compared to.) The SOTA method, developed using a complex network structure and trained on unshuffle-shuffle data pairs, serves as a robust baseline. Our results demonstrate that BBScoreV2 surpasses the SOTA method in global discrimination tasks with larger block sizes, underscoring its potential to capture more globalized temporal properties.

In shuffle tasks, most current high-performance methods, including the SOTA approach, rely on pairwise training and are unable to effectively compare articles of different lengths, as these models are typically constructed based on sentence-wise matching and comparisons. However, in the Mixed Shuffle test which evaluate the metric robustness across different articles, as shown in Table 1, BB-ScoreV2 surpasses these SOTA method by generating a metric that can be compared across different articles. We use the basic entity-grid method (Barzilay and Lapata, 2005) as a baseline and the result highlights that our score enables article-wise comparison. It also demonstrates significant potential in more complex tasks. Additionally, BBScoreV2 outperforms the BBScore in this article-wise comparison, underscoring a key contribution of our design—mitigating the effect of article length on score evaluation. This property allows for a more general comparison across diverse articles.

We also explore the effect of different LLM backbones. We tested our model using LLaMA3-1B and LLaMA3-3B, with GPT2-124M which is the LLM model used in the main section. As summarized in Table 1, we find: 1) In global shuffle task, LLaMA3-3B outperforms both GPT2-124M and the SOTA method, demonstrating its effectiveness in capturing global sequence structure; 2) In Mixed Shuffle Task, LLaMA3-3B surpasses GPT2-124M for smaller blocks (b=1), but its performance decreases for larger blocks (b=2, b=5, b=10). This suggests a trade-off where larger models excel at capturing local details (b=1) but might sacrifice robustness for global structures (b=10). This insight highlights an intriguing direction for future exploration — different LMs may facilitate learning stochastic representations in task-specific ways.

![](images/175f72a139a948a1a6acb43ff2d2949a333ee2c1df0219eae29e5d4380809f0c.jpg)

![](images/1f80e5409c678f500200db4961d782416dd5dc8ada118c32d89a21f25b14e373.jpg)

![](images/d02d8c403ac037fc07d9a1f56f86019b1d7ae1298262348a35572f33f78888e9.jpg)

![](images/0651ba047d0d355525147f2c9b5b65a579f7629596cb1a004b41432ffeb44a4d.jpg)

![](images/df5e8afbce61ca4c5ced02163ef7f6f8b476f2d9881b43283d13e31b5d05ac3d.jpg)

![](images/89fe0773ef8fc0f993bcd6dcc3f719467aeda098e02b429630a3c04166929805.jpg)

![](images/d222a601aa5f6efa9c67c37146eb22498df33862da0a245286f4e8928851852d.jpg)

![](images/637cbf23f4310ecc6a98ab5e687ab48d5a1d25458eb1e1467ada4e3b65399c61.jpg)

![](images/d49515a04f9a3028f7772bcc89c295306504add3c95acc21afd1b913560de7b3.jpg)

![](images/7223ca10ffc90f742e136778e9b17a78898d7f0a039e952c539b9def1e3277c9.jpg)

![](images/0b10b931eb8ff838d286e839cdb98bfb027bc7548b7713bfaf4e18ccd0e3eaf7.jpg)

![](images/b73b5e529713b12925459b3694772d02feed8b2f1ec18f143fa0d848acf160e2.jpg)  
Figure 3: PCA analysis of raw LM embeddings and CL encoder representations. (A.1) Projection of raw LM embeddings for an unshuffled article onto the first two PCs. The color gradient, from light to dark red, indicates the sequential position of each token within the article. (A.2, A.3) raw LM embeddings for shuffled versions of the same article. (B) Mean PC1 and PC2 values for all articles are plotted, with each article represented by a dot. (C.1) Latent representations from the CL encoder for an unshuffled article, where the color gradient (light to dark blue) signifies the component’s position in the article sequence. (C.2, C.3) Latent patterns observed in shuffled versions of the same article. (D) Visualization of the latent trajectories for all articles.

Moreover, we evaluate the robustness of the encoded stochastic representation on a broader dataset. As shown in Table 2, we train the encoder on WikiText and evaluate it on WikiSection (see Appendix C for details and comparisons about the datasets). The results indicate that our method remains highly robust in this O.O.D. setting, suggesting that the structural and temporal information captured by our model reflects fundamental patterns that generalize across different datasets.

## 5.3 Human-AI discrimination tasks

In this task, we hypothesize that human writing, compared to AI-generated text, displays temporal dynamics and structural patterns similar to those observed in other human-written articles. Specifically, we propose that an encoder trained on a human-written dataset will more accurately capture the characteristics of human writing than those of AI-generated text, resulting in a higher likelihood for human-authored content. As shown in Figure 4, BBScoreV2 consistently outperforms BBScore across all experimental settings. Notably, GPT2 (124M) surpasses the larger backbone models, suggesting that the quality of the learned stochastic representation does not necessarily improve with increased model size.Instead, it is the MLP module that plays the central role in shaping the stochastic representation. Among the models evaluated, GPT2 features a smaller hidden dimension of 768, whereas both LLaMA3 and Qwen3 utilize larger hidden dimensions of 2048. It implies that the hidden dimension of the backbone model may influence performance on this task.

Next, we use a WikiSection trained encoder to detect ChatGPT-generated answer in the Wikipedia subset of HC3. The results are shown in Table 3 under both Q&A and answer-only settings. To further highlight the flexibility and competitive performance of BBScoreV2 compared to LLMbased models, we assessed the perturbation discrepancy metric proposed in DetectGPT (Mitchell et al., 2023), which has high performance in AI detection tasks. Our results reveal that BBScoreV2 surpasses DetectGPT when using a comparable number of model inferences. DetectGPT’s performance is influenced by a hyperparameter—the number of perturbations—which directly affects both the number of model inferences and the computational complexity. As shown in Table 7 in the Appendix, we tested cases with 1 and 10 perturbations. With 1 perturbation, DetectGPT’s accuracy was approximately 64%, lower than BBScoreV2’s 70%, while requiring twice the number of model inferences per text. With 10 perturbations, DetectGPT’s accuracy increased to 84%, but this required 11 model

<table><tr><td rowspan="2">Methods</td><td colspan="4">Acc. (Shuffle Task)</td><td colspan="4">Acc. (Mixed Shuffle Task)</td></tr><tr><td> $\mathcal { D } _ { b = 1 }$ </td><td> $\mathscr { D } _ { b = 2 }$ </td><td> $\mathcal { D } _ { b = 5 }$ </td><td> $\mathcal { D } _ { b = 1 0 }$ </td><td> $\mathcal { D } _ { b = 1 }$ </td><td> $\mathcal { D } _ { b = 2 }$ </td><td> $\mathcal { D } _ { b = 5 }$ </td><td> $\mathcal { D } _ { b = 1 0 }$ </td></tr><tr><td>ENTITY GRID (Barzilay and Lapata, 2005)</td><td>85.73</td><td>82.79</td><td>75.81</td><td>64.65</td><td>46.10</td><td>52.29</td><td>53.69</td><td>63.02</td></tr><tr><td>UNIFIED COHERENCE (Moon et al., 2019)</td><td>99.73</td><td>97.86</td><td>96.90</td><td>96.09</td><td></td><td>一</td><td>一</td><td>一</td></tr><tr><td>BBSCORE (Sheng et al., 2024)</td><td>83.39</td><td>80.71</td><td>79.36</td><td>78.66</td><td>22.37</td><td>24.94</td><td>23.84</td><td>19.69</td></tr><tr><td>BBSCOREV2 (GPT2-124M)</td><td>99.03</td><td>98.11</td><td>98.02</td><td>98.17</td><td>94.78</td><td>89.24</td><td>79.64</td><td>70.83</td></tr><tr><td>BBSCOREV2 (LLAMA3-1B)</td><td>99.16</td><td>98.37</td><td>97.99</td><td>97.87</td><td>94.53</td><td>87.86</td><td>76.95</td><td>71.13</td></tr><tr><td>BBSCOREV2 (LLAMA3-3B)</td><td>99.57</td><td>98.74</td><td>98.14</td><td>98.74</td><td>94.97</td><td>86.34</td><td>73.88</td><td>68.87</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 1: Results of Global shuffle tasks on WikiSection. $\mathcal { D } _ { b = i } , i = 1 , 2 , 5 ,$ 10 refers to datasets constructed with varying levels of block shuffling.
<table><tr><td>Methods</td><td colspan="4">Shuffle Test tasks  $\mathbf { ( 0 . 0 . 0 . 0 . ) }$ </td></tr><tr><td></td><td> $\mathcal { D } _ { b = 1 }$ </td><td> $\mathcal { D } _ { b = 2 }$ </td><td> $\mathcal { D } _ { b = 5 }$ </td><td> $\mathcal { D } _ { b = 1 0 }$ </td></tr><tr><td>UNIFIED COHERENCE</td><td>60.02</td><td>9.63</td><td>44.80</td><td>66.51</td></tr><tr><td>BBSCORE</td><td>70.32</td><td>72.09</td><td>76.84</td><td>77.73</td></tr><tr><td>BBSCOREV2</td><td>91.30</td><td>87.22</td><td>86.14</td><td>88.18</td></tr></table>

Table 2: O.O.D. Task. Encoder was trained on the WikiText and evaluated on Shuffle Test tasks using the same WikiSection data to assess their performance.

![](images/c4148bf4f1d38e5747ad8e5e610ce77873b6968dffe596af398ec6bdb766034b.jpg)  
Figure 4: Compare different LM backbones.

inferences per text, making it significantly more computationally intensive than BBScoreV2.
<table><tr><td>Methods</td><td>HC3 (w/o Q&amp;A)</td><td>HC3 (w/ Q&amp;A)</td></tr><tr><td>BBSCORE</td><td>37.53</td><td>31.47</td></tr><tr><td>DETECTGPT</td><td>64.30</td><td>63.30</td></tr><tr><td>BBSCOREV2</td><td>70.67</td><td>69.71</td></tr></table>

Table 3: Accuracy of the Human-AI discrimination task.

## 5.4 Ablation analysis on CL encoder

As previously discussed, the CL encoder relies on a critical assumption of the independence and homogeneity among the dimensions of the encoded sequence which is $\Sigma = \mathbf { I } _ { d } .$ . To further examine this assumption, we employ two alternative methods:

1) A likelihood-based encoder, SP Encoder (see Appendix A) whose loss function is defined based on the likelihood of the Brownian bridge:

$$
\begin{array} { r l } & { L _ { \mathrm { N L L } } = \displaystyle \sum _ { j = 1 } ^ { m } \sum _ { i = 1 } ^ { n _ { j } } ( T _ { i } - 1 ) \log ( | \Sigma _ { j } | ) } \\ & { + \displaystyle \sum _ { j = 1 } ^ { m } \sum _ { i = 1 } ^ { n _ { j } } \mathrm { t r } ( \Sigma _ { j } ^ { - 1 } ( \mathbf { s } _ { i } ^ { \theta } - { \pmb \mu } _ { i } ^ { \theta } ) \Sigma _ { T _ { i } } ^ { - 1 } ( \mathbf { s } _ { i } ^ { \theta } - { \pmb \mu } _ { i } ^ { \theta } ) ^ { \top } ) . } \end{array}\tag{6}
$$

2) A contrastive loss-based encoder, whose loss function is AnInfoNCE Rusak et al. (2024) which is capable of learning Σ during training The loss function $L _ { \mathrm { A n I n f o N C E } }$ is defined with the same formate as CL loss (2), however, we replace the metric $d ( \mathbf { x } _ { 0 } , \mathbf { x } _ { t } , \mathbf { x } _ { T } )$ by a trainable metric $d ^ { * } \big ( \mathbf { x } _ { 0 } , \mathbf { x } _ { t } , \mathbf { x } _ { T } ; f _ { \theta } \big )$ defined as:

$$
\begin{array} { r l } & { d ^ { * } \big ( \mathbf { x } _ { 0 } , \mathbf { x } _ { t } , \mathbf { x } _ { T } ; f _ { \theta } \big ) } \\ & { = - \frac { \| f _ { \theta } \big ( \mathbf { x } _ { t } \big ) - \frac { T - t } { T } f _ { \theta } \big ( \mathbf { x } _ { 0 } \big ) - \frac { t } { T } f _ { \theta } \big ( \mathbf { x } _ { T } \big ) \| _ { \hat { \boldsymbol { \Lambda } } } ^ { 2 } } { 2 t ( T - t ) / T } . } \end{array}
$$

and $\hat { \Lambda }$ is a trainable diagonal scaling matrix. Let

$\begin{array} { r } { \mathbf { v } = f _ { \theta } ( \mathbf { x } _ { t } ) - \frac { T - t } { T } f _ { \theta } ( \mathbf { x } _ { 0 } ) - \frac { t } { T } f _ { \theta } ( \mathbf { x } _ { T } ) } \end{array}$ , then its corresponding norm is defined as:

$$
\| \mathbf { v } \| _ { \hat { \Lambda } } ^ { 2 } = \mathbf { v } ^ { \mathrm { T } } \cdot \hat { \boldsymbol { \Lambda } } \cdot \mathbf { v }
$$

As shown in Table 4, neither the likelihoodbased nor the CL encoder with AnInfoNCE loss yields significant improvements in the shuffle test. This suggests that the MLP layers do not capture meaningful structural correlations across latent dimensions—a phenomenon also noted by Wang et al. (2022)—and instead primarily reconstruct temporal information, which validate our assumptions on CL encoder training. The lack of performance improvement may also suggest that the pertinent correlation structure is likely inherent either within the statistical properties of the article domain or already captured within the high-dimensional embedding space of the pre-trained language model as we seen in the cluster analysis in Fig 3.

<table><tr><td>Loss Type</td><td> $\mathcal { D } _ { b = 1 }$ </td><td> $\mathcal { D } _ { b = 2 }$ </td><td> $\mathcal { D } _ { b = 5 }$ </td><td> $\mathcal { D } _ { b = 1 0 }$ </td></tr><tr><td>ANINFONCE</td><td>94.63</td><td>91.05</td><td>92.13</td><td>91.70</td></tr><tr><td>LIKELIHOOD</td><td>94.42</td><td>92.90</td><td>90.77</td><td>86.69</td></tr><tr><td>OURS</td><td>99.03</td><td>98.11</td><td>98.02</td><td>98.17</td></tr></table>

Table 4: Comparison of model performance with different loss function on WikiSection Dataset.

## 5.5 Computation efficiency analysis

We specifically analyze the computation efficiency of BBScoreV2, as shown in Figure 5, the y-axis represents computation time, while the x-axis indicates article length. The theoretical computational complexity of BBScoreV2 is $O ( T ^ { 2 } )$ , primarily due to matrix multiplications inherent in its definition. This complexity is fundamental to fully leveraging temporal information for sequence evaluation. Empirically, the observed computation time is slightly better than the theoretical prediction, thanks to the computational acceleration. These results demonstrate that BBScoreV2 is not only feasible for realtime applications but also retains its robust evaluation capabilities.

## 6 Conclusion

In this paper, we present both a theoretical and empirical investigation into the structural and temporal properties encoded in stochastic representations of latent trajectories for NLP tasks. We analyze and visualize these properties, and introduce BBScoreV2—a novel, length-invariant metric designed to quantify such information. First we present the learned representations recovers the time dependency of the input sequences. Then validated through shuffled and mixed-shuffle tests, we show that BBScoreV2 exhibits strong performance in capturing temporal structure and generalizes effectively to out-of-distribution tasks, suggesting that these properties reflect domain-independent textual signals. Moreover, BBScoreV2 shows promising capability in distinguishing humanwritten from AI-generated text by leveraging encoded structural and temporal features.

![](images/12949b650d972fa17f3c03c8ad92cb7d9d82e305056132e7e1a270b6a62f2fac.jpg)  
Figure 5: The computation time of BBScoreV2 for different article lengths. It reveals a quadratic relationship (experimentally 1.57, theoretically 2) between article length and computation time, with each article processed in approximately $\sim 1 0 ^ { - 3 }$ seconds.

Looking ahead, we aim to extend BBScoreV2 to multi-domain tasks such as domain identification, and to exploit its length-insensitive nature to develop generative models that maintain semantic coherence across varying sequence lengths. Its computational efficiency (see Fig. 5) also makes it suitable for large-scale applications. Finally, inspired by Albergo et al. (2023); Albergo and Vanden-Eijnden (2023), we plan to explore more expressive bridge processes to further enhance the representational capacity of the latent space and enable richer downstream analysis and generation.

## 7 Limitations

Our current study is constrained by limited computational resources and the lack of human-annotated data, which prevents us from evaluating BB-ScoreV2 against human preference—a key limitation in assessing its alignment with human judgment. Additionally, in the Human-AI discrimination task, we were unable to evaluate it on a broader range of datasets or conduct more extensive comparisons across more baselines. These limitations suggest directions for future work involving largescale human evaluation and broader benchmarking.

## References

Michael S Albergo, Nicholas M Boffi, and Eric Vanden-Eijnden. 2023. Stochastic interpolants: A unifying framework for flows and diffusions. arXiv preprint arXiv:2303.08797.

Michael Samuel Albergo and Eric Vanden-Eijnden. 2023. Building Normalizing Flows with Stochastic Interpolants. In The Eleventh International Conference on Learning Representations.

Sebastian Arnold, Benjamin Schrauwen, Verena Rieser, and Katja Filippova. 2019. SECTOR: A Neural Model for Coherent Topic Segmentation and Classification. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 241–253, Hong Kong, China. Association for Computational Linguistics.

Regina Barzilay and Mirella Lapata. 2005. Modeling Local Coherence: An Entity-Based Approach. In Proceedings of the 43rd Annual Meeting of the Association for Computational Linguistics (ACL’05), pages 141–148, Ann Arbor, Michigan. Association for Computational Linguistics.

Samuel R. Bowman, Luke Vilnis, Oriol Vinyals, Andrew Dai, Rafal Jozefowicz, and Samy Bengio. 2016. Generating Sentences from a Continuous Space. In Proceedings of the 20th SIGNLL Conference on Computational Natural Language Learning, pages 10–21, Berlin, Germany. Association for Computational Linguistics.

Yuntian Deng, Volodymyr Kuleshov, and Alexander Rush. 2022. Model Criticism for Long-Form Text Generation. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pages 11887–11912, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Conor Durkan, Iain Murray, and George Papamakarios. 2020. On Contrastive Learning for Likelihood-free Inference. In Proceedings ofthe 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 2771–2781. PMLR.

Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple Contrastive Learning of Sentence Embeddings. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, pages 6894–6910, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.

Borjan Geshkovski, Cyril Letrouit, Yury Polyanskiy, and Philippe Rigollet. 2023. The emergence of clusters in self-attention dynamics. In Advances in Neural Information Processing Systems, volume 36, pages 57026–57037. Curran Associates, Inc.

Biyang Guo, Xin Zhang, Ziyuan Wang, Minqi Jiang, Jinran Nie, Yuxuan Ding, Jianwei Yue, and Yupeng Wu. 2023. How close is chatgpt to human experts?

comparison corpus, evaluation, and detection. arXiv preprint arXiv:2301.07597.

Jon S Horne, Edward O Garton, Stephen M Krone, and Jesse S Lewis. 2007. Analyzing animal movements using Brownian bridges. Ecology, 88(9):2354–2363.

Sungho Jeon and Michael Strube. 2022. Entity-based Neural Local Coherence Modeling. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7787–7805, Dublin, Ireland. Association for Computational Linguistics.

Shafiq Joty, Muhammad Tasnim Mohiuddin, and Dat Tien Nguyen. 2018. Coherence Modeling of Asynchronous Conversations: A Neural Entity Grid Approach. In Proceedings ofthe 56th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 558–568, Melbourne, Australia. Association for Computational Linguistics.

Alice Lai and Joel Tetreault. 2018. Discourse Coherence in the Wild: A Dataset, Evaluation and Methods. In Proceedings of the 19th Annual SIGdial Meeting on Discourse and Dialogue, pages 214–223, Melbourne, Australia. Association for Computational Linguistics.

Bingbin Liu, Pradeep Ravikumar, and Andrej Risteski. 2021. Contrastive learning of strong-mixing continuous-time stochastic processes. In Proceedings ofThe 24th International Conference on Artificial Intelligence and Statistics, volume 130 of Proceedings ofMachine Learning Research, pages 3151– 3159. PMLR.

Aviya Maimon and Reut Tsarfaty. 2023. A Novel Computational and Modeling Foundation for Automatic Coherence Assessment. arXiv e-prints, arXiv:2310.00598.

Emile Mathieu, Adam Foster, and Yee Teh. 2021. On Contrastive Representations of Stochastic Processes. In Advances in Neural Information Processing Systems, volume 34, pages 28823–28835. Curran Associates, Inc.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016. Pointer Sentinel Mixture Models. Preprint, arXiv:1609.07843.

Eric Mitchell, Yoonho Lee, Alexander Khazatsky, Christopher D Manning, and Chelsea Finn. 2023. Detectgpt: Zero-shot machine-generated text detection using probability curvature. In International Conference on Machine Learning, pages 24950–24962. PMLR.

Han Cheol Moon, Tasnim Mohiuddin, Shafiq Joty, and Chi Xu. 2019. A Unified Neural Coherence Model. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2262– 2272, Hong Kong, China. Association for Computational Linguistics.

Bernt Øksendal and Bernt Øksendal. 2003. Stochastic differential equations. Springer.

Tanya Reinhart. 1980. Conditions for Text Coherence. Poetics Today, 1(4):161–180.

Evgenia Rusak, Patrik Reizinger, Attila Juhos, Oliver Bringmann, Roland S. Zimmermann, and Wieland Brendel. 2024. Infonce: Identifying the gap between theory and practice. Preprint, arXiv:2407.00143.

Zhecheng Sheng, Tianhao Zhang, Chen Jiang, and Dongyeop Kang. 2024. BBScore: A Brownian Bridge Based Metric for Assessing Text Coherence. Proceedings of the AAAI Conference on Artificial Intelligence, 38(13):14937–14945.

Aäron van den Oord, Yazhe Li, and Oriol Vinyals. 2018. Representation Learning with Contrastive Predictive Coding. CoRR, abs/1807.03748.

Rose E Wang, Esin Durmus, Noah Goodman, and Tatsunori Hashimoto. 2022. Language modeling via stochastic processes. In International Conference on Learning Representations.

Ling Yang, Zhilong Zhang, Yang Song, Shenda Hong, Runsheng Xu, Yue Zhao, Wentao Zhang, Bin Cui, and Ming-Hsuan Yang. 2023. Diffusion Models: A Comprehensive Survey of Methods and Applications. ACM Comput. Surv., 56(4).

Qinghua Yi, Xiaoyu Chen, Chen Zhang, Zhen Zhou, Ling Zhu, and Xin Kong. 2024. Diffusion models in text generation: a survey. PeerJ Computer Science, 10:e1905.

Heng Zhang, Daqing Liu, Qi Zheng, and Bing Su. 2023. Modeling Video As Stochastic Processes for Fine-Grained Video Representation Learning. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2225– 2234.

Hao Zou, Zae Myung Kim, and Dongyeop Kang. 2023. A Survey of Diffusion Models in Natural Language Processing. Preprint, arXiv:2305.14671.

## A Appendix: SP Encoder

## A.1 Definition

Consider a multi-domain problem with m domains $\mathcal { X } _ { 1 } , \mathcal { X } _ { 2 } , \ldots , \mathcal { X } _ { m }$ each associated with domainspecific true structural parameters $\Sigma _ { 1 } , \Sigma _ { 2 } , \dots , \Sigma _ { m } ,$ respectively. For each domain $\chi _ { j }$ , we have $n _ { j }$ independent raw inputs $\mathbf { x } _ { j 1 } , \dotsc , \mathbf { x } _ { j n _ { j } }$ We define the encoded sequences as $\bar { \mathbf { s } } _ { j i } ^ { \theta } = \dot { f } _ { \theta } ( \mathbf { x } _ { j i } )$ for $j = 1 , \dots , m$ and $i = 1 , \dotsc , n _ { j }$ , where $f _ { \theta }$ is the encoder parameterized by θ. When the encoder parameters reach their optimal values $\theta ^ { * }$ , the sequences $[ \bar { \bf s } _ { j i } ^ { \theta ^ { * } } ] _ { i = 1 } ^ { n _ { j } }$ are expected to be i.i.d. samples from BBs with parameters $\Sigma _ { j }$ for each domain $\mathcal { X } _ { j }$

We employ the negative log-likelihood (NLL) as the loss function to train the encoder. According to Proposition 1, for each $\theta ,$ the negative log-likelihood for domain $\chi _ { j }$ depends on $\Sigma _ { j }$ and the inputs $[ { \bf x } _ { j i } ] _ { i = 1 } ^ { n _ { j } }$ through the expression $\begin{array} { r } { \sum _ { i = 1 } ^ { n _ { j } } ( T _ { i } - 1 ) \log \bar { ( | \Sigma _ { j } | ) } + \sum _ { i = 1 } ^ { n _ { j } } \mathrm { t r } ( \Sigma _ { j } ^ { - 1 } \bar { ( \mathbf { s } _ { i } ^ { \theta } - } } \end{array}$ $\pmb { \mu } _ { i } ^ { \theta } ) \Sigma _ { T _ { i } } ^ { - 1 } ( \mathbf { s } _ { i } ^ { \theta } - \pmb { \mu } _ { i } ^ { \theta } ) ^ { \top } )$ . We consider the following training process.

Batch Processing: We divide the inputs $[ \mathbf { x } _ { j i } ] _ { i = 1 } ^ { n _ { j } }$ into several batches. For each batch $B { \mathrm { , } }$ we compute the batch loss using the current estimate $\widehat { \Sigma } _ { j }$ of $\begin{array} { r } { \Sigma _ { j } \colon \sum _ { i \in \mathcal { B } } \mathrm { t r } ( \widehat { \Sigma } _ { j } ^ { - 1 } ( \mathbf { s } _ { i } ^ { \theta } - { \pmb { \mu } } _ { i } ^ { \theta } ) { \Sigma } _ { T _ { i } } ^ { - 1 } ( \mathbf { s } _ { i } ^ { \theta } - { \pmb { \mu } } _ { i } ^ { \theta } ) ^ { \top } ) } \end{array}$ b. This bloss function measures how well the encoded sequences fit the assumed BB model with the current structural parameter estimate.

Handling Large Sequences: When the sequence lengths $T _ { i }$ are large, computing the full loss can be computationally intensive. To address this, we randomly sample a triplet of time points $t = ( t _ { 1 } , t _ { 2 } , t _ { 3 } )$ with $1 \leq t _ { 1 } < t _ { 2 } < t _ { 3 } \leq T _ { i } - 1$ We extract the corresponding sub-matrices $[ \mathbf { s } _ { i } ^ { \theta } ] _ { t }$ and $[ \mu _ { i } ^ { \theta } ] _ { \ i }$ of size d $l \times 3$ from $\mathbf { s } _ { i } ^ { \theta }$ and $\mu _ { i } ^ { \theta } .$ , respectively. Let $[ \Sigma _ { T _ { i } } ] _ { i }$ <sub>t</sub> be the $3 \times 3$ sub-matrix of $\Sigma _ { T _ { i } }$ corresponding to the selected time points. The loss for each i in the batch becomes $\mathrm { \hat { t r } } ( \widehat { \Sigma } _ { j } ^ { - 1 } ( [ \mathbf { s } _ { i } ^ { \theta } ] _ { t }$ $[ \pmb { \mu } _ { i } ^ { \theta } ] _ { t } ) [ \Sigma _ { T _ { i } } ] _ { t } ^ { - 1 } ( [ \mathbf { s } _ { i } ^ { \theta } ] _ { t } - [ \pmb { \mu } _ { i } ^ { \theta } ] _ { t } ) ^ { \top } )$ b. This approach reduces computational complexity while still capturing temporal dependencies at selected time points.

Updating Structural Parameters: After processing all batches for $\chi _ { j }$ , we update the estimate of $\Sigma _ { j }$ using the MLE: $\begin{array} { r } { \widehat { \Sigma } _ { j } \ = \ [ \sum _ { i = 1 } ^ { n _ { j } } ( T _ { i } \ - } \end{array}$ $\begin{array} { r } { 1 ) ] ^ { - 1 } [ \sum _ { i = 1 } ^ { { n _ { j } } ^ { - } } ( \mathbf s _ { i } ^ { \theta } - { \pmb \mu } _ { i } ^ { \theta } ) \Sigma _ { T _ { i } } ^ { - 1 } ( \mathbf s _ { i } ^ { \theta } - { \pmb \mu } _ { i } ^ { \theta } ) ^ { \top } ] } \end{array}$ . This update aggregates information from all sequences in the domain to refine the structural parameter estimate.

Regularization for Stability: To stabilize the training process, we regularize $\widehat { \Sigma } _ { j }$ by blendbing it with a scaled identity matrix. We compute the average variance ${ \widehat { \sigma } } _ { j } ^ { 2 }$ and update $\widehat { \Sigma } _ { j }$ b bas follows, using a small regularization parameter $\begin{array} { r c l c r c l } { \epsilon } & { > } & { 0 \colon } & { \widehat { \Sigma } _ { j } } & { = } & { ( 1 \ - \ \epsilon ) [ \sum _ { i = 1 } ^ { n _ { j } } ( T _ { i } \ - } \end{array}$ $1 ) ] ^ { - 1 } [ \sum _ { i = 1 } ^ { n _ { j } } ( \mathbf { s } _ { i } ^ { \theta } - { \bar { \mu _ { i } ^ { \theta } } } ) \Sigma _ { T _ { i } } ^ { - 1 } ( \mathbf { s } _ { i } ^ { \theta } - { \bar { \mu _ { i } ^ { \theta } } } ) ^ { \top } ] ^ { - } + \epsilon { \widehat { \sigma } } _ { j } ^ { 2 } \mathbf { I } _ { d }$ with $\begin{array} { r l r } { \widehat \sigma _ { j } ^ { 2 } } & { { } = } & { [ \sum _ { i = 1 } ^ { n _ { j } } ( T _ { i } - 1 ) d ] ^ { - 1 } [ \sum _ { i = 1 } ^ { n _ { j } } \mathrm { t r } ( ( { \bf s } _ { i } ^ { \widetilde { \theta } } - 1 ) \mathrm {  ~ \cdot ~ } } \end{array}$ $\pmb { \mu } _ { i } ^ { \theta } ) \Sigma _ { T _ { i } } ^ { - 1 } ( \mathbf { s } _ { i } ^ { \theta } - \pmb { \mu } _ { i } ^ { \theta } ) ^ { \top } ) ]$ . This regularization shifts $\widehat { \Sigma } _ { j }$ slightly towards isotropy, improving numerical bstability during optimization.

Total Empirical Loss Function: After iterating over all domains, the total empirical loss function

becomes

$$
\begin{array} { r l r } {  { L _ { \mathrm { N L L } } = \sum _ { j = 1 } ^ { m } \sum _ { i = 1 } ^ { n _ { j } } ( T _ { i } - 1 ) \log ( | \Sigma _ { j } | ) } } \\ & { } & { + \sum _ { j = 1 } ^ { m } \sum _ { i = 1 } ^ { n _ { j } } \mathrm { t r } \Big ( \Sigma _ { j } ^ { - 1 } ( \mathbf { s } _ { i } ^ { \theta } - \mu _ { i } ^ { \theta } ) } \\ & { } & { \cdot \Sigma _ { T _ { i } } ^ { - 1 } ( \mathbf { s } _ { i } ^ { \theta } - \mu _ { i } ^ { \theta } ) ^ { \top } \Big ) . } \end{array}
$$

Minimizing this loss over $\theta$ encourages the encoder to produce sequences that align with the assumed stochastic process model across all domains.

## A.2 Training Details

The WikiSection SP Encoder was trained on 1 A100 GPU for about 10 hours using the training set of WikiSection for 100 epochs. We used SGD optimizer and set the learning rate to be $I e { \cdot } 9 .$ . The ϵ in the loss function $L _ { \mathrm { { N L L } } }$ is chosen as $I e \mathrm { - } 7 .$ The WikiText SP Encoder was trained on 4 A100 GPUs for roughly 20 hours for 4 epochs with WikiText dataset. For this dataset, we trained with AdamW optimizer with learning rate $\mathit { l e } { \cdot } 9$ and batch size 32. The ϵ in the loss function $L _ { \mathrm { { N L L } } }$ is chosen as $1 e { \mathrm { - } } 3$ . Other hyperparameters can be accessed from the configuration file in the submitted code. Our empirical results show incorporating $\hat { \sigma } _ { j }$ into the $\widehat { \Sigma } _ { j }$ makes no significant results in the downstream btasks, thus we disregard $\hat { \sigma } _ { j }$ during encoder training.

## A.3 Hyper-parameter Tuning

While training the SP Encoder, we experimented with different ϵ in $L _ { \mathrm { { N L L } } }$ to see its impact on the performance of the trained encoder. Note that ϵ determines the perturbation added to the matrix $\widehat { \Sigma } .$ The eigenvalues of the initial $\widehat { \Sigma }$ range from $1 0 ^ { - 6 } ~ \mathrm { t o } ~ 1 0 ^ { - 1 }$ b, with the majority of which lying in $[ 1 0 ^ { - 3 } , 1 0 ^ { - 5 } ]$ . Thus we tested the following three different ϵ:

• Large $\epsilon = 1 0 ^ { - 3 }$ that is larger that most eigenvalues of $\widehat { \Sigma }$

• Medium $\epsilon = 1 0 ^ { - 5 }$ that is about the same scale of most eigenvalues of $\widehat { \Sigma }$

• Small $\epsilon = 1 0 ^ { - 7 }$ that is smaller than most eigenvalues of $\widehat { \Sigma }$

We choose the small ϵ based on the performance.

## B Proof

## B.1 Proof of Proposition 1

Proof. We fix the start and end points $s _ { 0 }$ and $s T$ and calculate the likelihood function of the input sequence s.

Given that $s _ { t } - \mu _ { t } = \mathbf { W } ( B _ { 1 } ( t ) , \ldots , B _ { d } ( t ) ) ^ { \top }$ and considering the independence of $B _ { 1 } ( t ) , \ldots , B _ { d } ( t )$ along with the properties of the standard BB, we have for any $t , t ^ { \prime } \in \{ 1 , 2 , \ldots , T - 1 \} : \mathrm { { \mathbb { E } } } [ s _ { t } - \mu _ { t } ] = 0 ,$ $\mathrm { V a r } [ s _ { t } ] = [ \Sigma _ { T } ] _ { t , t } \Sigma$ and $\mathrm { C o v } [ s _ { t } , s _ { t ^ { \prime } } ] = [ \Sigma _ { T } ] _ { t , t ^ { \prime } } \Sigma$ Therefore, the vectorized form of $\mathbf { s } - \pmb { \mu }$ follows a multivariate normal distribution:

$$
\mathrm { v e c } ( \mathbf { s } - { \pmb \mu } ) \sim N ( 0 , \Sigma _ { T } \otimes { \Sigma } ) ,
$$

where $\mathrm { v e c } ( \cdot )$ denotes vectorization and $\otimes$ represents the Kronecker product.

Using the likelihood function of the multivariate normal distribution, we have:

$$
\begin{array} { r l } & { \quad L ( \Sigma | \bar { \mathbf { s } } ) = ( 2 \pi ) ^ { - d ( T - 1 ) / 2 } | \Sigma _ { T } \otimes \Sigma | ^ { - 1 / 2 } } \\ & { \cdot \exp \left[ - \frac { 1 } { 2 } \mathrm { v e c } ( \mathbf { s } - \pmb { \mu } ) ^ { \top } [ \Sigma _ { T } \otimes \Sigma ] ^ { - 1 } \mathrm { v e c } ( \mathbf { s } - \pmb { \mu } ) \right] } \end{array}
$$

Using properties of the Kronecker product, we have $| \Sigma _ { T } \otimes \bar { \Sigma ^ { | } } = | \Sigma _ { T } | ^ { d } | \Sigma | ^ { T - 1 }$ and then

$$
\begin{array} { r l } & { \mathrm { v e c } ( \mathbf { s } - { \boldsymbol { \mu } } ) ^ { \top } [ { \boldsymbol { \Sigma } } _ { T } \otimes { \boldsymbol { \Sigma } } ] ^ { - 1 } \mathrm { v e c } ( \mathbf { s } - { \boldsymbol { \mu } } ) } \\ & { ~ = \mathrm { v e c } ( \mathbf { s } - { \boldsymbol { \mu } } ) ^ { \top } [ { \boldsymbol { \Sigma } } _ { T } ^ { - 1 } \otimes { \boldsymbol { \Sigma } } ^ { - 1 } ] \mathrm { v e c } ( \mathbf { s } - { \boldsymbol { \mu } } ) } \\ & { ~ = \mathrm { v e c } ( \mathbf { s } - { \boldsymbol { \mu } } ) ^ { \top } \mathrm { v e c } ( { \boldsymbol { \Sigma } } ^ { - 1 } ( \mathbf { s } - { \boldsymbol { \mu } } ) { \boldsymbol { \Sigma } } _ { T } ^ { - 1 } ) } \\ & { ~ = \mathrm { t r } ( ( \mathbf { s } - { \boldsymbol { \mu } } ) ^ { \top } { \boldsymbol { \Sigma } } ^ { - 1 } ( \mathbf { s } - { \boldsymbol { \mu } } ) { \boldsymbol { \Sigma } } _ { T } ^ { - 1 } ) } \\ & { ~ = \mathrm { t r } ( { \boldsymbol { \Sigma } } ^ { - 1 } ( \mathbf { s } - { \boldsymbol { \mu } } ) { \boldsymbol { \Sigma } } _ { T } ^ { - 1 } ( \mathbf { s } - { \boldsymbol { \mu } } ) ^ { \top } ) . } \end{array}
$$

Therefore, the likelihood function becomes:

$$
\begin{array} { r } { L ( \Sigma | \bar { \mathbf { s } } ) = ( 2 \pi ) ^ { - d ( T - 1 ) / 2 } | \Sigma _ { T } | ^ { - d / 2 } | \Sigma | ^ { - ( T - 1 ) / 2 } } \\ { \cdot \exp [ - \mathrm { t r } ( \Sigma ^ { - 1 } ( \mathbf { s } - \pmb { \mu } ) \Sigma _ { T } ^ { - 1 } ( \mathbf { s } - \pmb { \mu } ) ^ { \top } ) / 2 ] . } \end{array}
$$

Taking the logarithm, the log-likelihood function is:

$$
\begin{array} { r l } & { \ell ( { \Sigma } | \bar { \mathbf { s } } ) = - \frac { d ( T - 1 ) } { 2 } \log ( 2 \pi ) - \frac { d } { 2 } \log \left| { \Sigma } _ { T } \right| } \\ & { \phantom { { \Sigma } } - \frac { ( T - 1 ) } { 2 } \log \left| { \Sigma } \right| } \\ & { \phantom { { \Sigma } } - \frac { 1 } { 2 } \mathrm { t r } \Big ( { \Sigma } ^ { - 1 } ( { \mathbf { s } } - { \pmb \mu } ) { \Sigma } _ { T } ^ { - 1 } ( { \mathbf { s } } - { \pmb \mu } ) ^ { \top } \Big ) . } \end{array}
$$

For n independent input sequences $\bar { \bf s } _ { 1 } , \ldots , \bar { \bf s } _ { n }$ with lengths $T _ { 1 } + 1 , . . . , T _ { n } + 1$ , generated by the same $\Sigma ,$ the total likelihood is:

$$
L ( \Sigma | \{ \mathbf { s } _ { i } \} _ { i = 1 } ^ { n } ) = \Pi _ { i = 1 } ^ { n } L ( \Sigma | \mathbf { s } _ { i } ) .
$$

Then the total log-likelihood function is

$$
\begin{array} { r l } { \ell ( \Sigma | \{ \mathbf { s } _ { i } \} _ { n = 1 } ^ { n } ) = \displaystyle \sum _ { i = 1 } ^ { n } \ell ( \Sigma | \mathbf { s } _ { i } ) } \\ & { = - \frac { d \sum _ { i = 1 } ^ { n } ( T _ { i } - 1 ) } { 2 } \log ( 2 \pi ) } \\ & { \quad \quad \quad - \frac { d } { 2 } \displaystyle \sum _ { i = 1 } ^ { n } \log ( | \Sigma _ { T } | ) } \\ & { \quad \quad \quad - \displaystyle \sum _ { i = 1 } ^ { n - 1 } ( T _ { i } - 1 ) \log ( | \Sigma | ) } \\ & { \quad \quad \quad - \frac { 1 } { 2 } \displaystyle \sum _ { i = 1 } ^ { n } \mathrm { f o r } \left( \Sigma ^ { - 1 } ( \mathbf { s } _ { i } - \mu _ { i } ) \right. } \\ & { \quad \quad \quad \quad \left. - \frac { 1 } { 2 } \displaystyle \sum _ { i = 1 } ^ { n } ( s _ { i } - \mu _ { i } ) ^ { \top } \right) . } \end{array}
$$

## B.2 Proof of Proposition 2

Proof. To find the MLE of Σ, we need to minimize the negative log-likelihood function, which is equivalent to minimizing:

$$
\begin{array} { c } { { g ( \Sigma ) = \displaystyle \sum _ { i = 1 } ^ { n } ( T _ { i } - 1 ) \log | \Sigma | } } \\ { { + \displaystyle \sum _ { i = 1 } ^ { n } \mathrm { t r } \Big ( \Sigma ^ { - 1 } ( { \bf s } _ { i } - { \pmb \mu } _ { i } ) \Sigma _ { T _ { i } } ^ { - 1 } ( { \bf s } _ { i } - { \pmb \mu } _ { i } ) ^ { \top } \Big ) } } \end{array}
$$

Since $\begin{array} { r } { \Sigma = \mathbf { W } \mathbf { W } ^ { \top } } \end{array}$ is positive definite, we can compute the gradient of $g ( \Sigma )$ with respect to $\Sigma$ Note that:

$$
\begin{array} { r l } & { \frac { \mathrm { d } } { \mathrm { d } \Sigma } \log \left| \Sigma \right| = \Sigma ^ { - 1 } , } \\ & { \frac { \mathrm { d } } { \mathrm { d } \Sigma } \mathrm { t r } \Bigl ( \Sigma ^ { - 1 } ( { \bf s } _ { i } - { \pmb \mu } _ { i } ) \cdot \Sigma _ { T _ { i } } ^ { - 1 } ( { \bf s } _ { i } - { \pmb \mu } _ { i } ) ^ { \top } \Bigr ) } \\ & { = - \Sigma ^ { - 1 } ( { \bf s } _ { i } - { \pmb \mu } _ { i } ) \Sigma _ { T _ { i } } ^ { - 1 } \cdot ( { \bf s } _ { i } - { \pmb \mu } _ { i } ) ^ { \top } \Sigma ^ { - 1 } . } \end{array}
$$

We compute the gradient:

$$
\begin{array} { l } { \displaystyle \frac { \mathrm { d } } { \mathrm { d } \Sigma } g ( \Sigma ) = \Big ( \sum _ { i = 1 } ^ { n } ( T _ { i } - 1 ) \Big ) \Sigma ^ { - 1 } } \\ { - \Sigma ^ { - 1 } \Big ( \sum _ { i = 1 } ^ { n } ( \mathbf s _ { i } - \pmb \mu _ { i } ) \Sigma _ { T _ { i } } ^ { - 1 } ( \mathbf s _ { i } - \pmb \mu _ { i } ) ^ { \top } \Big ) } \\ { ~ \cdot \Sigma ^ { - 1 } . } \end{array}
$$

Setting the gradient to zero for minimization, we have:

$$
\begin{array} { c } { { \widehat { \Sigma } = \displaystyle \left( \sum _ { i = 1 } ^ { n } ( T _ { i } - 1 ) \right) ^ { - 1 } } } \\ { { \displaystyle \cdot \left( \sum _ { i = 1 } ^ { n } ( \mathbf { s } _ { i } - { \boldsymbol { \mu } } _ { i } ) \Sigma _ { T _ { i } } ^ { - 1 } ( \mathbf { s } _ { i } - { \boldsymbol { \mu } } _ { i } ) ^ { \top } \right) } } \end{array}
$$

As shown, the MLE estimate for Σ is obtained.

## C Datasets

WikiSection: We use dataset introduced in (Arnold et al., 2019) which contains selected Wikipedia articles on the topic of global cities and have clear topic structures. Each article in this collection follows a pattern certain sections such as abstract, history, geographics and demographics. The training split contains 2165 articles and the test split has 658 articles.

HC3: The Human ChatGPT Comparison Corpus (HC3) (Guo et al., 2023) includes comparative responses from human experts and ChatGPT, covering questions from various fields such as opendomain, finance, medicine, law, psychology and Wikipedia. We construct the input by concatenating the Question and Answers together as a single document and label whether it is ChatGPT generated by the source of the answers. We also use the data without Q&A settings and only treat the answer part as a single document.

WikiText: WikiText language modeling dataset (Merity et al., 2016) is a much larger set of verified good and featured articles extracted from Wikipedia compared to WikiSection,we further compare these two dataset (Section C) and show that there is only $\sim 1 \%$ potential overlap in topics. We used WikiText-103-v1 collection in specific for experiments. This dataset encompass over 100 million tokens from 29,061 full articles. The dataset is assessible through Huggingface <sup>1</sup>.

Difference between WikiSection and WikiText The WikiSection dataset comprises 2,165 articles describing cities from Wikipedia, while WikiText includes 29,061 featured or high-quality articles covering a broader range of topics. The Wiki-Section dataset is most similar to the “places" category in WikiText, which contains approximately 500 articles. To ensure dataset exclusivity, we used string match to check the overlapping. The regular expression query we used is ’(a|the) ([\w\s]\*)?(city|town) in’ as it is contained in 1,721 articles out of 2,165 in WikiSection dataset. Using the same query, we examined the WikiText dataset and checked the intersection of first word of the article from both search result. After manually getting rid of false positives, there are around 30 documents found overlap in both datasets. We argue that with that amount of ( 0.1%) contamination, WikiSection can be considered out of domain of WikiText.

## D Other scores used in this paper

Entity Grid Barzilay and Lapata (2005) is the most recognized entity-based approach. It creates a two-way contingency table for each input document to track the appearance of entities in each sentence. We use Stanford’s CoreNLP to annotate the documents and the implementation provided in the Coheoka library<sup>2</sup> to obtain the Entity Grid score.

Unified Coherence Moon et al. (2019) presents a neural-based entity-grid method that integrates sentence grammar, inter-sentence coherence relations, and global coherence patterns, achieving state-ofthe-art results in artificial tasks.

BBScore Sheng et al. (2024) introduces BB-Score, and also check the main text for a comprehensive comparison between BBScore and BB-ScoreV2.

## E Human-AI comparison test

Table 5 presents the performance of BBScoreV2 computed with different $\widehat { \Sigma } \in \mathbb { R } ^ { d }$ , while Table 6 bshows the performance of the BBScore with various ${ \widehat { \sigma } } \in \mathbb { R }$ , where the subscript indicates the bdataset used for approximation.

The clear improvement over the BBScore demonstrates that accurately capturing structural and temporal information can significantly enhance the model’s accuracy. Table 7 display the performance of DetectGPT with more inferences which significantly improves its performance while also takes much longer time to infer.

<table><tr><td rowspan="2"></td><td colspan="3">Human AI comparison</td><td colspan="3">Human AI comparison with Q&amp;A</td></tr><tr><td>Human  $( \widehat { \Sigma } _ { h u m a n } )$ </td><td>Human  $( \widehat { \Sigma } _ { a i } )$ </td><td>Human  $( \widehat { \Sigma } _ { w i k i } )$ </td><td>Human  $( \widehat { \Sigma } _ { h u m a n } )$ </td><td>Human  $( \widehat { \Sigma } _ { a i } )$ </td><td>Human  $( \widehat { \Sigma } _ { w i k i } )$ </td></tr><tr><td> $\mathrm { A I } ( \widehat { \Sigma } _ { h u m a n } )$ </td><td>70.07</td><td>70.55</td><td>1</td><td>69.00</td><td>69.60</td><td>1</td></tr><tr><td> $\widehat { \mathrm { A I } \left( \sum _ { a i } \right) }$ </td><td>59.98</td><td>61.52</td><td>=</td><td>58.19</td><td>59.74</td><td>1</td></tr><tr><td> $\widehat { \mathrm { A I } \left( \widehat { \Sigma } _ { w i k i } \right) }$ </td><td></td><td></td><td>70.67</td><td>=</td><td></td><td>69.71</td></tr></table>

Table 5: Combined accuracy of human AI comparison and human AI comparison with Q&A

Table 6: Human-AI Task Results with BBScore (Sheng et al., 2024).
<table><tr><td rowspan="2"></td><td colspan="3">Human AI comparison</td><td colspan="3">Human AI comparison with Q&amp;A</td></tr><tr><td>Human  $( \widehat { \sigma } _ { h u m a n } )$ </td><td>Human  $( \widehat \sigma _ { a i } )$ </td><td>Human  $( \widehat { \sigma } _ { w i k i } )$ </td><td>Human  $( \widehat { \sigma } _ { h u m a n } )$ </td><td>Human  $( \widehat \sigma _ { a i } )$ </td><td>Human  $( \widehat { \sigma } _ { w i k i } )$ </td></tr><tr><td> $\overline { { \mathrm { A I } \left( \widehat { \sigma } _ { h u m a n } \right) } }$ </td><td>35.99</td><td>45.13</td><td>1</td><td>35.04</td><td>38.84</td><td>1</td></tr><tr><td> $\mathrm { A I } \left( \widehat { \sigma } _ { a i } \right)$ </td><td>26.37</td><td>37.05</td><td>-</td><td>33.73</td><td>38.12</td><td>-</td></tr><tr><td> $\mathrm { A I } \left( \widehat { \sigma } _ { w i k i } \right)$ </td><td>=</td><td>=</td><td>37.53</td><td>-</td><td></td><td>31.47</td></tr></table>

<table><tr><td>Number of Perturbations</td><td>Human AI comparison 1 10</td><td>1</td><td>Human AI comparison with Q&amp;A 10</td></tr><tr><td>Number of LLM Inferences</td><td colspan="3">Number of Perturbations + 1</td></tr><tr><td>Accuracy</td><td>64.30 84.89</td><td>63.30</td><td>83.13</td></tr></table>

Table 7: Human-AI Task Results with DetectGPT (Mitchell et al., 2023). As a comparison, BBScoreV2 only requires one LLM inference.