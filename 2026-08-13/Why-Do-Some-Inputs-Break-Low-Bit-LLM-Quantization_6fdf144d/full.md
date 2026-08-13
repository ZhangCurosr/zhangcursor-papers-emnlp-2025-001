# Why Do Some Inputs Break Low-Bit LLM Quantization?

Ting-Yun Chang Muru Zhang Jesse Thomason Robin Jia University of Southern California, Los Angeles, CA, USA {tingyun, muruzhan, jessetho, robinjia}@usc.edu

## Abstract

Low-bit weight-only quantization significantly reduces the memory footprint of large language models (LLMs), but disproportionately affects certain examples. We analyze diverse 3–4 bit methods on LLMs ranging from 7B–70B in size and find that the quantization errors of 50 pairs of methods are strongly correlated (avg. ρ = 0.82) on FineWeb examples. Moreover, the residual stream magnitudes of full-precision models are indicative of future quantization errors. We further establish a hypothesis that relates the residual stream magnitudes to error amplification and accumulation over layers. Using LLM localization techniques, early exiting, and activation patching, we show that examples with large errors rely on precise residual activations in the later layers, and that the outputs of MLP gates play a crucial role in maintaining the perplexity. Our work reveals why certain examples result in large quantization errors and which model components are most critical for performance preservation.

## 1 Introduction

Many large language models (LLMs; AI@Meta, 2024; Gemma et al., 2024; OLMo et al., 2024; Qwen, 2025) use 16-bit precision to represent each parameter by default, requiring substantial GPU memory for inference. Quantization reduces memory usage by reducing the bit-precision of model weights or activations. Low-bit weight-only quantization (Frantar et al., 2023; Sheng et al., 2023; Park et al., 2024; Lee et al., 2024), which quantizes model weights to 4 bits and keeps the activations in 16 bits, greatly reduces the memory footprint while maintaining reasonable performance across various benchmarks (Kurtic et al., 2024).

However, Dettmers and Zettlemoyer (2023) find that at 3-bits, LLMs start to show noticeable accuracy degradation. Kumar et al. show that LLMs trained on more data become harder to quantize. Prior efforts focus on outliers, the abnormally large activations or weights that cause large errors in LLM quantization (Dettmers et al., 2022). Quantization methods that combat outliers lead to overall better performance (Xiao et al., 2023; Chee et al., 2023; Ashkboos et al., 2024; Luo et al., 2025); however, we observe that some examples still suffer from large quantization errors under various methods. Why does quantization disproportionately affect certain examples? What factors are indicative of quantization errors? Which parts of the LLMs lead to large errors? We study these underexplored questions across diverse 3- and 4-bit weight-only quantization methods.

First, we observe that the quantization errors of various pairs of methods are highly correlated (avg. $\rho = 0 . 8 2 )$ and that certain examples yield large errors across methods, where we define errors as the increase in negative log-likelihoods on sampled pretraining documents after quantization. A possible hypothesis is that the errors are largely predetermined by the full-precision model. Since Lin et al. (2024) show that activation outliers amplify the errors in quantized weights, we examine whether the activation magnitudes of the full-precision models are indicative of the future quantization errors.

Surprisingly, we find strong negative correlations between quantization errors and the magnitudes of the residual stream. For instance, the correlations computed on the residuals across the last 30 layers of Llama3-70B often exceed 0.6 and reach 0.8 in the final layer. The strong correlations support our hypothesis that the full-precision model largely decides the future quantization errors. Meanwhile, it is unexpected that smaller residual activations are associated with large quantization errors. Our further analysis reveals that the RMS LayerNorm (Zhang and Sennrich, 2019) tends to reverse the relative activation magnitudes. These findings altogether explain why certain examples have large quantization errors across methods: examples that inherently have smaller residual magnitudes will have larger post-RMSNorm magnitudes over multiple layers, which continually amplify the errors in the quantized weights and ultimately result in large quantization errors in outputs.

We further apply LLM localization techniques to study where quantization errors arise from. First, we use early exiting (nostalgebraist, 2020) to project the residual hidden states across layers to the output probabilities. Early exiting reveals that examples with large quantization errors rely heavily on later layers to adjust the model output distributions. However, restoring the weights of later layers to 16 bits only marginally reduces perplexity, implying that the problem lies with the accumulated errors in activations over layers. Next, we adapt cross-model activation patching (Prakash et al., 2024), which restores specific activations of the quantized model to the clean counterparts from the full-precision model. We find that restoring the outputs of the MLP gate projection (Shazeer, 2020) can substantially reduce the perplexity.

Finally, we investigate what kinds of data tend to have large quantization errors. We classify the large-error examples and find that they cover diverse topics. Because Ahia et al. (2021); Marchisio et al. (2024) show that model compression has a disparate impact on long-tail examples, we also study the relationship between quantization errors and long-tail examples. Surprisingly, our results on both FineWeb (Penedo et al., 2024) and PopQA (Mallen et al., 2023) datasets suggest that quantization has a milder impact on long-tail examples and that well-represented examples are not immune to large degradation in log-likelihood and accuracy.

Our work centers on why, where, and what leads to large quantization errors, offering systematic analyses across diverse methods. We provide a mechanistic explanation of why certain examples suffer from large quantization errors and identify key model components that are critical to maintaining the perplexity. Our study could motivate future methods to reduce errors from MLP gate projections or to introduce LayerNorm scaling factors (Wei et al., 2023; Lin et al., 2024) that consider the reversal effects in RMSNorm.<sup>1</sup>

## 2 Background

This section introduces the quantization methods in our study, covering diverse, widely-used approaches for low-bit weight-only quantization.

GPTQ. Frantar et al. (2023) propose GPTQ, an approximate second-order method that quantizes one weight at a time while updating not-yetquantized weights to compensate for the quantization loss. GPTQ minimizes the loss on a small calibration set of C4 (Raffel et al., 2020) samples. We adopt common GPTQ settings, using activation sorting and a Hessian damping coefficient of 0.01 when not otherwise specified.

AWQ. Lin et al. (2024) observe that the activation outliers magnify the quantization errors in the corresponding weight dimensions. They propose AWQ, which smoothes the activations by introducing a scaling factor corresponding to each input dimension of weights. AWQ grid-searches the scaling factors based on the activation magnitudes.

NormalFloat. Dettmers et al. (2023) propose NormalFloat (NF), a 4-bit data type that is optimal for normally distributed weights based on information theory. NF employs non-uniform quantization with Gaussian quantiles and requires a lookup table. Guo et al. (2024) accelerate the lookup table operation and extend NF to lower bit precision.

EfficientQAT. Chen et al. (2024) propose EfficientQAT (EQAT), a lightweight quantizationaware fine-tuning (Liu et al., 2024) method. EQAT applies straight-through estimator (Bengio et al., 2013) for backpropagation.

LayerNorm in Quantization. Kovaleva et al. (2021); Wei et al. (2022) find that outliers in the LayerNorm parameters of BERT (Devlin et al., 2019) cause difficulties in model compression. Given the importance of LayerNorm, all the quantization methods we discuss above leave LayerNorm unquantized. Specifically, LLMs in Llama (Touvron et al., 2023), Mistral (Jiang et al., 2023), and Qwen (Bai et al., 2023) families use RMSNorm (Zhang and Sennrich, 2019):

$$
\mathrm { R M S N o r m } ( z ) = \frac { z } { \mathrm { R M S } ( z ) } \odot \gamma ,\tag{1}
$$

$$
{ \mathrm { R M S } } ( z ) = { \sqrt { { \frac { 1 } { d } } \sum _ { i = 1 } ^ { d } z _ { i } ^ { 2 } } } ,\tag{2}
$$

where z is the d-dimensional input,  denotes element-wise multiplication, and $\gamma \in \mathbb { R } ^ { d }$ is the unqauntized weights for affine transformation.

## 3 Problem Setups

## 3.1 Datasets

FineWeb. We create a dataset of 10k examples sampled from the FineWeb corpus (Penedo et al., 2024), consisting of deduplicated English web documents. We evaluate the negative log-likelihood (NLL) of different models on . Formally, given a model θ and an example $x = [ x _ { 0 } , x _ { 1 } , \dotsc , x _ { T } ]$ of T tokens, prepended with a start of sentence token x<sub>0</sub>, we define the NLL of x as:

$$
\mathrm { N L L } ( \boldsymbol { x } ; \theta ) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } - \log p _ { \theta } ( x _ { t } | \boldsymbol { x } _ { < t } ) ,\tag{3}
$$

where $p _ { \theta }$ is the probability distribution of the model θ. We only sample from long documents and truncate all the sampled documents to 512 tokens.

Quantization Errors. NLL is a common loss term for LLM pretraining, and prior quantization research uses perplexity (exponentiated average NLL) as a standard metric. Therefore, we define the quantization error on a FineWeb example x as the increase in NLL. Formally, given the fullprecision LLM θ and its quantized counterpart ${ \tilde { \theta } } ,$ we define the quantization error of $\tilde { \theta }$ on x as:

$$
e ( x ; \widetilde { \theta } ) = \mathrm { N L L } ( x ; \widetilde { \theta } ) - \mathrm { N L L } ( x ; \theta )\tag{4}
$$

Large-Error Set and Control Set. For finegrained analyses, we create a large-error set $\mathcal { D } _ { \mathrm { l a r g e } } \subset \mathcal { D }$ , consisting of the top-10% (1k) examples by quantization errors and a control set $\mathcal { D } _ { \mathrm { c t r l } }$ of 1k examples sampled from the bottom-50%.<sup>2</sup>

PopQA. We use PopQA (Penedo et al., 2024) to study how quantization affects LLMs’ factual knowledge of entities under different popularity levels. PopQA contains 14k questions, each supplemented with the entity popularity. Following Penedo et al. (2024), we use 15-shot prompting, greedy decoding, and exact match accuracy.

## 3.2 Models and Quantization Methods

We study four LLMs: Qwen2.5-7B (Qwen, 2024), Llama3-8B (AI@Meta, 2024), Mistral-Nemo-12B,<sup>3</sup> and Llama3-70B. We study quantization methods in the 3–4 bit, weight-only quantization regime: GPTQ, AWQ, NF, and EQAT (see §2). We follow original implementations and detail the calibration sets, quantization configurations, and perplexity of each method in appendix A.

## 4 Do Methods Fail on the Same Inputs?

First, we investigate whether diverse quantization methods cause large errors on the same inputs. Recall that we define the example-level quantization error $e ( x ^ { ( i ) } ; \tilde { \theta } ) ~ \in ~ \mathbb { R }$ in Eq. 4. Given a dataset $\mathcal { D } = \{ x ^ { ( i ) } \} _ { i = 1 } ^ { N }$ of $N$ examples, we define the errors of the quantized model $\tilde { \theta }$ on the dataset as:

$$
\begin{array} { r } { \mathcal { E } ( \mathcal { D } ; \widetilde { \theta } ) = [ e ( x ^ { ( 1 ) } ; \widetilde { \theta } ) , \dotsc , e ( x ^ { ( N ) } ; \widetilde { \theta } ) ] \ \in \mathbb { R } ^ { N } , } \end{array}
$$

Let $\tilde { \theta } _ { 1 }$ and ${ \tilde { \theta } } _ { 2 }$ be the quantized models of the same base LLM θ, produced using different methods. We compute their quantization errors on the dataset, $\mathcal { E } _ { 1 } \triangleq \mathcal { E } ( \mathcal { D } ; \tilde { \theta } _ { 1 } )$ and $\mathcal { E } _ { 2 } \triangleq \mathcal { E } ( \mathcal { D } ; \tilde { \theta } _ { 2 } )$ , respectively, and report the Pearson correlation between the errors, denoted as $\rho ( \mathcal { E } _ { 1 } , \mathcal { E } _ { 2 } )$ .

## 4.1 Strong Correlations Between Methods

Table 1 shows the correlations between the quantization errors of every two methods. We include the results of Qwen2.5-7B and Llama3-8B in Table 6 in appendix. Across four LLMs, we observe high correlations, an average of $\rho ( \mathcal { E } _ { 1 } , \mathcal { E } _ { 2 } ) = 0 . 8 2$ over all 50 pairs of methods, although Qwen2.5- 7B has lower correlations (avg. 0.66) compared to other LLMs. The overall strong correlation indicates that diverse methods have a similar effect on

<table><tr><td>Model</td><td>(Method1, Method2)</td><td> $\rho ( \mathcal { E } _ { 1 } , \mathcal { E } _ { 2 } )$ </td></tr><tr><td rowspan="7">Mistral-12B</td><td>(AWQ4,NF4)</td><td>0.78</td></tr><tr><td>(AWQ4, GPTQ4)</td><td>0.86</td></tr><tr><td>(AWQ4,NF3)</td><td>0.80</td></tr><tr><td>(AWQ4, GPTQ3)</td><td>0.83</td></tr><tr><td>(AWQ3, NF3)</td><td>0.90 0.95</td></tr><tr><td>(AWQ3 , GPTQ3) (NF4, GPTQ4)</td><td>0.82</td></tr><tr><td>(NF4,AWQ3)</td><td>0.81</td></tr><tr><td>(NF4, GPTQ3)</td><td>0.78</td></tr><tr><td>(NF3 , GPTQ3)</td><td></td></tr><tr><td></td><td>0.89</td></tr><tr><td>(GPTQ4, AWQ3)</td><td>0.90</td></tr><tr><td>(GPTQ4, NF3)</td><td>0.85</td></tr><tr><td rowspan="9">Llama3-70B</td><td>(AWQ4,NF4)</td><td>0.80</td></tr><tr><td>(AWQ4, GPTQ4)</td><td>0.96</td></tr><tr><td>(AWQ4, GPTQ3)</td><td>0.86</td></tr><tr><td>(AWQ4, EQAT3)</td><td>0.81</td></tr><tr><td>(AWQ3, EQAT3)</td><td>0.85</td></tr><tr><td>(NF4, GPTQ4)</td><td>0.81</td></tr><tr><td>(NF4, GPTQ3)</td><td>0.80</td></tr><tr><td>(NF4, AWQ3)</td><td>0.81</td></tr><tr><td>(NF4, EQAT3)</td><td>0.75</td></tr><tr><td></td><td>(GPTQ4, AWQ3)</td><td>0.94</td></tr><tr><td></td><td>(GPTQ4, EQAT3)</td><td>0.83</td></tr><tr><td></td><td>(GPTQ3 , AWQ3 )</td><td>0.96</td></tr><tr><td></td><td>(GPTQ3, EQAT3)</td><td>0.83</td></tr><tr><td></td><td></td><td></td></tr></table>

Table 1: The quantization errors of different methods are strongly correlated on the FineWeb. The numbers in the method names denote 3- or 4-bit precision.

LLMs’ likelihoods. We further find that the top-10% large-error examples of different methods on the same LLMs are highly overlapped, with an average Jaccard similarity of 0.51, far above the $\sim 0 . 0 5$ expected values by chance.

Since the configurations of the GPTQ algorithm can greatly affect the quantization outcomes (Frantar et al., 2023; Chen et al., 2025), we further examine correlations in quantization errors across different GPTQ variants. We focus on two factors: (1) the damping coefficient that improves the numerical stability of the inverse Hessian, varied over the range [0.005, 0.01, 0.02], and (2) whether activation-based sorting is applied to change the weight quantization order. We find that the 3-bit Qwen2.5-7B model variants produced by different GPTQ configurations show a strong average correlation (0.93) in quantization errors on FineWeb (see Table 7 in the appendix).

## 5 Why Do Examples Have Large Errors?

We further explore why certain examples have large quantization errors across methods. Inspired by Lin et al. (2024), we first study whether the activation magnitudes from the full-precision models are predictive of the quantization errors.

## 5.1 Notations

Formally, a Transformer layer (Vaswani et al., 2017) in modern LLMs consists of a multi-headed attention (MHA), an MLP, and two Pre-LNs (Child et al., 2019), $\mathrm { L N _ { 1 } }$ and $\mathrm { L N _ { 2 } }$

$$
r ^ { ( l ) } = z ^ { ( l - 1 ) } + \mathrm { M H A } ^ { ( l ) } \left( \mathrm { L N } _ { 1 } ^ { ( l ) } ( z ^ { ( l - 1 ) } ) \right) ,\tag{5}
$$

$$
z ^ { ( l ) } = r ^ { ( l ) } + \mathrm { M L P } ^ { ( l ) } \left( \mathrm { L N } _ { 2 } ^ { ( l ) } ( r ^ { ( l ) } ) \right) ,\tag{6}
$$

where l is the layer index and $z ^ { ( 0 ) } \in \mathbb { R } ^ { d }$ is the input to the first layer. We observe that the representations after the residual operations, $r ^ { ( l ) }$ and $z ^ { ( l ) }$ are both indicative of quantization errors. $\mathbf { A } \mathbf { s } \ r ^ { ( l ) }$ shows clearer patterns, we name it as residual state and include the results of $z ^ { ( l ) }$ in appendix. Note that all the studied methods quantize the weights in MHA and MLP, but keep the weights $\gamma$ in $\mathrm { L N _ { 1 } }$ and $\mathrm { L N _ { 2 } }$ in 16 bits (see §2).

Given an example x of $T$ tokens, we quantify the magnitude of its residual states at layer l by averaging the Euclidean norm of each token representation $r _ { t } ^ { ( l ) } \in \mathbb { R } ^ { d }$ over all $T$ tokens:

$$
\mathrm { n o r m } \left( x ; r ^ { ( l ) } \right) = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \Vert r _ { t } ^ { ( l ) } \Vert _ { 2 }\tag{7}
$$

We name norm $\left( \boldsymbol { x } ; \boldsymbol { r } ^ { ( l ) } \right) \in \mathbb { R }$ as the residual magnitude of example x at layer l. Given a dataset $\mathcal { D } \ = \ \{ x ^ { ( i ) } \} _ { i = 1 } ^ { N }$ of N examples, we compute norm $( x ; r ^ { ( l ) } )$ for each example $x ^ { ( i ) }$ and denote the residual magnitudes on as $\mathbb { S } ( \mathcal { D } ; \boldsymbol { \theta } ; l ) \in \mathbb { R } ^ { N }$

Finally, given an full-precision model θ with a layer index $l ,$ a quantized model ${ \tilde { \theta } } ,$ and a dataset of N examples, we calculate the Pearson correlation between the residual magnitudes $\mathbb { S } ( \mathcal { D } ; \theta ; l )$ and the quantization errors $\mathcal { E } ( \mathcal { D } ; \tilde { \theta } )$ , denoted as $\rho \left( \mathbb { S } ( l ) , \mathcal { E } \right)$ . Note that $\mathcal { E } ( \mathcal { D } ; \tilde { \theta } )$ measures the final errors of the quantized model and thus does not depend on the layer index l.

## 5.2 Residual Magnitudes From Full-Precision Models Indicates Quantization Errors

We observe a strong negative correlation between the quantization error $\mathcal { E } ( \mathcal { D } ; \tilde { \theta } )$ and the residual magnitudes $\mathbb { S } ( \mathcal { D } ; \theta ; l )$ computed on the last few layers of the full-precision model, where $\mathcal { D }$ is the FineWeb dataset of 10k examples. For example, let $l : = L$ be the last Transformer layer, we report the correlation $\rho \left( \mathbb { S } ( L ) , \mathcal { E } \right)$ of different LLMs and quantization methods in Table 2. The average correlation over 3-bit methods on Qwen2.5- 7B, Llama3-8B, Mistral-12B, and Llama3-70B are 0.66, 0.76, 0.78, and 0.80. As shown in appendix C, the correlations become increasingly negative in upper layers. The strong negative correlations suggest that (1) the full-precision LLMs can indicate an example’s future quantization error, and (2) small residual magnitudes in upper layers are associated with large quantization errors.

<table><tr><td>Model</td><td>Method</td><td> $\rho \left( \mathbb { S } ( L ) , \mathcal { E } \right)$ </td></tr><tr><td rowspan="3">Qwen2.5-7B</td><td>AWQ3</td><td>-0.66</td></tr><tr><td>NF3</td><td>-0.58</td></tr><tr><td>GPTQ3</td><td>-0.74</td></tr><tr><td rowspan="3">Llama3-8B</td><td>AWQ3</td><td>-0.76</td></tr><tr><td>NF3</td><td>-0.76</td></tr><tr><td>EQAT3</td><td>-0.75</td></tr><tr><td rowspan="3">Mistral-12B</td><td>AWQ3</td><td>-0.80</td></tr><tr><td>NF3</td><td>-0.71</td></tr><tr><td>GPTQ3</td><td>-0.83</td></tr><tr><td rowspan="2">Llama3-70B</td><td>AWQ3</td><td>-0.78</td></tr><tr><td>GPTQ3</td><td>-0.81</td></tr></table>

Table 2: The final-layer residual magnitudes from the full-precision model have a strong negative correlation with the quantization errors of 3-bit methods.

![](images/fec17a68f8abb6a29d69311e9293d30462e156dabe6f47b2ddc5bf1032c80f72.jpg)

![](images/d6b8276634cdc2b9c352dc34054d87fa65e54862c74a9f2ad2ef8b9ae7f6c801.jpg)

![](images/51c00d87f639e3e266b96119b68887815e56ccf3de1a534617887ac00aba7842.jpg)  
Figure 1: (a) The residual magnitudes of the large-error set $\mathcal { D } _ { \mathrm { l a r g e } }$ are smaller than those of $\mathcal { D } _ { \mathrm { c t r l } }$ across layers. (b) & (c) RMSNorm reverses the relative activation magnitudes. Before RMSNorm (b), the residual magnitudes of $\mathcal { D } _ { \mathrm { l a r g e } }$ are smaller than $\mathcal { D } _ { \mathrm { c t r l } } .$ , as shown in (a). After applying RMSNorm (c), the activation magnitudes of $\mathcal { D } _ { \mathrm { l a r g e } }$ become larger. We observe clear reversal effects in the last 10 layers of the full-precision Llama3-70B.

To further investigate the cause of large errors, we compare the large-error set $\mathcal { D } _ { \mathrm { l a r g e } }$ with the control set $\mathcal { D } _ { \mathrm { c t r l } }$ (§3.1). We compute the average residual magnitudes over examples in $\mathcal { D } _ { \mathrm { l a r g e } }$ and $\mathcal { D } _ { \mathrm { c t r l } }$ respectively, and then compare their average magnitudes across the upper half layers of Llama3-70B. In Fig. 1 (a), we find that for both datasets, the average residual magnitudes increase over layers; however, the $\mathcal { D } _ { \mathrm { l a r g e } }$ (orange) consistently has smaller magnitudes than $\mathcal { D } _ { \mathrm { c t r l } }$ (blue). This implies that having a small residual magnitude in the final layer is the cumulative result of continually having smaller residual magnitudes over the upper layers. Our observation seems to contradict the previous belief that large activations are detrimental to weight-only quantization (Lin et al., 2024). We investigate the cause of this discrepancy in the next section.

## 5.3 From Residuals to Errors

The reversal effect of RMSNorm. Why are smaller residual magnitudes in upper layers associated with larger quantization errors? The RMSNorm applied to the residual states before MLP (Eq. 6), plays a crucial role. We observe a “reversal effect”, where the activations with smaller magnitudes often end up with larger magnitudes after applying RMSNorm. Formally, we denote the hidden state after RMSNorm as $\it { h ^ { ( l ) } }$ = RMSNorm ${ \bf \Xi } ^ { ( l ) } ( r ^ { ( l ) } )$ . Following Eq. 7, we can compute the magnitude of $\it { h ^ { ( l ) } }$ , denoted as norm $( x ; h ^ { ( l ) } )$ . Fig. 1 (b) & (c) present the density of norm $( x ; r ^ { ( l ) } )$ and norm $( x ; h ^ { ( l ) } )$ , respectively. Initially, $\mathcal { D } _ { \mathrm { l a r g e } }$ has smaller norm $( x ; r ^ { ( l ) } )$ than $\mathcal { D } _ { \mathrm { c t r l } } .$ , but after RMSNorm, it has larger norm $( x ; h ^ { ( l ) } )$ . We observe the reversal effect in the upper layers of all four LLMs.

Activations amplify errors in weights. Since the hidden state after RMSNorm is the immediate input to the quantized MLP (Eq. 6), we hypothesize that a larger norm $( x ; h ^ { ( l ) } )$ leads to amplification of errors in the quantized weights. Let $\it { h ^ { ( l ) } }$ and $\tilde { h } ^ { ( l ) }$ be the inputs to the l-th layer MLP in the fullprecision and quantized models, respectively, and let $o ^ { ( l ) }$ and $\tilde { o } ^ { ( \hat { l } ) }$ be the corresponding MLP outputs. We compute the mean squared error between $\bar { \mathbf { \chi } } _ { o } ( l )$ and $\tilde { o } ^ { ( l ) }$ to quantify the layer-wise output error, denoted as MSE<sup>(l)</sup>. We confirm that the input magnitudes norm $( x ; h ^ { ( l ) } )$ have positive correlations $( > 0 . 6 )$ with $\mathrm { M S E } ^ { ( l ) }$ in most upper layers.

Full Hypothesis. Combining our findings above, we establish a full hypothesis: certain examples inherently have smaller residual magnitudes over layers. Due to the reversal effect, they tend to have larger activation magnitudes after RMSNorms. When the model is quantized, large activations will amplify the quantization errors in the weights over multiple layers and eventually result in large quantization errors. Both the residual states and errors are cumulative over layers, which may explain why the final-layer residual magnitudes $\mathbb { S } ( \mathcal { D } ; \theta ; L )$ have the strongest correlation with the final errors $\mathcal { E } ( \mathcal { D } ; \tilde { \theta } )$

<table><tr><td>Notation</td><td>Definition</td></tr><tr><td> $\boldsymbol { r } ^ { ( l ) }$ </td><td>layer l residual states (Eq. 5)</td></tr><tr><td> $\it { h ^ { ( l ) } }$ </td><td> $\mathbb { R } ^ { d }$  RMSNorm  $^ { ( l ) } ( r ^ { ( l ) } )$   $\mathbb { R } ^ { d }$ </td></tr><tr><td> $o ^ { ( l ) }$ </td><td>ML  $\mathbf { \nabla } \cdot \mathbf { P } ^ { ( l ) } ( h ^ { ( l ) } )$ </td></tr><tr><td> $\gamma$ </td><td> $\mathbb { R } ^ { d }$  RMSNorm weights (Eq. 1)</td></tr><tr><td> $\left( x ; r ^ { \left( l \right) } \right)$  norm</td><td>layer l residual magnitude of x</td></tr><tr><td> $\mathbb { S } ( \mathcal { D } ; \theta ; l )$ </td><td> $\mathbb { R } ^ { N }$  layer l residual magnitudes of D</td></tr></table>

Table 3: Notation summary.

Why does $\mathcal { D } _ { \mathbf { l a r g e } }$ have smaller residual magnitudes? We find that examples in the large-error set have fewer/weaker outliers in the residual states $r ^ { ( l ) }$ . Following Bondarenko et al. (2023); Gong et al. (2024), we use the Kurtosis value to quantify the severity of outliers. We measure the Kurtosis value $\kappa ^ { ( l ) }$ of the l-th layer residual states $r ^ { ( l ) }$ , where $\begin{array} { r } { \kappa ^ { ( l ) } = \mathbb { E } \left\lceil \left( \frac { r ^ { ( l ) } - \mu } { \sigma } \right) ^ { 4 } \right\rceil } \end{array}$ . A large $\kappa ^ { ( l ) }$ indicates more outliers (Liu et al., 2025a). Fig. 6 in appendix shows that the Kurtosis values of $\mathcal { D } _ { \mathrm { l a r g e } }$ (orange) are smaller than those of $\mathcal { D } _ { \mathrm { c t r l } }$ (blue) across the upper layers of LLMs, suggesting that $\mathcal { D } _ { \mathrm { l a r g e } }$ has fewer outliers in the residual states. Because outliers dominate the activation magnitudes, having fewer outliers leads to smaller residual magnitudes.

How does RMSNorm reverse the relative magnitudes? We observe that the RMSNorm weights $\gamma \in \mathbb { R } ^ { d }$ suppress many outliers in the residual states by assigning smaller weight values to the corresponding dimensions. Specifically, we rank the entries in $\gamma$ by their absolute values and find that in the last few layers, the largest outliers<sup>4</sup> often have the smallest RMSNorm weights across all four LLMs. We include the statistics and discussion on other outlier dimensions in appendix D. We summarize the mechanism behind the reversal effect: a residual state $r ^ { ( l ) }$ with fewer outliers has a smaller magnitude, but RMSNorm normalizes the magnitude and downweights outliers with $\gamma ,$ yielding a hidden state $\it { h ^ { ( l ) } }$ with a relatively larger magnitude. Since $\mathcal { D } _ { \mathrm { l a r g e } }$ examples have smaller kurtosis values but larger norm ${ \big ( } x ; h ^ { ( l ) } { \big ) }$ , the large quantization errors are driven by the overall larger activations in $\it { h ^ { ( l ) } }$

## 6 Where Do Errors Arise From?

We study which parts of LLMs lead to large quantization errors with localization techniques: early exiting (nostalgebraist, 2020; Geva et al., 2022) and activation patching (Meng et al., 2022).

![](images/321d6fc036c3d6ddf88a69fdd976830abb41eedb4b3037ca4b2d4b9176e0fb57.jpg)  
Figure 2: Early exiting across layers shows that the NLLs of $\mathcal { D } _ { \mathrm { l a r g e } }$ (purple and orange) dramatically decrease in the late layers, suggesting that $\mathcal { D } _ { \mathrm { l a r g e } }$ heavily relies on the late layers to adjust the output probabilities. Meanwhile, the quantized model (orange) does not show obvious deviation from the full-precision one (purple) until layer 33.

## 6.1 Early Exiting

nostalgebraist (2020) first introduces logit lens, an early exiting technique that projects hidden representations into the vocabulary space using the LLMs’ output embeddings. We use early exiting to measure the degree to which a residual state is ready for decoding. Formally, let $r ^ { ( l ) }$ be the residual state at layer l and $\phi ( \cdot )$ be the final RMSNorm operation applied before the output embedding matrix U. We define the probability distribution from decoding $r ^ { ( l ) }$ as:

$$
p ^ { ( l ) } = \mathrm { S o f t m a x } \left( U \cdot \phi ( r ^ { ( l ) } ) \right)
$$

Let $p _ { \theta } ^ { ( l ) }$ and $p _ { \tilde { \theta } } ^ { ( l ) }$ be the early-exiting probabilities from the full-precision and quantized models. We use $p _ { \theta } ^ { ( l ) }$ and $p _ { \tilde { \theta } } ^ { ( l ) }$ to compute the average negative log-likelihood (NLL) over dataset at each layer, denoted $\mathrm { n l l } _ { \theta } ^ { ( l ) } ( \mathcal { D } )$ and $\mathrm { n l l } _ { \tilde { \theta } } ^ { ( l ) } ( \mathcal { D } )$ , respecitvely. A $\mathrm { n l l } _ { \theta } ^ { ( l ) } ( \mathcal { D } )$ value near the final model NLL means that the residual state at layer l contains sufficient information for decoding; otherwise, further layers are needed to refine the output probability.

We investigate two key questions: (1) Can we observe patterns from the early exiting dynamics that distinguish $\mathcal { D } _ { \mathrm { l a r g e } }$ and $ { \mathcal { D } _ { \mathrm { c t r l } } } ? \left( 2 \right)$ At which layer do $\mathrm { n l l } _ { \theta } ^ { ( l ) } ( \mathcal { D } )$ and $\mathrm { n l l } _ { \tilde { \theta } } ^ { ( l ) } ( \mathcal { D } )$ begin to diverge? To answer these questions, we perform four runs of early exiting, one for each combination of model type (full-precision and quantized) and dataset split $( \mathcal { D } _ { \mathrm { l a r g e } }$ and $\mathcal { D } _ { \mathrm { c t r l } } )$ . We then compute $\mathrm { n l l } _ { \theta } ^ { ( l ) } ( \mathcal { D } _ { \mathrm { l a r g e } } )$ n $\mathop { \mathrm { l l l } } _ { \tilde { \theta } } ^ { ( l ) } ( { \mathcal { D } } _ { \mathrm { l a r g e } } )$ $\mathrm { n l l } _ { \theta } ^ { ( l ) } ( \mathcal { D } _ { \mathrm { c t r l } } )$ , and $\mathrm { n l l } _ { \tilde { \theta } } ^ { ( l ) } ( \mathcal { D } _ { \mathrm { c t r l } } )$ across the upper half layers.

<table><tr><td>Model</td><td>Method</td><td>16-bit</td><td>3-bit</td><td>Patch  $h _ { \mathrm { d o w n } }$ </td><td> $\mathbf { P a t c h } h _ { \mathrm { g a t e } }$ </td><td>Patch  $h _ { \mathrm { u p } }$ </td><td>Patch  $h _ { \mathrm { a t t n } }$ </td></tr><tr><td>Qwen2.5-7B</td><td>AWQ3 NF3</td><td>9.42</td><td>12.26 (+2.84)  $1 2 . 5 3 \ : ( + 3 . 1 1 )$ </td><td> $9 . 7 3 \ ( + 0 . 3 1 )$   $9 . 6 6 \ : ( + 0 . 2 4 )$ </td><td>10.20 (+0.78) 10.61 (+1.19)</td><td>12.64 (+3.22) 12.97 (+3.55)</td><td>11.12 (+1.70) 11.35 (+1.93)</td></tr><tr><td>Llama3-8B</td><td>AWQ3 NF3</td><td>5.62</td><td>10.68 (+5.06) 11.02 (+5.40)</td><td> $5 . 9 5 \ ( + 0 . 3 3 )$  5.92 (+0.30)</td><td>6.42 (+0.80) 6.43 (+0.81)</td><td>9.97 (+4.35) 10.48 (+4.86)</td><td>9.27 (+3.65) 9.49 (+3.87)</td></tr><tr><td>Mistral-12B</td><td>AWQ3 NF3</td><td>5.60</td><td>8.67 (+3.07)  $1 0 . 1 4 \ ( + 4 . 5 4 )$ </td><td>6.05 (+0.45) 5.92 (+0.32)</td><td>6.13 (+0.53) 7.12 (+1.52)</td><td>12.49 (+6.89) 13.75 (+8.15)</td><td>7.61 (+2.01) 8.71 (+3.11)</td></tr><tr><td>Llama3-70B</td><td>AWQ3</td><td>2.66</td><td> $5 . 2 5 \ : ( + 2 . 5 9 )$ </td><td> $2 . 7 3 \ : ( + 0 . 0 7 )$ </td><td> $2 . 9 4 \ ( + 0 . 2 8 )$ </td><td> $3 . 2 4 \ : ( + 0 . 5 8 )$ </td><td> $4 . 7 3 \ : ( + 2 . 0 7 )$ </td></tr></table>

Table 4: The perplexity ( ) on $\mathcal { D } _ { \mathrm { l a r g e } }$ before/after patching activations from 16-bit full-precision models into 3-bit models, quantized with AWQ3 and NF3 methods, respectively. Both patching the entire MLP outputs $\left( { { h _ { \mathrm { { d o w n } } } } } \right)$ and patching only the MLP gate outputs $( h _ { \mathrm { g a t e } } )$ can substantially close the gap between the 3-bit and 16-bit models.

Fig. 2 compares how the four NLL values evolve across the upper layers of Mistral-12B, where the quantized model $\tilde { \theta }$ is produced by AWQ3. We observe a clear distinction between $\mathcal { D } _ { \mathrm { l a r g e } }$ and $\mathcal { D } _ { \mathrm { c t r l } }$ on the full-precision model: $\mathcal { D } _ { \mathrm { l a r g e } }$ (purple) has higher NLLs than $\mathcal { D } _ { \mathrm { c t r l } }$ (green) until layer 36, around which its NLLs begin to decrease sharply. This observation supports our claim in §5 that the properties of the full-precision model reflect the future quantization errors. Moreover, the sharp NLL decrease in later layers suggests that $\mathcal { D } _ { \mathrm { l a r g e } }$ heavily relies on these layers to make accurate predictions. This reliance requires the quantized model to keep the residual states precise through to the later layers, which is a challenging task because quantization errors propagate over layers. Otherwise, the model cannot decode the noisy representations, resulting in large quantization errors. By comparison, $\mathcal { D } _ { \mathrm { c t r l } }$ relies less on the later layers, and the quantized model (cyan) shows less deviation from the fullprecision one (green).

An alternative hypothesis is that the precision of weights in the later layers is crucial to $\mathcal { D } _ { \mathrm { l a r g e } }$ . To test this hypothesis, we keep all the weights above layer 32 in full-precision, because Fig. 2 shows that on $\mathcal { D } _ { \mathrm { l a r g e } }$ , the quantized model (orange) starts to diverge from the full-precision model (purple) at layer 33. This experiment restores 7 layers (18%) from 3 bits to 16 bits. However, the perplexity on $\mathcal { D } _ { \mathrm { l a r g e } }$ only slightly decreases from 8.67 to 8.05, versus 5.60 for the full-precision model. We show similar results on other LLMs in appendix E.

In summary, our early exiting experiment suggests that various methods struggle with $\mathcal { D } _ { \mathrm { l a r g e } }$ because it requires the quantized models to maintain precise residual states up to the late layers. We also find that leaving late layers unquantized has little effect on NLLs, implying that the cumulative error in residual states, not the precision of upper-layer weights, is the primary cause for large errors.

## 6.2 Cross-Model Activation Patching

Prakash et al. (2024) introduce cross-model activation-patching (CMAP) to identify how finetuned models improve over the pretrained ones. We adapt CMAP to track where the quantized models differ from the full-precision ones, aiming to identify the sources of quantization errors. For a given example from $\mathcal { D } _ { \mathrm { l a r g e } }$ , we first run a forward pass on the full-precision model and cache its activations. We then run the same example through the quantized model, but replace the activations of specific modules with those from the full-precision model. We patch the outputs of MHA, denoted $h _ { \mathrm { a t t n } } .$ , and the outputs of different modules in MLP. Specifically, all LLMs we study implement MLP with Gated Linear Units (Dauphin et al., 2017):

$$
\begin{array} { r l } & { h _ { \mathrm { g a t e } } = \sigma ( W _ { \mathrm { g a t e } } \ h ) , } \\ & { h _ { \mathrm { u p } } = W _ { \mathrm { u p } } \ h , } \\ & { h _ { \mathrm { d o w n } } = W _ { \mathrm { d o w n } } \ ( h _ { \mathrm { g a t e } } \odot h _ { \mathrm { u p } } ) , } \end{array}
$$

where h is the post-RMSNorm input to MLP, $W _ { \mathrm { g a t e } } ,$ $W _ { \mathsf { u p } } ,$ and $W _ { \mathrm { d o w n } }$ are the MLP weights, $h _ { \mathrm { d o w n } }$ is the output of MLP, and $\sigma ( \cdot )$ is the activation function. To trace quantization errors, we perform four runs of CMAP, corresponding to patching $h _ { \mathrm { g a t e } } , \ h _ { \mathrm { u p } } ,$ $h _ { \mathrm { d o w n } } .$ , and $h _ { \mathrm { a t t } \mathrm { n } }$ , in the upper half layers of the models. Patching $h _ { \mathrm { d o w n } }$ is a much stronger intervention than patching either $h _ { \mathrm { u p } }$ or $h _ { \mathrm { d o w n } }$ , as it overrides the final outputs of MLPs.

We apply AWQ3 and NF3 methods and report the perplexity on $\mathcal { D } _ { \mathrm { l a r g e } }$ in Table 4. First, we find that patching the outputs of MLP $h _ { \mathrm { d o w n } }$ greatly narrows the perplexity gaps between the 3-bit and 16-bit full-precision models, bringing the gaps below 0.5 across all LLMs. In contrast, patching the outputs of MHA $h _ { \mathrm { a t t n } }$ only leads to marginal effects, suggesting that the quantization errors from MLPs are dominating the results. We further find that inside MLPs, patching $h _ { \mathrm { u p } }$ alone is ineffective, but patching $h _ { \mathrm { g a t e } }$ results in substantial perplexity reduction, approaching the effect of patching the entire MLP outputs $h _ { \mathrm { d o w n } }$ . These results highlight the importance of having precise outputs from MLP gates, which control the information passed on to subsequent layers.

We conduct two additional experiments to understand the source of quantization errors in MLP gates: (1) only patching the inputs to $W _ { \mathrm { g a t e } }$ , and (2) restoring all $W _ { \mathrm { g a t e } }$ weights to 16-bit precision. We find that patching the gates’ inputs (1) performs nearly as well as patching the outputs, $h _ { \mathrm { g a t e } }$ , while restoring the weights (2) is ineffective. For instance, on Llama-3-8B quantized with NF3, (1) achieves a perplexity of 6.53 compared to 10.45 for (2). These results further support our conclusion in §6.1 that noisy activations are the primary source of large quantization errors.

## 7 What Data Cause Large Errors?

Understanding what kinds of data lead to large quantization errors is crucial for future improvement. This section studies whether examples in $\mathcal { D } _ { \mathrm { l a r g e } }$ cluster in certain domains or are unusual, long-tail data.

## 7.1 Categorizing Examples in $\mathcal { D } _ { \mathbf { l a r g e } }$

To better understand the distribution of $\mathcal { D } _ { \mathrm { l a r g e } }$ , we apply the topic classifier from Wettig et al. (2025), which categorizes web pre-training data into 24 topics.<sup>5</sup> We find that the $\mathcal { D } _ { \mathrm { l a r g e } }$ sets of both Mistral-12B and Llama3-70B cover all the 24 topics, and we plot the topic distributions in Fig. 3. For both models, Entertainment, Finance & Business, Politics, and Health are among the top five topics, although none account for more than 12% of the dataset. We also manually inspect $\mathcal { D } _ { \mathrm { l a r g e } }$ examples and do not find them unusual.

## 7.2 Quantization Errors vs. Long-Tail Data

Ogueji et al. (2022); Marchisio et al. (2024) show that model compression has a disparate effect for long-tail data; thus, we explore the relationship between quantization errors and data long-tailness.

![](images/ec0a103385d53e9e2a4e588689f685afdb288a83d791c09e7ec09dc6780f3420.jpg)  
Figure 3: Top five topics of $\mathcal { D } _ { \mathrm { l a r g e } }$ for Mistral-12B (up; orange) and Llama3-70B (down; blue), respectively.

FineWeb. We use the data likelihood assigned by a strong language model, Llama3-70B, to estimate the long-tail characteristics of the examples. We call the FineWeb examples that have small log-likelihoods (large NLLs) on the full-precision Llama3-70B as long-tail examples. Fig. 4 shows that long-tail examples $\left( \mathrm { x } { \cdot } \mathrm { a x i s } ~ \ge ~ 4 \right)$ have small quantization errors under both AWQ3 (avg. 0.14) and GPTQ3 (avg. 0.30) methods. On the other hand, examples that suffer from large errors have widespread NLLs before quantization, and many “head data" (small NL $\mathrm { { . L s } ; \mathrm { { x } { - a x i s } \leq 1 ) } }$ have large quantization errors (avg. 0.67 for AWQ3 and 1.00 for GPTQ3). We observe the same findings on the other LLMs (see Fig. 16 in appendix).

PopQA. We investigate how quantization affects LLMs’ factual knowledge on entities with different popularity levels. We partition the PopQA dataset into buckets based on the examples’ $\log _ { 1 0 } ( \mathrm { p o p u l a r i t y } )$ and calculate the accuracies within each bucket. Fig. 5 presents the accuracy degradation after quantization under different buckets, where we apply NF3 (blue) and AWQ3 (orange) to quantize Mistral-12B into 3 bits. We find that when the examples have the lowest and highest popularities (the leftmost and rightmost buckets, respectively), the 3-bit models show the lowest degradation. In contrast, examples in the middle buckets (log popularity between 2.8 and 4.8) are most affected by quantization, with accuracies dropped by $> 1 0 \%$ . Note that only 6% of examples have log popularity greater than 4.8, meaning that quantization can severely degrade

![](images/643ac97f653c8357eb841bbb34c153d0b39ae90c2c5a976fdbf568c4bc4ba458.jpg)  
Figure 4: Long-tail samples with large NLLs on the fullprecision Llama3-70B have small quantization errors. Each dot represents a FineWeb example. We quantize Llama3-70B with AWQ3 and GPTQ3 methods and compute their quantization errors with Eq. 4.

LLMs’ knowledge on entities that have intermediate to high popularity. We include results from other LLMs in Fig. 17, showing that low popularity examples usually have milder degradation.

Summary. We find that long-tail examples are relatively less affected by quantization on both FineWeb and PopQA. Meanwhile, wellrepresented data are not immune to large errors.

## 8 Discussion

We thoroughly study why certain examples are disproportionately affected by 3–4 bit, weight-only quantization methods. We reveal the distinct nature of large-error examples before quantization, showing that these examples have smaller residual magnitudes and rely on upper layers to refine output probabilities. Our patching and weight recovery experiments further highlight the importance of having precise activations at MLP gates, and reject naive mixed-precision approaches. Our kurtosis value analysis finds that large-error examples actually have fewer outliers in residual states across upper layers. We further discover the reversal effects of RMSNorm, showing that large quantization errors may stem from overall larger activations across dimensions, rather than from a few outlier dimensions. Finally, we explore the relationship between quantization errors and long-tail data, showing that long-tail examples tend to have less degradation after quantization than other examples.

Overall, our work sheds light on why and where large quantization errors occur. Our findings may inspire quantization error prediction frameworks and encourage future post-training methods or quantization-friendly architectures that specifically address MLP gates and RMSNorm; for instance, by avoiding the Gated Linear Units or redesigning the LayerNorm (Zhu et al., 2025).

![](images/1b65bafba804bb87db5c0740c159a0431b7fd3dcfa28d096aa3cfd51fe079990.jpg)  
Figure 5: The accuracy degradation of PopQA under different levels of popularity after 3-bit quantization. Samples with log popularity between 2.8 and 4.8 show the greatest degradation.

## 9 Limitations

We conduct our main experiments on FineWeb and use NLL as our evaluation metric, which does not consider downstream tasks. We believe this is a reasonable choice because Jin et al. (2024) show that perplexity (exponentiated average NLL) is a reliable performance indicator for quantized LLMs on various evaluation benchmarks. We invite future work beyond intrinsic analysis to extrinsic, downstream tasks.

We do not cover sub-3-bit settings because we observe that the methods in our study suffer from extremely large errors when quantized below 3 bits (with perplexity greater than $\mathrm { { \dot { 1 } 0 ^ { 3 } } }$ and near-random accuracy on commonsense QA tasks), making it hard to pinpoint the cause of errors. Because prior work also marks 3 bits as a turning point in quantization (Dettmers and Zettlemoyer, 2023; Liu et al., 2025b), our study of quantization errors under the 3–4 bit regime could be valuable to future efforts to improve quantization methods to lower bits.

## Acknowledgements

We thank Yuqing Yang for helpful discussions, Ming Zhong for guidance on table coloring, and the anonymous reviewers for their valuable feedback. This work was supported in part by the National Science Foundation under Grant No. IIS-2403436.

Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.

## References

Orevaoghene Ahia, Julia Kreutzer, and Sara Hooker. 2021. The low-resource double bind: An empirical study of pruning for low-resource machine translation. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 3316–3333, Punta Cana, Dominican Republic. Association for Computational Linguistics.

AI@Meta. 2024. Llama 3 model card.

Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. 2024. Quarot: Outlier-free 4-bit inference in rotated llms. Advances in Neural Information Processing Systems, 37:100213–100240.

Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer, Albert Q. Jiang, Jia Deng, Stella Biderman, and Sean Welleck. 2023. Llemma: An open language model for mathematics. Preprint, arXiv:2310.10631.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, Keming Lu, and 29 others. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.

Yoshua Bengio, Nicholas Léonard, and Aaron Courville. 2013. Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432.

Yelysei Bondarenko, Markus Nagel, and Tijmen Blankevoort. 2023. Quantizable transformers: Removing outliers by helping attention heads do nothing. Advances in Neural Information Processing Systems, 36:75067–75096.

Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, and Christopher M De Sa. 2023. Quip: 2-bit quantization of large language models with guarantees. Advances in Neural Information Processing Systems, 36:4396– 4429.

Jiale Chen, Torsten Hoefler, and Dan Alistarh. 2025. The geometry of llm quantization: Gptq as babai’s nearest plane algorithm. arXiv preprint arXiv:2507.18553.

Mengzhao Chen, Wenqi Shao, Peng Xu, Jiahao Wang, Peng Gao, Kaipeng Zhang, Yu Qiao, and Ping Luo. 2024. Efficientqat: Efficient quantization-aware training for large language models. arXiv preprint arXiv:2407.11062.

Rewon Child, Scott Gray, Alec Radford, and Ilya Sutskever. 2019. Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509.

Together Computer. 2023. Redpajama: An open source recipe to reproduce llama training dataset.

Yann N Dauphin, Angela Fan, Michael Auli, and David Grangier. 2017. Language modeling with gated convolutional networks. In International conference on machine learning, pages 933–941. PMLR.

Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. 2022. GPT3.int8(): 8-bit matrix multiplication for transformers at scale. In Advances in Neural Information Processing Systems, volume 35, pages 30318–30332. Curran Associates, Inc.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. QLoRA: Efficient finetuning of quantized LLMs. In Thirty-seventh Conference on Neural Information Processing Systems.

Tim Dettmers and Luke Zettlemoyer. 2023. The case for 4-bit precision: k-bit inference scaling laws. In International Conference on Machine Learning, pages 7750–7774. PMLR.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. 2023. GPTQ: Accurate post-training quantization for generative pre-trained transformers. In The Eleventh International Conference on Learning Representations.

Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, and 1 others. 2020. The pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027.

Team Gemma, Thomas Mesnard, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivière, Mihir Sanjay Kale, Juliette Love, and 1 others. 2024. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295.

Mor Geva, Avi Caciularu, Kevin Wang, and Yoav Goldberg. 2022. Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 30–45, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Ruihao Gong, Yang Yong, Shiqiao Gu, Yushi Huang, Chengtao Lv, Yunchen Zhang, Xianglong Liu, and Dacheng Tao. 2024. Llmc: Benchmarking large language model quantization with a versatile compression toolkit. arXiv preprint arXiv:2405.06001.

Han Guo, William Brandon, Radostin Cholakov, Jonathan Ragan-Kelley, Eric P. Xing, and Yoon Kim. 2024. Fast matrix multiplications for lookup tablequantized LLMs. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 12419–12433, Miami, Florida, USA. Association for Computational Linguistics.

Paul Jaccard. 1912. The distribution of the flora in the alpine zone. 1. New phytologist, 11(2):37–50.

Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, and 1 others. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825.

R Jin, J Du, W Huang, W Liu, J Luan, B Wang, and D Xiong. 2024. A comprehensive evaluation of quantization strategies for large language models. arxiv.

Olga Kovaleva, Saurabh Kulshreshtha, Anna Rogers, and Anna Rumshisky. 2021. BERT busters: Outlier dimensions that disrupt transformers. In Findings of the Associationfor Computational Linguistics: ACL-IJCNLP 2021, pages 3392–3405, Online. Association for Computational Linguistics.

Tanishq Kumar, Zachary Ankner, Benjamin Frederick Spector, Blake Bordelon, Niklas Muennighoff, Mansheej Paul, Cengiz Pehlevan, Christopher Re, and Aditi Raghunathan. Scaling laws for precision. In The Thirteenth International Conference on Learning Representations.

Eldar Kurtic, Alexandre Marques, Shubhra Pandit, Mark Kurtz, and Dan Alistarh. 2024. " give me bf16 or give me death"? accuracy-performance trade-offs in llm quantization. arXiv preprint arXiv:2411.02355.

Changhun Lee, Jungyu Jin, Taesu Kim, Hyungjun Kim, and Eunhyeok Park. 2024. Owq: Outlier-aware weight quantization for efficient fine-tuning and inference of large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 13355–13364.

Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. 2024. Awq: Activation-aware weight quantization for ondevice llm compression and acceleration. In Proceedings of Machine Learning and Systems, volume 6, pages 87–100.

Zechun Liu, Barlas Oguz, Changsheng Zhao, Ernie Chang, Pierre Stock, Yashar Mehdad, Yangyang Shi, Raghuraman Krishnamoorthi, and Vikas Chandra.

2024. LLM-QAT: Data-free quantization aware training for large language models. In Findings of the Associationfor Computational Linguistics: ACL 2024, pages 467–484, Bangkok, Thailand. Association for Computational Linguistics.

Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, and Tijmen Blankevoort. 2025a. Spinquant: LLM quantization with learned rotations. In The Thirteenth International Conference on Learning Representations.

Zechun Liu, Changsheng Zhao, Hanxian Huang, Sijia Chen, Jing Zhang, Jiawei Zhao, Scott Roy, Lisa Jin, Yunyang Xiong, Yangyang Shi, and 1 others. 2025b. Paretoq: Scaling laws in extremely low-bit llm quantization. arXiv preprint arXiv:2502.02631.

Haozheng Luo, Chenghao Qiu, Maojiang Su, Zhihan Zhou, Zoe Mehta, Guo Ye, Jerry Yao-Chieh Hu, and Han Liu. 2025. Fast and low-cost genomic foundation models via outlier removal. In Forty-second International Conference on Machine Learning.

Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Daniel Khashabi, and Hannaneh Hajishirzi. 2023. When not to trust language models: Investigating effectiveness of parametric and non-parametric memories. In Proceedings ofthe 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 9802–9822, Toronto, Canada. Association for Computational Linguistics.

Kelly Marchisio, Saurabh Dash, Hongyu Chen, Dennis Aumiller, Ahmet Üstün, Sara Hooker, and Sebastian Ruder. 2024. How does quantization affect multilingual LLMs? In Findings ofthe Association for Computational Linguistics: EMNLP 2024, pages 15928–15947, Miami, Florida, USA. Association for Computational Linguistics.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2022. Locating and editing factual associations in gpt. Advances in neural information processing systems, 35:17359–17372.

nostalgebraist. 2020. interpreting GPT: the logit lens.

Kelechi Ogueji, Orevaoghene Ahia, Gbemileke Onilude, Sebastian Gehrmann, Sara Hooker, and Julia Kreutzer. 2022. Intriguing properties of compression on multilingual models. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 9092–9110, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, Nathan Lambert, Dustin Schwenk, Oyvind Tafjord, Taira Anderson, David Atkinson, Faeze Brahman, Christopher Clark, Pradeep Dasigi, Nouha Dziri, and 21 others. 2024. 2 olmo 2 furious.

Gunho Park, Baeseong park, Minsub Kim, Sungjae Lee, Jeonghoon Kim, Beomseok Kwon, Se Jung Kwon, Byeongwook Kim, Youngjoo Lee, and Dongsoo Lee. 2024. LUT-GEMM: Quantized matrix multiplication based on LUTs for efficient inference in large-scale generative language models. In The Twelfth International Conference on Learning Representations.

Keiran Paster, Marco Dos Santos, Zhangir Azerbayev, and Jimmy Ba. 2023. Openwebmath: An open dataset of high-quality mathematical web text. Preprint, arXiv:2310.06786.

Guilherme Penedo, Hynek Kydlícek, Loubna Ben al-ˇ lal, Anton Lozhkov, Margaret Mitchell, Colin Raffel, Leandro Von Werra, and Thomas Wolf. 2024. The fineweb datasets: Decanting the web for the finest text data at scale. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Nikhil Prakash, Tamar Rott Shaham, Tal Haklay, Yonatan Belinkov, and David Bau. 2024. Fine-tuning enhances existing mechanisms: A case study on entity tracking. In Proceedings of the 2024 International Conference on Learning Representations. ArXiv:2402.14811.

Team Qwen. 2024. Qwen2.5: A party of foundation models.

Team Qwen. 2025. Qwen3.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofmachine learning research, 21(140):1–67.

Noam Shazeer. 2020. Glu variants improve transformer. arXiv preprint arXiv:2002.05202.

Ying Sheng, Lianmin Zheng, Binhang Yuan, Zhuohan Li, Max Ryabinin, Beidi Chen, Percy Liang, Christopher Re, Ion Stoica, and Ce Zhang. 2023. FlexGen: High-throughput generative inference of large language models with a single GPU. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings ofMachine Learning Research, pages 31094–31116. PMLR.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, and 1 others. 2023. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Xiuying Wei, Yunchen Zhang, Yuhang Li, Xiangguo Zhang, Ruihao Gong, Jinyang Guo, and Xianglong Liu. 2023. Outlier suppression+: Accurate quantization of large language models by equivalent and effective shifting and scaling. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 1648–1665.

Xiuying Wei, Yunchen Zhang, Xiangguo Zhang, Ruihao Gong, Shanghang Zhang, Qi Zhang, Fengwei Yu, and Xianglong Liu. 2022. Outlier suppression: Pushing the limit of low-bit transformer language models. Advances in Neural Information Processing Systems, 35:17402–17414.

Alexander Wettig, Kyle Lo, Sewon Min, Hannaneh Hajishirzi, Danqi Chen, and Luca Soldaini. 2025. Organize the web: Constructing domains enhances pretraining data curation. In Forty-second International Conference on Machine Learning.

Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. 2023. Smoothquant: Accurate and efficient post-training quantization for large language models. In International Conference on Machine Learning, pages 38087–38099. PMLR.

Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tieyan Liu. 2020. On layer normalization in the transformer architecture. In International conference on machine learning, pages 10524–10533. PMLR.

Biao Zhang and Rico Sennrich. 2019. Root mean square layer normalization. Advances in Neural Information Processing Systems, 32.

Jiachen Zhu, Xinlei Chen, Kaiming He, Yann LeCun, and Zhuang Liu. 2025. Transformers without normalization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

## A Implementation Details

We aim to answer whether different quantization methods make similar errors; thus, we follow the original implementations and do not unify their quantization configurations.

GPTQ. We use the GPTQModel<sup>6</sup> package for implementation. The calibration set consists of samples from English C4.<sup>7</sup> The quantization group size is 64 for 3-bit models and 128 for 4-bit models. We do not include the results of GPTQ3 with Llama-3-8B because we observe severe perplexity degradation with the latest GPTQModel package.

AWQ. We follow the original implementation and use the released model checkpoints when available. The calibration set consists of samples from the Pile (Gao et al., 2020). The quantization group size is 128 for both 3- and 4-bit models. Note that because AWQ applies scaling transformation, in our patching experiments §6.2, we also apply the same scaling factors on thefull-precision models, which does not change their outputs but aligns their activations with the AWQ-quantized models.

NF. We use the official packages: bitsandbytes<sup>9</sup> for 4-bit models and flute<sup>10</sup> for 3-bit model. The quantization group size is 64 for both 3- and 4-bit models. We do not have the results of NF3 on Llama-3-70B due to kernel incompatibility.

EQAT. EQAT train on thousands of samples from RedPajama (Computer, 2023). We use the officially released 3-bit model checkpoints,<sup>11</sup> which only cover the Llama3 models. The quantization group size is 128.

Perplexity. We report the perplexity of different 3- and 4-bit quantization methods in Table 5. We follow the standard implementation of WikiText perplexity that sets the sequence length to 2048. The sequence length of FineWeb is 512.

Data. We use FineWeb for our main experiments because it is a diverse, cleaned dataset, and none of the quantization methods considered above sample calibration data from it, making FineWeb an unbiased corpus for our study. We also explore more technical domains, Arxiv and OpenWeb-Math (Azerbayev et al., 2023; Paster et al., 2023). Our findings that the quantization errors of different methods are strongly correlated (§4) also hold on these domains. Specifically, the average correlation across 8 pairs of methods on Llama3-70B is 0.69 on ArXiv and 0.89 on OpenWebMath.

<table><tr><td>Model</td><td>Method</td><td>Fineweb</td><td>Wiki</td></tr><tr><td rowspan="6">Qwen2.5-7B</td><td>16 bits AWQ4</td><td>11.06</td><td>6.85</td></tr><tr><td></td><td>11.35</td><td>7.09</td></tr><tr><td>NF4</td><td>11.37</td><td>7.10</td></tr><tr><td>GPTQ4</td><td>11.33</td><td>7.16</td></tr><tr><td>AWQ3</td><td>12.77</td><td>8.20</td></tr><tr><td>NF3</td><td>13.47</td><td>8.51</td></tr><tr><td>Llama3-8B</td><td>GPTQ3</td><td>12.26</td><td>8.11</td></tr><tr><td rowspan="7"></td><td>16 bits</td><td>9.57</td><td>6.14</td></tr><tr><td>AWQ4</td><td>10.13</td><td>6.56</td></tr><tr><td>NF4</td><td>10.11</td><td>6.56</td></tr><tr><td>GPTQ4</td><td></td><td>6.86</td></tr><tr><td></td><td>10.12</td><td>8.19</td></tr><tr><td>AWQ3 NF3</td><td>12.13 12.45</td><td>8.44</td></tr><tr><td>EQAT3</td><td>10.77</td><td>7.10</td></tr><tr><td>Mistral-12B</td><td>16 bits</td><td>8.67</td><td>5.75</td></tr><tr><td rowspan="6"></td><td>AWQ4</td><td>9.09</td><td>6.01</td></tr><tr><td>NF4</td><td>9.13</td><td>6.06</td></tr><tr><td>GPTQ4</td><td></td><td>6.04</td></tr><tr><td></td><td>9.02</td><td></td></tr><tr><td>AWQ3</td><td>10.32</td><td>7.03</td></tr><tr><td>NF3 GPTQ3</td><td>12.48 10.44</td><td>8.75 7.20</td></tr><tr><td rowspan="6">Llama3-70B</td><td>16 bits</td><td></td><td></td></tr><tr><td>AWQ4</td><td>7.15</td><td>2.86</td></tr><tr><td></td><td>7.42</td><td>3.27</td></tr><tr><td>NF4</td><td>7.88</td><td>3.22</td></tr><tr><td>GPTQ4</td><td>7.52</td><td>3.58</td></tr><tr><td>GPTQ3 AWQ3</td><td>9.64 8.40</td><td>6.51 4.74</td></tr></table>

Table 5: The perplexity of different quantization methods on the 10k FineWeb dataset and WikiText. Our FineWeb examples have shorter sequence lengths than WikiText (512 vs. 2048), leading to higher perplexity.

Definition of Error. We also explore an alternative definition of quantization errors based on KL divergence. Let p<sub>θ</sub>, the probability distribution of the full-precision model, be the true distribution. We can define errors as the KL divergence between the quantized and full-precision model, KL $\left( p _ { \theta } \parallel p _ { \widetilde { \theta } } \right)$ . The results of these two definitions are highly correlated $( \rho \ge 0 . 9 7 )$ . Thus, we adopt Eq. 4 for its lower computation costs.

## B The Construction of the $\mathcal { D } _ { \mathrm { l a r g e } }$

To ensure the diversity of $\mathcal { D } _ { \mathrm { l a r g e } }$ examples, we apply aggressive data deduplication based on Jaccard similarity (Jaccard, 1912), filtering out examples with 5-gram Jaccard similarity $> ~ 0 . 1$ We create a separate $\mathcal { D } _ { \mathrm { l a r g e } }$ for each quantization method and observe similar findings across them. Besides, the $\mathcal { D } _ { \mathrm { l a r g e } }$ sets of different methods under the same LLMs often have overlapping examples, with the average Jaccard similarity $\textstyle { \overline { { | A \cap B | } } } = 0 . 3 0 , 0 . 4 9 , 0 . 6 6 , 0 . { \overline { { 6 6 } } }$ on Qwen2.5-7B, Meta-Llama-3-8B, Mistral-12B, and Meta-Llama-3-70B, respectively. Therefore, in the main content, we always use the one derived from AWQ3 as $\mathcal { D } _ { \mathrm { l a r g e } }$ for simplicity.

## C Correlations Between Residual Magnitudes and Quantization Errors

We present the correlations between the quantization errors $\mathcal { E } ( \mathcal { D } ; \tilde { \theta } )$ and the residual magnitudes $\mathbb { S } ( \mathcal { D } ; \theta ; l )$ across layers of Qwen2.5-7B, Llama3- 8B, Mistral-12B, and Llama3-70B in Fig. 8, 9, 10, 11, respectively. We find that different models and methods all show that the correlations become increasingly negative in deeper layers.

## D RMSNorm and Outliers

First, we select the serverest outlier dimensions in $r ^ { ( l ) }$ , namely, the most-activated dimensions in $r ^ { ( l ) }$ that have the highest average absolute activation over the 10k FineWeb dataset. Next, we rank the entries in RMSNorm weights γ by their absolute values. We compare the median of $\gamma$ with those entries corresponding to the outlier dimensions. Table 8 shows that in the upper half layers of Llama3-70B, all outlier dimensions have much smaller weights than the median. On the other hand, in Mistral-12B, the most severe outliers have small weights (rank 1 or 2) across layers, but other outlier dimensions sometimes have much larger weights than the median. These results show that RMSNorm weights suppress the largest outlier dimensions but sometimes also promote other outlier dimensions. In Fig. 6 and Fig. 7, we show the kurtosis values before and after RMSNorm, respectively. Comparing the scale of the kurtosis values (y-axis) of the two figures, we find that the scale after RM-SNorm is much smaller, implying that RMSNorm in the upper layers suppresses outliers overall.

Similar to Wei et al. (2022), we find that the LayerNorm parameters modulate activation outliers; however, we observe mostly opposite effects, likely due to the different architectural designs (pre-LN vs. post-LN Xiong et al., 2020).

<table><tr><td>Model</td><td>(Method1, Method2)</td><td> $\rho ( \mathcal { E } _ { 1 } , \mathcal { E } _ { 2 } )$ </td></tr><tr><td>Qwen2.5-7B</td><td>(AWQ4, NF4)</td><td>0.52</td></tr><tr><td></td><td>(AWQ4, GPTQ4)</td><td>0.66</td></tr><tr><td></td><td>(AWQ4, NF3)</td><td>0.61</td></tr><tr><td></td><td>(AWQ4, GPTQ3)</td><td>0.65</td></tr><tr><td></td><td>(AWQ3, NF3)</td><td>0.78</td></tr><tr><td></td><td>(AWQ3, GPTQ3)</td><td>0.84</td></tr><tr><td></td><td>(NF4, GPTQ4)</td><td>0.57</td></tr><tr><td></td><td>(NF4, AWQ3)</td><td>0.57</td></tr><tr><td></td><td>(NF4, GPTQ3)</td><td>0.55</td></tr><tr><td></td><td>(NF3, GPTQ3)</td><td>0.75</td></tr><tr><td></td><td>(GPTQ4, AWQ3)</td><td>0.75</td></tr><tr><td></td><td>(GPTQ4, NF3)</td><td>0.70</td></tr><tr><td>Llama3-8B</td><td>(AWQ4, NF4)</td><td>0.95</td></tr><tr><td></td><td>(AWQ4, GPTQ4)</td><td>0.96</td></tr><tr><td></td><td>(AWQ4, NF3)</td><td>0.88</td></tr><tr><td></td><td>(AWQ4, EQAT3)</td><td>0.92</td></tr><tr><td></td><td>(AWQ3, NF3)</td><td>0.98</td></tr><tr><td></td><td>(AWQ3, EQAT3)</td><td>0.97</td></tr><tr><td></td><td>(NF4, GPTQ4)</td><td>0.95</td></tr><tr><td></td><td>(NF4, AWQ3)</td><td>0.87</td></tr><tr><td></td><td>(NF4, EQAT3)</td><td>0.90</td></tr><tr><td></td><td>(NF3, EQAT3)</td><td>0.96</td></tr><tr><td></td><td>(GPTQ4, AWQ3)</td><td>0.90</td></tr><tr><td></td><td>(GPTQ4, NF3)</td><td>0.90</td></tr><tr><td></td><td>(GPTQ4, EQAT3)</td><td>0.93</td></tr></table>

Table 6: The quantization errors of different methods are strongly correlated on the FineWeb. The numbers in the method names denote 3 or 4-bit precision.
<table><tr><td>Variant Pairs</td><td>ρ</td></tr><tr><td>(damp-0.005-sort, damp-0.01-sort) (damp-0.005-sort, damp-0.02-sort)</td><td>0.95 0.95</td></tr><tr><td>(damp-0.01 -sort, damp-0.02-sort)</td><td>0.95</td></tr><tr><td>(damp-0.005-sort, damp-0.01-no_sort)</td><td>0.92</td></tr><tr><td>(damp-0.01 -sort, damp-0.01-no_sort) (damp-0.02 -sort, damp-0.01-no_sort)</td><td>0.92 0.92</td></tr></table>

Table 7: Quantization error correlations of 3-bit Qwen2.5-7B GPTQ variants with different damping coefficients and activation-sorting settings. All pairs are strongly correlated.

## E More Early Exiting Results

We apply early exiting on $z ^ { ( l ) }$ and $r ^ { ( l ) }$ , the hidden states after the residual operations (see Eq. 6), across layers. Figure 12, 13, 14, and 15 shows the results on Qwen2.5-7B, Llama3-8B, Mistral-12B, and Llama3-70B, respectively, where we use AWQ3 to produce the quantized models. All the LLMs show that the NLLs of $\mathcal { D } _ { \mathrm { l a r g e } }$ (purple) decrease dramatically in the later layers, suggesting that $\mathcal { D } _ { \mathrm { l a r g e } }$ relies more on the late layers to decode the representations than $\mathcal { D } _ { \mathrm { c t r l } }$ (green).

![](images/d59a62c7ad631d5170f9b15b3098ca937da957a6fc02fe6f7d130e4a10a1de50.jpg)

![](images/ca501e465ffd6711b4fc942a10d16aceacb4e7bd56b7a54408d9ff01d62e4c1b.jpg)

![](images/675ac75348232db140893e80a8bd8072a58d5069023b978a4f910e247ce3fce9.jpg)  
Figure 6: Kurtosis values of the residual states $r ^ { ( l ) }$ from different full-precision LLMs (log scale). The large-error set $\mathcal { D } _ { \mathrm { l a r g e } }$ shows lower kurtosis values than the control set $\mathcal { D } _ { \mathrm { c t r l } }$ across layers, indicating that $\mathcal { D } _ { \mathrm { l a r g e } }$ contains fewer or less extreme outliers in the residual states.

![](images/5e0b733c6fccb188f17d5309c0f86c4fe282f92c3cb08c46e9b99926aeb73c3c.jpg)

![](images/aef0d511f6c1405f6103362e329fde54d772e5c2ebf97febb2316673095eafb0.jpg)

![](images/0020fb2604cd8d863e722d99bcd1bec0d543d1de724e20b06ebe1c93570a4357.jpg)  
Figure 7: Kurtosis values of the post-RMSNorm hidden states $\it { h ^ { ( l ) } }$ from different full-precision LLMs (log scale). The large-error set $\mathcal { D } _ { \mathrm { l a r g e } }$ shows lower kurtosis values than the control set $\mathcal { D } _ { \mathrm { c t r l } } .$ except for the last layer of Llama3-8B, indicating that overall $\mathcal { D } _ { \mathrm { l a r g e } }$ contains fewer or less extreme outliers in the hidden states after RMSNorm. In addition, compared with Fig. 6 (before RMSNorm), the scale of the kurtosis values in this figure is much smaller (y-axis), implying that RMSNorm in the upper layers tends to suppress outliers.

Qwen2.5-7B  
![](images/464a7d384b5c739e7eec67058b3949b3f7a6b41170b4150f0068803466dbab03.jpg)  
Figure 8: Correlations between the quantization errors $\mathcal { E } ( \mathcal { D } ; \tilde { \theta } )$ and the residual magnitudes $\mathbb { S } ( \mathcal { D } ; \theta ; l )$ across the upper half layers of Qwen2.5-7B. All methods show negative correlations in the last few layers.

## F Full Results on PopQA

spectively. Fig. 19 shows the popularity histogram.

Fig. 17 and 18 show the accuracy degradation and absolute accuracy values of different models, re-

![](images/17996d41ad05d35f5a6999f9f2109ac8ea0f98b64675489dfb22dbb2b05501cd.jpg)

Figure 9: Correlations between the quantization errors $\mathcal { E } ( \mathcal { D } ; \tilde { \theta } )$ and the residual magnitudes $\mathbb { S } ( \mathcal { D } ; \theta ; l )$ across the upper half layers of Llama3-8B. All methods show negative correlations in the last few layers.  
![](images/5de97a81dc9c650798350c2792a518ced213086fd0d828396af51e5f909e8492.jpg)

Figure 10: Correlations between the quantization errors $\mathcal { E } ( \mathcal { D } ; \tilde { \theta } )$ and the residual magnitudes $\mathbb { S } ( \mathcal { D } ; \theta ; l )$ across the upper half layers of Mistral-12B. All methods show negative correlations in the last few layers.  
![](images/4f78d4826cb57b763fac6c4d20e8198f8da477814fef2b651bac6f82db933165.jpg)  
Figure 11: Correlations between the quantization errors $\mathcal { E } ( \mathcal { D } ; \tilde { \theta } )$ and the residual magnitudes $\mathbb { S } ( \mathcal { D } ; \theta ; l )$ across the upper half layers of Llama3-70B. All methods show negative correlations in the last few layers.

![](images/b7e376916bd3ba123054a4159ef8f505e0adbdd7ea465bc0bddb446933dfee15.jpg)  
Figure 12: Early Exiting results on Llama3-8B. We decode the $z ^ { ( l ) }$ (A#) and $r ^ { ( l ) }$ (M#) across layers.

![](images/be1af4938a8b38cb400bc9b296a5326204126752ec54d3cbbec8c6923654359c.jpg)  
Figure 13: Early Exiting results on Llama3-8B. We decode the inputs to $z ^ { ( l ) }$ (A#) and $r ^ { ( l ) }$ (M#) across layers.

![](images/aa1febcee0ad7bb73d1e5e55614ced8045a24c9f4dbfb2d0bd16dc6d087a0072.jpg)  
Figure 14: Early Exiting results on Llama3-8B. We decode the $z ^ { ( l ) }$ (A#) and $r ^ { ( l ) }$ (M#) across layers.

![](images/f87fee7afc9295b7262b6f95a07f06ed9a30bddbe9ec6bc6ca6b3fcb02ccd350.jpg)  
Figure 15: Early Exiting results on Llama3-8B. We decodez<sup>(l)</sup> (A#) and $r ^ { ( l ) }$ (M#) across layers.

![](images/1133a6ddfa6c0744b103723454c6f1ad3a8e989e4e90e57bf06af745e30664c1.jpg)

![](images/318105a0236aa5587848829aaa2d61341925c5858f6bb0e0b0deda6ed13c42d8.jpg)

![](images/67b2bc0b9bdd1e3907e670206eb0f4c81f281ea632928932c5d6814039466f45.jpg)  
Figure 16: NLLs before quantization vs. quantization errors on different LLMs. Each dot represents a sample from FineWeb. In general, the low-frequency data (x-axis values 4) have smaller errors. On the other hand, many samples that suffer from large quantization errors have small NLLs before quantization.

![](images/7d054878abbba47e9577781e47b8bd11861962479b6fc6e6a8cfc644fc3fa810.jpg)

![](images/1b59e6713f3ccb903f53f03d780e80ae972c49ed4f08ecb973b90a7c5797f612.jpg)

![](images/3ef2acc7284f41d7e0d36da923e73b4d2a03ecc1b6b8a6f5fc739ab4b6ef1b4a.jpg)  
Figure 17: Accuracy degradation of different models on PopQA.

![](images/e46c043a3695eb595553817c4eb69071d9cfcfbe353c8a52e05c6e30ad903317.jpg)  
Figure 18: Absolute accuracies of different models on PopQA.

![](images/3a1081833d11be6eab2640b8c0c394900903a29cb9fc2c7e8d6b245bbb11ddbc.jpg)  
Figure 19: The histogram of the PopQA dataset. We partition the dataset into 8 buckets based on the sample popularity and ensure that each partition contains at least 100 samples.

<table><tr><td rowspan=1 colspan=14>Layer  Rank of Outliers’ |γ|        Median |γ|                      Outliers’|γ|</td></tr><tr><td rowspan=1 colspan=7>40    [2, 9, 1, 10, 13]                 0.1553</td><td rowspan=1 colspan=5>[0.0195, 0.0376, 0.0186, 0.0386, 0.0413]</td><td rowspan=6 colspan=2></td></tr><tr><td rowspan=1 colspan=2>41</td><td rowspan=1 colspan=5>[2, 8, 1, 9, 10]                   0.1582</td><td rowspan=1 colspan=2>[0.0170, 0.0312,</td><td rowspan=1 colspan=3>0.0142, 0.0320, 0.0330]</td></tr><tr><td rowspan=1 colspan=2>42</td><td rowspan=1 colspan=5>[5, 1, 7, 10, 8]                   0.1602</td><td rowspan=1 colspan=2>[0.0215, 0.0128,</td><td rowspan=1 colspan=3>0.0271, 0.0312, 0.0292]</td></tr><tr><td rowspan=1 colspan=7>43    [1, 6, 2, 8, 9]                   0.1621</td><td rowspan=1 colspan=5>[0.0117, 0.0275, 0.0154, 0.0315, 0.0317]</td></tr><tr><td rowspan=1 colspan=7>44    [1, 7, 2, 9, 8]                    0.1631</td><td rowspan=1 colspan=5>[0.0171, 0.0369, 0.0209, 0.0396, 0.0374]</td></tr><tr><td rowspan=1 colspan=7>45    [1, 7, 10, 2, 7]                  0.1660</td><td rowspan=1 colspan=5>[0.0130, 0.0383, 0.0420, 0.0209, 0.0383]</td></tr><tr><td rowspan=1 colspan=7>46    [1, 7, 14, 12, 2]                 0.1689</td><td rowspan=1 colspan=5>[0.0088, 0.0342, 0.0405, 0.0396, 0.0223]</td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=7>47    [1, 7, 8, 3, 14]                  0.1699</td><td rowspan=1 colspan=5>[0.0090, 0.0320, 0.0364, 0.0206, 0.0396]</td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=7>48    [1, 6, 8, 2, 9]                   0.1709</td><td rowspan=1 colspan=4>[0.0098, 0.0359, 0.0420, 0.0247,</td><td rowspan=1 colspan=1>0.0427]</td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=7>49    [1, 5, 12, 3, 12]                 0.1738</td><td rowspan=1 colspan=4>[0.0128, 0.0354, 0.0476, 0.0297,</td><td rowspan=1 colspan=1>0.0476]</td><td rowspan=8 colspan=2></td></tr><tr><td rowspan=1 colspan=7>50    [1, 6, 13, 3, 8]                  0.1758</td><td rowspan=1 colspan=4>[0.0100, 0.0327, 0.0437, 0.0273,</td><td rowspan=1 colspan=1>0.0369]</td></tr><tr><td rowspan=1 colspan=7>51    [1, 6, 11, 3,8]                  0.1777</td><td rowspan=1 colspan=2>[0.0117, 0.0369,</td><td rowspan=1 colspan=2>0.0471, 0.0312,</td><td rowspan=1 colspan=1>0.0427]</td></tr><tr><td rowspan=1 colspan=7>52    [1, 7, 8, 3, 19]                   0.1787</td><td rowspan=1 colspan=2>[0.0126, 0.0422,</td><td rowspan=1 colspan=2>0.0454, 0.0315,</td><td rowspan=1 colspan=1>0.0591]</td></tr><tr><td rowspan=1 colspan=2>53</td><td rowspan=1 colspan=2>[1, 6, 11, 3, 14]</td><td rowspan=1 colspan=3>0.1816</td><td rowspan=1 colspan=2>[0.0136, 0.0337,</td><td rowspan=1 colspan=2>0.0422, 0.0255,</td><td rowspan=1 colspan=1>0.0461]</td></tr><tr><td rowspan=1 colspan=2>54    [</td><td rowspan=1 colspan=2>1, 6, 9, 2, 11]</td><td rowspan=1 colspan=3>0.1826</td><td rowspan=1 colspan=2>[0.0138, 0.0293,</td><td rowspan=1 colspan=2>0.0376, 0.0248,</td><td rowspan=1 colspan=1>0.0388]</td></tr><tr><td rowspan=1 colspan=2>55    [</td><td rowspan=1 colspan=5>1, 5,9, 3, 11]                  0.1846</td><td rowspan=1 colspan=2>[0.0110, 0.0330,</td><td rowspan=1 colspan=2>0.0427, 0.0277,</td><td rowspan=1 colspan=1>0.0454]</td></tr><tr><td rowspan=1 colspan=7>56    [1, 6, 9, 3, 13]                  0.1865</td><td rowspan=1 colspan=2>[0.0138, 0.0378,</td><td rowspan=1 colspan=3>0.0439, 0.0297, 0.0471]</td></tr><tr><td rowspan=1 colspan=7>57    [1, 6, 9, 15, 3]                  0.1885</td><td rowspan=1 colspan=2>[0.0124, 0.0344,</td><td rowspan=1 colspan=3>0.0408, 0.0459, 0.0282]</td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=7>58    [1, 6, 9, 15, 14]                 0.1904</td><td rowspan=1 colspan=2>[0.0121, 0.0322,</td><td rowspan=1 colspan=3>0.0388, 0.0457, 0.0437]</td><td rowspan=22 colspan=2></td></tr><tr><td rowspan=1 colspan=2>59</td><td rowspan=1 colspan=5>[1, 5, 12, 23, 14]                0.1934</td><td rowspan=1 colspan=2>[0.0206, 0.0420,</td><td rowspan=1 colspan=3>0.0559, 0.0684, 0.0583]</td></tr><tr><td rowspan=1 colspan=2>60</td><td rowspan=1 colspan=5>[1, 6, 12, 27, 14]                0.1953</td><td rowspan=1 colspan=2>[0.0197, 0.0461,</td><td rowspan=1 colspan=3>0.0547, 0.0776, 0.0554]</td></tr><tr><td rowspan=1 colspan=2>61</td><td rowspan=1 colspan=5>[1, 7, 12, 24, 14]                0.1982</td><td rowspan=1 colspan=2>[0.0177, 0.0459,</td><td rowspan=1 colspan=3>0.0532, 0.0693, 0.0562]</td></tr><tr><td rowspan=1 colspan=2>62</td><td rowspan=1 colspan=5>[1, 6, 22, 14, 20]                0.2002</td><td rowspan=1 colspan=2>[0.0160, 0.0435,</td><td rowspan=1 colspan=3>0.0659, 0.0579, 0.0635]</td></tr><tr><td rowspan=1 colspan=7>63    [1, 6, 12, 22, 15]                 0.2031</td><td rowspan=1 colspan=2>[0.0199, 0.0530,</td><td rowspan=1 colspan=3>0.0640, 0.0781, 0.0679]</td></tr><tr><td rowspan=1 colspan=7>64    [1, 8, 26, 18, 16]                0.2070</td><td rowspan=1 colspan=2>[0.0272, 0.0608,</td><td rowspan=1 colspan=3>0.0903, 0.0757, 0.0723]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>65    [1, 2, 9, 8, 26]</td><td rowspan=1 colspan=2>0.2109</td><td rowspan=1 colspan=2>[0.0225, 0.0403,</td><td rowspan=1 colspan=3>0.0635, 0.0603, 0.0903]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4>66    [1, 9, 5, 6, 22]</td><td rowspan=1 colspan=2>0.2119</td><td rowspan=1 colspan=2>[0.0259, 0.0591,</td><td rowspan=1 colspan=3>0.0496, 0.0520, 0.0825]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>67    [1, 10, 6, 26, 7]</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>0.2158</td><td rowspan=1 colspan=2>[0.0266, 0.0625,</td><td rowspan=1 colspan=3>0.0537, 0.0918, 0.0559]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>68    [1, 9, 5, 27, 6]</td><td rowspan=1 colspan=3>0.2227</td><td rowspan=1 colspan=2>[0.0293, 0.0762,</td><td rowspan=1 colspan=3>0.0659, 0.1079, 0.0674]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>69    [1, 9, 2, 36, 7]</td><td rowspan=1 colspan=3>0.2266</td><td rowspan=1 colspan=2>[0.0294, 0.0732,</td><td rowspan=1 colspan=3>0.0425, 0.1201, 0.0688]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3>70    [1, 12, 28, 6, 7]</td><td rowspan=1 colspan=3>0.2285</td><td rowspan=1 colspan=2>[0.0311, 0.0737,</td><td rowspan=1 colspan=3>0.1167, 0.0654, 0.0669]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=6>71    [1, 28, 16, 6, 7]                 0.2344</td><td rowspan=1 colspan=2>[0.0312, 0.1328,</td><td rowspan=1 colspan=3>0.1006, 0.0771, 0.0776]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=6>72    [1, 33, 11, 9, 3]                 0.2441</td><td rowspan=1 colspan=2>[0.0430, 0.1650,</td><td rowspan=1 colspan=3>0.0947, 0.0918, 0.0649]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=6>73    [1, 36, 9, 23, 3]                 0.2520</td><td rowspan=1 colspan=2>[0.0483, 0.1729,</td><td rowspan=1 colspan=3>0.1089, 0.1455, 0.0723]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>74</td><td rowspan=1 colspan=5>[1, 36, 8, 18, 3]                 0.2598</td><td rowspan=1 colspan=2>[0.0459, 0.1846,</td><td rowspan=1 colspan=3>0.1060, 0.1367, 0.0767]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>75</td><td rowspan=1 colspan=5>[1, 44, 11, 10, 3]                0.2676</td><td rowspan=1 colspan=2>[0.0796, 0.2139,</td><td rowspan=1 colspan=3>0.1250, 0.1230, 0.0923]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>76</td><td rowspan=1 colspan=5>[1, 43, 3, 15, 15]                0.3047</td><td rowspan=1 colspan=2>[0.0486, 0.2217,</td><td rowspan=1 colspan=2>0.0981, 0.1533,</td><td rowspan=1 colspan=1>0.1533]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=6>77    [1, 2, 52, 4, 5]                  0.2891</td><td rowspan=1 colspan=2>[0.0554, 0.0820,</td><td rowspan=1 colspan=2>0.2080, 0.0869,</td><td rowspan=1 colspan=1>0.0913]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=6>78    [1, 4, 2, 20, 6]                   0.2812</td><td rowspan=1 colspan=2>[0.0781, 0.0972,</td><td rowspan=1 colspan=2>0.0923, 0.1562,</td><td rowspan=1 colspan=1>0.1138]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=8>79    [9, 4, 1, 3, 13]                  0.2236           [0.0688, 0.0500,</td><td rowspan=1 colspan=3>0.0309, 0.0454, 0.0747]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=8>20    [1, 6, 22, 376, 4000]            0.8594     [0.001640, 0.137695, 0</td><td rowspan=2 colspan=5>.330078, 0.648438, 0.902344]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>[1, 5, 22, 289, 2463]</td><td rowspan=1 colspan=6>0.9062</td><td rowspan=1 colspan=4>0.324219, 0.644531, 0.906250</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>22</td><td rowspan=1 colspan=1>[1, 2, 22, 283, 2402]</td><td rowspan=1 colspan=6>0.9453     [0.002487, 0.044434, 0</td><td rowspan=1 colspan=1>.369141,</td><td rowspan=1 colspan=3>0.687500, 0.941406]</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>23    [1, 2, 22, 308, 3740]</td><td rowspan=1 colspan=7>0.9570     [0.001511, 0.089355, 0.410156,</td><td rowspan=1 colspan=3>0.726562, 0.988281]</td></tr><tr><td rowspan=1 colspan=3>24    [2, 1, 261, 21, 4599]</td><td rowspan=1 colspan=7>1.0078     [0.080078, 0.001457, 0.730469,</td><td rowspan=1 colspan=3>0.365234, 1.062500]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=3>25    [2, 1, 167, 20, 1615]</td><td rowspan=1 colspan=7>1.0469     [0.102539, 0.017334, 0.691406,</td><td rowspan=1 colspan=3>0.359375, 1.023438]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>26</td><td rowspan=1 colspan=1>[2, 1, 184, 21, 1145]</td><td rowspan=1 colspan=10>1.0859     [0.092285, 0.001183, 0.722656, 0.390625, 1.039062]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>27</td><td rowspan=1 colspan=1>[5, 1, 151, 21, 840]</td><td rowspan=1 colspan=10>1.1172     [0.140625, 0.025146, 0.722656, 0.410156, 1.046875]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>21</td><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=1>[2, 1, 150, 22, 792]</td><td rowspan=1 colspan=10>1.1484     [0.094727, 0.005585, 0.757812, 0.429688, 1.070312]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>29</td><td rowspan=1 colspan=1>[2, 1, 175, 21, 945]</td><td rowspan=1 colspan=10>1.1875     [0.121582, 0.016113, 0.785156, 0.453125, 1.132812]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>30</td><td rowspan=1 colspan=1>[2, 1, 181, 19, 1080]</td><td rowspan=1 colspan=10>1.2266     [0.116211, 0.009583, 0.855469, 0.458984, 1.187500]</td><td rowspan=5 colspan=1></td></tr><tr><td rowspan=1 colspan=1>is</td><td rowspan=1 colspan=2>31     [2, 1, 211, 3486, 21]</td><td rowspan=1 colspan=5>1.2578     [0.129883,</td><td rowspan=1 colspan=5>0.000584, 0.933594, 1.273438, 0.488281]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>32</td><td rowspan=1 colspan=1>[2, 1, 212, 4982, 18]</td><td rowspan=1 colspan=3>1.2812</td><td rowspan=1 colspan=2>[0.153320,</td><td rowspan=1 colspan=5>0.000881, 1.000000, 1.335938, 0.458984]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>33</td><td rowspan=1 colspan=1>[2, 1, 200, 5098, 18]</td><td rowspan=1 colspan=3>1.2812</td><td rowspan=1 colspan=2>[0.154297,</td><td rowspan=1 colspan=5>0.001419, 1.054688, 1.421875, 0.542969]</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>34</td><td rowspan=1 colspan=1>[1, 2, 5111, 666, 21]</td><td rowspan=1 colspan=3>1.3125</td><td rowspan=1 colspan=2>[0.000793,</td><td rowspan=1 colspan=5>0.210938, 1.640625, 1.273438, 0.636719]</td></tr><tr><td rowspan=1 colspan=2>35</td><td rowspan=1 colspan=1>[1, 2, 5112, 5054, 51</td><td rowspan=1 colspan=3>10]        1.3594</td><td rowspan=1 colspan=2>[0.110352,</td><td rowspan=1 colspan=5>0.213867, 1.773438, 1.484375, 1.742188]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>36</td><td rowspan=1 colspan=1>[1,2,5114,5111, 50</td><td rowspan=1 colspan=3>64]        1.3906</td><td rowspan=1 colspan=2>[0.001472,</td><td rowspan=1 colspan=5>0.277344, 1.984375, 1.906250, 1.578125]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>37</td><td rowspan=1 colspan=1>[1, 19, 5116, 5107, 5</td><td rowspan=1 colspan=3>095]        1.3828</td><td rowspan=1 colspan=2>[0.001205,</td><td rowspan=1 colspan=5>0.808594, 2.328125, 2.109375, 1.867188]</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=2 colspan=3>38    [1,5119,5115,5118,39    [1, 28, 61, 40, 5120]</td><td rowspan=1 colspan=5>5103]     1.4219     [0.316406,</td><td rowspan=1 colspan=5>3.078125, 2.921875, 3.031250, 2.359375]</td></tr><tr><td rowspan=1 colspan=10>1.6172     [0.601562, 1.265625, 1.476562, 1.406250, 5.437500]</td><td rowspan=1 colspan=1></td></tr></table>

Table 8: We rank the RMSNorm weights γ in descending order by the absolute value of each entry. We then compare the median weight with those of the outlier dimensions. We list the ranks and weight values corresponding to the 1st through 5th severest outliers. In Llama3-70B, all outlier dimensions have weights far below the median, suggesting that the model uses RMSNorm weights to suppress outliers. In Mistral-12B, the severest outliers have small weights (rank 1 or 2) across layers, but the other outliers sometimes have large weights (rank > 5000).