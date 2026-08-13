# TokenSkip: Controllable Chain-of-Thought Compression in LLMs

Heming Xia , Chak Tou Leong , Wenjie Wang , Yongqi Li \*, Wenjie Li Department of Computing, The Hong Kong Polytechnic University University of Science and Technology of China {he-ming.xia, chak-tou.leong}@connect.polyu.hk

## Abstract

Chain-of-Thought (CoT) has been proven effective in enhancing the reasoning capabilities of large language models (LLMs). Recent advancements, such as OpenAI’s o1 and DeepSeek-R1, suggest that scaling up the length of CoT sequences during inference could further boost LLM reasoning performance. However, due to the autoregressive nature of LLM decoding, longer CoT outputs lead to a linear increase in inference latency, adversely affecting user experience, particularly when the CoT exceeds 10,000 tokens. To address this limitation, we analyze the semantic importance of tokens within CoT outputs and reveal that their contributions to reasoning vary. Building on this insight, we propose TokenSkip, a simple yet effective approach that enables LLMs to selectively skip less important tokens, allowing for controllable CoT compression. Extensive experiments across various models and tasks demonstrate the effectiveness of TokenSkip in reducing CoT token usage while preserving strong reasoning performance. Notably, when applied to Qwen2.5- 14B-Instruct, TokenSkip reduces reasoning tokens by 40% (from 313 to 181) on GSM8K, with less than a 0.4% performance drop. We release our code and checkpoints in https: //github.com/hemingkx/TokenSkip.

## 1 Introduction

Chain-of-Thought (CoT) prompting (Nye et al., 2021; Wei et al., 2022; Kojima et al., 2022) has emerged as a cornerstone strategy for enhancing Large Language Models (LLMs) in complex reasoning tasks. By eliciting step-by-step inference, CoT enables LLMs to decompose intricate problems into manageable subtasks, thereby improving their problem-solving performance (Yao et al., 2023; Wang et al., 2023; Zhou et al., 2023; Shinn et al., 2023). Recent advancements, such as OpenAI’s o1 (OpenAI et al., 2024) and DeepSeek-R1 (DeepSeek-AI et al., 2025), further demonstrate that scaling up CoT lengths from hundreds to thousands of reasoning steps could continuously improve LLM reasoning. These breakthroughs have underscored CoT’s potential to advance LLM capabilities, expanding the boundaries of AI-driven problem-solving.

![](images/7a9a35a81bc1ac31741c6fa092c408bc0d9f65331cdcbb4432d9285e900e2928.jpg)  
Figure 1: In contrast to vanilla CoT that generates all reasoning tokens sequentially, TokenSkip enables LLMs to skip tokens with less semantic importance (e.g., ) and learn shortcuts between critical reasoning tokens, facilitating controllable CoT compression.

Despite its effectiveness, the increased length of CoT sequences introduces substantial computational overhead. Due to the autoregressive nature of LLM decoding, longer CoT outputs lead to proportional increases in both inference latency and memory footprints of key-value cache. Additionally, the quadratic computational cost of attention layers further exacerbates this burden. These issues become particularly pronounced when CoT sequences extend into thousands of reasoning steps, resulting in significant computational costs and prolonged response times. While prior research has explored methods for selectively skipping reasoning steps (Ding et al., 2024; Liu et al., 2024), recent findings (Jin et al., 2024; Merrill and Sabharwal, 2024) suggest that such reductions may conflict with test-time scaling (OpenAI, 2024; Snell et al., 2025), ultimately impairing LLM reasoning performance. Therefore, striking an optimal balance between CoT efficiency and reasoning accuracy remains a critical open challenge.

In this work, we delve into CoT efficiency and seek the answer to an important question: “Does every token in the CoT output contribute equally to deriving the answer?” We empirically analyze the semantic importance of tokens within CoT outputs and reveal that their contributions to the reasoning performance vary, as depicted in Figure 2. Building on this insight, we introduce TokenSkip, a simple yet effective approach that enables LLMs to skip less important tokens within CoT sequences and learn shortcuts between critical reasoning tokens, thereby allowing for controllable CoT compression with adjustable ratios. Specifically, as shown in Figure 1, TokenSkip constructs compressed CoT training data with various compression ratios, by pruning unimportant tokens from original LLM CoT trajectories. Then, it conducts a general supervised fine-tuning process on target LLMs with this training data, facilitating LLMs to automatically trim redundant tokens during reasoning.

We conduct extensive experiments across various models, including LLaMA-3.1-8B-Instruct and the Qwen2.5-Instruct series, using two widely recognized math reasoning benchmarks: GSM8K and MATH-500. The results validate the effectiveness of TokenSkip in compressing CoT outputs while maintaining robust reasoning performance. Notably, Qwen2.5-14B-Instruct exhibits almost NO performance drop (less than 0.4%) with a 40% reduction in token usage on GSM8K. On the challenging MATH-500 dataset, LLaMA-3.1- 8B-Instruct effectively reduces CoT token usage by 30% with a performance decline of less than 4%, resulting in a 1.4 inference speedup. Further analysis underscores the coherence of TokenSkip in specified compression ratios and its potential scalability with stronger compression techniques.

TokenSkip is distinguished by its low training cost. For Qwen2.5-14B-Instruct, TokenSkip finetunes only 0.2% of the model’s parameters using LoRA. The size of the compressed CoT training data is no larger than that of the original training set, with 7,473 examples in GSM8K and 7,500 in MATH. The training is completed in approximately 2 hours for the 7B model and 2.5 hours for the 14B model on two 3090 GPUs. These characteristics make TokenSkip an efficient and reproducible approach, suitable for use in efficient and cost-effective LLM deployment.

To sum up, our key contributions are:

1. To the best of our knowledge, this work is the first to investigate the potential of enhancing CoT efficiency through token skipping, inspired by the varying semantic importance of tokens in CoT trajectories of LLMs.

2. We introduce TokenSkip, a simple yet effective approach that enables LLMs to skip redundant tokens within CoTs and learn shortcuts between critical tokens, facilitating CoT compression with adjustable ratios.

3. Our experiments validate the effectiveness of TokenSkip. When applied to Qwen2.5-14B-Instruct, TokenSkip reduces reasoning tokens by 40% (from 313 to 181) on GSM8K, with less than a 0.4% performance drop.

## 2 Background and Preliminaries

In this section, we discuss the relevant research background and present preliminary studies on token efficiency in CoT sequences, exploring its impact on the reasoning performance of LLMs.

## 2.1 Token Importance

We first investigate a critical research question to CoT efficiency: “Does every token in the CoT output contribute equally to deriving the answer?” In other words, we would like to know if there is any token redundancy in CoT sequences that could be eliminated to improve CoT efficiency.

Token redundancy has been recognized as a longstanding and fundamental issue in LLM efficiency (Hou et al., 2022; Zhang et al., 2023; Lin et al., 2024; Chen et al., 2024). Recently, it has garnered intensive research attention in prompt compression (Li et al., 2023; Jiang et al., 2023; Pan et al., 2024), which focuses on removing redundant tokens from the input prompt to reduce API token usage. To address this issue, Selective Context (Li et al., 2023) proposed to measure the importance of tokens in a piece of text based on the semantic confidence of LLMs:

$$
I _ { 1 } \left( x _ { i } \right) = - \log P \left( x _ { i } \mid \pmb { x } _ { < i } ; \pmb { \theta } _ { \mathcal { M } _ { L } } \right) ,\tag{1}
$$

where $\pmb { x } = \{ x _ { i } \} _ { i = 1 } ^ { n }$ is the given text, $x _ { i }$ denotes a token, and $\mathcal { M } _ { L }$ denotes the LLM used to compute the confidence of each token. Intuitively, such a measurement could be seamlessly applied to CoT tokens generated by LLMs. We show an example of this measurement in Figure 2.

![](images/191aca160825cecdc31e35abf51243542b4c1a2c1154a376d6a881a8b74d6fab.jpg)  
Figure 2: Visualization of token importance within a CoT sequence, with darker colors indicating higher values. This figure compares two token importance measurements: Selective Context and LLMLingua-2.

Despite its simplicity, LLMLingua-2 (Pan et al., 2024) argued that there exist two major limitations in the aforementioned measurement that hinder the compression performance. Firstly, as shown in Figure 2, the intrinsic nature of LLM perplexity leads to lower importance measures (i.e., higher confidence) for tokens at the end of the sentence. Such position dependency impacts the factual importance measurement of each token. Furthermore, the unidirectional attention mechanism in causal LMs may fail to capture all essential information needed for token importance within the text.

To tackle these limitations, LLMLingua-2 introduced utilizing a bidirectional BERT-like LM (Devlin et al., 2019) for token importance measurement. It utilizes GPT-4 (OpenAI, 2023) to label each token as “important” or not and trains the bidirectional LM with a token classification objective. The token importance is measured by the predicted probability of each token:

$$
I _ { 2 } \left( x _ { i } \right) = P \left( x _ { i } \mid \pmb { x } _ { \le n } ; \pmb { \theta } _ { \mathcal { M } _ { B } } \right) ,\tag{2}
$$

where $\mathcal { M } _ { B }$ denotes the bidirectional LM.

This study applies LLMLingua-2 as the importance measurement to CoT tokens. Similar to plain text, we observe that the semantic importance of tokens within CoT outputs varies, as shown in Figure 2. For instance, mathematical equations tend to have a greater contribution to the final answer, consistent with recent research (Ma et al., 2024). In contrast, semantic connectors such as “so” and “since” generally contribute less. These findings highlight the token redundancy in CoT outputs of LLMs and the substantial potential to enhance CoT efficiency by trimming this redundancy.

![](images/9526e6ade254f70ffbdc7c5de1800a86715ef7960c45f99b80437d636063fb1d.jpg)  
Figure 3: Recovering the compressed CoT for GSM8K math word problem using LLaMA-3.1-8B-Instruct.

## 2.2 CoT Recovery

We further explore the following research question: “Are LLMs capable of restoring the CoT process from compressed outputs?” The answer is yes. As shown in Figure 3 and detailed in Appendix A, examples restored from compressed CoTs using LLaMA-3.1-8B-Instruct demonstrate that LLMs could effectively comprehend the semantic information encoded in the compressed CoT and restore the CoT process. This capability ensures that the interpretability of compressed CoTs is maintained. Additionally, when required by users, the complete CoT process can be recovered and presented.

In summary, the empirical analysis above underscores the potential of trimming redundant tokens to enhance CoT efficiency, as well as the ability of LLMs to restore CoT from compressed outputs. However, enabling LLMs to autonomously skip redundant CoT tokens and identify shortcuts between critical reasoning tokens presents a non-trivial challenge. To the best of our knowledge, this work is the first to explore CoT compression through token skipping. In the following sections, we present our proposed methodology in detail.

## 3 TokenSkip

We introduce TokenSkip, a simple yet effective approach that enables LLMs to skip less important tokens, enabling controllable CoT compression with adjustable ratios. This section demonstrates the details of our methodology, including token pruning (§3.1), training (§3.2), and inference (§3.3).

![](images/10a0479571017291ae9c221385b0c01b29896faec5ec963442fb3777d8c1e187.jpg)  
Figure 4: Illustration of TokenSkip. During training, TokenSkip first generates CoT trajectories from the target LLM. These CoTs are then compressed to various ratios sampled from the ratio set. TokenSkip fine-tunes the LLM using compressed CoTs with mixed ratios, enabling controllable CoT inference at any desired $\gamma \in \{ \gamma _ { 0 } , \ldots , \gamma _ { z } \}$

## 3.1 Token Pruning

The key insight behind TokenSkip is that “each reasoning token contributes differently to deriving the answer.” To enhance CoT efficiency, we propose to trim redundant CoT tokens from LLM outputs and fine-tune LLMs using these trimmed CoT trajectories. The token pruning process is guided by the concept of token importance, as detailed in Section 2.1.

Specifically, given a target LLM , one of its CoT trajectories $\mathbf { c } ~ = ~ \{ c _ { i } \} _ { i = 1 } ^ { m } ,$ and a specified compression ratio $\gamma \in [ 0 , 1 ]$ for the current $^ { c , }$ TokenSkip first calculates the semantic importance of each CoT token $\{ I ( c _ { i } ) \} _ { i = 1 } ^ { m }$ , as defined in Eq (2), and then ranks the resulting scores in descending order. The empirical γ-quantile of these importance values serves as the pruning threshold:

$$
I _ { \gamma } = Q _ { \gamma } \left( I \left( c _ { 1 } \right) , . . , I \left( c _ { m } \right) \right) ,\tag{3}
$$

where $Q _ { \gamma }$ denotes the $\gamma \cdot$ -quantile (i.e. the γ-th percentile) of the multiset $\{ I ( c _ { i } ) \} _ { i = 1 } ^ { m }$ . All CoT tokens whose importance value meets or exceeds this threshold are retained, yielding the compressed CoT trajectory:

$$
\begin{array} { r } { \widetilde { c } = \left\{ c _ { i } \ : | \ : I \left( c _ { i } \right) \geq I _ { \gamma } , 1 \leq i \leq m \right\} . } \end{array}\tag{4}
$$

## 3.2 Training

Given a training dataset with N samples and a target LLM $\mathcal { M } ,$ we first obtain N CoT trajectories with . Then, we filter out trajectories with incorrect answers to ensure data quality. For the remaining trajectories, we prune each CoT with a compression ratio $\gamma$ sampled from the ratio set $\{ \gamma _ { 0 } , \ldots , \gamma _ { z } \}$ , as demonstrated in Section 3.1. For each question, compressed CoT, answer , we inserted the compression ratio $\gamma$ after the question.

Each training sample is formatted as follows:

## [EOS] γ [EOS] Compressed CoT ,

where $\langle \mathcal { Q } , \mathcal { A } \rangle$ indicates the question, answer pair. Formally, given a question x, a compression ratio $\gamma$ randomly sampled from $\{ \gamma _ { 0 } , \ldots , \gamma _ { z } \}$ , and the output sequence $\pmb { y } = \{ y _ { i } \} _ { i = 1 } ^ { l }$ , which includes the compressed CoT c and the answer a, we fine-tunes ethe target LLM , enabling it to perform chainof-thought in a compressed pattern by minimizing

$$
\mathcal { L } = \sum _ { i = 1 } ^ { l } \log P \left( y _ { i } \mid \pmb { x } , \gamma , \pmb { y } _ { < i } ; \pmb { \theta } _ { \mathcal { M } } \right) ,\tag{5}
$$

where $\pmb { y } = \{ \widetilde { c } _ { 1 } , \cdot \cdot \cdot , \widetilde { c } _ { m ^ { \prime } } , a _ { 1 } , \cdot \cdot \cdot , a _ { t } \}$ . Note that e ethe compression is performed solely on CoT sequences, and we keep the answer $\pmb { a } = \{ a _ { i } \} _ { i = 1 } ^ { t }$ unchanged. To preserve LLMs’ reasoning capabilities, we also include a portion of the original CoT trajectories in the training data, with $\gamma$ set to 1.

## 3.3 Inference

The inference of TokenSkip follows autoregressive decoding. Compared to original CoT outputs that may contain redundancy, TokenSkip facilitates LLMs to skip unimportant CoT tokens, thereby enhancing reasoning efficiency. Formally, given a question x and a desired compression ratio $\gamma \in \{ \gamma _ { 0 } , \ldots , \gamma _ { z } \}$ , the input prompt of TokenSkip follows the same format adopted in fine-tuning, which is $\mathcal { Q } \left[ \mathrm { E O S } \right] \gamma \left[ \mathrm { E O S } \right]$ . The LLM sequentially predicts the output sequence $\hat { \pmb y }$ :

$$
\hat { \pmb y } = \arg \operatorname* { m a x } _ { \pmb y ^ { * } } \sum _ { j = 1 } ^ { l ^ { \prime } } \log P \left( y _ { j } \mid \pmb x , \gamma , \pmb y _ { < j } ; \pmb \theta _ { \mathcal { M } } \right) ,
$$

where $\pmb { \hat { y } } = \{ \hat { c } _ { 1 } , \cdot \cdot \cdot , \hat { c } _ { m ^ { \prime \prime } } , \hat { a } _ { 1 } , \cdot \cdot \cdot , \hat { a } _ { t ^ { \prime } } \}$ denotes the output sequence, which includes CoT tokens cˆ and the answer aˆ. We illustrate the training and inference process of TokenSkip in Figure 4.

## 4 Experiments

## 4.1 Experimental Setup

Models and Datasets We primarily evaluate our method using LLaMA-3.1-8B-Instruct (Dubey et al., 2024) and Qwen2.5-Instruct series (Yang et al., 2024). The evaluation leverages two widelyused math reasoning benchmarks: GSM8K (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021b). For training, we use the respective training sets from both datasets. Regarding the MATH dataset, due to the computation cost, we assess our method on a subset, MATH-500, which is identical to the test set used in Lightman et al. (2024).

Implementation Details We utilize LLMLingua-2 (Pan et al., 2024) as the token importance metric to generate our compressed CoT training data. The compression ratio γ is randomly selected from the ratio set 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 for each training sample. We adopt LoRA (Hu et al., 2022) to train our models. TokenSkip is characterized by its low training cost, with training taking 2 hours for the 7B model and 2.5 hours for the 14B model on 3090 GPUs. We include more implementation details in Appendix B.1.

Baselines We compare TokenSkip to three baselines: 1) Token-efficient Prompts. Following Lee et al. (2025), we select three advanced prompts, instructing LLMs to perform CoT efficiently. These prompts, denoted as BeConcise, OnlyNumbers, and AbbreWords, are detailed in Appendix B.3; 2) Length-control Prompts. We instruct the LLM to reduce a fixed proportion of output tokens in the CoT process, denoted as LC-Prompt in Table 1; 3) Truncation. This method involves brute-force length truncation, where the maximum number of output tokens is restricted, compressing the CoT output to a fixed ratio.

Evaluation Metrics We evaluate TokenSkip using three widely used metrics: accuracy, the number of CoT tokens, and inference latency per sample. Model performance is assessed using scripts from DeepSeek-Math<sup>1</sup>. Greedy decoding is employed to generate the outputs from the target LLM. Inference latency is measured on a single NVIDIA 3090 GPU with a batch size of 1. In addition to these metrics, we report the actual compression ratio of the CoTs to assess whether the compression aligns with the specified ratio.

![](images/8fc4df950953a669329020aed508f405e32c682f8a16da1510075908aa527bc1.jpg)  
Figure 5: Compression performance of TokenSkip on Qwen2.5-Instruct models. Qwen2.5-14B-Instruct shows almost no performance drop with 40% token trimming.

## 4.2 Main Results

The performance of TokenSkip on GSM8K using the Qwen2.5-Instruct series<sup>2</sup> is illustrated in Figure 5. As the model scale increases, there is less performance degradation at higher compression ratios, indicating that larger LLMs are better at identifying shortcuts between critical reasoning tokens, enabling more efficient CoT generation. Notably, Qwen2.5-14B-Instruct exhibits almost NO performance drop (less than 0.4%) with 40% token trimming. Even at a compression ratio of 0.5, the model maintains strong reasoning capabilities, with only 2% performance degradation. These results highlight the substantial potential of TokenSkip to reduce CoT token usage and accelerate reasoning in large-scale LLMs.

Table 1 compares TokenSkip with three widely used baselines. As shown, prompting methods, including token-efficient prompts and length-control ones, fail to achieve desired compression ratios. Specifically, token-efficient prompts achieve only 0.94-0.97 compression ratios on MATH-500, with nearly no efficiency improvements; the actual ratio of LC-Prompt exceeds 0.89 even when the target is set to 0.5. While Truncation adheres to the specified ratio, it results in significant degradation in reasoning performance. Concretely, at a compression ratio of 0.5, Truncation causes a 79% accuracy drop on GSM8K and a 21% drop on MATH-500. In contrast, TokenSkip ensures adherence to various desired compression ratios (see Figure 6) while preserving strong reasoning capabilities. Notably, TokenSkip achieves an actual compression ratio of 0.53 on GSM8K with merely a 10% performance drop, resulting in a 1.8 speedup in average latency. On MATH-500, TokenSkip effectively reduces CoT token usage by 30% with a performance drop of less than 4%. These results validate the effectiveness of TokenSkip.

<table><tr><td rowspan="2">Methods</td><td rowspan="2">Ratio</td><td colspan="4">GSM8K</td><td colspan="4">MATH-500</td></tr><tr><td>Accuracy ↑</td><td>Tokens↓</td><td>Latency (s) ↓ ActRatio</td><td></td><td>Accuracy ↑</td><td></td><td>Tokens ↓ Latency (s) ↓ ActRatio</td><td></td></tr><tr><td>Original</td><td></td><td> $8 6 . 2 _ { ( 0 . 0 \downarrow ) }$ </td><td>213.17</td><td> $5 . 9 6 _ { 1 . 0 \times }$ </td><td>-</td><td> $4 8 . 6 _ { ( 0 . 0 \downarrow ) }$ </td><td>502.60</td><td> $1 6 . 3 7 _ { 1 . 0 \times }$ </td><td>-</td></tr><tr><td>BeConcise</td><td></td><td> $8 2 . 9 _ { ( 3 . 3 \downarrow ) }$ </td><td>161.32</td><td> $4 . 7 3 _ { 1 . 3 \times }$ </td><td>0.76</td><td> $4 7 . 4 _ { ( 1 . 2 \downarrow ) }$ </td><td>471.34</td><td> $1 5 . 5 4 _ { 1 . 1 \times }$ </td><td>0.94</td></tr><tr><td>OnlyNumbers</td><td></td><td> $8 3 . 2 _ { ( 3 . 0 \downarrow ) }$ </td><td>165.27</td><td> $4 . 9 5 _ { 1 . 2 \times }$ </td><td>0.78</td><td> $4 6 . 4 _ { ( 2 . 2 \downarrow ) }$ </td><td>487.00</td><td> $1 5 . 9 3 _ { 1 . 0 \times }$ </td><td>0.97</td></tr><tr><td>AbbreWords</td><td></td><td> $8 3 . 7 _ { ( 2 . 5 \downarrow ) }$ </td><td>170.33</td><td> $5 . 1 5 _ { 1 . 2 \times }$ </td><td>0.80</td><td> $4 7 . 6 _ { ( 1 . 0 \downarrow ) }$ </td><td>489.07</td><td> $1 5 . 9 4 _ { 1 . 0 \times }$ </td><td>0.97</td></tr><tr><td rowspan="3">LC-Prompt</td><td>0.9</td><td> $8 4 . 1 _ { ( 2 . 1 \downarrow ) }$ </td><td>226.37</td><td> $6 . 1 2 _ { 1 . 0 \times }$ </td><td>1.06</td><td> $4 8 . 6 _ { ( 0 . 0 \downarrow ) }$ </td><td>468.04</td><td> $1 5 . 3 9 _ { 1 . 1 \times }$ </td><td>0.93</td></tr><tr><td>0.7</td><td> $8 4 . 9 _ { ( 1 . 3 \downarrow ) }$ </td><td>209.39</td><td> $5 . 5 1 _ { 1 . 1 \times }$ </td><td>0.98</td><td> $4 8 . 4 _ { ( 0 . 4 \downarrow ) }$ </td><td>472.13</td><td> $1 5 . 5 5 _ { 1 . 1 \times }$ </td><td>0.94</td></tr><tr><td>0.5</td><td> $8 3 . 7 _ { ( 2 . 5 \downarrow ) }$ </td><td>188.82</td><td> $4 . 9 7 _ { 1 . 2 \times }$ </td><td>0.89</td><td> $4 7 . 8 _ { ( 0 . 4 \downarrow ) }$ </td><td>471.11</td><td> $1 5 . 4 8 _ { 1 . 1 \times }$ </td><td>0.94</td></tr><tr><td rowspan="3">Truncation</td><td>0.9</td><td> $7 0 . 2 _ { ( 2 6 . 0 \downarrow ) }$ </td><td>202.06</td><td> $5 . 2 9 _ { 1 . 1 \times }$ </td><td>0.95</td><td> $4 7 . 8 _ { ( 0 . 8 \downarrow ) }$ </td><td>440.33</td><td> $1 4 . 5 6 _ { 1 . 1 \times }$ </td><td>0.88</td></tr><tr><td>0.7</td><td> $2 5 . 9 _ { ( 6 0 . 3 \downarrow ) }$ </td><td>149.99</td><td> $3 . 9 7 _ { 1 . 5 \times }$ </td><td>0.70</td><td> $4 5 . 0 _ { ( 3 . 6 \downarrow ) }$ </td><td>386.89</td><td> $1 2 . 8 5 _ { 1 . 3 \times }$ </td><td>0.77</td></tr><tr><td>0.5</td><td> $7 . 0 _ { ( 7 9 . 2 \downarrow ) }$ </td><td>103.69</td><td> $2 . 9 5 _ { 2 . 0 \times }$ </td><td>0.49</td><td> $2 7 . 4 _ { ( 2 1 . 2 \downarrow ) }$ </td><td>283.70</td><td> $9 . 4 0 _ { 1 . 7 \times }$ </td><td>0.56</td></tr><tr><td rowspan="6">TokenSkip</td><td>1.0</td><td> $8 6 . 7 _ { ( 0 . 5 \uparrow ) }$ </td><td>213.60</td><td> $5 . 9 8 _ { 1 . 0 \times }$ </td><td>1.00</td><td> $4 8 . 2 _ { ( 0 . 4 \downarrow ) }$ </td><td>504.79</td><td> $1 6 . 4 3 _ { 1 . 0 \times }$ </td><td>1.00</td></tr><tr><td>0.9</td><td> $8 6 . 1 _ { ( 0 . 1 \downarrow ) }$ </td><td>198.01</td><td> $5 . 6 5 _ { 1 . 1 \times }$ </td><td>0.93</td><td> $4 7 . 8 _ { ( 0 . 8 \downarrow ) }$ </td><td>448.31</td><td> $1 5 . 2 6 _ { 1 . 1 \times }$ </td><td>0.89</td></tr><tr><td>0.8</td><td> $8 4 . 3 _ { ( 1 . 9 \downarrow ) }$ </td><td>169.89</td><td> $5 . 1 3 _ { 1 . 2 \times }$ </td><td>0.80</td><td> $4 7 . 3 \dot { } _ { ( 1 . 3 \downarrow ) }$ </td><td>398.94</td><td> $1 3 . 3 9 _ { 1 . 2 \times }$ </td><td>0.79</td></tr><tr><td>0.7</td><td> $8 2 . 5 _ { ( 3 . 7 \downarrow ) }$ </td><td>150.12</td><td> $4 . 3 6 _ { 1 . 4 \times }$ </td><td>0.70</td><td> $4 6 . 7 _ { ( 1 . 9 \downarrow ) }$ </td><td>349.13</td><td> $1 1 . 5 5 _ { 1 . 4 \times }$ </td><td>0.69</td></tr><tr><td>0.6</td><td> $8 1 . 1 _ { ( 5 . 1 \downarrow ) }$ </td><td>129.38</td><td> $3 . 8 1 _ { 1 . 6 \times }$ </td><td>0.61</td><td> $4 2 . 0 _ { ( 6 . 6 \downarrow ) }$ </td><td>318.36</td><td> $1 0 . 5 8 _ { 1 . 6 \times }$ </td><td>0.63</td></tr><tr><td>0.5</td><td> $7 8 . 2 _ { ( 8 . 0 \downarrow ) }$ </td><td>113.05</td><td> $3 . 4 0 _ { 1 . 8 \times }$ </td><td>0.53</td><td> $4 0 . 2 _ { ( 8 . 4 \downarrow ) }$ </td><td>292.17</td><td> $9 . 6 7 _ { 1 . 7 \times }$ </td><td>0.58</td></tr></table>

Table 1: Experimental results of TokenSkip on LLaMA-3.1-8B-Instruct. We report accuracy, average CoT token count (Tokens), average latency per sample, and actual compression ratio (ActRatio) for comparison.

![](images/1173a1688febbcc9a3d08b5057fe740daa3129747b2d36ad948f3b124303ddbe.jpg)  
Figure 6: Comparison of ratio adherence across different compression ratio settings. The experimental results are obtained with LLaMA-3.1-8B-Instruct on GSM8K.

In Appendix C, we illustrate additional experiments to evaluate the out-of-domain performance of TokenSkip and validate its generalizability beyond mathematical reasoning.

![](images/12b5adc4f9dd1334b64846e7a5f73770c4cc3f9697a4bdeb6285bc84fc02d855.jpg)  
Figure 7: Distribution of token importance for skipped versus retained tokens. The LLM effectively learns to skip low-importance tokens and retain critical ones.

## 4.3 Analysis

Compression Ratio In our main results, we focus on compression ratios greater than 0.5. To further investigate the performance of TokenSkip at lower compression ratios, we train an additional variant, denoted as More Ratio, with extra compression ratios of 0.3 and 0.4. As shown in Figure 6, the ratio adherence of models largely degrades at these lower ratios. We attribute this decline to the excessive trimming of reasoning tokens, which likely causes a loss of critical information in the completions, hindering the effective training of LLMs to learn CoT compression. Furthermore, we observe that the overall adherence of More Ratio is not as good as TokenSkip with the default settings, which further supports our hypothesis.

Importance Distribution To validate that the LLM learns to skip less important tokens, we analyzed the distribution of the number of tokens with various token importance. Specifically, we instructed TokenSkip with Qwen2.5-14B-Instruct to generate full CoTs $( \gamma = 1 . 0 )$ and compressed CoTs $( \gamma = 0 . 7 )$ on the GSM8K test set. CoT Tokens appearing exclusively in full CoTs but not in compressed ones were identified as “skipped”

![](images/e9c1a7fbf454bbfb31be8759f196f7ce7021ec3e8e7d405ce1f9f2079297ca82.jpg)  
Figure 8: Performance comparison of TokenSkip using different token importance metrics, evaluated with LLaMA-3.1-8B-Instruct on GSM8K.

while those present in compressed CoTs were considered “retained”. As illustrated in Figure 7, the importance distribution of skipped tokens skews towards lower values, whereas retained tokens predominantly exhibit higher importance. This demonstrates that TokenSkip effectively enables LLMs to discard less critical CoT tokens during inference.

Importance Metric Figure 8 presents a comparison of TokenSkip across different importance metrics. In addition to the metrics discussed in Section 2.1, we include GPT-4o<sup>3</sup> as a token importance upperbound for comparison. Specifically, for a given CoT trajectory, we prompt GPT-4o to trim redundant tokens according to a specified compression ratio, without adding any additional tokens. As shown in Figure 8, TokenSkip utilizing LLMLingua-2 (Pan et al., 2024) outperforms the variant with Selective Context (Li et al., 2023), which aligns with our demonstrations in Section 2.1. Additionally, the results of GPT-4o suggest that the capabilities of effective token importance metrics (beyond LLMLingua-2) could be further improved. However, the API costs associated with GPT-4o make it impractical for processing largescale datasets. In contrast, LLMLingua-2, which includes a BERT-size model, offers a cost-effective and efficient alternative for training TokenSkip.

Length Budget As outlined in Section 4.1, we adjust the maximum length budget to max\_len γ when evaluating TokenSkip on MATH-500, ensuring a fair comparison of compression ratios. However, this brute-force length truncation inevitably impacts the reasoning performance of LLMs, as

![](images/630802db383111b9006bab768a019fb433a4a4db6494e0edc8c045a934d72073.jpg)  
Figure 9: Performance comparison of TokenSkip with varying maximum length constraints, evaluated with LLaMA-3.1-8B-Instruct on the MATH-500 dataset.

LLMs are unable to complete the full generation. In this analysis, we explore whether LLMs can “think” more effectively using a compressed CoT format. Specifically, we evaluate TokenSkip under the same length budget as the original LLM (e.g., 1024 for MATH-500). The experimental results, shown in Figure 9, demonstrate a significant performance improvement of TokenSkip under this length budget, compared to those adjusted by compression ratios. Notably, with compression ratios of 0.7, 0.8, and 0.9, TokenSkip outperforms the original LLM, yielding an absolute performance increase of 1.3 to 2.6 points. These findings highlight TokenSkip’s potential to enhance the reasoning capabilities of LLMs within the same length budget.

Case Study Figure 10 presents several examples of TokenSkip, derived from the test sets of GSM8K and MATH-500. These examples clearly illustrate that TokenSkip allows LLMs to learn shortcuts between critical reasoning tokens, rather than generating shorter CoTs from scratch. For instance, in the first case, TokenSkip facilitates LLaMA-3.1-8B-Instruct to skip semantic connectors such as “of ” and “the”, as well as expressions that contribute minimally to the reasoning, such as the first sentence. Notably, we observe that numeric values and mathematical equations are prioritized for retention in most cases. This finding aligns with recent research (Ma et al., 2024), which suggests that mathematical expressions may contribute more significantly to reasoning than CoT in natural language. Furthermore, we find that TokenSkip does not reduce the number of reasoning steps but instead trims redundant tokens within those steps.

## 5 Related Work

Efficient CoT While Chain-of-Thought (CoT) enhances the reasoning performance of LLMs, it introduces significant computational overhead. Researchers have sought methods to reduce this overhead while retaining the benefits of CoT. One intuitive approach is to simplify (Marconato et al., 2024), skip (Ding et al., 2024; Liu et al., 2024), or generate reasoning steps in parallel (Ning et al., 2023). Another research direction involves compressing CoTs into latent representations (Goyal et al., 2024; Deng et al., 2024; Hao et al., 2024; Cheng and Van Durme, 2024), allowing LLMs to reason without explicitly generating discrete tokens. To mitigate CoT redundancy, Han et al. (2024) guides token consumption through dynamic token budget estimation. Kang et al. (2024) prompts GPT-4 to shorten CoT trajectories, and then fine-tunes LLMs using compressed CoTs. In contrast, this work focuses on pruning CoT tokens based on their semantic importance. Moreover, TokenSkip leverages a small LM for token pruning, significantly reducing computational overhead.

![](images/3c53dc211c9da18c4df028da147a99c2854b47cb493bf9b5470d7047336fb423.jpg)  
Figure 10: Three CoT compression examples from TokenSkip. For each sample, we list the question, original CoT outputs from corresponding LLMs, and the compressed CoT by TokenSkip. The tokens that appear in both the original CoT and the compressed CoT are highlighted in red.

Prompt Compression The growing demand for long-context prompts has led to substantial computational and memory challenges. To address this, researchers have explored various prompt compression techniques. One intuitive approach involves using a lightweight LM to generate more concise prompts (Chuang et al., 2024). Considering that natural language formats inevitably contain redundancy, some studies have introduced implicit continuous tokens to represent long-context inputs (Chevalier et al., 2023; Ge et al., 2024; Mohtashami and Jaggi, 2023). Another line of research focuses on directly compressing prompts by filtering low-informative tokens (Li et al., 2023; Jiang et al., 2023; Pan et al., 2024). For instance, Selective Context uses the perplexity of LLMs to measure token importance and removes less important tokens. LLMLingua-2 (Pan et al., 2024) introduces a small bidirectional language model for token importance measurement and trains this LM with GPT-4 compression data, which serves as the token importance metric in this work.

## 6 Conclusion

This work introduces TokenSkip, a simple yet effective approach for controllable Chain-of-Thought (CoT) compression. TokenSkip is built upon the semantic importance of CoT tokens — By selectively skipping less important tokens while preserving critical ones, TokenSkip enables LLMs to generate compressed CoTs with adjustable ratios, thereby striking an expected balance between reasoning efficiency and accuracy. Extensive experiments across various LLMs and tasks validate the effectiveness of TokenSkip. We hope our investigations in token skipping will offer valuable insights for advancing efficient CoT research and inspire future studies in this area.

## Limitations

Due to computational constraints, experiments with larger LLMs, such as Qwen2.5-32B-Instruct and Qwen2.5-72B-Instruct, were not conducted. We believe that TokenSkip could achieve a more favorable trade-off between reasoning performance and CoT token usage on these models. Additionally, the token importance measurement used in our study, derived from the LLMLingua-2 compressor (Pan et al., 2024), was not specifically trained on mathematical data. This limitation may affect the compression effectiveness, as the model is not optimized for handling numerical tokens and mathematical expressions. Furthermore, experiments with long-CoT LLMs, such as QwQ-32B-Preview, were also excluded due to computational constraints. We plan to explore these aspects in future work, as we anticipate that TokenSkip ’s potential can be further realized in these contexts.

## Acknowledgements

We thank all anonymous reviewers for their insightful comments and valuable feedback during the review process. The work described in this paper was supported by Research Grants Council of Hong Kong (PolyU/15207122, PolyU/15209724, PolyU/15207821, PolyU/15213323) and PolyU internal grants (BDWP).

## Ethics Statement

The datasets used in our experiment are publicly released and labeled through interaction with humans in English. In this process, user privacy is protected, and no personal information is contained in the dataset. The scientific artifacts that we used are available for research with permissive licenses. And the use of these artifacts in this paper is consistent with their intended use. Therefore, we believe that our research work meets the ethics of ACL.

## References

Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. 2024. An image is worth 1/2 tokens after layer 2: Plug-andplay inference acceleration for large vision-language models. In Computer Vision - ECCV 2024 - 18th European Conference, Milan, Italy, September 29-October 4, 2024, Proceedings, Part LXXXI, volume 15139 of Lecture Notes in Computer Science, pages 19–35. Springer.

Jeffrey Cheng and Benjamin Van Durme. 2024. Compressed chain of thought: Efficient reasoning through dense representations. arXiv preprint arXiv:2412.13171.

Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. 2023. Adapting language models to compress contexts. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 3829–3846, Singapore. Association for Computational Linguistics.

Yu-Neng Chuang, Tianwei Xing, Chia-Yuan Chang, Zirui Liu, Xun Chen, and Xia Hu. 2024. Learning to compress prompt in natural language formats. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 7756– 7767, Mexico City, Mexico. Association for Computational Linguistics.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. CoRR, abs/2110.14168.

DeepSeek-AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, et al. 2025. Deepseekr1: Incentivizing reasoning capability in llms via reinforcement learning. Preprint, arXiv:2501.12948.

Yuntian Deng, Yejin Choi, and Stuart Shieber. 2024. From explicit cot to implicit cot: Learning to internalize cot step by step. arXiv preprint arXiv:2405.14838.

Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171–4186, Minneapolis, Minnesota. Association for Computational Linguistics.

Mengru Ding, Hanmeng Liu, Zhizhang Fu, Jian Song, Wenbo Xie, and Yue Zhang. 2024. Break the chain: Large language models can be shortcut reasoners. CoRR, abs/2406.06580.

Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, et al. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.

Tao Ge, Hu Jing, Lei Wang, Xun Wang, Si-Qing Chen, and Furu Wei. 2024. In-context autoencoder for context compression in a large language model. In The Twelfth International Conference on Learning Representations.

Sachin Goyal, Ziwei Ji, Ankit Singh Rawat, Aditya Krishna Menon, Sanjiv Kumar, and Vaishnavh Nagarajan. 2024. Think before you speak: Training language models with pause tokens. In The Twelfth International Conference on Learning Representations.

Tingxu Han, Chunrong Fang, Shiyu Zhao, Shiqing Ma, Zhenyu Chen, and Zhenting Wang. 2024. Token-budget-aware llm reasoning. arXiv preprint arXiv:2412.18547.

Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. 2024. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021a. Measuring massive multitask language understanding. In International Conference on Learning Representations.

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021b. Measuring mathematical problem solving with the MATH dataset. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2).

Le Hou, Richard Yuanzhe Pang, Tianyi Zhou, Yuexin Wu, Xinying Song, Xiaodan Song, and Denny Zhou. 2022. Token dropping for efficient BERT pretraining. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3774–3784, Dublin, Ireland. Association for Computational Linguistics.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. 2022. Lora: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022. OpenReview.net.

Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023. LLMLingua: Compressing prompts for accelerated inference of large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 13358–13376, Singapore. Association for Computational Linguistics.

Mingyu Jin, Qinkai Yu, Dong Shu, Haiyan Zhao, Wenyue Hua, Yanda Meng, Yongfeng Zhang, and Mengnan Du. 2024. The impact of reasoning step length on large language models. In Findings of the Association for Computational Linguistics: ACL 2024, pages 1830–1842, Bangkok, Thailand. Association for Computational Linguistics.

Yu Kang, Xianghui Sun, Liangyu Chen, and Wei Zou. 2024. C3ot: Generating shorter chain-of-thought without compromising effectiveness. arXiv preprint arXiv:2412.11664.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022.

Ayeong Lee, Ethan Che, and Tianyi Peng. 2025. How well do llms compress their own chain-of-thought? A token complexity approach. CoRR, abs/2503.01141.

Yucheng Li, Bo Dong, Frank Guerin, and Chenghua Lin. 2023. Compressing context to enhance inference efficiency of large language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 6342–6353, Singapore. Association for Computational Linguistics.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Zhenghao Lin, Zhibin Gou, Yeyun Gong, Xiao Liu, yelong shen, Ruochen Xu, Chen Lin, Yujiu Yang, Jian Jiao, Nan Duan, and Weizhu Chen. 2024. Not all tokens are what you need for pretraining. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Tengxiao Liu, Qipeng Guo, Xiangkun Hu, Cheng Jiayang, Yue Zhang, Xipeng Qiu, and Zheng Zhang. 2024. Can language models learn to skip steps? In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Ilya Loshchilov and Frank Hutter. 2019. Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019. Open-Review.net.

Yiran Ma, Zui Chen, Tianqiao Liu, Mi Tian, Zhuo Liu, Zitao Liu, and Weiqi Luo. 2024. What are step-level reward models rewarding? counterintuitive findings from mcts-boosted mathematical reasoning. CoRR, abs/2412.15904.

Emanuele Marconato, Stefano Teso, Antonio Vergari, and Andrea Passerini. 2024. Not all neuro-symbolic concepts are created equal: Analysis and mitigation of reasoning shortcuts. Advances in Neural Information Processing Systems, 36.

William Merrill and Ashish Sabharwal. 2024. The expressive power of transformers with chain of thought. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Amirkeivan Mohtashami and Martin Jaggi. 2023. Random-access infinite context length for transformers. In Thirty-seventh Conference on Neural Information Processing Systems.

Xuefei Ning, Zinan Lin, Zixuan Zhou, Zifu Wang, Huazhong Yang, and Yu Wang. 2023. Skeleton-ofthought: Large language models can do parallel decoding. Proceedings ENLSP-III.

Maxwell I. Nye, Anders Johan Andreassen, Guy Gur-Ari, Henryk Michalewski, Jacob Austin, David Bieber, David Dohan, Aitor Lewkowycz, Maarten Bosma, David Luan, Charles Sutton, and Augustus Odena. 2021. Show your work: Scratchpads for intermediate computation with language models. CoRR, abs/2112.00114.

OpenAI. 2023. GPT-4 technical report. CoRR, abs/2303.08774.

OpenAI. 2024. Learning to reason with llms.

OpenAI et al. 2024. Openai o1 system card. Preprint, arXiv:2412.16720.

Zhuoshi Pan, Qianhui Wu, Huiqiang Jiang, Menglin Xia, Xufang Luo, Jue Zhang, Qingwei Lin, Victor Rühle, Yuqing Yang, Chin-Yew Lin, H. Vicky Zhao, Lili Qiu, and Dongmei Zhang. 2024. LLMLingua-2: Data distillation for efficient and faithful task-agnostic prompt compression. In Findings of the Association for Computational Linguistics: ACL 2024, pages 963– 981, Bangkok, Thailand. Association for Computational Linguistics.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Charlie Victor Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. 2025. Scaling test-time compute optimally can be more effective than scaling LLM parameters. In The Thirteenth International Conference on Learning Representations.

Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant. 2019. CommonsenseQA: A question answering challenge targeting commonsense knowledge. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4149–4158, Minneapolis, Minnesota. Association for Computational Linguistics.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference

on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022.

An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, Guanting Dong, Haoran Wei, Huan Lin, Jialong Tang, Jialin Wang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Ma, Jianxin Yang, Jin Xu, Jingren Zhou, Jinze Bai, Jinzheng He, Junyang Lin, Kai Dang, Keming Lu, Keqin Chen, Kexin Yang, Mei Li, Mingfeng Xue, Na Ni, Pei Zhang, Peng Wang, Ru Peng, Rui Men, Ruize Gao, Runji Lin, Shijie Wang, Shuai Bai, Sinan Tan, Tianhang Zhu, Tianhao Li, Tianyu Liu, Wenbin Ge, Xiaodong Deng, Xiaohuan Zhou, Xingzhang Ren, Xinyu Zhang, Xipin Wei, Xuancheng Ren, Xuejing Liu, Yang Fan, Yang Yao, Yichang Zhang, Yu Wan, Yunfei Chu, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, Zhifang Guo, and Zhihao Fan. 2024. Qwen2 technical report. CoRR, abs/2407.10671.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Zhenyu Zhang, Ying Sheng, Tianyi Zhou, Tianlong Chen, Lianmin Zheng, Ruisi Cai, Zhao Song, Yuandong Tian, Christopher Re, Clark Barrett, Zhangyang Wang, and Beidi Chen. 2023. H2o: Heavy-hitter oracle for efficient generative inference of large language models. In Thirty-seventh Conference on Neural Information Processing Systems.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, and Zheyan Luo. 2024. LlamaFactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pages 400– 410, Bangkok, Thailand. Association for Computational Linguistics.

Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc V. Le, and Ed H. Chi. 2023. Least-to-most prompting enables complex reasoning in large language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

## Appendix

## A CoT Recovery

In this section, we provide the detailed prompt for our recovery experiments, which is illustrated in Figure 11. Besides, we present the CoT recovery result from GPT-4o (OpenAI, 2023) in Figure 12. The recovered results demonstrate that GPT-4o could understand the compressed CoT content and correctly restore the original CoT process.

![](images/75f030fc5add1933956a69b1d960e2c0c5428aa20e54724b50a7f3611adfd535.jpg)  
Figure 11: Input prompt for LLaMA-3.1-8B-Instruct designed to recover the compressed CoT from a GSM8K math problem.

Revovering the Compressed Chain-of-Thought   
Compressed CoT: break down Deanna 26 Marcus   
five younger 26 - 5 21 Marcus half Leo’s age twice   
Marcus Marcus 21, Leo’s age 2 x 21 = 42.   
Recovered CoT: 1. We know that Deanna is 26 years   
old. 2. Marcus is five years younger than Deanna.   
So, Marcus’s age is 26 5 = 21. 3. Marcus is also   
half of Leo’s age, which means Leo’s age is twice   
Marcus’s age. 4. Since Marcus is 21 years old, Leo’s   
age is 2 21 = 42. So, Leo is 42 years old.  
Figure 12: Recovering the compressed CoT for GSM8K math word problem using GPT-4o.

## B Experimental Details

## B.1 Implementation Details

We utilize LLMLingua-2 (Pan et al., 2024) as the token importance metric to generate our compressed CoT training data. The compression ratio γ is randomly selected from 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 for each training sample. We adopt LoRA (Hu et al., 2022), an efficient and reproducible approach that has been widely verified as effective in LLM fine-tuning, to train our models. The rank r is set to 8, and the scaling parameter α is set to 16. We train the models for 3 epochs on both datasets. The peak learning rate is set to 5e-5, following a cosine decay schedule. We use AdamW (Loshchilov and Hutter, 2019) for optimization, with a warmup ratio of 0.1. We implement our training process using the LLaMA-Factory (Zheng et al., 2024) library. Inference for both our method and all baselines is performed using the Huggingface transformers package. During inference, the maximum number of tokens max\_len is set to 512 for GSM8K and 1024 for MATH<sup>4</sup>. All experiments are conducted using Pytorch 2.1.0 on 2 NVIDIA GeForce RTX 3090 GPU (24GB) with CUDA 12.1, and an Intel(R) Xeon(R) Platinum 8370C CPU with 32 cores.

## B.2 Detailed Results with Qwen

We provide detailed experimental results of the Qwen2.5-Instruct series evaluated on GSM8K in Table 2. As the model scale increases, there is less performance degradation at higher compression ratios, indicating that larger LLMs are better at identifying shortcuts between critical reasoning tokens, enabling more efficient CoT generation.

<table><tr><td>Scale</td><td>Methods</td><td>Ratio</td><td>Accuracy</td><td>Tokens</td><td>ActRatio</td></tr><tr><td rowspan="5">3B</td><td>Original</td><td>-</td><td> $8 3 . 7 _ { ( 0 . 0 \downarrow ) }$ </td><td>314.87</td><td>-</td></tr><tr><td rowspan="5">TokenSkip</td><td>1.0</td><td> $8 3 . 4 _ { ( 0 . 3 \downarrow ) }$ </td><td>318.79</td><td>1.00</td></tr><tr><td>0.9</td><td> $8 3 . 2 _ { ( 0 . 5 \downarrow ) }$ </td><td>262.99</td><td>0.83</td></tr><tr><td>0.8</td><td> $8 1 . 6 _ { ( 2 . 1 \downarrow ) }$ </td><td>250.71</td><td>0.79</td></tr><tr><td>0.7</td><td> $8 0 . 1 _ { ( 3 . 6 \downarrow ) }$ </td><td>233.03</td><td>0.73</td></tr><tr><td>0.6 0.5</td><td> $7 7 . 3 \dot { ( 6 . 4 \downarrow ) }$   $7 4 . 4 _ { ( 9 . 3 \downarrow ) }$ </td><td>199.55 170.55</td><td>0.63 0.54</td></tr><tr><td rowspan="6">7B</td><td>Original</td><td>-</td><td> $9 1 . 4 _ { ( 0 . 0 \downarrow ) }$ </td><td>297.83</td><td>-</td></tr><tr><td></td><td>1.0</td><td> $9 1 . 7 _ { ( 0 . 3 \uparrow ) }$ </td><td>295.78</td><td>1.00</td></tr><tr><td rowspan="5">TokenSkip</td><td>0.9</td><td> $9 1 . 1 _ { ( 0 . 3 \downarrow ) }$ </td><td>254.77</td><td>0.86</td></tr><tr><td>0.8</td><td> $9 0 . 1 \dot { _ { ( 1 . 3 \downarrow ) } }$ </td><td>237.27</td><td>0.80</td></tr><tr><td>0.7</td><td> $8 9 . 9 _ { ( 1 . 5 \downarrow ) }$ </td><td>216.73</td><td>0.73</td></tr><tr><td>0.6</td><td> $8 7 . 9 _ { ( 3 . 5 \downarrow ) }$ </td><td>178.07</td><td>0.60</td></tr><tr><td>0.5</td><td> $8 6 . 0 _ { ( 5 . 4 \downarrow ) }$ </td><td>151.44</td><td>0.51</td></tr><tr><td rowspan="5">14B</td><td>Original</td><td>-</td><td> $9 3 . 1 _ { ( 0 . 0 \downarrow ) }$ </td><td>313.11</td><td>-</td></tr><tr><td rowspan="5">TokenSkip</td><td>1.0</td><td> $9 3 . 0 _ { ( 0 . 1 \downarrow ) }$ </td><td>314.55</td><td>1.00</td></tr><tr><td>0.9</td><td> $9 3 . 3 _ { ( 0 . 2 \uparrow ) }$ </td><td>269.22</td><td>0.86</td></tr><tr><td>0.8</td><td> $^ { 9 3 . 2 } _ { \textnormal { c a . } } ( 0 . 1 \uparrow )$ </td><td>247.24</td><td>0.79</td></tr><tr><td>0.7</td><td> $9 3 . 4 _ { ( 0 . 3 \uparrow ) }$ </td><td>218.62</td><td>0.70</td></tr><tr><td>0.6</td><td> $9 2 . 7 _ { ( 0 . 4 \downarrow ) }$ </td><td></td><td>180.68</td><td>0.57</td></tr><tr><td rowspan="2"></td><td>0.5</td><td></td><td>156.85</td><td>0.50</td></tr><tr><td></td><td> $9 1 . 4 _ { ( 1 . 7 \downarrow ) }$ </td><td></td><td></td></tr></table>

Table 2: Experimental results on the Qwen2.5-Instruct series. We report accuracy, average CoT token count, and actual compression ratio (ActRatio) for comparison.

## B.3 Detailed Prompts

We demonstrate the detailed prompts used in our main experiments in Table 3.

<table><tr><td>Methods</td><td>Detailed Prompts</td></tr><tr><td>BeConcise</td><td>Be concise.</td></tr><tr><td>OnlyNumbers</td><td>Only use numbers or equations.</td></tr><tr><td>AbbreWords</td><td>Abbreviate words as much as possible.</td></tr><tr><td>LC-Prompt</td><td>Please reduce 50% of the words in your Chain-of-Thought process.</td></tr></table>

Table 3: Details for prompt-based baselines.

## C Additional Experiments

## C.1 Out-of-domain Evaluation

To assess the generalizability of TokenSkip beyond the training domain data, we conducted an additional out-of-domain evaluation. Specifically, we fine-tuned LLaMA-3.1-8B-Instruct on the MATH training data and evaluated TokenSkip on both the in-domain MATH-500 and two outof-domain benchmarks, GSM8K and MMLU-STEM (Hendrycks et al., 2021a). MMLU-STEM includes a diverse set of STEM subjects from the full MMLU dataset.

The results in Table 4 suggest that TokenSkip maintains strong generalizability on out-of-domain scenarios. The model adheres closely to specified compression ratios while preserving accuracy. Notably, on the MMLU-STEM test set, TokenSkip exhibits comparable performance to the original LLM with 40% token trimming. Even at a compression ratio of 0.5, the model maintains strong reasoning capabilities, with only 0.4% absolute performance degradation.

## C.2 Evaluation Beyond Math

To demonstrate the generalizability of TokenSkip beyond mathematical reasoning, we present results on CommonsenseQA (Talmor et al., 2019), a widely used multiple-choice question answering dataset that requires diverse commonsense knowledge to predict correct answers. For this experiment, we used 9,700 samples from the training set and evaluated TokenSkip on the validation set.

Experimental results on Qwen2.5-Instruct models are shown in Table 5, which demonstrate that TokenSkip effectively reduces CoT length by 50% without any performance degradation. These findings further highlight the generalizability of TokenSkip beyond the mathematical reasoning.

<table><tr><td>Methods</td><td>Ratio</td><td>Accuracy</td><td>Tokens ActRatio</td><td></td></tr><tr><td colspan="5">MATH-500 (in-domain)</td></tr><tr><td>Original</td><td>-</td><td> $4 8 . 6 _ { ( 0 . 0 \downarrow ) }$ </td><td>502.60</td><td>-</td></tr><tr><td rowspan="6"></td><td>1.0</td><td> $4 8 . 2 _ { ( 0 . 4 \downarrow ) }$ </td><td>504.79</td><td>1.00</td></tr><tr><td>0.9</td><td> $4 7 . 8 _ { ( 0 . 8 \downarrow ) }$ </td><td>448.31</td><td>0.89</td></tr><tr><td>0.8</td><td> $4 7 . 3 _ { ( 1 . 3 \downarrow ) }$ </td><td>398.94</td><td>0.79</td></tr><tr><td>0.7</td><td> $4 6 . 7 _ { ( 1 . 9 \downarrow ) }$ </td><td>349.13</td><td>0.69</td></tr><tr><td>0.6</td><td> $4 2 . 0 _ { ( 6 . 6 \downarrow ) }$ </td><td>318.36</td><td>0.63</td></tr><tr><td>0.5</td><td> $4 0 . 2 \dot { } _ { ( 8 . 4 \dot { \downarrow } ) }$ </td><td>292.17</td><td>0.58</td></tr><tr><td colspan="5">GSM8K (out-of-domain)</td></tr><tr><td>Original</td><td>-</td><td> $8 6 . 2 _ { ( 0 . 0 \downarrow ) }$ </td><td>213.17</td><td>一</td></tr><tr><td rowspan="6">TokenSkip</td><td>1.0</td><td> $8 6 . 0 _ { ( 0 . 2 \downarrow ) }$ </td><td>214.49</td><td>1.00</td></tr><tr><td>0.9</td><td> $8 4 . 9 _ { ( 1 . 3 \downarrow ) }$ </td><td>201.84</td><td>0.95</td></tr><tr><td>0.8</td><td> $8 3 . 7 _ { ( 2 . 5 \downarrow ) }$ </td><td>175.24</td><td>0.82</td></tr><tr><td>0.7</td><td> $8 2 . 6 _ { ( 3 . 6 \downarrow ) }$ </td><td>152.32</td><td>0.71</td></tr><tr><td>0.6</td><td> $7 9 . 8 _ { ( 6 . 4 \downarrow ) }$ </td><td>136.95</td><td>0.64</td></tr><tr><td>0.5</td><td> $7 6 . 6 _ { ( 9 . 6 \downarrow ) }$ </td><td>122.55</td><td>0.58</td></tr><tr><td colspan="5">MMLU-STEM (out-of-domain)</td></tr><tr><td>Original</td><td>-</td><td> $5 8 . 5 _ { ( 0 . 0 \downarrow ) }$ </td><td>356.31</td><td>-</td></tr><tr><td rowspan="6"></td><td>1.0</td><td> $5 8 . 4 _ { ( 0 . 1 \downarrow ) }$ </td><td>354.25</td><td>1.00</td></tr><tr><td>0.9</td><td> $5 9 . 4 _ { ( 0 . 9 \uparrow ) }$ </td><td>327.18</td><td>0.92</td></tr><tr><td>0.8</td><td> $5 9 . 3 _ { ( 0 . 8 \uparrow ) }$ </td><td>286.15</td><td>0.80</td></tr><tr><td>0.7</td><td> $5 8 . 9 _ { ( 0 . 4 \uparrow ) }$ </td><td>257.26</td><td>0.72</td></tr><tr><td>0.6</td><td> $5 9 . 2 _ { ( 0 . 7 \uparrow ) }$ </td><td>225.33</td><td>0.63</td></tr><tr><td>0.5</td><td> $5 8 . 1 _ { ( 0 . 4 \downarrow ) }$ </td><td>188.87</td><td>0.53</td></tr></table>

Table 4: Out-of-domain results on LLaMA-3.1-8B-Instruct. We report accuracy, average CoT token count, and actual compression ratio (ActRatio) for comparison.

<table><tr><td>Methods</td><td>Ratio</td><td>Accuracy</td><td>Tokens ActRatio</td><td></td></tr><tr><td colspan="5">Qwen2.5-7B-Instruct</td></tr><tr><td>Original</td><td>-</td><td> $8 0 . 3 _ { ( 0 . 0 \downarrow ) }$ </td><td>272.13</td><td></td></tr><tr><td>TokenSkip</td><td>1.0 0.9 0.8 0.7 0.6</td><td> $8 0 . 4 _ { ( 0 . 1 \uparrow ) }$   $8 0 . 9 _ { ( 0 . 6 \uparrow ) }$   $8 1 . 1 _ { ( 0 . 8 \uparrow ) }$   $8 2 . 0 \dot { _ { ( 1 . 7 \uparrow ) } }$   $8 1 . 5 _ { ( 1 . 2 \uparrow ) }$ </td><td>273.64 245.70 218.73 188.78 153.17</td><td>1.00 0.90 0.80 0.69 0.56</td></tr><tr><td></td><td>0.5 Qwen2.5-14B-Instruct</td><td> $8 0 . 6 _ { ( 0 . 3 \uparrow ) }$ </td><td>128.43</td><td>0.47</td></tr><tr><td>Original</td><td>-</td><td> $8 2 . 1 _ { ( 0 . 0 \downarrow ) }$ </td><td>247.81</td><td></td></tr><tr><td colspan="4">1.0 0.9 0.8 TokenSkip 0.7 0.6 0.5</td><td>247.34 1.00 221.75 0.95 199.07 0.82</td></tr></table>

Table 5: Experimental results on CommonsenseQA with Qwen2.5-Instruct models. We report accuracy, average CoT token count, and actual compression ratio (ActRatio) for comparison.