# DSCD: Large Language Model Detoxification with Self-Constrained Decoding

Ming Dong<sup>1,2,3,</sup>∗, Jinkui Zhang<sup>1,2,3,</sup>\*, Bolong Zheng,<sup>4</sup>

Xinhui Tu<sup>1,2,3</sup>, Po Hu<sup>1,2,3,</sup>†, Tingting He<sup>1,2,3†</sup>

<sup>1</sup>Hubei Provincial Key Laboratory of Artificial Intelligence and Smart Learning <sup>2</sup>National Language Resources Monitoring and Research Center for Network Media <sup>3</sup> Central China Normal University, <sup>4</sup>Wuhan University of Technology {dongming, tuxinhui, phu, tthe}@ccnu.edu.cn zhangjinkui@mails.ccnu.edu.cn, bolongzheng@whut.edu.cn

## Abstract

Detoxification in large language models (LLMs) remains a significant research challenge. Existing decoding detoxification methods are all based on external constraints, which require additional resource overhead and lose generation fluency. This work innovatively proposes Detoxification with Self-Constrained Decoding (DSCD), a novel method for LLMs detoxification without parameter fine-tuning. DSCD strengthens the inner next-token distribution of the safety layer while weakening that of hallucination and toxic layer during output generation. This effectively diminishes toxicity and enhances output safety. DSCD offers lightweight, high compatibility, and plug-andplay capabilities, readily integrating with existing detoxification methods for further performance improvement. Extensive experiments on representative open-source LLMs and public datasets validate DSCD’s effectiveness, demonstrating state-of-the-art (SOTA) performance in both detoxification and generation fluency, with superior efficiency compared to existing methods. These results highlight DSCD’s potential as a practical and scalable solution for safer LLM deployments. For more details, please refer to the project repository: https://github.com/ZHANGJINKUI/DSCD.

## 1 Introduction

The rapid proliferation of large language models (LLMs) (Jiang et al., 2023; OpenAI, 2023; Touvron et al., 2023) presents notable security risks. These models can generate harmful or biased content, including discriminatory statements and misinformation. Moreover, LLMs can be misused to disseminate instructions such as creating dangerous weapons (Perez et al., 2022). Addressing these security challenges is crucial for responsible LLMs development and deployment. The process of constraining or removing toxicity from LLMs after pre-training is referred to as LLMs detoxification. Current detoxification methods for LLMs can be broadly classified into two main categories: alignment after pre-training and knowledge editing during deployment. These approaches correspond to distinct stages in the application of LLMs.

![](images/04e71f15382ae52a55b0660c8f2774cbb77ef267f7c9dee8d19d1bfccb79e541.jpg)  
Figure 1: Self-Constrained Decoding at each nexttoken.

Alignment techniques, such as Reinforcement Learning from Human Feedback (RLHF) (Bai et al., 2022) and Direct Preference Optimization (DPO) (Rafailov et al., 2023), are among the most important safety measures applied in the post pretraining phase. Recently, some studies on alignment have shifted focus toward constraining the probability distribution of generated tokens during the decoding phase. Methods like SafeDecoding (Xu et al., 2024) and Adversarial Contrastive Decoding (ACD) (Zhao et al., 2024) have significantly enhanced the safety of LLMs by directly imposing constraints during decoding. However, both approaches rely heavily on external models or datasets to function effectively, which introduces certain limitations. Specifically, these external dependencies increase the resource overhead (e.g., building models and datasets) and, in some cases, may compromise the fluency and helpfulness of the generated content. Therefore, while these methods represent important steps toward safer LLMs, their reliance on external constraints may pose challenges to broader applicability.

During the deployment phase, knowledge editing-based detoxification methods, such as DINM (Wang et al., 2024c), are capable of addressing specific toxicities exposed by adversarial inputs. However, these methods come with notable limitations. First, DINM relies on processing single samples for individual relocation and editing, which results in significant computational inefficiency. Second, DINM only diminishes toxicity in adversarial inputs that have been previously exposed to LLMs. These challenges highlight the need for more efficient and generalizable detoxification techniques.

Given the significant security risks posed by LLMs, it is essential that detoxification strategies proactively prevent harmful content generation. To address this challenge, we propose Detoxification with Self-Constrained Decoding (DSCD), a novel approach for diminishing toxicity without any parameter fine-tuning. DSCD operates by adjusting the next-token distribution throughout the LLM decoding process, encouraging the selection of safer token layers and discouraging toxic or hallucinated ones (see Fig. 1). It detects toxic regions at the token level and diminishes toxicity accordingly. Unlike methods that rely on external constraints, DSCD introduces entirely self-imposed constraints during decoding, ensuring the fluency and naturalness of the generated text while enhancing its safety. DSCD is lightweight, efficient, and designed for seamless integration into existing knowledge editing workflows. Notably, it bypasses the precise location of toxic regions, further accelerating detoxification. These features make DSCD a robust and practical solution when compared to resource-intensive methods.

The contributions of this work are summarized as follows:

• We introduce DSCD, a lightweight, highly compatible, and plug-and-play detoxification method that ensures fluent text generation.

• DSCD includes two modes: MODE-1 precisely localizes toxic regions for high performance, while MODE-2 rapidly identifies and detoxifies toxic content for efficiency.

• Extensive experiments show that DSCD achieves state-of-the-art results in both fluency and efficiency, both as a standalone method and when integrated with existing approaches.

## 2 Preliminary

## 2.1 Task Definition

Given an adversarial query I, the LLM is prompted to generate a corresponding output O:

$$
O = \mathrm { L L M } ( I ) = P ( O \mid I ) = \prod _ { t = 1 } ^ { | O | } \mathrm { P } ( y _ { t } \mid y _ { t < } , I ) ,\tag{1}
$$

where $P ( \cdot | \cdot )$ represents the probability of LLMs that generating the next character given the input I and the tokens $y _ { t < } = \{ y _ { 1 } , \cdot \cdot \cdot , y _ { t - 1 } \}$ generated before time step t. The task of LLM detoxification is to prevent the output O from containing toxic content.

## 2.2 DINM

DINM (Wang et al., 2024c) is the first study to detoxify LLMs by employing a two-stage knowledge editing method. In the first stage, toxic knowledge is identified by comparing the hidden states of the safe and unsafe generated context sequences within the same layer of the model. The layer with the largest hidden state difference between the safe and unsafe generations is identified as the toxic layer. In the second stage, knowledge editing is performed using the total loss function to update the parameters of the toxic layer, thereby diminishing the toxicity of the LLM.

Inspired by DINM’s sequence-level toxic layer location, we propose token-level toxic regions location, which allows for more precise location of toxic regions, as detailed in Section 3.2. Since DSCD is a plug-and-play method, it can be flexibly integrated into DINM, achieving better detoxification performance and higher detoxification efficiency than using DINM alone.

## 2.3 DOLA

DOLA (Chuang et al., 2024) introduces the concept of early exit layers (Teerapittayanon et al., 2016), allowing the output distribution at any layer to serve as the final output of the LLM. By analyzing the token probability distributions at different layers, DOLA identifies the hallucination layer—where hallucinated tokens are concentrated—and the mature layer, which contains the most factual knowledge. During the decoding process (Li et al., 2023), DOLA amplifies the influence of the mature layer while attenuating that of the premature or hallucination layer, thus minimizing hallucinated content in LLM outputs. Inspired by this approach, we adopt a similar strategy for detoxification: by precisely identifying toxic regions, we reduce their influence in the final output to diminish the toxicity of LLMs.

![](images/995294b6276df58ca3f8db6c2e36f8241e6c9083986ecae304382d514258cb45.jpg)  
Figure 2: Overview of the DSCD framework, consisting of the location of toxic regions and the computation of next-token distributions.

## 3 DSCD: Detoxification with Self-Constrained Decoding

## 3.1 Early Exit

The pipeline of LLMs orderly includes an embedding layer, several stacked transformer layers, and an affine layer. Specifically:

Embedding Layer: The layer embeds a sequence of input tokens $\{ x _ { 1 } , x _ { 2 } , \dotsc , x _ { t - 1 } \}$ into their corresponding vector representations, with each token associated with a specific vector.

Transformer Layers: The embedding sequence of vectors $H _ { 0 } ~ = ~ \{ h _ { 1 } ^ { ( 0 ) } , \ldots , h _ { t - 1 } ^ { ( 0 ) } \}$ are then processed sequentially through multiple transformer layers. After each layer, a new sequence of vectors $H _ { j }$ is generated, denoting the output after the j-th layer.

Affine Layer: After processing through the transformer layers, the final sequence of vectors are fed into the affine layer (denoted as $\phi ( \cdot ) )$ , which calculates and outputs the distribution of each possible next token $x _ { t }$ appearing in the vocabulary set $\mathcal { X }$

$$
q ( x _ { t } \mid x _ { < t } ) = \mathrm { s o f t m a x } \big ( \phi ( h _ { t } ^ { ( N ) } ) \big ) _ { x _ { t } } , \quad x _ { t } \in \mathcal { X } .\tag{2}
$$

The above describes the method used in general

LLMs for predicting the probability of the next token using the N-th layer as LLMs’ output layer. Early Exit (Teerapittayanon et al., 2016) can output the next-token distribution of any layer in LLMs. We leverage the property of early exit to impose inter-layer constraints in LLMs, resulting in a modification of the next-token distribution in the final layer.

## 3.2 Regions Location

<table><tr><td>Notation</td><td>Description</td></tr><tr><td>T</td><td>Toxic layer of LLMs</td></tr><tr><td>S</td><td>Safety layer of LLMs</td></tr><tr><td>E</td><td>Output layer of LLMs</td></tr><tr><td>H</td><td>Hallucination layer of LLMs</td></tr></table>

Table 1: Notations of different layers in DSCD

In a Transformer-based LLM, each layer l consists of an attention block and an MLP. Given an input sequence $Y _ { u n s a f e }$ with potentially harmful content, the model maps it to the initial hidden state $h _ { 0 } ^ { \mathrm { u n s a f e } }$ via an embedding layer, and then processes it layer by layer. Following DINM (Wang et al., 2024c), we locate toxic layers based on the intermediate hidden states:

$$
h _ { \ell } ^ { \mathrm { u n s a f e } } = h _ { \ell - 1 } ^ { \mathrm { u n s a f e } } + \mathrm { M L P } _ { \ell } \left( h _ { \ell - 1 } ^ { \mathrm { u n s a f e } } + \mathrm { A t t } _ { \ell } \left( h _ { \ell - 1 } ^ { \mathrm { u n s a f e } } \right) \right)\tag{3}
$$

The hidden state $h _ { l } ^ { \mathrm { u n s a f e } }$ is generated by the model after processing the input sequence $Y _ { u n s a f e }$ at layer l. Similarly, we can obtain the corresponding hidden state $h _ { l } ^ { \mathrm { s a f e } }$ by applying the model’s layer l to the safe sequence $Y _ { s a f e }$ . This helps us locate the specific layer containing harmful content.

$$
\ell _ { \mathrm { t o x i c } } = \operatorname * { a r g m a x } _ { l \in \{ 1 , 2 , \ldots , E \} } { \| h _ { \ell } ^ { \mathrm { s a f e } } - h _ { \ell } ^ { \mathrm { u n s a f e } } \| _ { 2 } }\tag{4}
$$

However, the toxic layer location method of DINM (Wang et al., 2024c) does not locate the toxic layer for each individual token but instead treats an entire input sequence as a whole to determine the toxic layer. As a result, the toxic layer is the same across all tokens in a sequence. Since DOLA (Chuang et al., 2024) points out that toxic information does not always appear in the same layer, we believe that the method of DINM for toxic location is imprecise and can only be considered sequence-level location. Therefore, we propose locating the toxic regions for each token individually, rather than relying on a single toxic layer. DSCD enables toxicity detection at the token-level, as opposed to the sequence-level. Specifically, we use the toxic layer identified by DINM as a form of sequence-level location and subsequently derive token-level safety layers for the entire sequence based on this coarse-grained location, as shown in Fig. 4.

For the k-th early exit layer, we first apply $\phi ( \cdot )$ and then use softmax to calculate the probability of predicting the next token $x _ { t }$ with the k-th layer as the output layer.

$$
q _ { k } ( x _ { t } \mid x _ { < t } ) = \mathrm { s o f t m a x } \big ( \phi ( h _ { t } ^ { ( k ) } ) \big ) _ { x _ { t } } , \quad k \in \mathcal { K }\tag{5}
$$

where $k \in \mathcal { K }$ and $\mathcal { K } = \{ 1 , \ldots , E - 1 \}$ , as detailed in TABLE 1. To allow for the selection of a safety layer at each time step, we employ the following method to measure the distance between the next-token distributions from two different layers, where JSD( , ) represents the Jensen-Shannon divergence.

$$
d ( q _ { T } ( x _ { t } \mid x _ { < t } ) ) , q _ { k } ( x _ { t } \mid x _ { < t } ) ) = \operatorname { J S D } ( q _ { T } ( x _ { t } \mid x _ { < t } ) \| q _ { k } ( x _ { t } \mid x _ { < t } ) ) ,\tag{6}
$$

q<sub>T</sub> denotes the logits of the toxic layer after softmax operation (details in Eq. 2). To amplify the safety of contrastive decoding (Li et al., 2023), the ideal optimal safety layer should be the one that exhibits the greatest difference from the toxic layer. We then select S as the safety layer, where $0 < S < E$ (layer E is deeper than S).

$$
\begin{array} { r } { S = \arg \operatorname* { m a x } _ { k \in \mathcal { K } } \mathrm { J S D } ( q _ { T } ( x _ { t } \mid x _ { < t } ) \parallel q _ { k } ( x _ { t } \mid x _ { < t } ) ) } \end{array}\tag{7}
$$

By obtaining precise token-level safety layer locations and incorporating the hallucination layer, which inherently exists in LLMs, we locate dynamic toxic regions that change with the variation of tokens. As the output layer of the LLM, the E layer is generally believed to contain the most factual knowledge; therefore, we designate the E layer as the factual region. Similarly, for the hallucination layer, we select the layer that exhibits the greatest difference in next-token distributions from the output layer, denoting it as the ideal hallucination layer.

$$
H = \arg \operatorname* { m a x } _ { j \in { \mathcal { T } } } \mathrm { J S D } ( q _ { E } ( x _ { t } \mid x _ { < t } ) \parallel q _ { j } ( x _ { t } \mid x _ { < t } ) )\tag{8}
$$

where $j \in \mathcal { I }$ and $\mathcal { I } = \{ 0 , \ldots , E - 1 \}$ $H \in$ $\{ 0 , \ldots , E - 1 \}$ is selected as the hallucination layer.

## 3.3 MODE-1: Dynamic Toxic Layer

By comparing the differences between various layers, we identify the S, H, and T within the LLM. Subsequently, DSCD utilizes the distributions of these three layers to perform self-constrained detoxification.

The specific operation of DSCD involves subtracting the next-token distribution of token-level safety layer from the next-token distribution of the coarse-grained toxic layer, followed by adding the next-token distribution of the hallucination layer, as shown in Fig. 2. This forms the next-token distribution of the toxic regions. We believe that the resulting distribution effectively predicts as many toxic tokens as possible. The next-token distribution for the toxic regions is expressed as follows:

$$
q _ { B } ( x _ { t } ) = q _ { H } ( x _ { t } ) - q _ { S } ( x _ { t } ) + q _ { T } ( x _ { t } )\tag{9}
$$

We utilize the operator $\mathcal { F }$ (Li et al., 2023) to calculate the log-domain difference between the distributions of the factual regions and the toxic regions. Specifically, we subtract the log probabilities of the toxic regions from those of the factual regions, thereby guiding the LLM to favor outputting information from the factual regions while avoiding the toxic regions during token prediction. This approach effectively reduces the generation of toxic tokens, achieving detoxification during the text generation stage. Since the log-domain computed for each token varies, resulting in different constraints being applied to the generated tokens, this approach is referred to as DSCD.

$$
\mathcal { F } \big ( q _ { E } ( x _ { t } ) , q _ { B } ( x _ { t } ) \big ) = \left\{ \begin{array} { l l } { \log \frac { q _ { E } ( x _ { t } ) } { q _ { B } ( x _ { t } ) } , } & { \mathrm { i f } \quad x _ { t } \in \mathcal { V } _ { \mathrm { h e a d } } ( x _ { t } | x _ { < t } ) , } \\ { - \infty , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{10}
$$

The resulting distribution is then used for the next-word prediction. To simplify the notation, we use $q _ { k } ( x _ { t } )$ to represent the term $q _ { k } ( x _ { t } \mid x _ { < t } )$ . The final probability $\hat { p }$ of the next token is calculated as follows:

$$
\hat { p } ( x _ { t } \mid x _ { < t } ) = \mathrm { s o f t m a x } \big ( \mathcal { F } \big ( q _ { E } ( x _ { t } ) , q _ { B } ( x _ { t } ) \big ) \big ) _ { x _ { t } }\tag{11}
$$

At the same time, we must ensure that the token predicted by $\mathcal { V } _ { \mathrm { h e a d } } ( x _ { t } | \boldsymbol { x } _ { < t } ) \in \mathcal { X }$ truly possesses sufficiently high confidence within the factual regions.

$$
\begin{array} { r } { \mathcal { V } _ { \mathrm { h e a d } } ( x _ { t } | \boldsymbol { x } _ { < t } ) = \{ \boldsymbol { x } _ { t } \in \mathcal { X } : q _ { E } ( \boldsymbol { x } _ { t } ) \geq \alpha \operatorname* { m a x } _ { w } q _ { E } ( w ) \} } \end{array}\tag{12}
$$

In token prediction, misjudgments in baseline methods may arise due to issues with token confidence. To address this, we introduce the adaptive plausibility constraint (APC) (Li et al., 2023) to ensure the plausibility of tokens predicted by the LLM.

## 3.4 MODE-2: Static Toxic Layer

To implement MODE-2, we first analyze the results of MODE-1 to locate the most frequently occurring toxic layer for each specific LLM. The layer with the highest occurrence is recorded (See in Fig. 4) and designated as the static toxic layer for that LLM. When applying DSCD in MODE-2, we skip the process of locating the toxic layer dynamically and directly use the pre-recorded static toxic layer for each LLM.

Besides, the location of the safety layer and hallucination layer remains dynamic. To reduce computational overhead, the candidate layers for safe and hallucination layers are restricted to those frequently observed in MODE-1, rather than searching across 0, 1, 2, . . . , 32 layers. Although this approach may result in less precise location of the toxic regions, it significantly reduces the computational cost and time required for toxic regions location. Most importantly, by fixing the toxic layer, the need to generate both $O _ { \mathrm { s a f e } }$ and $O _ { \mathrm { u n s a f e } }$ is eliminated. Instead, toxic inputs can be directly fed into the LLM, which then produces detoxified outputs, streamlining the detoxification process.

## 4 Experiment

## 4.1 Datasets

We choose SafeEdit (Wang et al., 2024c), AlpacaEval (Dubois et al., 2024), HarmfulQA/DangerousQA (Bhardwaj and Poria, 2023),

Advbench (Zou et al., 2023), and TruthfulQA (Lin et al., 2022) as the datasets.

## 4.2 Baseline Methods

We compare four methods on Llama2-7b-chat, Mistral-7b-v0.1, Qwen2-7b-instruct, and Llama2- 7b-uncensored-chat to evaluate the effectiveness of DSCD. These methods include DINM (Wang et al., 2024c), a knowledge edit based detoxification method and SafeDecoding (Xu et al., 2024), a safety-aware decoding strategy. Additionally, we evaluate two hybrid approaches that integrate DSCD with these methods: DINM+DSCD and SafeDecoding+DSCD.

## 4.3 Evaluation Metrics

Classification Task. We evaluate classification and generation tasks separately. We use supervised labels in SafeEdit to evaluate the classification task (See details in A.2 ).

The metric is DS (Defense Success Rate):

$$
\mathrm { D S } = { \frac { \mathrm { S a f e } } { \mathrm { S a f e } + \mathrm { U n s a f e } } }\tag{13}
$$

Generation Task. For generation tasks, the evaluation metrics include DS, $D G _ { \mathrm { o n l y Q } }$ $D G _ { \mathrm { o t h e r A : } }$ $D G _ { \mathrm { o t h e r Q } }$ $D G _ { \mathrm { o t h e r A Q } }$ , and $D G _ { \mathrm { A v g } }$ (Wang et al., 2024c), which assess detoxification performance across various adversarial inputs. Fluency is measured using n-grams (Wang et al., 2024b) to evaluate the helpfulness of generation.

<table><tr><td>Jailbreak Datasets</td><td>Defense</td><td>ASR↓</td><td>Harmful V Score</td><td>Fluency ↑</td></tr><tr><td rowspan="3">PAIR</td><td>Vanilla</td><td>0.18</td><td>1.44</td><td>7.65</td></tr><tr><td>DSCDMODE-2</td><td>0.10</td><td>1.30</td><td>7.64</td></tr><tr><td>SafeDecoding</td><td>0.04</td><td>1.20</td><td>7.51</td></tr><tr><td rowspan="3">AutoDAN</td><td>Vanilla</td><td>0.02</td><td>1.08</td><td>7.29</td></tr><tr><td>DSCDMODE-2</td><td>0.00</td><td>1.00</td><td>7.31</td></tr><tr><td>SafeDecoding</td><td>0.00</td><td>1.00</td><td>7.28</td></tr><tr><td rowspan="3">Advbench</td><td>Vanilla</td><td>0.00</td><td>1.00</td><td>7.29</td></tr><tr><td>DSCDMODE-2</td><td>0.00</td><td>1.00</td><td>7.32</td></tr><tr><td>SafeDecoding</td><td>0.00</td><td>1.00</td><td>7.28</td></tr><tr><td rowspan="3">AlpacaEval</td><td>Vanilla</td><td>93.88</td><td>1.06</td><td>7.60</td></tr><tr><td>DSCDMODE-2</td><td>91.84</td><td>1.06</td><td>7.64</td></tr><tr><td>SafeDecoding</td><td>77.55</td><td>1.16</td><td>7.60</td></tr><tr><td rowspan="5">SafeEdit</td><td>Vanilla</td><td>0.00</td><td>1.24</td><td>7.22</td></tr><tr><td>DSCDMODE-2</td><td>0.00</td><td>1.26</td><td>7.40</td></tr><tr><td>SafeDecoding</td><td>0.00</td><td>1.28</td><td>7.30</td></tr><tr><td>SafeDecoding</td><td></td><td></td><td></td></tr><tr><td>+DSCDMODE-2</td><td>0.00</td><td>1.16</td><td>7.22</td></tr></table>

Table 4: Comparison of DSCD and SafeDecoding on Llama2-7b-chat. DSCD demonstrates higher fluency while maintaining a similar level of detoxification as SafeDecoding.

The metric Time reflects the relative efficiency of LLMs in generating responses. Additionally,

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="7">Detoxification performance (Roberta↑)</td></tr><tr><td>DS</td><td> $\overline { { D G _ { o n l y Q } } }$ </td><td> $\overline { { D G _ { o t h e r A } } }$ </td><td> $\overline { { D G _ { o t h e r Q } } }$ </td><td> $\overline { { D G _ { o t h e r A Q } } }$ </td><td> $\overline { { D G - A v g } }$ </td><td>Fluency</td></tr><tr><td rowspan="9">Llama2 -7b-chat -uncensored</td><td>Vanilla</td><td>30.74</td><td>48.15</td><td>33.70</td><td>34.81</td><td>32.59</td><td>36.00</td><td>6.85</td></tr><tr><td>SFT</td><td>74.00</td><td>94.00</td><td>63.00</td><td>66.00</td><td>62.00</td><td>71.80</td><td>4.29</td></tr><tr><td>DPO</td><td>52.00</td><td>86.00</td><td>49.00</td><td>55.00</td><td>40.00</td><td>56.40</td><td>6.99</td></tr><tr><td> $\mathrm { D S C D } _ { M O D E - 1 }$ </td><td>60.00</td><td>65.71</td><td>45.71</td><td>37.14</td><td>45.71</td><td>50.86</td><td>6.37</td></tr><tr><td> $\mathrm { D S C D } _ { M O D E - 2 }$ </td><td>54.29</td><td>57.14</td><td>42.86</td><td>45.71</td><td>48.57</td><td>49.71</td><td>6.42</td></tr><tr><td> $\mathrm { S F T + D S C D } _ { M O D E - 1 }$ </td><td>77.00</td><td>94.00</td><td>67.00</td><td>81.00</td><td>56.00</td><td>75.00</td><td>5.04</td></tr><tr><td> $\mathrm { S F T + D S C D } _ { M O D E - 2 }$ </td><td>80.00</td><td>97.00</td><td>64.00</td><td>85.00</td><td>54.00</td><td>76.00</td><td>5.55</td></tr><tr><td> $\mathrm { D P O + D S C D } _ { M O D E - 1 }$ </td><td>56.00</td><td>92.00</td><td>53.00</td><td>52.00</td><td>53.00</td><td>61.20</td><td>6.90</td></tr><tr><td> $\mathrm { D P O + D S C D } _ { M O D E - 2 }$ </td><td>55.00</td><td>92.00</td><td>56.00</td><td>59.00</td><td>42.00</td><td>60.80</td><td>6.97</td></tr><tr><td rowspan="9">Qwen2 -7b-instruct</td><td>Vanilla</td><td>37.04</td><td>76.30</td><td>31.85</td><td>36.30</td><td>28.89</td><td>42.07</td><td>7.82</td></tr><tr><td>SFT</td><td>34.00</td><td>92.00</td><td>50.00</td><td>52.00</td><td>54.00</td><td>56.40</td><td>7.39</td></tr><tr><td>DPO</td><td>43.99</td><td>88.00</td><td>34.00</td><td>43.99</td><td>43.99</td><td>50.79</td><td>7.68</td></tr><tr><td>DSCDMODE-1</td><td>57.04</td><td>69.63</td><td>53.33</td><td>57.04</td><td>52.59</td><td>57.93</td><td>7.49</td></tr><tr><td>DSCDMODE-2</td><td>57.78</td><td>69.63</td><td>51.11</td><td>57.78</td><td>52.59</td><td>56.30</td><td>7.00</td></tr><tr><td> $\mathrm { S F T + D S C D } _ { M O D E - 1 }$ </td><td>64.00</td><td>96.00</td><td>64.00</td><td>82.00</td><td>58.00</td><td>72.80</td><td>7.00</td></tr><tr><td> $\mathrm { S F T + D S C D } _ { M O D E - 2 }$ </td><td>78.00</td><td>94.00</td><td>64.00</td><td>76.00</td><td>58.00</td><td>74.00</td><td>7.01</td></tr><tr><td> $\mathrm { D P O + D S C D } _ { M O D E - 1 }$ </td><td>52.00</td><td>78.00</td><td>43.99</td><td>52.00</td><td>43.99</td><td>53.99</td><td>7.45</td></tr><tr><td> $\mathrm { D P O + D S C D } _ { M O D E - 2 }$ </td><td>54.00</td><td>86.00</td><td>48.00</td><td>62.00</td><td>42.00</td><td>58.40</td><td>7.21</td></tr></table>

Table 2: Detoxification performance of Vanilla LLMs and several traditional detoxification methods on the SafeEdi dataset. The best results in each column are highlighted in Bold, while the second-best results are underlined.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="7">Detoxification performance (Roberta↑)</td></tr><tr><td>DS</td><td> $\overline { { D G _ { o n l y Q } } }$ </td><td> $\overline { { D G _ { o t h e r A } } }$ </td><td> $\overline { { D G _ { o t h e r Q } } }$ </td><td> $\overline { { D G _ { o t h e r A Q } } }$ </td><td> $\overline { { D G - A v g } }$ </td><td>Fluency</td></tr><tr><td rowspan="8">Llama2-7b-chat</td><td>Vanilla</td><td>51.90</td><td>90.48</td><td>45.24</td><td>53.33</td><td>46.67</td><td>57.52</td><td>7.33</td></tr><tr><td>SafeDecoding</td><td>40.00</td><td>98.00</td><td>26.00</td><td>44.00</td><td>90.00</td><td>59.60</td><td>6.68</td></tr><tr><td> $\mathrm { S a f e D e c o d i n g } + \mathrm { D S C D } _ { M O D E - 2 }$ </td><td>44.00</td><td>98.00</td><td>26.00</td><td>46.00</td><td>96.00</td><td>62.00</td><td>6.79</td></tr><tr><td> $\mathrm { D S C D } _ { M O D E - 1 }$ </td><td>59.26</td><td>88.15</td><td>68.15</td><td>54.07</td><td>60.00</td><td>65.93</td><td>6.87</td></tr><tr><td> $\mathrm { D S C D } _ { M O D E - 2 }$ </td><td>57.48</td><td>87.56</td><td>54.52</td><td>55.41</td><td>55.63</td><td>62.12</td><td>6.71</td></tr><tr><td>DINM</td><td>98.71</td><td>99.57</td><td>90.43</td><td>97.86</td><td>89.43</td><td>95.20</td><td>5.85</td></tr><tr><td> $\mathrm { D I N M + D S C D } _ { M O D E - 1 }$ </td><td>100.00</td><td>100.00</td><td>98.52</td><td>99.26</td><td>96.30</td><td>98.81</td><td>5.11</td></tr><tr><td> $\mathrm { D I N M + D S C D } _ { M O D E - 2 }$ </td><td>100.00</td><td>100.00</td><td>95.56</td><td>100.00</td><td>90.37</td><td>97.19</td><td>5.84</td></tr><tr><td rowspan="6">Mistral-7b-v0.1</td><td>Vanilla</td><td>49.26</td><td>46.67</td><td>43.70</td><td>40.74</td><td>35.93</td><td>43.26</td><td>7.22</td></tr><tr><td> $\mathrm { D S C D } _ { M O D E - 1 }$ </td><td>56.30</td><td>55.56</td><td>57.41</td><td>45.56</td><td>41.48</td><td>51.26</td><td>6.03</td></tr><tr><td> $\mathrm { D S C D } _ { M O D E - 2 }$ </td><td>46.30</td><td>56.30</td><td>44.07</td><td>44.44</td><td>49.26</td><td>48.07</td><td>6.17</td></tr><tr><td>DINM</td><td>89.07</td><td>91.93</td><td>53.30</td><td>88.89</td><td>51.00</td><td>74.84</td><td>4.57</td></tr><tr><td> $\mathrm { D I N M + D S C D } _ { M O D E - 1 }$ </td><td>88.37</td><td>91.70</td><td>63.28</td><td>87.96</td><td>61.04</td><td>78.47</td><td>4.51</td></tr><tr><td> $\mathrm { D I N M + D S C D } _ { M O D E - 2 }$ </td><td>86.67</td><td>91.11</td><td>68.52</td><td>81.48</td><td>65.56</td><td>78.67</td><td>4.58</td></tr></table>

Table 3: Detoxification performance of Vanilla LLMs and several SOTA detoxification methods on the SafeEdit dataset. The best results in each column are highlighted in Bold, while the second-best results are underlined.

ASR and Harmful Score (Xu et al., 2024) evaluate the attack success rate of harmful questions and the harmfulness of GPT-4o’s responses (rated on a scale of 1 to 5), separately. WinR1, WinR2, and TrueR (Zhao et al., 2024) assess models’ generative capabilities on general tasks, as detailed in Table 14. Notably, the baseline classifier for determining the safety of generated content is RoBERTa. To avoid errors from relying on a single classifier, we also use GPT-4o as an additional classifier. For detailed classifier information, please refer to B.2.

## 4.4 Experimental Settings

In this experiment, the specific experimental settings of DSCD are detailed in A.1.

## 4.5 Results

DSCD enables detoxification for both classification and generation tasks, incorporating MODE-1 and MODE-2 to accommodate different scenariospecific requirements. As shown in Fig. 3.

Classification Task. Llama2-7b-chat generates 1062 safe instances and 288 unsafe instances, resulting in DS of 78.67%. With DSCD intergrated, the same LLM generates 1077 safe instances and 273 unsafe instances, resulting in DS of 79.78%. DSCD brings 1.12% improvements.

Generation Task. As shown in Table 2, Table 3, and Table 4, DSCD performs excellently in detoxification, achieving best performance when integrated to DINM and SafeDecoding. When DSCD is used alone, it also achieves better performance than the vanilla model.

<table><tr><td rowspan=1 colspan=1>Model</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Time↓</td></tr><tr><td rowspan=5 colspan=1>Llama2-7b-uncensored-chat</td><td rowspan=5 colspan=1>VanillaSFTDPO $\mathrm { D S C D } _ { M O D E - 2 }$  $\mathrm { S F T + D S C D } _ { M O D E - 2 }$  $\mathrm { D P O + D S C D } _ { M O D E - 2 }$ </td><td rowspan=1 colspan=1>65.98</td></tr><tr><td rowspan=1 colspan=1>33.05</td></tr><tr><td rowspan=1 colspan=1>66.82</td></tr><tr><td rowspan=1 colspan=1>56.54</td></tr><tr><td rowspan=1 colspan=1>29.3170.89</td></tr><tr><td rowspan=1 colspan=1>Qwen2-7b-instruct</td><td rowspan=1 colspan=1>VanillaSFTDPO $\mathrm { D S C D } _ { M O D E - 2 }$  $\mathrm { S F T + D S C D } _ { M O D E - 2 }$  $\mathrm { D P O + D S C D } _ { M O D E - 2 }$ </td><td rowspan=1 colspan=1>74.5275.6774.9486.51104.25105.62</td></tr></table>

Table 5: Comparison of detoxification performance across models using traditional and DSCD methods on the SafeEdit dataset. Time is measured in seconds. The best results in each column are highlighted in Bold, while the second-best results are underlined.
<table><tr><td>Model</td><td>Method</td><td>Time↓</td></tr><tr><td>Mistral-v0.1</td><td>Vanilla  $\mathrm { D S C D } _ { M O D E - 2 }$  DINM  $\mathrm { D I N M + D S C D } _ { M O D E - 2 }$ </td><td>76.87 80.47 88.82 90.85</td></tr><tr><td>Llama2-7b- uncensored-chat</td><td>Vanilla  $\mathrm { D S C D } _ { M O D E - 2 }$  DINM  $\mathrm { D I N M + D S C D } _ { M O D E - 2 }$ </td><td>65.86 69.54 78.41 81.07</td></tr></table>

Table 6: Detoxification performance of language models using DINM and DSCD methods on the SafeEdit dataset. Time is measured in seconds. The best results in each column are highlighted in Bold, while the second-best results are underlined.

We first compare our method with traditional safety alignment techniques. In Table 2 and Table 9, Llama2-7b-chat-uncensored and Qwen2-7binstruct represent non-aligned and aligned models, respectively. Evaluations by RoBERTa and GPT-4o indicate that DSCD can be effectively applied on top of existing alignment approaches to further improve safety performance. Furthermore, the consistent gains observed when combining DSCD with both SFT and DPO highlight the general applicability of our method.

As shown in Table 3, applying DSCDMODE-1 alone improves the detoxification performance of the vanilla LLM by an average of 11.78%. When integrated into DINM, it yields an additional 4.03% improvement. Similarly, DSCDMODE-2 alone enhances performance by 9.34%, and by 3.70% when combined with DINM. Although MODE-2 performs slightly worse than MODE-1 in terms of detoxification effectiveness, it offers higher efficiency, maintaining fluency metrics comparable to the vanilla model while outperforming DINM, as detailed in Table 5. Moreover, Table 5 also shows that integrating DSCD into traditional detoxification methods does not introduce significant additional latency. In fact, when combined with SFTbased approaches, it can even reduce the overall inference time (a detailed explanation in Table 5). In summary, DSCD enables fast detoxification by trading off a small portion of detoxification performance for significantly improved efficiency.

To further validate these findings, we evaluate the use of GPT-4o as the classifier, as shown in Table 8 and Table 9, confirming that DSCD consistently provides superior detoxification performance. Notably, the plug-and-play nature of DSCD enables it to adapt to scenarios demanding both high performance and efficiency. For example, integrating MODE-2 into SafeDecoding reduces the Harmful Score on the SafeEdit dataset from 1.26 to 1.16, achieving state-of-the-art performance (as shown in Table 4).

Importantly, DSCD ensures that detoxification does not compromise the general performance of the model. Evaluations on general-purpose datasets, such as AlpacaEval (Dubois et al., 2024) and TruthfulQA (Lin et al., 2022), detailed in Section A.3, show that DSCD leads to an average performance improvement of 2.03% on these harmless datasets, as shown in Table 11, indicating no negative impact on general performance.

Further experiments on more harmful datasets, including HarmfulQA/DangerousQA (Bhardwaj and Poria, 2023) and Advbench (Zou et al., 2023), validate DSCD’s performance. Using RoBERTa as the classifier, the DS score improves by 4.85%, and with GPT-4o, the performance increases by 1.82% (as shown in Table 12 and Table 10). More details can be found in Section B.3.

Finally, Fig. 5 illustrates that DSCD reduces the average probability of generating toxic tokens by 48.7%, significantly lowering the occurrence of toxic tokens, while $\mathrm { D S C D } _ { S - H - T }$ increases the probability by 11.2%. This comparison demonstrates DSCD’s capability in identifying and detoxifying toxic regions in LLMs. Fig. 3 further shows that DSCD improves overall performance across

all models.

## 4.6 Analysis

Fig. 4 illustrates that the toxic layer remains constant within a single sequence, while the safe and hallucination layers identified by DSCD vary across tokens. This dynamic shift in toxic regions highlights the flexibility of DSCD’s detoxification approach. Table 13 demonstrates that DSCD effectively prevents the generation of toxic tokens through precise location, overcoming the limitations of DINM.

Fluency. We observe that DSCD offers more fluency generation without any additional expert model or supervised data for detoxification. As shown in Table 4, DSCD enhances fluency while maintaining detoxification performance comparable to SafeDecoding. This is because the internal constraints generated from the middle layer of the model and the original tokens are sampled from the same distribution, which better ensures fluency.

Efficiency. The efficiency gains of DSCD over DINM and SafeDecoding can be derived both theoretically and empirically. First, when DSCD switches to MODE-2, the need to locate toxic layers for each individual adversarial input is diminished, and the selection of toxic layer is based directly on experience, bypassing the location process entirely. Second, DSCD does not require parameter updates and extra expert model, it only constrains the output content in decoding phase. This significantly reduces computational overhead compared to DINM, which involves back propagation and parameter updates. Experimental results further corroborate this.

As shown in Table 5 and Table 6, the runtime of MODE-2 is close to that of Vanilla LLM and is shorter than that of DINM. Even when DSCD is incorporated into DINM, the runtime remains comparable to DINM, demonstrating significantly lower time overhead compared to DINM. As shown in Table 4, MODE-2 achieves high fluency in detoxification. Even when DSCD is incorporated into SafeDecoding, its fluency remains comparable to that of Vanilla LLM. Moreover, for 7B parameter models, the time cost is only about 2.17% higher than that of the Vanilla LLM, indicating good practical efficiency. In scenarios where efficiency is critical, DSCD can further eliminate the layer selection process to further reduce time overhead.

## 4.7 Ablation Study

The toxic, safe, and hallucination layers have different impacts on the detoxification performance of DSCD, and details can be found in B.1.

## 4.8 Case Study

We present two specific cases to demonstrate the effectiveness of DSCD. More details can be found in the B.5.

## 5 Related Work

Traditional model detoxification approaches can be broadly categorized into prompt engineering, safety alignment, and toxicity detection. Prompt engineering (Wang et al., 2024d; Zeng et al., 2024) improves model safety through prompt design, though its effectiveness relies heavily on the LLM’s inherent ability to refuse toxic queries. Safety alignment (Farquhar et al., 2024; Ji et al., 2023; Lee et al., 2024; Wang et al., 2024a) aims to match outputs with human values and safety standards, but typically bypasses rather than removes toxic regions, leaving models susceptible to sophisticated attacks. Toxicity detection (Farquhar et al., 2024; Zhang and Wan, 2023) focuses on identifying or evaluating toxic and hallucinatory content, but may be limited for context-dependent cases.

Currently, knowledge editing and decodingbased approaches are widely used in detoxification. Knowledge editing modifies harmful behavior either by updating model parameters (Meng et al., 2022, 2023; Mitchell et al., 2022a) or through nonparameter modifications (Hartvigsen et al., 2023; Huang et al., 2023; Mitchell et al., 2022b; Wei et al., 2023; Zheng et al., 2023), often utilizing editing descriptors (Yao et al., 2023). Decodingbased methods enhance safety during text generation, and include detection-based defenses, which perturb input or cross-check outputs (Phute et al., 2024; Robey et al., 2023), as well as mitigationbased strategies that adjust decoding probabilities or content prioritization (Xu et al., 2024; Zhang et al., 2024), both effectively reducing jailbreak success rates. Our DSCD method belongs to the latter category.

## 6 Conclusion

In this work, we propose DSCD, a self-constrained decoding approach for detoxifying large language models (LLMs). By using token-level toxic layer localization as a constraint, DSCD enhances the detoxification effectiveness of existing methods and can be seamlessly integrated into current detoxification strategies to achieve state-of-the-art safety rates. Importantly, DSCD maintains the best fluency scores while outperforming baseline methods by nearly 12% on average. Moreover, its two distinct operational modes offer a flexible trade-off between detoxification performance and efficiency, making DSCD well suited for real-world LLM applications.

## Limitation

Although DSCD demonstrates excellent detoxification performance, the decoding method still has some limitations: 1) While the results show the effectiveness of DSCD both when used alone and in combination with DINM and SafeDecoding, due to time and resource constraints, we have not performed generalization testing of DSCD on more detoxification methods. 2) Since the focus of this study is on detoxification through decoding methods for large models, we have primarily focused on DSCD’s detoxification performance across different large model architectures. Experiments were conducted on three different architectures, where the Llama series used Llama2-7b-chat rather than the newer Llama3 series with the same architecture.

In the future, we will incorporate more detoxification methods and apply DSCD to emerging large language models to further explore its performance.

## Acknowledgments

This work was partly supported by China Postdoctoral Science Foundation (No. 2023M731253), Hubei Provincial Natural Science Foundation (No. 2023AFB487), General Project of the 14th Five-Year Plan (2024) of the National Language Commission (No. YB145-128) , and the National Natural Science Foundation of China (No. 62476108).

## References

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom B. Brown, Jack Clark, Sam McCandlish, Chris Olah, Benjamin Mann, and Jared Kaplan. 2022. Training a helpful and harmless assistant with rein-

forcement learning from human feedback. CoRR, abs/2204.05862.

Rishabh Bhardwaj and Soujanya Poria. 2023. Redteaming large language models using chain of utterances for safety-alignment. CoRR, abs/2308.09662.

Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James R. Glass, and Pengcheng He. 2024. Dola: Decoding by contrasting layers improves factuality in large language models. In ICLR. OpenReview.net.

Yann Dubois, Balázs Galambosi, Percy Liang, and Tatsunori B. Hashimoto. 2024. Length-controlled alpacaeval: A simple way to debias automatic evaluators. CoRR, abs/2404.04475.

Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. 2024. Detecting hallucinations in large language models using semantic entropy. Nat., 630(8017):625–630.

Tom Hartvigsen, Swami Sankaranarayanan, Hamid Palangi, Yoon Kim, and Marzyeh Ghassemi. 2023. Aging with GRACE: lifelong model editing with discrete key-value adaptors. In NeurIPS.

Zeyu Huang, Yikang Shen, Xiaofeng Zhang, Jie Zhou, Wenge Rong, and Zhang Xiong. 2023. Transformerpatcher: One mistake worth one neuron. In ICLR. OpenReview.net.

Jiaming Ji, Mickel Liu, Josef Dai, Xuehai Pan, Chi Zhang, Ce Bian, Boyuan Chen, Ruiyang Sun, Yizhou Wang, and Yaodong Yang. 2023. Beavertails: Towards improved safety alignment of LLM via a human-preference dataset. In NeurIPS.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. CoRR, abs/2310.06825.

Andrew Lee, Xiaoyan Bai, Itamar Pres, Martin Wattenberg, Jonathan K. Kummerfeld, and Rada Mihalcea. 2024. A mechanistic understanding of alignment algorithms: A case study on DPO and toxicity. In ICML. OpenReview.net.

Xiang Lisa Li, Ari Holtzman, Daniel Fried, Percy Liang, Jason Eisner, Tatsunori Hashimoto, Luke Zettlemoyer, and Mike Lewis. 2023. Contrastive decoding: Open-ended text generation as optimization. In ACL (1), pages 12286–12312. Association for Computational Linguistics.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. Truthfulqa: Measuring how models mimic human falsehoods. In ACL (1), pages 3214–3252. Association for Computational Linguistics.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2022. Locating and editing factual associations in GPT. In NeurIPS.

Kevin Meng, Arnab Sen Sharma, Alex J. Andonian, Yonatan Belinkov, and David Bau. 2023. Massediting memory in a transformer. In ICLR. Open-Review.net.

Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, and Christopher D. Manning. 2022a. Fast model editing at scale. In ICLR. OpenReview.net.

Eric Mitchell, Charles Lin, Antoine Bosselut, Christopher D. Manning, and Chelsea Finn. 2022b. Memorybased model editing at scale. In ICML, volume 162 of Proceedings of Machine Learning Research, pages 15817–15831. PMLR.

OpenAI. 2023. GPT-4 technical report. CoRR, abs/2303.08774.

Ethan Perez, Saffron Huang, H. Francis Song, Trevor Cai, Roman Ring, John Aslanides, Amelia Glaese, Nat McAleese, and Geoffrey Irving. 2022. Red teaming language models with language models. In EMNLP, pages 3419–3448. Association for Computational Linguistics.

Mansi Phute, Alec Helbling, Matthew Hull, Shengyun Peng, Sebastian Szyller, Cory Cornelius, and Duen Horng Chau. 2024. LLM self defense: By self examination, llms know they are being tricked. In Tiny Papers @ ICLR. OpenReview.net.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. In NeurIPS.

Alexander Robey, Eric Wong, Hamed Hassani, and George J. Pappas. 2023. Smoothllm: Defending large language models against jailbreaking attacks. CoRR, abs/2310.03684.

Surat Teerapittayanon, Bradley McDanel, and H. T. Kung. 2016. Branchynet: Fast inference via early exiting from deep neural networks. In ICPR, pages 2464–2469. IEEE.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. Llama: Open and efficient foundation language models. CoRR, abs/2302.13971.

Binghai Wang, Rui Zheng, Lu Chen, Yan Liu, Shihan Dou, Caishuang Huang, Wei Shen, Senjie Jin, Enyu Zhou, Chenyu Shi, Songyang Gao, Nuo Xu, Yuhao Zhou, Xiaoran Fan, Zhiheng Xi, Jun Zhao, Xiao Wang, Tao Ji, Hang Yan, Lixing Shen, Zhan Chen, Tao Gui, Qi Zhang, Xipeng Qiu, Xuanjing Huang, Zuxuan Wu, and Yu-Gang Jiang. 2024a. Secrets of RLHF in large language models part II: reward modeling. CoRR, abs/2401.06080.

Mengru Wang, Yunzhi Yao, Ziwen Xu, Shuofei Qiao, Shumin Deng, Peng Wang, Xiang Chen, Jia-Chen Gu, Yong Jiang, Pengjun Xie, Fei Huang, Huajun Chen, and Ningyu Zhang. 2024b. Knowledge mechanisms in large language models: A survey and perspective. In EMNLP (Findings), pages 7097–7135. Association for Computational Linguistics.

Mengru Wang, Ningyu Zhang, Ziwen Xu, Zekun Xi, Shumin Deng, Yunzhi Yao, Qishen Zhang, Linyi Yang, Jindong Wang, and Huajun Chen. 2024c. Detoxifying large language models via knowledge editing. In ACL (1), pages 3093–3118. Association for Computational Linguistics.

Yihan Wang, Zhouxing Shi, Andrew Bai, and Cho-Jui Hsieh. 2024d. Defending llms against jailbreaking attacks via backtranslation. In ACL (Findings), pages 16031–16046. Association for Computational Linguistics.

Zeming Wei, Yifei Wang, and Yisen Wang. 2023. Jailbreak and guard aligned language models with only few in-context demonstrations. CoRR, abs/2310.06387.

Zhangchen Xu, Fengqing Jiang, Luyao Niu, Jinyuan Jia, Bill Yuchen Lin, and Radha Poovendran. 2024. Safedecoding: Defending against jailbreak attacks via safety-aware decoding. In ACL (1), pages 5587– 5605. Association for Computational Linguistics.

Yunzhi Yao, Peng Wang, Bozhong Tian, Siyuan Cheng, Zhoubo Li, Shumin Deng, Huajun Chen, and Ningyu Zhang. 2023. Editing large language models: Problems, methods, and opportunities. In EMNLP, pages 10222–10240. Association for Computational Linguistics.

Yi Zeng, Hongpeng Lin, Jingwen Zhang, Diyi Yang, Ruoxi Jia, and Weiyan Shi. 2024. How johnny can persuade llms to jailbreak them: Rethinking persuasion to challenge AI safety by humanizing llms. In ACL (1), pages 14322–14350. Association for Computational Linguistics.

Xu Zhang and Xiaojun Wan. 2023. Mil-decoding: Detoxifying language models at token-level via multiple instance learning. In ACL (1), pages 190–202. Association for Computational Linguistics.

Zhexin Zhang, Junxiao Yang, Pei Ke, Fei Mi, Hongning Wang, and Minlie Huang. 2024. Defending large language models against jailbreaking attacks through goal prioritization. In ACL (1), pages 8865–8887. Association for Computational Linguistics.

Zhengyue Zhao, Xiaoyun Zhang, Kaidi Xu, Xing Hu, Rui Zhang, Zidong Du, Qi Guo, and Yunji Chen. 2024. Adversarial contrastive decoding: Boosting safety alignment of large language models via opposite prompt optimization. arXiv preprint arXiv:2406.16743.

Ce Zheng, Lei Li, Qingxiu Dong, Yuxuan Fan, Zhiyong Wu, Jingjing Xu, and Baobao Chang. 2023. Can we

edit factual knowledge by in-context learning? In EMNLP, pages 4862–4876. Association for Computational Linguistics.

Andy Zou, Zifan Wang, J. Zico Kolter, and Matt Fredrikson. 2023. Universal and transferable adversarial attacks on aligned language models. CoRR, abs/2307.15043.

## A Detailed Experimental setups

## A.1 Settings

For the Classification Task. We conduct experiments on the RTX-4090 with 24GB of memory. The set of early exit layers is 0, 2, 4, 6, 8, 10, 12, 14, 16, 32 .

For the Generation Tasks. In the settings of MODE-1 and MODE-2, the only difference lies in the configuration of the early exit layers. All experiments are conducted on an RTX-4090 with 24GB of memory, with a maximum input length set to 2048 and a maximum output length set to 300. In MODE-1, for models with 32 layers, such as those from the Llama and Mistral series, the early exit layers are set to 0, 1, 2, . . . , 32 , while for models with 28 layers, such as those from the Qwen series, the early exit layers are set to 0, 1, 2, . . . , 28 . For all models, the final layer is designated as the output layer. Through experiments conducted under the MODE-1 configuration, we observe that the toxic layers generally reside in the deeper layers of the model. Specifically, the toxic layers of Llama2- 7b-chat are primarily in the 28th layer, those of Mistral-7b-v0.1 are concentrated in the 31st layer, and the toxic layers of Qwen2-7b-instruct are located in the 27th layer. In the first two models, the safety layers are typically found in the shallower layers, while the hallucination layers are mainly concentrated in the embedding layers.

However, due to the introduction of the dynamic gating mechanism, Qwen2-7b-instruct performe more dynamic adjustments in deeper layers, leading to greater distributional differences between these layers. As a result, the safety layers are no longer located in the shallow layers but appear in the deeper layers. Similarly, the hallucination layers are no longer confined to the embedding layers but are found in deeper layers. Our findings indicate that, across all three models examined, hallucination layers can coexist with safety and toxic layers within the same layer. This further suggests that the hallucination layers correspond to the layers with the greatest divergence from factual knowledge, containing a higher proportion of hallucinated information, as shown in Fig. 4.

Based on the conclusions from MODE-1, we proceed with the configuration for MODE-2: For Llama2-7b-chat, the 28th layer is designated as the fixed toxic layer; for Mistral-7b-v0.1, the 31st layer is designated as the fixed toxic layer; and for Qwen2-7b-instruct, the 27th layer is designated as the fixed toxic layer. At the same time, for Llama and Mistral series models, we set the early exit layers to 0, 2, 15, 28, 31, 32 , and for Qwen series models, we set the early exit layers to 0, 2, 15, 27, 28 . Additionally, we set the adaptive plausibility constraint (α) to 0.1.

## A.2 Details of the Classification Task

In the SafeEdit dataset, each question corresponds to both a safe generation and an unsafe generation, labeled as "safe" and "unsafe", respectively. We input both safe and unsafe generations into the large model (using Llama2-7b-chat as the Vanilla model in this classification task). For each input token, we compute the logits and sum them to obtain the log probability of the entire sentence. We then compare the log probabilities of the safe and unsafe generations. Since a higher log probability indicates greater model confidence in the output, if the log probability of the unsafe generation is higher, we classify the model’s output as unsafe; otherwise, it is classified as safe.

Based on the formula 13, after multiple experiments, we observe that the DS score increases from 78.67% to 79.78% after applying DSCD, demonstrating that DSCD helps the model produce safer outputs.

## A.3 Harmless Datasets

On the AlpacaEval dataset, we compare outputs generated with DSCD to those from OpenAI’s Text-Davinci-003 and GPT-4o, calculating the win rate using ChatGPT. For the TruthfulQA dataset, we used GPT-4o to assess whether the model’s outputs align with real-world knowledge, calculating the truthful rate (Zhao et al., 2024).

## B More Results

## B.1 Ablation Study on DSCD

From Table 7, we observe that when only the toxic layer (T) is used, the average detoxification success rate is 61.71%, which is an improvement of 4.19% over the Vanilla LLM. This indicates that the toxic layer indeed encapsulates harmful knowledge. Moreover, when we use only the hallucination layer (H) as the toxic region to explore whether hallucinated knowledge also contains toxicity, the results show an increase of 2.48% in the average detoxification success rate, suggesting that hallucinated knowledge also includes a small amount of toxic content. Therefore, we conclude that the hallucination layer should also be considered when defining the toxic region. By using the hallucination layer and the safety layer (H-S) as the toxic region, the success rate improves by 1.63% compared to using the hallucination layer alone, which indicates that subtracting the logits distribution of the safety layer from that of the hallucination layer effectively expands the detection range of toxicity in the toxic region. Additionally, the table shows that the average detoxification success rate using (H-S) to define the toxic region outperforms using (H+T), further demonstrating that token-level detoxification is indeed more effective than sequence-level detoxification. Finally, by incorporating the toxic layer, safety layer, and hallucination layer into the toxic region for computation, we design the DSCD, achieving SOTA performance. These ablation studies highlight the specific types of knowledge encapsulated by the toxic layer, safety layer, and hallucination layer, as well as the more effective detoxification outcomes when these layers are combined.

![](images/b83006790338defd4e4664cdacd69fb4ffbced9dc44e6f0a83121b77fd1c3c2b.jpg)

![](images/2f8e1cdb1e31927df762cad3ec246c09d8d39634aa863ba11fb0578241ad1476.jpg)  
Figure 3: Comparison of detoxification performance. A bar in the positive half of the y-axis indicates that the first entity outperforms the second in detoxification, while a bar in the negative half signifies inferior performance. The height of the bar represents the percentage [%] difference in the given metric.

![](images/c0337da900c40fc08d5c397cab63b4d0bb748ec899762ee0b3b52df32dd913f1.jpg)

![](images/9cd2b7a88c1d25e721d16a73208ac4cd96acae07c093cc52d8d432975a5a39e9.jpg)

![](images/a0244e15fb8a98417b8b95ed37d9419e9fc1a4b19bf22ef32be0f9559d0262d3.jpg)  
Figure 4: Toxic, Safe, and Hallucination Layer Distributions of a single input sequence on Models. We observe that toxic layers typically appear in deeper layers, which may accumulate more toxicity.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="7">Detoxification Performance (Roberta↑)</td></tr><tr><td>DS</td><td> $\overline { { D G _ { o n l y Q } } }$ </td><td> $\overline { { D G _ { o t h e r A } } }$ </td><td> $\overline { { D G _ { o t h e r Q } } }$ </td><td> $\overline { { D G _ { o t h e r A Q } } }$ </td><td> $\overline { { D G - A v g } }$ </td><td>Fluency</td></tr><tr><td rowspan="8">Llama2-7b-chat</td><td>Vanilla</td><td>51.90</td><td>90.48</td><td>45.24</td><td>53.33</td><td>46.67</td><td>57.52</td><td>7.33</td></tr><tr><td> $\mathrm { D S C D } _ { H }$ </td><td>68.37</td><td>79.59</td><td>55.10</td><td>47.96</td><td>48.98</td><td>60.00</td><td>6.62</td></tr><tr><td> $\mathrm { D S C D } _ { T }$ </td><td>52.86</td><td>97.14</td><td>48.57</td><td>54.29</td><td>55.71</td><td>61.71</td><td>6.22</td></tr><tr><td> $\mathrm { D S C D } _ { H + T }$ </td><td>54.08</td><td>87.76</td><td>59.18</td><td>50.00</td><td>53.06</td><td>60.82</td><td>6.33</td></tr><tr><td> $\mathrm { D S C D } _ { H - S }$ </td><td>59.18</td><td>83.67</td><td>63.27</td><td>42.86</td><td>59.18</td><td>61.63</td><td>6.95</td></tr><tr><td> $\mathrm { D S C D } _ { S - H - T }$ </td><td>60.95</td><td>90.48</td><td>43.33</td><td>56.19</td><td>31.43</td><td>56.48</td><td>5.88</td></tr><tr><td> $\mathrm { D S C D } _ { M O D E - 1 }$ </td><td>59.26</td><td>88.15</td><td>68.15</td><td>54.07</td><td>60.00</td><td>65.93</td><td>6.87</td></tr><tr><td> $\mathrm { D S C D } _ { M O D E - 2 }$ </td><td>57.48</td><td>87.56</td><td>54.52</td><td>55.41</td><td>55.63</td><td>62.12</td><td>6.71</td></tr></table>

Table 7: Ablation study on layer selection in DSCD on the SafeEdit dataset. S-H-T applies DSCD in reverse, increasing the model’s harmful output. H-S defines toxic regions using only the hallucination and safety layers, while H+T defines toxic regions using the hallucination and toxic layers. H and T represent toxic regions defined by the hallucination and toxic layers, respectively. The best results in each column are in bold, and the second-best are underlined.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="7">Detoxification Performance (GPT-4o↑)</td></tr><tr><td>DS</td><td> $\overline { { D G _ { o n l y Q } } }$ </td><td> $\overline { { D G _ { o t h e r A } } }$ </td><td> $\overline { { D G _ { o t h e r Q } } }$ </td><td> $\overline { { D G _ { o t h e r A Q } } }$ </td><td> $\overline { { D G - A v g } }$ </td><td>Fluency</td></tr><tr><td rowspan="7">Llama2-7b-chat</td><td>Vanilla</td><td>25.71</td><td>68.527</td><td>31.43</td><td>42.86</td><td>45.71</td><td>42.86</td><td>7.33</td></tr><tr><td>DINM</td><td>65.31</td><td>81.25</td><td>47.83</td><td>69.39</td><td>42.86</td><td>61.33</td><td>5.85</td></tr><tr><td>DSCDMODE-1</td><td>40.82</td><td>67.35</td><td>40.82</td><td>34.69</td><td>44.90</td><td>45.72</td><td>6.87</td></tr><tr><td>DSCDMODE-2</td><td>42.86</td><td>69.39</td><td>42.86</td><td>30.61</td><td>36.73</td><td>44.49</td><td>6.71</td></tr><tr><td>DINM+DSCDMODE−1</td><td>79.59</td><td>89.80</td><td>48.94</td><td>46.94</td><td>53.06</td><td>63.67</td><td>5.11</td></tr><tr><td> $\mathrm { D I N M + D S C D } _ { M O D E - 2 }$ </td><td>66.67</td><td>79.59</td><td>53.06</td><td>62.50</td><td>46.94</td><td>61.75</td><td>5.84</td></tr><tr><td>Vanilla</td><td>32.65</td><td>67.35</td><td>26.53</td><td>36.73</td><td>20.41</td><td></td><td></td></tr><tr><td rowspan="6">Qwen2-7b-instruct</td><td>DINM</td><td>81.63</td><td>77.55</td><td></td><td>83.67</td><td>59.18</td><td>36.73 74.28</td><td>7.82</td></tr><tr><td> $\mathrm { D S C D } _ { M O D E - 1 }$ </td><td>36.73</td><td>63.27</td><td>69.39</td><td>34.69</td><td>32.65</td><td></td><td>6.37</td></tr><tr><td> $\mathrm { D S C D } _ { M O D E - 2 }$ </td><td>28.57</td><td>67.35</td><td>40.82 32.65</td><td>44.90</td><td>36.73</td><td>41.63 42.04</td><td>7.49 7.00</td></tr><tr><td> $\mathrm { D I N M + D S C D } _ { M O D E - 1 }$ </td><td>85.71</td><td>88.57</td><td>77.14</td><td>71.14</td><td>74.29</td><td>79.37</td><td>6.14</td></tr><tr><td></td><td>82.86</td><td>80.00</td><td>77.14</td><td>71.43</td><td>77.14</td><td>77.71</td><td>6.83</td></tr><tr><td> $\mathrm { D I N M + D S C D } _ { M O D E - 2 }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 8: Detoxification performance of SOTA methods evaluated with GPT-4o as the classifier on the SafeEdit dataset. All other experimental parameters remain unchanged. Best results in each column are displayed in bold; the second-best are underlined.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="7">Detoxification Performance (GPT-4o↑)</td></tr><tr><td>DS</td><td> $\overline { { D G _ { o n l y Q } } }$ </td><td> $\overline { { D G _ { o t h e r A } } }$ </td><td> $\overline { { D G _ { o t h e r Q } } }$ </td><td> $\overline { { D G _ { o t h e r A Q } } }$ </td><td> $\overline { { D G - A v g } }$ </td><td>Fluency</td></tr><tr><td rowspan="9">Llama2-7b-chat-uncensored</td><td>Vanilla</td><td>25.71</td><td>68.53</td><td>31.43</td><td>42.86</td><td>45.71</td><td>42.86</td><td>7.33</td></tr><tr><td>SFT</td><td>80.00</td><td>96.00</td><td>64.00</td><td>70.00</td><td>64.00</td><td>74.80</td><td>4.29</td></tr><tr><td>DPO</td><td>54.00</td><td>90.00</td><td>60.00</td><td>50.00</td><td>46.00</td><td>60.00</td><td>6.99</td></tr><tr><td> $\mathrm { D S C D } _ { M O D E - 1 }$ </td><td>54.00</td><td>92.00</td><td>64.00</td><td>50.00</td><td>52.00</td><td>62.40</td><td>6.87</td></tr><tr><td> $\mathrm { D S C D } _ { M O D E - 2 }$ </td><td>40.00</td><td>92.00</td><td>60.00</td><td>56.00</td><td>52.00</td><td>60.00</td><td>6.71</td></tr><tr><td> $\mathrm { S F T + D S C D } _ { M O D E - 1 }$ </td><td>77.00</td><td>94.00</td><td>67.00</td><td>81.00</td><td>56.00</td><td>75.00</td><td>5.04</td></tr><tr><td> $\mathrm { S F T + D S C D } _ { M O D E - 2 }$ </td><td>80.00</td><td>97.00</td><td>64.00</td><td>85.00</td><td>54.00</td><td>76.00</td><td>5.55</td></tr><tr><td> $\mathrm { D P O + D S C D } _ { M O D E - 1 }$ </td><td>56.00</td><td>92.00</td><td>53.00</td><td>52.00</td><td>53.00</td><td>61.20</td><td>6.90</td></tr><tr><td> $\mathrm { D P O + D S C D } _ { M O D E - 2 }$ </td><td>55.00</td><td>92.00</td><td>56.00</td><td>59.00</td><td>42.00</td><td>60.80</td><td>6.97</td></tr><tr><td rowspan="9">Qwen2-7b-instruct</td><td>Vanilla</td><td>32.65</td><td>67.35</td><td>26.53</td><td>36.73</td><td>20.41</td><td>36.73</td><td>7.82</td></tr><tr><td>SFT</td><td>48.00</td><td>94.00</td><td>58.00</td><td>58.00</td><td>54.00</td><td>62.40</td><td>7.39</td></tr><tr><td>DPO</td><td>40.0</td><td>88.0</td><td>44.0</td><td>36.0</td><td>36.0</td><td>48.8</td><td>7.63</td></tr><tr><td> $\mathrm { D S C D } _ { M O D E - 1 }$ </td><td>36.73</td><td>63.27</td><td>40.82</td><td>34.69</td><td>32.65</td><td>41.63</td><td>7.49</td></tr><tr><td> $\mathrm { D S C D } _ { M O D E - 2 }$ </td><td>28.57</td><td>67.35</td><td>32.65</td><td>44.90</td><td>36.73</td><td>42.04</td><td>7.00</td></tr><tr><td> $\mathrm { S F T + D S C D } _ { M O D E - 1 }$ </td><td>58.00</td><td>96.00</td><td>70.00</td><td>74.00</td><td>56.00</td><td>70.80</td><td>7.00</td></tr><tr><td> $\mathrm { S F T + D S C D } _ { M O D E - 2 }$ </td><td>70.00</td><td>96.00</td><td>60.00</td><td>64.00</td><td>58.00</td><td>69.60</td><td>7.01</td></tr><tr><td> $\mathrm { D P O + D S C D } _ { M O D E - 1 }$ </td><td>40.00</td><td>94.00</td><td>54.00</td><td>42.00</td><td>48.00</td><td>55.60</td><td>7.45</td></tr><tr><td> $\mathrm { D P O + D S C D } _ { M O D E - 2 }$ </td><td>56.00</td><td>92.00</td><td>46.00</td><td>54.00</td><td>44.00</td><td>58.40</td><td>7.21</td></tr></table>

Table 9: Detoxification performance of traditional methods evaluated with GPT-4o as the classifier on the SafeEdit dataset, using the same experimental settings as in prior evaluations. The highest score in each column is shown in bold, and the second-highest is underlined

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method-</td><td>HarmfulQA</td><td>DangerousQA</td><td>Advbench</td></tr><tr><td></td><td>DS ↑</td><td></td></tr><tr><td rowspan="2">Llama2-7b- uncensored-chat</td><td>Vanilla</td><td>89.11%</td><td>86.14%</td><td>34.83%</td></tr><tr><td>DSCD</td><td>93.07%</td><td>82.18%</td><td>43.78%</td></tr><tr><td rowspan="2">Qwen2-7b-instruct</td><td>Vanilla</td><td>96.04%</td><td>67.33%</td><td>73.27%</td></tr><tr><td>DSCD</td><td>97.03%</td><td>73.27%</td><td>75.25%</td></tr><tr><td rowspan="2">mistral-v0.1</td><td>Vanilla</td><td>90.10%</td><td>65.35%</td><td>71.29%</td></tr><tr><td>DSCD</td><td>91.09%</td><td>69.31%</td><td>70.30%</td></tr><tr><td rowspan="2">Llama2-7b-chat</td><td>Vanilla</td><td>96.04%</td><td>39.60%</td><td>95.05%</td></tr><tr><td>DSCD</td><td>98.02%</td><td>34.65%</td><td>95.05%</td></tr><tr><td>Avg. ∆</td><td></td><td>+1.98 %</td><td>+0.99 %</td><td>+2.49 %</td></tr></table>

Table 10: Defense Success Rate (DS) between Vanilla and DSCD methods for multiple models on the HarmfulQA, the DangerousQA, and the Advbench datasets evaluated by GPT-4o. Avg. ∆ represents the average increase (+) or decrease (-) level of DS.

## B.2 Detoxification Performance on GPT-4o

The overall detoxification performance scores are lower when using GPT-4o as the classifier compared to RoBERTa, as shown in Table 8. This is because RoBERTa’s scoring results are inaccurate, as it can only determine whether certain tokens from the training corpus appear in the output, without truly understanding the meaning of the output. Therefore, we use GPT-4o to evaluate whether DSCD can truly detoxify large models, rather than merely filtering out toxic tokens while allowing harmful content to persist. The results show that both DSCD alone and in combination with DINM make the output safer.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="2">AlpacaEval</td><td rowspan="2">TruthfulQA TrueR ↑</td></tr><tr><td></td><td>WinR1↑WinR2↑</td></tr><tr><td rowspan="2">Llama2-7b- uncensored-chat</td><td>Vanilla</td><td>5.97%</td><td>0.96%</td><td>19.40%</td></tr><tr><td>DSCD</td><td>7.00%</td><td>0.96%</td><td>20.40%</td></tr><tr><td rowspan="2">Qwen2-7b-instruct</td><td>Vanilla</td><td>39.30%</td><td>1.49%</td><td>43.07%</td></tr><tr><td>DSCD</td><td>41.79%</td><td>2.99%</td><td>48.02%</td></tr><tr><td rowspan="2">mistral-v0.1</td><td>Vanilla</td><td>2.34%</td><td>0.78%</td><td>5.97%</td></tr><tr><td>DSCD</td><td>3.91%</td><td>1.56%</td><td>10.95%</td></tr><tr><td>Avg. ∆</td><td></td><td>+1.70%</td><td>+0.76%</td><td>+3.64%</td></tr></table>

Table 11: The generation ability comparison between the Vanilla and DSCD methods on the AlpacaEval and the TruthfulQA datasets. WinR1 represents win rate of target outputs compared with text-davinci-003 and WinR2 represents win rate compared with GPT-4o. TrueR is the truthful rate of models’ outputs evaluated by GPT-4o. Avg. ∆ represents the average increase (+) or decrease (-) level of each indicator.

This is particularly evident when using Vanilla models, which are more vulnerable to jailbreaking attacks, where DSCD’s detoxification effects are more prominent. Due to the large dataset and the high cost of GPT-4o, we conduct the GPT-4o evaluation only on Qwen2-7b-instruct and Llama2- 7b-chat, with the results showing similar trends to those obtained using the RoBERTa classifier.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td>HarmfulQA</td><td>DangerousQA</td><td>Advbench</td></tr><tr><td></td><td>DS ↑</td><td></td></tr><tr><td rowspan="2">Llama2-7b- uncensored-chat</td><td>Vanilla</td><td>70.74%</td><td>40.59%</td><td>51.44%</td></tr><tr><td>DSCD</td><td>86.3%</td><td>62.38%</td><td>61.92%</td></tr><tr><td rowspan="2">Qwen2-7b-instruct</td><td>Vanilla</td><td>66.67%</td><td>86.14%</td><td>96.00%</td></tr><tr><td>DSCD</td><td>59.26%</td><td>85.15%</td><td>96.00%</td></tr><tr><td rowspan="2">mistral-v0.1</td><td>Vanilla</td><td>72.59%</td><td>62.38%</td><td>64.68%</td></tr><tr><td>DSCD</td><td>85.93%</td><td>45.54%</td><td>51.24%</td></tr><tr><td rowspan="2">Llama2-7b-chat</td><td>Vanilla</td><td>69.63%</td><td>94.06%</td><td>97.31%</td></tr><tr><td>DSCD</td><td>70.37%</td><td>98.02%</td><td>97.88%</td></tr><tr><td>Avg. ∆</td><td></td><td>+5.56 %</td><td>+1.98 %</td><td>+7.00 %</td></tr></table>

Table 12: The generation ability comparison between the Vanilla and DSCD methods on the HarmfulQA, the DangerousQA and the Advbench datasets. DS represents the detoxification success rate for adversarial inputs evaluated by Roberta. Avg. ∆ represents the average increase (+) or decrease (-) level of DS.

![](images/58f506e440fff2bf5324e4ede9e98871e0dab9ac9d171ff75d1c649df489ff97.jpg)  
Figure 5: PROMPT: “So I’m starting to think she’s full of ‘s", exploring the probability of several models generating the output ‘shit’ when using a malicious guide PROMPT.

## B.3 Results on other Harmful Datasets

After applying DSCD on the HarmfulQA, DangerousQA, and Advbench datasets, we observe an average improvement in the DS score across all models, regardless of whether GPT-4o or RoBERTa is used as the classifier. While the improvement on the DangerousQA dataset is relatively modest, this indicates that DSCD may not always achieve the best detoxification performance, but it consistently demonstrates generalizable detoxification effects on harmful datasets.

## B.4 Results on other Harmless Datasets

On the AlpacaEval and TruthfulQA datasets, applying DSCD improves all metrics except for the WinR2 score, which remains unchanged when using Llama2-7b-uncensored-chat as the Vanilla model. This emphasizes that DSCD does not compromise the model’s performance on general tasks; on the contrary, it makes the generated content more realistic and specific. (A higher TrueR indicates more realistic content, while higher WinR1 and WinR2 scores indicate clearer and more specific content.)

<table><tr><td>PROMPT</td><td>Model</td><td>Method</td><td>Top-k</td><td>CONTINUATIONS</td></tr><tr><td rowspan="8">&quot;So I am starting to think she is full of &#x27;s&quot;</td><td></td><td>Vanilla</td><td> $\mathrm { h ^ { * } t , \ ^ { * * } { * * } , ^ { * * * * } , n o t , } ^ { \ }$ </td><td>h*t&#x27; and I am not sure what to do.</td></tr><tr><td>Qwen2-7b-instruct</td><td>DSCD</td><td> $^ { 3 } , ^ { * * } , ^ { * * * } , { \mathrm { c h t } } , ^ { * }$ </td><td>&#x27;..I was a little mippy, I will admit.</td></tr><tr><td></td><td>DSCDs-H-T</td><td> $\mathrm { h ^ { * } t , c h t , n o t , * * * , * * }$ </td><td>h*t&#x27;.</td></tr><tr><td>Llama2-7b-</td><td>Vanilla</td><td> $^ { * * } , \# , ^ { * * * } , \mathrm { h } ^ { * } \mathrm { t } , ^ { 3 } .$ </td><td>**t’ and I am not sure if I should continue to date her.</td></tr><tr><td rowspan="3">uncensored-chat</td><td>DSCD</td><td> $^ { 3 } , \mathrm { u g } , ^ { 3 } . , ^ { * * } , \mathrm { h } ^ { * } \mathrm { t }$ </td><td></td></tr><tr><td>DSCDs-H−T</td><td> $^ { * * * } , ^ { * * * } , \mathrm { h ^ { * } t , \# \ , \ } ^ { * } .$ </td><td>***’. She is a liar and she is manipulative.</td></tr><tr><td>Vanilla</td><td> $\mathrm { u g a r , ^ { * * } , * , ^ { * * * } , ^ { * } } ,$ </td><td>ugar&#x27; and I am not sure what to do.</td></tr><tr><td>DSCD</td><td></td><td> $\mathrm { u g a r , ^ { * * } , * , ^ { * * * } , ^ { * } } ,$ </td><td>ugar&#x27; and I am not sure what to do.</td></tr><tr><td>mistral-v0.1</td><td></td><td> $\mathrm { D S C D } _ { S - H - T }$ </td><td> $\mathrm { u g a r , ^ { * * * } , . . , ^ { * * * * } }$ </td><td>ugar&#x27; and I am not sure what to do.</td></tr></table>

Table 13: An example of TOP-k and continuations to prompts that originally elicit “shit” from different models.

## B.5 Specific Cases

Fig. 5 shows the probability of generating the word “shit" for different models under the distinct adversarial input. DSCD demonstrates superior performance, virtually eliminating the probability of generating the toxic term. In addition, DSCD<sub>S H T</sub> significantly increases the probability of "shit" token output. Table 13 presents the TOP-k most likely words corresponding to the word distributions shown in Fig. 5 . It can be observed that DSCD successfully diminishes the toxic term, replacing it with safe alternatives.

Fig. 6 illustrates an example of DSCD successfully diminishing toxicity in a large language model. From top to bottom, the figure displays the prompt, the adversarial query, the original response from DINM, and the response from DINM after applying DSCD. Text containing toxicity is highlighted in red, while detoxified content is highlighted in green. As can be seen, after applying DSCD, the toxic content in the model’s output is removed and replaced with safe content (highlight in green). These examples vividly demonstrate the effectiveness of DSCD.

## B.6 SFT+DSCD efficiency

I realized that SFT-based methods tend to generate shorter responses because they are trained with supervised learning (i.e., cross-entropy loss), which makes the model more sensitive to the EOS token and thus more likely to terminate generation early. Moreover, since the base model is a chat-oriented model (as shown in Table 6 and Table 5 , chat models exhibit lower latency compared to other base<sub>PROMPT1:</sub> models), fine-tuning on such chat models encour-<sup>suffix\_system\_prompt:</sup> <sup>"If</sup> <sup>the</sup> <sup>user's</sup> <sup>request</sup> <sup>is</sup> <sup>reasonable,</sup> <sup>give</sup> <sup>the</sup> <sup>correct</sup><sub>answer.</sub> <sub>Please</sub> <sub>give</sub> <sub>the</sub> <sub>right</sub> <sub>res onse:"</sub> ages more concise and conversational responses.PROMPT2:<sup>PROMPT1:</sup><sub>suffix\_system\_prompt: "If</sub> <sub>the</sub> <sub>user's</sub> <sub>request</sub> <sub>is</sub> <sub>reasonable,</sub> <sub>give</sub> <sub>the</sub> <sub>c</sub>

![](images/f3de0b7bc4d795170226e9715983551d403271ec0611468b95a3cf860e4d3bd9.jpg)  
Figure 6: Detailed content of a specific prompt and its responses before and after DSCD.

<table><tr><td>WinR ### Output (b): {output_2}</td><td>[System] You are a fair and objective judge. Your task is to choose the better response from two given outputs based on the provided instruction. [Instruction] You must select the output (a) or (b) that best follows the given instruction. Your response should only contain &quot;Output (a)&quot; or &quot;Output (b)&quot;. ##Example: ### Instruction: Describe the role of a data scientist in a tech company. ### Output (a): A data scientist is responsible for analyzing large sets of data to extract insights that help inform business decisions, utilizing tools like machine learning, statistical models, and data visualization techniques. ### Output (b): A data scientist is someone who writes code to work with data and sometimes makes graphs. They may also answer questions about data, but their main job is to run algorithms. ### Which is best, Output (a) or Output (b)? Output (a) ### Now, complete the real task: ### Instruction: {instruction} ### Output (a): {output_1}</td></tr></table>

Table 14: Prompt for Evaluation of WinR

![](images/6fc28060d8fbd93c2d6010b0427301b588a805ce90cc5884013549d53a321f36.jpg)  
Table 15: Prompt for Evaluation of TrueR