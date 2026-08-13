# Training a Utility-based Retriever Through Shared Context Attribution for Retrieval-Augmented Language Models

Yilong Xu<sup>1,2,3</sup> Jinhua Gao<sup>1,2</sup>\* Xiaoming Yu<sup>1,2</sup> Yuanhai Xue<sup>1,2</sup> Baolong Bi<sup>1,2,3</sup>

Huawei Shen<sup>1,2,3</sup> Xueqi Cheng<sup>1,2,3</sup>

<sup>1</sup>State Key Lab of AI Safety, Institute of Computing Technology, CAS

<sup>2</sup>Key Lab of AI Safety, Chinese Academy of Sciences

<sup>3</sup>University of Chinese Academy of Sciences

{xuyilong23s, gaojinhua}@ict.ac.cn

## Abstract

Retrieval-Augmented Language Models boost task performance, owing to the retriever that provides external knowledge. Although crucial, the retriever primarily focuses on semantics relevance, which may not always be effective for generation. Thus, utility-based retrieval has emerged as a promising topic, prioritizing passages that provide valid benefits for downstream tasks. However, due to insufficient understanding, capturing passage utility accurately remains unexplored. This work proposes SCARLet, a framework for training utility-based retrievers in RALMs, which incorporates two key factors, multi-task generalization and inter-passage interaction. First, SCAR-Let constructs shared context on which training data for various tasks is synthesized. This mitigates semantic bias from context differences, allowing retrievers to focus on learning task-specific utility and generalize across tasks. Next, SCARLet uses a perturbation-based attribution method to estimate passage-level utility for shared context, which reflects interactions between passages and provides more accurate feedback. We evaluate our approach on ten datasets across various tasks, both indomain and out-of-domain, showing that retrievers trained by SCARLet consistently improve the overall performance of RALMs.

Resources: § github.com/ylXuu/SCARLet

## 1 Introduction

Retrieval-Augmented Language Models (RALMs; Lewis et al., 2020) typically comprise two parts: the retriever and the generator. The retriever collects up-to-date task-related external information, while the generator incorporates the collected nonparametric knowledge into inference. RALMs have achieved enhanced performance across various downstream tasks, including question answering, fact checking, and dialogue generation (Shao et al., 2023; Cheng et al., 2023). As a crucial role, the optimization of the retrievers in RALMs has become a trending research topic.

Early RALMs adopt relevance-based retrievers, including both sparse (Robertson and Zaragoza, 2009) and dense (Karpukhin et al., 2020) models. However, these retrievers are primarily biased toward semantic relevance (Wu et al., 2024), failing to consider the passage utility and leading to misalignment in RALMs. The utility, measuring the valid gain that a passage contributes to the downstream generation (Zhang et al., 2024), can bridge the gap between the retriever and the generator. Some recent works have proposed to optimize retrievers by constructing feedback from generators (Shi et al., 2023; Yu et al., 2023; Wei et al., 2024), achieving promising results. Nonetheless, how to align retrievers to better capture utility remains an open yet challenging problem.

Different from relevance, which is mainly determined by the query and the passage (Xia et al., 2015; Guo et al., 2019), utility needs a more comprehensive measurement. In this paper, we propose the following two vital yet overlooked factors for utility modeling in RALMs:

Multi-task Generalization RALMs need to accommodate various downstream tasks, where the utility of a passage can vary accordingly. Existing methods typically optimize retrievers using the pooling strategy, i.e., mixing data from different tasks for training, to learn task-specific retrieval criteria (Lin et al., 2024; Zamani and Bendersky, 2024). However, since pooled samples from different tasks typically have different contexts, the trained retrievers might tend to capture semantic relevance signals instead of utility features. Such unexpected preference will downgrade the retrievers’ generalization ability, especially for those with weaker linguistic capabilities (Liu et al., 2024).

Inter-passage Interaction In some complex tasks, the utility of a certain passage cannot be solely determined by itself. For example, when handling multi-hop question-answering tasks, the model should rely on preceding and even succeeding contexts in the reasoning chain to judge a passage’s utility. However, the utility signals constructed in previous works either fail to provide passage-level feedback (Zamani and Bendersky, 2024; Sohn et al., 2024) or evaluate each passage independently (Yu et al., 2023; Shi et al., 2023), leading to imprecise utility measurements.

In this paper, we propose a novel framework to train utility-based retrievers for RALMs, named SCARLet, representing shared context attribution supervised training for utility-based retrievers.

Specifically, SCARLet first introduces a training data synthesis pipeline. Contrary to the previous pooling strategy that mixes training data with different contexts, our pipeline first constructs a shared context, and subsequently synthesizes training data for various downstream tasks derived from the shared context. This method mitigates the semantic interference by achieving single-variable control, and enables the retriever to focus on learning taskspecific utility. To better assess the utility of certain passages, SCARLet employs a passage-level perturbation-based technique, which randomly removes some passages from the context and measures the fluctuations in the generated output. Such a design can effectively capture the synergy between passages, thereby accurately reflecting their utility. Finally, SCARLet collects positive and negative samples based on the utility scores and trains the retriever in a contrastive way.

We conduct extensive experiments to evaluate the performance gain brought by SCARLet. Our experiments adopt ten datasets, covering eight distinct tasks that are frequently used for RALMs evaluation. The results show that RALMs equipped with retrievers trained by SCARLet, consistently achieve optimal or suboptimal downstream performance across all datasets. Moreover, further analysis and case studies demonstrate that SCARLet can better capture utility signals.

To summarize, our main contributions include:

• We argue that utility should be preferred in RALMs and propose two critical factors for training utility-based retrievers.

• We propose SCARLet, a novel framework to train utility-based retrievers through shared context synthesis and utility attribution.

• We conduct extensive experiments across various tasks, demonstrating that our proposed SCARLet can improve the overall performance of RALMs.

## 2 Related Work

RALMs Large Language Models (LLMs; Brown et al., 2020) exhibit remarkable performance across a wide range of tasks (Zhao et al., 2024; Naveed et al., 2024; Wei et al., 2022). However, LLMs also face the challenge of hallucinations, often performing poorly when addressing factual issues (Huang et al., 2024; Bi et al., 2024). The emergence of RALMs effectively alleviates the weakness of insufficient factuality (Gao et al., 2024). A RALM system typically comprise a retriever and a generator, where the retriever recalls external information to enhance the generator to respond more accurately. To further optimize RALMs and improve the synergy between the two parts, existing methods generally fall into three categories: 1) overall optimization (Lin et al., 2024; Zamani and Bendersky, 2024); 2) generator-only optimization (Fang et al., 2024; Yu et al., 2024; Bi et al., 2025); 3) retriever-only optimization (Shi et al., 2023; Yu et al., 2023). Optimizing only the retriever is a more efficient and cost-effective way that offers plug-and-play capabilities, enhancing the overall efficiency and stability of the RALM system.

Utility-based Retrieval In RALMs, early exploration of retrieval utility focuses on capturing the downstream feedback of generators. Salemi and Zamani (2024) propose supervision based on downstream task metrics, but fail to provide passagelevel utility feedback. Shi et al. (2023); Yu et al. (2023) assess utility of each passage using generator outputs, but they ignore the interactions between passages. Sohn et al. (2024); Wei et al. (2024) employ the generator’s self-reflection to evaluate utility, which may bring hallucinations as the language models can be dishonest (Madsen et al., 2024). Asai et al. (2023); Glass et al. (2022) notice the multi-task nature of the retrieval stage, but fail to account for the training biases introduced by contextual differences in the pooling strategy.

Therefore, our proposed SCARLet framework comprehensively considers the above issues of multi-task generalization and utility assessment, offering a novel pipeline with shared context synthesis and utility attribution to effectively train utilitybased retrievers in RALMs.

![](images/de6bde737646e9b574405137bff9414dda7e4255a9f1dabe74337170dda67b27.jpg)  
Figure 1: The illustration of SCARLet. The upper left part describes the inference process of RALMs. In SCARLet, there are three main stages. First, the shared context is constructed by retrieving external corpus based on the seed data. The synthesizer is instructed with shared context and task information from the task pool, to generate synthetic data. Next, using the shared context as the data source, SCARLet applies perturbation-based utility attribution on the generator, and then, based on the utility scores, samples positive and negative passages for retriever training.

## 3 Method

In this section, we first define the RALMs system, then we introduce the SCARLet pipeline.

## 3.1 Definitions

A typical RALM system consists of a retriever and a generator. During the retrieval stage, we employ a dense retriever based on an encoder Enc with parameters ϕ. And the retriever interacts with an external corpus $\mathcal { C } .$ For a query q, we calculate the dot product of the embeddings of $q$ and each passage d in ${ \mathcal { C } } ,$ as follows:

$$
\operatorname { s c o r e } \left( q , d \right) = \mathbf { E n c } _ { \phi } \left( q \right) \cdot \mathbf { E n c } _ { \phi } \left( d \right) , d \in \mathcal { C } .\tag{1}
$$

The top-k passages with the highest scores are selected and added to the context, denoted as $D = [ d _ { 1 } , \ldots , d _ { k } ]$ . Note that RALMs need to accommodate various downstream tasks, for a task $T$ and an input $x$ from its dataset, we define the query format as $q = I \oplus x ,$ , where I denotes the instruction description of task $T$

In the generation stage, a language model LM with parameters θ serves as the generator. The context D is used to enhance generation, ultimately producing the predicted output $\hat { \mathbf { y } } .$ , as shown below:

$$
{ \hat { \mathbf { y } } } = \operatorname { L M } _ { \theta } \left( I \oplus x \oplus D \right) ,\tag{2}
$$

where $\hat { \mathbf { y } }$ is a sequence and $\hat { \mathbf { y } } _ { t }$ denotes the t-th token. We denote the ground truth of x as $\mathbf { y }$

## 3.2 Overview of SCARLet

The overall architecture of our proposed SCARLet is shown in Figure 1, including shared context synthesis and training data construction (§3.3), utility attribution modeling (§3.4), as well as data sampling and retriever tuning (§3.5).

Shared context refers to the common context for data of different tasks in the training stage, which is then used to enhance downstream generation. Previous studies employ the pooling strategy (Lin et al., 2024; Zamani and Bendersky, 2024), where each instance has a distinct context for training. Learning task-specific features to improve multitask generalization of utility-based retrieval might be disturbed by the semantically relevant noise introduced by differences in context, leading to unexpected preference, particularly in retrievers with weaker linguistic capabilities. To tackle the above challenges, our proposed SCARLet adopts a reverse strategy, first constructing shared context to narrow the semantic gap, and then synthesizing task-specific data based on this context. Sharing context across tasks can highlight utility feature differences, making it easier to learn. Moreover, LLM-driven data synthesis has been shown to be a promising way (Long et al., 2024; Kim and Baek, 2025), which can effectively reduce labor costs.

Utility attribution modeling refers to local explanation techniques to build utility signals from the downstream generation. More specifically, we adopt the contributive attribution model, which measures how the input context contributes to the model’s output and aligns well with the definition of utility in RALMs. Previous research on optimizing retrievers from downstream generation, either fails to construct passage-level feedback or only considers the individual impact of each passage, overlooking the synergy effects between passages. Therefore, taking the shared context as the source data, SCARLet uses a passage-level perturbationbased utility attribution approach, where fluctuations in generation caused by perturbations can reflect interactions between passages and then be quantified as utility scores.

## 3.3 Shared Context Synthesis

Specifically, we first define a task pool $\tau ,$ which is linked to various downstream tasks and their datasets, such as multi-hop QA, long-form $\mathrm { Q A }$ and fact checking. We begin by collecting seed data from datasets of $\tau _ { \ast }$ , including task instructions, inputs, and ground truth. In line with the motivation behind shared context, passages within this context need to be closely related to facilitate the synthesis of high-quality data. Therefore, we employ an approach based on associated entities, which extracts entities from the seed data, searches their adjacent entities by querying Wikidata<sup>1</sup>, and merges them to obtain a related entity list. We then use this list to retrieve relevant passages from the Wikipedia corpus, and treat the recalled passages as the shared context $D _ { \mathrm { s h a r e d } }$ . Subsequently, we instruct the synthesizer model S to generate new training data, using $D _ { \mathrm { s h a r e d } }$ as the information source and task information (including instructions and examples) from $\tau$ . The process is formalized as follows:

$$
\left( x _ { T _ { 1 } } ^ { \mathrm { n e w } } , \mathbf { y } _ { T _ { 1 } } ^ { \mathrm { n e w } } \right) , \ldots , \left( x _ { T _ { l } } ^ { \mathrm { n e w } } , \mathbf { y } _ { T _ { l } } ^ { \mathrm { n e w } } \right) = { \mathsf { S } } \left( D _ { \mathrm { s h a r e d } } , { \mathcal { T } } \right) ,\tag{3}
$$

where $x _ { T _ { i } } ^ { \mathrm { n e w } }$ and $\mathbf { y } _ { T _ { i } } ^ { \mathrm { n e w } }$ represent input and ground truth of the synthetic data of task $T _ { i }$ , respectively. l is the total number of tasks in $\tau$

To improve the quality of synthetic data, the task pool not only provides the task instructions but also offers example data. To further improve robustness, following Fang et al. (2024); Zhang et al. (2024), we also introduce synthetic noise into the shared context by instructing the synthesizer to generate semantically relevant but useless passages. In addition, we incorporate a data filtering step that instructs the synthesizer to eliminate samples containing faults. For more details, please refer to Appendix A. We also provide an example of shared context in Appendix F.

## 3.4 Passage-level Utility Attribution

Specifically, the context D recalled by the upstream retriever consists of k passages. To evaluate the utility of each individual passage with inter-passage interactions, we adopt a perturbation-based method where we remove certain passages and inspect the changes in the final output. The approach is implemented via introducing a perturbation vector $\bar { \mathbf { v } } \in \{ 0 , 1 \} ^ { k }$ , where 0 and 1 indicate whether the corresponding passage is removed or included, respectively. However, running all generations of $2 ^ { \bar { k } }$ possible perturbation vectors can result in significant computational overhead. Inspired by the method of Local Interpretable Model-agnostic Explanations (LIME; Ribeiro et al., 2016; Mardaoui and Garreau, 2021), we first sample n perturbation vectors randomly and then fit a surrogate model for predicting the utility score, as shown below:

$$
\hat { \pmb { \alpha } } \in \underset { { \pmb { \alpha } } \in \mathbb { R } ^ { k + 1 } } { \arg \operatorname* { m i n } } \left\{ \sum _ { i = 1 } ^ { n } \left( z _ { i } - { \pmb { \alpha } } ^ { T } { \mathbf { v } } _ { i } \right) ^ { 2 } + \lambda \left\| { \pmb { \alpha } } \right\| ^ { 2 } \right\} ,\tag{4}
$$

where we adopt the ridge regression (Hilt and Seegrist, 1977) as our surrogate model, α represents the parameters to be fitted, $\lambda$ is a hyperparameter for regularization, and $z _ { i }$ is the observed value under $\mathbf { v } _ { i }$ . More specifically, $\mathbf { \alpha } \alpha ^ { ( i ) }$ denotes the utility score of passage $d _ { i } , { \pmb { \alpha } } ^ { ( 0 ) }$ represents the intercept term. And $z _ { i }$ , which quantifies the fluctuation caused by ${ \bf v } _ { i } .$ , is calculated using the logit values of the tokens in the ground truth y at each time step, as shown below:

$$
z _ { i } = \sum _ { t } \log \mathrm { i t } \left( \mathbf { y } _ { t } ^ { ( i ) } \right) .\tag{5}
$$

To evaluate the effectiveness of the above utility attribution method, we conduct a preliminary experiment on the GTI benchmark (Zhang et al., 2024), which includes three datasets: HotpotQA (Yang et al., 2018), Natural Questions (NQ; Kwiatkowski et al., 2019), and MSMARCO-QA (Bajaj et al., 2018). Each test sample comprises input, ground truth, and ten passages including correct passages and other noise passages. We use the utility score to rank the passages. The results, measured using nDCG, demonstrate that our method shows a high accuracy in reflecting passage utility, as shown in Figure 2. We also compare our method to other attribution approaches, and our method outperforms them by over 20%. For further details of the experiment, please refer to Appendix B.

![](images/e2a9388b90c6bf55ff1f07e0f1d399707fb781038804d1b523c5b0cb5cbcf3b3.jpg)  
Figure 2: The performance of the perturbation-based attribution method on the GTI benchmark. The nDCG metrics show that it achieves at least about 80% performance on three datasets, with some exceeding 90%.

## 3.5 Sampling and Training

After calculating the utility score for each passage in the shared context, we then collect positive and negative samples based on these scores for training the retriever. When sorted in descending order of the scores, the utility distribution follows an inverse S-shaped curve, as depicted in Figure 3. Passages with higher scores correspond to positive samples, while those with lower scores represent negative samples. To effectively sample these two types of data, we employ a one-dimensional clustering approach. Specifically, we take the utility score list as the input and divide it into three clusters: one for the positive samples, one for the intermediate samples that will be discarded, and another for the negative samples. This method can dynamically adjust the number of useful passages in the context on various tasks and data.

After obtaining positive and negative samples, following Xiong et al. (2020), the loss function is calculated as follows:

$$
\begin{array} { c } { { \mathcal { L } = \displaystyle \sum _ { x } \sum _ { d ^ { + } \in D ^ { + } } \sum _ { d ^ { - } \in D ^ { - } } } } \\ { { l \left( \mathrm { s c o r e } \left( x , d ^ { + } \right) , \mathrm { s c o r e } \left( x , d ^ { - } \right) \right) , } } \end{array}\tag{6}
$$

where l represents the cross-entropy loss.

![](images/d921002481dbfcd85592e93cc0f5ced76dc0adbc62b4027957e7c485b580c61d.jpg)

![](images/c7f063d78e73256de81c7844569f5dcca46d8294b828c42853f12bc6cf33cd01.jpg)

Figure 3: The illustration of the 1D clustering sampling. Based on the utility score, this method clusters the passages into three labels: the high-score passages (green) corresponding to positive samples, the middle-score passages (orange) that will be discarded, and the low-score passages (red) corresponding to negative samples.
<table><tr><td>Dataset</td><td>Task</td><td>Corpus</td><td>Metric</td></tr><tr><td colspan="4">In-domain</td></tr><tr><td>NQ (Kwiatkowski et al., 2019)</td><td>Single-hop QA</td><td>Wikipedia</td><td>Accuracy</td></tr><tr><td>HotpotQA (Yang et al., 2018)</td><td>Multi-hop QA</td><td>Wikipedia</td><td>Accuracy</td></tr><tr><td>ELI5 (Fan et al., 2019)</td><td>Long-form QA</td><td>Wikipedia</td><td>ROUGE-L</td></tr><tr><td>FEVER (Thorne et al., 2018)</td><td>Fact checking</td><td>Wikipedia</td><td>Accuracy</td></tr><tr><td>WoW (Dinan et al., 2019)</td><td>Dialogue generation</td><td>Wikipedia</td><td>Fl</td></tr><tr><td>T-REx (Elsahar et al., 2018)</td><td>Slot filling</td><td>Wikipedia</td><td>Accuracy</td></tr><tr><td colspan="4">Out-of-domain</td></tr><tr><td>zs-RE (Levy et al., 2017)</td><td>Relation extraction</td><td>Wikipedia</td><td>Accuracy</td></tr><tr><td>SciFact (Wadden et al., 2020)</td><td>Fact checking</td><td>BeIR</td><td>Accuracy</td></tr><tr><td>Climate-FEVER (Diggelmann et al., 2021)</td><td>Fact checking</td><td>BeIR</td><td>Accuracy</td></tr><tr><td>FiQA (Maia et al., 2018)</td><td>Financial QA</td><td>BeIR</td><td>ROUGE-L</td></tr></table>

Table 1: The datasets used in the main experiment. Climate-Fever is a four-class classification task, while the other two fact-checking tasks are binary. For metrics, NQ, HotpotQA, T-REx, and zs-RE all calculate accuracy based on exact substring matching.

## 4 Experimental Setup

This section introduces the main experiment setup, including datasets, baselines and implementation.

## 4.1 Datasets and Evaluation

We collect both in-domain and out-of-domain datasets for our experiments. In-domain datasets are utilized for providing seed data to construct synthetic training data, while out-of-domain datasets possess different tasks and corpora and are collected for further generalization tests. We collect seven datasets from KILT (Petroni et al., 2021), and three from BeIR (Thakur et al., 2021), as detailed in Table 1. All KILT datasets utilize Wikipedia dump dated $2 0 1 9 - 0 8 – 0 1 ^ { 2 }$ as the corpus. Following Wang et al. (2019), we split the original articles into segments with a maximum length of 100 words, resulting in a total of 28,773,800 passages. For test sets of BeIR, we adopt their self-constructed corpora. For retrieval, we follow the closed corpus setup(Asai et al., 2023), where RALMs only retrieve from the corpus of the current dataset. For the test data, we randomly sample 1,000 data from the test split of each dataset.

<table><tr><td rowspan="2">Method</td><td colspan="6">In-domain</td><td colspan="4">Out-of-domain</td></tr><tr><td>NQ</td><td>HotpotQA</td><td>ELI5</td><td>FEVER</td><td>WoW</td><td>T-REx</td><td>Zs-RE</td><td>SciFact</td><td>C-FEVER</td><td>FiQA</td></tr><tr><td></td><td colspan="8">LLaMA-3-8B-Instruct</td><td>45.8</td><td></td></tr><tr><td>No retrieval</td><td>43.5</td><td>36.8</td><td>14.8</td><td>79.8</td><td>9.3</td><td>34.5</td><td>21.7</td><td>68.0</td><td></td><td>17.2</td></tr><tr><td>Contriever</td><td>43.8</td><td>36.7</td><td>14.5</td><td>78.5</td><td>8.6</td><td>33.6</td><td>20.4</td><td>70.1</td><td>38.2</td><td>16.2</td></tr><tr><td>BGE</td><td>47.5</td><td>41.6</td><td>15.2</td><td>83.5</td><td>8.7</td><td>36.4</td><td>22.7</td><td>83.3</td><td>44.9</td><td>21.0</td></tr><tr><td>AARContriever</td><td>44.9</td><td>39.9</td><td>15.0 13.8</td><td>77.2</td><td>8.3</td><td>34.4</td><td>21.0</td><td>73.6</td><td>39.2</td><td>16.7</td></tr><tr><td>REPLUGContriever</td><td>43.3</td><td>38.9</td><td></td><td>80.0</td><td>9.4</td><td>33.1</td><td>22.8</td><td>74.6</td><td>41.2</td><td>18.9</td></tr><tr><td>SCARLetContriever SCARLetBGE</td><td>44.6</td><td>40.5</td><td>15.8</td><td>80.6</td><td>11.0</td><td>35.8</td><td>21.0</td><td>75.5</td><td>42.8</td><td>17.7</td></tr><tr><td>49.2</td><td></td><td>47.0</td><td>16.3</td><td>81.3</td><td>12.2</td><td>37.0</td><td>24.4</td><td>82.2</td><td>46.1</td><td>22.9</td></tr><tr><td colspan="9">Qwen2.5-3B-Instruct</td><td></td></tr><tr><td>No retrieval</td><td>27.4</td><td>26.5</td><td>15.2</td><td>66.1</td><td>11.5</td><td>26.0</td><td>7.3</td><td>58.2</td><td>40.4</td><td>17.7</td></tr><tr><td>Contriever</td><td>32.6</td><td>28.8</td><td>14.3</td><td>67.0</td><td>10.5</td><td>27.2</td><td>14.3</td><td>64.9</td><td>31.6</td><td>15.5</td></tr><tr><td>BGE</td><td>46.8</td><td>39.6</td><td>13.7</td><td>78.2</td><td>10.4</td><td>29.3</td><td>15.5</td><td>70.6</td><td>30.2</td><td>18.7</td></tr><tr><td>AARContriever</td><td>34.1</td><td>29.7</td><td>13.8</td><td>66.6</td><td>10.1</td><td>28.7</td><td>15.2</td><td>63.6</td><td>32.2</td><td>16.1</td></tr><tr><td> $\mathtt { R E P L U G } _ { \mathrm { C o n t r i e v e r } }$ </td><td>33.7</td><td>34.0</td><td>14.0</td><td>71.4</td><td>12.2</td><td>26.9</td><td>16.2</td><td>61.1</td><td>30.6</td><td>19.0</td></tr><tr><td>SCARLetContriever</td><td>38.2</td><td>35.4</td><td>14.9</td><td>70.8</td><td>11.7</td><td>28.0</td><td>19.1</td><td>65.3</td><td>31.7</td><td>17.3</td></tr><tr><td>SCARLetBGE</td><td>44.9</td><td>41.1</td><td>15.2</td><td>74.3</td><td>12.6</td><td>29.7</td><td>16.6</td><td>62.3</td><td>33.0</td><td>20.4</td></tr></table>

Table 2: Results of the main experiment across datasets on different downstream generators. AAR<sub>Contriever</sub>, REPLUG<sub>Contriever</sub>, SCARLet<sub>Contriever</sub> represent the baselines initilized from Contriever, and SCARLet<sub>BGE</sub> represents the baseline initialized from BGE-base-v1.5. The bold score means the best performance of the corresponding dataset among baselines within the same generator, while the underline score means the second best.

For evaluation metrics, we mainly assess the performance of downstream tasks. For WoW, we use F1. For ELI5 and FiQA, we use ROUGE-L. For other datasets, we use accuracy.

## 4.2 Baselines

The baselines are categorized into three settings:

No Retrieval The downstream generators operate without any retrieval.

Vanilla RAG Retrievers are added and the recalled passages are incorporated into the generation process. We choose two well-trained embedding models, Contriever (Izacard et al., 2022) and BGEbase-v1.5 (Xiao et al., 2023) as the retrievers.

Retriever-only Optimization Retrievers are optimized using feedback from the generator. We select two recent methods, RePlug (Shi et al., 2023) and AAR (Yu et al., 2023), both of which are initialized from Contriever.

We utilize LLaMA-3-8B-Instruct (AI@Meta, 2024) and Qwen2.5-3B-Instruct (Team, 2024) as the generators in RALMs. All retrieval-based baselines use the top-3 passages. Given that some retrievers may not be tuned by instructions, the query format for Contriever and its baselines only contains x, without task instruction I. For BGE and its baselines, the query format follows the definition in Section 3.1, which contains both x and I.

## 4.3 Implementation Details

In the shared context synthesis stage, we add the six tasks of the in-domain datasets into the task pool. We then randomly sample 1,000 data from the training split of each dataset to construct the seed dataset. We only consider one-hop relation when searching adjacent entities. For each entity, the top-10 passages are retrieved from , and the shared context is formed by selecting the top-10 passages across all retrieved passages. We utilize gpt-4o-2024-11-20 (OpenAI, 2024) as the synthesizer model. For more implementation details and meta data, please see Appendix C.

## 5 Results

In this section, we present the results of main experiment (§5.1), ablation study (§5.2), retrieval evaluation (§5.3), and case study (§5.4).

## 5.1 Overall Performance

The main experimental results are shown in Table 2. Our proposed SCARLet method achieves either optimal or suboptimal performance across various datasets and generators, demonstrating its effectiveness. Our detailed analysis from different perspectives is as follows:

In-domain Performance In the evaluation on six in-domain datasets, the retrievers trained by SCAR-Let achieve the best performance in five datasets when using LLaMA-3-8B as the generator, and in four datasets when using Qwen-2.5-3B as the generator. Except for NQ and FEVER, SCARLet consistently outperforms the initial baselines, including Contriever and BGE.

Out-of-domain Performance In the evaluation on four out-of-domain datasets, SCARLet also achieves optimal or suboptimal results. Specifically, SCARLet can still show progress in Sci-Fact, Climate-FEVER, and FiQA, whose corpora differ from the Wikipedia corpus used in training and whose domains are notably different from the in-domain datasets, highlighting its generalization across corpora. In addition, SCARLet can achieve overall improvements when using two different downstream LLMs, preliminarily indicating its adaptability across generators.

## 5.2 Ablation Study

According to the pipeline of SCARLet, we design the ablation experiments from three stages: 1) In the data synthesis stage, we evaluate the method of removing the step of retrieving adjacent entities, and instead directly retrieving the top-k passages from the corpus using only the entities extracted from the seed data; 2) In the utility attribution stage, since Section 3.4 already compares various attribution methods and demonstrates the superiority of our perturbation-based approach, we no longer conduct ablation study for this part; 3) In the sampling and training stage, we assess the effect of removing the one-dimensional clustering step and instead directly selecting the highest-scoring passage as the positive sample and the five lowest-scoring passages as negative samples based on the scores.

The comparison results, presented in Figure 4, show that removing either of the two components leads to a significant performance drop. Without adjacent entities retrieval, we believe that the original entity list may contain insufficient information, making it challenging to construct a shared context that effectively supports multi-task data synthesis. And the weaker entity associations can disrupt the connection between peer passages in the shared context, ultimately degrading the quality of the synthesized data. Furthermore, without onedimensional clustering sampling, we suggest that it reduces the number of positive samples, which can be particularly detrimental to retrieval tasks requiring multiple reasoning hops.

![](images/422cfc28bf23d14752fccf3d1361d7ca3a32455d83ef64df4f128b3345cbd0cd.jpg)  
Figure 4: Ablation Study on six in-domain datasets, using BGE as retriever, with two generators. The values in the charts correspond to the metrics of each dataset.

## 5.3 Aspects of Retrieval Utility

The previous experiment evaluates the overall performance improvement of RALMs brought by SCARLet. However, in essence, SCARLet is an optimization method of the retrieval stage. Moreover, despite discussing the utility as the valid gain for downstream generation in RALMs, neither existing work nor this study can explicitly define utility-based retrieval. To assess the effectiveness of SCARLet in improving retrieval performance, we select three retrieval benchmarks, each representing a distinct aspect of retrieval utility based on our understanding, as shown below:

GTI This benchmark was introduced in Section 3.4. Its goal is to evaluate whether retrievers can bypass pitfalls of semantic relevance and prioritize passages that are useful for answering questions.

BRIGHT This benchmark focuses on the reasoning implied in retrieval (Su et al., 2024), particularly for complex queries that require the retriever to engage in deep reasoning to identify useful passages, beyond simple semantic relevance. Dai et al. (2024) also argue that the entailment reasoning between passages and queries is essential for enhancing retrieval capabilities. We believe that recognizing retrieval utility requires reasoning, such as distinguishing task-specific features and determining the appropriate number of hops.

X<sup>2</sup>-Retrieval This benchmark focuses on retrieval across multiple tasks and scenarios (Asai et al., 2023), where understanding the intent behind user’s queries becomes crucial. We suggest that this corresponds to identifying the target utility anticipated by the downstream tasks.

<table><tr><td></td><td colspan="2">HotpotQA</td><td colspan="2">NQ</td><td colspan="2">MSMARCO-QA</td></tr><tr><td>Method</td><td>NDCG@1</td><td>NDCG@5</td><td>NDCG@1</td><td>NDCG@5</td><td>NDCG@1</td><td>NDCG@5</td></tr><tr><td>Contriever</td><td>33.3</td><td>48.0</td><td>10.0</td><td>35.8</td><td>16.8</td><td>37.0</td></tr><tr><td>SCARLetContriever</td><td>41.3(+8.0)</td><td>52.1(+4.1)</td><td>17.5(+7.5)</td><td>45.3(+9.5)</td><td>21.9(+5.1)</td><td>44.1(+7.1)</td></tr><tr><td>BGE</td><td>70.3</td><td>70.1</td><td>30.3</td><td>60.2</td><td>47.8</td><td>71.9</td></tr><tr><td>SCARLetBGE</td><td>72.8(+2.5)</td><td>76.7(+6.6)</td><td>33.4(+3.1)</td><td>64.4(+4.2)</td><td>53.2(+5.4)</td><td>77.0(+5.1)</td></tr></table>

Table 3: Evaluation results on GTI, reporting nDCG for each datasets. Bracketed values indicate the changes in metrics compared to the initial model.
<table><tr><td>Model</td><td>StackExchange</td><td>Coding</td><td>Theorem-based</td></tr><tr><td>Contriever</td><td>10.5</td><td>19.6</td><td>6.9</td></tr><tr><td>SCARLetContriever</td><td> $1 3 . 3 _ { ( + 2 . 8 ) }$ </td><td> $1 9 . 2 _ { ( - 0 . 4 ) }$ </td><td> $8 . 7 _ { ( + 1 . 8 ) }$ </td></tr><tr><td>BGE</td><td>14.9</td><td>16.0</td><td>8.1</td></tr><tr><td> $\mathbf { S C A R L e t } _ { \mathrm { B G E } }$ </td><td> $1 6 . 2 _ { ( + 1 . 3 ) }$ </td><td> $1 4 . 4 _ { ( - 1 . 6 ) }$ </td><td> $9 . 2 _ { ( + 1 . 1 ) }$ </td></tr></table>

Table 4: Evaluation results on BRIGHT, reporting nDCG@10 for each datasets. Bracketed values indicate the changes in metrics compared to the initial model.
<table><tr><td>Model</td><td>AMB</td><td>WQA</td><td>GAT</td><td>LSO</td><td>CSP</td></tr><tr><td>Contriever</td><td>96.8</td><td>80.9</td><td>73.2</td><td>28.0</td><td>36.7</td></tr><tr><td>SCARLetContriever</td><td> $9 7 . 5 _ { ( + 0 . 7 ) }$ </td><td>85.8(+5.1)</td><td> $7 1 . 6 _ { ( - 1 . 6 ) }$ </td><td> $2 0 . 9 _ { ( - 7 . 1 ) }$ </td><td> $2 4 . 8 _ { ( - 1 1 . 9 ) }$ </td></tr><tr><td>BGE</td><td>97.3</td><td>84.0</td><td>77.4</td><td>30.1</td><td>38.2</td></tr><tr><td>SCARLetBGE</td><td> $9 8 . 3 _ { ( + 1 . 0 ) }$ </td><td>86.1(+2.1)</td><td> $7 7 . 8 _ { ( + 0 . 4 ) }$ </td><td> $2 7 . 5 _ { ( - 2 . 6 ) }$ </td><td> $3 4 . 9 _ { ( - 3 . 3 ) }$ </td></tr></table>

Table 5: Evaluation results on $\mathbb { X } ^ { 2 } .$ -Retrieval, averaged nDCG@10 for each datasets. Bracketed values indicate the changes in metrics compared to the initial model.

We choose Contriever and BGE as the retriever models, using LLaMA-3-8B-Instruct as the downstream generator to implement SCARLet training. We compare the performance of the trained retrievers with the initial retrievers on two benchmarks, as shown in Table 3, 4 and 5, respectively. The results indicate that SCARLet improves performance on some datasets, but its effectiveness is generally limited for code-related tasks, such as LinkSo (Liu et al., 2018) and CodeSearchNet (Husain et al., 2020). The reasons could be: 1) the significant difference between the code domain and our selected in-domain datasets, which may hinder generalization; 2) the retriever models used are relatively lightweight, making it susceptible to catastrophic forgetting during training; 3) the optimization is related to downstream generators, but feedback related to the code domain cannot be obtained.

## 5.4 Case Study

Multi-hop QA is a task that requires multiple pieces of information and multi-step reasoning to derive the solution (Mavi et al., 2024). Given the characteristics of the task, we believe that retrieval utility should point to passages that may contain information necessary for the reasoning chain. We select a representative example from the test split of the HotpotQA dataset, as shown in Figure 5. To answer the question, the reasoning chain is: knowing information about William Preston, identifying the 1996 American historical drama he appeared in, finding information about that drama, and determining its writer. Directly relevant information about William Preston is relatively easy to define. However, the shown passage which corresponds to the final reasoning step, has a poor match with the question in terms of semantic relevance. And BGE ranks it 8th. After training by SCARLet, the passage achieves a higher ranking of 3rd. For more case studies, please refer to Appendix E.

![](images/00bcf1ce2f591deb60448e640fde3fd4fb1e3bca001fbfa638ec195d7f0151ba.jpg)  
Figure 5: Case Study on HotpotQA. The passage is ranked variously by different retrievers. Orange text indicates necessary reasoning information.

## 6 Conclusion

This study focuses on utility-based retrieval, a paradigm that moves beyond semantic relevance to prioritize downstream task performance in RALMs. We highlight two key challenges faced by existing research. To solve the limitations, we propose SCARLet, a novel framework to enhance utilitybased retrieval. To mitigate semantic interference on utility features during training, SCARLet incorporates a shared context synthesis method, which narrows the semantic gap between different tasks. To address the issue of inaccurate passage-level utility estimation, SCARLet employs a perturbationbased attribution method to capture the synergy between passages. Lastly, SCARLet utilizes a onedimensional clustering method to sample positive and negative passages for retriever optimization. Through experiments, we demonstrate that SCAR-Let can effectively enhance the overall performance of RALMs, and brings improvements in complex retrieval benchmarks. We hope this study can inspire further research on utility-based retrieval.

## Limitations

This study only covers several classic downstream datasets. We believe that incorporating a task augmentation stage could further enhance generalization, which we leave for future work. Moreover, there is a noticeable decline in retrieval performance in the code domain during generalization tests. Therefore, future work should also focus on improving the integration of different corpus structures. In addition, due to environmental constraints, this study does not evaluate larger-scale retrievers and generators. Furthermore, we also try GPT-4o-mini as the synthesizer, which performed poorly. Thus our framework should be equipped with models with stronger reasoning capabilities.

## Ethics Statement

All datasets and corpora involved in this study are publicly available, and we ensure that all used data comply with the usage and privacy policies established by the original authors. The synthetic data is exclusively used for training the retriever model. Moreover, given the security assurance of the synthesizer model, the probability of generating harmful passages and data is extremely minimal.

## Acknowledgments

This work was supported by the National Key R&D Program of China (2023YFC3303800).

## References

AI@Meta. 2024. Llama 3 model card.

Akari Asai, Timo Schick, Patrick Lewis, Xilun Chen, Gautier Izacard, Sebastian Riedel, Hannaneh Hajishirzi, and Wen-tau Yih. 2023. Task-aware retrieval with instructions. In Findings of the Association for Computational Linguistics: ACL 2023, pages 3650– 3675, Toronto, Canada. Association for Computational Linguistics.

Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, Mir Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary, and Tong Wang. 2018. Ms marco: A human

generated machine reading comprehension dataset. Preprint, arXiv:1611.09268.

Baolong Bi, Shenghua Liu, Yiwei Wang, Lingrui Mei, Hongcheng Gao, Yilong Xu, and Xueqi Cheng. 2024. Adaptive token biaser: Knowledge editing via biasing key entities. Preprint, arXiv:2406.12468.

Baolong Bi, Shenghua Liu, Yiwei Wang, Yilong Xu, Junfeng Fang, Lingrui Mei, and Xueqi Cheng. 2025. Parameters vs. context: Fine-grained control of knowledge reliance in language models. arXiv preprint arXiv:2503.15888.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877–1901. Curran Associates, Inc.

Canyu Chen and Kai Shu. 2024. Can llmgenerated misinformation be detected? Preprint, arXiv:2309.13788.

Xin Cheng, Di Luo, Xiuying Chen, Lemao Liu, Dongyan Zhao, and Rui Yan. 2023. Lift yourself up: Retrieval-augmented text generation with selfmemory. In Advances in Neural Information Processing Systems, volume 36, pages 43780–43799. Curran Associates, Inc.

Lu Dai, Hao Liu, and Hui Xiong. 2024. Improve dense passage retrieval with entailment tuning. Preprint, arXiv:2410.15801.

Misha Denil, Alban Demiraj, and Nando de Freitas. 2015. Extraction of salient sentences from labelled documents. Preprint, arXiv:1412.6815.

Thomas Diggelmann, Jordan Boyd-Graber, Jannis Bulian, Massimiliano Ciaramita, and Markus Leippold. 2021. Climate-fever: A dataset for verification of real-world climate claims. Preprint, arXiv:2012.00614.

Emily Dinan, Stephen Roller, Kurt Shuster, Angela Fan, Michael Auli, and Jason Weston. 2019. Wizard of Wikipedia: Knowledge-powered conversational agents. In Proceedings ofthe International Conference on Learning Representations (ICLR).

Hady Elsahar, Pavlos Vougiouklis, Arslen Remaci, Christophe Gravier, Jonathon Hare, Frederique Laforest, and Elena Simperl. 2018. T-REx: A large scale alignment of natural language with knowledge base triples. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), Miyazaki, Japan. European Language Resources Association (ELRA).

Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. 2019. ELI5: Long form question answering. In Proceedings of

the 57th Annual Meeting of the Association for Computational Linguistics, pages 3558–3567, Florence, Italy. Association for Computational Linguistics.

Feiteng Fang, Yuelin Bai, Shiwen Ni, Min Yang, Xiaojun Chen, and Ruifeng Xu. 2024. Enhancing noise robustness of retrieval-augmented language models with adaptive adversarial training. Preprint, arXiv:2405.20978.

Yunfan Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-augmented generation for large language models: A survey. Preprint, arXiv:2312.10997.

Michael Glass, Gaetano Rossiello, Md Faisal Mahbub Chowdhury, Ankita Naik, Pengshan Cai, and Alfio Gliozzo. 2022. Re2G: Retrieve, rerank, generate. In Proceedings ofthe 2022 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2701–2715, Seattle, United States. Association for Computational Linguistics.

Jiafeng Guo, Yixing Fan, Xiang Ji, and Xueqi Cheng. 2019. Matchzoo: A learning, practicing, and developing system for neural text matching. In Proceedings of the 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR’19, page 1297–1300, New York, NY, USA. Association for Computing Machinery.

Xiaochuang Han and Yulia Tsvetkov. 2022. Orca: Interpreting prompted language models via locating supporting data evidence in the ocean of pretraining data. Preprint, arXiv:2205.12600.

Donald E. Hilt and Donald W. Seegrist. 1977. Ridge: a computer program for calculating ridge regression estimates.

Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. 2024. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Transactions on Information Systems.

Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc Brockschmidt. 2020. Codesearchnet challenge: Evaluating the state of semantic code search. Preprint, arXiv:1909.09436.

Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2022. Unsupervised dense information retrieval with contrastive learning. Preprint, arXiv:2112.09118.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for opendomain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural

Language Processing (EMNLP), pages 6769–6781, Online. Association for Computational Linguistics.

Minsang Kim and Seungjun Baek. 2025. Syntriever: How to train your retriever with synthetic data from llms. Preprint, arXiv:2502.03824.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:452–466.

Omer Levy, Minjoon Seo, Eunsol Choi, and Luke Zettlemoyer. 2017. Zero-shot relation extraction via reading comprehension. In Proceedings of the 21st Conference on Computational Natural Language Learning (CoNLL 2017), pages 333–342, Vancouver, Canada. Association for Computational Linguistics.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledgeintensive nlp tasks. In Advances in Neural Information Processing Systems, volume 33, pages 9459– 9474. Curran Associates, Inc.

Dongfang Li, Zetian Sun, Xinshuo Hu, Zhenyu Liu, Ziyang Chen, Baotian Hu, Aiguo Wu, and Min Zhang. 2023. A survey of large language models attribution. Preprint, arXiv:2311.03731.

Xinze Li, Yixin Cao, Liangming Pan, Yubo Ma, and Aixin Sun. 2024. Towards verifiable generation: A benchmark for knowledge-aware language model attribution. Preprint, arXiv:2310.05634.

Xi Victoria Lin, Xilun Chen, Mingda Chen, Weijia Shi, Maria Lomeli, Rich James, Pedro Rodriguez, Jacob Kahn, Gergely Szilvasy, Mike Lewis, Luke Zettlemoyer, and Scott Yih. 2024. Radit: Retrieval-augmented dual instruction tuning. Preprint, arXiv:2310.01352.

Xueqing Liu, Chi Wang, Yue Leng, and ChengXiang Zhai. 2018. Linkso: a dataset for learning to retrieve similar question answer pairs on software development forums. In Proceedings of the 4th ACM SIG-SOFT International Workshop on NLP for Software Engineering, NL4SE 2018, page 2–5, New York, NY, USA. Association for Computing Machinery.

Yuhang Liu, Xueyu Hu, Shengyu Zhang, Jingyuan Chen, Fan Wu, and Fei Wu. 2024. Fine-grained guidance for retrievers: Leveraging llms’ feedback in retrieval-augmented generation. Preprint, arXiv:2411.03957.

Lin Long, Rui Wang, Ruixuan Xiao, Junbo Zhao, Xiao Ding, Gang Chen, and Haobo Wang. 2024. On llmsdriven synthetic data generation, curation, and evaluation: A survey. Preprint, arXiv:2406.15126.

Gianluigi Lopardo, Frederic Precioso, and Damien Garreau. 2024. Attention meets post-hoc interpretability: A mathematical perspective. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 32781–32800. PMLR.

Andreas Madsen, Sarath Chandar, and Siva Reddy. 2024. Are self-explanations from large language models faithful? In Findings of the Association for Computational Linguistics: ACL 2024, pages 295–337, Bangkok, Thailand. Association for Computational Linguistics.

Macedo Maia, Siegfried Handschuh, André Freitas, Brian Davis, Ross McDermott, Manel Zarrouk, and Alexandra Balahur. 2018. Www’18 open challenge: Financial opinion mining and question answering. In Companion Proceedings ofthe The Web Conference 2018, WWW ’18, page 1941–1942, Republic and Canton of Geneva, CHE. International World Wide Web Conferences Steering Committee.

Dina Mardaoui and Damien Garreau. 2021. An analysis of lime for text data. In Proceedings of The 24th International Conference on Artificial Intelligence and Statistics, volume 130 of Proceedings of Machine Learning Research, pages 3493–3501. PMLR.

Vaibhav Mavi, Anubhav Jangra, and Adam Jatowt. 2024. Multi-hop question answering. Preprint, arXiv:2204.09140.

Nikolaos Mylonas, Ioannis Mollas, and Grigorios Tsoumakas. 2022. An attention matrix for every decision: Faithfulness-based arbitration among multiple attention-based interpretations of transformers in text classification. Preprint, arXiv:2209.10876.

Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad Saqib, Saeed Anwar, Muhammad Usman, Naveed Akhtar, Nick Barnes, and Ajmal Mian. 2024. A comprehensive overview of large language models. Preprint, arXiv:2307.06435.

Ian E. Nielsen, Dimah Dera, Ghulam Rasool, Ravi P. Ramachandran, and Nidhal Carla Bouaynaya. 2022. Robust explainability: A tutorial on gradient-based attribution methods for deep neural networks. IEEE Signal Processing Magazine, 39(4):73–84.

OpenAI. 2024. Gpt-4o system card.

Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard, Vassilis Plachouras, Tim Rocktäschel, and Sebastian Riedel. 2021. KILT: a benchmark for knowledge intensive language tasks. In Proceedings ofthe 2021 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human

Language Technologies, pages 2523–2544, Online. Association for Computational Linguistics.

Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. 2016. Model-agnostic interpretability of machine learning. Preprint, arXiv:1606.05386.

Stephen Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: Bm25 and beyond. Foundations and Trends® in Information Retrieval, 3(4):333–389.

Alireza Salemi and Hamed Zamani. 2024. Towards a search engine for machines: Unified ranking for multiple retrieval-augmented large language models. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’24, page 741–751, New York, NY, USA. Association for Computing Machinery.

Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu Chen. 2023. Enhancing retrieval-augmented large language models with iterative retrieval-generation synergy. Preprint, arXiv:2305.15294.

Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen tau Yih. 2023. Replug: Retrievalaugmented black-box language models. Preprint, arXiv:2301.12652.

Kurt Shuster, Spencer Poff, Moya Chen, Douwe Kiela, and Jason Weston. 2021. Retrieval augmentation reduces hallucination in conversation. In Findings of the Association for Computational Linguistics: EMNLP 2021, pages 3784–3803, Punta Cana, Dominican Republic. Association for Computational Linguistics.

Jiwoong Sohn, Yein Park, Chanwoong Yoon, Sihyeon Park, Hyeon Hwang, Mujeen Sung, Hyunjae Kim, and Jaewoo Kang. 2024. Rationale-guided retrieval augmented generation for medical question answering. Preprint, arXiv:2411.00300.

Hongjin Su, Howard Yen, Mengzhou Xia, Weijia Shi, Niklas Muennighoff, Han yu Wang, Haisu Liu, Quan Shi, Zachary S. Siegel, Michael Tang, Ruoxi Sun, Jinsung Yoon, Sercan O. Arik, Danqi Chen, and Tao Yu. 2024. Bright: A realistic and challenging benchmark for reasoning-intensive retrieval. Preprint, arXiv:2407.12883.

Qwen Team. 2024. Qwen2.5: A party of foundation models.

Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. Beir: A heterogenous benchmark for zero-shot evaluation of information retrieval models. Preprint, arXiv:2104.08663.

James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. 2018. FEVER: a large-scale dataset for fact extraction and VERification. In NAACL-HLT.

David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi. 2020. Fact or fiction: Verifying scientific claims. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 7534–7550, Online. Association for Computational Linguistics.

Yongjie Wang, Tong Zhang, Xu Guo, and Zhiqi Shen. 2024. Gradient based feature attribution in explainable ai: A technical review. Preprint, arXiv:2403.10415.

Zhiguo Wang, Patrick Ng, Xiaofei Ma, Ramesh Nallapati, and Bing Xiang. 2019. Multi-passage BERT: A globally normalized BERT model for open-domain question answering. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5878–5882, Hong Kong, China. Association for Computational Linguistics.

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. 2022. Emergent abilities of large language models. Preprint, arXiv:2206.07682.

Zhepei Wei, Wei-Lin Chen, and Yu Meng. 2024. InstructRAG: Instructing retrieval augmented generation via self-synthesized rationales. In Adaptive Foundation Models: Evolving AI for Personalized and Efficient Learning.

Orion Weller, Marc Marone, Nathaniel Weir, Dawn Lawrie, Daniel Khashabi, and Benjamin Van Durme. 2024. “according to . . . ”: Prompting language models improves quoting from pre-training data. In Proceedings ofthe 18th Conference ofthe European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2288–2301, St. Julian’s, Malta. Association for Computational Linguistics.

Siye Wu, Jian Xie, Jiangjie Chen, Tinghui Zhu, Kai Zhang, and Yanghua Xiao. 2024. How easily do irrelevant inputs skew the responses of large language models? Preprint, arXiv:2404.03302.

Long Xia, Jun Xu, Yanyan Lan, Jiafeng Guo, and Xueqi Cheng. 2015. Learning maximal marginal relevance model via directly optimizing diversity evaluation measures. In Proceedings of the 38th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’15, page 113–122, New York, NY, USA. Association for Computing Machinery.

Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. 2023. C-pack: Packaged resources to advance general chinese embedding. Preprint, arXiv:2309.07597.

Lee Xiong, Chenyan Xiong, Ye Li, Kwok-Fung Tang, Jialin Liu, Paul Bennett, Junaid Ahmed, and Arnold Overwijk. 2020. Approximate nearest neighbor negative contrastive learning for dense text retrieval. Preprint, arXiv:2007.00808.

Yilong Xu, Jinhua Gao, Xiaoming Yu, Baolong Bi, Huawei Shen, and Xueqi Cheng. 2024. Aliice: Evaluating positional fine-grained citation generation. Preprint, arXiv:2406.13375.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2369–2380, Brussels, Belgium. Association for Computational Linguistics.

Yue Yu, Wei Ping, Zihan Liu, Boxin Wang, Jiaxuan You, Chao Zhang, Mohammad Shoeybi, and Bryan Catanzaro. 2024. Rankrag: Unifying context ranking with retrieval-augmented generation in llms. Preprint, arXiv:2407.02485.

Zichun Yu, Chenyan Xiong, Shi Yu, and Zhiyuan Liu. 2023. Augmentation-adapted retriever improves generalization of language models as generic plug-in. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 2421–2436, Toronto, Canada. Association for Computational Linguistics.

Hamed Zamani and Michael Bendersky. 2024. Stochastic rag: End-to-end retrieval-augmented generation through expected utility maximization. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’24, page 2641–2646, New York, NY, USA. Association for Computing Machinery.

Hengran Zhang, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Yixing Fan, and Xueqi Cheng. 2024. Are large language models good at utility judgments? In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’24, page 1941–1951, New York, NY, USA. Association for Computing Machinery.

Wayne Xin Zhao, Kun Zhou, Junyi Li, Tianyi Tang, Xiaolei Wang, Yupeng Hou, Yingqian Min, Beichen Zhang, Junjie Zhang, Zican Dong, Yifan Du, Chen Yang, Yushuo Chen, Zhipeng Chen, Jinhao Jiang, Ruiyang Ren, Yifan Li, Xinyu Tang, Zikang Liu, Peiyu Liu, Jian-Yun Nie, and Ji-Rong Wen. 2024. A survey of large language models. Preprint, arXiv:2303.18223.

## A Details of Data Synthesis

The detailed steps of the data synthesis pipeline in SCARLet are described as follows:

Seed Datasets Collection We first collect seed data for the data synthesis pipeline. A task pool is defined, including the selected tasks and their corresponding datasets. Each task is associated with task instruction and retrieval instruction, as shown in Table 6. For each dataset, we randomly sample 1,000 instances from its training split, including both input and ground truth. Every sampling uses the same random seed for every dataset.

Entities Extraction For each seed data instance, we extract entities for subsequent passages retrieval. We utilize the SpaCy<sup>3</sup> toolkit to extract entities from both the input and ground truth. Data instances without extractable entities are discarded.

Entities Retrieval This stage is to retrieve more relevant entities based on the extracted ones. This serves two purposes: 1) to enhance diversity, and 2) to strengthen relationships between entities, facilitating better construction of the shared context. We retrieve neighboring entities from Wikidata, considering only the direct related entities of each existing entity. To achieve this, we write the SPARQL query for retrieval, as shown below:

```awk
SELECT ? property ? propertyLabel ? object
? objectLabel
WHERE {
wd :{ id} ? property ? object .
? property rdfs : label ? propertyLabel .
? object rdfs : label ? objectLabel .
FILTER ( LANG (? propertyLabel ) = "en")
FILTER ( LANG (? objectLabel ) = "en")
}
LIMIT { limit }
```

Passages Retrieval After obtaining the expanded entity list, we retrieve relevant passages based on these entities to construct the shared context.

Data Synthesis At this stage, training data is synthesized for different tasks in the task pool based on the shared context. First, a synthesizer model is selected, which must possess sufficient reasoning and generation capabilities to ensure the quality of the synthetic data. To help the synthesizer understand the task definition and follow the correct format, we provide task instruction, task description, and example data in the prompt. The synthetic

data should include both the input and ground truth.   
The prompt template we use is shown in Table 9.

Data Filtering In this stage, the data synthesized in the previous phase is cleaned to further ensure data quality and training stability. We prompt the synthesizer model to check the synthetic data for logical consistency and format correctness based on the shared context. The prompt used for this stage is shown in Table 10.

Passages Enhancement To enhance the robustness of the training, we inject noise into the shared context. We instruct the synthesizer model to generate a passage that is semantically relevant but useless for downstream task, and then add this passage to the shared context. The prompt used for this stage is shown in Table 11.

## B Details of Utility Attribution

Introduction to Attribution Attribution is a local-interpretable technique used to provide evidence for the model generation (Li et al., 2023; Xu et al., 2024). The data source of attribution can be training data (Han and Tsvetkov, 2022; Weller et al., 2024), whereas in RALMs, the source is often retrieved external passages (Shuster et al., 2021; Li et al., 2024), which we denote as context attribution. Furthermore, contributive attribution is a form of attribution that quantifies the contribution of each data source unit to the generation process. It assigns an attribution score to each unit, where a higher score indicates a greater contribution. In this study, we propose the SCARLet framework, which employs a perturbation-based attribution method to estimate the utility score of each passage within the shared context. Additionally, we evaluate other attribution methods, including attention-based method, gradient-based method, and LLM-based method.

Perturbation-based Method This method is described in Section 3.4. Notably, unlike the classical LIME method, we remove the weight of $\mathbf { v } _ { i }$ in the surrogate model, which measures the the cosine distance from the original text. The reason behind this is that for different perturbation vectors, the weight would exacerbate the unfair evaluation of passage utility, as utility features cannot be directly measured by semantic relevance. For passages that are semantically relevant but essentially useless, the variation they bring would be downweighted, as such passages typically cause greater logit fluctuations due to their lack of utility.

<table><tr><td>Dataset</td><td>Task</td><td>Task Instruction</td><td>Retrieval Instruction</td></tr><tr><td>NQ</td><td>Single-hop QA</td><td>Answer the question based on the given passages.</td><td>Retrieve passages to answer the question.</td></tr><tr><td>HotpotQA</td><td>Multi-hop QA</td><td>Answer the question based on the given passages. You may need to refer to multiple passages.</td><td>Find passages that provide useful information to answer this question.</td></tr><tr><td>ELI5</td><td>Long-form QA</td><td>Answer the question based on the given passages. The answer needs to be detailed, paragraph-level, and with explanations.</td><td>Retrieve passages that provide a piece of good evidence for the answer.</td></tr><tr><td>FEVER</td><td>Fact Checking</td><td>Verify whether the claim is correct based on the given passages. If it is correct, output &quot;SUPPORTS&quot;, if it is wrong, output &quot;REFUTES&quot;.</td><td>Retrieve passages to verify this claim.</td></tr><tr><td>WoW</td><td>Dialogue Generation</td><td>Generate an appropriate, reasonable and meaningful response based on previous conversations and the following relevant passages.</td><td>Find passages related to the conversation topic.</td></tr><tr><td>T-REx</td><td>Slot Filling</td><td>Given an entity and an attribute (or relationship), fill in the specific value of the attribute based on the following passages. The entity and the attribute are separated by &quot;[SEP]&quot;.</td><td>Find passages related to the entities.</td></tr><tr><td>SciFact</td><td>Fact Checking</td><td>Verify whether the claim is correct based on the given passages. If it is correct, output &quot;SUPPORT&quot;, if it is wrong, output &quot;CONTRADICT&quot;.</td><td>Retrieve passages to verify this claim.</td></tr><tr><td>zs-RE</td><td>Relation Extraction</td><td>Given an entity and an attribute (or relationship), fill in the specific value of the attribute based on the following passages. The entity and the attribute are separated by &quot;[SEP]&quot;.</td><td>Find passages related to the entities.</td></tr><tr><td>FiQA</td><td>Financial QA</td><td>Answer the question based on the given passages.</td><td>Find passages to answer the question.</td></tr><tr><td>Climate- FEVER</td><td>Fact Checking</td><td>Verify whether the claim is correct based on the given passages. If it is correct, output &quot;SUPPORTS&quot;, if it is wrong, output &quot;REFUTES&quot;, if the information is insufficient, output &quot;NOT_ENOUGH_INFO&quot;, if can&#x27;t get a sufficiently confident judgment, output &quot;DISPUTED&quot;.</td><td>Retrieve passages to verify this claim.</td></tr></table>

Table 6: Task instructions and retrieval instructions of the datasets in the task pool.

Please first provide the answer based on the   
passages that you have ranked in utility and then   
write the ranked passages in descending order of   
utility in answering the question, like "My rank:   
[i]>[j]>...>[k]".   
Context: {context}   
Question: {query}  
Table 7: The prompt template for LLM-based method.

Attention-based Method This method takes the attention score received by each source unit during inference as the attribution score (Mylonas et al., 2022; Lopardo et al., 2024). We construct the attention-based baseline by averaging the attention values of each token within each passage, as shown below:

$$
\pmb { \alpha } _ { d _ { i } } = \frac { 1 } { K \cdot | d _ { i } | } \sum _ { t \in d _ { i } } \sum _ { i = 1 } ^ { K } a _ { t } ^ { ( i ) } , t \in d _ { i } ,\tag{7}
$$

where $\alpha _ { d _ { i } }$ represents the utility score for passage $d _ { i } , K$ indicates the number of attention heads, and $a _ { t } ^ { \left( i \right) }$ indicates the attention value of the t-th token in passage $d _ { i }$ of the i-th attention head.

Gradient-based Method This approach determines the utility scores from the gradient of each token in the source unit during backward propagation(Nielsen et al., 2022; Wang et al., 2024). Specifically, we employ the Gradient times Input $( G \times I ;$ Denil et al., 2015), which computes the score of each token by performing the dot product as follows:

$$
\begin{array} { r } { f _ { G \times I } ( t ) = e _ { t } \cdot \nabla _ { e _ { t } } f _ { \mathrm { L M } } \left( x , D \right) , } \end{array}\tag{8}
$$

where $e _ { t }$ represents the embedding vector of token $t ,$ and $f _ { \mathrm { L M } }$ denotes the function of LM. The utility score of each passage is then obtained by averaging the $G \times I$ scores of each token contained within it.

LLM-based Method This approach, which can also be referred to as rationale-based method or self-rationalization, is in line with the work of Sohn et al. (2024); Wei et al. (2024), where the LLM generator simultaneously attributes the utility of passages in the context while performing the task. Although this method is theoretically flawed due to the potential influence of hallucinations from

<table><tr><td rowspan="2">Method</td><td colspan="2">HotpotQA</td><td colspan="2">NQ</td><td colspan="2">MSMARCO-QA</td></tr><tr><td>NDCG@1</td><td>NDCG@5</td><td>NDCG@1</td><td>NDCG@5</td><td>NDCG@1</td><td>NDCG@5</td></tr><tr><td>Att.-based</td><td>31.54</td><td>27.25</td><td>29.14</td><td>25.77</td><td>29.92</td><td>22.15</td></tr><tr><td>Grad.-based</td><td>49.90</td><td>38.83</td><td>50.58</td><td>44.56</td><td>59.09</td><td>53.35</td></tr><tr><td>LLM-based</td><td>76.34</td><td>76.84</td><td>28.35</td><td>32.16</td><td>31.88</td><td>59.97</td></tr><tr><td>Pert.-based</td><td>93.28</td><td>83.04</td><td>78.16</td><td>84.12</td><td>91.65</td><td>85.36</td></tr><tr><td>w/o G.T.</td><td>92.34</td><td>81.03</td><td>77.85</td><td>80.67</td><td>91.10</td><td>83.73</td></tr></table>

Table 8: The experimental results comparing various utility attribution methods on the GTI benchmark. Attn., Grad., Pert., and G.T. represent Attention, Gradient, Perturbation and Ground Truth, respectively.

LLMs (Chen and Shu, 2024), we still believe that it represents one of the future directions of utility attribution. Following Zhang et al. (2024), we instruct the generator to rank the passages from the context in a list-wise setup while generating the answer. The prompt is shown at Table 7.

GTI Benchmark This benchmark (Ground-Truth Inclusion; Zhang et al., 2024) is designed to assess the utility of retrieved passages including three QA datasets: NQ, with 1,868 data; HotpotQA, with 4,407 data; and MSMARCO-QA, with 3,121 data. It manually constructs 10 passages per query, including ground truth (correct passages), counterfactual passages, highly relevant noisy passages, and weakly relevant noisy passages. We evaluate the above methods on this benchmark using LLaMA-3-8B-Instruct as the generator, with the experimental results presented in Table 8. The results demonstrate that the perturbation-based method outperforms all other baselines by a significant margin, highlighting its considerable advantage as an indicator for utility in RALMs.

Attribution Forms Additionally, we investigate two different attribution forms: 1) The first form directly uses the ground truth provided by the dataset as the output of the generator, which is adopted in our proposed SCARLet; 2) The second form is let the generator to produce a response first, followed by attribution based on that response. The first form reflects the contribution of each passage within the context to the production of the correct answer. While the second form requires an additional comparison between the generated response and the ground truth, where we believe that the attribution process can be valid only if the the two are consistent. We compare the performance of the above two forms in the perturbation-based method, as shown in Table 8. We find that the performance difference between the two forms is minimal, but in terms of mechanism and difficulty of implementation, we choose the first form.

![](images/9ebbc16faa8b176bc41067d2e2c19bc85baba2f63794982b42bd425753630abb.jpg)  
Figure 6: Case Study on WoW. Blue text indicates clues more relevant to semantics, while orange text highlights clues more align with the target utility in dialogue generation task. Responses are generated by LLaMA-3-8B. The generated response augmented by SCARLet<sub>BGE</sub> achieves a higher F1 score than the response augmented by BGE.

## C Details of Implementation

Meta data of data synthesis We present the meta data from the data synthesis pipeline of one run in our experiment, as shown in Table 12. As observed, although the amount of training data is sufficient for tuning the retriever, the SCARLet pipeline leads to data loss at each stage, sometimes resulting in significant loss rates, which causes an increase in costs. The reasons for the loss include issues with the seed data, network problems, model generation errors, among others.

Hyperparameters During the data synthesis stage, the temperature of the synthesizer model is set to 0.5. In the utility attribution stage, the number of sampled perturbation vectors n is set to 64, with a perturbation probability of 0.5. During training, we set the learning rate as 6e-5, and epochs as 1. All experiments are conducted on NVIDIA A100 GPUs in torch.float32 precision.

## D Additional Experimental Results

The results presented in Table 2 are under the closed corpus setup, i.e., retrievers search passages only from the corpus of the corresponding dataset. In contrast, the pooled corpus setup refers to merging the corpora of different datasets into a single corpus, where all retrieval is performed with the unified corpus. This setup better simulates realworld retrieval scenarios and enables a fairer evaluation of generalization. The experimental results under the pooled corpus setup are shown in Table 13. All baselines perform similarly to those in the closed corpus setup, and some outperform them, demonstrating generalization of SCARLet on the unified corpus.

## E Additional Case Study

The QA tasks typically focus more on precise answers, whereas dialogue tasks prioritize the coherence between the generated response and the preceding conversation. These two tasks have distinct retrieval utility, with the latter being more vaguely defined. To analyze whether the retriever trained by SCARLet exhibits a diversified retrieval criteria, we select a case from the test split of the WoW dataset, as shown in Figure 6. In this case, a retriever relying on semantic relevance may primarily focus on topic words such as "red" and "spectrum". However, for dialogue generation, it is also crucial to consider the intent of the previous speaker. Passage 2 is ranked higher by the retriever trained by SCARLet, because it is directly tied to the deeper meaning of the key clue "bold", making it more helpful in sustaining conversational coherence. At comparable recall levels, SCARLet prioritizes passages that offer greater task-specific utility.

## F Example of Shared Context

In this section, we provide an example of the shared context constructed during one run of SCARLet in our experiment, as shown in Figure 7, along with its corresponding synthetic data for various tasks, as shown in Figure 8 and Figure 9.

![](images/26cef39713474fd6585cd1b4ade95ca6a94ff53f35742b6d7f6895f58719807a.jpg)  
Figure 7: An example of the shared context. Based on this context, SCARLet synthesizes training data, as shown in Figure 8 and 9.

![](images/b576db15fd2550cce70aadceb12eded7d3c550562657a77bab7df5c805ef46fa.jpg)  
Figure 8: The training data of long-form QA, synthesized by SCARLet based on the context in Figure 7.

![](images/fe21effa61f74c9e89a2611ffa1b085294f9a5a8738d4662b0a385c4a6f4dd51.jpg)

![](images/1fb04ff1a197d4ab7d7f6ea8ad0042d33cb0f034f7acbf5516fcbd9ea3a41d53.jpg)

![](images/e3f3913f721976df5dd12051c190683ae1d3a5a1d4b19076f30baebfa2c122c7.jpg)  
Figure 9: The training data of multi-hop QA, fact checking and slot filling, synthesized by SCARLet based on the context in Figure 7.

![](images/d1992b86e73960ae077826f967ceb10ab868694dd2439859317f22fac15d9a91.jpg)  
Table 9: The prompt template for data synthesis.

![](images/dbe37c0bc6f594ec635283080dce680c1087a2f8346f7f4bae09c6e2d2fbbad6.jpg)  
Table 10: The prompt template for data filtering.

![](images/e748ac44e00c478fe9778f63862abf28dd959ace44c8cdaa91ef7fa9caa1c71c.jpg)  
Table 11: The prompt template for passages enhancement.

<table><tr><td></td><td>NQ</td><td>HotpotQA</td><td>ELI5</td><td>FEVER</td><td>WoW</td><td>T-REx</td></tr><tr><td colspan="7">Entities Extraction</td></tr><tr><td>Loss Rate</td><td>12.2%</td><td>0.6%</td><td>24.1%</td><td>5.1%</td><td>17.1%</td><td>9.2%</td></tr><tr><td>Averaged Number of Entities</td><td>1.7</td><td>3.4</td><td>5.8</td><td>2.0</td><td>5.1</td><td>1.8</td></tr><tr><td colspan="7">Entities Retrieval</td></tr><tr><td>Expansion Rate</td><td>91.0%</td><td>90.5%</td><td>96.0%</td><td>89.1%</td><td>97.5%</td><td>98.0%</td></tr><tr><td>Averaged Number of New Entities</td><td>5.1</td><td>6.5</td><td>17.1</td><td>3.7</td><td>14.9</td><td>5.2</td></tr><tr><td>Averaged Number of Entities</td><td>6.3</td><td>9.3</td><td>22.2</td><td>5.3</td><td>19.6</td><td>6.9</td></tr><tr><td colspan="7">Data Synthesis</td></tr><tr><td>Number of Synthetic Data</td><td>5230</td><td>5950</td><td>4492</td><td>5580</td><td>4872</td><td>5317</td></tr><tr><td>Loss Rate</td><td>12.8%</td><td>0.8%</td><td>25.1%</td><td>7.0%</td><td>18.8%</td><td>11.4%</td></tr></table>

Table 12: Meta data from the synthesis pipeline of one run in our experiment. Loss Rate means the proportion of discarded data caused by the process. Expansion Rate means the proportion of data with new entities added. In this run, the data filtering achieves a loss rate of 44.2%, and the total amount of data used for utility attribution is 17,529.

<table><tr><td rowspan="2">Method</td><td colspan="6">In-domain</td><td colspan="4">Out-of-domain</td></tr><tr><td></td><td>HotpotQA</td><td>ELI5</td><td>FEVER</td><td>WoW</td><td>T-REx</td><td>ZS-RE</td><td>SciFact</td><td>C-FEVER</td><td>FiQA</td></tr><tr><td colspan="10">LLaMA-3-8B-Instruct</td></tr><tr><td>Contriever</td><td>44.0</td><td>36.7</td><td>14.5</td><td>79.2</td><td>8.6</td><td>33.8</td><td>20.9</td><td>68.1</td><td>38.0</td><td>16.5</td></tr><tr><td>BGE</td><td>48.0</td><td>45.4</td><td>15.2</td><td>85.6</td><td>8.8</td><td>39.6</td><td>24.1</td><td>80.2</td><td>45.9</td><td>20.8</td></tr><tr><td>AARContriever</td><td>46.2</td><td>41.8</td><td>15.0</td><td>77.8</td><td>8.2</td><td>35.1</td><td>24.2</td><td>70.3</td><td>42.6</td><td>16.7</td></tr><tr><td>REPLUGContriever</td><td>44.5</td><td>39.7</td><td>13.8</td><td>81.3</td><td>9.2</td><td>33.7</td><td>23.6</td><td>72.9</td><td>41.0</td><td>18.8</td></tr><tr><td>SCARLetContriever SCARLetBGE</td><td>45.1</td><td>42.0</td><td>15.9</td><td>80.6</td><td>10.4</td><td>36.4</td><td>22.2</td><td>74.7</td><td>42.0</td><td>17.7</td></tr><tr><td>49.8</td><td></td><td>48.3</td><td>16.6</td><td>81.2</td><td>12.7</td><td>37.0</td><td>24.7</td><td>81.5</td><td>45.9</td><td>23.1</td></tr><tr><td colspan="9">Qwen2.5-3B-Instruct</td></tr><tr><td>Contriever</td><td>31.9</td><td>28.5</td><td>14.2</td><td>67.1</td><td>10.5</td><td>27.1</td><td>14.0</td><td>66.5</td><td>32.8</td><td>15.5</td></tr><tr><td>BGE</td><td>48.5</td><td>44.0</td><td>13.7</td><td>80.4</td><td>10.2</td><td>34.5</td><td>18.6</td><td>65.5</td><td>37.1</td><td>18.6</td></tr><tr><td>AARContriever</td><td>34.8</td><td>30.9</td><td>13.8</td><td>66.2</td><td>10.6</td><td>28.3</td><td>15.5</td><td>63.2</td><td>32.0</td><td>16.3</td></tr><tr><td>REPLUGContriever</td><td>34.2</td><td>35.8</td><td>14.0</td><td>71.2</td><td>12.8</td><td>26.8</td><td>16.9</td><td>60.6</td><td>30.9</td><td>18.7</td></tr><tr><td>SCARLetContriever</td><td>39.3</td><td>36.0</td><td>14.4</td><td>70.0</td><td>11.9</td><td>28.2</td><td>19.1</td><td>64.9</td><td>31.8</td><td>17.3</td></tr><tr><td>SCARLetBGE</td><td>45.1</td><td>44.7</td><td>15.6</td><td>74.1</td><td>12.3</td><td>30.1</td><td>18.7</td><td>64.4</td><td>36.3</td><td>20.5</td></tr></table>

Table 13: Results of the main experiment in the pooled corpus setup. The unified corpus includes corpora of Wikipedia dump, BeIR-SciFact, BeIR-ClimateFEVER and BeIR-FiQA.