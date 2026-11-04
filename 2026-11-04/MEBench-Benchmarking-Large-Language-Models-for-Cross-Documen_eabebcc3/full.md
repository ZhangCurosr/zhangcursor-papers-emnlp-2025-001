# MEBench: Benchmarking Large Language Models for Cross-Document Multi-Entity Question Answering

Teng Lin<sup>1</sup>, Yuyu Luo<sup>1,2</sup>, Honglin Zhang<sup>3</sup>, Jicheng Zhang<sup>3</sup>, Chunlin Liu<sup>3</sup>, Kaishun Wu<sup>1</sup>, Nan Tang<sup>1,2</sup>\*

<sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou) <sup>2</sup>The Hong Kong University of Science and Technology <sup>3</sup>China Mobile Information Technology Company Limited tlin280@connect.hkust-gz.edu.cn {yuyuluo,wuks,nantang}@hkust-gz.edu.cn {zhanghonglin, zhangjichengit, liuchunlin}@chinamobile.com

## Abstract

Cross-Document Multi-entity Question Answering (MEQA) demands the integration of scattered information across documents to resolve complex queries involving entities, relationships, and contextual dependencies. Al though Large Language Models (LLMs) and Retrieval-augmented Generation (RAG) systems show promise, their performance on crossdocument MEQA remains underexplored due to the absence of tailored benchmarks. To ad dress this gap, we introduce MEBench, a scal able multi-document, multi-entity benchmark designed to systematically evaluate LLMs’ capacity to retrieve, consolidate, and reason over scattered and dense information. Our benchmark comprises 4,780 questions which are systematically categorized into three primary categories: Comparative Reasoning, Statistical Reasoning and Relational Reasoning, further divided into eight distinct types, ensuring broad coverage of real-world multi-entity reasoning scenarios. Our experiments on state-of-the art LLMs reveal critical limitations: even ad vanced models achieve only 59% accuracy on MEBench. Our benchmark emphasizes the importance of completeness and factual precision of information extraction in MEQA tasks, using Entity-Attributed F1 (EA-F1) metric for granular evaluation of entity-level correctness and attribution validity. MEBench not only highlights systemic weaknesses in current LLM frameworks but also provides a foundation for advancing robust, entity-aware QA architectures.<sup>1</sup>

## 1 Introduction

The emergence of large language models (LLMs) has significantly advanced natural language processing capabilities, demonstrating exceptional performance in diverse tasks spanning text generation to data science and databases (Achiam et al.,

2023; Lin, 2025a; Liu et al., 2025a; Li et al., 2025a; Zhang et al., 2025a; Li et al., 2024a; Chen et al., 2025; Li et al., 2025b; Fan et al., 2024a). Nevertheless, long-context LLMs exhibit notable limitations in processing entity-dense analytical reasoning, particularly when contextual dependencies are distributed across multiple documents (Wu et al., 2025c,a; Shi et al., 2025), and we analytically argue that context window limitations, over-reliance on parametric knowledge, and poor cross-document attention as the key bottlenecks (Tang et al., 2024b; Zhu et al., 2024; Xiang et al., 2025; Lin et al., 2025b). On the other hand, current implementations of retrieval-augmented generation (RAG) architectures (Yang et al., 2024; Wu et al., 2025b; Fan et al., 2024b; Tang et al., 2024a; Liu et al., 2025b; Zhang et al., 2024b; Lin et al., 2025a; Lin, 2025b; Hong et al., 2025) frameworks’ effectiveness in addressing cross-document Multi-entity Question Answering (MEQA) remains insufficiently investi gated. Furthermore, the field lacks comprehensive benchmarking frameworks specifically designed to evaluate the performance of LLMs and RAG systems for cross-document entity-intensive tasks. As shown in Figure 1, existing evaluation metrics frequently inadequately represent the complexities inherent in real-world MEQA applications (Song et al., 2024), where queries such as “What is the number distribution of all Turing Award winners by fields of study by 2023?” necessitate not only highprecision information retrieval but also reasoning over fragmented, entity-specific information across heterogeneous document sources.

To address this methodological gap, we present MEbench, a novel benchmarking framework specifically designed to assess the performance of large language models and RAG systems in crossdocument multi-entity question answering scenarios. The benchmark simulates real-world information integration challenges where correct answers require synthesizing entity-centric evidence distributed across multiple documents, with a single instance of document omission or entity misinterpretation can propagate errors through the reasoning chain. As shown in Table 2, MEBench features a mean entity density of 409 entities per query, with systematically varied entity cardinality across three operational tiers: low (0-10 entities), medium (10-100 entities), and high complexity (>100 entities). This stratified design enables granular performance evaluation across different entity scales and task difficulty levels. The framework comprises 4,780 validated question-answer pairs systematically categorized into three primary categories and eight distinct types, MEBench spans diverse real-world scenarios, from academic field distributions to geopolitical event analysis. Our experiments with state-of-the-art models, including GPT-4 and Llama-3, reveal significant shortcomings: even the most advanced LLMs achieve only 59% accuracy on MEBench. This underscores systemic weaknesses in current frameworks, for example, models frequently fail to locate all entity and their attributes or infer implicit relationships, highlighting the need for architectures that prioritize entity-aware retrieval and contextual consolidation.

Our contributions are summarized as follows:

Development of MEBench. A scalable benchmark designed to evaluate LLMs and RAG systems in cross-document aggregation and reasoning. It includes 4,780 validated question-answer pairs spanning three categories and eight types, simulating real-world scenarios that demand integration of scattered, entity-specific information.

Entity-centric Task Categories and Evaluation. Utilization of Entity-Attributed F1 (EA-F1), a granular metric for assessing entity-level correctness and attribution validity, alongside a stratified entity density design (low: 0–10, medium: 11–100, high: >100 entities per query). Our framework emphasizes completeness and factual precision in information extraction, addressing gaps in existing metrics for entity-dense MEQA tasks.

Scalable Benchmark Construction. A scalable, automated pipeline: Knowledge graph extraction from structured Wikipedia for cross-document relationship discovery; Relational table generation to preserve entity-property relationships; Templatebased QA generation ensuring reproducibility and reducing cost and labor.

![](images/92afb76618ceb8ac3efebd677073d55302b2104385ac778dacb2334a765ae9d7.jpg)  
Figure 1: Existing benchmarks vs. MEBench. Unlike existing benchmarks which feature centralized evidence distributions and sparse entity mentions, MEBench presents entity-dense scene where critical evidences are dispersed across multiple documents, necessitating that when seeking an answer, no document or entity can be ignored.

## 2 Related Work

Recent advancements in question answering (QA) have been driven by breakthroughs in LLMs and RAG systems. While these technologies excel in single or a few document settings, demonstrating proficiency in tasks like fact extraction, summarization, and reasoning within a single source, their performance in cross-document, multi-entity scenarios remains underexplored. This section contextualizes our work within three key research areas: single-document QA, cross-document aggregation, and entity-centric evaluation.

## 2.1 Single-Document QA and LLM Progress

Many QA benchmarks, such as SQuAD (Rajpurkar et al., 2016), Natural Questions (Kwiatkowski et al., 2019), L-eval (An et al., 2024) and needlein-a-haystack (Kamradt, 2023), focus on extracting answers from individual document. Modern LLMs like GPT-4 (Achiam et al., 2023), Llama-3 (Meta Llama3, 2024), and PaLM (Chowdhery et al., 2023) have achieved near-human performance on these tasks, leveraging their ability to parse and reason within localized contexts. However, these benchmarks do not address the complexities of integrating information across multiple documents, a critical limitation for real-world applications (Luo et al., 2018a,b, 2021, 2022; Qin et al., 2020; Zhu et al., 2025; Liu et al., 2025c,a).

## 2.2 Cross-Document Aggregation Challenges

Efforts to extend QA to multi-document settings include datasets like HotpotQA (Yang et al., 2018), MuSiQue (Trivedi et al., 2021), LooGLE (Li et al., 2024b), LM-Infinit (Han et al., 2024), Bench (Zhang et al., 2024a), CLongEval (Qiu et al., 2024), BAMBOO (Dong et al., 2024), Loong (Wang et al., 2024) and Symphony (Chen et al., 2023), which emphasize multi-hop reasoning and cross-source synthesis. While these benchmarks highlight the need for systems to connect disparate information, they often prioritize breadth over depth in entity-centric reasoning. For instance, questions in these datasets rarely demand the consolidation of attributes for dozens or more entities (e.g., aggregating ACM Fellows’ expertise across fields), a gap that limits their utility in evaluating entity-dense scenarios. Recent RAG frameworks (Fan et al., 2024b; Zhang et al., 2025b) aim to enhance retrieval-augmented QA but struggle with ensuring completeness and attribution validity when handling multi-entity queries.

## 2.3 Entity-Centric Evaluation Metrics.

Existing evaluation metrics for QA, such as F1 score and exact match (EM), focus on answer surface-form correctness but overlook granular entity-level attribution (Rostampour et al., 2010). Metrics in FEVER (Thorne et al., 2018), Attributed QA (Bohnet et al., 2023) and emphasize source verification, yet they lack the specificity to assess multi-entity integration. For example, they do not systematically measure whether all relevant entities are retrieved, their attributes are correctly extracted, or their sources are accurately used, which is a shortcoming that becomes critical in MEQA tasks.

## 2.4 The Gap in Multi-Entity QA Benchmarks.

Prior work has yet to establish a benchmark that systematically evaluates LLMs and RAG systems on entity-dense, cross-document reasoning. Current datasets either lack the scale and diversity of real-world multi-entity questions or fail to provide fine-grained metrics for assessing entity-level completeness and attribution (Song et al., 2024; Wang et al., 2024; Bai et al., 2025). MEBench addresses these limitations by introducing a comprehensive evaluation framework that challenges models to retrieve, consolidate, and reason over scattered entitycentric data across heterogeneous sources. By incorporating the Entity-Attributed F1 (EA-F1) metric, our benchmark advances the field toward more precise, entity-aware QA systems.

## 3 MEBench

## 3.1 Task overview

MEBench is a structured evaluation framework designed to systematically assess the capabilities of LLMs and RAG systems in performing crossdocument multi-entity question answering. This framework targets three core reasoning modalities: comparative analysis, statistical inference, and relational reasoning, and each decomposed into specialized subtasks that rigorously test distinct facets of LLM performance, ensuring broad coverage of real-world multi-entity reasoning scenarios. Examples of tasks are provided in Table 1. Each of three primary task categories addresses distinct reasoning challenges:

Comparative Reasoning Comparative reasoning tasks evaluate LLM’s ability to juxtapose entities across heterogeneous documents, demanding both attribute alignment and contextual synthesis.

Statistical Reasoning Statistical tasks assess LLM’s proficiency in quantitative synthesis, including aggregation, distributional analysis, correlation analysis, and variance analysis across multidocument.

Relational Reasoning Relational tasks probe model’s capacity to model explicit interactions and counterfactual dependencies among entities.

## 3.2 Benchmark Construction

MEBench was constructed through a systematic pipeline, comprising the following steps.

## 3.2.1 Data Collection

Concept Topic Identification. In the initial phase of data collection for MEbench, a meticulous process is employed to determine the concept topics that are applicable to multi-entity scenarios. These topics are carefully selected based on their significance, prevalence, and the potential for generating complex multi-entity questions, and examples can be seen in Appendix Table 5.

Entity and Property Identification. Once the concept topics are determined, we input descriptions related to the concept topics into a LLM (we use GPT-4), which then processes the text to identify concept entity and property, as illustrated in Figure 2-a1. After the LLM identifies the entity and property via iterative semantic refinement, we map them to entity IDs and property IDs in the

Table 1: Examples of multi-entities queries.
<table><tr><td>Categories</td><td>Types</td><td>Examples</td></tr><tr><td rowspan="2">Comparison</td><td>Intercomparison</td><td>Which has more ACM fellow, UK or USA?</td></tr><tr><td>Superlative</td><td>Which city has the highest population?</td></tr><tr><td rowspan="4">Statistics</td><td>Aggregation</td><td>How many ACM fellow are from MIT? Does the nationality of ACM fellows follow a</td></tr><tr><td>Distribution Compliance</td><td>normal distribution?</td></tr><tr><td>Correlation Analysis</td><td>Is there a linear relationship between number of events and records broken in Olympic Games?</td></tr><tr><td>Variance Analysis</td><td>Do the variances in the number of participat- ing countries and total events in the Summer Olympics differ significantly?</td></tr><tr><td rowspan="2">Relationship</td><td>Descriptive Relationship</td><td>Is there a relationship between the year of ACM fellowship induction and the fellows’ areas of expertise?</td></tr><tr><td>Hypothetical Scenarios</td><td>If China wins one more gold medal, will it over- take the US in the gold medal tally at the 2024 Olympics?</td></tr></table>

Table 2: Statistics of MEBench benchmark.
<table><tr><td>Categories</td><td>MEBench-train</td><td>MEBench-test</td><td>MEBench-total</td></tr><tr><td>#-Queries</td><td>3406</td><td>1374</td><td>4780</td></tr><tr><td>#-Topics</td><td>165</td><td>76</td><td>241</td></tr><tr><td>Ave. #-entities /Q</td><td>460</td><td>391</td><td>409</td></tr><tr><td colspan="4">Hops</td></tr><tr><td>#-one-hop Q</td><td>1406</td><td>606</td><td>2012</td></tr><tr><td>#-multi-hop Q</td><td>1322</td><td>768</td><td>2090</td></tr><tr><td colspan="4">Categories</td></tr><tr><td>#-Comparison</td><td>1107</td><td>438</td><td>1545</td></tr><tr><td>#-Statistics</td><td>1440</td><td>585</td><td>2025</td></tr><tr><td>#-Relationship</td><td>859</td><td>351</td><td>1210</td></tr><tr><td colspan="4">Entity Density</td></tr><tr><td>#-low (0-10)</td><td>487</td><td>196</td><td>683</td></tr><tr><td>#-medium(11-100)</td><td>973</td><td>393</td><td>1366</td></tr><tr><td>#-high (&gt;100)</td><td>1946</td><td>785</td><td>2731</td></tr></table>

Wiki graph. This mapping is crucial as it allows for seamless integration with the vast amount of structured data available in Wikipedia. The detailed method is in Appendix A.1. Using the entity ID and property ID, we synthesise SPARQL. We then utilize the API provided by Wikipedia to retrieve the wiki web pages of all entities related to the topic. For example, if our concept topic is "ACM Fellows", we would obtain the Wikipedia pages of all ACM Fellows, which contain their detailed information. We use GPT-4 to generate a set of interesting entity attributes. These attributes are carefully chosen based on general interest and relevance in the domain. For ACM Fellows, as an example, nationality, research field, institution, and academic contribution maybe the attributes that people commonly pay attention to.

![](images/abe3e123d63d173ae7a694b3d62f90b5db82cfa292a2b5fe71b8172d1840544d.jpg)  
Figure 2: The systematic pipeline of benchmark construction. It comprising three phases: documents collection, information extraction and question-answer generation. In the documents collection phase, concept topics relevant to multi-entity scenarios are selected, followed by GPT-4 processing descriptions to extract entities and properties mapped to Wikipedia IDs for integration with structured Wiki data. Structured information from Wikipedia documents is processed using small language models (SLMs) due to the structured nature of the documents, culminating in table creation with entity attributes as columns. For QA generation, questions are generated following a "template-driven, entity-attribute coupling" paradigm using GPT-4 with predefined templates, and undergo syntactic, semantic, and ambiguity checks, while answers are programmatically derived via SQL queries against the table and standardized into canonical forms. The final dataset ensures traceability (SQL-derived answers), scalability (template-driven approach), and rigor (execution-based answering reduces hallucination risks).

Structured Information Processing. Once the document set is collected, we proceed to the structured information processing stage. The documents we have gathered from Wikipedia have welldefined and accurate structural relations. Due to the structured nature of the documents, we do not need to rely on the long context ability of large language models. Instead, we can use small language models (SLMs) for information extraction. They are well-suited for tasks where the information is already structured and the focus is on extracting specific details (Fan et al., 2025).

Table Generation. The final step in the data collection process is to generate a table, as shown in Figure 2-b1. We use the the entity attributes as the column headers of the table. Each row in the table represents an individual entity. For example, in the case of ACM Fellows, each row would correspond to an individual ACM Fellow.

## 3.2.2 Question and answer Generation

The question and answer generation framework for MEBench is a structured, multi-phase process that leverages LLM and tabular data to produce both semantically coherent questions and computationally verifiable answers.

Question Generation. The foundational input for the QA generation pipeline is the table generated in last step. The generation of questions follows a "template-driven, entity-attribute coupling" paradigm, implemented through LLM (GPT-4), as illustrated in Figure 2-c1. Predefined syntactic and semantic templates govern the grammatical structure and intent of questions. These templates are shown in Appendix Table 6. The LLM instantiates templates with entity-attribute pairs, ensuring syntactic diversity while adhering to logical constraints. Generated questions undergo validation via: Syntactic checks, ensuring grammatical correctness; Semantic grounding, verifying that the question is answerable using the table’s data; Ambiguity reduction, pruning underspecified questions (e.g., “Describe the economy” revised to “Describe the GDP growth rate of Brazil in 2023”).

Answer Generation. Answers are derived programmatically through automated SQL query execution, ensuring reproducibility and alignment with the table’s ground-truth data. The synthesized SQL is executed against the table, yielding direct answers or sub-tables (Intermediate results requiring post-processing), as illustrated in Figure 2-c3. Answers are standardized to ensure consistency: Numeric results are rounded to significant figures; Categorical answers are converted to canonical forms (e.g., "USA" to "United States").

## 3.3 Data Statistics

The benchmark comprises 4,780 methodically structured questions partitioned into two subsets: a training set (3,406 questions) for model fine-tuning or train, and a test set (1,374 questions) for rigorous evaluation. Based on entity count, the data is divided into three groups: “low” (0-10), “Medium” (11-100), and “high” (>100), containing 683, 1366, and 2731 entries, respectively. Table 2 details comprehensive statistics of the benchmark. We also analyze the proportion of questions rejected during manual review and about 21% of the questions are failure to meet quality standards.

## 4 Experiment

## 4.1 Experiment Setup

Models. For open-source LLMs, we conduct experiments using the representative Meta-Llama-3-8B-Instruct (Meta Llama3, 2024) and apply QLoRA (Dettmers et al., 2023) to fine-tune it with the training set of MEBench. For proprietary LLMs, we select the widely recognized GPT models, including GPT-3.5-turbo (Ouyang et al., 2022) and GPT-4 (Achiam et al., 2023).

RAG. We implement a hierarchical retrieval framework that explicitly incorporates document organizational structures into the RAG pipeline to explore whether RAG can enhance the model’s performance on MEBench. For the Embedding choice, we employ the OpenAI Embedding model (OpenAI), and the chunk size is 1024. For each document, we retrieve the top-5 most related chunks and concatenate them in their original order to form the context input for the model.

Evaluation Metrics. We adopt Accuracy (Acc) as the primary metric to assess the performance of LLMs on MEBench tasks. For the subcategories of Variance Analysis, Correlation Analysis, and Distribution Compliance within the Statistics tasks, which are shown in Table 1, we focus solely on prompting LLMs to identify relevant columns and applicable methods, evaluating the accuracy of their selections instead of the computational results, as LLMs’ abilities in precise calculations are not the central focus of this study. In addition, we evaluate performance of information extraction using Entity-Attributed F1 (EA-F1). This is an F1 score applied to the predicted vs. gold sets of the (entity, atrribution, value) . All three elements in the tuple must exactly match the tuple in the ground truth to be marked correct.

## 4.2 Results and Analysis

Various models exhibit notable variations in performance on MEBench. Table 3 presents experimental results alongside overall accuracy on MEBench, and Figure 3 shows accuracy on eight furtherdivided tasks.

Main result. GPT-4 + RAG achieved superior accuracy (59.3%), outperforming the second-ranked model (FT Llama-3-Instruct: 55.6% ) by a statistically significant margin. Notably, GPT-4 + RAG excelled in relational (68.7%) and comparative (76.3%) queries, likely due to its superior contextual understanding. However, all models exhibited markedly lower accuracy in statistical queries (GPT-4 + RAG: 41.0%), suggesting inherent challenges in numerical reasoning. In our evaluation, we focused on analyzing the capability of LLMs to extract question-related data. This assessment aimed to understand how well these sophisticated models can organize and present data for the question. The result is shown in Table 4. These results underscore the critical role of information extraction architectures in mitigating hallucinations and grounding outputs in factual data. Introducing RAG significantly improves overall performance, particularly in comparison tasks, while fine-tuning LLaMA-3-Instruct alone does not yield substantial gains without RAG. On MEBench, open-source models like LLaMA-3-Instruct, even with RAG, can’t match proprietary models like GPT-4, which achieves a 59.3% accuracy compared to LLaMA-3-Instruct’s 32.5%.

Fine-grained Performance on Sub-tasks. Figure 3 shows that vanilla LLMs perform well in correlation analysis and descriptive relationship tasks, while RAG significantly improves intercomparison and superlative tasks. However, neither fine-tuning nor RAG overcomes challenges in variance analysis and aggregation tasks, while GPT-4 + RAG achieves accuracy of 15.3% and 32.1%.

Entity density Analysis. As we can see from Table 3, our experiments underscore the impact of entity density on model performance in MEQA tasks. This phenomenon arises because higher entity densities amplify two critical challenges inherent to MEQA systems: (1) Semantic ambiguity due to overlapping relational predicates among entities (e.g., distinguishing "Paris [person]" vs. "Paris [location]" within narrow contexts), and (2) computational overhead in attention-based architectures attempting parallel reasoning over entangled entityattribution pairs (e.g. transformer self-attention weights saturate under dense cross-entity dependencies).

Table 3: Experimental results for MEBench.
<table><tr><td rowspan="2">Models</td><td colspan="4">Accuracy</td></tr><tr><td>Comparison</td><td>Statistics</td><td>Relationship</td><td>Overall</td></tr><tr><td colspan="5">All sets</td></tr><tr><td>GPT-3.5-turbo</td><td>0.105</td><td>0.198</td><td>0.476</td><td>0.239</td></tr><tr><td>GPT-3.5-turbo + RAG</td><td>0.605</td><td>0.260</td><td>0.476</td><td>0.425</td></tr><tr><td>GPT-4</td><td>0.199</td><td>0.289</td><td>0.507</td><td>0.316</td></tr><tr><td>GPT-4 + RAG</td><td>0.763</td><td>0.410</td><td>0.687</td><td>0.593</td></tr><tr><td>Llama-3-Instruct</td><td>0.046</td><td>0.118</td><td>0.256</td><td>0.130</td></tr><tr><td>Llama-3-Instruct + RAG</td><td>0.447</td><td>0.181</td><td>0.410</td><td>0.325</td></tr><tr><td>FT Llama-3-Instruct</td><td>0.046</td><td>0.253</td><td>0.259</td><td>0.189</td></tr><tr><td>FT Llama-3-Instruct + RAG</td><td>0.687</td><td>0.448</td><td>0.573</td><td>0.556</td></tr><tr><td colspan="5">Set1 (0-10)</td></tr><tr><td>GPT-3.5-turbo</td><td>0.435</td><td>0.583</td><td>0.560</td><td>0.530</td></tr><tr><td>GPT-3.5-turbo + RAG</td><td>0.548</td><td>0.654</td><td>0.620</td><td>0.612</td></tr><tr><td>GPT-4</td><td>0.451</td><td>0.595</td><td>0.540</td><td>0.535</td></tr><tr><td>GPT-4 + RAG</td><td>0.870</td><td>0.619</td><td>0.740</td><td>0.729</td></tr><tr><td>Llama-3-Instruct</td><td>0.322</td><td>0.500</td><td>0.400</td><td>0.418</td></tr><tr><td>Llama-3-Instruct + RAG</td><td>0.419</td><td>0.571</td><td>0.480</td><td>0.500</td></tr><tr><td>FT Llama-3-Instruct</td><td>0.322</td><td>0.511</td><td>0.380</td><td>0.418</td></tr><tr><td>FT Llama-3-Instruct + RAG</td><td>0.580</td><td>0.677</td><td>0.690</td><td>0.676</td></tr><tr><td colspan="5">Set2 (11-100)</td></tr><tr><td>GPT-3.5-turbo</td><td>0.364</td><td>0.495</td><td>0.544</td><td>0.466</td></tr><tr><td>GPT-3.5-turbo + RAG</td><td>0.613</td><td>0.581</td><td>0.640</td><td>0.607</td></tr><tr><td>GPT-4</td><td>0.348</td><td>0.476</td><td>0.521</td><td>0.447</td></tr><tr><td>GPT-4 + RAG</td><td>0.791</td><td>0.511</td><td>0.661</td><td>0.638</td></tr><tr><td>Llama-3-Instruct</td><td>0.240</td><td>0.385</td><td>0.357</td><td>0.332</td></tr><tr><td>Llama-3-Instruct + RAG</td><td>0.428</td><td>0.454</td><td>0.459</td><td>0.447</td></tr><tr><td>FT Llama-3-Instruct</td><td>0.240</td><td>0.434</td><td>0.344</td><td>0.349</td></tr><tr><td>FT Llama-3-Instruct + RAG</td><td>0.612</td><td>0.608</td><td>0.655</td><td>0.640</td></tr><tr><td colspan="5">Set3 (&gt;100)</td></tr><tr><td>GPT-3.5-turbo</td><td>0.09</td><td>0.158</td><td>0.291</td><td>0.173</td></tr><tr><td>GPT-3.5-turbo + RAG</td><td>0.389</td><td>0.191</td><td>0.311</td><td>0.285</td></tr><tr><td>GPT-4</td><td>0.142</td><td>0.202</td><td>0.309</td><td>0.210</td></tr><tr><td>GPT-4 + RAG</td><td>0.436</td><td>0.270</td><td>0.405</td><td>0.357</td></tr><tr><td>Llama-3-Instruct</td><td>0.055</td><td>0.108</td><td>0.168</td><td>0.106</td></tr><tr><td>Llama-3-Instruct + RAG</td><td>0.265</td><td>0.147</td><td>0.253</td><td>0.212</td></tr><tr><td>FT Llama-3-Instruct</td><td>0.055</td><td>0.177</td><td>0.167</td><td>0.136</td></tr><tr><td>FT Llama-3-Instruct + RAG</td><td>0.401</td><td>0.291</td><td>0.355</td><td>0.345</td></tr></table>

![](images/a6564ddc98fcb5155a9fd5b01458b9baca1b93d8b03fa00f24596507f06c23e8.jpg)  
Figure 3: The Experimental results for eight subtasks of each model.

Table 4: Quality of Large Language Models (LLMs) in EA-F1.
<table><tr><td rowspan=1 colspan=1>Models                           EA- F1</td></tr><tr><td rowspan=1 colspan=1>GPT-3.5-turbo                      0.25 $\mathrm { G P T } { - } 3 . 5 { \mathrm { - } } \mathrm { t u r b o } + \mathrm { R A G }$              0.43</td></tr><tr><td rowspan=1 colspan=1>GPT-4                                0.36</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { G P T } \mathrm { - } 4 + \mathrm { R A G }$                       0.71</td></tr><tr><td rowspan=1 colspan=1>Llama-3-Instruct                   0.21</td></tr><tr><td rowspan=1 colspan=1>Llama-3-Instruct + RAG          0.39</td></tr><tr><td rowspan=1 colspan=1>FT Llama-3-Instruct               0.21</td></tr><tr><td rowspan=1 colspan=1> $\mathrm { F T \ L l a m a { - } } 3 { \mathrm { - } } \mathrm { I n s t r u c t + R A G }$      0.59</td></tr></table>

• Low Entity Density: Models generally performed well in low-density scenarios. The simplicity of context allowed for accurate entity recognition and minimal ambiguity.

• Medium Entity Density: Performance began to decrise among models in medium-density scenarios by 6% average acc. This variance suggests differences in how models handle increased entity complexity and overlapping contexts.

• High Entity Density: High-density questions posed significant challenges, with an average acc drop to 22.8% across models. The result highlighting limitations in current architectures’ ability to handle complex multi-entity questions.

## 5 Limitations

While MEBench provides a comprehensive framework for evaluating cross-document multi-entity reasoning, our work has several limitations that warrant further investigation. Although MEBench covers eight distinct reasoning types across three broad categories, real-world MEQA scenarios may involve even more intricate combinations of logical, temporal, or causal dependencies. The current benchmark does not explicitly model dynamic or time-sensitive entity interactions, which could limit its applicability to domains like financial forecasting or event-driven narratives. The benchmark relies on a curated collection of documents to ensure controlled evaluation. While this design choice minimizes noise, it may not fully replicate the challenges of real-world environments where documents vary widely in quality, redundancy, and structure. Future iterations could incorporate noisy or incomplete data sources to better simulate practical scenarios. While the Entity-Attributed F1 (EA-F1) metric rigorously assesses entity-level correctness and attribution validity, it prioritizes factual precision over semantic coherence. This may undervalue partially correct answers that demonstrate valid reasoning chains but contain minor factual inaccuracies. A hybrid evaluation framework combining EA-F1 with human judgment could provide a more holistic assessment.

## 6 Conclusion

In this study, we have comprehensively addressed the significant challenges that Multi-entity Question Answering (MEQA) poses to LLMs and RAG systems. The limitations of existing methods in handling cross-document aggregation, especially when dealing with entity-dense questions, have been clearly identified and analyzed. We introduced MEBench, a groundbreaking multidocument, multi-entity benchmark. Our experiments on state-of-the-art LLMs such as GPT-4 and Llama-3, along with RAG pipelines, have shed light on the critical limitations of these advanced models. The fact that even these leading models achieve only 59% accuracy on MEBench underscores the magnitude of the challenges in MEQA. MEBench has effectively highlighted the systemic weaknesses in current LLM frameworks. These weaknesses serve as valuable insights for future research directions. For instance, the need for improved algorithms to retrieve and consolidate fragmented information from heterogeneous sources is evident. Additionally, there is a pressing need to develop more robust entity-aware QA architectures that can better handle the complexities of MEQA.

## 7 Acknowledgment

This work is supported by Guangdong provincial project 2023CX10X008.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Chenxin An, Shansan Gong, Ming Zhong, Xingjian Zhao, Mukai Li, Jun Zhang, Lingpeng Kong, and Xipeng Qiu. 2024. L-eval: Instituting standardized

evaluation for long context language models. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 14388–14411, Bangkok, Thailand. Association for Computational Linguistics.

Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei Hou, Yuxiao Dong, Jie Tang, and Juanzi Li. 2025. Longbench v2: Towards deeper understanding and reasoning on realistic long-context multitasks.

Bernd Bohnet, Vinh Q. Tran, Pat Verga, Roee Aharoni, Daniel Andor, Livio Baldini Soares, Massimiliano Ciaramita, Jacob Eisenstein, Kuzman Ganchev, Jonathan Herzig, Kai Hui, Tom Kwiatkowski, Ji Ma, Jianmo Ni, Lierni Sestorain Saralegui, Tal Schuster, William W. Cohen, Michael Collins, Dipanjan Das, Donald Metzler, Slav Petrov, and Kellie Webster. 2023. Attributed question answering: Evaluation and modeling for attributed large language models.

Sibei Chen, Ju Fan, Bin Wu, Nan Tang, Chao Deng, Pengyi Wang, Ye Li, Jian Tan, Feifei Li, Jingren Zhou, and Xiaoyong Du. 2025. Automatic database configuration debugging using retrievalaugmented language models. Proc. ACM Manag. Data, 3(1):13:1–13:27.

Zui Chen, Zihui Gu, Lei Cao, Ju Fan, Samuel Madden, and Nan Tang. 2023. Symphony: Towards natural language query answering over multi-modal data lakes. In 13th Conference on Innovative Data Systems Research, CIDR 2023, Amsterdam, The Netherlands, January 8-11, 2023. www.cidrdb.org.

Aakanksha Chowdhery, Sharan Narang, and Jacob Devlin. 2023. Palm: Scaling language modeling with pathways. Journal ofMachine Learning Research, 24.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. Qlora: Efficient finetuning of quantized llms. arXiv preprint arXiv:2305.14314.

Zican Dong, Tianyi Tang, Junyi Li, Wayne Xin Zhao, and Ji-Rong Wen. 2024. BAMBOO: A comprehensive benchmark for evaluating long text modeling capacities of large language models. In Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024), pages 2086–2099, Torino, Italia. ELRA and ICCL.

Meihao Fan, Xiaoyue Han, Ju Fan, Chengliang Chai, Nan Tang, Guoliang Li, and Xiaoyong Du. 2024a. Cost-effective in-context learning for entity resolution: A design space exploration. In 40th IEEE International Conference on Data Engineering, ICDE 2024, Utrecht, The Netherlands, May 13-16, 2024, pages 3696–3709. IEEE.

Tianyu Fan, Jingyuan Wang, Xubin Ren, and Chao Huang. 2025. Minirag: Towards extremely simple retrieval-augmented generation.

Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat Seng Chua, and Qing Li. 2024b. A survey on rag meeting llms: Towards retrieval-augmented large language models.

Chi Han, Qifan Wang, Hao Peng, Wenhan Xiong, Yu Chen, Heng Ji, and Sinong Wang. 2024. LMinfinite: Zero-shot extreme length generalization for large language models. In Proceedings ofthe 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3991–4008, Mexico City, Mexico. Association for Computational Linguistics.

Sirui Hong, Yizhang Lin, Bang Liu, Bangbang Liu, Binhao Wu, Ceyao Zhang, Danyang Li, Jiaqi Chen, Jiayi Zhang, Jinlin Wang, Li Zhang, Lingyao Zhang, Min Yang, Mingchen Zhuge, Taicheng Guo, Tuo Zhou, Wei Tao, Robert Tang, Xiangtao Lu, Xiawu Zheng, Xinbing Liang, Yaying Fei, Yuheng Cheng, Yongxin Ni, Zhibin Gou, Zongze Xu, Yuyu Luo, and Chenglin Wu. 2025. Data interpreter: An LLM agent for data science. In ACL (Findings), pages 19796– 19821. Association for Computational Linguistics.

Greg Kamradt. 2023. Needle in a haystack- pressure testing llms. https://github.com/gkamradt/ LLMTest\_NeedleInAHaystack.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, and Slav Petrov. 2019. Natural questions: A benchmark for question answering research. Transactions ofthe Associationfor Computational Linguistics, 7(15):453–466.

Boyan Li, Yuyu Luo, Chengliang Chai, Guoliang Li, and Nan Tang. 2024a. The dawn of natural language to SQL: are we fully ready? Proc. VLDB Endow., 17(11):3318–3331.

Boyan Li, Jiayi Zhang, Ju Fan, Yanwei Xu, Chong Chen, Nan Tang, and Yuyu Luo. 2025a. Alpha-sql: Zeroshot text-to-sql using monte carlo tree search. CoRR, abs/2502.17248.

Changlun Li, Chenyu Yang, Yuyu Luo, Ju Fan, and Nan Tang. 2025b. Weak-to-strong prompts with lightweight-to-powerful llms for high-accuracy, lowcost, and explainable data transformation. Proc. VLDB Endow., 18(8):2371–2384.

Jiaqi Li, Mengmeng Wang, Zilong Zheng, and Muhan Zhang. 2024b. Loogle: Can long-context language models understand long contexts?

Teng Lin. 2025a. Simplifying data integration: Slmdriven systems for unified semantic queries across heterogeneous databases. In 2025 IEEE 41st International Conference on Data Engineering (ICDE), pages 4690–4693.

Teng Lin. 2025b. Structured retrieval-augmented generation for multi-entity question answering over heterogeneous sources. In 2025 IEEE 41st International Conference on Data Engineering Workshops (ICDEW), pages 253–258.

Teng Lin, Yizhang Zhu, Yuyu Luo, and Nan Tang. 2025a. Srag: Structured retrieval-augmented generation for multi-entity question answering over wikipedia graph. CoRR, abs/2503.01346.

Xiaotian Lin, Yanlin Qi, Yizhang Zhu, Themis Palpanas, Chengliang Chai, Nan Tang, and Yuyu Luo. 2025b. LEAD: iterative data selection for efficient LLM instruction tuning. CoRR, abs/2505.07437.

Bang Liu, Xinfeng Li, Jiayi Zhang, Jinlin Wang, Tanjin He, Sirui Hong, Hongzhang Liu, Shaokun Zhang, Kaitao Song, Kunlun Zhu, Yuheng Cheng, Suyuchen Wang, Xiaoqiang Wang, Yuyu Luo, Haibo Jin, Peiyan Zhang, Ollie Liu, Jiaqi Chen, Huan Zhang, Zhaoyang Yu, Haochen Shi, Boyan Li, Dekun Wu, Fengwei Teng, Xiaojun Jia, Jiawei Xu, Jinyu Xiang, Yizhang Lin, Tianming Liu, Tongliang Liu, Yu Su, Huan Sun, Glen Berseth, Jianyun Nie, Ian Foster, Logan T. Ward, Qingyun Wu, Yu Gu, Mingchen Zhuge, Xiangru Tang, Haohan Wang, Jiaxuan You, Chi Wang, Jian Pei, Qiang Yang, Xiaoliang Qi, and Chenglin Wu. 2025a. Advances and challenges in foundation agents: From brain-inspired intelligence to evolutionary, collaborative, and safe systems. CoRR, abs/2504.01990.

Chunwei Liu, Matthew Russo, Michael Cafarella, Lei Cao, Peter Baille Chen, Zui Chen, Michael Franklin, Tim Kraska, Samuel Madden, and Gerardo Vitagliano. 2024. A declarative system for optimizing ai workloads. arXiv preprint arXiv:2405.14696.

Xinyu Liu, Shuyu Shen, Boyan Li, Peixian Ma, Runzhi Jiang, Yuxin Zhang, Ju Fan, Guoliang Li, Nan Tang, and Yuyu Luo. 2025b. A survey of text-to-sql in the era of llms: Where are we, and where are we going?

Xinyu Liu, Shuyu Shen, Boyan Li, Nan Tang, and Yuyu Luo. 2025c. Nl2sql-bugs: A benchmark for detecting semantic errors in NL2SQL translation. CoRR, abs/2503.11984.

Yuyu Luo, Xuedi Qin, Nan Tang, and Guoliang Li. 2018a. Deepeye: Towards automatic data visualization. In ICDE, pages 101–112. IEEE Computer Society.

Yuyu Luo, Xuedi Qin, Nan Tang, Guoliang Li, and Xinran Wang. 2018b. Deepeye: Creating good data visualizations by keyword search. In SIGMOD Conference, pages 1733–1736. ACM.

Yuyu Luo, Nan Tang, Guoliang Li, Chengliang Chai, Wenbo Li, and Xuedi Qin. 2021. Synthesizing natural language to visualization (NL2VIS) benchmarks from NL2SQL benchmarks. In SIGMOD Conference, pages 1235–1247. ACM.

Yuyu Luo, Nan Tang, Guoliang Li, Jiawei Tang, Chengliang Chai, and Xuedi Qin. 2022. Natural language to visualization by neural machine translation. IEEE Trans. Vis. Comput. Graph., 28(1):217–226.

Meta Llama3. 2024. Meta llama3. https://llama. meta.com/llama3/. Accessed: 2024-04-10.

OpenAI. Openai embedding model. https://huggingface.co/Xenova/text-embedding-ada-002. Accessed [Date of access].

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744.

Pratyush Patel, Esha Choukse, Chaojie Zhang, Íñigo Goiri, Aashaka Shah, Saeed Maleki, and Ricardo Bianchini. 2023. Splitwise: Efficient generative llm inference using phase splitting. arXiv preprint arXiv:2311.18677.

Xuedi Qin, Yuyu Luo, Nan Tang, and Guoliang Li. 2020. Making data visualization more efficient and effective: a survey. VLDB J., 29(1):93–117.

Zexuan Qiu, Jingjing Li, Shijue Huang, Xiaoqi Jiao, Wanjun Zhong, and Irwin King. 2024. Clongeval: A chinese benchmark for evaluating long-context large language models.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. Squad: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing.

A. Rostampour, A. Kazemi, F. Shams, A. Zamiri, and P. Jamshidi. 2010. A metric for measuring the degree of entity-centric service cohesion. In 2010 IEEE International Conference on Service-Oriented Computing and Applications (SOCA), pages 1–5.

Jingze Shi, Yifan Wu, Bingheng Wu, Yiran Peng, Liangdong Wang, Guang Liu, and Yuyu Luo. 2025. Trainable dynamic mask sparse attention.

Mingyang Song, Mao Zheng, and Xuan Luo. 2024. Counting-stars: A multi-evidence, position-aware, and scalable benchmark for evaluating long-context large language models.

Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Lixin Su, Suqi Cheng, Dawei Yin, and Chao Huang. 2024a. Graphgpt: Graph instruction tuning for large language models. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR ’24, page 491–500, New York, NY, USA. Association for Computing Machinery.

Nan Tang, Chenyu Yang, Ju Fan, Lei Cao, Yuyu Luo, and Alon Y. Halevy. 2024b. Verifai: Verified generative AI. In CIDR. www.cidrdb.org.

James Thorne, Andreas Vlachos, Christos Christodoulopoulos, and Arpit Mittal. 2018. Fever: a large-scale dataset for fact extraction and verification.

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2021. Musique: Multi-hop questions via single-hop question composition.

Minzheng Wang, Longze Chen, Cheng Fu, Shengyi Liao, Xinghua Zhang, Bingli Wu, Haiyang Yu, Nan Xu, Lei Zhang, and Run Luo. 2024. Leave no document behind: Benchmarking long-context llms with extended multi-doc qa.

Bingheng Wu, Jingze Shi, Yifan Wu, Nan Tang, and Yuyu Luo. 2025a. Transxssm: A hybrid transformer state space model with unified rotary position embedding. CoRR, abs/2506.09507.

Kevin Wu, Eric Wu, and James Zou. 2025b. Clasheval: Quantifying the tug-of-war between an llm’s internal prior and external evidence.

Yifan Wu, Jingze Shi, Bingheng Wu, Jiayi Zhang, Xiaotian Lin, Nan Tang, and Yuyu Luo. 2025c. Concise reasoning, big gains: Pruning long reasoning trace with difficulty-aware prompting. CoRR, abs/2505.19716.

Jinyu Xiang, Jiayi Zhang, Zhaoyang Yu, Fengwei Teng, Jinhao Tu, Xinbing Liang, Sirui Hong, Chenglin Wu, and Yuyu Luo. 2025. Self-supervised prompt optimization. CoRR, abs/2502.06855.

Xiao Yang, Kai Sun, Hao Xin, Yushi Sun, Nikita Bhalla, Xiangsen Chen, Sajal Choudhary, Rongze Daniel Gui, Ziran Will Jiang, Ziyu Jiang, Lingkun Kong, Brian Moran, Jiaqi Wang, Yifan Xu, An Yan, Chenyu Yang, Eting Yuan, Hanwen Zha, Nan Tang, Lei Chen, Nicolas Scheffer, Yue Liu, Nirav Shah, Rakesh Wanga, Anuj Kumar, Scott Yih, and Xin Dong. 2024. CRAG - comprehensive RAG benchmark. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W Cohen, Ruslan Salakhutdinov, and Christopher D Manning. 2018. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing.

Jiayi Zhang, Jinyu Xiang, Zhaoyang Yu, Fengwei Teng, Xionghui Chen, Jiaqi Chen, Mingchen Zhuge, Xin Cheng, Sirui Hong, Jinlin Wang, Bingnan Zheng, Bang Liu, Yuyu Luo, and Chenglin Wu. 2025a. Aflow: Automating agentic workflow generation. In ICLR. OpenReview.net.

Xinrong Zhang, Yingfa Chen, Shengding Hu, Zihang Xu, Junhao Chen, Moo Khai Hao, Xu Han, Zhen Leng Thai, Shuo Wang, Zhiyuan Liu, and Maosong Sun. 2024a. bench: Extending long context evaluation beyond 100k tokens.

Zhengxuan Zhang, Zhuowen Liang, Yin Wu, Teng Lin, Yuyu Luo, and Nan Tang. 2025b. Datamosaic: Explainable and verifiable multi-modal

data analytics through extract-reason-verify. CoRR, abs/2504.10036.

Zhengxuan Zhang, Yin Wu, Yuyu Luo, and Nan Tang. 2024b. MAR: matching-augmented reasoning for enhancing visual-based entity question answering. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024, Miami, FL, USA, November 12-16, 2024, pages 1520–1530. Association for Computational Linguistics.

Yizhang Zhu, Shiyin Du, Boyan Li, Yuyu Luo, and Nan Tang. 2024. Are large language models good statisticians? In NeurIPS.

Yizhang Zhu, Runzhi Jiang, Boyan Li, Nan Tang, and Yuyu Luo. 2025. Elliesql: Cost-efficient text-to-sql with complexity-aware routing. CoRR, abs/2503.22402.

|<sup>V</sup> <sup>alidResults</sup>t|<sub>TotalResults</sub> ≥ <sup>θ</sup>precision   
(θ = 0.98 empirically)   
or maximum iteration thresholds.

## A Appendix

A.1 Methodology for composite SPARQL Generation via Iterative Semantic Refinement

## A.1.1 Initial Query Parsing Using GPT-4

We employ a transformer-based large language model (LLM), specifically GPT-4, to perform preliminary natural language question decomposition. This stage generates a proto-SPARQL query containing candidate triple patterns with hypothesized entity-property relationships. While this initial output captures broad syntactic structures (e.g., basic graph pattern groupings), it frequently exhibits two critical inaccuracies:

Entity Misalignment: Incorrect Wikidata Q-ID assignments due to lexical ambiguity (e.g., "Java" as programming language vs. Indonesian island)

Property Mismatch: Invalid P-ID selections arising from underspecified predicate semantics (e.g., using P19 [place of birth] instead of P20 [place of death])

## A.1.2 Semantic Validation Layer

To address these limitations, we implement a multistage correction framework:

(a) Structured Knowledge Anchoring

The system interfaces with the Wikipedia API through programmatic endpoints that map surface forms to canonical entities.

(b) Neural-Semantic Disambiguation Module

GPT-4 serves as our semantic analysis engine, performing three key operations:

• Contextual disambiguation using entity linking algorithms enhanced by Wikifier-style mention detection.

• Property type validation against Wikidata’s ontology constraints (rdf:type, owl:ObjectProperty).

• Temporal scope verification for time-sensitive queries requiring qualifiers like P585 [point in time].

## A.1.3 Iterative Refinement Protocol

The system implements closed-loop feedback through successive cycles of:

• Executing candidate SPARQL on the Wikidata Query Service endpoint.

• Analyzing result cardinality and type consistency.

• Applying constraint satisfaction heuristics:

FILTER (?population > 1e6 && ?country   
IN wd:Q30) # Example numerical/entity con  
straints

Each iteration tightens precision metrics until meeting termination criteria defined by either:

## A.1.4 Final Query Synthesis

Through combining LLM-based semantic parsing with knowledge-grounded verification, we converge on an optimized SPARQL template satisfying both syntactic validity and functional correctness requirements for structured knowledge extraction.

## A.2 Optimization

Two aspects of optimization are included in MEBench system to enhance the overall performance:

Model Selection. Model selection is straightforward yet highly effective for optimization Liu et al. (2024). Our system comprises multiple tasks, necessitating the selection of the most suitable model for different tasks. For basic tasks, more affordable and faster LLMs can suffice, while utilization of the most advanced LLMs is essential in more complex tasks to ensure optimal performance. Specifically, our system employs powerful yet resourceintensive GPT-4 for tasks such as semantic analysis or generation of table schemas and SQL queries. In contrast, for more basic information extraction, we utilize open-source Mistral-7B, thereby achieving a balance between cost efficiency and functional performance.

LLM Input/Output Control SplitWise Patel et al. (2023) shows that LLM inference time is generally proportional to the size of input and output tokens. Since GPT models decide the cost based on the input token, we try to minimize the input of large models. Meanwhile, we use the instructive prompt to reduce the size of the outputs generated by LLM without changing the quality of these outputs. The example of prompt is in Appendix A.2.1.

Table 5: Example Topics and Their Entities Attributions.
<table><tr><td>Topics</td><td>Entities Attributions</td><td>#-Entities</td></tr><tr><td>ACM fellow</td><td>nationality, field of study, affiliation</td><td>1115</td></tr><tr><td>Presidents of the US</td><td>term lengths, political parties, vice-presidents, birth states, previous occupations</td><td>55</td></tr><tr><td>Chemical Elements</td><td>atomic number, atomic mass, boiling point, melting point, electron configuration</td><td>166</td></tr><tr><td>Summer Olympic Games</td><td>host cities, number of participating countries, total number of events, medal tally, records broken</td><td>35</td></tr><tr><td>Nobel Prize in Chemistry</td><td>categories, year of award, country of origin, field of contribution.</td><td>194</td></tr><tr><td>Cities of the World</td><td>population, geographic coordinates, altitude, GDP</td><td>7040</td></tr></table>

Table 6: Template example for questions generated by the LLM (GPT-4).
<table><tr><td>Types</td><td>Sub-types</td><td>Template Examples</td></tr><tr><td rowspan="2">Comparison</td><td>Intercomparison</td><td>Which has high [property], [entity A] or [entity B]?</td></tr><tr><td>Superlative</td><td>Which [entity] has the highest/lowest [property]?</td></tr><tr><td rowspan="4">Statistics</td><td>Aggregation</td><td>How many [entities] have [specific property value]?</td></tr><tr><td>Distribution Compliance</td><td>Does [property] follow a normal distribution?</td></tr><tr><td>Correlation Analysis</td><td>Is there a linear relationship between [property A] and [property B]?</td></tr><tr><td>Variance Analysis</td><td>Are the variances in [property A] and [property B] significantly different?</td></tr><tr><td rowspan="2">Relationship</td><td>Descriptive Relationship</td><td>How is [entity A] related to [entity B]?</td></tr><tr><td>Hypothetical Scenarios</td><td>What would be the impact if [entity A] collabo- rates with [entity B]?</td></tr></table>

## A.2.1 Prompt for Output Control

Review your output to ensure it meets all the above criteria. Your goal is to produce a clear, accurate, and well-structured output. Just output the result, no other word or symbol.

## A.2.2 Quality Control

We devise several strategies to ensure the integrity and effectiveness of questions.

Question Templates. The use of templates ensures that every question is crafted with a clear structure, making it easier for respondents to understand and answer them accurately. For relationship and complex statistic questions we turn the questions in a closed-ended style, as they require a specific response of either "yes" or "no", which make the answer in a standardized format. The examples of Question Templates is in the Appendix 6.

Question Refinement. After initial development, each question undergoes a refinement process which we used GPT-3.5-Turbo. This stage is critical for enhancing the clarity, relevance, and neutrality of the questions. It involves reviewing the questions for bias. This strategy helps in reducing misunderstandings and improving the overall quality of the questions.

Manual review. We assess the questions for accuracy, ensuring they are factually correct and relevant to our purpose. Manual reviews can also provide insights into whether the questions are likely to effectively elicit the intended information from answers, thereby contributing to the reliability and validity of the benchmark.

## A.3 Tables

Table 5 shows examples of topics and their entities attributions. Table 6 shows examples of question templates to synthesize questions.