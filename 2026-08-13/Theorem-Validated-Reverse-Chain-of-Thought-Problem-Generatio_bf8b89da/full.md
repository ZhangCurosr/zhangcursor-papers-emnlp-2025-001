# Theorem-Validated Reverse Chain-of-Thought Problem Generation for Geometric Reasoning

Linger Deng<sup>1</sup>∗, Linghao Zhu<sup>1</sup>∗, Yuliang Liu<sup>1</sup>†, Yu Wang<sup>2</sup>, Qunyi Xie<sup>2</sup>, Jingjing Wu<sup>2</sup>, Gang Zhang<sup>2</sup>, Yingying Zhu<sup>1</sup>, Xiang Bai<sup>1</sup>,

<sup>1</sup>Huazhong University of Science and Technology, <sup>2</sup>Department of Computer Vision Technology, Baidu Inc, Correspondence: lingerdeng, ylliu@hust.edu.cn

## Abstract

Large Multimodal Models (LMMs) face limitations in geometric reasoning due to insufficient Chain of Thought (CoT) image-text training data. While existing approaches leverage template-based or LLM-assisted methods for geometric CoT data creation, they often face challenges in achieving both diversity and precision. To bridge this gap, we introduce a twostage Theorem-Validated Reverse Chain-of-Thought Reasoning Synthesis (TR-CoT) framework. The first stage, TR-Engine, synthesizes theorem-grounded geometric diagrams with structured descriptions and properties. The second stage, TR-Reasoner, employs reverse reasoning to iteratively refine question-answer pairs by cross-validating geometric properties and description fragments. Our approach expands theorem-type coverage, corrects longstanding misunderstandings, and enhances geometric reasoning. Fine-grained CoT enhances theorem understanding and improves logical consistency by 24.5%.. Our best models surpass the baselines in MathVista and GeoQA by 10.1% and 4.7%, outperforming advanced closed-source models like GPT-4o. The code is available at https://github.com/dle666/ R-CoT.

## 1 Introduction

Large Language Models (LLMs) (OpenAI, 2024; Guo et al., 2025) have revolutionized textual mathematical reasoning through advanced inferential mechanisms. While architectural innovations now enable these models to process multimodal inputs via parameter-efficient vision-language alignment (e.g., GPT-4o (Islam and Moushi, 2024), Gemini (Team et al., 2023)), achieving human-competitive VQA performance (Fan et al., 2024), their geometric reasoning remains constrained (Wang et al., 2025). This limitation stems from training data dominated by natural scenes, which lack the geometric specificity required for rigorous spatial problem-solving.

Current methods for generating geometric reasoning data through Chain-of-Thought (CoT) frameworks face three fundamental limitations. First, rephrasing approaches (Gao et al., 2023b) use LLM to transform the CoT format of existing problems, which requires scarce high-quality annotations and domain-specific expertise to ensure theorem consistency (Fig. 1 (a)). Second, templatebased methods (Kazemi et al., 2023a; Zhang et al., 2024b) generate geometrically oversimplified images by combining predefined polygons in rigid configurations, lacking theorem-aware element interactions, limiting their applicability to advanced reasoning, as shown in Fig. 1 (b). Thirdly, while LMM-based reasoning (Peng et al., 2024) ensures reasoning diversity, insufficient mathematical priors often lead to incorrect reasoning, e.g., misusing theorems in the wrong situation, leading to logically invalid chains of reasoning (Fig. 1 (c)).

We introduce Theorem-Validated Reverse Chainof-Thought (TR-CoT), a two-stage framework designed to generate geometric reasoning data and verify logical flows, as shown in Fig. 1 (d). We first develop the theorem-driven image and property generation engine (TR-Engine) to create images paired with geometric properties, ensuring dependencies among elements. Then, TR-Reasoner derives questions from answers by segmenting image descriptions, generating single-step reasoning, and combining them into multi-step reasoning chains. Each step is verified against geometric properties, discarding pairs that violate mathematical rules, ensuring the logical rigor of generated data.

With TR-CoT, we create TR-GeoMM and TR-GeoSup, comprehensive datasets of diverse geometric theorems, which fully leverage CoT information. TR-CoT can bring notable and consistent improvements across a range of LMM baselines such as LLaVA, Qwen, and InternVL. Using the recent LMM baselines, we achieve a new performance record in 2B, 7B, and 8B settings for solving geometry problems. The main advantages of our method are summarized as follows:

![](images/377d8fe147f8a7002da03c49889d87cbc5ef9db36f0549d6a527fff99b70ba21.jpg)  
Figure 1: Comparison of TR-CoT with existing CoT data generation approaches. (a) Rephrase existing Q&A pairs using LLMs, relying on existing CoT data. (b) Generate images and CoT data using pre-defined templates containing a limited number of theorems. (c) Generate CoT using LMMs, where accuracy is limited by the performance of the LMMs. (d) Design the TR-Engine to generate images, corresponding descriptions, and geometric properties from theorems. And input the descriptions and properties into TR-Reasoner to generate reliable CoT Q&A pairs.

• Compared to traditional template-based methods, our approach covers twice the number of theorem types, effectively correcting long-standing theorem misunderstandings in models and enhancing their geometric reasoning.

• Generating geometric data with fine-grained CoTs enhances the model’s understanding of theorems, increasing the proportion of logically consistent and clear outputs by 24.5%.

• Our most advanced models achieve a 10.1% performance gain on MathVista and 4.7% on GeoQA over the baseline, outperforming advanced closed-source models such as GPT-4o.

## 2 Related Work

Enhancing Reasoning with CoT in Inference. Chain-of-thought (CoT) prompting has improved reasoning in math tasks. KQG-CoT (Liang et al., 2023a) selects logical forms from unlabeled data via CoT-based KBQG. In general math, code-based self-verification (Zhou et al., 2023) and SSC-CoT (Zhao et al., 2024b) enhance reliability by combining reasoning with structured knowledge. Other prompting strategies, including PEP (Liao et al., 2024), Plan-and-Solve (Wang et al., 2023), and incontext demonstrations (Didolkar et al., 2024), further refine inference. In geometry, visual-symbolic

CoT methods (Zhao et al., 2024a; Hu et al., 2024) align reasoning with multimodal representations.

Enhancing Reasoning in Geometry Training. Training geometric solvers requires scalable and diverse data. Early symbolic systems (e.g., GeoS (Seo et al., 2015), Inter-GPS (Lu et al., 2021)) relied on small benchmarks, while neural approaches like UniGeo (Chen et al., 2022) and PGPS9K (Zhang et al., 2023a) scaled up with costly manual annotations. Recent methods automate data generation using visual-language models (e.g., G-LLaVA (Gao et al., 2023a)) or code-based engines (Kazemi et al., 2023b; Zhang et al., 2024b). Geo-Eval (Zhang et al., 2024a) provides fine-grained evaluation across diverse reasoning settings. LLMgenerated CoT traces (Peng et al., 2024) offer new avenues for training data synthesis.

Recently, reverse engineering has helped diagnose and refine LLM reasoning. Techniques such as condition-answer swapping (Jiang et al., 2024; Weng et al., 2023), error localization (Xue et al., 2023), and prompt optimization (Yuan et al., 2024) validate reasoning consistency without model updates. However, they often lack integration into training. Our approach embeds reverse reasoning into CoT generation, producing fine-grained, theorem-aware supervision for model training.

## 3 Theorem-Validated Reverse Chain-of-Thought

There are two key challenges for generating geometry reasoning data: (1) Direct generation of question-answer pairs often leads to errors or unsolvable problems due to oversimplified scenarios. (2) Single-step reasoning processes lack validation of intermediate steps, compromising reliability.

![](images/f07e9980a0a0d8c6153220db34fd37c00bc051782c317c984a3929ca12f0cffa.jpg)  
Figure 2: The TR-Engine generates diverse images, corresponding descriptions, and geometric properties step by step based on geometric theorems. Subsequently, the TR-Reasoner is utilized to obtain accurate geometric Q&A pairs from descriptions and properties.

We propose Theorem-Validated Reverse Chainof-Thought (TR-CoT), a two-stage framework for creating geometry reasoning data with verified logical flow, as shown in Fig. 2. The pseudo-code of TR-CoT is shown in Appendix A.

1) Stage 1: Theorem-Driven Image & Property Generation. We collect 110 fundamental geometry theorems (Complete theorems and collection method are shown in Appendix K) and develop TR-Engine, a structured method to generate images paired with textual descriptions and geometric properties (e.g., angles, lengths). Unlike random image generation, TR-Engine guides image generation based on the sampled theorems and enforces dependencies between geometric elements across generation steps. Each step must operate on the geometric primitives—such as lines, angles, and points—produced in the preceding step.

2) Stage 2: Q&A Generation with Stepwise Validation. Using the descriptions and properties from Stage 1, TR-Reasoner generates questions from answers through three steps: First, the image description is divided into logical segments (e.g., “Triangle ABC is isosceles with AB = AC”). An LLM processes these parts step-by-step, generating individual inferences that are then combined into multi-step reasoning chains. Secondly, for each reasoning step, the system creates corresponding questions. For instance, the inference $^ { * } \angle B = \angle C ^ { * }$ generates the question: “If triangle ABC is isosceles with AB=AC, which angles are equal?” Finally, all Q&A pairs are cross-checked against the geometric properties from Stage 1. Pairs violating mathematical rules (e.g., claiming ${ } ^ { \ast } \angle A = 9 0 ^ { \circ \mathfrak { s } }$ for a non-right isosceles triangle) are discarded.

## 3.1 TR-Engine

TR-Engine is a theorem-guided framework for synthesizing geometrically valid images with rich relational structures, corresponding descriptions, and geometric properties. TR-Engine operates through four key components (Fig. 3):

1) Geometric Theorem Library. The 110 fundamental geometric theorems are classified into substrate-related theorems and line-element-related theorems. During the image generation process, 1 to 3 theorems from each category are sampled to guide the selection of geometric substrates and the addition of line elements.

2) Geometric Substrate Library. We curate 20 fundamental geometric shapes (substrates), such as triangles, circles, and quadrilaterals. Each substrate is paired with a set of relevant geometric theorems and description templates. During image generation, appropriate substrates are selected according to the sampled theorems. The description templates encode geometric conditions (e.g., “In triangle ABC, AB = 5 cm and BC = 6 cm”) to anchor subsequent reasoning steps.

3) Theorem-Based Dynamic Element Injection. This component strategically injects elements to enable complex reasoning scenarios based on theorem requirements. For example: Adding parallel lines to invoke properties of alternate angles. Introducing auxiliary lines (e.g., medians, altitudes) to create congruent sub-shapes. Such operations expand reasoning opportunities while maintaining geometric validity. In addition, TR-Engine assigns line segment values and angle degrees using exact vertex coordinates, preventing numerical conflicts from geometric constraints.

![](images/72d23fe39bb967cefa9c2c465af1cb3b6bbd853f99eef89a7f6805f38a83a7f8.jpg)  
Figure 3: Overview of the TR-Engine. Starting from a Geometric Substrate Library, dynamically injecting elements based on theorems, and integrating a property computation module to enable multi-step geometric reasoning and validation in image generation.

4) Property Computation Module. As elements are added, the vertex coordinates are used to automatically calculate: Metric properties: Lengths, angles, areas. Relational properties: Parallelism, congruence, symmetry. These properties serve as ground truth for verifying generated Q&A pairs. Additionally, we perform a visual fidelity check on geometric properties, filtering out distorted images with abnormal vertex spacing (the ratio of the maximum distance to the minimum distance exceeds a threshold) or extreme angles (less than 15 degrees or more than 160 degrees).

By integrating theorem-driven construction with stepwise validation, TR-Engine ensures images inherently support multi-step geometric reasoning, which is a critical advance over prior generation methods in practice. Additionally, to expand the range of potential element relationships and support the introduction of new substrates, we explore all possible interactions between elements. Distorted images with abnormal vertex spacing or extreme angles are filtered out by automatically analyzing geometric properties.

## 3.2 TR-Reasoner

Despite advances in LLMs, generating accurate and educationally viable geometric question-answer (Q&A) pairs remains challenging due to three persistent issues: (1) misapplication of geometric theorems in multi-step proofs, (2) diagram-text misalignment in problem formulation, and (3) inability to maintain answerability constraints during question generation. To address these limitations, we propose the TR-Reasoner to generate theoremgrounded Q&A pairs through coordinated interaction between geometric properties and structured reasoning chains (Fig. 4).

Description Patch Reasoning Fusion Building on the geometrically valid descriptions from TR-Engine, this module enforces logical coherence through causal dependencies between reasoning steps. Let $D = \{ p _ { 1 } , p _ { 2 } , . . . , p _ { x } \}$ denote the x description patches extracted from an image, where each patch $p _ { i }$ corresponds to a geometrically meaningful component (e.g., “Circle O with chord AB and tangent CD”). The single-step reasoning $r _ { i }$ for patch $p _ { i }$ is generated through theorem-constrained transformation:

$$
r _ { i } = \mathcal { F } _ { \mathrm { L L M } } ( p _ { i } | \boldsymbol { r } _ { < i } , \mathcal { T } ) ,\tag{1}
$$

where $r _ { < i } = \{ r _ { 1 } , . . . , r _ { { i - 1 } } \}$ represents preceding reasoning states, and  denotes the applicable theorem set (e.g., intersecting chords theorem for patch $p _ { i }$ describing chord intersections). This chained formulation ensures cumulative reasoning: later steps automatically inherit and extend prior conclusions (e.g., deriving arc lengths after establishing chord congruence).

Reverse Question Generation To prevent answerability drift, we implement answerconstrained reverse generation rather than open-ended question synthesis. Given a verified reasoning chain $R = \{ r _ { 1 } , r _ { 2 } , . . . , r _ { n } \}$ , each step $r _ { i }$ undergoes answerability assessment through a theorem-aware discriminator:

![](images/c75c59ce40c58b0110b4d7de9570ec2d114de7c1c6679d55a3562cad8566d50d.jpg)  
Figure 4: Overview of the TR-Reasoner. Image descriptions are segmented into patches to generate single-step reasoning results. Single-step reasoning results are fused progressively to get multi-step reasoning results. Then questions are generated based on the multi-step reasoning results. Finally, Q&A pairs that contradict geometric properties are filtered.

$$
f _ { a q } ( r _ { i } ) = \left\{ \begin{array} { l l } { f _ { q } ( r _ { i } ; \Phi _ { \mathrm { g e o } } ) , } & { \mathrm { i f } \ \mathcal { V } ( r _ { i } , G _ { \mathrm { p r o p s } } ) = \mathrm { T r u e } } \\ { \varnothing , } & { \mathrm { o t h e r w i s e } } \end{array} \right.\tag{2}
$$

where $G _ { \mathrm { p r o p s } }$ denotes geometric properties from TR-Engine (e.g., coordinate-derived lengths), performs theorem-based validation (e.g., checking triangle congruence rules), and $f _ { q }$ generates questions using a geometry-specialized LLM with instruction prompt $\Phi _ { \mathrm { g e o } }$ . This approach leverages the granular reasoning steps from the patch reasoning stage to generate theorem-aware Q&A pairs.

To comprehensively capture multi-level reasoning and ensure the quality of the dataset, we included Q&A pairs generated at all reasoning steps while implementing measures to eliminate redundancy and overly simplistic entries. Specifically: 1) Semantic Similarity Filtering: Using a pre-trained language model (BERT), we calculated the semantic similarity between Q&A pairs. Highly similar pairs were either merged or removed to reduce redundancy. 2) Length-Based Filtering: Simple Q&A pairs are often shorter in length. We set a minimum length threshold and excluded excessively short pairs that lacked sufficient information.

Error A&Q Filtering The final verification stage applies bidirectional cross-validation to ensure Q&A quality. The forward validation aligns generated answers with deterministic geometric properties computed by TR-Engine’s analytical algorithms, removing cases with (1) discrepancies between final answers and properties, and (2) inconsistencies in intermediate reasoning steps from verified properties. The reverse validation identifies ill-posed questions through semantic analysis, excluding those exhibiting answer ambiguity or logical indeterminacy. Both of the validation processes are conducted through single-round LLM inference, and only Q&A pairs that satisfy both verifications are reserved. Quantitative analysis revealed four main error patterns that were filtered out: Theorem Violation (36.3%): incorrect geometric principle application; Metric Discrepancy (24.9%): numerical inconsistency with problem constraints; Diagram-Text Mismatch (12.2%): references to non-existent diagram elements; and Answerability Ambiguity (26.6%): ill-defined problem statements.

Our proposed filtering mechanism can effectively reduce model hallucination and accumulate errors in previous reasoning steps. Among a sample of 200 generated Q&A pairs, the framework successfully suppresses reasoning error, reducing overall error rates from 16.0% (pre-validation) to 5.0% (post-validation). Showcases of invalid samples in Appendix D.

Context-Aware Prompt Engineering We deploy an instruction-based context-aware prompting strategy to optimize reasoning. We construct a reasoning instruction template pool containing prototypical geometric problems with a corresponding reasoning process. For each input, 3-4 optimal templates that are most relevant to the theorem and content is selected and integrated into the prompt. Additionally, the pool also contains a series of geometric relationships that are easily misunderstood by LLM. We use the same sample strategy to integrate them into the prompt as well, referred to as Basic Knowledge. The sampled instruction templates and basic knowledge serve as examples to assist the LLM to perform correct reasoning. Such context-aware prompt engineering ensures a relatively ideal reasoning accuracy, improving the efficiency of data generation. More details about the prompt strategy in Appendix B.

## 3.3 TR-GeoMM

Through the TR-CoT pipeline, we construct the TR-GeoMM dataset to enhance LMM’s geometric reasoning ability. From 15k figures, we obtain 45k high-quality Q&A pairs after error filtering, averaging 3.49 questions per figure. Detailed dataset statistics are shown in Fig. 6.

At the image level, TR-GeoMM covers 20 substrate shapes, mainly triangles, quadrilaterals, and circles. Unlike conventional polygon-based designs, TR-Engine builds figures from lines as primitive elements. It emphasizes key lines with geometric significance, e.g., midlines, angle bisectors, and radii, which frequently appear in theorems. As illustrated in Fig. 5 (a), 1.7k unique patterns are formed through theorem-guided line combinations, where each addition must interact with existing elements(e.g., a new line’s vertex must align with previously generated lines). At the text level, questions are categorized into four core types: side lengths, angles, areas, and geometric relationships. The hierarchical figure construction induces interdependent questions, where earlier solutions serve as prerequisites for subsequent ones. This subproblem design supports step-by-step learning of geometric concepts and reasoning. As shown in Fig. 5 (b), TR-GeoMM contains a theorem repository twice as large as existing synthetic datasets (MAVIS and GeomVerse). Furthermore, Fig. 5 (c) demonstrates superior data diversity through higher Q&A pair cosine distances. More information is provided in Appendix C and Appendix F.

## 3.4 TR-GeoSup

TR-CoT can not only generate reliable CoT geometric data but also be used to augment existing datasets. Real-world geometry CoTs often include key intermediate steps rich in problem-solving insights, yet these are typically implicit or oversimplified, relying on human prior knowledge. This lack of explicit reasoning may hinder model learning due to limited background knowledge and inference capability. Leveraging the TR-CoT pipeline, we decompose the original CoT process into explicit theorem-aware steps, then reverse generate new Q&A pairs with TR-Reasoner.

Specifically, our augmentation involves three steps: generating a comprehensive multi-step analysis of the geometric figure, segmenting it into essential problem-solving steps, and creating new Q&A pairs for each step. These fine-grained Q&A pairs explicitly guide the model with theorems and knowledge implicitly expressed in the original data, leading to improvement in comprehension and reasoning abilities. We applied TR-Reasoner to the GeoQA dataset, producing the TR-GeoSup dataset with 20k new entries. The final TR-GeoSup dataset does not contain the original GeoQA data. Examples of TR-GeoSup are shown in Appendix E.

During the augmentation, LLM receives the original question and its corresponding CoT answer to produce a more complete analysis, supplementing missing theorems and steps not explicitly stated in the original CoT. We sampled 200 examples from both the analysis and Q&A generation stages and observed no errors, confirming the reliability of our design. To streamline the data generation process, we did not introduce additional independent validation. After generation, 10% of the data was manually reviewed and corrected.

## 4 Experiments

## 4.1 Setup

We train multiple LMMs (Wang et al., 2024; Liu et al., 2024; Chen et al., 2024c) using existing geometric instruction datasets (Chen et al., 2021; Gao et al., 2023b) and our TR-CoT generated data (TR-GeoMM and TR-GeoSup). Both the projected linear layer and the language model are trainable. The models are trained for two epochs with a batch size of 128 on 16 64G NPU, and learning rate set to 5e-6. For evaluation, we assess these models on the geometry problem solving on the testmini set of MathVista (Lu et al., 2023) and GeoQA (Chen et al., 2021) following Gao et al. (2023b). Top-1 accuracy serves as the metric, with predictions and ground truth evaluated via ERNIE Bot 4.0. Ablation experiments were done on Intern-VL-2.0-8B.

![](images/d49a4f4e552bf968de35857c89c579eb195599662cc77cdae268ca6be1c777df.jpg)

![](images/6315241b98d32b3f1ba982d504fb87d6a314007c1ef3292b830e13bc95f4593a.jpg)

![](images/aab5daf32e7f32b9b2adb6e17fc637a3740163876fd257db5f78908034d63762.jpg)  
Figure 5: Diversity analysis of TR-GeoMM.

![](images/c1ea638e667bb47bfac9ad6fa89cfaafe7230d2502aa441357587cd1ee4da515.jpg)  
Figure 6: Statistical information about TR-GeoMM.

## 4.2 Ablation Study

Data generating procedures. To evaluate the contributions of TR-CoT components, we construct ablated variants by removing specific modules, as summarized in Tab. 1. Each variant is used to generate training data, and the resulting models are evaluated on MathVista and GeoQA. Generating Q&A pairs from descriptions yields better performance than from images, with gains of 5.3% on MathVista and 6.3% on GeoQA. Incorporating reverse generation further improves accuracy by 2.9% and 2.6% on the two datasets, respectively. The full TR-CoT pipeline achieves the best performance, confirming the effectiveness of each component.

Separate validity of synthetic and augmented data. We evaluated the impact of the TR-GeoSup and TR-GeoMM datasets on model performance, as shown in Tab. 2. Training with TR-GeoSup improved performance by 1.4% on MathVista and 7.9% on GeoQA compared to the baseline. Combining GeoQA with TR-GeoSup improves performance by 2.9% on MathVista and 3.9% on GeoQA compared to GeoQA alone, indicating their complementarity. It suggests that TR-GeoSup effectively enhances in-domain performance with better extracted knowledge. A deeper understanding of knowledge may contribute to improved generalization on mixed out-of-domain datasets.

Table 1: Ablation study on the data generating procedures. ‘Description’ represents generation based on descriptions. ‘Reverse’ represents generating reasoning followed by reverse question generation. ‘Filter’ represents filtering errors based on geometric properties.
<table><tr><td colspan="3">Configurations</td><td rowspan="2">MathVista</td><td rowspan="2">GeoQA</td></tr><tr><td>Description</td><td>Reverse</td><td>Filter</td></tr><tr><td>x</td><td>X</td><td>X</td><td>55.3</td><td>44.2</td></tr><tr><td>√</td><td>x</td><td>x</td><td>60.6</td><td>50.5</td></tr><tr><td>√</td><td>√</td><td>x</td><td>63.5</td><td>53.1</td></tr><tr><td>√</td><td>√</td><td>√</td><td>64.4</td><td>54.0</td></tr></table>

Table 2: Ablation study on the TR-CoT generated data.
<table><tr><td colspan="3">Configurations</td><td rowspan="2">MathVista</td><td rowspan="2">GeoQA</td></tr><tr><td>GeoQA</td><td>TR-GeoSup</td><td>TR-GeoMM</td></tr><tr><td>X</td><td>X</td><td>X</td><td>63.0</td><td>52.4</td></tr><tr><td>√</td><td>X</td><td>x</td><td>64.9</td><td>64.8</td></tr><tr><td>x</td><td>√</td><td>x</td><td>64.4</td><td>60.3</td></tr><tr><td>X</td><td>x</td><td>√</td><td>64.4</td><td>54.0</td></tr><tr><td>√</td><td>√</td><td>X</td><td>67.8</td><td>68.7</td></tr><tr><td>√</td><td>x</td><td>√</td><td>65.4</td><td>67.9</td></tr><tr><td>√</td><td>√</td><td>√</td><td>68.3</td><td>69.0</td></tr></table>

Second, training with TR-GeoMM improved performance by 1.4% on MathVista and 1.6% on GeoQA, confirming the strong generalization of TR-CoT synthetic data to real data. Moreover, joint training with GeoQA further improved performance, highlighting the effectiveness of synthetic data in supplementing real data. Finally, when jointly training on all three datasets (GeoQA, TR-GeoSup and TR-GeoMM). The model achieved the best performance, with improvements of 5.3% on MathVista and 6.6% on GeoQA over the baseline. These results support that TR-CoT-generated data compensate for the limitations of existing datasets and enhance the model’s reasoning capability.

![](images/4b5daf84ec808867a170781f749184a6570a3947c8364de933ec947186b99a27.jpg)  
Figure 7: Comparison of model problem solving before and after training.

Compared with other synthesis datasets. We train InternVL-2.0-8B using TR-GeoMM and two recent synthetic datasets for geometric problems, i.e., MAVIS (synthesis part) (Zhang et al., 2024b) and GeomVerse (Kazemi et al., 2023a), as summarized in Tab. 3. Compared to the baseline, models trained with GeomVerse or MAVIS show a slight performance gain on GeoQA and a decline on MathVista, both lower than TR-GeoMM. We attribute this to the limited diversity of image and Q&A pairs in these datasets, which benefits the simpler distribution of GeoQA but struggles with the diverse distributions in MathVista. In contrast, TR-GeoMM, with its diverse image and Q&A pairs, improves performance on both datasets.

Table 3: Compared with other synthesis datasets.
<table><tr><td>Dataset</td><td>MathVista</td><td>GeoQA</td></tr><tr><td>1</td><td>63.0</td><td>52.4</td></tr><tr><td>GeomVerse(9k)</td><td>58.2</td><td>53.6</td></tr><tr><td>MAVIS(sample 48k)</td><td>57.2</td><td>53.2</td></tr><tr><td>TR-GeoMM(45k)</td><td>64.4</td><td>54.0</td></tr><tr><td>TR-GeoMM(sample 9k)</td><td>63.0</td><td>55.6</td></tr></table>

## 4.3 Comparison with Previous State-of-the-Art

With the proposed method, we train three specialized models for geometry problem solving named TR-CoT-InternVL-2.0-2B, TR-CoT-Qwen2.5-VL-7B, and TR-CoT-InternVL-2.5-8B on the joint dataset of Geo170K and TR-CoT-generated data (TR-GeoMM and TR-GeoSup). We compare our models with both general and mathematical LMMs on the geometry problems from testmini set of MathVista and the test set of GeoQA. As shown in Tab. 4, TR-CoT-InternVL-2.5-8B outperforms GPT-4o by 17.3% on MathVista and TR-CoT-Qwen2.5-VL-7B outperforms GPT-4o by 17.8% on GeoQA. Compared to mathematical LMMs, TR-CoT-InternVL-2.5-8B maintains a 11.1% lead on MathVista, and TR-CoT-Qwen2.5-VL-7B achieves a 12.5% advantage on GeoQA. For performance analysis on more baselines and benchmarks, please refer to Appendix H.

Table 4: Top-1 Accuracy (%) on geometry problem solving on the testmini set of MathVista and the GeoQA test set. \* represents the results from the existing papers.
<table><tr><td>Model</td><td>MathVista</td><td>GeoQA</td></tr><tr><td colspan="3">Closed-source LMMs</td></tr><tr><td>GPT-4o (Islam and Moushi, 2024)</td><td>60.6</td><td>61.4</td></tr><tr><td>GPT-4V</td><td>51.0*</td><td>43.4*</td></tr><tr><td>Gemini Ultra (Team et al., 2023)</td><td>56.3*</td><td></td></tr><tr><td colspan="3">Open-source LMMs</td></tr><tr><td>LLaVA2-13B (Liu et al., 2024)</td><td>29.3*</td><td>20.3*</td></tr><tr><td>mPLUG-Owl2-7B (Ye et al., 2024)</td><td>25.5</td><td>21.4</td></tr><tr><td>Qwen-VL-Chat-7B (Bai et al., 2023)</td><td>35.6</td><td>26.1</td></tr><tr><td>Monkey-Chat-7B (Li et al., 2024)</td><td>24.5</td><td>28.5</td></tr><tr><td>Deepseek-VL-7B (Lu et al., 2024)</td><td>34.6</td><td>33.7</td></tr><tr><td>InternVL-2.0-2B (Chen et al., 2024c)</td><td>46.2</td><td>38.2</td></tr><tr><td>InternLM-XC2-7B (Zhang et al., 2023b)</td><td>51.4</td><td>44.7</td></tr><tr><td>InternVL-1.5-20B (Chen et al., 2024b)</td><td>60.1</td><td>49.7</td></tr><tr><td>Qwen2-VL-7B (Wang et al., 2024)</td><td>55.1</td><td>55.7</td></tr><tr><td>InternVL-2.0-8B (Chen et al., 2024c)</td><td>65.9</td><td>56.5</td></tr><tr><td>InternVL-2.5-8B (Chen et al., 2024a)</td><td>67.8</td><td>59.0</td></tr><tr><td>Qwen2.5-VL-7B (Wang et al., 2024)</td><td>71.6</td><td>74.5</td></tr><tr><td colspan="3">Open-source Mathematical LMMs</td></tr><tr><td>UniMath (Liang et al., 2023b)</td><td></td><td>50.0*</td></tr><tr><td>Math-LLaVA-13B (Shi et al., 2024)</td><td>56.5*</td><td>47.8</td></tr><tr><td>G-LLaVA-7B (Gao et al., 2023b)</td><td>53.4*</td><td>62.8*</td></tr><tr><td>MAVIS-7B (Zhang et al., 2024b)</td><td></td><td>66.7*</td></tr><tr><td>PUMA-Qwen2-7B (Zhuang et al., 2024)</td><td>48.1*</td><td>一</td></tr><tr><td>MultiMath-7B (Peng et al., 2024)</td><td>66.8*</td><td></td></tr><tr><td>TR-CoT-InternVL-2.0-2B</td><td>56.3</td><td>63.4</td></tr><tr><td>TR-CoT-Qwen2.5-VL-7B</td><td>74.5</td><td>79.2</td></tr><tr><td>TR-CoT-InternVL-2.5-8B</td><td>77.9</td><td>76.7</td></tr></table>

## 5 Discussion

Fig. 7 highlights consistent improvements: posttrained models produce concise, logical CoTs with accurate conclusions, demonstrating robust geometric understanding. Pre-trained models show recurring errors (e.g., misdefining isosceles triangles as having two equal sides), reflecting foundational gaps in theorem comprehension. Our approach trains models on diverse theorems with structured reasoning, addressing these errors and enhancing general geometric problem-solving.

![](images/c4410af509964fc73f5379c3f4b352abd9332e3b2030efdc189de3bf900ad1cc.jpg)

![](images/c30aa05fafcbdaff86ddb06da4160bb8c483351915eba136b2baa2c4fb5ef279.jpg)  
Figure 8: Comparison of model output quality and token length before and after training.

We use DeepSeek R1 and ERNIE Bot 4.0 to quantitatively evaluate model outputs before and after training, focusing on logical consistency, clarity, and lack of ambiguity (see Appendix I for detailed information). We use the average score of the two models as the final score. As shown in Fig. 8 (a), the total mean score increased by 0.37 after training, the mean score for correct answers increased by 0.70, and outputs with scores of 8 or higher increased by 24.5%. We attribute these improvements to TR-CoT’s explicit focus on the reasoning process, where step decomposition enhances the model’s logical consistency and rigor.

We further compare the token usage for correct answers before and after training. As shown in Fig. 8 (b), the model after training requires fewer tokens on average, with the percentage of correct answers within 200 tokens increasing by 35%. We assume this improvement results from the data diversity, which enables the model to find more efficient solutions across different theorems, while a deeper understanding of the theorems allows for more concise reasoning.

## 6 Conclusion

We introduce TR-CoT, a theorem-validated reverse chain-of-thought framework that generates logically consistent geometric reasoning data. By combining theorem-driven diagram-property synthesis with reverse reasoning validation, TR-CoT expands theorem coverage, enhances fine-grained theorem understanding, and effectively corrects common reasoning errors in LMMs. Models trained on our TR-CoT generated data achieve substantial gains on MathVista and GeoQA, surpassing strong openand closed-source baselines. In future work, we expect to explore the role of theorem-aware reasoning further beyond multimodal geometry problem

solving.

## Limitations

For our method, one major constraint is that there is still room for further improvement in the generation efficiency. The overall efficiency can be divided into time efficiency and data efficiency. First, in our process, LLM is called multiple times for reasoning generation. The limited reasoning speed of LLM becomes the bottleneck of time efficiency. In addition, although we have adopted various methods to improve the reasoning accuracy of LLM, due to the limitations of model performance, there is still a certain proportion of errors in the direct output of the model. We observe that about 10% of the direct output is deleted in the Error A&Q Filtering stage.

## Acknowledgments

This research was supported by the NSFC (62576147 and 62225603).

## References

Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966.

Jiaqi Chen, Tong Li, Jinghui Qin, Pan Lu, Liang Lin, Chongyu Chen, and Xiaodan Liang. 2022. Unigeo: Unifying geometry logical reasoning via reformulating mathematical expression. Preprint, arXiv:2212.02746.

Jiaqi Chen, Jianheng Tang, Jinghui Qin, Xiaodan Liang, Lingbo Liu, Eric Xing, and Liang Lin. 2021. Geoqa: A geometric question answering benchmark towards multimodal numerical reasoning. In Findings of the Associationfor Computational Linguistics: ACL-IJCNLP 2021, pages 513–523.

Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, and 1 others. 2024a. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271.

Zhe Chen, Weiyun Wang, Hao Tian, Shenglong Ye, Zhangwei Gao, Erfei Cui, Wenwen Tong, Kongzhi Hu, Jiapeng Luo, Zheng Ma, and 1 others. 2024b. How far are we to gpt-4v? closing the gap to commercial multimodal models with open-source suites. arXiv preprint arXiv:2404.16821.

Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, and 1 others. 2024c. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198.

Jacob Devlin. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

Aniket Didolkar, Anirudh Goyal, Nan Rosemary Ke, Siyuan Guo, Michal Valko, Timothy Lillicrap, Danilo Rezende, Yoshua Bengio, Michael Mozer, and Sanjeev Arora. 2024. Metacognitive capabilities of llms: An exploration in mathematical problem solving. Preprint, arXiv:2405.12205.

Yue Fan, Jing Gu, Kaiwen Zhou, Qianqi Yan, Shan Jiang, Ching-Chen Kuo, Yang Zhao, Xinze Guan, and Xin Wang. 2024. Muffin or chihuahua? challenging multimodal large language models with multipanel vqa. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 6845–6863.

Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, and Lingpeng Kong. 2023a. G-llava: Solving geometric problem with multi-modal large language model. Preprint, arXiv:2312.11370.

Jiahui Gao, Renjie Pi, Jipeng Zhang, Jiacheng Ye, Wanjun Zhong, Yufei Wang, Lanqing Hong, Jianhua Han, Hang Xu, Zhenguo Li, and 1 others. 2023b. G-llava: Solving geometric problem with multi-modal large language model. arXiv preprint arXiv:2312.11370.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, and 1 others. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948.

Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A Smith, and Ranjay Krishna. 2024. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. Preprint, arXiv:2406.09403.

Raisa Islam and Owana Marzia Moushi. 2024. Gpt-4o: The cutting-edge advancement in multimodal llm. Authorea Preprints.

Weisen Jiang, Han Shi, Longhui Yu, Zhengying Liu, Yu Zhang, Zhenguo Li, and James T. Kwok. 2024. Forward-backward reasoning in large language models for mathematical verification. Preprint, arXiv:2308.07758.

Mehran Kazemi, Hamidreza Alvari, Ankit Anand, Jialin Wu, Xi Chen, and Radu Soricut. 2023a. Geomverse: A systematic evaluation of large models for geometric reasoning. arXiv preprint arXiv:2312.12241.

Mehran Kazemi, Hamidreza Alvari, Ankit Anand, Jialin Wu, Xi Chen, and Radu Soricut. 2023b. Geomverse: A systematic evaluation of large models for geometric reasoning. Preprint, arXiv:2312.12241.

Zhang Li, Biao Yang, Qiang Liu, Zhiyin Ma, Shuo Zhang, Jingxu Yang, Yabo Sun, Yuliang Liu, and Xiang Bai. 2024. Monkey: Image resolution and text label are important things for large multi-modal models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26763–26773.

Yuanyuan Liang, Jianing Wang, Hanlun Zhu, Lei Wang, Weining Qian, and Yunshi Lan. 2023a. Prompting large language models with chain-of-thought for fewshot knowledge base question generation. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 4329– 4343, Singapore. Association for Computational Linguistics.

Zhenwen Liang, Tianyu Yang, Jipeng Zhang, and Xiangliang Zhang. 2023b. Unimath: A foundational and multimodal mathematical reasoner. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7126–7133.

Haoran Liao, Jidong Tian, Shaohua Hu, Hao He, and Yaohui Jin. 2024. Look before you leap: Problem elaboration prompting improves mathematical reasoning in large language models. Preprint, arXiv:2402.15764.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2024. Visual instruction tuning. Advances in neural information processing systems, 36.

Haoyu Lu, Wen Liu, Bo Zhang, Bingxuan Wang, Kai Dong, Bo Liu, Jingxiang Sun, Tongzheng Ren, Zhuoshu Li, Hao Yang, and 1 others. 2024. Deepseek-vl: Towards real-world vision-language understanding. CoRR.

Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. 2023. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In The 3rd Workshop on Mathematical Reasoning and AI at NeurIPS’23.

Pan Lu, Ran Gong, Shibiao Jiang, Liang Qiu, Siyuan Huang, Xiaodan Liang, and Song-Chun Zhu. 2021. Inter-gps: Interpretable geometry problem solving with formal language and symbolic reasoning. Preprint, arXiv:2105.04165.

OpenAI. 2024. Openai o1 system card. preprint.

Shuai Peng, Di Fu, Liangcai Gao, Xiuqin Zhong, Hongguang Fu, and Zhi Tang. 2024. Multimath: Bridging visual and mathematical reasoning for large language models. arXiv preprint arXiv:2409.00147.

Minjoon Seo, Hannaneh Hajishirzi, Ali Farhadi, Oren Etzioni, and Clint Malcolm. 2015. Solving geometry problems: Combining text and diagram interpretation. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 1466–1476, Lisbon, Portugal. Association for Computational Linguistics.

Wenhao Shi, Zhiqiang Hu, Yi Bin, Junhua Liu, Yang Yang, See-Kiong Ng, Lidong Bing, and Roy Ka-Wei Lee. 2024. Math-llava: Bootstrapping mathematical reasoning for multimodal large language models. arXiv preprint arXiv:2406.17294.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, and 1 others. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.

Lei Wang, Wanyu Xu, Yihuai Lan, Zhiqiang Hu, Yunshi Lan, Roy Ka-Wei Lee, and Ee-Peng Lim. 2023. Planand-solve prompting: Improving zero-shot chain-ofthought reasoning by large language models. In Proceedings of the 61st Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 2609–2634, Toronto, Canada. Association for Computational Linguistics.

Peijie Wang, Zhong-Zhi Li, Fei Yin, Xin Yang, Dekang Ran, and Cheng-Lin Liu. 2025. Mv-math: Evaluating multimodal math reasoning in multi-visual contexts. arXiv preprint arXiv:2502.20808.

Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, and 1 others. 2024. Qwen2- vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191.

Yixuan Weng, Minjun Zhu, Fei Xia, Bin Li, Shizhu He, Shengping Liu, Bin Sun, Kang Liu, and Jun Zhao. 2023. Large language models are better reasoners with self-verification. Preprint, arXiv:2212.09561.

Tianci Xue, Ziqi Wang, Zhenhailong Wang, Chi Han, Pengfei Yu, and Heng Ji. 2023. Rcot: Detecting and rectifying factual inconsistency in reasoning by reversing chain-of-thought. Preprint, arXiv:2305.11499.

Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, and Fei Huang. 2024. mplug-owl2: Revolutionizing multimodal large language model with modality collaboration. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13040–13051.

Jiahao Yuan, Dehui Du, Hao Zhang, Zixiang Di, and Usman Naseem. 2024. Reversal of thought: Enhancing large language models with preferenceguided reverse reasoning warm-up. Preprint, arXiv:2410.12323.

Jiaxin Zhang, Zhong-Zhi Li, Ming-Liang Zhang, Fei Yin, Cheng-Lin Liu, and Yashar Moshfeghi. 2024a. GeoEval: Benchmark for evaluating LLMs and multimodal models on geometry problem-solving. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 1258–1276, Bangkok, Thailand. Association for Computational Linguistics.

Ming-Liang Zhang, Fei Yin, and Cheng-Lin Liu. 2023a. A multi-modal neural geometric solver with textual clauses parsed from diagram. In Proceedings ofthe Thirty-Second International Joint Conference on Artificial Intelligence, pages 3374–3382.

Pan Zhang, Xiaoyi Dong Bin Wang, Yuhang Cao, Chao Xu, Linke Ouyang, Zhiyuan Zhao, Shuangrui Ding, Songyang Zhang, Haodong Duan, Hang Yan, and 1 others. 2023b. Internlm-xcomposer: A vision-language large model for advanced text-image comprehension and composition. arXiv preprint arXiv:2309.15112.

Renrui Zhang, Xinyu Wei, Dongzhi Jiang, Yichi Zhang, Ziyu Guo, Chengzhuo Tong, Jiaming Liu, Aojun Zhou, Bin Wei, Shanghang Zhang, and 1 others. 2024b. Mavis: Mathematical visual instruction tuning. arXiv preprint arXiv:2407.08739.

Xueliang Zhao, Xinting Huang, Tingchen Fu, Qintong Li, Shansan Gong, Lemao Liu, Wei Bi, and Lingpeng Kong. 2024a. Bba: Bi-modal behavioral alignment for reasoning with large vision-language models. Preprint, arXiv:2402.13577.

Zilong Zhao, Yao Rong, Dongyang Guo, Emek Gö- zlüklü, Emir Gülboy, and Enkelejda Kasneci. 2024b. Stepwise self-consistent mathematical reasoning with large language models. Preprint, arXiv:2402.17786.

Aojun Zhou, Ke Wang, Zimu Lu, Weikang Shi, Sichun Luo, Zipeng Qin, Shaoqing Lu, Anya Jia, Linqi Song, Mingjie Zhan, and Hongsheng Li. 2023. Solving challenging math word problems using gpt-4 code interpreter with code-based self-verification. Preprint, arXiv:2308.07921.

Wenwen Zhuang, Xin Huang, Xiantao Zhang, and Jin Zeng. 2024. Math-puma: Progressive upward multimodal alignment to enhance mathematical reasoning. arXiv preprint arXiv:2408.08640.

## A Pseudo Code

We have written pseudo-code for the overall flow of TR-CoT, the details of which are given in Algor. 1.

Algorithm 1: Pseudo-code of TR-CoT   
Input: Geometry substrates sampling rounds n, plot   
function ${ \bf { \bar { \alpha } } } _ { f , \ }$ image-description pair sets $s ,$ , line   
sampling rounds $k ,$ geometric property   
calculation module $\nu ,$ large language model   
$\mathcal { M }$   
Output: Generated Image $\mathcal { T } ,$ Description $\mathcal { D } ,$   
Geometric Properties , Question $\mathcal { Q } ;$   
Answer $\mathcal { A }$   
1 Initialization: $\mathcal { T }  \emptyset , \mathcal { D }  \emptyset , \mathcal { T }  \emptyset ,$ vertex   
coordinate ${ \mathcal { C } } \gets \emptyset , { { r } _ { s } } \gets \emptyset$   
2 for $i \gets I$ to n do   
3 Sample geometry substrate $\mathcal { G } _ { i }$ and description   
$\mathcal { D } _ { i }$ from image-description pair sets   
4 Refresh $\mathcal { T }$ using plot function: ${ \mathcal { T } } \gets f ( { \mathcal { T } } , { \mathcal { G } } _ { i } )$   
5 Refresh corresponding description:   
$\mathcal { D }  \mathcal { D } \cup \mathcal { D } _ { i }$   
6 Refresh vertex coordinate: $c  { \mathcal { C } } \cup { \mathcal { C } } _ { i }$   
7 end   
8 for j 1 to k do   
9 Select line drawing position $\mathcal { P } _ { j }$   
10 Draw line and label length: $\mathcal { T }  f ( \mathcal { T } , \mathcal { P } _ { j } )$   
11 Refresh corresponding description:   
$\mathcal { D }  \mathcal { D } \cup \mathcal { P } _ { j } ^ { \cdot }$   
12 Refresh vertex coordinate: ${ \mathcal { C } } \gets { \mathcal { C } } \cup { \mathcal { C } } _ { i }$   
13 $\mathbf { i f } ~ j = k$ then   
14 Calculate all angle information $\mathcal { R }$   
15 Draw angles and label degrees:   
${ \mathcal { T } } \gets f ( { \mathcal { T } } , { \mathcal { R } } )$   
16 Refresh corresponding description:   
$\mathcal { D }  \mathcal { D } \cup \mathcal { R }$   
17 end   
18 end   
19 Refresh Geometric Properties: $\tau  \mathcal { V } ( C )$   
20 Produce single-step reasoning result $r _ { c }$ using prompt   
$P _ { s } \colon r _ { c } \gets \mathsf { \bar { M } } ( \hat { T } , \hat { P } _ { s } )$   
21 Generate answer $A _ { e }$ and its corresponding question   
$Q _ { e }$ using prompt $P _ { q } \colon A _ { e } , Q _ { e } \gets \mathsf { \bar { M } } ( r _ { c } , \bar { P _ { q } } )$   
22 Filtering for correct answer A and its corresponding   
question $Q$ using prompt $P _ { e } \colon$   
$\hat { A } , Q \gets \dot { \mathcal { M } } ( A _ { e } , \mathsf { \bar { Q } } _ { e } , \hat { T } , P _ { e } )$   
23 Return: , , ,

## B Details of prompt in TR-Reasoner

We used ERNIE Bot 4.0 to implement TR-Reasoner. We describe the prompts used in TR-Reasoner, including the prompts for the Description Patch Reasoning Fusion (Fig. 9), the Reverse Question Generation (Fig. 10), and the Error A&Q Filtering (Fig. 11). In these figures, the texts in blue represent the Task Description, while the texts in orange represent the input information. Each prompt includes three contextual examples, and we show only one of them, with the remaining examples replaced by ellipses. In addition to the examples, some prompts also include an instruction section that specifies more detailed requirements, some incorporate additional basic knowledge, and others outline more specific goals that must be achieved.

Description Patch Reasoning Fusion Prompt   
Assume expertise in geometry. Perform accurate inferences based on image   
descriptions, summarize analysis, and provide logical, correct reasoning for   
each level of description based on the provided data.   
Instructions:   
1. Inference Process: Analyze each description step by step and derive conclusions.   
2. Relevance: If a description is irrelevant, state why and ignore it.   
3. Output Format: Present inference as numbered points, concise and precise   
Input:   
Step 1: In triangle ABC, length of side AB is 4, length of side AC is 6,   
length of side BC is 6.Step 2: CD is the bisector of angle ACB, D lies on   
line segment AB. Step 3: E is the midpoint of CD, connect E and B   
Output:   
Reasoning 1: Triangle ABC has sides $\mathbf { A B } = 4 , \mathbf { A C } = 6 , \mathbf { B C } = 6 .$ Since AC =   
BC, triangle ABC is isosceles. The perimeter of triangle ABC is AB + AC +   
BC = 4 + 6 + 6 = 16.Reasoning 2: CD bisects angle ACB, forming 2 equa   
angle ACD and BCD. Also, according to the property of isosceles triangle,   
CD is also Perpendicular bisector of AB. We can know $\mathbf { A D } = \mathbf { B D } = 0 . 5$   
\times AB = 2, and using the Pythagorean theorem, we get CD = √(AC^2 -   
AD^2) CD = 4 √ 2. Reasoning 3: As E being the midpoint of CD, CE = ED   
= 0.5 \times C ${ \bf D } = 2 \mathrm { ~ } \sqrt { \mathrm { ~ } } 2 .$ Connecting BE form a new triangle BDE, and   
angle BDE = 90 degrees as CD is perpendicular to AB. Using the   
Pythagorean theorem, we can get BE = √(BD^2 + DE^2) = 2\sqrt(3)   
Input: [DESCRIPTION]   
Output: [INFERENCE]   
Basic knowledge:   
1. sin(15 °) =cos(75 °) ≈ 0.2588 , cos(15 °)= sin(75 °) ≈ 0.9659.   
2. The radius of circle equals the side length of its inscribed regular hexagon.   
4. In a hexagon, diagonals CA, AC, EA, AE, DB, BD, FB, BF, EC, CE are   
√ 3\*side length; diagonals DA, AD, EB, BE, FC, CF are 2 \*side length.   
5. In a square, the radius of the circle is r, and the side length of the   
inscribed square is √ 2\*r.   
6. In an equilateral triangle, the radius of the circle is r , and the side length   
of the inscribed triangle is √ 3\*r.  
Figure 9: The prompt of the Description Patch Reasoning Fusion.

In preliminary experiments, we observed that LLMs often failed to accurately interpret certain geometric relationships. To systematically identify such issues, we selected 50 representative instances per geometric substrate from the TR-GeoMM dataset and applied the TR-Reasoner framework for Patch Reasoning. We analyzed the most frequently misinterpreted relationships and formalized their correct representations into a base knowledge library. During formal generation, prompts are dynamically constructed by retrieving relevant geometric relationships from this library based on the target substrate.

## C More information of TR-GeoMM

Through the TR-CoT, we construct a high-quality geometric dataset, TR-GeoMM. In Fig. 12, we provide a detailed overview of specific cases from TR-GeoMM. These cases demonstrate the variety of mathematical geometry question types covered by TR-GeoMM, including solving for lengths, angles, areas, and geometry elemental relations. Each of these categories is critical for improving the geometric reasoning ability of LMMs.

For Cosine distance based data diversity, we first randomly sample 5000 instances from each dataset(MAVIS, GeomVerse, and TR-GeoMM), then we encode the instances into embedding features using pretrained BERT model (Devlin, 2018). Finally, we calculate the average cosine distance of each dataset using the BERT output features. Higher distance score indicates better diversity, and our TR-GeoMM has the highest distance score among the three datasets.

![](images/5b34f08dc2d1cbd89ac88a1fbcebfc52b53fdb6acae0244b273df20f48f1f025.jpg)  
Figure 10: The prompt of the Reverse Question Generation.

![](images/e3432b985378725e133b9131d773db3ffb6d6310c03bccc31ce3485c8d6dc245.jpg)  
Figure 11: The prompt of the Error A&Q Filtering.

![](images/8913271e2fbbc265e71fb87c93ea1c55c1e8a3fa26974d89969d9a3affcc6f5f.jpg)  
Figure 12: Examples of TR-GeoMM dataset.

We further report the computation and generation cost of the generation pipeline here. Geometric images, descriptions, and properties are all generated simultaneously by the TR-Engine, our Python-built graphics rendering engine. A total of 15,000 geometric images were generated during the geometric image generation phase, and it could be done in approximately 10 minutes by a 12th-gen Intel Core i7-12700. TR-Reasoner, our LLM-based module for Patch Reasoning, Q&A generation, and error filtering, runs via cloud APIs with 32 parallel processes. These stages take approximately 6 hours in total ( 2 hours per stage). As a one-time cost, the generated data can be reused across model training tasks.

## D Qualitative examples of filtered errors

Fig. 13 presents four representative types of errors identified by the Error A&Q Filtering module. Theorem Violation refers to cases where conclusions or assumptions contradict established mathematical theorems. Metric Discrepancies involve inconsistencies between the given numerical values or angles and the geometric properties. Diagramtext Mismatches occur when elements described in the problem statement are either absent from the diagram or inconsistent with it. Ambiguous Answerability denotes problems in which the information provided is insufficient to derive a unique solution, or essential data is not explicitly stated in the question.

Table 5: Statistic comparison between Geo170K, Geom-Verse, and our data. En. Exis means Enhance Existing data. Fully syn. means Fully synthesized.‘/’indicates the same number as GeoQA
<table><tr><td>Dataset name</td><td>Data type</td><td>Img.</td><td>Q&amp;A</td><td>Theorem</td></tr><tr><td>Geo170K</td><td>En. Exis.</td><td>6.4K</td><td>110K</td><td>1</td></tr><tr><td>GeomVerse</td><td>Fully syn.</td><td>9.3K</td><td>9.3K</td><td>60</td></tr><tr><td>TR-GeoMM</td><td>Fully syn.</td><td>15K</td><td>45K</td><td>110</td></tr><tr><td>TR-GeoSup</td><td>En. Exis.</td><td>6.4K</td><td>20K</td><td>1</td></tr></table>

## E Examples of TR-GeoSup dataset

Fig. 14 illustrates an example from the TR-GeoSup dataset, showcasing the transformation of a multistep reasoning problem from the original GeoQA dataset. In the original Q&A pair, the reasoning process is condensed and lacks explicit intermediate steps, relying on implicit knowledge. TR-GeoSup decomposes the original reasoning process into three hierarchical sub-questions, each accompanied by a detailed and theorem-aware reasoning chain. This augmentation not only clarifies the implicit knowledge embedded in the original data but also provides a step-by-step guide for model training.

## F Statistical Comparison With Related Datasets

Here we make a brief comparison between our proposed dataset with some related academic datasets.

• GeomVerse is a representative template-based method that generates geometrically oversimplified images by combining predefined polygons in fixed configurations. These images only contain polygon compositions and lack theorem-aware elements (e.g., midlines and angle bisectors). It has 9.3k synthetic images accompanied by Q&A pairs, but their richness was limited by the absence of theorem-aware elements, covering only 60 theorems.

• Geo170K represents an augmented version of the existing GeoQA dataset. It primarily focuses on rephrasing Q&A pairs, such as altering wording, swapping conditions and answers, or scaling numerical values while keeping the underlying theorems identical. This approach does not enhance the diversity of theorems covered in the dataset.

• TR-CoT enables theorem-driven multimodal reasoning by designing substrates and embedding theorem-aware elements based on theorem conditions, allowing generated images to support complex Q&A construction. Unlike prior approaches, TR-CoT is not limited by existing data coverage and can expand a model’s geometric knowledge. The framework supports both new data synthesis and augmentation of existing datasets, and covers 110 theorems through its structured generation process.

As shown in Tab. 5, our data possess a notable diversity in theorem coverage, image distribution, and Q&A quantity. Ablation studies in Section 4.2 further discuss the training effectiveness of our proposed data.

## G Robustness of polygon distribution

We conducted robustness experiments for different polygon distributions, where the details of the polygon distributions are shown in Tab. 6. From top to bottom, the percentage of triangles and quads gradually decreases, and the percentage of pentagons and hexagons gradually increases. There is also a clear difference in the percentage of circles.

Similar quantitative results within 0.6% in Tab. 7 show that the impact of polygon distributions is almost negligible, demonstrating the strong robustness of our method to different polygon distributions. Therefore, the performance gain is mainly attributed to the diverse geometry representation and reasoning knowledge provided by our method.

Table 6: Details of polygon distribution for distributional robust ablation studies.
<table><tr><td rowspan="2">Method</td><td colspan="5">Polygon Distribution</td></tr><tr><td>triangle quad</td><td></td><td>circle</td><td>pentagon</td><td>hexagon</td></tr><tr><td>Group I</td><td>29%</td><td>46%</td><td>17%</td><td>5%</td><td>3%</td></tr><tr><td>Group II</td><td>32%</td><td>40%</td><td>14%</td><td>8%</td><td>6%</td></tr><tr><td>Group III</td><td>25%</td><td>33%</td><td>21%</td><td>12%</td><td>8%</td></tr></table>

Table 7: Ablation study on the robustness to polygonal distributions.
<table><tr><td>Polygon Distribution</td><td>MathVista</td><td>GeoQA</td></tr><tr><td>Group I</td><td>64.4</td><td>54.0</td></tr><tr><td>Group II</td><td>64.4</td><td>53.7</td></tr><tr><td>Group III</td><td>63.9</td><td>53.4</td></tr></table>

![](images/4f7cdbc707cdaf77936e4dbbe3cdbbcef3b4a3e15ecc1c1fd07c3f3aa0902da5.jpg)  
Figure 13: Examples of filtered errors.

![](images/490d14ba617cc7a0e922472e16d32eb6fe4569c5932a328b3dd7961bcff25c31.jpg)  
Figure 14: Examples of TR-GeoSup dataset.

## H Effectiveness of TR-CoT

As shown in Tab. 8, models jointly trained on Geo170K and TR-CoT-generated data (TR-GeoMM and TR-GeoSup) consistently outperform those trained solely on Geo170K (‘Geo-’). InternVL2.5-8B receives 1.5% improvements on MathVista and GeoQA, and Qwen2.5-VL-7B improves by 1.0% and 2.0% on MathVista and GeoQA, respectively. These results indicate that TR-CoT-generated data can supplement existing datasets and is widely effective in various LMMs.

We further evaluate model performance on more diverse benchmarks, i.e., MathVerse, MathVision

Table 8: TR-CoT generated data effectiveness validation on different models. ‘Geo-’ indicates the model is finetuned only with geometric instruction data of Geo170K. Consistent and significant improvement without adding any additional parameters.

<table><tr><td>Model</td><td>MathVista</td><td>GeoQA</td></tr><tr><td>Geo-InternVL-2.0-2B TR-CoT-InternVL-2.0-2B</td><td>51.9 56.3 (4.4↑)</td><td>62.5 63.4 (0.9↑)</td></tr><tr><td>Geo-LLaVA-1.5-7B TR-CoT-LLaVA-7B</td><td>27.9 29.3 (1.4↑)</td><td>47.6 51.7 (4.1↑)</td></tr><tr><td>Geo-Qwen2-VL-7B TR-CoT-Qwen2-VL-7B</td><td>59.9 67.6 (7.7↑)</td><td>69.1 70.4 (1.3↑)</td></tr><tr><td>Geo-InternVL-2.0-8B TR-CoT-InternVL-2.0-8B</td><td>70.2 72.1 (1.9↑)</td><td>74.9 76.7 (1.8↑)</td></tr><tr><td>Geo-InternVL-2.5-8B TR-CoT-InternVL-2.5-8B</td><td>76.4 77.9 (1.5↑)</td><td>75.2 76.7 (1.5↑)</td></tr><tr><td>Geo-Qwen2.5-VL-7B TR-CoT-Qwen2.5-VL-7B</td><td>73.5 74.5 (1.0↑)</td><td>77.2 79.2 (2.0↑)</td></tr></table>

and WeMath, to investigate the generalizability of TR-CoT generated data. As shown in the Tab. 9, models of various scale trained with TR-CoT demonstrate an average performance improvement compared to the baseline, which validates the potential generalization ability of TR-CoT on mathematical tasks beyond geometry problem solving.

## I Details of CoT quality evaluation

We used ERNIE Bot 4.0 and DeepSeek R1 to evaluate model outputs. For each response, the evaluation model gives a score between 0 and 10 to judge the logical consistency, clarity, and lack of ambiguity. We use the average score of the two models as the final score. To ensure a more accurate evaluation, we include specific judging standards. The prompts used are shown in Fig. 15. The blue part represents the Task Description.

Table 9: TR-CoT generated data effectiveness validation on more diverse math-related benchmarks. ‘+TR-CoT indicates model fine-tuned on TR-CoT generated data. ‘∆’ denotes the relative change in accuracy.
<table><tr><td>Model</td><td>MathVerse</td><td>MathVision</td><td>WeMath</td><td>Total</td></tr><tr><td>InternVL-2.0-2B</td><td>19.2</td><td>7.5</td><td>32.9</td><td>59.6</td></tr><tr><td>+TR-CoT</td><td>24.3</td><td>12.3</td><td>36.5</td><td>73.1</td></tr><tr><td>∆</td><td>5.1↑</td><td>4.8↑</td><td>3.6↑</td><td>13.5↑</td></tr><tr><td>Qwen2.5-VL-7B</td><td>42.6</td><td>25.7</td><td>63.1</td><td>131.4</td></tr><tr><td>+TR-CoT</td><td>45.7</td><td>24.8</td><td>61.3</td><td>131.8</td></tr><tr><td>∆</td><td>3.1↑</td><td>0.9↓</td><td>1.8↓</td><td>0.4↑</td></tr><tr><td>InternVL-2.5-2B</td><td>40.1</td><td>17.0</td><td>61.3</td><td>118.4</td></tr><tr><td>+TR-CoT</td><td>41.8</td><td>20.0</td><td>59.9</td><td>121.7</td></tr><tr><td>∆</td><td>1.7个</td><td>3.0↑</td><td>1.4↓</td><td>3.3↑</td></tr></table>

Quality Judgment Prompt   
You are provided with a language model’s response to a geometric question.   
Your mission is to judge the quality of the response based on the following   
standards, and give a score between 0 to 10.   
Judging Standards:   
1.Logic consistency. Assess whether the response is self-consistent,   
logically coherent, and free from contradictions or illogical reasoning.   
2.Clarity. Evaluate whether the response is clear and easy to understand,   
avoiding ambiguity or vague expressions。   
3.Outputformat: Score: your score(from 0 to 10)  
Figure 15: Comparison of model problem solving before and after training.

## J The Case of Direct Generation and TR-Reasoner Generation

The core idea of the TR-Reasoner is to improve the accuracy of Q&A pairs by simplifying the reasoning based on descriptions and then generating corresponding questions from the answers in a reversed manner. A straightforward approach is directly prompting ERNIE Bot 4.0 to generate Q&A pairs from the input image description. However, as shown on the left of Fig. 16, this approach often fails to determine the correct answer. In contrast, the Q&A pairs produced by TR-Reasoner are correct for all three instances with our design.

## K Details of the theorems

The support of mathematical theorems is crucial for the accuracy of TR-Engine. In Tab. 10, we present the geometric theorems and properties that we used. These define the rules for combining elements, establishing a logically coherent chain throughout the figure construction process. They serve as the foundation for extending reasoning scenarios and also assist in the computation and verification of question-answer pairs.

We collect and organize geometric theorems through three main approaches: (1) Systematic Textbook Mining: We analyzed standard textbooks and online educational resources to compile core geometric axioms and theorems from primary and secondary school mathematics curricula in Mainland China. (2) Alignment with Public Academic Datasets: We extracted theorems referenced in public academic datasets (e.g., PGPS9K, MAVIS, GeomVerse) to ensure consistency with commonly used training corpora. (3) Expert Consultation: We consulted primary and secondary school educators to identify important theorems and conclusions grounded in real-world teaching practices.

![](images/6a898219e7286771c3dbced7294cf320d8b2e273a1c00c0de07af7c636aa50f2.jpg)  
Figure 16: The Case of Direct Generation and TR-Reasoner Generation.

Table 10: Summary of Geometric Theorems and Properties
<table><tr><td>Category</td><td>Properties</td><td>Criteria</td></tr><tr><td>Parallel Lines</td><td>Corresponding angles equal; Alternate inte- rior angles equal; Consecutive interior an- gles supplementary</td><td>Equal corresponding angles; Supplemen- tary consecutive angles; Équal alternate an- gles; Parallel to the same line</td></tr><tr><td>General Triangles</td><td>Interior angles sum to 180°</td><td>AA similarity; SSS/SAS/ASA/AAS/HL congruence</td></tr><tr><td>Isosceles Trian- gles</td><td>Equal base angles; Three-line coincidence (angle bisector, median, altitude) ;Base an- gles are  $4 5 ^ { \circ }$  in right-isosceles case</td><td>Two equal angles ; Two equal sides</td></tr><tr><td>Equilateral Trian- gles</td><td>All angles are  $\overline { { 6 0 ^ { \circ } } }$  ; Three - line coincidence</td><td>Three equal sides ; Three equal angles ; Isosceles triangle with a  $6 0 ^ { \circ }$  angle</td></tr><tr><td>Right Triangles  $\scriptstyle { \dot { c } } ^ { 2 }$ </td><td>Acute angles are complementary ; Side op- posite  $3 0 ^ { \circ }$  angle is half of the hypotenuse ; Median on the hypotenuse is half of the  $\mathrm { { h y - } }$  potenuse ; Pythagorean theorem:  $a ^ { 2 } + b ^ { 2 } =$ </td><td>Contains a right angle ; HL congruence for right - triangles</td></tr><tr><td>Angle Bisector</td><td>Points on the perpendicular bisector are equidistant from the endpoints</td><td>A ray that divides an angle into two equal parts</td></tr><tr><td>Triangle Midline</td><td>Parallel to the third side and half of its length</td><td>Connects the mid-points of two sides</td></tr><tr><td>Parallelogram</td><td>Opposite sides are equal ; Diagonals bisect each other ; Area = base × height</td><td>Both pairs of opposite sides are parallel; Diagonals bisect each other; Opposite sides are equal</td></tr><tr><td>Rectangle</td><td>All angles are  $\overline { { 9 0 ^ { \circ } } }$  ; Diagonals are equal</td><td>A parallelogram with a right angle; A quadrilateral with three right angles</td></tr><tr><td>Rhombus</td><td>All sides are equal; Diagonals are perpen- dicular to each other</td><td>A parallelogram with adjacent sides equal; A quadrilateral with four equal sides</td></tr><tr><td>Square</td><td>All sides and angles are equal; Diagonals are equal and perpendicular</td><td>Prove it is both a rectangle and a rhombus</td></tr><tr><td>Isosceles Trape- zoid</td><td>Legs are equal; Base angles on the same base are equal</td><td>Two equal legs; Equal base angles on the same base</td></tr><tr><td>Trigonometric Functions</td><td>sin  $3 0 ^ { \circ } = { \frac { 1 } { 2 } }$  ; sin  $\overline { { 4 5 ^ { \circ } = \frac { \sqrt { 2 } } { 2 } } }$  ; sin  $6 0 ^ { \circ } =$   ${ \begin{array} { l } { { \frac { { \sqrt { 3 } } } { 2 } } } \end{array} } ;$  sin  $9 0 ^ { \circ } \ = \ 1 \ ;$  COS  $3 0 ^ { \circ } \ = \ { \frac { \sqrt { 3 } } { 2 } }$   $\cos 4 5 ^ { \circ } = { \frac { \sqrt { 2 } } { 2 } }$  ; COs  $6 0 ^ { \circ } = { \frac { 1 } { 2 } }$  ; cos  $9 0 ^ { \circ } =$  0 ; tan  $3 0 ^ { \circ } \ = \ { \frac { \sqrt { 3 } } { 3 } }$  ; tan  $4 5 ^ { \circ } \ = \ 1$ </td><td>1</td></tr><tr><td>Circle</td><td> $\tan 6 0 ^ { \circ } = { \sqrt { 3 } }$  The perpendicular bisector of a chord is per- pendicular to the chord; The perpendicular bisector of a chord passes through the cen-</td><td></td></tr><tr><td>Central Angle</td><td>ter Equal central angles subtend equal chords and arcs</td><td></td></tr><tr><td>Inscribed Angle</td><td>An inscribed angle is half of the central angle subtended by the same arc; An angle</td><td></td></tr><tr><td>Cyclic Quadrilat- eral</td><td>subtended by a diameter is a right angle Opposite angles are supplementary</td><td></td></tr><tr><td>Tangent</td><td>A tangent is perpendicular to the radius at the point of contact; Tangents from an ex- ternal point to a circle are equal in length</td><td>A line perpendicular to the radius at the endpoint on the circle is a tangent</td></tr><tr><td>Regular Polygon</td><td>For an equilateral triangle inscribed in a circle of radius R, side length  $a = R { \sqrt { 3 } }$  For a square inscribed in a circle of radius R, side length  $a = R { \sqrt { 2 } }$ </td><td></td></tr></table>