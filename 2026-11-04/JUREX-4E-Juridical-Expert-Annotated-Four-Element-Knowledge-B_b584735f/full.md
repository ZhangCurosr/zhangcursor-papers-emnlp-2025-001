# JUREX-4E: Juridical Expert-Annotated Four-Element Knowledge Base for Legal Reasoning

Huanghai Liu<sup>1</sup>\*, Quzhe Huang<sup>2\*</sup>, Qingjing Chen<sup>3</sup>, Yiran Hu<sup>1</sup>,

Jiayu Ma<sup>1</sup>, Yun Liu<sup>1</sup>, Weixing Shen<sup>1†</sup>, Yansong Feng<sup>2</sup>†

<sup>1</sup>School of Law, Tsinghua University

<sup>2</sup>Wangxuan Institute of Computer Technology, Peking University <sup>3</sup>Department of Legal Studies, University of Bologna

{liuhh23,chenqj21,huyr21,ma-jy24}@mails.tsinghua.edu.cn {huangquzhe,fengyansong}@pku.edu.cn {liuyun89,wxshen}@mail.tsinghua.edu.cn

## Abstract

In recent years, Large Language Models (LLMs) have been widely applied to legal tasks. To enhance their understanding of legal texts and improve reasoning accuracy, a promising approach is to incorporate legal theories. One of the most widely adopted theories is the Four-Element Theory (FET), which defines the crime constitution through four elements: Subject, Object, Subjective Aspect, and Objective Aspect. While recent work has explored prompting LLMs to follow FET, our evaluation demonstrates that LLM-generated four-elements are often incomplete and less representative, limiting their effectiveness in legal reasoning. To address these issues, we present JUREX-4E, an expert-annotated fourelement knowledge base covering 155 criminal charges. The annotations follow a progressive hierarchical framework grounded in legal source validity and incorporate diverse interpretive methods to ensure precision and authority. We evaluate JUREX-4E on the Similar Charge Disambiguation task and apply it to Legal Case Retrieval. Experimental results validate the high quality of JUREX-4E and its substantial impact on downstream legal tasks, underscoring its potential for advancing legal AI applications. The dataset and code are available at: https://github.com/THUlawtech/JUREX

## 1 Introduction

Large Language Models (LLMs) have recently demonstrated impressive performance in legal tasks such as charge prediction (Yuan et al., 2024) and legal case retrieval (Feng et al., 2024). In these applications, a key challenge is accurately understanding complex legal language. To address this, recent studies have introduced legal theories into LLM workflows (Jiang and Yang, 2023; Servantez et al., 2024; Yuan et al., 2024; Deng et al., 2023), as these theories provide structured reasoning frameworks and domain knowledge. Among these theories, the Four-Element Theory (FET) in Chinese criminal law (Liang, 2017) is particularly important, as it defines the legal criteria for establishing criminal liability. FET breaks down a criminal charge into four elements: Subject, Object, Subjective Aspect, and Objective Aspect, which serve as the essential criteria for determining whether a defendant’s behavior constitutes a specific crime.

![](images/d888731483c962cb6081ff12c422b3f93590d84027f01aec54a220e10089898d.jpg)  
Figure 1: An example of LLM-generated four-elements.

Most current approaches rely on the LLM’s internal knowledge to incorporate the FET. A common method is to ask LLMs to emulate expert reasoning processes. For example, designing four separate prompts to guide the LLM outputs in the form of four-elements (Deng et al., 2023). This raises a critical question: Can LLMs reliably understand and apply the FET?

To investigate this, we conducted a pilot study where we provide LLMs with legal articles and asked them to generate the four-elements for several representative charges (Ouyang et al., 1999). Results show that the LLM-generated fourelements are often not accurate enough. As shown in Figure 1, in the charge of misappropriation of public funds, the LLM failed to identify the right to benefit from the use of public funds, a core part of the Object. These results suggest that LLMs lack the domain knowledge and legal reasoning precision required for reliable FET application.

To help LLMs better utilize the FET in legal tasks, we construct JUREX-4E: JURidical EXpert-annotated 4-Element knowledge base for legal reasoning. The knowledge base covers 155 high-frequency criminal charges, each decomposed into Subject, Object, Subjective Aspect, and Objective Aspect. JUREX-4E is built through a fourstage Hierarchical Legal Interpretation System. In this process, legal experts refine each element by referencing sources in descending order of legal validity—Criminal Articles, Judicial Interpretations, Guiding Cases, and Academic Discourses—while applying appropriate interpretive methods at each stage. Each charge was annotated over a sevenmonth period, yielding knowledge-rich representations with an average annotation length of 472.5 words.

To assess the quality of JUREX-4E, we conduct a human evaluation on four independent dimensions, Precision, Completeness, Representativeness, and Standardization, grounded in legal scholarship on how criminal elements should be normatively defined and expressed in judicial contexts (Zhang, 2007a). The expert-annotated fourelements achieved an average score of 4.60 on a 5-point scale, significantly outperforming the LLMgenerated ones, which scored 3.96. Among the four dimensions, the largest performance gaps appeared in Completeness and Representativeness, as expert annotations provided more comprehensive legal interpretations and summarized typical application scenarios, which are often overlooked by LLMs.

To further evaluate the quality and utility of JUREX-4E, we conducted two downstream tasks: Similar Charge Disambiguation (SCD) and Legal Case Retrieval (LCR). In the SCD task (Liu et al., 2021), we tested whether different charges could be more effectively distinguished by incorporating four-element knowledge. Results show that expert-annotated four-elements from JUREX-4E consistently outperformed LLM-generated counterparts across various prompting strategies and model types, improving average accuracy by 0.70% and F1-score by 0.75%. In the LCR task (Li et al., 2024d), we incorporated JUREX-4E into the retrieval pipeline to guide case-level four-element generation and similarity matching, achieving better retrieval accuracy. Together, these findings validate the high quality and practical value of JUREX-4E in enhancing legal understanding and decisionmaking.

Our contributions are as follows:

(1) We demonstrate that while LLMs can assist legal reasoning to some extent, they still fall short in accurately understanding and applying the Four-Element Theory.

(2) We construct the JUREX-4E, the first expertannotated legal knowledge base grounded in a hierarchical legal interpretation framework based on legal source validity.

(3) We validate the quality and effectiveness of JUREX-4E on two representative legal tasks, Similar Charge Disambiguation (SCD) and Legal Case Retrieval (LCR), where it consistently outperforms LLM-generated representations across various prompting strategies.

## 2 Background

The Four-Element Theory (FET) of crime constitution is a fundamental framework in Chinese criminal law (Liang, 2017). It provides a standardized structure to determine criminal liability through four elements: Subject, Object, Subjective Aspect, and Objective Aspect.

For example, the four-elements of Robbery can be briefly summarized as follows:

(1) Subject (the person who commits a criminal act and should bear criminal responsibility): General subjects above the age of criminal responsibility.

(2) Object (the legal interest harmed by the act): A compound object, combining both property ownership and personal rights of the victim.

(3) Subjective Aspect (the offender’s mental state regarding the harmful act): Direct intent with the purpose of unlawfully appropriating another’s property.

(4) Objective Aspect (the external facts of the criminal activity, including key actions and their outcomes): On-the-spot taking of property from an owner, custodian, or possessor through violence, coercion, or other methods.

For the legal community, FET plays a central role in doctrinal analysis and judicial reasoning. It serves as the legal basis for both legislation and adjudication, ensuring internal consistency and normative rigor in criminal law application (Li, 2006; Zhang, 2007a). For the legal AI community, FET offers a task-agnostic and interpretable framework for modeling legal reasoning (Deng et al., 2023; Yuan et al., 2024).

Compared to general reasoning templates (e.g., legal syllogism (Gold, 2018)) or alternative theories such as the Three-Tier System (Zhou, 2017; Zhang, 2010), FET has become the dominant approach in China for assessing criminal liability (Wang, 2017). Its clearer and more interpretable decomposition of crimes into objective and subjective elements makes it particularly suitable for structured legal reasoning tasks.

## 3 Related Work

With the rise of open-source base LLMs, lots of legal LLMs have emerged, such as Lawyer LLaMA (Huang et al., 2023), DiSC-LawLLM (Yue et al., 2023), ChatLaw (Cui et al., 2024), and TongyiFarui<sup>1</sup>. These models are typically adapted from general-purpose LLMs via domain-specific post-training or Retrieval-Augmented Generation (RAG), incorporating legal texts like cases and laws.

Although these models achieve notable improvements on legal tasks, they still struggle with complex legal reasoning, such as charge disambiguation, legal question answering, statutory interpretation, and structured explanation generation (Hu et al., 2025). LegalDiscourse shows that LLMs often fail to capture when laws apply and to whom (Spangher et al., 2024), while LegalBench demonstrates that even state-of-the-art models underperform on diverse reasoning-intensive legal tasks (Guha et al., 2023).

To further enhance model performance, particularly in tasks requiring complex legal reasoning, some studies draw inspiration from established legal reasoning paradigms. For example, introducing the legal syllogism for legal judgment prediction (Jiang and Yang, 2023); using the IRAC paradigm to guide LLMs in reasoning about compositional rules (Servantez et al., 2024). Several works have drawn on the FET in the context of Chinese criminal law. For example, breaking down legal rules into FET-aligned components using automated planning techniques (Yuan et al., 2024); employing model-generated FETs as minor premises in legal judgment analysis (Deng et al., 2023).

While these methods have demonstrated improved performance on downstream tasks, they generally assume that the LLMs inherently understand the Four-Element Theory, without systematically validating this assumption.

## 4 Can LLMs Grasp Legal Theory?

To examine whether LLMs can understand and apply the Four-Element Theory (FET), we ask them to generate the four elements (FETs) for several representative charges and then analyze the outputs against expert annotations.

We select GPT-4o as the target LLM, as it achieves state-of-the-art performance on opensource legal benchmarks (Fei et al., 2023; Li et al., 2024c) and also outperforms others in our pilot study (Appendix D), indicating a strong capacity to understand and apply legal knowledge. Following prior work (Deng et al., 2023; Cui et al., 2024; Zhou et al., 2023), for each charge, we prompt GPT-4o with corresponding criminal articles (see prompt template in Appendix C).

We invite legal experts who passed the National Judicial Examination to analyze the LLMgenerated FETs and identify two main issues:

(1) Inaccurate elements: LLMs may produce inaccurate FETs. For example, in Figure 1, for misappropriation ofpublicfunds, the LLM-generated Object is “the management order of public funds and the integrity of officials’ conduct”, missing the right to benefit from the use of public funds, which is necessary to identify this charge.

(2) Insufficient interpretive ability: LLMs fail to recognize when statutory language requires deeper interpretation. As shown in Figure 1, the model simply extracts “misappropriating public funds for personal use” to describe the Objective Aspect. However, this phrase is far too general for practice. In judicial interpretations<sup>2</sup>, the term “for personal use” should be interpreted with three situations: (1) using public funds for oneself, relatives, or other individuals; (2) lending public funds to other entities in one’s own name; or (3) using public funds in the name of one’s organization for another entity to gain personal benefits.

## 5 Dataset Construction

The lack of accuracy and interpretation in the generated FETs undermines the reliability of legal reasoning tasks. To address this, we introduce an expert-annotated FET dataset that captures both formal legal definitions and practical interpretive nuances, supporting more trustworthy and adaptable legal AI systems.

![](images/64eeac967d07e8d2305476f7841fe4f13434d6f43005894fa7d322cb8c25f2b2.jpg)  
Figure 2: Hierarchical Legal Interpretation System based on legal source validity. The system consists of four annotation rounds, each using different interpretive methods based on different legal sources. Solid arrows indicate the primary method applied; dashed arrows represent supplementary use.

To ensure both legal validity and interpretive clarity, we design a hierarchical annotation framework rooted in statutory sources and authoritative interpretive methods.

## 5.1 Hierarchical Legal Interpretation System

Given a specific charge, we ask five legal experts to annotate the FETs based on relevant legal materials like articles and cases. This annotation process is essentially an act of legal interpretation. Legal interpretation refers to the application of various methods to analyze and understand legal texts, to determine their meaning and application in specific legal contexts (Barak, 2005). In our task, it involves applying different interpretation methods to the different materials in order to analyze and define the connotation and extension of each of the FETs of a charge. In designing our annotation framework, we address the following two questions:

(1) What sources are interpreted. Legal interpretation draws upon various legal sources with different levels of validity. In legal studies, these sources are categorized based on their legal validity into formal sources (which carry legal force in judgments) and informal sources (which serve as references without legal force) (Pound, 1925; Watson, 1982; Pound, 1932). Articles and judicial interpretations are considered formal sources, whereas case precedents and academic discourses are regarded as informal sources under the Chinese legal system (Zhang and Zhou, 2007). Accordingly, we organize legal sources by their level of validity, with the following order of priority: Article → Judicial Interpretations → Guiding Cases → Academic

Discourses.

(2) How the law is interpreted. When interpreting the above sources, different interpretation methods are required. These methods follow a hierarchical order (Sutherland, 1891; Kim and Division, 2008; Eig, 2014): Legal interpretation should begin with literal interpretation (interpreting the text based on its plain meaning). If the intended meaning cannot be clearly derived from the article alone, systematic interpretation (considering the article’s role within the legal system) and purposive interpretation (considering the legislative intent) should be applied. If ambiguity remains, historical interpretation (based on the legislative history), sociological interpretation (based on the article’s social function and consequences), and other interpretation methods may be used to further clarify the legal meaning. The specific definition of legal interpretation methods is in Appendix B.

We also consider the nature of each source. For example, Guiding Cases, as informal sources, do not define elements literally but instead supplement statutory interpretation through purposive, sociological, and other interpretive methods. The correspondence between interpretation methods and legal sources is illustrated by the orange arrows in Figure 2.

## 5.2 Annotation Process

As shown in the left part of Figure 2, our annotation process takes charges as input and outputs corresponding FETs, following a Hierarchical Legal Interpretation System to organize legal sources by validity and apply interpretation methods. The annotators are five experts, all of whom have passed the National Judicial Examination and are familiar with FET. The entire annotation process took 7 months and involved four rounds.

Stage One: Literal interpretation using the core article. The interpretation of the FETs begins with the core article of each charge, which carries the highest legal validity, mainly through literal interpretation.

At this stage, annotators analyze the article’s subject–predicate–object structure to identify candidate FETs, mapping the subject to the Subject (who commits the crime), the predicate (verb phrase) to the Objective Aspect (the conduct carried out), the object to the Object (the legal interest infringed), and adverbial phrases to the Subjective Aspect (the offender’s mental state).

For example, Article 263 of the Chinese Criminal Law, concerning robbery, states that “forcibly seizing public or private property through violence, coercion, or other means” describes the Objective Aspect. Since the article lacks an explicit subject, a general subject is assumed by default. The adverbs “violence” and “coercion” indicate an intentional act. This stage spans two months.

Stage Two: Systematic interpretation using related articles and judicial interpretations. While Stage One relies on the primary article of each charge, literal interpretation often leaves FETs underspecified. Stage Two, therefore, applies systematic interpretation, situating underspecified terms within the broader legal framework. By considering the provision’s function in the Criminal Law and its links to related articles and judicial interpretations (broadly understood here to include SPC guiding opinions and other interpretative documents), annotators clarify the scope and meaning of the element.

For example, Article 263 of the Criminal Law does not specify whether “violence” must be directed only at persons or may also extend to property, nor does it make clear which borderline conduct should be excluded. Article 289 (Congress, 2017) provides that in mass “smashing, looting, and robbing,” the destruction or seizure of property by ringleaders may be punished as robbery, suggesting that violence may also cover acts against property. Conversely, the SPC’s 2016 Guiding Opinion on Robbery (spc, 2016) clarifies that if an offender uses only minor force to escape after theft, fraud, or snatching, and no injury above the statutory threshold results, such conduct is not deemed “violence” and does not requalify the offense as robbery.

Stage Three: Purposive and sociological interpretation using guiding cases. Although the first two stages define the FETs based on articles and judicial interpretations, these sources remain abstract. In legal practice, courts also refer to Guiding Cases, designated by the Supreme People’s Court since 2011, to illustrate how legal articles are applied in concrete cases and interpreted in light of social purposes (Chen et al., 2024).

At this stage, annotators refine FETs by incorporating specific case scenarios from Guiding Cases. Since the number of Guiding Cases is limited, for rarer charges, we also consult model cases from the People’s Court Case Database and Gazette cases<sup>3</sup>. Annotators mainly apply purposive and sociological interpretation to examine how legal elements are concretized in the reasoning process of practical cases, considering both legislative intent and social context.

For example, in defining how “violence” in robbery operates in practice, annotators refer to a case involving molestation (Ma, 2021). The offender bound the victim to commit indecent acts and then took her phone. Under purposive and sociological interpretation, the ongoing molestation maintained the victim’s restrained state and thus constituted new violence. Such cases clarify that acts like molestation, though not physically injurious, can suppress resistance and therefore qualify as violence. This stage spans two months.

Stage Four: Diverse interpretations using academic discourses. Although Stages One to Three refine both the abstract definitions and concrete scenarios of each element from various legal sources, certain elements still involve unresolved issues that rarely appear in practice and therefore lack clear judicial standards.

At this stage, annotators consult academic discourses and apply diverse interpretations such as comparative, purposive, and sociological interpretation. For elements where disagreement exists, they record both mainstream and minority views, providing concise annotations that explain the underlying legal reasoning.

For example, in defining “violence” in robbery, mainstream views in China, the former Soviet Union, North Korea, and Japan require that it endanger the victim’s life or health (Zhang, 2007b), while others argue that any force sufficient to subdue the victim should qualify (Yang, 2010). The annotations record both the dominant consensus and minority positions. This stage spans one month.

The four stages represent the main interpretive approaches, but they are not mutually exclusive. In practice, annotators often combine methods when clarifying a particular element. As shown in Figure 2, the dashed orange arrows mark cross-stage interactions where multiple interpretive methods operate in a complementary way.

## 5.3 Data Statistics

<table><tr><td>Metric</td><td>LLMMean</td><td>LLMMedian</td><td>ExpertMean</td><td>ExpertMedian</td></tr><tr><td>Avg. Length</td><td>115.43</td><td>-</td><td>472.53</td><td>-</td></tr><tr><td>Subject</td><td>23.12</td><td>27</td><td>51.64</td><td>17</td></tr><tr><td>Object</td><td>15.86</td><td>15</td><td>36.01</td><td>25</td></tr><tr><td>Subjective Aspect</td><td>28.00</td><td>30</td><td>42.38</td><td>21</td></tr><tr><td>Objective Aspect</td><td>48.45</td><td>45</td><td>342.5</td><td>230</td></tr></table>

Table 1: Comparison of element lengths.

Our dataset comprises 155 common criminal charges. These charges are selected based on their frequency in over 2.6 million publicly available criminal cases in China: specifically, we include all charges that appear more than 3,000 times, which together account for 91.71% of all cases (Appendix A), ensuring coverage of the most common real-world judicial scenarios.

To compare the quality of expert-annotated FETs (expert-FETs) and LLM-generated FETs, we selected 105 charges in JUREX-4E that also appear in the widely used LeCaRDv2 dataset (Li et al., 2024d). LLM-generated FETs were produced using the same setup as before, with a maximum output of 8192 tokens. Table 1 summarizes the differences in element length, with full length distributions available in Appendix A.

Overall, expert-FETs are significantly longer, with an average total length of 472.53 words compared to 115.43 for LLM-generated ones. The most pronounced gap appears in the Objective Aspect (OA) (mean: 342.5 vs. 48.45), where experts provide detailed factual descriptions, such as action, result, time, and location, often underdeveloped in LLM outputs. While the Subject (SB), Object (OB), and Subjective Aspect (SA) show smaller median differences, notable variation remains, especially in SB (mean: 51.64 vs. 17), which in certain charges involves complex legal interpretations (e.g., “work” in copyright infringement) requiring more elaborate legal definitions.

<table><tr><td>Dimension</td><td>LLM</td><td>Expert</td><td>δ</td></tr><tr><td>Precision</td><td>4.12</td><td>4.69</td><td>+0.57</td></tr><tr><td>Completeness</td><td>3.79</td><td>4.65</td><td>+ 0.86</td></tr><tr><td>Representativeness</td><td>3.60</td><td>4.48</td><td>+ 0.88</td></tr><tr><td>Standardization</td><td>4.33</td><td>4.56</td><td>+ 0.23</td></tr></table>

Table 2: Performance comparison of four elements across methods. δ represents the score difference between expert and LLM-generated FETs, with experts outperforming LLMs in all dimensions.

## 6 Human Evaluation

To compare the quality of expert-annotated and LLM-generated FETs, we selected six complicated charges in Chinese judicial practice (Ouyang et al., 1999). Based on prior theoretical framework (Zhang, 2007a), we assess the quality of FETs along four independent dimensions: Precision, Completeness, Representativeness, and Standardization.

• Precision: Whether each element accurately aligns with its statutory definition, reflecting key terms in the corresponding legal article.

• Completeness: Whether each element includes all practically necessary information, ensuring the definition is sufficient to guide legal reasoning.

• Representativeness: Whether the annotations reflect the most typical and practically significant scenarios in judicial practice.

• Standardization: Whether the expressions of elements are consistent across different charges, with clear, concise, and unambiguous language that facilitates understanding and minimizes interpretive variance.

Evaluation was conducted by experts from two backgrounds: one group with a purely legal background and another with a combined background in law and AI, all of whom have passed the National Judicial Examination. The experts were selected to balance domain expertise and interdisciplinary perspectives. Scores were averaged across the two groups. Details about the 1-5 scale criteria and annotator background are provided in Appendix E.

As shown in Table 2, expert annotations consistently outperform LLM-generated elements across all four dimensions. The most pronounced deficiencies are observed in Completeness (+0.86) and Representativeness (+0.88). This aligns with our earlier analyses, where expert-generated elements include more factual details and representative descriptions. The gap in Precision (+0.57) suggests a tendency toward vague or legally irrelevant content, while the smaller difference in Standardization (+0.23) shows that LLMs can mimic structural patterns but lack deeper normative consistency. These results demonstrate the importance of expert supervision in providing reliable legal knowledge.

<table><tr><td>Model</td><td colspan="2">F&amp;E</td><td colspan="2">E&amp;MPF</td><td colspan="2">AP&amp;DD</td><td colspan="2">Average</td></tr><tr><td></td><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td><td>Acc</td><td>F1</td></tr><tr><td>GPT-40</td><td>94.36</td><td>95.81</td><td>86.49</td><td>89.76</td><td>85.54</td><td>87.12</td><td>88.72</td><td>90.07</td></tr><tr><td>GPT-4o+Article</td><td>95.34</td><td>96.30</td><td>92.64</td><td>93.03</td><td>88.30</td><td>89.33</td><td>92.09</td><td>92.89</td></tr><tr><td>Legal-COT</td><td>94.99</td><td>96.27</td><td>90.50</td><td>90.99</td><td>87.81</td><td>88.14</td><td>89.95</td><td>90.85</td></tr><tr><td>MALR</td><td>94.62</td><td>95.82</td><td>86.99</td><td>86.98</td><td>87.86</td><td>88.68</td><td>89.82</td><td>90.49</td></tr><tr><td> $\mathrm { F a r u i – p l u s + F E T _ { 4 0 } }$ </td><td>89.09</td><td>90.27</td><td>86.32</td><td>88.00</td><td>75.90</td><td>77.67</td><td>83.77</td><td>85.31</td></tr><tr><td> $\mathrm { F a r u i – p l u s + F E T _ { E x p e r t } }$ </td><td>89.29</td><td>90.98</td><td>86.13</td><td>87.54</td><td>76.25</td><td>78.12</td><td>83.89</td><td>85.55</td></tr><tr><td> $\mathrm { Q w e n } { \bar { 2 } } . 5 { \cdot } 7 2 \mathrm { b } { + } \mathrm { F E } { \cdot } \mathrm { \bar { T } } _ { 4 0 }$ </td><td>93.15</td><td>95.06</td><td>90.99</td><td>93.56</td><td>87.71</td><td>88.56</td><td>90.62</td><td>92.39</td></tr><tr><td> $\mathrm { Q w e n } 2 . 5 { \cdot } 7 2 \mathrm { b } { + } \mathrm { F E } \mathrm { T } _ { \mathrm { E x p e r t } }$ </td><td>93.29</td><td>95.18</td><td>91.18</td><td>93.66</td><td>87.81</td><td>89.45</td><td>90.76</td><td>92.76</td></tr><tr><td> $\mathrm { G P T \mathrm { - } 4 o \mathrm { + F E T _ { \mathrm { f a r u i } } } }$ </td><td>94.86</td><td>96.12</td><td>91.84</td><td>92.64</td><td>89.35</td><td>89.85</td><td>92.02</td><td>92.87</td></tr><tr><td> $\mathrm { G P T - 4 o + F E T _ { q w e n } }$ </td><td>95.53</td><td>96.53</td><td>91.82</td><td>92.96</td><td>89.48</td><td>90.09</td><td>92.28</td><td>93.19</td></tr><tr><td> $\mathrm { G P T - 4 o + F E T _ { 4 0 + f a r u i + q w e n } }$ </td><td>94.97</td><td>96.24</td><td>91.84</td><td>92.73</td><td>89.69</td><td>90.12</td><td>92.17</td><td>93.03</td></tr><tr><td> $\mathrm { G P T } { - } 4 0 { + } \mathrm { F E T } _ { 4 0 }$ </td><td>95.73</td><td>96.56</td><td>91.87</td><td>92.01</td><td>89.61</td><td>89.69</td><td>92.40</td><td>92.75</td></tr><tr><td> $\mathrm { G P T - 4 o + F E T _ { 4 0 + I C L } }$ </td><td>95.74</td><td>96.36</td><td>91.84</td><td>92.01</td><td>90.48</td><td>90.63</td><td>92.69</td><td>93.00</td></tr><tr><td> $\mathrm { G P T - 4 o + F E T _ { E x p e r t } }$ </td><td>96.06</td><td>96.69</td><td>92.57</td><td>93.05</td><td>90.53</td><td>90.62</td><td>93.05</td><td>93.45</td></tr></table>

Table 3: Performance on the Similar Charge Disambiguation (SCD) task. “Expert” refers to our expert-annotated FET, while “4o”, “qwen”, and “farui” refer to FET generated by different LLMs. Highest results are in bold.

## 7 Evaluation on Similar Charge Disambiguation

To further validate annotation quality, we introduce the Similar Charge Disambiguation (SCD) task (Yuan et al., 2024; Li et al., 2024a). Given the case fact and a set of similar charges, SCD task requires the model to identify which charge is correct. We evaluate whether similar charges can be effectively distinguished based on their FETs, and whether expert-annotated FETs perform better than LLM-generated FETs.

## 7.1 Experiment Settings

## 7.1.1 Dataset and metrics

We chose the SCD dataset released by (Liu et al., 2021). Following previous work (Yuan et al., 2024), we selected three 2-label classification groups: Fraud & Extortion (F&E), Embezzlement & Misappropriation of Public Funds (E&MPF), and Abuse of Power & Dereliction of Duty (AP&DD). Each charge has over 1.9k cases, with a total of 13,962 cases. The details of the groups are shown in Appendix F. Following previous work (Liu et al., 2021; Yuan et al., 2024), we use Average Accuracy (Acc) and macro-F1 (F1) as evaluation metrics.

## 7.1.2 Baselines and Methods

We compared the following baselines: GPT-4o (Achiam et al., 2023) and GPT-4o+Article, which explicitly supplies relevant legal articles;

Legal-COT (Kojima et al., 2022), a Chain-of-Thought variant that applies the Four-Element Theory step by step, and MALR (Yuan et al., 2024), a multi-agent framework that decomposes legal tasks into FET-aligned subtasks. Details are in Appendix F.

Methods: Following Section 4, our main model is GPT-4o. We also compared Farui-plus (the latest version of Tongyifarui, representative legal LLM) and Qwen2.5-72B (Bai et al., 2023) (representative open-source LLM). To incorporate FET knowledge, each group of similar charges is augmented with four-element descriptions, either generated by LLMs or sourced from JUREX-4E. For example, $\mathrm { G P T - 4 o + F E T _ { L L M } }$ uses LLM-generated FETs, while $\mathrm { G P T - 4 o + F E T _ { E x p e r t } }$ uses expert-annotated ones. The input format is fixed across methods, differing only in the [four-elements of candidate charges] (Appendix F). All experiments are zeroshot, with max\_tokens set to $3 , 0 0 0 \ ( 1 0 , 0 0 0$ for Legal-COT and MALR) and a temperature of 0 or 0.0001 in repeated runs.

To further explore generation setups, we also evaluate an ICL variant, GPT-4o+FET<sub>4o + ICL</sub>, where two representative FET exemplars (Theft and Snatching) are provided in the prompt to guide generation.

## 7.2 Results

The SCD results are shown in Table 3, where we can observe that:

Effectiveness of Structured FET Knowledge: Providing specific structured charge FETs yields the highest accuracy among all legal knowledge integration methods. Compared to implicit approaches, such as prompts (GPT-4o+Article, Acc

<table><tr><td>Model</td><td>NDCG@10</td><td>NDCG@20</td><td>NDCG@30</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@20</td><td>R@30</td><td>MRR</td></tr><tr><td>BGE (case_fact only)</td><td>0.4737</td><td>0.5539</td><td>0.5937</td><td>0.0793</td><td>0.2945</td><td>0.4298</td><td>0.6500</td><td>0.7394</td><td>0.1926</td></tr><tr><td>BGE+FET (Qwen2.5)</td><td>0.5125</td><td>0.5858</td><td>0.6350</td><td>0.1104</td><td>0.2870</td><td>0.4653</td><td>0.6679</td><td>0.7836</td><td>0.2168</td></tr><tr><td rowspan="2">FET only BGE+FET (Expert, Qwen2.5)</td><td>0.3367</td><td>0.3971</td><td>0.4487</td><td>0.0622</td><td>0.2006</td><td>0.3279</td><td>0.4806</td><td>0.6037</td><td>0.1524</td></tr><tr><td>0.5295</td><td>0.5979</td><td>0.6416</td><td>0.1124</td><td>0.3122</td><td>0.4838</td><td>0.6791</td><td>0.7824</td><td>0.2206</td></tr><tr><td>FET only</td><td>0.3354</td><td>0.4035</td><td>0.4541</td><td>0.0849</td><td>0.1923</td><td>0.3076</td><td>0.4839</td><td>0.6097</td><td>0.1606</td></tr><tr><td>BGE+FET (GPT-40) FET only</td><td>0.5139</td><td>0.5862</td><td>0.6291</td><td>0.0980</td><td>0.2967</td><td>0.4769</td><td>0.6802</td><td>0.7828</td><td>0.2140</td></tr><tr><td rowspan="2">BGE+FET (Expert, GPT-40)</td><td>0.3583</td><td>0.4293</td><td>0.4798</td><td>0.0506</td><td>0.2240</td><td>0.3644</td><td>0.5383</td><td>0.6652</td><td>0.1453</td></tr><tr><td>0.5211</td><td>0.5920</td><td>0.6379</td><td>0.1024</td><td>0.3049</td><td>0.4883</td><td>0.6885</td><td>0.7967</td><td>0.2155</td></tr><tr><td>FET only</td><td>0.3766</td><td>0.4584</td><td>0.5111</td><td>0.0715</td><td>0.1894</td><td>0.3709</td><td>0.5891</td><td>0.7203</td><td>0.1624</td></tr></table>

Table 4: Performance on the Legal Case Retrieval (LCR) task. The highest results are in bold. “FET only” indicates using the four-element descriptions without case facts.

92.09) or reasoning chains (Legal-COT, Acc 89.95), structured FET knowledge offers more effective support for legal decision-making (e.g., GPT-$4 0 { + } \mathrm { F E T _ { E x p e r t } } ,$ Acc 93.05)

Superiority of Expert-Annotated FET: Expertannotated FET consistently outperforms LLMgenerated FET across three representative LLMs, including $\mathrm { F E T _ { f a r u i } , F E T _ { q w e n } , F E T _ { 4 0 } }$ , and their combination $\left( \mathrm { F E T _ { 4 0 + f a r u i + q w e n } } \right)$ . For example, GPT-$4 0 { + } \mathrm { F E T _ { E x p e r t } }$ surpasses $\mathrm { G P T - 4 o + F E T _ { 4 0 } }$ by 0.65 in average accuracy and 0.70 in F1-score.

Consistent Gains Across Models: Expertannotated FETs yield consistent performance gains across different SCD models. When applied to Farui-plus, Qwen2.5-72b, and GPT-4o, it improves F1-score by +0.24, +0.37, and +0.70, respectively over their LLM-generated FET baselines.

The ICL-based variant yields consistent improvements over direct prompting (Table 3), demonstrating the benefit of exemplar guidance, though it still lags behind expert-FETs. We also conducted Mc-Nemar’s test on paired samples between each LLMgenerated FET and the expert-FET, which shows statistically significant improvements $( p < 0 . 0 5 )$ across all tasks (see Table 11 in Appendix F).

## 8 Application in Legal Case Retrieval

In this section, we design a simple expert-guided FET method to apply JUREX-4E to the Legal Case Retrieval (LCR) task, which retrieves relevant cases based on case facts. This task is well-suited for FET because it requires a comprehensive comparison of the four elements across different charges in cases.

## 8.1 Dataset and Metrics

LeCaRDv2 (Li et al., 2024d) is the latest version of LeCaRD (Ma et al., 2021), which is widely used in legal tasks (Li et al., 2024b; Zhou et al., 2023). It comprises 800 queries and 55,192 candidates extracted from 4.3 million criminal case documents. Following previous work (Qin et al., 2024), we chose 1390 candidates and used NDCG@10, 20, 30, Recall@1, 5, 10, 20, and MRR as metrics. We also tested different candidate pool settings (see Appendix G). The results are consistent.

## 8.2 Baselines and Methods

We adopt a dense retrieval framework based on BGE-m3 (Chen et al., 2023), a strong embedding model for legal and general-domain texts. Given a query q and a candidate case c, we compute their vector representations using a shared BGE-m3 encoder. Retrieval is performed by computing cosine similarities between the query and all candidates and selecting the top-k candidates.

To enhance retrieval accuracy, we compare the following three methods:

(1) BGE(case\_fact only): Standard dense retrieval using only BGE-m3 embeddings of the raw case facts.

(2) BGE+FET $( \mathcal { M } _ { g } ) \mathrm { : }$ We prompt different LLMs $\mathcal { M } _ { g }$ to generate a structured four-element description of each case (case-FET) based solely on its facts, without using external knowledge. These case-FETs are then embedded with BGE-m3 and used to compute similarity. Because the FET abstracts away case-specific details, we combine the original fact-based similarity and the FET-based similarity in a ratio of 7:3.

(3) BGE+FET (Expert, $\mathcal { M } _ { g } )$ : An expertguided FET method that incorporates JUREX-4E to guide case-FET generation. It consists of four steps:

1. Charge Prediction. A charge prediction model $\mathcal { M } _ { p }$ (Qwen-plus, details see Appendix D) predicts the set of likely charges $Z = \{ z _ { 1 } , . . . , z _ { k } \}$ for the query case.

2. Expert-FET Matching. Retrieving corresponding charge’s four-elements $\{ f _ { z } \} _ { z \in Z }$ for each predicted charge in JUREX-4E. These provide theoretical guidance for subsequent reasoning.

<table><tr><td>Model</td><td>F&amp;EAcc</td><td>F&amp;EF1</td><td>E&amp;MPF Acc</td><td>E&amp;MPF F1</td><td>AP&amp;DD Acc</td><td>AP&amp;DD F1</td><td>Avg. Acc</td><td>Avg. F1</td></tr><tr><td> $\overline { { \mathrm { G P T } \mathrm { - } 4 0 \mathrm { + F E T } _ { 4 0 } } }$ </td><td>95.73</td><td>96.56</td><td>91.87</td><td>92.01</td><td>89.61</td><td>89.69</td><td>92.40</td><td>92.75</td></tr><tr><td>GPT-40+FETCollaboration</td><td>95.99</td><td>96.56</td><td>91.51</td><td>91.61</td><td>91.02</td><td>90.95</td><td>92.84</td><td>93.04</td></tr><tr><td>GPT-40+FETExpert</td><td>96.06</td><td>96.69</td><td>92.57</td><td>93.05</td><td>90.53</td><td>90.62</td><td>93.05</td><td>93.45</td></tr></table>

Table 5: Results of the collaboration study on Similar Charge Disambiguation (SCD).

3. Case-FET Generation. Guided by $\{ f _ { z } \}$ the LLM $\mathcal { M } _ { g }$ generates case-specific fourelements $f e t _ { c }$ for candidate c.

4. Dense retrieval. We embed the generated FETs using BGE-m3 and compute similarity scores as in Method (2), combining both factual and FET-based similarities.

For the $\mathcal { M } _ { g } ,$ , we chose Qwen2.5-72b and GPT-4o. The retrieval framework is implemented with the FlagEmbedding Toolkit<sup>4</sup> with an RTX 3090. Following prior work (Li et al., 2024d; Qin et al., 2024), we also compare some dense retrieval methods to examine the representativeness of BGE-m3. Results of baselines and prompt templates are available in Appendix G.

## 8.3 Results

The LCR results are shown in Table 4, where we can observe that: (1) FET Enhances Retrieval. Integrating the FET improves retrieval performance across all metrics. For instance, BGE+FET(GPT-4o) improves MRR by 11.11%, and BGE+FET (Expert, GPT-4o) achieves an even larger gain of 11.89%, indicating that structured legal theory benefits retrieval quality. (2) Expert Knowledge is Important. Expert-guided case-FET consistently outperforms LLM-generated variants across both Qwen2.5-72b and GPT-4o backbones. For example, BGE+FET (Expert, GPT-4o) achieves higher Recall@30 (0.7967 vs. 0.7828) and MRR (0.2155 vs. 0.2140). The gap is even larger in the FET only setting (e.g., MRR 0.1624 vs. 0.1453 for GPT-4o), demonstrating that expert knowledge captures critical legal reasoning that LLMs may overlook.

We provide a case study in Appendix I. It illustrates that the expert-annotated FETs of charges provide practical judgment points and key narratives (e.g., the special subject of Embezzlement) that help the LLM focus on essential facts to analyze the case-FET.

## 9 Discussion

To examine whether LLMs can effectively support expert annotation, we designed a preliminary collaboration pipeline that is roughly aligned with our annotation stages.

For each charge, the LLM first retrieved relevant statutory articles (the retrieval of legal sources in stages 1–2). It then extracted the top 10 factual keywords from cases cited by experts (the use of factual cues as in stage 3). Experts validated these keywords for each element, after which the LLM generated refined FET annotations by reasoning over the combined legal texts and factual cues. This pipeline yielded measurable improvements over unguided prompting (Table 5), illustrating the feasibility of expert-in-the-loop workflows.

Nevertheless, the study also highlights limitations. High-frequency keywords often lacked discriminative power for similar charges—for example, descriptors such as “beating” and “stabbing” occurred in both intentional injury and intentional homicide, reducing their utility. The refined FETs still fell short of the depth and precision of expert annotations. Future improvements could include decomposing FET generation into subtasks aligned with annotation stages and enriching each stage with more authoritative sources, thereby strengthening both coverage and normative awareness.

## 10 Conclusion

This paper presents JUREX-4E, an expertannotated FET knowledge base built through a structured legal interpretation process and validated on downstream tasks. Grounded in widely accepted interpretive methods, our framework is adaptable across different branches of law and legal traditions, making it applicable beyond Chinese criminal law. Moreover, the structured approach to integrating expert domain knowledge may inspire applications in other areas where expert judgment is critical.

## 11 Ethical Considerations

The datasets used in our evaluation are sourced from publicly available legal datasets, with all defendant information anonymized to ensure privacy.

## 12 Limitations

Our current knowledge base is limited to 155 charges under Chinese Criminal Law due to the high cost of expert annotation. Future work will explore extending it to other legal domains and jurisdictions.

Another limitation lies in our current integration of factual and legal information. In the LCR task, although case facts are used to generate FETs, the FET only variant excludes the original case facts during retrieval, resulting in performance loss (e.g., MRR 0.1624 vs. 0.2155). This suggests that our current method remains coarse-grained, and more fine-grained fusion strategies, such as multi-agent coordination or retrieval-time integration, deserve future exploration.

## Acknowledgments

This work is supported in part by Beijing Science and Technology Program (Z231100007423011)

## References

2016. Supreme people’s court guiding opinion on adjudicating robbery cases. SPC, Fa Fa [2016] No. 2. Judicial guiding document.

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. 2023. Qwen technical report. arXiv preprint arXiv:2309.16609.

Aharon Barak. 2005. Chapter one. what is legal interpretation? In Purposive Interpretation in Law, pages 3–60. Princeton University Press, Princeton.

Iz Beltagy, Matthew E Peters, and Arman Cohan. 2020. Longformer: The long-document transformer. arXiv preprint arXiv:2004.05150.

Ilias Chalkidis, Manos Fergadiotis, Prodromos Malakasiotis, Nikolaos Aletras, and Ion Androutsopoulos. 2020. Legal-bert: The muppets straight out of law school. arXiv preprint arXiv:2010.02559.

Benjamin M Chen, Zhiyu Li, David Cai, and Elliott Ash. 2024. Detecting the influence of the chinese guiding cases: a text reuse approach. Artificial Intelligence and Law, 32(2):463–486.

Jianlv Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. 2023. Bge m3-embedding: Multi-lingual, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. Preprint, arXiv:2309.07597.

National People’s Congress. 2017. Criminal Law ofthe People’s Republic ofChina. China Legal Publishing House.

Jiaxi Cui, Zongjian Li, Yang Yan, Bohua Chen, and Li Yuan. 2023. Chatlaw: Open-source legal large language model with integrated external knowledge bases. arXiv preprint arXiv:2306.16092.

Jiaxi Cui, Munan Ning, Zongjian Li, Bohua Chen, Yang Yan, Hao Li, Bin Ling, Yonghong Tian, and Li Yuan. 2024. Chatlaw: A multi-agent collaborative legal assistant with knowledge graph enhanced mixtureof-experts large language model. arXiv preprint arXiv:2306.16092.

Wentao Deng, Jiahuan Pei, Keyi Kong, Zhe Chen, Furu Wei, Yujun Li, Zhaochun Ren, Zhumin Chen, and Pengjie Ren. 2023. Syllogistic reasoning for legal judgment analysis. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 13997–14009.

Jacob Devlin. 2018. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

Larry M. Eig. 2014. Statutory interpretation: General principles and recent trends. Technical Report 97- 589, Congressional Research Service.

Zhiwei Fei, Xiaoyu Shen, Dawei Zhu, Fengzhe Zhou, Zhuo Han, Songyang Zhang, Kai Chen, Zongwen Shen, and Jidong Ge. 2023. Lawbench: Benchmarking legal knowledge of large language models. arXiv preprint arXiv:2309.16289.

Yi Feng, Chuanyi Li, and Vincent Ng. 2024. Legal case retrieval: A survey of the state of the art. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 6472–6485.

Michael Evan Gold. 2018. A Primer on Legal Reasoning. Cornell University Press, Ithaca, New York.

Neel Guha, Julian Nyarko, Daniel Ho, Christopher Ré, Adam Chilton, Alex Chohlas-Wood, Austin Peters, Brandon Waldon, Daniel Rockmore, Diego Zambrano, et al. 2023. Legalbench: A collaboratively built benchmark for measuring legal reasoning in large language models. Advances in neural information processing systems, 36:44123–44279.

Yiran Hu, Huanghai Liu, Qingjing Chen, Ning Zheng, Chong Wang, Yun Liu, Charles LA Clarke, and Weixing Shen. 2025. J&h: Evaluating the robustness of large language models under knowledge-injection attacks in legal domain. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pages 28106–28115.

Quzhe Huang, Mingxu Tao, Chen Zhang, Zhenwei An, Cong Jiang, Zhibin Chen, Zirui Wu, and Yansong Feng. 2023. Lawyer llama technical report. arXiv preprint arXiv:2305.15062.

Cong Jiang and Xiaolei Yang. 2023. Legal syllogism prompting: Teaching large language models for legal judgment prediction. In Proceedings of the Nineteenth International Conference on Artificial Intelligence and Law, pages 417–421.

Yule Kim and American Law Division. 2008. Statutory interpretation: General principles and recent trends. Congressional Research Service Washington, DC.

Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. 2022. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199– 22213.

Ang Li, Qiangchao Chen, Yiquan Wu, Ming Cai, Xiang Zhou, Fei Wu, and Kun Kuang. 2024a. From graph to word bag: Introducing domain knowledge to confusing charge prediction. arXiv preprint arXiv:2403.04369.

Haitao Li, Qingyao Ai, Jia Chen, Qian Dong, Yueyue Wu, Yiqun Liu, Chong Chen, and Qi Tian. 2023. Sailer: structure-aware pre-trained language model for legal case retrieval. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 1035–1044.

Haitao Li, Qingyao Ai, Xinyan Han, Jia Chen, Qian Dong, Yiqun Liu, Chong Chen, and Qi Tian. 2024b. Delta: Pre-train a discriminative encoder for legal case retrieval via structural word alignment. arXiv preprint arXiv:2403.18435.

Haitao Li, You Chen, Qingyao Ai, Yueyue Wu, Ruizhe Zhang, and Yiqun Liu. 2024c. Lexeval: A comprehensive chinese legal benchmark for evaluating large language models. arXiv preprint arXiv:2409.20288.

Haitao Li, Yunqiu Shao, Yueyue Wu, Qingyao Ai, Yixiao Ma, and Yiqun Liu. 2024d. Lecardv2: A largescale chinese legal case retrieval dataset. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2251–2260.

Hong Li. 2006. No need to reconstruct china’s crime constitution system. Chinese Journal of Law, (1):32– 51.

Genlin Liang. 2017. The vicissitudes of chinese criminal law and theory: A study in history, culture and politics. Peking University Law Journal, 5(1):25–49.

Xiao Liu, Da Yin, Yansong Feng, Yuting Wu, and Dongyan Zhao. 2021. Everything has a cause: Leveraging causal inference in legal text analysis. arXiv preprint arXiv:2104.09420.

Yinxiang Ma. 2021. The spiritualization and limitation of the concept of violence in robbery. Law Science, (06):76–91.

Yixiao Ma, Yunqiu Shao, Yueyue Wu, Yiqun Liu, Ruizhe Zhang, Min Zhang, and Shaoping Ma. 2021. Lecard: a legal case retrieval dataset for chinese law system. In Proceedings of the 44th international ACM SIGIR conference on research and development in information retrieval, pages 2342–2348.

Tao Ouyang, Kejia Wei, and Renwen Liu. 1999. Confusing crimes, noncrime, and boundaries between crimes.

Roscoe Pound. 1925. Jurisprudence.

Roscoe Pound. 1932. Hierarchy of sources and forms in different systems of law. Tul. L. Rev., 7:475.

Weicong Qin, Zelin Cao, Weijie Yu, Zihua Si, Sirui Chen, and Jun Xu. 2024. Explicitly integrating judgment prediction with legal document retrieval: A law-guided generative approach. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 2210–2220.

Sergio Servantez, Joe Barrow, Kristian Hammond, and Rajiv Jain. 2024. Chain of logic: Rule-based reasoning with large language models. arXiv preprint arXiv:2402.10400.

Alexander Spangher, Zihan Xue, Te-Lin Wu, Mark Hansen, and Jonathan May. 2024. Legaldiscourse: Interpreting when laws apply and to whom. In Proceedings ofthe 2024 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 8528–8551.

Jabez Gridley Sutherland. 1891. Statutes and Statutory Construction: Including a Discussion ofLegislative Powers, Constitutional Regulations Relative to the Forms of Legislation and to Legislative Procedure, Together with an Exposition at Length of the Principles ofInterpretation and Cognate Topics. Callaghan.

Shizhou Wang. 2017. Criminal law in china.

Alan Watson. 1982. Legal change: sources of law and legal culture. U. Pa. L. Rev., 131:1121.

Chaojun Xiao, Xueyu Hu, Zhiyuan Liu, Cunchao Tu, and Maosong Sun. 2021. Lawformer: A pre-trained language model for chinese legal long documents. AI Open, 2:79–84.

Chaojun Xiao, Haoxi Zhong, Zhipeng Guo, Cunchao Tu, Zhiyuan Liu, Maosong Sun, Yansong Feng, Xianpei Han, Zhen Hu, Heng Wang, et al. 2018. Cail2018: A large-scale legal dataset for judgment prediction. arXiv preprint arXiv:1807.02478.

Kai Yang. 2010. On the distinction between violence in robbery and theft. http://www.jsfy.gov.cn/ article/78069.html. Accessed: 2025-02-16.

Weikang Yuan, Junjie Cao, Zhuoren Jiang, Yangyang Kang, Jun Lin, Kaisong Song, Pengwei Yan, Changlong Sun, Xiaozhong Liu, et al. 2024. Can large language models grasp legal theories? enhance legal reasoning with insights from multi-agent collaboration. arXiv preprint arXiv:2410.02507.

Shengbin Yue, Wei Chen, Siyuan Wang, Bingxuan Li, Chenchen Shen, Shujun Liu, Yuxuan Zhou, Yao Xiao, Song Yun, Xuanjing Huang, et al. 2023. Disc-lawllm: Fine-tuning large language models for intelligent legal services. arXiv preprint arXiv:2309.11325.

Mingkai Zhang. 2007a. Normative elements of the constitutive requirements. Studies in Law, (06):76– 93.

Mingkai Zhang. 2010. Justification grounds and the system of crime constitution. The Jurist, (1):31–39, 176–177.

Wenxian Zhang and Wangsheng Zhou. 2007. Jurisprudence (3rd Edition). Higher Education Press, Beijing.

Zhihai Zhang. 2007b. On the violent elements of robbery. Legal System and Society, (1):222.

Guangquan Zhou. 2017. The hierarchical theory of crime and its practical development. Tsinghua Law Journal, 11(5):84–104.

Youchao Zhou, Heyan Huang, and Zhijing Wu. 2023. Boosting legal case retrieval by query content selection with large language models. In Proceedings of the Annual International ACM SIGIR Conference on Research and Development in Information Retrieval in the Asia Pacific Region, pages 176–184.

## A Charge Selection and Detailed Data Distribution

Charge Selection: To systematically determine charge frequency, we analyzed the CAIL2018 dataset (Xiao et al., 2018), which contains 2,676,075 criminal cases annotated with 183 criminal law articles and 202 criminal charges and a total of 3,010,000 criminal charges. Apart from few charges that have been merged or changed name, our dataset largely covers all criminal charges from CAIL2018 that have a frequency of over 3,000 (>0.099%) occurrences.

Length distribution for each element: Table 6.

## B Interpretation Methods

## 1. Literal Interpretation

A strict textual analysis method that adheres to the ordinary meaning of words as understood by a reasonable person at the time of enactment, excluding subjective intent inference

## 2. Systematic Interpretation

An approach interpreting legal articles through their position within the codified legal hierarchy and logical connections with related norms, maintaining the integrity of the legal system (aligned with Dworkin’s "law as integrity" theory).

## 3. Purposive Interpretation

A method discerning the objective legislative purpose through analysis of statutory structure and functional goals, distinct from subjective legislative intent (following Hart & Sacks’ legal process school).

## 4. Historical Interpretation

Interpretation based on legislative history materials, including drafts, debates, and official commentaries, while distinguishing original meaning from framers’ subjective intentions (as per Brest’s original understanding theory).

## 5. Comparative Interpretation

A methodology referencing functionally comparable legal systems sharing common juridical traditions, employing analogical reasoning while considering local legal culture (developed through Gottfried Wilhelm Leibniz’s comparative law framework).

## 6. Sociological Interpretation

Interpretation evaluating social efficacy through empirical analysis of implementation effects, guided by Pound’s sociological jurisprudence principle that “law must be measured by its achieved results”.

## C Prompt for LLM-generated FET

See Table 6.

## D Details about Pilot Study

We selected candidate models from LawBench (Fei et al., 2023) and LexEval (Li et al., 2024c), which contain the broadest and most up-to-date evaluation of legal LLMs. From these, we chose topperforming models such as GPT-4, Qwen-14Bchat, and representative legal-specific LLMs.

For best performance, we used GPT-4o (the latest version of GPT-4 at that time) and Qwen-plus (a stronger commercial variant of Qwen2.5-72B. (Aliyun model-studio official site))

![](images/8e68247e1ad5479d2e882c012590198f2573505587f755e5145d1e161c99493f.jpg)  
Figure 3: The average length distribution of the total four elements annotated by experts.

![](images/9b15a7b5bbfa6033855005bbcbd66fd3e8b66bd32d6966254cc246d6ee36f759.jpg)

![](images/eb6e84b9a959916966635378aef417a112a8fffe48d7da39e7d0afdd44ca21e3.jpg)

![](images/5577b76d9fdbfcea78ade80dbc9bd5b9d1573841e5f2ea270750176785129d33.jpg)

![](images/e2dbe837008932d0d573440785dd30ea44a1ab4c66e224491d0fb116d3b8fafc.jpg)  
Figure 4: The length distribution of each element annotated by experts.

![](images/d4b370baa6f22ff4f2cf47f73cca132ffdb0e1b2b0ceaead9c0aa6ac09f53d00.jpg)  
Figure 5: The average length distribution of the total four elements generated by LLM.

![](images/0d293fb6e9c7434206259e8fe7928f837be1c0ea4e8244ae5156aeee4a40e1a2.jpg)

![](images/25925383977c275589d7d7305bd1081706cae5adcf98cea26f3cd88ea47ce919.jpg)

![](images/1a86c2abdfd9cd156a91392c90acbeb59024a4b79672a4b002129c4f1599801b.jpg)

![](images/c7b466d6db9243d329675f7a7e954b00dd3e8583604d88a99662092a6440f2f1.jpg)  
Figure 6: The length distribution of each element generated by LLM.

<table><tr><td>You are an expert in criminal law. Based on the given charge, please analyze it according to China&#x27;s criminal law and output the four elements of the charge in order, including: - Object: The concretization of a certain abstract social interest. For example, the object of charges that infringe on personal rights is the right to life, while the object of property-related charges could be items such as mobile phones or wallets. - Objective Aspect: The objective facts of the criminal act, including the key actions that trigger</td></tr><tr><td>the charge (e.g., theft, robbery) and the consequences caused by the act (e.g., serious injury, death, property loss).</td></tr><tr><td>- Subject: Typically, the general subject of the charge, but in some cases, a specific subject is required (e.g., government officials in certain offenses). - Subjective Aspect: The mental state of the perpetrator, such as intent or negligence.</td></tr><tr><td>Relevant Legal Articles: [] Please synthesize the above information to generate a refined set of four elements that represent the</td></tr><tr><td>characteristics of the charge. Output format: { &quot;Crime&quot;: &quot;&quot;, &quot;Four-elements of the Crime&quot;: { &quot;Crime Object&quot;: &quot;&quot;, &quot;Objective</td></tr><tr><td>Aspect&quot;: &quot;&quot;, &quot;Subject&quot;: &quot;&quot;, &quot;Subjective Aspect&quot;: &quot;&quot; } } Crime: []</td></tr></table>

Table 6: Prompt template for generating the four elements of a Crime using LLMs

During implementation, we found that most legal LLMs were unavailable. The only stably accessible one was Farui (A leading legal LLM built on Qwen, (Aliyun model-studio official site, Tongyi Farui)), specifically the version “tongyifarui-890” from its official API.

To compare GPT-4o, Qwen-plus, and tongyifarui-890, we sampled 300 cases from our legal retrieval dataset and asked models to perform charge prediction, which is the pre-task for generating case-FETs.

(For each case in legal retrieval, the model was required to predict charges, so we can match charges’ expert-FETs, and use them to generate case-FET.)

This task involved all criminal charges, including multi-defendant and multi-charge scenarios, and requires models to predict charges from open text without a predefined list, making it a challenging legal task.

The result showed that GPT-4o (59.78%) > Qwen-plus (58.70%) » tongyifarui-890 (21%). Given Farui’s poor performance, we did not include it in subsequent experiments.

We further evaluated GPT-4o and Qwen-plus based on their ability to generate case-FETs. The results showed that GPT-4o outperformed Qwenplus (MRR 0.2140 vs. 0.2052). Considering both results, we adopted GPT-4o as our primary model in the paper.

Subsequently, in efforts to improve charge prediction for matching charges’ expert-FETs, we found that Qwen-plus performed better than GPT-4o when a charge list was provided (58.70%- >80.43% vs. 59.78%->71.74%). Therefore, in this specific setting for charge prediction before retrieval, we used Qwen-plus.

For fair and reproducible presentation of results on specific downstream tasks (SCD and LCR), as mentioned in the main text, we present the results of open source Qwen2.5-72b.

## E Human Evaluation Guidance

The annotators included three postgraduate students specializing in criminal law and one master’s student in legal science and technology. The annotators scored independently, without knowledge of each other’s results. Before scoring, they were asked to read the descriptions and scoring guidelines (as shown in Table 7) for each evaluation dimension. In order to ensure the fairness of the evaluation, they do not know the source of the four elements, and they do not know that the four elements include those generated by LLMs.

When assigning scores, they were also required to provide brief justifications. For example, for the Completeness dimension: 3 (The description of Objective Aspect is too brief, and does not specify the intent of illegal possession).

## F Details for Similar Charge Disambiguation

For LLM baselines, we evaluate both generalpurpose and task-specific methods.

GPT-4o is an optimized version of GPT-4 (Achiam et al., 2023) that has well performance in specific tasks through domain adaptation.

To explore the effectiveness of expert-FETs, we further consider other methods that introduced the Four-element Theory into LLMs.

$\mathbf { G P T - 4 0 _ { L a w } }$ , which introduces articles related to corresponding charges into the instruction to provide legal context.

Legal-COT is a variant of COT (Kojima et al., 2022) that guides the LLM to perform step-by-step legal reasoning by incorporating explanations of the Four-element theory into the instruction.

MALR is an up-to-date multi-agent framework designed to enhance complex legal reasoning (Yuan et al., 2024), enabling LLMs to autonomously decompose legal tasks and extract insights from legal rules. As its full implementation is not publicly available, we use the released code for the autoplanner module and implement the legal insight extraction following the specified steps and prompts, with necessary refinements. Experiments on the paper’s reported examples show that our implementation produces task decompositions and outputs largely consistent with the original results.

As shown in Table 10, different methods differ in their prompts for generating and explaining the Four-Element Theory, but generally follow a similar process. For the SCD output, except for COT and MALR, which require reasoning processes and prediction results, all other methods only require the output of prediction results.

## G Baselines in Legal Case Retrieval

BERT (Devlin, 2018) is a language model widely used in retrieval tasks. In this paper, we chose BERT-base-Chinese<sup>5</sup>. Legal-BERT<sup>6</sup> (Chalkidis et al., 2020) is a variant of BERT that is specifically trained on legal corpora. Lawformer (Xiao et al., 2021)is a Chinese legal pre-trained model based on Longformer (Beltagy et al., 2020), which is able to process long texts in the legal domain. ChatLaw-Text2Vec<sup>7</sup> (Cui et al., 2023) is a Chinese legal LLM trained on 936,727 legal cases for similarity calculation of legal-related texts. SAILER (Li et al., 2023) is a structure-aware legal case retrieval model utilizing the structural information in legal case documents.

Baseline results are provided in Table 12.

To support reproducibility, we provide the full prompt templates used in our pipeline. Table 13 shows the prompt for charge prediction, and Table 14 presents the prompt used for generating fourelement annotations in both BGE+FET(LLM) and BGE+FET(Expert, LLM).

## H SCR results on the full LeCaRDv2 Dataset

As presented in Table 15, we selected several representative methods based on sparse retrieval and dense retrieval for experiments on the full LeCaRDv2 dataset. All language models were not fine-tuned. The expert-guided FET method achieved the best performance among all language models, attaining top results in both R@500 and R@1000. The results indicate that the conclusions drawn from the full dataset are consistent with those from the subset, and the expert-guided method demonstrates strong performance.

## I A Case Study of LCR

Table 16 presents a case study on the Crime of Embezzlement. By comparing the expert-FETs for the charges in JUREX-4E, the case-FETs generated directly by the LLM, and those generated by the LLM with expert-FETs of charge as guidance, we can observe that:

1) Incorporating expert fine-grained annotations enables the model to better grasp the elements of a crime, thereby providing more precise element comparison. For example, LLMs can identify the “integrity of official duties”, and the subjective aspect “Intentional” can be interpreted as “having the purpose of illegally possessing public or private property”, highlighting the characteristics of “official duties”. Capturing the core information of the case is crucial for matching cases with similar facts.

2) LLMs can conduct case-tailored specific analysis based on the constitutive elements of a crime.

<table><tr><td>Dimension</td><td>Precision</td><td>Completeness</td><td>Representativeness</td><td>Standardization</td></tr><tr><td>Definition</td><td>Whether there are errors in key elements</td><td>Whether the four- elements are complete</td><td>Whether key elements and scenarios are empha- sized</td><td>Whether language and format are clear and stan- dardized</td></tr><tr><td>Score 1</td><td>Contains numerous obvi- ous errors, severely im- peding the judgment of culpability, exculpation, and conviction, leading to significant deviations.</td><td>Severe omission of key content, unable to present a complete picture of the crime structure, greatly hinder- ing analysis of criminal</td><td>Completely fails to men- tion any key elements or scenarios, unable to high- light essential points for crime recognition, offer- ing no assistance in con-</td><td>Language is extremely chaotic and obscure; for- mat lacks any standard- ization, greatly hindering comprehension and ap- plication.</td></tr><tr><td>Score 2</td><td>Contains multiple notice- able errors, significantly interfering with culpabil- ity, exculpation, and con- viction judgments, poten- tially leading to partial er-</td><td>behavior. Noticeable omissions in content, failing to com- prehensively cover crime elements, affecting thor- ough analysis of criminal behavior.</td><td>viction. Only highlights a mini- mal and unimportant por- tion of the key elements, providing weak support for understanding key crime features.</td><td>Language is relatively vague and inaccurate, with a casual format that makes content com- prehension significantly challenging.</td></tr><tr><td>Score 3</td><td>rors. Contains a few errors, but the overall accuracy in determining culpabil- ity, exculpation, and con- viction is relatively unaf- fected, unlikely to lead to</td><td>Some key content descriptions are incom- plete, but they generally present the framework of the crime structure.</td><td>Highlights some rela- tively important key ele- ments but lacks compre- hensiveness and promi- nence, offering limited assistance in crime iden-</td><td>Language is generally clear but may have minor deviations in phrasing or formatting.</td></tr><tr><td>Score 4</td><td>judgment errors. Almost error-free, key elements accurately serve culpability, excul- pation, and conviction judgments, ensuring the</td><td>Key elements are mostly complete, with only very slight and non-critical deficiencies that do not hinder a comprehensive analysis of the crime.</td><td>tification. Clearly and relatively comprehensively high- lights key elements, aiding in accurately iden- tifying crucial aspects of criminal behavior.</td><td>Language is clear and accurate, format is rel- atively standardized, fa- cilitating comprehension and application of rele- vant content.</td></tr><tr><td>Score 5</td><td>accuracy of results. Completely error-free, key elements are pre- cisely defined, achieving highly accurate culpa- bility, exculpation, and conviction judgments without any flaws.</td><td>All four elements are complete and detailed, covering every aspect of the crime, perfectly pre- senting the crime struc- ture.</td><td>Precisely and compre- hensively highlights all crucial elements, en- abling immediate grasp of the core aspects of the crime, significantly aiding conviction.</td><td>Language is extremely clear, standardized, and concise; format perfectly meets requirements, with no barriers to understand- ing, ensuring efficient in- formation delivery.</td></tr></table>

Table 7: The four dimensions of the human evaluation and the specific score description.

<table><tr><td>Charge Sets</td><td>Charges</td><td>Cases</td></tr><tr><td>F&amp;E</td><td>Fraud &amp; Extortion</td><td>3536 / 2149</td></tr><tr><td>E&amp;MPF</td><td>Embezzlement &amp; Mis- appropriation of Public Funds</td><td>2391 / 1998</td></tr><tr><td>AP&amp;DD</td><td>Abuse of Power &amp; Dere- liction of Duty</td><td>1950 / 1938</td></tr></table>

Table 8: Distribution of charges in the GCI dataset. Cases denote the number of cases in each category. Following (Liu et al., 2021), for a case with both confusable charges, the prediction of any one of the charges is considered correct.

Blue parts show the LLMs can better analyze the defendant’s workplace and the actions taken in the case, which reflects the significance of specific and accurate legal knowledge.

<table><tr><td>Prompt:</td></tr><tr><td>You are a lawyer specializing in criminal law. Based on Chinese criminal law,</td></tr><tr><td>Please determine which of the following candidate charges the given facts align with.</td></tr><tr><td>The candidate charges and their corresponding four-elements are as follows:</td></tr><tr><td>[four-elements of Candidate Charges].</td></tr><tr><td>The four elements represent the core factors for determining the constitution of a criminal charge.</td></tr><tr><td>[The basic concepts of the Four-Element Theory]</td></tr><tr><td>Please compare the case facts to determine which charge&#x27;s four elements they align with, thereby identifying the charge.</td></tr></table>

Table 9: Prompt template for adding the Four-Element Theory and specific four-elements of crime in charge disambiguation.

<table><tr><td>Method</td><td>GPT-40</td><td>GPT- 40+Article</td><td>Legal-COT</td><td>GPT- 40+FETLLM</td><td>GPT- 40+FETExperts</td></tr><tr><td>Pre-task</td><td>None</td><td>None</td><td>None</td><td>LLM- generated FETs</td><td>Expert- annotated FETs</td></tr><tr><td rowspan="3">Prompt</td><td colspan="5">You are a lawyer specializing in criminal law. Based on Chinese criminal law, please determine which of the following candidate charges the given facts align with. The candidate Please ana- The candidate charges and their charges and rel- lyze using the corresponding four-elements are</td></tr><tr><td>Candidate charges are as follows: #Candidate Charges</td><td>evant legal arti- four-elements cles are as fol- Theory step by lows: #Candi- step: #details date Charges + about each step. #Articles</td><td>The candidate charges are as follows: #Candidate Charges</td><td>as follows: #four-elements of candidate charges. The four elements represent the four core factors of a charge. Compare the case facts to determine which charge&#x27;s four elements they align with, thereby identifying the charge.</td><td></td></tr><tr><td colspan="4">Output format: #Format. Note: Only output the charge, no additional information, Case facts: #Case Facts.</td><td></td><td></td></tr></table>

Table 10: Prompts of different methods in Similar Charge Disambiguation. # represents a format input.

<table><tr><td>FET-LLM vs FET-Expert</td><td>F&amp;E</td><td>E&amp;MPF</td><td>AP&amp;DD</td></tr><tr><td>FET-qwen vs FET-Expert</td><td>0.00215</td><td>0.00070</td><td>0.00509</td></tr><tr><td>FET-farui vs FET-Expert</td><td>0.00000</td><td>0.00126</td><td>0.00516</td></tr><tr><td>FET-4o+farui+qwen vs FET-Expert</td><td>0.00000</td><td>0.00996</td><td>0.03415</td></tr><tr><td>FET-4o vs FET-Expert</td><td>0.02246</td><td>0.00156</td><td>0.02251</td></tr></table>

Table 11: McNemar’s test results (p-values) comparing LLM-generated FET and expert-FET. Statistically significant improvements $( p < 0 . 0 5 )$ are observed across all tasks.

<table><tr><td>Model</td><td>NDCG@10</td><td>NDCG@20</td><td>NDCG@30</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@20</td><td>R@30</td><td>MRR</td></tr><tr><td>BERT</td><td>0.1511</td><td>0.1794</td><td>0.1978</td><td>0.0199</td><td>0.0753</td><td>0.1299</td><td>0.2157</td><td>0.2579</td><td>0.1136</td></tr><tr><td>Legal-BERT</td><td>0.1300</td><td>0.1487</td><td>0.1649</td><td>0.0186</td><td>0.0542</td><td>0.1309</td><td>0.1822</td><td>0.2172</td><td>0.0573</td></tr><tr><td>Lawformer</td><td>0.2684</td><td>0.3049</td><td>0.3560</td><td>0.0432</td><td>0.1479</td><td>0.2330</td><td>0.3349</td><td>0.4683</td><td>0.1096</td></tr><tr><td>ChatLaw-Text2Vec</td><td>0.2049</td><td>0.2328</td><td>0.2745</td><td>0.0353</td><td>0.1306</td><td>0.1913</td><td>0.2684</td><td>0.3751</td><td>0.1285</td></tr><tr><td>SAILER</td><td>0.3142</td><td>0.4133</td><td>0.4745</td><td>0.0539</td><td>0.1780</td><td>0.3442</td><td>0.5688</td><td>0.7092</td><td>0.1427</td></tr><tr><td>BGE (case_fact only)</td><td>0.4737</td><td>0.5539</td><td>0.5937</td><td>0.0793</td><td>0.2945</td><td>0.4298</td><td>0.6500</td><td>0.7394</td><td>0.1926</td></tr><tr><td>BGE+FET (Qwen2.5)</td><td>0.5125</td><td>0.5858</td><td>0.6350</td><td>0.1104</td><td>0.2870</td><td>0.4653</td><td>0.6679</td><td>0.7836</td><td>0.2168</td></tr><tr><td>FET only</td><td>0.3367</td><td>0.3971</td><td>0.4487</td><td>0.0622</td><td>0.2006</td><td>0.3279</td><td>0.4806</td><td>0.6037</td><td>0.1524</td></tr><tr><td>BGE+FET (Expert, Qwen2.5)</td><td>0.5295</td><td>0.5979</td><td>0.6416</td><td>0.1124</td><td>0.3122</td><td>0.4838</td><td>0.6791</td><td>0.7824</td><td>0.2206</td></tr><tr><td>FET only</td><td>0.3354</td><td>0.4035</td><td>0.4541</td><td>0.0849</td><td>0.1923</td><td>0.3076</td><td>0.4839</td><td>0.6097</td><td>0.1606</td></tr><tr><td>BGE+FET (GPT-40)</td><td>0.5139</td><td>0.5862</td><td>0.6291</td><td>0.0980</td><td>0.2967</td><td>0.4769</td><td>0.6802</td><td>0.7828</td><td>0.2140</td></tr><tr><td>FET only</td><td>0.3583</td><td>0.4293</td><td>0.4798</td><td>0.0506</td><td>0.2240</td><td>0.3644</td><td>0.5383</td><td>0.6652</td><td>0.1453</td></tr><tr><td>BGE+FET (Expert, GPT-40)</td><td>0.5211</td><td>0.5920</td><td>0.6379</td><td>0.1024</td><td>0.3049</td><td>0.4883</td><td>0.6885</td><td>0.7967</td><td>0.2155</td></tr><tr><td>FET only</td><td>0.3766</td><td>0.4584</td><td>0.5111</td><td>0.0715</td><td>0.1894</td><td>0.3709</td><td>0.5891</td><td>0.7203</td><td>0.1624</td></tr></table>

Table 12: Performance on the Legal Charge Retrieval (LCR) task with baselines. Highest results are in bold. “FET only” indicates using the four-element descriptions without case facts.

![](images/853afcc3752f5f1902105b43110742508026b9c7241c3afdc1e925dd59b67ab6.jpg)  
Table 13: Prompt used for charge prediction.

Prompt 2: BGE+FET(LLM) and BGE+FET(Expert, LLM).   
You are a legal expert specializing in criminal law. Based on Chinese criminal law knowledge, analyze   
the following case facts and provide the following information in sequence:   
1. The four elements of the crime:   
- Criminal Object: The tangible or intangible interests being infringed upon (e.g., personal rights   
such as life, or property rights such as money, vehicles).   
- Objective Aspect: The objective facts of the criminal activity, including key actions (e.g., theft,   
robbery) and consequences (e.g., injury, death, loss).   
- Criminal Subject: Typically general subjects; special subjects in certain crimes (e.g., government   
officials).   
- Subjective Aspect: Whether the act was intentional or negligent.   
2. Charge: Only output the specific crime name(s).   
3. Relevant Legal Articles: Only output the article number(s) of the relevant laws.   
Output format: JSON. For each crime involved in the case, provide a separate dictionary entry.   
[Output Sample]   
"Crime 1": {   
"Four elements": {   
"Criminal Object": "Personal rights: the victim Wang’s right to life; Property rights: vehicle.",   
"Objective Aspect": "The defendant Wu drove under the influence and collided with the victim   
Wang, causing Wang’s immediate death and vehicle damage.",   
"Criminal Subject": "Defendant Wu, the driver.",   
"Subjective Aspect": "Negligence"   
},   
"Charge": "Traffic Accident Crime",   
"Relevant Legal Article": "Article 133"   
},   
"Crime 2": {   
"Four elements": {   
"Criminal Object": "Social management order: infringement on the state’s document management   
system; Property rights: forged documents and related items.",   
"Objective Aspect": "Defendant 1 purchased equipment and materials to forge documents.   
Defendant 2 delivered the forged documents. Defendant 3 facilitated transactions via the internet,   
handling payments and document transfers.",   
"Criminal Subject": "Multiple defendants, all individuals with full criminal responsibility.",   
"Subjective Aspect": "Intentional"   
},   
"Charge": "Forgery, Alteration, or Sale of Official Documents, Certificates, and Seals of State   
Organs",   
"Relevant Legal Article": "Article 280, Paragraph 1"   
}   
} }  
Table 14: Prompt for generating four-element annotations used in $\mathrm { F E T } _ { \mathrm { L L M } }$ and $\mathrm { F E T _ { E x p e r t \_ G u i d e d } } .$

<table><tr><td>Model</td><td>R@100</td><td>R@200</td><td>R@500</td><td>R@1000</td></tr><tr><td>Legal-BERT Lawformer</td><td>0.1116 0.2432</td><td>0.1493 0.304</td><td>0.2174 0.4054</td><td>0.2819 0.4833</td></tr><tr><td>ChatLaw-Text2Vec SAILER</td><td>0.1045 0.2834</td><td>0.1628 0.4033</td><td>0.2791 0.6104</td><td>0.3999 0.7568</td></tr><tr><td>BGE</td><td>0.4085</td><td>0.5246</td><td>0.6855</td><td>0.7912</td></tr><tr><td>BGE+FET(GPT-40)</td><td>0.4167</td><td>0.5388</td><td>0.7006</td><td>0.7925</td></tr><tr><td>BGE+FET(Expert, GPT-4o)</td><td>0.4201</td><td>0.5396</td><td>0.7010</td><td>0.7927</td></tr></table>

Table 15: SCR results on the full set of LeCaRDv2. Bold fonts indicate leading results in each setting. The expert-guided FET method achieved the best performance among all language models and attained the top results in both R@500 and R@1000.

<table><tr><td rowspan=1 colspan=1>Document</td><td rowspan=1 colspan=4>[Head of document]...In April 201X, Company A appointed B as the Sales Managerand Deputy Manager of the Catering Department, responsible for collecting outstand-ing debts. From December 201X to April 201Y, B embezzled X yuan from six units’payments for personal investment and consumption. In May 201Y, knowing that thecompany required him to reconcile accounts with the finance department, B fled afterchanging his phone number. It was found that after the case broke out, B&#x27;s father Chad returned the X yuan to Company A.</td></tr><tr><td rowspan=1 colspan=1>Charge</td><td rowspan=1 colspan=4>Crime of Embezzlement</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Expert-FETs</td><td rowspan=1 colspan=2>LLM-generated</td><td rowspan=1 colspan=1>Expert-guided</td></tr><tr><td rowspan=3 colspan=1>FETs</td><td rowspan=3 colspan=1>Object: The ownership ofproperty and integrity ofduties in companies, en-terprises, or other organi-zations. The objects in-clude movable and immov-able, tangible and intangi-ble property of the perpe-trator&#x27;s organization.Objective Aspect: Usingpositional convenience toillegally appropriate orga-nizational property in asignificant amount.Subject: Special sub-ject, namely personnel ofcompanies, enterprises, orother organizations.Subjective            As-pect:Intentional,   withthe purpose of illegallypossessing property.</td><td rowspan=2 colspan=2>Object: The ownership ofproperty of companies, en-terprises, or other organi-zations.Objective Aspect: Usingthe convenience of one&#x27;sposition to illegally appro-priate the property of theorganization, in a signifi-cant amount.Subject: Personnel ofcompanies, enterprises, orother organizations. Thedefendant is B.Subjective Aspect: Inten-tional.</td><td rowspan=3 colspan=1>Object: The ownership ofproperty of companies, en-terprises, or other organi-zations and the integrity ofofficial duties. In this case,crime objects are the prop-erties of the company, en-terprise, or other organiza-tion where the perpetratoris employed.Objective Aspect: Usingone&#x27;s position for personalgain by illegally appropri-ating the property of theorganization, where theamount involved is signifi-cant. Specifically, B usedhis position to embezzle Xyuan from the paymentsof six units for personalinvestment and consump-tion.Subject: Special sub-ject, namely personnel ofcompanies, enterprises, orother organizations. De-fendant B is the SalesManager and Deputy Man-ager of the Catering De-partment of the company.Subjective Aspect: Inten-tional, with the purpose ofillegally possessing publicor private property.</td></tr><tr><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2></td></tr></table>

Table 16: Results of the case-FETs using different methods for LCR. Expert-FETs refer to expert annotations of the predicted charges. LLM-generated and Expert-guided indicate whether the Expert-FETs of predicted charges are used to guide the LLM in generating the case-FETs. Red parts indicate knowledge from experts, while blue parts represent the LLM’s internal knowledge. By incorporating JUREX-4E, the model better emphasizes convictionand sentencing-related information and provides more detailed descriptions of key case facts.