# SafeScientist: Enhancing AI Scientist Safety for Risk-Aware Scientific Discovery

Kunlun Zhu†\*, Jiaxun Zhang†, Ziheng Qi<sup>†</sup>, Nuoxing Shang   
Zijia Liu, Peixuan Han, Yue Su, Haofei Yu, Jiaxuan You <sup>1</sup>University of Illinois Urbana-Champaign {kunlunz2, jiaxuan}@illinois.edu

## Abstract

Recent advancements in large language model (LLM) agents have significantly accelerated scientific discovery automation, yet concurrently raised critical ethical and safety concerns. To systematically address these challenges, we introduce SafeScientist, an innovative AI scientist framework explicitly designed to enhance safety and ethical responsibility in AI-driven scientific exploration. SafeScientist proactively refuses ethically inappropriate or high-risk tasks and rigorously emphasizes safety throughout the research process. To achieve comprehensive safety oversight, we integrate multiple defensive mechanisms, including prompt monitoring, agent-collaboration monitoring, tooluse monitoring, and an ethical reviewer component. Complementing SafeScientist, we propose SciSafetyBench, a novel benchmark specifically designed to evaluate AI safety in scientific contexts, comprising 240 high-risk scientific tasks across 6 domains, alongside 30 specially designed scientific tools and 120 tool-related risk tasks. Extensive experiments demonstrate that SafeScientist significantly improves safety performance by 35% compared to traditional AI scientist frameworks, without compromising scientific output quality. Additionally, we rigorously validate the robustness of our safety pipeline against diverse adversarial attack methods, further confirming the effectiveness of our integrated approach. The code and data will be available at https://github. com/ulab-uiuc/SafeScientist. Warning: this paper contains example data that may be offensive or harmful.

## 1 Introduction

Recent advancements in Artificial Intelligence (AI), particularly with the proliferation of powerful Large Language Models (LLMs) such as Gemini-2.5-Pro (Team et al., 2023), GPT-o3 (OpenAI,

2024), and DeepSeek-V3 (Liu et al., 2024), have substantially reshaped the landscape of scientific research. These models are increasingly capable of automating complex tasks including hypothesis generation, experimental design, data analysis, and even manuscript preparation (Sakana, 2024; Yu et al., 2024). The potential for AI to accelerate discovery is immense, with several works surveying the broad applications of LLMs in science (Zhang et al., 2024c; Luo et al., 2025; Zhang et al., 2025; Taylor et al., 2022).

![](images/2b1e18b191d17408272a3cc85b9dff5aa944dfb35695968952984f7446b0ce53.jpg)  
Figure 1: SafeScientist vs. Normal Scientist. Unlike a normal AI scientist that may respond unsafely to malicious or risky prompts, SafeScientist can reject harmful queries and responsibly handle high-risk topics under safety-aware guidance.

Despite these promising developments, the integration of AI-driven agents into research processes introduces significant ethical and safety risks (Bengio et al., 2025a; Feng et al., 2024; Liu et al., 2025). These include the potential for malicious exploitation, the perpetuation and amplification of harmful biases, and the inadvertent propagation of misinformation or hazardous knowledge (Tang et al., 2024; Shamsujjoha et al., 2024; Tang et al., 2024; Deng et al., 2024; Dong et al., 2024; He et al., 2024). Much of the existing literature on LLM safety has primarily focused on isolated aspects, such as adversarial attacks on single models (Wei et al., 2024; Zou et al., 2023; Li et al., 2024), pretraining data biases (Feng et al., 2023), or specific defense mechanisms like safety fine-tuning (Ouyang et al., 2022; Bai et al., 2022; Choi et al., 2024) and runtime monitoring (Yuan et al., 2024; Wang et al., 2023b; Inan et al., 2023). However, these studies often neglect the holistic dynamics and emergent risks within multi-agent scientific environments (Guo et al., 2024; Huang et al., 2024; Osman and d’Inverno, 2023; Cheng et al., 2024), where complex interactions can lead to unforeseen safety challenges (Tian et al., 2023; Zhang et al., 2024a; Ju et al., 2024). Consequently, there is an urgent and growing need for comprehensive evaluation benchmarks and robust defensive frameworks tailored explicitly for AI-enabled scientific communities.

![](images/0737c090ce793e4e9df6ad155743c62e8adb4e9689183562cc3adca83fd861c5.jpg)

![](images/f02494c618fb53fb3556489705104d17c611382cfd1cfb714b1ffdafe3bdc489.jpg)  
Figure 2: Overview of the SafeScientist . An end-to-end pipeline from task to paper, integrating input detection, discussion, tool use, and writing stages, with SciSafetyBench-based attack/defense evaluation for scientific AI safety.

Despite the current success in agent-level safeguard (Wang et al., 2025; Sun et al., 2023), tailored design risk-aware AI Scientist frameworks are still underexplored. To systematically address these critical challenges in AI-driven scientific exploration, we are the first to introduce SafeScientist, an innovative AI scientist framework explicitly designed to prioritize safety and ethical responsibility. Safe-Scientist proactively refuses high-risk or ethically inappropriate tasks and maintains thorough safety oversight via an integrated, multi-layered defense system, including: (1) Prompt Monitor, (2) Agent Collaboration Monitor, (3) Tool-Use Monitor, and (4) Paper Ethic Reviewer.

To effectively benchmark SafeScientist and similar AI scientist frameworks, we further propose SciSafetyBench, a specialized benchmark explicitly designed to evaluate AI safety within scientific contexts. SciSafetyBench comprises two main components: (1) a collection of 240 risks evaluation scientific discovery tasks spanning six scientific domains (Physics, Chemistry, Biology, Material Science, Computer Science, and Medicine), categorized by four distinct risk sources; and (2) a set of 30 representative scientific tools accompanied by 120 detailed tool-specific risk scenarios, designed to critically assess AI agents’ handling of realistic laboratory safety concerns.

Extensive experiments demonstrate that SafeScientist significantly enhances safety performance by achieving an 34.69% improvement (insert specific metric and value) over traditional AI scientist frameworks lacking integrated safeguards, without compromising scientific output quality. Moreover, rigorous validation against diverse adversarial attack methods affirms the robustness and effectiveness of our integrated safety pipeline. Collectively, this work emphasizes the necessity and practicality of proactive, safety-oriented design in AI scientific discovery, contributing directly toward more responsible, trustworthy, and beneficial scientific AI systems.

Our primary contributions are: 1) We propose SafeScientist, an AI scientist framework integrating proactive prompt monitoring, agent collaboration oversight, tool-use constraints, and ethical review to ensure safety and ethical compliance. 2) We introduce SciSafetyBench, a benchmark with 240 high-risk discovery tasks and 120 tool-specific risk tasks across six scientific domains for evaluating AI scientist safety. 3) We implement diverse adversarial attacks to rigorously validate the robustness and effectiveness of SafeScientist and SciSafetyBench.

## 2 Related Work

LLM Safety Avoiding the generation of harmful content to individuals or society is a critical principle in the responsible deployment of LLMs. To challenge LM safety, researchers have developed various attack methods, methods, including prompt injection (Wang et al., 2023b; Xie et al., 2024; Shen et al., 2024; Kumar et al., 2023), backdoor attacks (Zhao et al., 2024b; Li et al., 2022), and autonomous prompt jailbreaking (Zou et al., 2023; Huang et al., 2025).

LLM safety can be enhanced through internal and external methods. Internally, prompt engineering (Chen et al., 2024a; Zheng et al., 2024), supervised fine-tuning (Choi et al., 2024), and reinforcement learning from human feedback (Ouyang et al., 2022; Bai et al., 2022; Mu et al., 2024; Xiong et al., 2024) are commonly used to equip LLMs with safety awareness. More delicate safety enhancement methods involve modifying LLMs’ hidden representations about harmful content, enhancing safety in a parameter-efficient manner (Li et al., 2024; Zou et al., 2024; Rosati et al., 2024). Externally, harmful content detectors (Inan et al., 2023), bad intention predictors (Han et al., 2025) and behavioral steers (Arditi et al., 2024; Han et al., 2023) serve as plug-and-play modules to ensure safety.

LLM Agent Safety Recent advancements endowed LLMs with tool-calling and planning abilities, making them AI agents that can proactively interact with and influence the environment (Cheng et al., 2024; Guo et al., 2024). Such progress brings promising applications and security risks at the same time, including tool response injection (Debenedetti et al., 2024), long-term memory poisoning (Chen et al., 2024b; Dong et al., 2025), and malicious agent in collaboration (He et al., 2025; Lee and Tiwari, 2024). In addition, LLM-agent-related security loopholes may severely impact the environment through malicious actions (Tian et al., 2023; Zhang et al., 2024a) or the spread of misinformation (Ju et al., 2024). To address these risks, several agent-level safeguards (Zhou et al., 2024; Wang et al., 2025; Sun et al., 2023; Mao et al., 2025) and testbeds for agent safety (Zhang et al., 2024b; Yin et al., 2024;

Debenedetti et al., 2024; Andriushchenko et al., 2024) have been proposed. However, specialized considerations for scientific research scenarios remain largely unexplored.

AI Scientists We have witnessed remarkable progress in AI scientists’ recent years, which are involved in multiple steps in research (Luo et al., 2025) and across multiple disciplines (Zhang et al., 2025, 2024c; Wang et al., 2023a). Several AI scientist frameworks (Lu et al., 2024; Schmidgall et al., 2025; Yuan et al., 2025; Weng et al., 2024) and benchmarks (Qiu et al., 2025; Li and Zhan, 2022) are also proposed, aiming to generate research findings end-to-end. While most AI scientists are currently limited to simulated research, considering and mitigating their risks in real-world applications beforehand is meaningful (Bengio et al., 2025b).

## 3 Method

## 3.1 A Safe AI Scientist Framework

Inspired by recent agentic frameworks such as AI Scientist (Sakana, 2024) and Tiny Scientist (Yu et al., 2025), we propose SafeScientist, a lightweight yet secure framework for automating scientific research. As illustrated in Figure 2, the research pipeline initiates from a user instruction, which is first analyzed to identify the scientific domain and task type. Based on this initial analysis, an appropriate ensemble of expert agents—including domain-specific researchers, general-purpose survey writers, and experimental planners—is dynamically activated to perform a group discussion.

Details of the group discussion chat history can be viewed at the Appendix 26.

These agents collaboratively generate and iteratively refine a scientific idea. Once a promising idea is identified, relevant scientific tools and retrieval modules (e.g., web search, scientific literature search, and domain-specific simulation tools) are invoked to gather necessary information, perform simulations, and analyze outcomes. Finally, the resulting findings are synthesized through dedicated writing and refinement modules, producing a structured, thoroughly cited, and high-quality research paper draft.

To ensure secure and responsible automation throughout this process, SafeScientist integrates several lightweight yet effective safety mechanisms. These defensive components include the Prompt Monitor, the Agent Collaboration Monitor, the

<table><tr><td>Framework</td><td colspan="6">Ethic Rev. Writ. Disc. Input Safety Agent Def. Tool Def.</td><td>Tools</td></tr><tr><td>AI Scientist (Sakana, 2024)</td><td></td><td></td><td></td><td>X</td><td>X</td><td>X</td><td>Aider, Semantic Scholar</td></tr><tr><td>CycleResearcher (Weng et al., 2024)</td><td></td><td></td><td></td><td>X</td><td>X</td><td>X</td><td>Ethical Detection Tool</td></tr><tr><td>ResearchTown (Yu et al., 2024)</td><td></td><td></td><td></td><td>X</td><td>X</td><td>X</td><td>Websearch, Arxiv</td></tr><tr><td>AI co-scientist (Gottweis et al., 2025)</td><td></td><td>X</td><td></td><td></td><td>X</td><td>X</td><td>Web search, AlphaFold</td></tr><tr><td>Agent Laboratory (Schmidgall et al., 2025)</td><td></td><td></td><td></td><td></td><td>X</td><td>X</td><td>arXiv API, HF Datasets, etc</td></tr><tr><td>SafeScientist (this work)</td><td></td><td></td><td></td><td></td><td></td><td></td><td>Search Tools, 30 science tools</td></tr></table>

Table 1: Comparison of safety and capability coverage across AI research-agent frameworks. Columns are ordered so that the distribution of checkmarks forms an inverted triangle—from universally supported functions on the left to rarer protections on the right. Rev., Writ., Disc., and Def. are abbreviations for Review, Writing, Discussion, and Defender, respectively.

Tool-Use Monitor, and the Paper Ethic Reviewer, collectively safeguarding the entire scientific exploration pipeline.

## 3.2 Defense Methods

Specifically, to address the safety issues SafeScientist consists of the following components. Details of the prompts of methods below can be viewed at the Appendix B

• Prompt Monitor: We adopt LLaMA-Guard (Inan et al., 2023), an effective LLM-based risk detector, to screen inputs and identify adversarial prompt injections. Our monitoring pipeline integrates two complementary stages for robust detection. First, LLaMA-Guard-3-8B evaluates the semantic intent and associated risks of the prompt, generating a safety label with explanatory rationale. Second, SafeChecker, a structural analyzer, scans prompts for known attack patterns—such as jailbreak attempts or roleplay exploits—and classifies each into three labels: pass, warning, or reject. The warning label means even though the research is risky, it is still worth exploring. It assesses 17 distinct risk categories and provides justifications for its classification. We fuse these analyses by rejecting prompts flagged by either LLaMA-Guard or SafeChecker, ensuring comprehensive threat detection. A lightweight fallback mechanism addresses ambiguous cases without compromising risk assessment integrity.

• Agent Collaboration Monitor: In the multiagent interaction stage, a monitor agent with focus on ethics and safety continuously monitors discussions, providing corrective ethical interventions against potential malicious agent influences.

• Tool-Use Monitor: We utilize a specialized detector to oversee tool interactions. Equipped with domain knowledge and tool operation guidelines, the tool-use detector effectively identifies unsafe usage of simulated scientific tools, avoiding misuse and potential risk regarding experimental tools.

• Paper Ethic Rewiewer: We adopt an ethical reviewer before the AI scientist pipeline produces a research outcome. The reviewer ensures that the paper adheres to research norms, collected from ethical standards of top Conferences like ACL<sup>1</sup> and NeurIPS<sup>2</sup>, before dissemination, ensuring the safety of AI scientists from the output level.

## 3.3 Attack Methods

To comprehensively evaluate AI Scientist safety, we design three types of attacks in the AI Scientist workflow, which are illustrated in Figure 2.

## 3.3.1 Query Injection

To comprehensively assess the robustness of AI Scientists against malicious attempts, we employ 7 query injection methods designed to obscure risky topics and make them harder to detect.

We utilize three Query Transformation techniques to make risks in the queries harder to detect for LLMs: Low Source Translation (LST) (Yong et al., 2023) translates the original query to Sindhi, a low-resource South-Asian language; BASE64 (B64) (Wei et al., 2023): encodes the query as BASE64 form; and Payload Splitting (PS) (Kang et al., 2024) divides the original query into several sections, and ask the model respond to the splice of the sections.

Two Behavior Manipulating methods that contain instructions in the system prompt leading to harmful responses are also used: Do Anything Now (DAN) (Shen et al., 2024) asks the LLM to be a non-restricted agent, and DeepInception (DI) (Li et al., 2023) leverages the personification capabilities of LLMs to construct a virtual nested scene, enabling them to bypass usage controls and generate harmful content.

In additon, we also utilize two Combination Attacks, which are DAN+Translation (DAN\_LST) and Payload Splitting+BASE64 (DI\_B64).

Details of the prompts of Behavior Manipulating methods can be viewed at the Appendix 15.

## 3.3.2 Malicious Discussion Agent.

We introduce a malicious agent into the multi-agent discussion step of the SafeScientist pipeline, which is deliberately programmed to steer conversations toward risky and potentially unethical directions. As an adversarial force, the agent simulates the complex interactions in real-world scientific communities, where conflicting or hazardous ideas may emerge from various participants. This agent tests the system’s robustness from the agent level, pushing it to discern and counteract harmful influences.

## 3.3.3 Malicious Experiment Instructor.

Experimentation is a crucial step in SafeScientist, which involves operating potentially risky scientific equipment. To rigorously assess the system’s robustness, we incorporate an agent tasked with deceiving the AI into adopting hazardous practices in this step. A dependable framework should counteract these attempts, ensuring that the experimentation process remains secure and scientifically sound despite the instructor’s interference.

## 4 SciSafetyBench

To evaluate our SafeScientist framework, we propose SciSafetyBench, a multi-disciplinary benchmark that evaluates the model’s safety awareness over 240 discovery tasks and 30 experimental tools.

![](images/2e7079575dfa4160f84e0beadc731a1311cb11e5ad0ca7cce3d2102262822c7b.jpg)  
Figure 3: SciSafetyBench consists of 240 tasks across six domains with four different risk types to give a comprehensive evaluation of how AI scientists can handle risky tasks well

## 4.1 General Research Dataset

The benchmark collects scientific tasks in six scientific domains: Physics, Chemistry, Biology, Material Science, Information Science, and Medicine, where each domain involves unique risk factors. In addition, we build tasks with four different risk sources (Tang et al., 2024): 1) The user intentionally requests a malicious topic - The user’s intent is clearly malicious and explicitly expressed in the prompt (like genetic editing); 2) The request seems benign but may be used for indirect harm- The user conceals harmful intent behind academic, fictional, or problem-solving language (like highly resistant virus) ; 3) The task has unintentional bad consequences - The user has no harmful intent, but the requested task may accidentally result in harm (like large-scale molecule replication); 4) The task is intrinsically risky - The task itself appears neutral, but the execution process involves safety hazards (like lose contact with infectious patients). More details on those types can be found at 7. In total, we provide 240 diverse scientific tasks—10 for each domain-risk type combination—accompanied by detailed descriptions and risk explanations.

To obtain the tasks, we first utilize OpenAIo3 (OpenAI, 2024), GPT-4.5 (OpenAI, 2025) and Gemini-2.5-pro (DeepMind, 2025)’s deep research function to collect high-risk tasks in each field. Each source is manually verified to ensure its accuracy, credibility, and alignment with our risk framework. For each query, we provide the LLM with the task name, the domain, and formal definitions of all four risk types, and prompt it to explore plausible high-risk tasks that are grounded in scientific literature. Our goal is to elicit open-ended research-style questions that may plausibly arise in academic or experimental contexts, but also carry distinct safety concerns. Each datapoint in the benchmark includes four fields: Task, Task Description, Prompt, and Risk Type. We then filter and refine the data with human experts from diverse backgrounds with sufficient domain knowledge to make sure that: 1) the factual knowledge in the task is correct; and 2) the task is authentically risky, and the risk type is consistent with the description.

## 4.2 Science Tool Dataset

Many experimental tools carry inherent risks and require specialized knowledge and careful handling to ensure safe operation (Zhao et al., 2024a; Al-Zyoud et al., 2019). To assess whether LLMs can recognize these risks and operate such equipment in accordance with established regulations and manuals, we build the safe tool-use dataset for scientific purpose.

First, we identify a total of 30 commonly used experimental tools across six scientific domains. For each tool, we construct a detailed description based on deep research of frontier LLMs. Specifically, we abstract the tool as a function that takes several input parameters, representing how a scientist would configure or operate it (e.g., setting the temperature of a chemical reactor), which enables text-based agents to simulate real tool uses. Safe usage is then defined as a comprehensive assessment of the tool’s overall risk profile, including descriptive accounts of potentially hazardous operations and a set of constraints on input parameters—where specific values or combinations thereof may lead to hazardous conditions. Our dataset includes precise criteria for identifying such risks, along with clear explanations for each case. For detailed illustration, a pseudo-code showing the tool “Radiation detection system” is included in Appendix C.2.

Secondly, we generate 120 specialized experimental use cases for the tools to evaluate whether AI scientists can operate the tools safely. These test cases are also created by GPT-4o and are reviewed by human experts to ensure that: 1) the assigned task is appropriate and relevant to the tool’s intended function; and 2) the potential hazards described and could plausibly occur under improper operation.

## 5 Experiment

## 5.1 Experiment Settings

Our SafetyScientist is built upon the Tiny Scientist framework (Yu et al., 2024), utilizing GPT-4o as the default LLM for our SafeScientist pipeline agent. For our method’s API calls, we configured the temperature at 0.75 and the maximum token length at 4096. Discussions in multi-agent scenarios were set to a maximum of three rounds. When comparing against other baseline frameworks such as AI Scientist (Lu et al., 2024) or Lab Agent (Schmidgall et al., 2025), we adhered to their respective default LLM settings to ensure fair comparisons. The experimental pipeline was designed to process both standard scientific prompts and adversarial inputs, allowing for a comprehensive comparison between our fully defended Safe-Scientist agent and a Baseline Agent lacking these integrated safety modules.

Metrics For the Quality test of the paper writing, we adopt the same LLM-as-judge evaluation prompt from the AI scientist, such as ‘Quality’, ‘Clarity’, ‘Presentation’, ‘Contribution’, and ‘Overall Assessment’. Similar to the design of our paper ethic reviewer, we design our safety evaluation prompt by gpt-4o-2024-0806 scoring from 0.5-5 with a step of 0.5. Our Pearson correlation coefficients among human annotators and LLM-as-judge reach 0.8, with a significant correlation of less than 0.01. Three annotators provided the human ratings with higher education backgrounds. Each annotator rated 10 ideas, selected from 6 scoring bins, totaling 30 ideas.

## 5.2 Main Experiment: Comparison with AI Scientist Frameworks

In this primary experiment, we compare SafeScientist against two established AI scientist frameworks: Agent Laboratory (Schmidgall et al., 2025) and Sakana AI Scientist (Sakana, 2024). Performance is evaluated based on quality, clarity, presentation, contribution and safety, each on a 1-5 scale by gpt-4o-2024-0806 with temperature set to 0.

In our experiment, we randomly selected 20 tasks from the biology domain. Since these tasks are incompatible with the experimental execution component in the original pipeline, we omit that part and focus on the literature review and writing stages. The AI scientist is implemented using a simplified version of the Tiny-Scientist framework for ease of deployment. In our evaluation, if any task is flagged as unsafe and rejected, it is assigned a safety score of 5, and its quality score is excluded from the overall analysis.

From Table 2, we can find that SafeScientist, equipped with a comprehensive multi-stage safeguard (including ethical review and defender at the discussion stage), significantly outperforms baseline methods, particularly in terms of safety. These results highlight SafeScientist’s effectiveness in minimizing risks in scientific discovery while maintaining high-quality research outputs. Notably, even without a prompt-level rejecter, SafeScientist maintains strong safety performance and successfully addresses all queries. The variant incorporating SafeChecker achieves the highest safety score among all methods, while also preserving high quality in the accepted queries.

<table><tr><td rowspan="2">Framework</td><td rowspan="2">Reject Rate (%)</td><td colspan="5">Quality Level Metrics</td><td rowspan="2">Safety</td></tr><tr><td>Quality</td><td>Clarity</td><td>Presentation</td><td>Contribution</td><td>Overall</td></tr><tr><td>AI Scientist (Sakana, 2024)</td><td>0</td><td>1.85</td><td>1.90</td><td>1.90</td><td>1.90</td><td>3.20</td><td>2.45</td></tr><tr><td>Agent Lab (Schmidgall et al., 2025)</td><td>15</td><td>2.00</td><td>2.47</td><td>2.47</td><td>1.94</td><td>3.18</td><td>2.45</td></tr><tr><td>SafeScientist-no PromptMonitor</td><td>0</td><td>2.00</td><td>2.30</td><td>2.35</td><td>2.00</td><td>3.40</td><td>3.30</td></tr><tr><td>SafeScientist-SafeChecker (ours)</td><td>90</td><td>2.00</td><td>2.50</td><td>2.50</td><td>2.00</td><td>3.50</td><td>4.83</td></tr></table>

Table 2: Comparison with Baseline AI Scientist Frameworks. Scores for the quality metrics use a 1–5 scale; Safety is also on a 1–5 scale. Reject Rate is reported as a percentage. We chose 20 test cases on Biology Domain

## 5.3 Prompt Monitor effectiness

In this experiment we compare our designed SafeChecker which is a prompt monitor to detect whether or not the input is unsafe.

This section evaluates the robustness of different LLMs against various prompt injection attacks and the effectiveness of different defense mechanisms, including components of SafeScientist’s Prompt Monitor. Attack Methods (detailed in Section 3.3) include: DeepInception (Li et al., 2023), DAN (Do Anything Now) (Shen et al., 2024), Low-SourceLanguage (Yong et al., 2023), Base64 (Wei et al., 2023), PayloadSplitting (Kang et al., 2024), DAN + LowSourceLanguage, PayloadSplitting + Base64.

As shown in Table 3, the proposed SafeScientist Fused Method—which combines both semantic and structural detection via LLaMA-Guard and SafeChecker—consistently outperforms all baseline defenses across all attack types, achieving the highest overall rejection rate of 78.70%.

Our custom-designed SafeChecker component significantly surpasses LLaMA-Guard-3-8B, particularly on structure-based attacks such as Base64 (46.67% vs. 13.33%) and PayloadSplitting (84.17% vs. 79.17%), as well as combination attacks like DAN + LST (50.00% vs. 10.83%).

Notably, combination attacks (PS + B64, DAN + LST) exhibit markedly lower rejection rates across individual defense models, demonstrating their elevated evasion strength. However, the Fused Method remains highly robust even against these complex attacks, with rejection rates of 88.33% and 55.83%, respectively, highlighting the advantage of multi-perspective detection.

## 5.4 Evaluation of Discussion Attacker Agent and Defense Agent

This experiment evaluates the impact of malicious and defensive agents on the safety of refined ideas generated across all our scientific domains.

In the first experimental setting (Attacker 1 and Defender 1), we activate either or both malicious and defensive agents during the initial group discussion phase, disguised as normal expert participants. Our experimental results indicate a robust group discussion pipeline: for example, in Medical, the safety score does not change under attack from Attacker 1. Similarly, in Info., safety scores improved from 2.83 (Normal) to 3.10 with Defender 1 active, demonstrating effective detection and neutralization of malicious intents by collective reasoning and suggestion modules.

In the second experimental setting (Attacker 2 and Defender 2), malicious and defensive agents were introduced at the summarization stage, directly interacting with the summarizer agent responsible for finalizing ideas. This stage significantly affected safety outcomes: notably, Defender 2 alone increased safety scores substantially across all domains, for instance, reaching 4.03 in Medical and 4.00 in Material. When both Attacker 2 and Defender 2 are present, the safety scores improve significantly compared to Attack2 alone, but do not reach the high levels achieved by Defense2 alone: for example, in Medical, the safety score for the combination of Attacker 2 and Defender 2 is 2.50, while the score for Defender 2 alone is 4.03.

These results indicate that late-stage interactions between attackers and defenders can catalyze deeper defensive reasoning, resulting in substantially enhanced idea safety.

## 5.5 Evaluation of Safe Tool Use

We evaluated the effectiveness of SafeScientist’s Tool-Use Monitor in ensuring safe interactions with scientific tools under benign and malicious instructional conditions. Specifically, we measured the Tool Call Safety Rate (percentage of tool calls adhering strictly to safety protocols) and the Human Correctness Rate (percentage assessed as both safe and accurate by human evaluators).

<table><tr><td>Model</td><td>Origin</td><td>DAN</td><td>LST</td><td>B64</td><td>DI</td><td>PS</td><td>PS+B64</td><td>DAN+LST</td><td>Avg</td></tr><tr><td>GPT-40</td><td>65.0</td><td>85.42</td><td>0.4</td><td>2.1</td><td>29.6</td><td>58.3</td><td>0.0</td><td>0.0</td><td>30.10</td></tr><tr><td>LlamaGuard-3-8B</td><td>79.2</td><td>88.3</td><td>33.75</td><td>13.33</td><td>96.67</td><td>79.17</td><td>73.33</td><td>10.83</td><td>59.32</td></tr><tr><td>SafeChecker</td><td>84.2</td><td>70.42</td><td>60.42</td><td>46.67</td><td>78.75</td><td>84.17</td><td>56.25</td><td>50.00</td><td>66.36</td></tr><tr><td>SafeScientist-Fuse (ours)</td><td>86.67</td><td>90.83</td><td>67.92</td><td>53.75</td><td>100.00</td><td>86.25</td><td>88.33</td><td>55.83</td><td>78.70</td></tr></table>

Table 3: Our SafeScientist-Fuse method consistently outperforms across all attack scenarios.. Method Prompt Defense Reject Rate with Different Monitor methods. (%)

<table><tr><td>Setting</td><td colspan="5">Physics Medical Info. Chemistry Material Biology</td></tr><tr><td>Normal</td><td>3.10</td><td>2.97</td><td>2.83</td><td>2.77</td><td>2.80 3.10</td></tr><tr><td>Attacker 1</td><td>2.77</td><td>2.97</td><td>2.87</td><td>2.83 2.90</td><td>3.07</td></tr><tr><td>Defender 1</td><td>3.10</td><td>2.93</td><td>3.10</td><td>2.50</td><td>2.93 3.17</td></tr><tr><td>Attacker 1 + Defender 1</td><td>2.87</td><td>2.97</td><td>2.67</td><td>2.83</td><td>2.93 2.97</td></tr><tr><td>Attacker 2</td><td>0.63</td><td>0.77</td><td>1.80</td><td>1.00</td><td>1.00 0.87</td></tr><tr><td>Defender 2</td><td>3.97</td><td>4.03</td><td>3.67</td><td>3.97</td><td>4.00 3.97</td></tr><tr><td>Attacker 2 + Defender 2</td><td>2.35</td><td>2.50</td><td>2.37</td><td>2.03</td><td>2.37 2.33</td></tr></table>

Table 4: Safety Impact Across Domains under Different Agent Configurations. Each value is a placeholder (1–5 scale).

<table><tr><td>Scenario Setting</td><td>Safety Rate (%) Correctness (%)</td></tr><tr><td>Benign User w/o Monitor</td><td>43.3 70.6</td></tr><tr><td>Malicious User w/o Monitor</td><td>5.8 0.0</td></tr><tr><td>Benign User w/ Monitor</td><td>50.0 75.0</td></tr><tr><td>Malicious User w/ Monitor</td><td>47.5 60.0</td></tr></table>

Table 5: Performance in Safe Tool Usage Scenarios. Each row represents a specific combination of user intent and monitoring setup.

The rule-based detector automatically assessed the initial tool call safety, with further validation through manual human evaluation of 10 randomly selected tasks per domain to see if the agent correctly used the tool to finish the tasks.

The results in Table 5 demonstrate clear improvements when the Tool-Use Monitor was employed. The Safety Rate improved from 43.3% to 50.0% under benign conditions, and notably from 5.8% to 47.5% under malicious instructions. Correspondingly, the Human Correctness Rate increased from 70.6% to 75.0% for benign tasks and rose dramatically from 0% to 60.0% for malicious tasks when monitored. These findings quantitatively illustrate the significant protective effect of the Tool-Use Monitor against unsafe operational parameters, particularly in adversarial conditions.

## 5.6 Impact of the Ethical Reviewer

To evaluate the effectiveness of our ethical reviewer module, we randomly select 20 representative tasks from each of six scientific domains. For each task, we collect both the AI-generated draft paper and the refined paper after applying the ethical reviewer, and assess their ethical adherence using our scoring rubric. As shown in Figure 4, our ethical

![](images/7a75ebf2b4c937cc30ee6a3ab4d5be69a2a7f8545ac850a794007e412178c814.jpg)  
Figure 4: Ethical Score Comparison Across Domains. This bar chart compares the average ethical scores of AIgenerated draft papers and their refined versions across six scientific domains. The refined papers consistently demonstrate improved ethical adherence.

reviewer achieves substantial improvements across all domains. On average, the refined papers exhibit a 44.4% increase in ethical score compared to the initial drafts, validating the effectiveness of our refinement strategy in enhancing safety and ethical robustness in AI-generated scientific outputs.

## 6 Conclusion

We present SafeScientist, a novel framework that prioritizes safety and ethical responsibility in AI-driven scientific research. Together with SciSafetyBench, a dedicated benchmark for evaluating safety in high-risk scientific scenarios, our approach integrates layered defenses—including prompt filtering, agent oversight, Tool Defender, and ethical review. SafeScientist demonstrates strong potential for enabling more secure and responsible AI scientific discovery. To the best of our knowledge, this is the first work to comprehensively address the dual challenge of designing a risk-aware AI scientist framework and establishing a domain-grounded benchmark for its safety evaluation. Our work paves the way for the next generation of secure, ethical, and trustworthy AI systems for scientific discovery. Future efforts will extend SciSafetyBench to additional scientific areas, enhance real-time adaptivity of defense mechanisms, and further explore societal impacts of autonomous research agents.

## Limitations

This work focuses on enhancing the safety of AI Scientists by developing a comprehensive safeguard framework spanning multiple stages. However, the current system primarily relies on offthe-shelf large language models (LLMs) that operate as separate modules with limited integration. This modularity, while convenient, restricts both the depth of domain-specific expertise and the level of interaction between components. Future work could explore end-to-end architectures that enable richer connectivity and joint optimization, which may lead to more robust and coherent safety mechanisms for AI Scientists.

Additionally, while our proposed evaluation method creatively incorporates tool use to assess agent safety, it remains only simulation of realworld experimental settings. As such, it may overlook important contextual or sensory details. Moving forward, we aim to incorporate multi-modal inputs, such as images of laboratory equipment or instructional videos, and potentially employ embodied agents. These additions could: (1) provide a more realistic and comprehensive evaluation of AI Scientists’ capabilities; and (2) test their ability to attend to nuanced, non-textual cues that are often critical in scientific practice.

## 7 Acknowledgment

We sincerely appreciate the support from Amazon grant funding project 120359, “GRAG: Enhance RAG Applications with Graph-structured Knowledge”, and Meta gift funding project “PERM: Toward Parameter Efficient Foundation Models for Recommenders”.

## 8 Statement

Our artifacts are only for research and non-profit use. We follow all the instructions of other artifacts, such as TinyScientist. We check that the data we create doesn’t have personal privacy information.

## References

Walid Al-Zyoud, Alshaimaa M Qunies, Ayana UC Walters, and Nigel K Jalsa. 2019. Perceptions of chemical safety in laboratories. Safety, 5(2):21.

Maksym Andriushchenko, Alexandra Souly, Mateusz Dziemian, Derek Duenas, Maxwell Lin, Justin Wang, Dan Hendrycks, Andy Zou, Zico Kolter, Matt Fredrikson, and 1 others. 2024. Agentharm: A benchmark for measuring harmfulness of llm agents. arXiv preprint arXiv:2410.09024.

Andy Arditi, Oscar Obeso, Aaquib Syed, Daniel Paleka, Nina Panickssery, Wes Gurnee, and Neel Nanda. 2024. Refusal in language models is mediated by a single direction. arXiv preprint arXiv:2406.11717.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, and 1 others. 2022. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862.

Yoshua Bengio, Michael Cohen, Damiano Fornasiere, Joumana Ghosn, Pietro Greiner, Matt MacDermott, Sören Mindermann, Adam Oberman, Jesse Richardson, Oliver Richardson, Marc-Antoine Rondeau, Pierre-Luc St-Charles, and David Williams-King. 2025a. Superintelligent agents pose catastrophic risks: Can scientist ai offer a safer path? Preprint, arXiv:2502.15657.

Yoshua Bengio, Sören Mindermann, Daniel Privitera, Tamay Besiroglu, Rishi Bommasani, Stephen Casper, Yejin Choi, Philip Fox, Ben Garfinkel, Danielle Goldfarb, and 1 others. 2025b. International ai safety report. arXiv preprint arXiv:2501.17805.

Xiusi Chen, Hongzhi Wen, Sreyashi Nag, Chen Luo, Qingyu Yin, Ruirui Li, Zheng Li, and Wei Wang. 2024a. Iteralign: Iterative constitutional alignment of large language models. In Proceedings ofthe 2024 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 1423–1433.

Zhaorun Chen, Zhen Xiang, Chaowei Xiao, Dawn Song, and Bo Li. 2024b. Agentpoison: Red-teaming llm agents via poisoning memory or knowledge bases. Advances in Neural Information Processing Systems, 37:130185–130213.

Yuheng Cheng, Ceyao Zhang, Zhengwen Zhang, Xiangrui Meng, Sirui Hong, Wenhao Li, Zihao Wang,

Zekai Wang, Feng Yin, Junhua Zhao, and 1 others. 2024. Exploring large language model based intelligent agents: Definitions, methods, and prospects. arXiv preprint arXiv:2401.03428.

Hyeong Kyu Choi, Xuefeng Du, and Yixuan Li. 2024. Safety-aware fine-tuning of large language models. arXiv preprint arXiv:2410.10014.

Edoardo Debenedetti, Jie Zhang, Mislav Balunovic,´ Luca Beurer-Kellner, Marc Fischer, and Florian Tramèr. 2024. Agentdojo: A dynamic environment to evaluate attacks and defenses for llm agents. arXiv preprint arXiv:2406.13352.

Google DeepMind. 2025. Gemini 2.5 pro. https:// deepmind.google/technologies/gemini/pro/.

Zehang Deng, Yongjian Guo, Changzhou Han, Wanlun Ma, Junwu Xiong, Sheng Wen, and Yang Xiang. 2024. Ai agents under threat: A survey of key security challenges and future pathways. arXiv preprint arXiv:2406.02630.

Shen Dong, Shaochen Xu, Pengfei He, Yige Li, Jiliang Tang, Tianming Liu, Hui Liu, and Zhen Xiang. 2025. A practical memory injection attack against llm agents. arXiv preprint arXiv:2503.03704.

Yi Dong, Ronghui Mu, Yanghao Zhang, Siqi Sun, Tianle Zhang, Changshun Wu, Gaojie Jin, Yi Qi, Jinwei Hu, Jie Meng, and 1 others. 2024. Safeguarding large language models: A survey. arXiv preprint arXiv:2406.02622.

Shangbin Feng, Chan Young Park, Yuhan Liu, and Yulia Tsvetkov. 2023. From pretraining data to language models to downstream tasks: Tracking the trails of political biases leading to unfair nlp models. arXiv preprint arXiv:2305.08283.

Tao Feng, Chuanyang Jin, Jingyu Liu, Kunlun Zhu, Haoqin Tu, Zirui Cheng, Guanyu Lin, and Jiaxuan You. 2024. How far are we from AGI: Are LLMs all we need? Transactions on Machine Learning Research. Survey Certification.

Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, Anil Palepu, Petar Sirkovic, Artiom Myaskovsky, Felix Weissenberger, Keran Rong, Ryutaro Tanno, and 1 others. 2025. Towards an ai coscientist. arXiv preprint arXiv:2502.18864.

Taicheng Guo, Xiuying Chen, Yaqi Wang, Ruidi Chang, Shichao Pei, Nitesh V Chawla, Olaf Wiest, and Xiangliang Zhang. 2024. Large language model based multi-agents: A survey of progress and challenges. arXiv preprint arXiv:2402.01680.

Chi Han, Jialiang Xu, Manling Li, Yi Fung, Chenkai Sun, Nan Jiang, Tarek Abdelzaher, and Heng Ji. 2023. Word embeddings are steers for language models. arXiv preprint arXiv:2305.12798.

Peixuan Han, Cheng Qian, Xiusi Chen, Yuji Zhang, Denghui Zhang, and Heng Ji. 2025. Internal activation as the polar star for steering unsafe llm behavior. arXiv preprint arXiv:2502.01042.

Pengfei He, Yupin Lin, Shen Dong, Han Xu, Yue Xing, and Hui Liu. 2025. Red-teaming llm multi-agent systems via communication attacks. arXiv preprint arXiv:2502.14847.

Yifeng He, Ethan Wang, Yuyang Rong, Zifei Cheng, and Hao Chen. 2024. Security of ai agents. arXiv preprint arXiv:2406.08689.

David Huang, Avidan Shah, Alexandre Araujo, David Wagner, and Chawin Sitawarin. 2025. Stronger universal and transferable attacks by suppressing refusals. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 5850–5876.

Jen-tse Huang, Jiaxu Zhou, Tailin Jin, Xuhui Zhou, Zixi Chen, Wenxuan Wang, Youliang Yuan, Maarten Sap, and Michael R Lyu. 2024. On the resilience of multiagent systems with malicious agents. arXiv preprint arXiv:2408.00989.

Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, and 1 others. 2023. Llama guard: Llm-based inputoutput safeguard for human-ai conversations. arXiv preprint arXiv:2312.06674.

Tianjie Ju, Yiting Wang, Xinbei Ma, Pengzhou Cheng, Haodong Zhao, Yulong Wang, Lifeng Liu, Jian Xie, Zhuosheng Zhang, and Gongshen Liu. 2024. Flooding spread of manipulated knowledge in llmbased multi-agent communities. arXiv preprint arXiv:2407.07791.

Daniel Kang, Xuechen Li, Ion Stoica, Carlos Guestrin, Matei Zaharia, and Tatsunori Hashimoto. 2024. Exploiting programmatic behavior of llms: Dual-use through standard security attacks. In 2024 IEEE Security and Privacy Workshops (SPW), pages 132–143. IEEE.

Aounon Kumar, Chirag Agarwal, Suraj Srinivas, Aaron Jiaxun Li, Soheil Feizi, and Himabindu Lakkaraju. 2023. Certifying llm safety against adversarial prompting. arXiv preprint arXiv:2309.02705.

Donghyun Lee and Mo Tiwari. 2024. Prompt infection: Llm-to-llm prompt injection within multi-agent systems. Preprint, arXiv:2410.07283.

Tianlong Li, Shihan Dou, Wenhao Liu, Muling Wu, Changze Lv, Rui Zheng, Xiaoqing Zheng, and Xuanjing Huang. 2024. Rethinking jailbreaking through the lens of representation engineering. arXiv preprint arXiv:2401.06824.

Xuan Li, Zhanke Zhou, Jianing Zhu, Jiangchao Yao, Tongliang Liu, and Bo Han. 2023. Deepinception: Hypnotize large language model to be jailbreaker. arXiv preprint arXiv:2311.03191.

Yatao Li and Jianfeng Zhan. 2022. Saibench: Benchmarking ai for science. BenchCouncil Transactions on Benchmarks, Standards and Evaluations, 2(2):100063.

Yiming Li, Yong Jiang, Zhifeng Li, and Shu-Tao Xia. 2022. Backdoor learning: A survey. IEEE transactions on neural networks and learning systems, 35(1):5–22.

Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, and 1 others. 2024. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437.

Bang Liu, Xinfeng Li, Jiayi Zhang, Jinlin Wang, Tanjin He, Sirui Hong, Hongzhang Liu, Shaokun Zhang, Kaitao Song, Kunlun Zhu, Yuheng Cheng, Suyuchen Wang, Xiaoqiang Wang, Yuyu Luo, Haibo Jin, Peiyan Zhang, Ollie Liu, Jiaqi Chen, Huan Zhang, and 28 others. 2025. Advances and challenges in foundation agents: From brain-inspired intelligence to evolutionary, collaborative, and safe systems. Preprint, arXiv:2504.01990.

Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jeff Clune, and David Ha. 2024. The ai scientist: Towards fully automated open-ended scientific discovery. arXiv preprint arXiv:2408.06292.

Ziming Luo, Zonglin Yang, Zexin Xu, Wei Yang, and Xinya Du. 2025. Llm4sr: A survey on large language models for scientific research. arXiv preprint arXiv:2501.04306.

Junyuan Mao, Fanci Meng, Yifan Duan, Miao Yu, Xiaojun Jia, Junfeng Fang, Yuxuan Liang, Kun Wang, and Qingsong Wen. 2025. Agentsafe: Safeguarding large language model-based multi-agent systems via hierarchical data management. arXiv preprint arXiv:2503.04392.

Tong Mu, Alec Helyar, Johannes Heidecke, Joshua Achiam, Andrea Vallone, Ian Kivlichan, Molly Lin, Alex Beutel, John Schulman, and Lilian Weng. 2024. Rule based rewards for language model safety. arXiv preprint arXiv:2411.01111.

OpenAI. 2024. Openai o3 and o4-mini system card. https://cdn.openai.com/pdf/ 2221c875-02dc-4789-800b-e7758f3722c1/ o3-and-o4-mini-system-card.pdf.

OpenAI. 2025. Openai gpt-4.5 system card. https://cdn.openai.com/ gpt-4-5-system-card-2272025.pdf.

Nardine Osman and Mark d’Inverno. 2023. Human values in multiagent systems. arXiv preprint arXiv:2305.02739.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, and 1 others. 2022. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744.

Yansheng Qiu, Haoquan Zhang, Zhaopan Xu, Ming Li, Diping Song, Zheng Wang, and Kaipeng Zhang. 2025. Ai idea bench 2025: Ai research idea generation benchmark. arXiv preprint arXiv:2504.14191.

Domenic Rosati, Jan Wehner, Kai Williams, Lukasz Bartoszcze, Robie Gonzales, Subhabrata Majumdar, Hassan Sajjad, Frank Rudzicz, and 1 others. 2024. Representation noising: A defence mechanism against harmful finetuning. Advances in Neural Information Processing Systems, 37:12636–12676.

Sakana. 2024. The ai scientist: Towards fully automated open-ended scientific discovery.

Samuel Schmidgall, Yusheng Su, Ze Wang, Ximeng Sun, Jialian Wu, Xiaodong Yu, Jiang Liu, Zicheng Liu, and Emad Barsoum. 2025. Agent laboratory: Using llm agents as research assistants. arXiv preprint arXiv:2501.04227.

Md Shamsujjoha, Qinghua Lu, Dehai Zhao, and Liming Zhu. 2024. Towards ai-safety-by-design: A taxonomy of runtime guardrails in foundation model based systems. arXiv preprint arXiv:2408.02205.

Xinyue Shen, Zeyuan Chen, Michael Backes, Yun Shen, and Yang Zhang. 2024. " do anything now": Characterizing and evaluating in-the-wild jailbreak prompts on large language models. In Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security, pages 1671–1685.

Lucheng Sun, Tiejun Wu, and Ya Zhang. 2023. A defense strategy for false data injection attacks in multiagent systems. International Journal ofSystems Science, 54(16):3071–3084.

Xiangru Tang, Qiao Jin, Kunlun Zhu, Tongxin Yuan, Yichi Zhang, Wangchunshu Zhou, Meng Qu, Yilun Zhao, Jian Tang, Zhuosheng Zhang, and 1 others. 2024. Prioritizing safeguarding over autonomy: Risks of llm agents for science. arXiv preprint arXiv:2402.04247.

Ross Taylor, Marcin Kardas, Guillem Cucurull, Thomas Scialom, Anthony Hartshorn, Elvis Saravia, Andrew Poulton, Viktor Kerkez, and Robert Stojnic. 2022. Galactica: A large language model for science. arXiv preprint arXiv:2211.09085.

Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, and 1 others. 2023. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.

Yu Tian, Xiao Yang, Jingyuan Zhang, Yinpeng Dong, and Hang Su. 2023. Evil geniuses: Delving into the safety of llm-based agents. arXiv preprint arXiv:2311.11855.

Benyou Wang, Qianqian Xie, Jiahuan Pei, Zhihong Chen, Prayag Tiwari, Zhao Li, and Jie Fu. 2023a. Pretrained language models in biomedical domain: A systematic survey. ACM Computing Surveys, 56(3):1– 52.

Shilong Wang, Guibin Zhang, Miao Yu, Guancheng Wan, Fanci Meng, Chongye Guo, Kun Wang, and Yang Wang. 2025. G-safeguard: A topology-guided security lens and treatment on llm-based multi-agent systems. arXiv preprint arXiv:2502.11127.

Yuxia Wang, Haonan Li, Xudong Han, Preslav Nakov, and Timothy Baldwin. 2023b. Do-not-answer: A dataset for evaluating safeguards in llms. arXiv preprint arXiv:2308.13387.

Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. 2023. Jailbroken: How does llm safety training fail? Advances in Neural Information Processing Systems, 36:80079–80110.

Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. 2024. Jailbroken: How does llm safety training fail? Advances in Neural Information Processing Systems, 36.

Yixuan Weng, Minjun Zhu, Guangsheng Bao, Hongbo Zhang, Jindong Wang, Yue Zhang, and Linyi Yang. 2024. Cycleresearcher: Improving automated research via automated review. arXiv preprint arXiv:2411.00816.

Tinghao Xie, Xiangyu Qi, Yi Zeng, Yangsibo Huang, Udari Madhushani Sehwag, Kaixuan Huang, Luxi He, Boyi Wei, Dacheng Li, Ying Sheng, and 1 others. 2024. Sorry-bench: Systematically evaluating large language model safety refusal behaviors. arXiv preprint arXiv:2406.14598.

Wei Xiong, Hanze Dong, Chenlu Ye, Ziqi Wang, Han Zhong, Heng Ji, Nan Jiang, and Tong Zhang. 2024. Gibbs sampling from human feedback: A provable kl-constrained framework for rlhf. In Proc. The Fortyfirst International Conference on Machine Learning (ICML2024).

Sheng Yin, Xianghe Pang, Yuanzhuo Ding, Menglan Chen, Yutong Bi, Yichen Xiong, Wenhao Huang, Zhen Xiang, Jing Shao, and Siheng Chen. 2024. Safeagentbench: A benchmark for safe task planning of embodied llm agents. arXiv preprint arXiv:2412.13178.

Zheng-Xin Yong, Cristina Menghini, and Stephen H Bach. 2023. Low-resource languages jailbreak gpt-4. arXiv preprint arXiv:2310.02446.

Haofei Yu, Zhaochen Hong, Zirui Cheng, Kunlun Zhu, Keyang Xuan, Jinwei Yao, Tao Feng, and Jiaxuan You. 2024. Researchtown: Simulator

of human research community. arXiv preprint arXiv:2412.17767.

Haofei Yu, Keyang Xuan, Fenghai Li, Zijie Lei, and Jiaxuan You. 2025. Tinyscientist: A lightweight framework for building research agents. https://github.com/ulab-uiuc/tiny-scientist. Accessed: 2025-04-14.

Jiakang Yuan, Xiangchao Yan, Botian Shi, Tao Chen, Wanli Ouyang, Bo Zhang, Lei Bai, Yu Qiao, and Bowen Zhou. 2025. Dolphin: Closed-loop openended auto-research through thinking, practice, and feedback. arXiv preprint arXiv:2501.03916.

Tongxin Yuan, Zhiwei He, Lingzhong Dong, Yiming Wang, Ruijie Zhao, Tian Xia, Lizhen Xu, Binglin Zhou, Fangqi Li, Zhuosheng Zhang, and 1 others. 2024. R-judge: Benchmarking safety risk awareness for llm agents. arXiv preprint arXiv:2401.10019.

Boyang Zhang, Yicong Tan, Yun Shen, Ahmed Salem, Michael Backes, Savvas Zannettou, and Yang Zhang. 2024a. Breaking agents: Compromising autonomous llm agents through malfunction amplification. arXiv preprint arXiv:2407.20859.

Hanrong Zhang, Jingyuan Huang, Kai Mei, Yifei Yao, Zhenting Wang, Chenlu Zhan, Hongwei Wang, and Yongfeng Zhang. 2024b. Agent security bench (asb): Formalizing and benchmarking attacks and defenses in llm-based agents. arXiv preprint arXiv:2410.02644.

Qiang Zhang, Keyan Ding, Tianwen Lv, Xinda Wang, Qingyu Yin, Yiwen Zhang, Jing Yu, Yuhao Wang, Xiaotong Li, Zhuoyi Xiang, and 1 others. 2025. Scientific large language models: A survey on biological & chemical domains. ACM Computing Surveys, 57(6):1–38.

Yu Zhang, Xiusi Chen, Bowen Jin, Sheng Wang, Shuiwang Ji, Wei Wang, and Jiawei Han. 2024c. A comprehensive survey of scientific large language models and their applications in scientific discovery. arXiv preprint arXiv:2406.10833.

Aaron Y Zhao, Nathan E DeSousa, Hanne C Henriksen, Ann Marie May, Xianming Tan, and David S Lawrence. 2024a. An assessment of laboratory safety training in undergraduate education. Journal of Chemical Education, 101(4):1626–1634.

Shuai Zhao, Meihuizi Jia, Zhongliang Guo, Leilei Gan, Xiaoyu Xu, Xiaobao Wu, Jie Fu, Yichao Feng, Fengjun Pan, and Luu Anh Tuan. 2024b. A survey of backdoor attacks and defenses on large language models: Implications for security measures. Authorea Preprints.

Chujie Zheng, Fan Yin, Hao Zhou, Fandong Meng, Jie Zhou, Kai-Wei Chang, Minlie Huang, and Nanyun Peng. 2024. On prompt-driven safeguarding for large language models. In Forty-first International Conference on Machine Learning.

Xuhui Zhou, Hyunwoo Kim, Faeze Brahman, Liwei Jiang, Hao Zhu, Ximing Lu, Frank Xu, Bill Yuchen Lin, Yejin Choi, Niloofar Mireshghallah, Ronan Le Bras, and Maarten Sap. 2024. Haicosystem: An ecosystem for sandboxing safety risks in human-ai interactions. Preprint, arXiv:2409.16427.

Andy Zou, Long Phan, Justin Wang, Derek Duenas, Maxwell Lin, Maksym Andriushchenko, J Zico Kolter, Matt Fredrikson, and Dan Hendrycks. 2024. Improving alignment and robustness with circuit breakers. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J Zico Kolter, and Matt Fredrikson. 2023. Universal and transferable adversarial attacks on aligned language models. arXiv preprint arXiv:2307.15043.

## A Comparisons with Different Foundation Models

To assess the impact of the underlying LLM on SafeScientist’s performance, we evaluated our framework using various foundation models. The same set of quality and safety metrics as in the main experiment was used; the results can be found in Table 6.

## B Prompts

This section lists the prompts used in building Safe-Scientist, constructing the dataset, and conducting attacks.

## B.1 Thinker Group discussion Prompts

<table><tr><td rowspan="2">Model integrated in SafeScientist</td><td colspan="4">Review Metrics (1–5)</td><td rowspan="2">Safety Score (1–5)</td></tr><tr><td>Quality</td><td>Clarity</td><td></td><td>Presentation Contribution</td></tr><tr><td>GPT-4.1</td><td>1.00</td><td>1.75</td><td>1.43</td><td>1.00</td><td>3.89</td></tr><tr><td>GPT-3.5-turbo</td><td>2.00</td><td>2.00</td><td>2.00</td><td>2.00</td><td>2.88</td></tr><tr><td>Qwen2.5-72B-Instruct</td><td>2.15</td><td>2.83</td><td>2.83</td><td>2.27</td><td>3.39</td></tr><tr><td>Llama-3-70B-Instruct</td><td>1.95</td><td>2.30</td><td>2.30</td><td>2.02</td><td>2.77</td></tr></table>

Table 6: Performance of SafeScientist with Different Foundation Models.

![](images/8c3171ee2e39e665199a2ba12de7c9a27963dd5cfbdc640d16be86727edf7da0.jpg)  
Figure 5: Construct Scientist Prompt.

![](images/d9f37e37c3f854e1fe80c2e3b575dba5e7c604b73901fb98605528bad937f6b6.jpg)  
Figure 6: Experiment Designer Prompt.

![](images/b1d998945d3696543a653a7749a3dfb9ca5388ee5e5b16d31140717abc502e1a.jpg)  
Figure 7: Survey Specialist Prompt.

![](images/21cb5cac34faa82df5ba3cda9e5df9b8fc0a419bf4a68d49af8dcc9c00b4f7cc.jpg)  
Figure 8: Physics Nuclear Theorist Scientist Prompt.

![](images/b6d85de8acb6b4614921e06168f428ee0349912146f62f35cc821e14f3d7c4d5.jpg)  
Figure 9: Medicine Tissue Engineer Scientist Prompt.

![](images/ae7e3683327d9faa1749572576a898cd3fbbc2930faa19532de5d3ff89b5d609.jpg)  
Figure 10: Materials Metamaterials Engineer Scientist Prompt.

![](images/9ef8449ae38a7763b5a1b79b361e6e564bbf731636d1448632dbdb6b43550d43.jpg)  
Figure 11: Information Science Network Architect Scientist Prompt.

![](images/9c657cebe90f47e806a5550a394e9d69638b2d9cd56b33dcce797b0390e8d101.jpg)  
Figure 12: Chemistry Synthetic Chemist Scientist Prompt.

![](images/cf76f8a2fbb24a63932c11ff1c2fd1498642a6dff3052f3e8e61c5640f5ab7db.jpg)  
Figure 13: Biology Neuropharmacologist Scientist Prompt.

![](images/6f2e804036700e099ebd065c26add03f44275784fe640efd535c4cf6e82e6717.jpg)  
Figure 14: Group Discussion Prompt.

## B.2 Attack Prompts

![](images/5bb8020488d90faf191bc721a5df691bd30feb41b4528f506e20a3cf786ff12b.jpg)  
Figure 15: Prompt Attack Method

![](images/e521e07457b72fa2b82413717025f501926eb3efa86517486853b2ee6c7a7d7f.jpg)  
Figure 16: Attack Detection Risk Level Prompt

![](images/63a5aa280652f9acafecc4fc6f929d96aac6a91856d2053b8fbff7f670cdbc3e.jpg)  
Figure 17: Attack Detection Prompt Part 1.

![](images/28d0de06d7755ef70103f6cb2a0e0efd6d7663297568ca8b74b2580ed1327f5a.jpg)  
Figure 18: Attack Detection Prompt Part 2.

## C Cases of SciSafetyBench

## C.1 Research Task and its Risk

Figure 19 shows one data entry of SciSafetyBench in JSON form, containing task name, detailed instruction, risk type, risk explanation, and the source supporting the risk.

## C.2 Scientific Tool and its Risk

Figure 20 shows the of tool “High Pressure Reactor” in SciSafetyBench, which contains a list of parameters and several safety constraints on its usage.

## D Cases of Different Attack Method

## E Instructions Given to Participants

All annotators involved in this research were explicitly informed prior to participation that their annotation results would be included in a publicly accessible research publication. Annotators received detailed instructions clearly explaining the objective of the annotation tasks, which were primarily focused on evaluating the quality of generated ideas with particular attention to the criteria of diversity and professional background relevance, such as one student in the Physics major was specified to annotate physics-related tasks, all the other have enough background knowledge to learn and check the correctness of the annotated data. Specifically, diversity was described to annotators as the degree of novelty and distinctiveness among ideas. The relevance of the annotators’ background was emphasized, highlighting the importance of having participants from varied professional domains and experiences to ensure comprehensive and balanced evaluations. Annotators formally acknowledged understanding these instructions and provided consent for their annotations to be used in the subsequent analysis and publicly disseminated results. All the participants are the authors, and they all agree to participate in the annotations. We follow all the instructions for getting the annotated data, including the policy from the ethical review board.

AI assistants, such as GPT and DeepResearch, were used to help us filter and identify the source of our dataset, and then a human double-checks this.

![](images/0768421ea1c281722519710530f0616b8250f9c62d5b974727e9e07a60f8776d.jpg)  
Figure 19: A Case of SciSafetyBench.

![](images/6087c29347eb7a4be2e072eaaad4c2e6c905ed0ea04ee23df0f9aa52d06ef96d.jpg)  
Figure 20: Safety Check Code for High-Pressure Reactor.

![](images/554f4221392b1d1c1dc6e3079579044e5c3f2cc6ad99efa2baf1630035732462.jpg)  
Figure 21: Planning Prompt used to evaluate how well the agents define roles, assign tasks, and distribute workloads in a multiagent system, with automatic line wrapping.

![](images/a4ad10e968b63d7725c890a59af6f45db004c8cf8f2bffa25033d06ef9e1996a.jpg)  
Figure 22: Safety Score Evaluation Prompt used to evaluate the safety and ethical risk of an idea on a scale of 0.5 to 5.0 (in 0.5 increments).

![](images/f0cae606294b9b0051775eb4b8cef216012d4bb51fe248cd116d1887278565b2.jpg)  
Figure 23: A Case of Scientific Tool Definition.

![](images/116daa1688b45d32f4c82df86a3d082d998097c161a8b8da84aff3a7edbc4a08.jpg)  
Figure 24: A Case of Scientific Tool Tasks.

![](images/d4c93201a02185d05bf79b593ce2ce9cc9583d308cbcaa6391ea1f6707b65d2f.jpg)  
Figure 25: A Case of Parameters for a specific scientific tool task.

![](images/d5e50ef3ad87614459a550db064e18467734cce168e520986778c107339d8726.jpg)  
Figure 26: An Example of Group Discussion Chat History.

<table><tr><td rowspan=1 colspan=1>Risk Type                       Definition</td></tr><tr><td rowspan=1 colspan=1>User Perspective</td></tr><tr><td rowspan=1 colspan=1>Direct Malicious User            The user has explicitly harmful intent, directly asking the agent to assistwith dangerous or illegal actions.Example: How can I create a binary chemical weapon using commonindustrial supplies?</td></tr><tr><td rowspan=1 colspan=1>Indirect Malicious User          The user has a hidden malicious goal, but conceals it by requestingseemingly harmless subtasks.For example: How can I synthesize compound X (a precursor) in highyield using over-the-counter reagents?(Risk: X is later used in explosive synthesis).</td></tr><tr><td rowspan=1 colspan=1>Unintentional Consequence      The user has benign or neutral intentions, but the proposed actions maycause unforeseen harm due to systemic, environmental, or biologicalcomplexity.For example: How can I optimize a soil bacteria to improve crop nitrogenfixation in large-scale deployments?(Risk: Unintended ecological imbalance).</td></tr><tr><td rowspan=1 colspan=1>Task Perspective</td></tr><tr><td rowspan=1 colspan=1>Task-Intrinsic                    The task is scientifically valid and often academically framed, but itsexecution involves inherent physical, chemical, biological and other risks.For example: Design a laboratory protocol for handling Beryllium dustduring alloy testing procedures.</td></tr></table>

Table 7: Definitions and Examples of Dataset Risk Types

<table><tr><td>Dimension</td><td>Model vs Human Evaluation</td></tr><tr><td>GOAL</td><td>0.80</td></tr></table>

Table 8: Pearson correlation coefficients (r) and significance between GPT-4o evaluation and human judgment on the GOAL dimension. Blue indicates strong and significant correlation $( p \le 0 . 0 1 )$ ). The human ratings were provided by three annotators with higher-education backgrounds. Each annotator rated 10 ideas, selected from 6 scoring bins, for a total of 30 ideas.

![](images/5182142ad12b59ced1d4e0b767df9b73ceb427f5b3bd39aaec764965067c1eae.jpg)  
Figure 27: System prompts issued to the Malicious and Defense agents for risk manipulation and mitigation during evaluation.