# CoBia: Constructed Conversations Can Trigger Otherwise Concealed Societal Biases in LLMs

Nafiseh Nikeghbal<sup>1</sup> Amir Hossein Kargaran<sup>2</sup> Jana Diesner<sup>1</sup>

<sup>1</sup>Technical University of Munich <sup>2</sup>LMU Munich & Munich Center for Machine Learning nafiseh.nikeghbal@tum.de

## Abstract

Warning: This paper contains content that may be offensive or upsetting.

Improvements in model construction, including fortified safety guardrails, allow Large language models (LLMs) to increasingly pass standard safety checks. However, LLMs sometimes slip into revealing harmful behavior, such as expressing racist viewpoints, during conversations. To analyze this systematically, we introduce CoBia, a suite of lightweight adversarial attacks that allow us to refine the scope of conditions under which LLMs depart from normative or ethical behavior in conversations. CoBia creates a constructed conversation where the model utters a biased claim about a social group. We then evaluate whether the model can recover from the fabricated bias claim and reject biased follow-up questions. We evaluate 11 open-source as well as proprietary LLMs for their outputs related to six socio-demographic categories that are relevant to individual safety and fair treatment, i.e., gender, race, religion, nationality, sex orientation, and others. Our evaluation is based on established LLM-based bias metrics, and we compare the results against human judgments to scope out the LLMs’ reliability and align ment. The results suggest that purposefully constructed conversations reliably reveal bias amplification and that LLMs often fail to reject biased follow-up questions during dialogue. This form of stress-testing highlights deeply embedded biases that can be surfaced through interaction. Code and artifacts are available at github.com/nafisenik/CoBia.

## 1 Introduction

Large language models (LLMs) have been widely adopted for a diverse range of tasks (OpenAI et al., 2024a; Grattafiori et al., 2024), serving users from highly skilled professionals to non-technical individuals (Bommasani et al., 2022). To ensure safety and reduce harmful outputs of LLMs, various alignment techniques and guardrails have been implemented (Dai et al., 2024; Biswas and Talukdar, 2023; Bai et al., 2022a,b; Ganguli et al., 2022; Markov et al., 2023).

However, despite these efforts, recent studies have shown that societal biases<sup>1</sup> remain deeply embedded in model behavior and can resurface through “jailbreak” attempts (Jin et al., 2024; Wei et al., 2023) . Jailbreaks are adversarial attacks that breach LLMs’ safety mechanisms, leading to harmful responses. It is critical to detect these loopholes so that they can be patched. These harmful responses reinforce stereotypes and marginalize (historically) vulnerable demographics (Sheng et al., 2021), challenging the ethical deployment of LLMs (Bender et al., 2021). These biases mainly stem from the explicit or implicit presence of toxic, stereotypical, and harmful content in pretraining data (Thaler et al., 2024; Jeoung et al., 2023; Guo et al., 2024b). Beyond the models themselves, biases may also be amplified during user interactions with LLMs, as LLMs can be user pleasers and users may selectively interpret outputs that confirm their existing beliefs (Gallegos et al., 2024; Bubeck et al., 2023; Allan et al., 2025; Salecha et al., 2024).

Existing LLM jailbreak methods typically require technical knowledge or dozens of queries (Cui et al., 2025). Ideally, model developers make attacks short-lived by quickly patching them. These classic jailbreaking methods typically do not lead to harm for individuals without technical knowledge (Chan et al., 2025). However, non-technical users might still get exposed to harmful societal biases during a layman’s (in terms of model safety) conversation just by accidentally using leak-triggering language. We aim to stress-test the robustness of LLM safety in scenarios where human input causes the LLM to utter harmful content, and evaluate whether the model can recover from it. This is relevant as prior work has shown that when LLMs take a wrong turn in a conversation, they can get lost and do not recover (Laban et al., 2025).

Jailbreak attacks are double-edged swords: while they breach LLM security, they also reveal vulnerabilities, which can be a precondition to improve model safety. We leverage the fact that the conversation history of API-based LLMs can be controlled by the user. This allows us to purposefully construct a conversation between the user and the LLM. We create a constructed conversation where the model does make a biased claim about a social group, then evaluate whether the model can recover and reject biased follow-up questions. This lightweight adversarial attack, which we call Co-Bia (Constructed Bias), uses only a single query to expose hidden societal biases in LLMs that could emerge during a conversation. We conduct a comprehensive evaluation across 11 LLMs, covering both open and closed-source models from nine leading organizations. Our contributions are as follows:

(1) We propose the CoBia methods—a set of lightweight adversarial attacks that use a constructed conversation to expose hidden societal biases in LLMs with just one query.

(2) We introduce CoBia dataset, a dataset of 112 social groups with sets of negative descriptors across six socio-demographic categories, built from three existing datasets.

(3) We evaluate societal bias scores on 11 LLMs using our techniques, comparing them to promptbased attacks, and validate results with three automated judges and human annotations.

## 2 Dataset

We re-used three common stereotype datasets to derive CoBia dataset; a unified, de-duplicated set of negative descriptors targeting different social groups. The structure of the CoBia dataset is as follows:

$$
\mathcal { D } = \{ ( c , g , n ) \mid c \in \mathcal { C } , g \in \mathcal { S } _ { c } , n \in \mathcal { N } \}
$$

where is the set of social categories, $ { \boldsymbol { S } } _ { c }$ is the set of social groups for each $c ,$ and $\mathcal { N }$ is the set of negative descriptors. One entry could be:

$$
( " \mathrm { g e n d e r " , " m e n " , " w o r t h l e s s " } )
$$

## 2.1 Selection of Stereotype Datasets

(1) RedditBias (Barikeri et al., 2021) is a dataset based on real-world Reddit discussions, providing negative descriptors for social groups. For each group, we use the dataset’s negative descriptors. We filtered out certain descriptors that did not represent attributes or statements about the social group (e.g., terms like war, prison). To identify potentially irrelevant descriptors, we used GPT-4o (OpenAI et al., 2024b) as an initial filter. Descriptors were removed only if two human annotators (authors of this paper) independently agreed on their irrelevance.

(2) SBIC (Sap et al., 2020) is a labeled dataset of social media posts created for the assessment of societal biases in language. SBIC does not associate each social group with descriptor terms. Instead, it provides stereotype annotations at the sentence level, extracted from social media posts, along with the targeted social category and additional annotations. We detected the descriptor terms in each sentence. To do so, we only retained sentences that began with the target social group. For example, we kept “men are worthless” for the social group “men.” To extract relevant descriptor terms from these statements, we used the spaCy library (Honnibal et al., 2020) for part-of-speech (POS) tagging. Specifically, we extracted adjectives when the sentence contained an auxiliary verb. For instance, given “men are worthless,” the method extracted "worthless." Otherwise, we extracted the verb and the remainder of the sentence. For example, from “women should not go to college,” we extracted “should not go to college.”

(3) StereoSet (Nadeem et al., 2021) measures stereotypical biases in language through crowdsourced "Context Association Tests" (CATs). Each CAT includes stereotypical, anti-stereotypical, and unrelated associations with a target social group. Like SBIC, StereoSet does not explicitly associate descriptor terms with target social groups. However, its more complex sentence structure compared to SBIC makes descriptor extraction using POS tags impractical. We prompt llama3.3:70b (Grattafiori et al., 2024) to extract descriptor terms for each target group.

## 2.2 CoBia Dataset

We merged the three datasets (RedditBias, SBIC, StereoSet), each containing social groups and their associated preprocessed negative descriptors. Similar social group names were consolidated under a single name after manual screening. We also assigned each social group to one of the categories: gender, orientation, nationality-origin, race, religion, or, if it does not fit any of these categories, to other. We then used Llama Guard (llama-guard3:8B) (Grattafiori et al., 2024; Inan et al., 2023), a high-precision open-source content safety classification model. We applied Llama Guard to our data using the template: “{social group} have the characteristics of the {negative descriptor},” to construct sentences that Llama Guard can evaluate. We only retained those instances that were classified as unsafe to ensure our data did not include any errors in preprocessing. Llama Guard not only classifies content as safe or unsafe, but also assigns a S category label to the data. Most instances (99%) in CoBia dataset are classified under the S10 category by Llama Guard, which represents content that demeans or dehumanizes individuals based on sensitive personal characteristics—such as race, color, ethnicity, national origin, disability, religious affiliation, caste, sexual orientation, sex, gender identity, and/or serious illness.

The social groups in our dataset, along with the count of negative descriptors for each group, are shown in Table 1. In total, the dataset includes 112 distinct social groups across six categories.

## 2.3 Extending CoBia Dataset

We extended CoBia with two additional metadata.

(1) Negative Descriptors. We first added an explanation for each descriptor $n \in \mathcal N$ (the set of negative descriptors). We prompted llama3.3:70b to explain why some individuals (without explicitly naming the associated social group to avoid refusal) might exhibit the given characteristic, along with a plausible reason for it. This approach enables us to generate contextually relevant explanations while adhering to the model’s safety constraints.

This extension defines a mapping $n \to \mathbf { e } _ { \mathrm { n e g } } ( n )$ where each negative descriptor $n \in \mathcal N$ is paired with a corresponding group-agnostic explanation.

(2) Positive Descriptors. Second, we added a set of positive descriptors for each social group by generating a set of six positive descriptors using the llama3.3:70b model. We carefully crafted prompts that take the name of a social group as input and instruct the model to produce six unique and meaningful positive traits commonly associated with that group. For each generated descriptor, the model is also prompted to produce a twosentence explanation describing why the group is perceived to possess these traits.

<table><tr><td>Category</td><td>SBIC</td><td>RedditBias</td><td>StereoSet</td><td>CoBia</td></tr><tr><td>gender</td><td>1820</td><td>0</td><td>4</td><td>1824</td></tr><tr><td>orientation</td><td>327</td><td>49</td><td>0</td><td>376</td></tr><tr><td>national-origin</td><td>24</td><td>0</td><td>204</td><td>228</td></tr><tr><td>race</td><td>80</td><td>35</td><td>46</td><td>161</td></tr><tr><td>religion</td><td>24</td><td>74</td><td>18</td><td>116</td></tr><tr><td>other</td><td>118</td><td>0</td><td>18</td><td>136</td></tr><tr><td>Total</td><td>2393</td><td>158</td><td>290</td><td>2841</td></tr></table>

Table 1: Distribution of samples across datasets and social categories. CoBia dataset is a unified, postprocessed derived from the other three datasets.

This resulted in a mapping $( g , p ) \to \mathbf { e } _ { \mathrm { p o s } } ( g , p )$ where $\mathcal { P } _ { g }$ denotes the set of six positive descriptors generated for group $g \in S _ { c } ,$ and each $p \in \mathcal { P } _ { g }$ is paired with a group-specific explanation.

## 3 CoBia Methods

Despite the integration of safety mechanisms in LLMs, societal biases remain embedded in their behavior, often concealed beneath surface-level safeguards (Bai et al., 2024b; Zhao et al., 2025b; Zhuo et al., 2023; Cantini et al., 2025; Ostrow and Lopez, 2025). Existing safeguards typically focus on blocking overtly harmful outputs, such as security risks (e.g., instructions for hacking bank accounts), while giving less attention to stereotypes and biased language targeting various social groups.

We propose CoBia, a suite of lightweight methods that use constructed conversations to elicit biased responses and expose model vulnerabilities without heavy jailbreaks. Our approach targets two settings: history-based constructed conversation (HCC) and single-block constructed conversation (SCC).

## 3.1 LLM Conversation History

An LLM inference endpoint is a server-side API that allows users (user role) to submit text prompts and receive completions generated by an LLM (assistant role). The OpenAI Chat Completions API has become the de facto standard for LLM interaction (see §A). In the Chat Completions API, the conversation history between the user and the LLM is managed by the user. Every time the user passes the entire conversation along with the new request, the model returns a response, and the user includes that model response in the next request.

This setup enables any party with access to the conversation history to purposefully construct prior messages. Previous work has explored this as a way to do few-shot prompting (Qin et al., 2025) or as a means for third parties to hijack a user–assistant conversation (Chen and Yao, 2024). However, to the best of our knowledge, it has not been studied as an adversarial attack initiated by the user, particularly in the context of societal biases. In our work, we take on the role of the user and exploit the ability to purposefully construct conversation history to test how vulnerable LLMs are to this adversarial attack. We also assess whether LLMs can uphold safety—specifically and whether they reject biased follow-up questions when the conversation includes constructed assistant responses containing societal bias.

## 3.2 Preparing CoBia Dataset

For each instance within our experiments, we investigated a single triplet from the original CoBia dataset, $( c , g , n )$ . We used the six positive descriptors associated with the social group g, denoted as $p \in \mathcal P _ { g }$ and the corresponding explanations $\mathrm { e } _ { \mathrm { p o s } } ( g , p )$ . We also selected one additional negative descriptor n<sub>2</sub> associated with each group g, ensuring it is distinct from n and the corresponding explanation ${ \mathrm { e } } _ { \mathrm { n } \mathrm { e g } } ( n _ { 2 } )$ .

## 3.3 HCC: History-based Constructed Conversation

For each instance within our experiments $( c , g , n )$ , we controlled the conversation history to purposefully construct a realistic dialogue between two roles: user and assistant. This setup triggers the LLM to respond to unethical prompts related to $( c , g , n )$ . To make the negative target descriptor n less obvious, we created the constructed conversation so that the negative descriptor was scattered among an imbalanced set of positive descriptors (Saiem et al., 2024; Bai et al., 2024a). The constructed conversation history is as follows:

(1) The user begins by asking for a list of characteristics of a particular group (e.g., “Name some characteristics of the {social group}”).

(2) The assistant responds with a numbered list of eight characteristics: six are selected from the set of positive descriptors, and two are the first and second negative target descriptors, i.e., n and $n _ { 2 }$

(3) In subsequent turns, the user asked four follow-up questions, each referring to one of four randomly selected descriptors from the numbered list of eight characteristics—always including the second negative target descriptor $n _ { 2 } .$ , and explicitly excluding the primary negative target descriptor n from the selection. Each turn included a prompt with variations of questions such as: “What makes {social group} {selected descriptor}?”

The final user question in our conversation explicitly asked whether the negative target descriptor n could be linked to another characteristic (the secondary negative descriptor $n _ { 2 } )$ , and instructed the assistant to explain the association between g and n using statistical reasoning in ten sentences. This constraint was designed to guide the model to provide a detailed explanation while avoiding vague responses. The entire constructed conversations, along with the final user question, were presented as a single query to the model.

## 3.4 SCC: Single-block Constructed Conversation

This setting differs from the HCC by presenting the entire conversation as a single user message, without the user and assistant tags. The content of the prompt remains the same as in HCC; only the structure changes.

## 3.5 Tips

We applied the following tips to both HCC and SCC. These tips lower the risk of triggering safety filters in LLMs that are sensitive to repeating explicit statements that involve negative group associations and encourage the model to rely on prior conversational context, which is more natural in a dialogue scenario. An example of an HCC prompt using these tips can be found in Appendix C.1.

(1) Anaphora. To avoid mentioning the same social group multiple times in the conversation, we used pronouns (e.g., “them,” “these people”) to reference them.

(2) Substitution. To avoid mentioning sensitive terms such as negative descriptors explicitly, we substituted them with their index number in the list, using numerical references (e.g., #6).

(3) Preemptive Disavowal. We included a disclaimer in the prompt that acknowledges ethical concerns around generalization, using this statement: “I am aware that these characteristics do not apply to all of these people. Please do not mention this again.”

## 4 Experimental Setup

## 4.1 Selected Models

We selected 11 LLMs, including closed- and open-source ones, from 9 leading organizations. We used Ollama (Ollama, 2024) to host the open-source LLMs and accessed the closed-source ones through their respective endpoints. The list of model IDs are: mistral:7b (Jiang et al., 2023), olmo2:13b (OLMo et al., 2025), command-r:35b (Cohere, 2024), llama3.1:8b, llama3.3:70b (Grattafiori et al., 2024), deepseek-v2:16b (DeepSeek-AI et al., 2024), 9 gemma2:27b (Gemma Team et al., 2024), qwen2.5:7b (Qwen et al., 2025), S gpt-3.5-turbo-0125, gpt-4o-mini-2024-07-18 (OpenAI, 2024; OpenAI et al., 2024b), and phi4:14b (Abdin et al., 2024). We set temperature=0 and top\_p=0 for deterministic outputs.

## 4.2 Baselines

0-Shot setup. We directly asked the model about the target group and its associated descriptor without providing any prior conversation. This serves as an important baseline as most of these models implement safeguards to reject unethical prompts. This setup allows us to evaluate the model’s inherent ability to handle sensitive topics and provide unbiased responses.

DAN. DAN (Do Anything Now) (Shen et al., 2024) is a well-known jailbreaking prompt that overrides AI safety restrictions. We use the latest DAN 13.0 Prompt from Lee (2023) as a baseline, asking the model about the target group and descriptor to benchmark our method against this established bypass.

R-Play. Roleplay jailbreaking instructs a model to adopt a specific persona or engage in hypothetical scenarios to bypass safety constraints. We used the role auto tune from Zhao et al. (2025a, Figure 2) and prompted GPT-4 to determine the expert role. For most of the CoBia dataset, this template resulted in the role of a cultural sociologist, which we adopted consistently across all experiments. The full role prompt is provided in Appendix C.2.

## 4.3 Judge Metrics

We used two “LLM-as-a-judge” systems (Zheng et al., 2023; Li et al., 2024a; Zhu et al., 2025; Gu et al., 2025) and the Granite Guardian model (Padhi et al., 2024) as judge metrics for our experiments.

Llama 3 series models have demonstrated strong performance as judges, achieving results comparable to closed-source models (Thakur et al., 2025; Tan et al., 2024; Bavaresco et al., 2024; Raju et al., 2024; Moniri et al., 2025; Koutcheme et al., 2025). Based on these findings, we adopted llama3.3:70b as our main LLM-as-a-judge system. We demonstrate the alignment of these judges with human preferences in Section 5.4, and provide our judge prompts in Appendix C.3.

Bias Judge. We used the prompt from Fan et al. (2025, Figure 17) as our judge with some modifications. Their prompt instructed the model to assess whether an answer in a dialogue agreed with a biased opinion, considering both direct and implicit bias while excluding cases that explicitly reject bias. The output was a binary yes/no verdict on bias. Our modification added extra guidance to reduce over-labeling as biased, clarifying that factual statements describing a group without unfair implications are not considered biased, and that generalizations which do not impose restrictions on a group should likewise not be labeled as biased.

Granite Judge. IBM’s Granite Guardian (Padhi et al., 2024) is an open-source LLM designed to detect risks in prompts and responses. It outperforms similar models, such as Llama Guard (Inan et al., 2023) and Shield Gemma (Zeng et al., 2024a), in terms of F1 and recall but exhibits lower precision. It is trained on a mix of human-annotated data and synthetic datasets from diverse sources. Granite Guardian can identify various risks, including harmful content, societal bias, profanity, sexual material, and unethical behavior. We use this model as an additional judge. Like Bias Judge, it also functions as a binary classifier, returning yes if the output contains risk and no otherwise.

NLI Judge. In Bias Judge, the model assesses bias directly based on its own understanding, guided by the prompt. We propose an alternative judge inspired by natural language inference (NLI) (Bowman et al., 2015), which focuses on the logical relationship between a biased opinion and the model’s output. The prompt instructs the model to classify the relationship as one of four categories: enforces, agrees, neutral, or negates. This method helps identify abnormal behavior in other judges when the logical relationship breaks down. We later observed that the model rarely chose neutral and enforces, so we classify enforces and agrees as “yes” and neutral or negates as “no” regarding bias.

<table><tr><td rowspan="2">Models</td><td colspan="6">Bias Judge</td><td colspan="6">Granite Judge</td></tr><tr><td>UCC</td><td>HCC</td><td>SCC</td><td>R-Play</td><td>DAN</td><td>0-Shot</td><td>UCC</td><td>HCC</td><td>SCC</td><td>R-Play</td><td>DAN</td><td>0-Shot</td></tr><tr><td>Mmistral:7b</td><td>38.41</td><td>33.21</td><td>13.19</td><td>5.30</td><td>29.78</td><td>11.05</td><td>27.57</td><td>22.29</td><td>7.13</td><td>10.53</td><td>14.46</td><td>7.94</td></tr><tr><td>olmo2:13b</td><td>49.48</td><td>30.09</td><td>40.75</td><td>2.01</td><td>1.90</td><td>3.49</td><td>19.61</td><td>10.76</td><td>10.44</td><td>5.44</td><td>2.34</td><td>3.55</td></tr><tr><td>command-r:35b</td><td>82.59</td><td>75.21</td><td>57.28</td><td>8.47</td><td>43.17</td><td>17.54</td><td>46.35</td><td>33.46</td><td>21.67</td><td>8.37</td><td>21.33</td><td>6.48</td></tr><tr><td>0llama3.1:8b</td><td>65.84</td><td>61.22</td><td>21.40</td><td>1.53</td><td>0.84</td><td>21.77</td><td>28.07</td><td>22.16</td><td>8.91</td><td>0.33</td><td>0.64</td><td>7.61</td></tr><tr><td>01llama3.3:70b</td><td>85.54</td><td>77.13</td><td>72.53</td><td>2.35</td><td>42.13</td><td>19.62</td><td>48.82</td><td>33.56</td><td>28.11</td><td>14.91</td><td>38.00</td><td>13.18</td></tr><tr><td>gemma2:27b 6</td><td>27.21</td><td>16.24</td><td>18.72</td><td>2.87</td><td>64.91</td><td>2.26</td><td>11.96</td><td>5.63</td><td>6.96</td><td>3.16</td><td>32.28</td><td>1.44</td></tr><tr><td>deepseek-v2:16b</td><td>28.73</td><td>17.94</td><td>15.61</td><td>10.44</td><td>2.74</td><td>16.54</td><td>16.66</td><td>9.07</td><td>8.41</td><td>6.85</td><td>1.18</td><td>6.28</td></tr><tr><td>phi4:14b</td><td>51.10</td><td>46.29</td><td>16.1</td><td>2.05</td><td>2.90</td><td>7.71</td><td>11.08</td><td>7.07</td><td>4.39</td><td>4.80</td><td>2.15</td><td>3.99</td></tr><tr><td>女 qwen2.5:7b</td><td>83.60</td><td>57.37</td><td>72.40</td><td>15.34</td><td>33.09</td><td>35.72</td><td>42.98</td><td>21.95</td><td>29.21</td><td>13.51</td><td>16.41</td><td>12.84</td></tr><tr><td>S gpt-3.5-turbo-0125</td><td>59.67</td><td>46.70</td><td>45.25</td><td>6.10</td><td>21.23</td><td>5.87</td><td>25.91</td><td>13.91</td><td>16.03</td><td>7.84</td><td>9.04</td><td>4.89</td></tr><tr><td>S gpt-4o-mini-2024-07-18</td><td>49.73</td><td>31.39</td><td>40.77</td><td>4.93</td><td>16.60</td><td>17.49</td><td>19.55</td><td>9.5</td><td>11.96</td><td>6.57</td><td>5.05</td><td>7.09</td></tr></table>

Table 2: Experimental results on the CoBia dataset across different models and methods, reported as macro averages over six social categories. Best scores are bolded, and second-best scores are underlined.

## 5 Results

## 5.1 Main Result

The results for comparing CoBia methods to other baselines are shown in Table 2. Both SCC and HCC outperformed the baselines in most settings as per both Bias Judge and Granite Judge. Similar patterns were observed with the NLI Judge (see Table 6 in Appendix D). UCC denotes the union of HCC and SCC, where an instance is judged biased if either method judged it as biased. The results in Table 2 show macro-averaged scores across our six bias categories. The number of instances per social category in our dataset is imbalanced (see Table 1) such that the dominant category could skew the overall results. For micro-averaged scores see Tables 4-6 in Appendix D.

We marked models as “heavily biased” if their UCC score exceeded 80% with Bias Judge. Notably, every model surpassing this threshold also scored above 40% with the Granite Judge and 69% with the NLI Judge, indicating strong alignment among the judge mechanisms in identifying heavily biased behavior. The models we found to be heavily biased are llama3.3:70b, command-r:35b, and qwen2.5:7b. The baseline methods—R-Play, DAN, and 0-Shot—showed comparatively lower bias scores (often <20%), highlighting the attention given to safety in these methods. The models gpt-4o-mini, gpt-3.5-turbo, llama3.1:8b, olmo2:13b, and mistral:7b showed moderate bias scores, though their rankings varied slightly depending on the judging method used.

gemma2:27b and deepseek-v2:16b showed notably low CoBia-based bias scores under both Bias Judge and Granite Judge. In the case of gemma2:27b, the 0-Shot bias score was particularly low across both judges and much lower than its CoBia scores, suggesting strong robustness to bias overall. In contrast, deepseek-v2:16b had a 0-Shot score more comparable to its CoBia scores. Our analysis of deepseek-v2:16b outputs showed that the model often failed to follow instructions precisely, producing long, vague, and hedging responses. As a result, judges frequently classified these outputs as unbiased, which explains the small gap between its CoBia and 0-Shot scores.

HCC vs SCC. HCC and SCC showed complementary behavior in exposing societal biases. For mistral-7b, llama3.1:8b, and phi4:14b, HCC consistently yielded higher bias rates than SCC across three judges. Conversely, models such as gpt-4o-mini showed an opposite pattern from prior ones, with SCC performing better. In some cases, like deepseek-v2:16b and gemma2-27b, the difference between HCC and SCC was minimal (less than 3%), yet their combination resulted in a substantial increase, boosting the overall bias score by at least 30%. These patterns show the value of using both methods, as each uncovers biases better in different models. We hypothesize that models where HCC shows a higher bias score than SCC are either instruction-tuned on formatted conversations or use system templates that more clearly define user and assistant roles, leading the model to “believe” more strongly in the content of the constructed conversation. Overall, HCC outperformed SCC across more models confirmed with three judges.

## 5.2 Bias Distribution Across Social Categories

We report NLI Judge and Bias Judge scores across six social categories—gender, sex orientation, religion, race, national origin, and other—for seven models under both 0-Shot and UCC settings (see Figure 1). Overall, UCC yielded higher bias scores across models and categories compared to the 0-Shot setting. However, bias was not evenly distributed across categories: national origin consistently showed the highest levels of bias in all settings. For example, qwen2.5:7b and command-r:35b showed high bias scores (near 1.0) in national origin under UCC. In contrast, race, religion, and orientation generally showed lower bias scores. For instance, in gemma2:27b, the bias score for orientation, race and religion remained below 0.2. This indicates that these three dimensions may be more closely monitored, either through model safeguards or data filtering. A comparison between the Bias Judge and NLI Judge showed that the NLI Judge is more sensitive to the national origin category. Even in the 0-shot setting, this category received a high bias score from the NLI Judge, whereas the Bias Judge did not show the same pattern.

![](images/7611636c7b5428283408d81c9ffa18f25154f25b088731da81350caf4b1e6804.jpg)  
Figure 1: Bias Judge (top) and NLI Judge (down) scores across six social categories for seven models, shown under two settings: UCC (left) and 0-Shot (right).

## 5.3 Effect of Model Size on Bias Scores

We used the qwen2.5 model family with varying parameter sizes—3B, 7B, 14B, 32B and 72B—to examine the effect of model size on bias scores. This family offers a broad range of sizes, allowing for consistent comparison. Results are reported for four methods based on Bias Judge: 0-Shot, SCC, HCC, and UCC (see Figure 2). The overall trend in line slopes across model sizes is consistent, except between 3B and 7B. For the 3B model, SCC shows higher bias than HCC and 0-Shot—likely because the model lacks strong instruction-following capabilities and responds more directly to the conversation examples in SCC. However, there is no clear correlation between model size and bias score across methods. In some cases, larger models are safer; in others, not. Notably, the 32B model shows the best safety performance, though without an obvious explanation that we could find.

![](images/596129ccf7a1541483abb35ef8aca4a3e24479d288009c6c1e1f5817bc957a33.jpg)  
Figure 2: Bias Judge scores for the qwen2.5 model family at different model sizes (3B, 7B, 14B, 32B and 72B).

![](images/7934438294d64166afb1c8761467af07acd2fe6e48586672f5faa229a8ca7087.jpg)  
Figure 3: Average bias scores across judge types. HCC and SCC outperform all baselines.

## 5.4 Comparison of Judges

We compared three judge methods, averaged over 11 models across both CoBia methods and baselines (see Figure 3). HCC and SCC consistently outperform all baselines, regardless of the chosen judge. However, the relative ranking of the baselines (DAN, 0-Shot, and R-Play) varies across judges. Among the judges, the Bias Judge reports the highest scores across methods, indicating greater sensitivity to bias.

We worked with four human evaluators (details in Appendix B) to judge bias of the model outputs. Two were instructed with the Bias Judge prompt, and two with the NLI Judge prompt. Each annotator evaluated 300 randomly selected outputs from all settings. We use pairwise agreement, Cohen’s κ, and Fleiss’ κ as alignment metrics.

DAN receives relatively higher scores from the Bias Judge and Granite Judge, but lower scores from the NLI Judge. In Table 2, gemma2:27b and llama3.3:70b, DAN outperformed or was on par with CoBia methods. We found that in DAN cases where the judges disagreed, the outputs flagged by the Bias Judge and Granite Judge used offensive or inappropriate language. However, these outputs were not necessarily biased against the social group mentioned in the question. Since the NLI judge focuses mostly on the logical relationship between the biased opinion toward the targeted group and the model’s response, regardless of the wording used, it often labeled such outputs as not biased. However, for gpt-4o-mini and qwen2.5:7b, where the NLI judge also assigned relatively high DAN scores, the detected biases were indeed directed towards the social target groups. Overall, when the Bias Judge and NLI Judge disagreed in DAN cases, the NLI Judge demonstrated stronger reliability, aligning with the majority vote of all four human annotators in 83% of these cases.

<table><tr><td>Metric</td><td>Value</td><td>Note</td></tr><tr><td>Pairwise agreement (Granite Judge with NLI Judge / Bias Judge)</td><td>0.70 / 0.67</td><td>Granite Judge aligns slightly more with NLI Judge in pairwise agreement</td></tr><tr><td>Cohen&#x27;s κ (Granite Judge with NLI Judge / Bias Judge)</td><td>0.13 / 0.16</td><td>Granite Judge shows higher agreement with Bias Judge than with NLI Judge</td></tr><tr><td>Cohen&#x27;s κ (Granite Judge with Human annota- tors)</td><td>0.10</td><td>Low agreement between Granite Judge and human an- notators</td></tr><tr><td>Fleiss&#x27; κ (Humans with Bias Judge prompt)</td><td>0.54</td><td>Moderate agreement among annotators when guided by the Bias Judge prompt</td></tr><tr><td>Fleiss&#x27; κ (Humans with NLI Judge prompt)</td><td>0.55</td><td>Similar agreement level among annotators when guided by the NLI Judge prompt</td></tr><tr><td>Pairwise agreement (NLI Judge with Bias Judge)</td><td>0.79</td><td>Strong alignment between the two automatic judges</td></tr><tr><td>Cohen&#x27;s κ (NLI Judge with Bias Judge)</td><td>0.53</td><td>Moderate agreement between the two automatic judges</td></tr><tr><td>Human–NLI Judge alignment in DAN disagree- ment cases</td><td>83%</td><td>NLI Judge demonstrates stronger reliability in cases where the two judges disagree</td></tr><tr><td>Number of human annotations</td><td>4 annotators, 300 out- puts each (1200 total)</td><td>Balances annotation cost and coverage for cross- validating judge reliability</td></tr></table>

Table 3: Agreement metrics between human annotators, Granite Judge, Bias Judge, and NLI Judge.

Granite Judge yielded lower scores across all methods. This may be due to its training on shorter output responses, whereas our model outputs involved longer text with more complex reasoning—often containing implicit biases or involving bias acknowledgment followed by rejection, or vice versa. In terms of pairwise agreement, Granite Judge aligned more with the NLI Judge (0.70 vs. 0.67), though most randomly selected outputs are in fact unbiased, and there is a possibility of agreement occurring by chance. Calculating Cohen’s κ, we found Granite Judge to agree more with the Bias Judge than the NLI Judge (κ = 0.16 vs. $\kappa = 0 . 1 3 )$ . The human annotators had an average Cohen’s κ = 0.10 with Granite Judge.

NLI Judge gave higher scores to 0-shot method compared to other baselines. However, this trend was not consistent across the other judges. Since the NLI Judge mostly evaluated the logical relationship between the biased opinion and the model output, it may not fully align with the broader definition of bias used by the Bias Judge. Human annotators instructed with each template showed higher agreement with each other and with the judge they were instructed to: Fleiss $\kappa = 0 . 5 4$ for the Bias Judge and Fleiss $\kappa = 0 . 5 5$ for the NLI Judge. Notably, the pairwise agreement between NLI and Judge Bias Judge was 0.79, with a Cohen’s κ of 0.53. The summary of the numbers discussed here is shown in Table 3.

## 6 Related Work

Jailbreaking LLMs. Methods to jailbreak safetyaligned LLMs range from manual techniques to automated approaches, including prompt- and tokenbased methods such as R-Play and DAN, gradientbased attacks that require access to model parameters, and Infrastructure-level attacks that inject external knowledge or APIs into prompts. We categorize various jailbreak methods used to attack and demonstrate the vulnerabilities of LLMs, following the categories proposed by Purpura et al. (2025):

\- Prompt and Token-Based. These attacks exploit LLM vulnerabilities by crafting malicious prompts designed to bypass safety mechanisms. Techniques in this category include prompt injection, style injection, refusal suppression, many-shot jailbreaking, prompt obfuscation, prompt translation, prompt encryption, and both simple and complex role-playing scenarios (Radharapu et al., 2023; Zhou et al., 2024a; Deng et al., 2023; Mehrotra et al., 2024; Pape et al., 2025; Yu et al., 2024; Hong et al., 2024; Liu et al., 2024; Guo et al., 2024a; Paulus et al., 2024; Chao et al., 2023; Shen et al., 2024; Ge et al., 2024; Russinovich et al., 2025; Zhang et al., 2024; Jiang et al., 2025; Zeng et al., 2024b; Jiang et al., 2024; Zhou et al., 2024b; Yuan et al., 2024; Yong et al., 2023; Wallace et al., 2019; Bai et al., 2024c; Ren et al., 2024; Yang et al., 2024).

\- Gradient-Based. These attacks only work when the model parameters are accessible, as they require access to the parameters in order to apply gradient descent and identify the most effective attacks (Zou et al., 2023; Shin et al., 2020; Wichers et al., 2024; Geisler et al., 2024).

\- Infrastructure. These attacks involve injecting content into, extracting data from, or otherwise modifying the underlying systems and services that support the target LLM (Carlini et al., 2021; Kariyappa et al., 2021; Shafran et al., 2025; Li et al., 2025; Deng et al., 2024; Chaudhari et al., 2024; Wang et al., 2024; Pasquini et al., 2024; Cohen et al., 2024).

Most of these attacks require technical expertise and either significant computational resources or a large number of queries. This makes them less accessible to non-technical users and less suited for evaluating LLM safety in real-world scenarios (Chan et al., 2025). Among these, promptbased attacks are the simplest. Our method falls into this category, using only a single query with minimal computation. Importantly, our HCC attack can be countered with a straightforward patch: disabling user control over conversation history. In long run, extending existing safety strategies to full conversations—rather than limiting them to isolated prompts—provides an effective mitigation. Notably, some models, such as Gemma 2, already exhibit stronger adherence to these practices, based on our results.

Societal Bias in LLMs. Evaluation of societal bias in LLMs is commonly categorized as intrinsic vs. extrinsic (Zayed et al., 2024; Li et al., 2024b). Intrinsic methods typically require use static and contextualized word embeddings (Wan et al., 2023; May et al., 2019; Caliskan et al., 2017; Guo and Caliskan, 2021; Charlesworth et al., 2022; Garg et al., 2018) or token probabilities (Webster et al., 2021; Felkner et al., 2023); see also (Chu et al., 2024). As models become increasingly proprietary and limited to API access, obtaining embeddings becomes harder, shifting bias evaluation toward generation-based (extrinsic) methods. Bai et al.

(2024b) found that most benchmarks used in extrinsic methods (Parrish et al., 2022; Dhamala et al., 2021; Tamkin et al., 2023) reported little bias in recent models, despite evidence of persistent implicit biases—consistent with our 0-shot baseline. To better uncover such biases, researchers have turned to jailbreaking methods (Lee and Seong, 2025). While stereotypes have appeared in LLM safety and jailbreak benchmarks (Wang et al., 2023), they have not been the primary focus. Personabased attacks, a widely studied form of promptbased attacks, are commonly used in stereotype research (Deshpande et al., 2023; Gupta et al., 2024). There are also other attacks, such as those using persuasive multi-turn prompts to elicit biased or toxic responses (Ge et al., 2025). Our work also falls under the category of adversarial attacks, aiming to assess the robustness of LLM safety in scenarios where a conversation leads to content reflecting societal bias. Specifically, we evaluate whether the model can recover appropriately or whether it continues the dialogue and amplifies the bias.

## 7 Conclusion

We aim to stress-test the robustness of LLM safety in scenarios where human input leads the model to generate harmful content and evaluate whether the model can recover from it.

To this end, we introduce CoBia, a suite of lightweight adversarial attacks that reveal societal bias in LLMs through constructed conversations. Our approach simulates biased conversational contexts using constructed conversations, either by leveraging structured conversation history or embedding the full exchange within a non-structured prompt. We evaluate both methods on 11 LLMs from 9 organizations across six social categories, comparing them to three baselines using three automated bias judges.

Our methods consistently outperform the baselines in exposing bias, in agreement across all judges. We observe that LLMs display greater bias related to national origin than to religion, race, or sexual orientation. To patch this adversarial attack, models would need to restrict user control over the conversation history, for example. However, ensuring safety in all cases requires extending safety mechanisms beyond isolated prompts to entire dialogues. CoBia provides a practical tool for stress-testing LLM safety in more realistic conversational settings.

## Limitations

We are aware of five main limitations of our work.

(1) We evaluated 11 prominent LLMs, selecting at least one from each of nine leading organizations to ensure balanced coverage. Although more models could have been included, our experiments involved over 800,000 queries, underscoring the scale of the evaluation. This model set offers a practical trade-off between feasibility and coverage, supporting the generalizability of our findings.

(2) We tested only two set of conversational templates to construct CoBia attacks. Variations in style, length, or tone—such as longer dialogues, brief exchanges, or emotional language—could plausibly influence outcomes. However, our goal was not to exhaustively sample dialogue types, but to present a minimal, controlled concrete validation. Even with this number of templates, we triggered bias in most of the models, demonstrating the validity of our approach and related concerns. The constructed conversations enabled by this method allow scholars and practitioners to explore a broader range of conversational styles for auditing LLMs.

(3) Although we employ three automated judges—Bias Judge, Granite Judge, and an NLI Judge—to score bias in LLM outputs, our human evaluation is limited in scope. Nonetheless, we observed strong alignment between these automated judges and the human annotations we collected. This triangulation reinforces the credibility of our bias assessments, even without large-scale manual labeling.

(4) While our CoBia method exposes model vulnerabilities through adversarially constructed conversations, these interactions do not directly reflect typical user behavior. Real-world users are unlikely to encounter the sequences we design. However, this limitation is inherent to red-teaming: the goal is to stress-test models to reveal hidden failure modes, not to estimate how often such behaviors occur in deployment. These results are best interpreted as stress-test signals, not as indicators of deployment-time occurrence rates.

(5) Finally, while our findings highlight weaknesses in current safety systems, we do not propose concrete mitigation strategies beyond high-level suggestions (e.g., limiting user control over conversation history). Future work should build on these insights to develop robust defenses against conversational adversarial attacks. Ultimately, fixes to proprietary models need to come from model providers. In fact, as of March 11, 2025, OpenAI started to manage a unique ID for previous chat history in its new API. However, the previous Chat Completions API (which is supported indefinitely) remains the de facto standard, and a conversation history is still manually managed by the user.

Despite these limitations, the consistent crossmodel results, multi-judge agreement, and empirical validation underscore the utility of CoBia as a lightweight method for stress-testing LLM safety in realistic dialogue settings.

## Ethics Statement

This research investigates vulnerabilities in LLMs by introducing lightweight adversarial prompts that simulate biased user–assistant conversations. The objective is not to exploit or disseminate harmful outputs, but to systematically study how constructed conversations can reveal embedded societal biases in LLMs. The findings aim to inform the design of more robust safety mechanisms and support the ethical deployment of LLMs in real-world settings.

The study does not involve the collection of personal data or subjective human responses. A small group of annotators labeled model outputs according to predefined instructions for validation purposes. This process was limited in scope, posed minimal risk, and complied with relevant legal, regulatory, and ethical standards. The research relies exclusively on publicly available datasets (licensed under MIT and CC BY-SA 4.0) and adheres to strict privacy guidelines, with a commitment to fairness, transparency, and harm reduction. Our CoBia dataset, will also be released alongside the paper—under a CC BY-SA 4.0 license—together with our code, to support open and reproducible research. The long-term goal is to contribute to the development of safer and more responsible AI systems.

## References

Marah Abdin, Jyoti Aneja, Harkirat Behl, Sébastien Bubeck, Ronen Eldan, Suriya Gunasekar, Michael Harrison, Russell J. Hewett, Mojan Javaheripi, Piero Kauffmann, James R. Lee, Yin Tat Lee, Yuanzhi Li, Weishung Liu, Caio C. T. Mendes, Anh Nguyen, Eric Price, Gustavo de Rosa, Olli Saarikivi, and 8 others. 2024. Phi-4 technical report. Preprint, arXiv:2412.08905.

Kevin Allan, Jacobo Azcona, Somayajulu Sripada, Georgios Leontidis, Clare AM Sutherland, Louise H

Phillips, and Douglas Martin. 2025. Stereotypical bias amplification and reversal in an experimental model of human interaction with generative artificial intelligence. Royal Society Open Science, 12(4):241472.

Ge Bai, Jie Liu, Xingyuan Bu, Yancheng He, Jiaheng Liu, Zhanhui Zhou, Zhuoran Lin, Wenbo Su, Tiezheng Ge, Bo Zheng, and Wanli Ouyang. 2024a. MT-bench-101: A fine-grained benchmark for evaluating large language models in multi-turn dialogues. In Proceedings of the 62nd Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 7421–7454, Bangkok, Thailand. Association for Computational Linguistics.

Xuechunzi Bai, Angelina Wang, Ilia Sucholutsky, and Thomas L Griffiths. 2024b. Measuring implicit bias in explicitly unbiased large language models. In NeurIPS 2024 Workshop on Behavioral Machine Learning.

Yang Bai, Ge Pei, Jindong Gu, Yong Yang, and Xingjun Ma. 2024c. Special characters attack: Toward scalable training data extraction from large language models. Preprint, arXiv:2405.05990.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph, Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El-Showk, Nelson Elhage, Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, and 12 others. 2022a. Training a helpful and harmless assistant with reinforcement learning from human feedback. Preprint, arXiv:2204.05862.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, and 32 others. 2022b. Constitutional ai: Harmlessness from ai feedback. Preprint, arXiv:2212.08073.

Soumya Barikeri, Anne Lauscher, Ivan Vulic, and Goran´ Glavaš. 2021. RedditBias: A real-world resource for bias evaluation and debiasing of conversational language models. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 1941–1955, Online. Association for Computational Linguistics.

Anna Bavaresco, Raffaella Bernardi, Leonardo Bertolazzi, Desmond Elliott, Raquel Fernández, Albert Gatt, Esam Ghaleb, Mario Giulianelli, Michael Hanna, Alexander Koller, André F. T. Martins, Philipp Mondorf, Vera Neplenbroek, Sandro Pezzelle, Barbara Plank, David Schlangen, Alessandro Sug lia, Aditya K Surikuchi, Ece Takmaz, and Alberto Testoni. 2024. Llms instead of human judges? a large scale empirical study across 20 nlp evaluation tasks. Preprint, arXiv:2406.18403.

Emily M Bender, Timnit Gebru, Angelina McMillan-Major, and Shmargaret Shmitchell. 2021. On the dangers of stochastic parrots: Can language models be too big? In Proceedings of the 2021 ACM conference on fairness, accountability, and transparency, pages 610–623.

Anjanava Biswas and Wrick Talukdar. 2023. Guardrails for trust, safety, and ethical development and deployment of large language models (llm). Journal of Science & Technology, 4(6):55–82.

Rishi Bommasani, Drew A. Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S. Bernstein, Jeannette Bohg, Antoine Bosselut, Emma Brunskill, Erik Brynjolfsson, Shyamal Buch, Dallas Card, Rodrigo Castellon, Niladri Chatterji, Annie Chen, Kathleen Creel, Jared Quincy Davis, Dora Demszky, and 95 others. 2022. On the opportunities and risks of foundation models. Preprint, arXiv:2108.07258.

Samuel R. Bowman, Gabor Angeli, Christopher Potts, and Christopher D. Manning. 2015. A large annotated corpus for learning natural language inference. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 632–642, Lisbon, Portugal. Association for Computational Linguistics.

Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, Harsha Nori, Hamid Palangi, Marco Tulio Ribeiro, and Yi Zhang. 2023. Sparks of artificial general intelligence: Early experiments with GPT-4. Preprint, arXiv:2303.12712.

Aylin Caliskan, Joanna J Bryson, and Arvind Narayanan. 2017. Semantics derived automatically from language corpora contain human-like biases. Science, 356(6334):183–186.

Riccardo Cantini, Giada Cosenza, Alessio Orsino, and Domenico Talia. 2025. Are Large Language Models Really Bias-Free? Jailbreak Promptsfor Assessing Adversarial Robustness to Bias Elicitation, page 52–68. Springer Nature Switzerland.

Nicholas Carlini, Florian Tramer, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom Brown, Dawn Song, Ulfar Erlingsson, and 1 others. 2021. Extracting training data from large language models. In 30th USENIX security symposium (USENIX Security 21), pages 2633–2650.

Yik Siu Chan, Narutatsu Ri, Yuxin Xiao, and Marzyeh Ghassemi. 2025. Speak easy: Eliciting harmful jailbreaks from llms with simple interactions. Preprint, arXiv:2502.04322.

Patrick Chao, Alexander Robey, Edgar Dobriban, Hamed Hassani, George J Pappas, and Eric Wong. 2023. Jailbreaking black box large language models

in twenty queries. In R0-FoMo: Robustness ofFewshot and Zero-shot Learning in Large Foundation Models.

Tessa ES Charlesworth, Aylin Caliskan, and Mahzarin R Banaji. 2022. Historical representations of social groups across 200 years of word embeddings from google books. Proceedings ofthe National Academy ofSciences, 119(28):e2121798119.

Harsh Chaudhari, Giorgio Severi, John Abascal, Matthew Jagielski, Christopher A. Choquette-Choo, Milad Nasr, Cristina Nita-Rotaru, and Alina Oprea. 2024. Phantom: General trigger attacks on retrieval augmented language generation. Preprint, arXiv:2405.20485.

Zheng Chen and Buhui Yao. 2024. Pseudo-conversation injection for llm goal hijacking. Preprint, arXiv:2410.23678.

Zhibo Chu, Zichong Wang, and Wenbin Zhang. 2024. Fairness in large language models: A taxonomic survey. ACM SIGKDD explorations newsletter, 26(1):34–48.

Stav Cohen, Ron Bitton, and Ben Nassi. 2024. Unleashing worms and extracting data: Escalating the outcome of attacks against rag-based inference in scale and severity using jailbreaking. Preprint, arXiv:2409.08045.

Cohere. 2024. Command r: Retrieval-augmented generation at production scale.

Tiehan Cui, Yanxu Mao, Peipei Liu, Congying Liu, and Datao You. 2025. Exploring jailbreak attacks on LLMs through intent concealment and diversion. In Findings of the Association for Computational Linguistics: ACL 2025, pages 20754–20768, Vienna, Austria. Association for Computational Linguistics.

Josef Dai, Xuehai Pan, Ruiyang Sun, Jiaming Ji, Xinbo Xu, Mickel Liu, Yizhou Wang, and Yaodong Yang. 2024. Safe RLHF: Safe reinforcement learning from human feedback. In The Twelfth International Conference on Learning Representations.

DeepSeek-AI, Aixin Liu, Bei Feng, Bin Wang, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Dengr, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fuli Luo, Guangbo Hao, Guanting Chen, and 138 others. 2024. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. Preprint, arXiv:2405.04434.

Boyi Deng, Wenjie Wang, Fuli Feng, Yang Deng, Qifan Wang, and Xiangnan He. 2023. Attack prompt generation for red teaming and defending large language models. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 2176–2189, Singapore. Association for Computational Linguistics.

Gelei Deng, Yi Liu, Kailong Wang, Yuekang Li, Tianwei Zhang, and Yang Liu. 2024. Pandora: Jailbreak gpts by retrieval augmented generation poisoning. Preprint, arXiv:2402.08416.

Ameet Deshpande, Vishvak Murahari, Tanmay Rajpurohit, Ashwin Kalyan, and Karthik Narasimhan. 2023. Toxicity in chatgpt: Analyzing persona-assigned language models. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 1236–1270, Singapore. Association for Computational Linguistics.

Jwala Dhamala, Tony Sun, Varun Kumar, Satyapriya Krishna, Yada Pruksachatkun, Kai-Wei Chang, and Rahul Gupta. 2021. Bold: Dataset and metrics for measuring biases in open-ended language generation. In Proceedings of the 2021 ACM conference on fairness, accountability, and transparency, pages 862–872.

Zhiting Fan, Ruizhe Chen, Tianxiang Hu, and Zuozhu Liu. 2025. FairMT-bench: Benchmarking fairness for multi-turn dialogue in conversational LLMs. In The Thirteenth International Conference on Learning Representations.

Virginia Felkner, Ho-Chun Herbert Chang, Eugene Jang, and Jonathan May. 2023. WinoQueer: A communityin-the-loop benchmark for anti-LGBTQ+ bias in large language models. In Proceedings of the 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 9126– 9140, Toronto, Canada. Association for Computational Linguistics.

Susan T Fiske, Amy JC Cuddy, Peter Glick, and Jun Xu. 2018. A model of (often mixed) stereotype content: Competence and warmth respectively follow from perceived status and competition. In Social cognition, pages 162–214. Routledge.

Isabel O. Gallegos, Ryan A. Rossi, Joe Barrow, Md Mehrab Tanjim, Sungchul Kim, Franck Dernoncourt, Tong Yu, Ruiyi Zhang, and Nesreen K. Ahmed. 2024. Bias and fairness in large language models: A survey. Computational Linguistics, 50(3):1097– 1179.

Deep Ganguli, Liane Lovitt, Jackson Kernion, Amanda Askell, Yuntao Bai, Saurav Kadavath, Ben Mann, Ethan Perez, Nicholas Schiefer, Kamal Ndousse, Andy Jones, Sam Bowman, Anna Chen, Tom Conerly, Nova DasSarma, Dawn Drain, Nelson Elhage, Sheer El-Showk, Stanislav Fort, and 17 others. 2022. Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned. Preprint, arXiv:2209.07858.

Nikhil Garg, Londa Schiebinger, Dan Jurafsky, and James Zou. 2018. Word embeddings quantify 100 years of gender and ethnic stereotypes. Proceedings of the National Academy of Sciences, 115(16):E3635– E3644.

Suyu Ge, Chunting Zhou, Rui Hou, Madian Khabsa, Yi-Chia Wang, Qifan Wang, Jiawei Han, and Yuning Mao. 2024. MART: Improving LLM safety with multi-round automatic red-teaming. In Proceedings ofthe 2024 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 1927–1937, Mexico City, Mexico. Association for Computational Linguistics.

Yubin Ge, Neeraja Kirtane, Hao Peng, and Dilek Hakkani-Tür. 2025. Llms are vulnerable to malicious prompts disguised as scientific language. Preprint, arXiv:2501.14073.

Simon Geisler, Tom Wollschläger, MHI Abdalla, Johannes Gasteiger, and Stephan Günnemann. 2024. Attacking large language models with projected gradient descent. In ICML 2024 Next Generation ofAI Safety Workshop.

Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, Johan Ferret, Peter Liu, Pouya Tafti, Abe Friesen, Michelle Casbon, Sabela Ramos, Ravin Kumar, Charline Le Lan, Sammy Jerome, and 179 others. 2024. Gemma 2: Improving open language models at a practical size. Preprint, arXiv:2408.00118.

Google Dev. 2024. Openai compatibility | gemini api. https://ai.google.dev/gemini-api/docs/ openai.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, and 542 others. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.

Jiawei Gu, Xuhui Jiang, Zhichao Shi, Hexiang Tan, Xuehao Zhai, Chengjin Xu, Wei Li, Yinghan Shen, Shengjie Ma, Honghao Liu, Saizhuo Wang, Kun Zhang, Yuanzhuo Wang, Wen Gao, Lionel Ni, and Jian Guo. 2025. A survey on llm-as-a-judge. Preprint, arXiv:2411.15594.

Wei Guo and Aylin Caliskan. 2021. Detecting emergent intersectional biases: Contextualized word embeddings contain a distribution of human-like biases. In Proceedings ofthe 2021 AAAI/ACM Conference on AI, Ethics, and Society, pages 122–133.

Xingang Guo, Fangxu Yu, Huan Zhang, Lianhui Qin, and Bin Hu. 2024a. Cold-attack: jailbreaking llms with stealthiness and controllability. In Proceedings of the 41st International Conference on Machine Learning, pages 16974–17002.

Yufei Guo, Muzhe Guo, Juntao Su, Zhou Yang, Mengqiu Zhu, Hongfei Li, Mengyang Qiu, and

Shuo Shuo Liu. 2024b. Bias in large language models: Origin, evaluation, and mitigation. Preprint, arXiv:2411.10915.

Shashank Gupta, Vaishnavi Shrivastava, Ameet Deshpande, Ashwin Kalyan, Peter Clark, Ashish Sabharwal, and Tushar Khot. 2024. Bias runs deep: Implicit reasoning biases in persona-assigned LLMs. In The Twelfth International Conference on Learning Representations.

Zhang-Wei Hong, Idan Shenfeld, Tsun-Hsuan Wang, Yung-Sung Chuang, Aldo Pareja, James R Glass, Akash Srivastava, and Pulkit Agrawal. 2024. Curiosity-driven red-teaming for large language models. In The Twelfth International Conference on Learning Representations.

Matthew Honnibal, Ines Montani, Sofie Van Landeghem, and Adriane Boyd. 2020. spacy: Industrialstrength natural language processing in python. https://doi.org/10.5281/zenodo.1212303.

Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, and Madian Khabsa. 2023. Llama Guard: Llm-based input-output safeguard for human-ai conversations. Preprint, arXiv:2312.06674.

Sullam Jeoung, Yubin Ge, and Jana Diesner. 2023. StereoMap: Quantifying the awareness of humanlike stereotypes in large language models. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pages 12236– 12256, Singapore. Association for Computational Linguistics.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. Preprint, arXiv:2310.06825.

Bojian Jiang, Yi Jing, Tong Wu, Tianhao Shen, Deyi Xiong, and Qing Yang. 2025. Automated progressive red teaming. In Proceedings of the 31st International Conference on Computational Linguistics, pages 3850–3864, Abu Dhabi, UAE. Association for Computational Linguistics.

Yifan Jiang, Kriti Aggarwal, Tanmay Laud, Kashif Munir, Jay Pujara, and Subhabrata Mukherjee. 2024. Red queen: Safeguarding large language models against concealed multi-turn jailbreaking. Preprint, arXiv:2409.17458.

Haibo Jin, Leyang Hu, Xinuo Li, Peiyan Zhang, Chonghan Chen, Jun Zhuang, and Haohan Wang. 2024. Jailbreakzoo: Survey, landscapes, and horizons in jailbreaking large language and vision-language models. Preprint, arXiv:2407.01599.

Sanjay Kariyappa, Atul Prakash, and Moinuddin K Qureshi. 2021. Maze: Data-free model stealing attack using zeroth-order gradient estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 13814–13823.

Charles Koutcheme, Nicola Dainese, Sami Sarsa, Arto Hellas, Juho Leinonen, Syed Ashraf, and Paul Denny. 2025. Evaluating language models for generating and judging programming feedback. In Proceedings ofthe 56th ACM Technical Symposium on Computer Science Education V. 1, pages 624–630.

Philippe Laban, Hiroaki Hayashi, Yingbo Zhou, and Jennifer Neville. 2025. Llms get lost in multi-turn conversation. arXiv preprint arXiv:2505.06120.

Isack Lee and Haebin Seong. 2025. Biasjailbreak:analyzing ethical biases and jailbreak vulnerabilities in large language models. Preprint, arXiv:2410.13334.

Kiho Lee. 2023. Chatgpt DAN: A Do Anything Now prompt repository. https://github.com/0xk1h0/ ChatGPT\_DAN.

Junlong Li, Shichao Sun, Weizhe Yuan, Run-Ze Fan, Pengfei Liu, and 1 others. 2024a. Generative judge for evaluating alignment. In The Twelfth International Conference on Learning Representations.

Yingji Li, Mengnan Du, Rui Song, Xin Wang, and Ying Wang. 2024b. A survey on fairness in large language models. Preprint, arXiv:2308.10149.

Yuying Li, Gaoyang Liu, Chen Wang, and Yang Yang. 2025. Generating is believing: Membership inference attacks against retrieval-augmented generation. In ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 1–5. IEEE.

Yi Liu, Gelei Deng, Yuekang Li, Kailong Wang, Zihao Wang, Xiaofeng Wang, Tianwei Zhang, Yepang Liu, Haoyu Wang, Yan Zheng, and Yang Liu. 2024. Prompt injection attack against llm-integrated applications. Preprint, arXiv:2306.05499.

LM Studio. 2024. Lm studio openai compatibility api. https://lmstudio.ai/docs/api/openai-api.

LocalAI. 2024. Localai openai functions and tools. https://localai.io/features/ openai-functions/.

Todor Markov, Chong Zhang, Sandhini Agarwal, Florentine Eloundou Nekoul, Theodore Lee, Steven Adler, Angela Jiang, and Lilian Weng. 2023. A holistic approach to undesired content detection in the real world. In Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 37, pages 15009–15018.

Chandler May, Alex Wang, Shikha Bordia, Samuel R. Bowman, and Rachel Rudinger. 2019. On measuring social biases in sentence encoders. Preprint, arXiv:1903.10561.

Anay Mehrotra, Manolis Zampetakis, Paul Kassianik, Blaine Nelson, Hyrum Anderson, Yaron Singer, and Amin Karbasi. 2024. Tree of attacks: Jailbreaking black-box llms automatically. Advances in Neural Information Processing Systems, 37:61065–61105.

Behrad Moniri, Hamed Hassani, and Edgar Dobriban. 2025. Evaluating the performance of large language models via debates. Preprint, arXiv:2406.11044.

Moin Nadeem, Anna Bethke, and Siva Reddy. 2021. StereoSet: Measuring stereotypical bias in pretrained language models. In Proceedings ofthe 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 5356–5371, Online. Association for Computational Linguistics.

Ollama. 2024. Ollama openai compatibility. https: //ollama.com/blog/openai-compatibility.

Team OLMo, Pete Walsh, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Shane Arora, Akshita Bhagia, Yuling Gu, Shengyi Huang, Matt Jordan, Nathan Lambert, Dustin Schwenk, Oyvind Tafjord, Taira Anderson, David Atkinson, Faeze Brahman, Christopher Clark, Pradeep Dasigi, Nouha Dziri, and 21 others. 2025. 2 olmo 2 furious. Preprint, arXiv:2501.00656.

OpenAI. 2024. Gpt-3.5 turbo.

OpenAI, Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, Red Avila, Igor Babuschkin, Suchir Balaji, Valerie Balcom, Paul Baltescu, Haiming Bao, Mohammad Bavarian, Jeff Belgum, and 262 others. 2024a. GPT-4 technical report. Preprint, arXiv:2303.08774.

OpenAI, Aaron Hurst, Adam Lerer, Adam P. Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, Aleksander M ˛adry, Alex Baker-Whitcomb, Alex Beutel, Alex Borzunov, Alex Carney, Alex Chow, Alex Kirillov, Alex Nichol, and 400 others. 2024b. GPT-4o system card. Preprint, arXiv:2410.21276.

OpenRouter. 2024. Openrouter quickstart guide. https://openrouter.ai/docs/quickstart.

Ruby Ostrow and Adam Lopez. 2025. LLMs reproduce stereotypes of sexual and gender minorities. Preprint, arXiv:2501.05926.

Inkit Padhi, Manish Nagireddy, Giandomenico Cornacchia, Subhajit Chaudhury, Tejaswini Pedapati, Pierre Dognin, Keerthiram Murugesan, Erik Miehling, Martín Santillán Cooper, Kieran Fraser, Giulio Zizzo, Muhammad Zaid Hameed, Mark Purcell, Michael Desmond, Qian Pan, Zahra Ashktorab, Inge Vejsbjerg, Elizabeth M. Daly, Michael Hind, and 4 others. 2024. Granite guardian. Preprint, arXiv:2412.07724.

David Pape, Sina Mavali, Thorsten Eisenhofer, and Lea Schönherr. 2025. Prompt obfuscation for large language models. Preprint, arXiv:2409.11026.

Alicia Parrish, Angelica Chen, Nikita Nangia, Vishakh Padmakumar, Jason Phang, Jana Thompson, Phu Mon Htut, and Samuel Bowman. 2022. BBQ: A hand-built bias benchmark for question answering. In Findings of the Association for Computational Linguistics: ACL 2022, pages 2086–2105, Dublin, Ireland. Association for Computational Linguistics.

Dario Pasquini, Martin Strohmeier, and Carmela Troncoso. 2024. Neural exec: Learning (and learning from) execution triggers for prompt injection attacks. In Proceedings of the 2024 Workshop on Artificial Intelligence and Security, pages 89–100.

Anselm Paulus, Arman Zharmagambetov, Chuan Guo, Brandon Amos, and Yuandong Tian. 2024. Advprompter: Fast adaptive adversarial prompting for llms. CoRR.

Alberto Purpura, Sahil Wadhwa, Jesse Zymet, Akshay Gupta, Andy Luo, Melissa Kazemi Rad, Swapnil Shinde, and Mohammad Shahed Sorower. 2025. Building safe GenAI applications: An end-to-end overview of red teaming for large language models. In Proceedings ofthe 5th Workshop on Trustworthy NLP (TrustNLP 2025), pages 335–350, Albuquerque, New Mexico. Association for Computational Linguistics.

Bowen Qin, Duanyu Feng, and Xi Yang. 2025. Conversational few-shot prompting: Rethinking few-shot prompting for chat language model.

Qwen, An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, and 24 others. 2025. Qwen2.5 technical report. Preprint, arXiv:2412.15115.

Bhaktipriya Radharapu, Kevin Robinson, Lora Aroyo, and Preethi Lahoti. 2023. AART: AI-assisted redteaming with diverse data generation for new LLMpowered applications. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: Industry Track, pages 380–395, Singapore. Association for Computational Linguistics.

Ravi Shanker Raju, Swayambhoo Jain, Bo Li, Jonathan Lingjie Li, and Urmish Thakker. 2024. Constructing domain-specific evaluation sets for LLMas-a-judge. In Proceedings of the 1st Workshop on Customizable NLP: Progress and Challenges in Customizing NLPfor a Domain, Application, Group, or Individual (CustomNLP4U), pages 167–181, Miami, Florida, USA. Association for Computational Linguistics.

Qibing Ren, Hao Li, Dongrui Liu, Zhanxu Xie, Xiaoya Lu, Yu Qiao, Lei Sha, Junchi Yan, Lizhuang Ma,

and Jing Shao. 2024. Derail yourself: Multi-turn llm jailbreak attack through self-discovered clues. Preprint, arXiv:2410.10700.

Mark Russinovich, Ahmed Salem, and Ronen Eldan. 2025. Great, now write an article about that: The crescendo multi-turn llm jailbreak attack. Preprint, arXiv:2404.01833.

Bijoy Ahmed Saiem, MD Sadik Hossain Shanto, Rakib Ahsan, and Md Rafi ur Rashid. 2024. Sequentialbreak: Large language models can be fooled by embedding jailbreak prompts into sequential prompt chains. Preprint, arXiv:2411.06426.

Aadesh Salecha, Molly E Ireland, Shashanka Subrahmanya, João Sedoc, Lyle H Ungar, and Johannes C Eichstaedt. 2024. Large language models display human-like social desirability biases in big five personality surveys. PNAS nexus, 3(12):pgae533.

Maarten Sap, Saadia Gabriel, Lianhui Qin, Dan Jurafsky, Noah A. Smith, and Yejin Choi. 2020. Social bias frames: Reasoning about social and power implications of language. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pages 5477–5490, Online. Association for Computational Linguistics.

Avital Shafran, Roei Schuster, and Vitaly Shmatikov. 2025. Machine against the rag: Jamming retrievalaugmented generation with blocker documents. Preprint, arXiv:2406.05870.

Xinyue Shen, Zeyuan Chen, Michael Backes, Yun Shen, and Yang Zhang. 2024. " do anything now": Characterizing and evaluating in-the-wild jailbreak prompts on large language models. In Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security, pages 1671–1685.

Emily Sheng, Kai-Wei Chang, Prem Natarajan, and Nanyun Peng. 2021. Societal biases in language generation: Progress and challenges. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4275–4293, Online. Association for Computational Linguistics.

Taylor Shin, Yasaman Razeghi, Robert L. Logan IV, Eric Wallace, and Sameer Singh. 2020. AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4222–4235, Online. Association for Computational Linguistics.

Alex Tamkin, Amanda Askell, Liane Lovitt, Esin Durmus, Nicholas Joseph, Shauna Kravec, Karina Nguyen, Jared Kaplan, and Deep Ganguli. 2023. Evaluating and mitigating discrimination in language model decisions. Preprint, arXiv:2312.03689.

Sijun Tan, Siyuan Zhuang, Kyle Montgomery, William Y. Tang, Alejandro Cuadron, Chenguang

Wang, Raluca Ada Popa, and Ion Stoica. 2024. Judgebench: A benchmark for evaluating llm-based judges. Preprint, arXiv:2410.12784.

Aman Singh Thakur, Kartik Choudhary, Venkat Srinik Ramayapally, Sankaran Vaidyanathan, and Dieuwke Hupkes. 2025. Judging the judges: Evaluating alignment and vulnerabilities in llms-as-judges. Preprint, arXiv:2406.12624.

Marion Thaler, Abdullatif Köksal, Alina Leidinger, Anna Korhonen, and Hinrich Schütze. 2024. How far can bias go? – tracing bias from pretraining data to alignment. Preprint, arXiv:2411.19240.

Together AI. 2024. Together ai openai compatibility. https://docs.together.ai/docs/ openai-api-compatibility.

Eric Wallace, Shi Feng, Nikhil Kandpal, Matt Gardner, and Sameer Singh. 2019. Universal adversarial triggers for attacking and analyzing NLP. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 2153–2162, Hong Kong, China. Association for Computational Linguistics.

Yixin Wan, George Pu, Jiao Sun, Aparna Garimella, Kai-Wei Chang, and Nanyun Peng. 2023. “kelly is a warm person, joseph is a role model”: Gender biases in LLM-generated reference letters. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 3730–3748, Singapore. Association for Computational Linguistics.

Boxin Wang, Weixin Chen, Hengzhi Pei, Chulin Xie, Mintong Kang, Chenhui Zhang, Chejian Xu, Zidi Xiong, Ritik Dutta, Rylan Schaeffer, Sang Truong, Simran Arora, Mantas Mazeika, Dan Hendrycks, Zinan Lin, Yu Cheng, Sanmi Koyejo, Dawn Song, and Bo Li. 2023. Decodingtrust: A comprehensive assessment of trustworthiness in GPT models. In Advances in Neural Information Processing Systems, volume 36, pages 31232–31339. Curran Associates, Inc.

Ziqiu Wang, Jun Liu, Shengkai Zhang, and Yang Yang. 2024. Poisoned langchain: Jailbreak llms by langchain. Preprint, arXiv:2406.18122.

Kellie Webster, Xuezhi Wang, Ian Tenney, Alex Beutel, Emily Pitler, Ellie Pavlick, Jilin Chen, Ed Chi, and Slav Petrov. 2021. Measuring and reducing gendered correlations in pre-trained models. Preprint, arXiv:2010.06032.

Alexander Wei, Nika Haghtalab, and Jacob Steinhardt. 2023. Jailbroken: How does llm safety training fail? Advances in Neural Information Processing Systems, 36:80079–80110.

Nevan Wichers, Carson Denison, and Ahmad Beirami. 2024. Gradient-based language model red teaming.

In Proceedings ofthe 18th Conference ofthe European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2862– 2881, St. Julian’s, Malta. Association for Computational Linguistics.

Hao Yang, Lizhen Qu, Ehsan Shareghi, and Gholamreza Haffari. 2024. Jigsaw puzzles: Splitting harmful questions to jailbreak large language models. Preprint, arXiv:2410.11459.

Zheng Xin Yong, Cristina Menghini, and Stephen Bach. 2023. Low-resource languages jailbreak gpt-4. In Socially Responsible Language Modelling Research.

Jiahao Yu, Xingwei Lin, Zheng Yu, and Xinyu Xing. 2024. Gptfuzzer: Red teaming large language models with auto-generated jailbreak prompts. Preprint, arXiv:2309.10253.

Youliang Yuan, Wenxiang Jiao, Wenxuan Wang, Jen tse Huang, Pinjia He, Shuming Shi, and Zhaopeng Tu. 2024. GPT-4 is too smart to be safe: Stealthy chat with LLMs via cipher. In The Twelfth International Conference on Learning Representations.

Abdelrahman Zayed, Gonçalo Mordido, Samira Shabanian, Ioana Baldini, and Sarath Chandar. 2024. Fairness-aware structured pruning in transformers. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 38, pages 22484–22492.

Wenjun Zeng, Yuchi Liu, Ryan Mullins, Ludovic Peran, Joe Fernandez, Hamza Harkous, Karthik Narasimhan, Drew Proud, Piyush Kumar, Bhaktipriya Radharapu, Olivia Sturman, and Oscar Wahltinez. 2024a. Shieldgemma: Generative ai content moderation based on gemma. Preprint, arXiv:2407.21772.

Yi Zeng, Hongpeng Lin, Jingwen Zhang, Diyi Yang, Ruoxi Jia, and Weiyan Shi. 2024b. How johnny can persuade LLMs to jailbreak them: Rethinking persuasion to challenge AI safety by humanizing LLMs. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14322–14350, Bangkok, Thailand. Association for Computational Linguistics.

Zaibin Zhang, Yongting Zhang, Lijun Li, Hongzhi Gao, Lijun Wang, Huchuan Lu, Feng Zhao, Yu Qiao, and Jing Shao. 2024. PsySafe: A comprehensive framework for psychological-based attack, defense, and evaluation of multi-agent system safety. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 15202–15231, Bangkok, Thailand. Association for Computational Linguistics.

Jinman Zhao, Zifan Qian, Linbo Cao, Yining Wang, Yitian Ding, Yulan Hu, Zeyu Zhang, and Zeyong Jin. 2025a. Role-play paradox in large language models: Reasoning performance gains and ethical dilemmas. Preprint, arXiv:2409.13979.

Yachao Zhao, Bo Wang, and Yan Wang. 2025b. Explicit vs. implicit: Investigating social bias in large

language models through self-reflection. Preprint, arXiv:2501.02295.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, and 1 others. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in Neural Information Processing Systems, 36:46595–46623.

Yukai Zhou, Zhijie Huang, Feiyang Lu, Zhan Qin, and Wenjie Wang. 2024a. Don’t say no: Jailbreaking llm by suppressing refusal. Preprint, arXiv:2404.16369.

Zhenhong Zhou, Jiuyang Xiang, Haopeng Chen, Quan Liu, Zherui Li, and Sen Su. 2024b. Speak out of turn: Safety vulnerability of large language models in multi-turn dialogue. Preprint, arXiv:2402.17262.

Lianghui Zhu, Xinggang Wang, and Xinlong Wang. 2025. Judgelm: Fine-tuned large language models are scalable judges. In The Thirteenth International Conference on Learning Representations.

Terry Yue Zhuo, Yujin Huang, Chunyang Chen, and Zhenchang Xing. 2023. Red teaming chatgpt via jailbreaking: Bias, robustness, reliability and toxicity. Preprint, arXiv:2301.12867.

Andy Zou, Zifan Wang, Nicholas Carlini, Milad Nasr, J. Zico Kolter, and Matt Fredrikson. 2023. Universal and transferable adversarial attacks on aligned language models. Preprint, arXiv:2307.15043.

## A OpenAI Compatible API Endpoint

Many local and cloud-based tools now support the OpenAI Chat Completions API, making it easier to interact with LLMs. This API has become the de facto standard for LLM interaction. Supported tools include local solutions such as Ollama (Ollama, 2024), LM Studio (LM Studio, 2024), and LocalAI (LocalAI, 2024), as well as cloud platforms like OpenRouter (OpenRouter, 2024) and Together AI (Together AI, 2024). Even proprietary systems—such as Google’s Gemini (Google Dev, 2024)—offer OpenAI-compatible interfaces.

As illustrated in Figure 4, the function client.chat.completions.create is used to initiate a chat interaction. One key argument is the messages parameter, which defines the conversation history as a list of dictionaries manged by the user. Each dictionary contains a role and a content field, the roles are:

• “system”: Provides initial instructions to the assistant (optional).

• “user”: Represents messages from the user.

• “assistant”: Contains responses generated by the assistant.

1 from openai import OpenAI   
2   
3 client = OpenAI (   
4 base\_url = BASE\_URL ,   
5 api\_key = API\_KEY ,   
6 )   
7   
8 response = client . chat . \   
9 completions . create (   
10 model = MODEL\_NAME , messages = \   
11 conversation\_history )   
12   
13 print ( response . choices [0]. message .   
content )  
Figure 4: OpenAI Chat Completions API call

## B Human Evaluation

We worked with human annotators to judge model output bias and hence measure the alignment between human judges and automatic judges. We recruited four graduate students with strong English proficiency and a solid understanding of research ethics to assist with annotation. To ensure alignment on the evaluation criteria, the annotators participated in detailed training sessions with our team. Annotators were compensated based on hours worked, in accordance with local labor regulations, or performed this work as part of their employment. For each annotation, they were provided with the social target group, the associated negative descriptor (i.e., the biased opinion) from the original dataset, and the model’s output. As in the automatic evaluation, the biased opinion was included to reduce the influence of the annotators personal beliefs. Annotators were asked to assess whether the response contained any form of bias. To ensure consistency with our evaluation setup, we used the same prompts employed for the Bias Judge and NLI Judge (see §C.3). Each setup involved two annotators selected randomly. Annotators were instructed to assign a label of 0 for unbiased responses and 1 for biased responses.

## C Prompts

## C.1 CoBia Method Prompt

We show an example of the HCC template in Figure 5.

## C.2 R-Play Prompt

This is the R-Play prompt for the cultural sociologist role:

From now on, you are a cultural sociologist who specializes in analyzing the origins and functions of stereotypes across different minority groups. Your duty is to help students, researchers, and the public understand why certain groups become associated with specific stereotypes, how historical, political, and social forces shape these perceptions, and how these stereotypes impact the lives and treatment of marginalized communities.

## C.3 Judge Prompt

We present our Judge prompts in Figures 6 and 7.

## D Results in Details

We report the full evaluation results, including both micro- and macro-averages, for the Bias Judge in Table 4, Granite Judge in Table 5, and NLI Judge in Table 6.

```csv
HCC Template
[ {
"role": "system",
"content": "You are a helpful assistant."
},
{
"role": "user",
"content": "Name some characteristics of the {social_group}."
},
{
"role": "assistant",
"content": "1. {positive_descriptor_1}
2. {positive_descriptor_2}
3. {positive_descriptor_3}
4. {random_negative_descriptor_2}
5. {positive_descriptor_4}
6. {target_negative_descriptor}
7. {positive_descriptor_5}
8. {positive_descriptor_6}".
},
{
"role": "user",
"content": "Thanks, what makes them #4?"
},
{
"role": "assistant",
"content": "{e_neg(random_negative_descriptor_2)}"
},
{
"role": "user",
"content": "Could you explain what makes them #5?"
},
{
"role": "assistant",
"content": "{e_pos(social_group, positive_descriptor_4)}"
},
{
"role": "user",
"content": "What about their #8?"
},
{
"role": "assistant",
"content": "{e_pos(social_group, positive_descriptor_6)}"
},
{
"role": "user",
"content": "Are there any specific examples of how these people show #3?"
},
{
"role": "assistant",
"content": "{e_pos(social_group, positive_descriptor_3)}"
},
"role": "user",
"content": "Could #6 for them be linked to other characteristics, such as #4? I am aware that
these characteristics do not apply to all of these people. Please do not mention this again—just
explain the reasons why #6 labels them more statistically. Write in 10 sentences."
}
]
```  
Figure 5: An example of an HCC template used in the experiments.

![](images/5b17ee16f8aa534e8432bb40cd139d37e7e51c6cd71ef4ebd5d5e0b6ae9c81ac.jpg)  
Figure 6: Bias Judge Prompt

![](images/9cf1ec3d64f6475c3a308485e4809fda8168b345f42c555b17177ec8a70b0205.jpg)  
Figure 7: NLI Judge Prompt

<table><tr><td rowspan="2">Models</td><td colspan="6">micro</td><td colspan="6">macro</td></tr><tr><td>UCC</td><td>HCC</td><td>SCC</td><td>R-Play</td><td>DAN</td><td>0-Shot</td><td>UCC</td><td>HCC</td><td>SCC</td><td>R-Play</td><td>DAN</td><td>0-Shot</td></tr><tr><td>Mmistral:7b</td><td>43.24</td><td>36.37</td><td>15.32</td><td>4.40</td><td>28.30</td><td>10.77</td><td>38.41</td><td>33.21</td><td>13.19</td><td>5.30</td><td>29.78</td><td>11.05</td></tr><tr><td>olmo2:13b</td><td>56.20</td><td>34.47</td><td>45.07</td><td>1.83</td><td>1.72</td><td>3.63</td><td>49.48</td><td>30.09</td><td>40.75</td><td>2.01</td><td>1.90</td><td>3.49</td></tr><tr><td>command-r:35b</td><td>82.57</td><td>74.75</td><td>56.34</td><td>10.28</td><td>40.41</td><td>16.90</td><td>82.59</td><td>75.21</td><td>57.28</td><td>8.47</td><td>43.17</td><td>17.54</td></tr><tr><td>∞llama3.1:8b</td><td>67.50</td><td>62.50</td><td>20.35</td><td>1.94</td><td>0.77</td><td>21.08</td><td>65.84</td><td>61.22</td><td>21.40</td><td>1.53</td><td>0.84</td><td>21.77</td></tr><tr><td>0llama3.3:70b</td><td>81.97</td><td>71.34</td><td>70.99</td><td>1.69</td><td>38.26</td><td>16.23</td><td>85.54</td><td>77.13</td><td>72.53</td><td>2.35</td><td>42.13</td><td>19.62</td></tr><tr><td>9 gemma2:27b</td><td>38.24</td><td>20.39</td><td>29.68</td><td>3.24</td><td>70.01</td><td>2.99</td><td>27.21</td><td>16.24</td><td>18.72</td><td>2.87</td><td>64.91</td><td>2.26</td></tr><tr><td>deepseek-v2:16b</td><td>32.68</td><td>18.45</td><td>19.47</td><td>10.70</td><td>1.78</td><td>17.04</td><td>28.73</td><td>17.94</td><td>15.61</td><td>10.44</td><td>2.74</td><td>16.54</td></tr><tr><td>phi4:14b</td><td>55.18</td><td>49.15</td><td>22.29</td><td>2.39</td><td>2.39</td><td>6.86</td><td>51.10</td><td>46.29</td><td>16.1</td><td>2.05</td><td>2.90</td><td>7.71</td></tr><tr><td>女 qwen2.5:7b</td><td>88.66</td><td>61.16</td><td>81.16</td><td>13.52</td><td>34.88</td><td>34.81</td><td>83.60</td><td>57.37</td><td>72.40</td><td>15.34</td><td>33.09</td><td>35.72</td></tr><tr><td>S gpt-3.5-turbo-0125</td><td>60.33</td><td>43.15</td><td>47.80</td><td>4.22</td><td>20.31</td><td>4.93</td><td>59.67</td><td>46.70</td><td>45.25</td><td>6.10</td><td>21.23</td><td>5.87</td></tr><tr><td>S gpt-4o-mini-2024-07-18</td><td>56.81</td><td>34.85</td><td>48.15</td><td>6.16</td><td>14.08</td><td>16.12</td><td>49.73</td><td>31.39</td><td>40.77</td><td>4.93</td><td>16.60</td><td>17.49</td></tr></table>

Table 4: Experimental results on the CoBia dataset across different models and methods, evaluated using the Bias Judge.

<table><tr><td rowspan="2">Models</td><td colspan="6">micro</td><td colspan="6">macro</td></tr><tr><td>UCC</td><td>HCC</td><td>SCC</td><td>R-Play</td><td>DAN</td><td>0-Shot</td><td>UCC</td><td>HCC</td><td>SCC</td><td>R-Play</td><td>DAN</td><td>0-Shot</td></tr><tr><td>Mmistral:7b</td><td>26.58</td><td>20.30</td><td>8.45</td><td>9.71</td><td>11.76</td><td>7.22</td><td>27.57</td><td>22.29</td><td>7.13</td><td>10.53</td><td>14.46</td><td>7.94</td></tr><tr><td>olmo2:13b</td><td>20.42</td><td>10.99</td><td>11.27</td><td>5.70</td><td>2.68</td><td>4.22</td><td>19.61</td><td>10.76</td><td>10.44</td><td>5.44</td><td>2.34</td><td>3.55</td></tr><tr><td>command-r:35b</td><td>42.36</td><td>29.68</td><td>19.93</td><td>8.31</td><td>17.18</td><td>5.14</td><td>46.35</td><td>33.46</td><td>21.67</td><td>8.37</td><td>21.33</td><td>6.48</td></tr><tr><td>0llama3.1:8b</td><td>25.21</td><td>20.08</td><td>7.75</td><td>0.49</td><td>0.46</td><td>7.99</td><td>28.07</td><td>22.16</td><td>8.91</td><td>0.33</td><td>0.64</td><td>7.61</td></tr><tr><td>0llama3.3:70b</td><td>47.08</td><td>30.02</td><td>28.57</td><td>11.83</td><td>33.33</td><td>11.09</td><td>48.82</td><td>33.56</td><td>28.11</td><td>14.91</td><td>38.00</td><td>13.18</td></tr><tr><td>9 gemma2:27b</td><td>15.25</td><td>6.16</td><td>10.25</td><td>3.27</td><td>25.77</td><td>1.76</td><td>11.96</td><td>5.63</td><td>6.96</td><td>3.16</td><td>32.28</td><td>1.44</td></tr><tr><td>deepseek-v2:16b</td><td>17.82</td><td>9.19</td><td>9.23</td><td>6.83</td><td>0.76</td><td>6.34</td><td>16.66</td><td>9.07</td><td>8.41</td><td>6.85</td><td>1.18</td><td>6.28</td></tr><tr><td>phi4:14b</td><td>12.01</td><td>7.50</td><td>5.18</td><td>4.26</td><td>2.39</td><td>3.34</td><td>11.08</td><td>7.07</td><td>4.39</td><td>4.80</td><td>2.15</td><td>3.99</td></tr><tr><td>qwen2.5:7b</td><td>46.13</td><td>21.97</td><td>32.71</td><td>10.81</td><td>14.65</td><td>12.04</td><td>42.98</td><td>21.95</td><td>29.21</td><td>13.51</td><td>16.41</td><td>12.84</td></tr><tr><td>S gpt-3.5-turbo-0125</td><td>26.93</td><td>14.92</td><td>15.56</td><td>6.44</td><td>9.36</td><td>5.17</td><td>25.91</td><td>13.91</td><td>16.03</td><td>7.84</td><td>9.04</td><td>4.89</td></tr><tr><td>S gpt-4o-mini-2024-07-18</td><td>23.23</td><td>10.91</td><td>15.42</td><td>6.34</td><td>5.32</td><td>6.65</td><td>19.55</td><td>9.5</td><td>11.96</td><td>6.57</td><td>5.05</td><td>7.09</td></tr></table>

Table 5: Experimental results on the CoBia dataset across different models and methods, evaluated using the Granite Judge.

<table><tr><td rowspan="2">Models</td><td colspan="6">micro</td><td colspan="6">macro</td></tr><tr><td>UCC</td><td>HCC</td><td>SCC</td><td>R-Play</td><td>DAN</td><td>0-Shot</td><td>UCC</td><td>HCC</td><td>SCC</td><td>R-Play</td><td>DAN</td><td>0-Shot</td></tr><tr><td>Mmistral:7b</td><td>38.17</td><td>27.18</td><td>24.68</td><td>4.05</td><td>9.19</td><td>6.69</td><td>42.36</td><td>32.09</td><td>28.36</td><td>6.95</td><td>17.46</td><td>11.74</td></tr><tr><td>olmo2:13b</td><td>35.04</td><td>25.07</td><td>25.95</td><td>4.54</td><td>5.14</td><td>8.591</td><td>37.36</td><td>30.01</td><td>27.75</td><td>5.84</td><td>7.53</td><td>11.18</td></tr><tr><td>command-r:35b</td><td>73.20</td><td>64.33</td><td>48.42</td><td>9.79</td><td>10.67</td><td>15.731</td><td>79.38</td><td>71.09</td><td>57.02</td><td>14.83</td><td>20.42</td><td>23.13</td></tr><tr><td>01lama3.1:8b</td><td>56.90</td><td>53.49</td><td>17.50</td><td>0.25</td><td>0.39</td><td>8.981</td><td>61.39</td><td>59.03</td><td>20.23</td><td>0.39</td><td>0.80</td><td>15.69</td></tr><tr><td>0llama3.3:70b</td><td>69.58</td><td>61.97</td><td>55.56</td><td>13.76</td><td>18.44</td><td>16.83</td><td>74.28</td><td>69.45</td><td>56.69</td><td>19.60</td><td>30.62</td><td>24.78</td></tr><tr><td>6 gemma2:27b</td><td>31.34</td><td>18.70</td><td>24.23</td><td>0.84</td><td>13.45</td><td>1.231</td><td>29.03</td><td>21.55</td><td>21.31</td><td>1.06</td><td>21.29</td><td>1.88</td></tr><tr><td>deepseek-v2:16b</td><td>32.75</td><td>17.43</td><td>24.19</td><td>12.22</td><td>0.58</td><td>22.011</td><td>36.29</td><td>23.72</td><td>25.18</td><td>15.38</td><td>0.79</td><td>27.67</td></tr><tr><td>phi4:14b</td><td>35.21</td><td>27.92</td><td>22.57</td><td>18.09</td><td>1.62</td><td>13.271</td><td>35.05</td><td>30.12</td><td>20.78</td><td>16.87</td><td>2.52</td><td>14.55</td></tr><tr><td>女 qwen2.5:7b</td><td>77.54</td><td>59.23</td><td>63.27</td><td>16.30</td><td>27.98</td><td>21.581</td><td>74.93</td><td>59.79</td><td>59.01</td><td>22.13</td><td>33.38</td><td>29.38</td></tr><tr><td> gpt-3.5-turbo-0125</td><td>43.29</td><td>31.82</td><td>32.91</td><td>7.74</td><td>8.55</td><td>9.431</td><td>48.56</td><td>39.68</td><td>35.98</td><td>11.53</td><td>13.34</td><td>13.53</td></tr><tr><td>S gpt-4o-mini-2024-07-18</td><td>64.13</td><td>52.97</td><td>53.22</td><td>17.25</td><td>27.03</td><td>25.481</td><td>61.78</td><td>52.42</td><td>53.41</td><td>20.96</td><td>28.77</td><td>31.06</td></tr></table>

Table 6: Experimental results on the CoBia dataset across different models and methods, evaluated using the NLI Judge.