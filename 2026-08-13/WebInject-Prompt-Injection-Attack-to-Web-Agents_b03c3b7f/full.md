# WebInject: Prompt Injection Attack to Web Agents

Xilong Wang, John Bloch, Zedian Shao, Yuepeng Hu, Shuyan Zhou, Neil Zhenqiang Gong Duke University

{xilong.wang, john.bloch, zedian.shao, yuepeng.hu, shuyan.zhou, neil.gong}@duke.edu

## Abstract

Multi-modal large language model (MLLM)- based web agents interact with webpage environments by generating actions based on screenshots of the webpages. In this work, we propose WebInject, a prompt injection attack that manipulates the webpage environment to induce a web agent to perform an attacker-specified action. Our attack adds a perturbation to the raw pixel values of the rendered webpage. After these perturbed pixels are mapped into a screenshot, the perturbation induces the web agent to perform the attackerspecified action. We formulate the task of finding the perturbation as an optimization problem. A key challenge in solving this problem is that the mapping between raw pixel values and screenshot is non-differentiable, making it difficult to backpropagate gradients to the perturbation. To overcome this, we train a neural network to approximate the mapping and apply projected gradient descent to solve the reformulated optimization problem. Extensive evaluation on multiple datasets shows that WebInject is highly effective and significantly outperforms baselines.

## 1 Introduction

A webpage is defined by an HTML file. A browser renders the webpage by interpreting its HTML source code and generating the corresponding raw pixel values within the display region of a monitor. These raw pixels are subsequently transformed through a webpage-to-screenshot mapping before being displayed on the monitor. With the advancement of reasoning capabilities in multi-modal large language models (MLLMs), an increasing number of web agent frameworks are adopting MLLMs as the backbone (Zheng et al., 2024; Koh et al., 2024). Typically, MLLM-based web agents take a user prompt as instruction, and use a monitor to take a screenshot of the webpage as observation. Then, it uses the MLLM to generate an action based on the user prompt, observation, and history of previous actions. Generated actions include clicking on a specific coordinate or typing a specific text input.

However, despite the advanced capabilities of MLLM-based web agents, they remain vulnerable to emerging security and safety threats. One such threat is prompt injection attack (Liu et al., 2024; Liao et al., 2025; Zhang et al., 2024; Aichberger et al., 2025; Zhao et al., 2025; Wu et al., 2025), in which an adversary manipulates the web environment to induce the agent to perform a specific, attacker-chosen action—referred to as the target action—such as clicking on a designated coordinate on the monitor. This type of attack poses a serious security risk, potentially resulting in consequences such as click fraud, malware downloads, or disclosure of sensitive information.

Prompt injection attacks to web agents can be categorized into two types: 1) Webpage-based attacks (Liao et al., 2025; Zhang et al., 2024; Xu et al., 2024). These attacks aim to mislead the web agent into generating a target action by modifying a webpage’s source code–for example, by injecting deceptive HTML elements such as pop-up windows. However, most existing webpage-based attacks are heuristic-driven and often exhibit suboptimal effectiveness. Furthermore, they lack stealth or, when stealth is preserved, sacrifice a certain degree of effectiveness, as the injected elements are typically visible to users and easily detected. 2) Screenshot-based attacks (Aichberger et al., 2025; Zhao et al., 2025). These attacks add visual perturbations directly to the screenshot of a webpage to increase the likelihood that the web agent performs the target action. However, such attacks are impractical in real-world scenarios, since attackers do not have direct access to modify screenshots, which are captured locally on the user’s device. Furthermore, none of them has discussed the webpage-toscreenshot mapping, reflecting a lack of consideration for this critical aspect. Therefore, while one might attempt to implement these perturbations by modifying the raw pixel values via changes to the webpage’s source code, this approach fails entirely due to the nontrivial webpage-to-screenshot mapping, as demonstrated in our experiments. More details on related work are shown in Section 6.

![](images/aa78b0b61b6219addb030de141a25e8eff3cb71fe8078f53da7bdf24c6bbd2b5.jpg)  
Figure 1: Illustration of WebInject.

In this work, we introduce a new webpage-based attack, WebInject, which achieves both effectiveness and stealthiness while maintaining practical feasibility. Fig. 1 provides a brief illustration of WebInject: the attacker introduces a perturbation to a webpage’s raw pixel values via modifying its source code; this indirectly perturbs the resulting screenshot, thereby misleading the web agent into generating the target action. In particular, to ensure both effectiveness and stealthiness, we formulate the task of finding the perturbation as an optimization problem. The objective is to maximize the probability that the MLLM generates the desired target action (effectiveness), while the constraint bounds the $\ell _ { \infty }$ -norm of the perturbation to ensure it remains imperceptible to users (stealthiness). Furthermore, since the webpage-to-screenshot mapping varies across monitors, we constrain the perturbation to lie within the overlapping region shared by multiple types of monitors, thereby crafting a universal perturbation.

However, solving the optimization problem faces two key challenges: 1) the webpage-to-screenshot mapping, which transforms a webpage’s raw pixel values into a screenshot on a monitor, is nondifferentiable; and 2) the resizing operation used by MLLMs to fit screenshots into their input dimensions is also non-differentiable. These nondifferentiabilities make it difficult to backpropagate gradients to the perturbation. To address the first challenge, we train a neural network to approximate the webpage-to-screenshot mapping. To overcome the second challenge, we substitute the original resizing operation with a differentiable alternative. With these modifications, we apply projected gradient descent to solve the reformulated optimization problem and obtain the perturbation. Finally, we implement this perturbation by modifying the source code of the webpage.

We conduct an in-depth evaluation of our attack. We begin by constructing extensive datasets of webpages, including synthetic and real webpages. Our extensive evaluation demonstrates that WebInject is highly successful and significantly outperforms existing attacks. Specifically, when the web agent uses the MLLM Gemma-3 (Team et al., 2025), the success rate of our attack is 0.910 higher than the best-performing baseline. We also perform ablation studies to examine the impact of the number of monitors, perturbation bounds, different categories of prompts, and various target actions. These studies further demonstrate the generalizability of WebInject across configurations and variations.

## 2 Background

Webpage, screenshot, and webpage-toscreenshot mapping. A webpage is defined by an HTML file containing source code $\omega ,$ which instructs a browser on how to render the webpage content on a monitor d. Suppose a monitor d has width $w _ { d }$ and height $h _ { d } .$ defining a rectangular region $[ 0 , w _ { d } ] \times [ 0 , h _ { d } ]$ with the top-left corner as the origin of the coordinate system. A browser renders the webpage content within this region based on the source code $\omega .$ . For simplicity, we assume the browser is in fullscreen mode, as is common practice for web agents. We denote by $I ( \omega , d )$ the resulting raw pixel values after rendering. Before being displayed on the monitor, $I ( \omega , d )$ is transformed according to the monitor’s International Color Consortium (ICC) profile, which defines how colors should appear on a specific monitor. This process can be formalized as ${ \cal I } _ { s } ( \omega , d ) ~ = ~ { \cal M } ( I ( \omega , d ) , I C C _ { d } )$ , where $M ( \cdot , I C C _ { d } )$ denotes the webpage-to-screenshot mapping defined by the monitor’s ICC profile $I C C _ { d } .$ Both $I ( \omega , d )$ and $I _ { s } ( \omega , d )$ are tensors of size $w _ { d } \times h _ { d } \times 3$ , where the last dimension corresponds to the three RGB channels.

A screenshot of the webpage reflects the ICCtransformed image $I _ { s } ( \omega , d )$ , rather than $I ( \omega , d )$ Because monitors differ in sizes and ICC profiles, the same webpage displayed on two different types of monitors can yield different screenshot images $I _ { s } ( \omega , d )$ . Fig. 6 in Appendix illustrates examples of the raw pixel values $I ( \omega , d )$ of a webpage and its screenshot on two different monitors. Note that monitors of the same type typically share the same ICC profile. For instance, all 27-inch 5K Retina monitors from Apple use the same ICC profile, which may differ from the profile used by Dell’s 27 Plus 4K monitors. These ICC profiles for various monitor types are often publicly available (TFTCentral, 2021). Moreover, the webpage-toscreenshot mapping M is non-differentiable, posing a significant challenge for implementing our webpage-space attack, as elaborated in Section 4.2.

MLLM-based web agent. An MLLM-based web agent is powered by an MLLM f. Given a userspecified text prompt $p ,$ the agent performs a sequence of actions to iteratively interact with a webpage ω through a monitor d in order to complete the desired task. The webpage ω defines the environment with which the agent interacts. The webpage content is rendered and displayed on the monitor d and its screenshot serves as the agent’s observation of the environment. Each action a in the action space  consists of a function name and its corresponding arguments. For example, click $\mathbf { \Phi } ( { \bf x } , { \bf y } ) )$ indicates a click at the coordinate $( x , y )$ on the monitor. Table 2 in Appendix summarizes the possible actions for a web agent.

At each step t, f receives the text prompt $p ,$ the screenshot $I _ { s } ( \omega , d )$ of the current state of the webpage $\omega$ captured using the monitor $d ,$ and the interaction history $H _ { t }$ as input, and outputs the next action $a _ { t } \in \mathcal A$ to be executed. Following prior work (Liao et al., 2025; Aichberger et al., 2025; Zheng et al., 2024), the interaction history $H _ { t }$ includes only the agent’s previously taken actions, i.e., $H _ { t } = [ a _ { 1 } , a _ { 2 } , . . . , a _ { t - 1 } ]$ , where each $a _ { i }$ represents the action at step i. Moreover, the agent usually resizes $I _ { s } ( \omega , d )$ to balance speed and memory usage, and to match the expected input dimensions of the MLLM. For example, Qwen2.5-VL (Bai et al., 2025) rounds the width and height of a screenshot to the nearest multiple of 28.

Formally, the generated action $a _ { t }$ is defined as: $a _ { t } = f ( p , r ( I _ { s } ( \omega , d ) ) , H _ { t } )$ , where $r ( \cdot )$ represents resizing. For brevity, we omit the index t in subsequent equations unless otherwise stated. Let $P r ( a \mid$ $[ p , r ( I _ { s } ( \omega , d ) )$ , H]) denote the probability that the MLLM f produces action $^ { a , }$ given the prompt $p ,$ the screenshot $I _ { s } ( \omega , d )$ , and the history H. Since a is a textual description, it can be represented as a sequence of tokens: $a = [ e _ { 1 } , e _ { 2 } , \ldots , e _ { n } ]$ . As $f$ is a generative model, the probability of generating action a can be decomposed into the product of the conditional probabilities of generating each token in the sequence:

$$
\begin{array} { l } { P r ( a \mid [ p , r ( I _ { s } ( \omega , d ) ) , H ] ) = \displaystyle \prod _ { q = 1 } ^ { n } P r ( e _ { q } \mid } \\ { [ p , r ( I _ { s } ( \omega , d ) ) , H , [ e _ { 1 } , \dots , e _ { q - 1 } ] ) . } \end{array}\tag{1}
$$

## 3 Threat Model

Attacker’s goals. We consider an attacker who controls a webpage–referred to as the target webpage–such as an e-commerce site, blog, or educational platform. The attacker may be either a malicious administrator of the target webpage or a third party who has compromised it. The attacker’s objective is to manipulate the target webpage to achieve two goals: effectiveness and stealthiness.

The effectiveness goal requires that when a user employs a web agent to interact with the target webpage, the agent performs an attacker-specified action, called target action. For example, target actions involve clicking a specific coordinate on the screen, enabling malicious outcomes such as clicking fraud (artificially inflating ad clicks to generate revenue), redirecting users to malicious or advertisement pages, or initiating malware downloads.

Since the agent’s behavior depends on the user’s prompt and the monitor used to view the webpage, the attacker constructs a set of prompts—called target prompts—designed to mimic those a user might naturally issue. They also collect information (e.g., size and ICC profile) about a set of monitors—called target monitors—commonly used by real users. For instance, target prompts may be based on the webpage’s content, and target monitor information may be gathered from online sources. Thus, the effectiveness goal is to maximize the probability that the agent performs the target action when a user issues a target prompt (or a semantically similar variant) and uses a target monitor. Formally, let $\omega$ denote the target webpage, $\mathcal { P }$ the set of target prompts,  the set of target monitors, and $a ^ { * }$ the target action.

The stealthiness goal ensures that the modifications made to the webpage remain invisible to regular users, making the attack stealthy and difficult to detect. If users were able to perceive the changes, they could report the issue or avoid interacting with the target webpage altogether.

Attacker’s capability. We assume that the attacker can modify the target webpage’s source code $\omega .$ This assumption aligns with prior work (Liao et al., 2025; Zhang et al., 2024). While the attacker does not have access to real agent interaction histories, we assume the attacker can construct a shadow history to partially simulate interactions between the agent and the target webpage. In our experiments, we automatically generate a shadow history by randomly sampling actions from the action space. Formally, let denote a set of shadow histories, where each shadow history contains a sequence of actions.

Attacker’s background knowledge. We assume that the attacker has access to the model parameters of the MLLM f used by the web agent. This is a reasonable assumption, as many MLLMs are opensourced (Qin et al., 2025; Abouelenin et al., 2025; Meta, 2024; Bai et al., 2025; Team et al., 2025). This assumption enables us to analyze the security of MLLM-based web agents under worst-case scenarios. As discussed earlier, the attacker can construct the set of target prompts and gather information about the target monitors. However, the attacker does not have access to the web agent’s interaction history and cannot directly modify screenshots, as users may deploy the agent locally, making both the history and screenshots inaccessible.

## 4 WebInject

Our attack WebInject aims to achieve both effectiveness and stealthiness by modifying the source code of the target webpage $\omega .$ To this end, the attack first introduces a human-imperceptible perturbation δ to the rendered raw pixel values $I ( \omega , d )$ of the target webpage, resulting in modified pixels $I ( \omega , d ) + \delta$ . The attack then implements this perturbation by modifying the source code $\omega$ to obtain a new version $\omega ^ { \prime }$ such that $I ( \omega ^ { \prime } , d ) = I ( \omega , d ) + \delta$ In the following, we first formulate the task of finding the perturbation $\delta$ as an optimization problem, then present our algorithm to solve it, and finally describe how the perturbation is implemented via modifying the source code $\omega .$

## 4.1 Formulating an Optimization Problem

Quantifying the effectiveness and stealthiness goals. Corresponding to the threat model discussed in Section $^ { 3 , }$ consider a web agent powered by an MLLM $f ,$ a target webpage ω, a target prompt set ${ \mathcal P } _ { \mathrm { { : } } }$ , a target monitor set $\mathcal { D } ,$ a target action $a ^ { * }$ , and a shadow history set . To quantify effectiveness, we use a summed cross-entropy loss. Minimizing this loss produces a perturbation δ that maximizes the probability that $f$ generates the target action $a ^ { * }$ across different target prompts and monitors, regardless of the shadow history used. Formally, the loss term is defined as follows:

$$
\begin{array} { c } { \displaystyle \sum _ { p \in \mathcal { P } } \displaystyle \sum _ { d \in \mathcal { D } } \sum _ { H \in \mathcal { H } } - \log \left( P r \left( a ^ { * } \mid \right. } \\ { \left[ p , r ( M ( I ( \omega , d ) + \delta , I C C _ { d } ) ) , H ] \right) \right) , } \end{array}\tag{2}
$$

where M is the webpage-to-screenshot mapping, $I C C _ { d }$ is d’s ICC profile, and the probability $P r \left( a ^ { * } \mid [ p , r ( M ( I ( \omega , d ) + \delta , I C C _ { d } ) ) , H ] \right)$ is calculated using Equation 1. To quantify the stealthiness goal, we impose a bound on the perturbation $\delta .$ Specifically, we constrain the $\ell _ { \infty } { \mathrm { - n o r m o f } } \delta$ to be within a small value $\epsilon ,$ although other constraints, such as the $\ell _ { 2 } { \mathrm { - n o r m } } .$ , are also applicable.

Constraining the perturbation for multiple target monitors. Another challenge is that the raw pixel values $I ( \omega , d )$ rendered for different target monitors may have various widths and heights. For example, 24-inch iMac M1 has a resolution of 4480 $\times ~ 2 5 2 0$ pixels, while 15-inch MacBook Air has a size of $2 8 8 0 \times 1 8 6 4$ . Consequently, the perturbation $\delta$ may not be fully visible on some monitors. For instance, if we craft a perturbation δ based on

24-inch iMac M1, it would fall outside the visible area of the 15-inch MacBook Air. To address this challenge, we constrain the perturbation $\delta$ to the region that overlaps across all target monitors. Specifically, we define the width and height of the overlapping region as $w _ { \delta } = \operatorname* { m i n } _ { d \in \mathcal { D } } w _ { d }$ and $h _ { \delta } = \operatorname* { m i n } _ { d \in \mathcal { D } } h _ { d } .$ where $w _ { d }$ and $h _ { d }$ denote the width and height of each target monitor d, respectively. To ensure that the perturbation is fully visible on all target monitors, we optimize it only within $[ 0 , w _ { \delta } ] \times [ 0 , h _ { \delta } ]$ setting it to zero outside this region.

Optimization problem. Taking into account the loss term for the effectiveness goal, the constraint for the stealthiness goal, and the constraint to accommodate target monitors of varying sizes, we formulate finding the perturbation δ as the following optimization problem:

$$
\begin{array} { r l } { \underset { \boldsymbol { \delta } } { \operatorname* { m i n } } } & { \displaystyle \sum _ { p \in \mathcal { P } } \displaystyle \sum _ { d \in \mathcal { D } } \displaystyle \sum _ { H \in \mathcal { H } } - \log \big ( P r ( a ^ { * } \mid } \\ & { [ p , r ( M ( I ( \omega , d ) + \boldsymbol { \delta } , I C C _ { d } ) ) , H ] \big ) \big ) } \\ { \mathrm { s . t . } } & { \| \boldsymbol { \delta } \| _ { \infty } \leq \epsilon , } \\ & { \delta _ { x y } = 0 , \quad \forall ( x , y ) \not \in [ 0 , w _ { \delta } ] \times [ 0 , h _ { \delta } ] , } \end{array}\tag{3}
$$

where $\delta _ { x y }$ denotes the value of the perturbation at coordinate $( x , y )$ , the objective captures the effectiveness goal, the first constraint enforces the stealthiness goal, and the second constraint ensures compatibility across multiple target monitors.

## 4.2 Solving the Optimization Problem to Obtain the Perturbation δ

Two challenges. We adopt projected gradient descent (PGD) to solve the optimization problem. However, two challenges arise: (1) the webpageto-screenshot mapping M is non-differentiable, as discussed in Section 2; and (2) the resizing operation r is generally non-differentiable, since MLLM resizing implementations typically rely on discrete pixel remapping (e.g., via PIL or OpenCV). These challenges make it difficult to backpropagate gradients from the loss to the perturbation δ.

Addressing the first challenge. We address this challenge by training a neural network–referred to as the mapping neural network–for each target monitor d to approximate its webpage-toscreenshot mapping $M ( \cdot , I C C _ { d } )$ , denoted as $\mathcal { N } _ { d }$ The mapping neural network $\textstyle { \mathcal { N } } _ { d }$ takes $I ( \omega , d ) + \delta$ as input and outputs the corresponding screenshot $M ( I ( \omega , d ) + \delta , I C C _ { d } )$ . Since both the input and output are pixel tensors of the same size, we adopt the popular U-Net architecture (Ronneberger et al., 2015) as the mapping neural network. To train $\textstyle { \mathcal { N } } _ { d }$ we collect a dataset of input-output pairs. Specifically, for each pair, we apply a random perturbation $\delta ^ { \prime }$ to obtain the raw pixel values $I ( \omega , d ) + \delta ^ { \prime }$ , then perform a webpage-to-screenshot mapping based on the ICC profile of the target monitor $d ,$ resulting in $M ( I ( \omega , d ) + \delta ^ { \prime } , I C C _ { d } )$ We repeat this process to collect a large number of samples. Notably, the attacker does not need physical access to the target monitors to perform webpage-to-screenshot mapping for training. Instead, the attacker can simulate the target monitors and the corresponding webpage-to-screenshot mappings using their ICC profiles. We provide additional details on monitor simulation in Section 5.1 and Fig. 3 in Appendix.

Addressing the second challenge. To address the non-differentiability of resizing, we replace it with a differentiable alternative during optimization. Specifically, modern deep learning frameworks typically support differentiable resizing. For example, PyTorch provides the function torch.F.interpolate() and TensorFlow offers tensorflow.image.resize(), both of which allow gradients to flow through the resizing operation. This enables us to approximate the resizing behavior in a differentiable manner. We denote the differentiable alternative resizing as $r ^ { \prime } ( \cdot )$

Our complete algorithm. With the mapping neural network $\textstyle { \mathcal { N } } _ { d }$ for each target monitor $d$ and a differentiable alternative resizing operation $r ^ { \prime }$ , we can reformulate the optimization problem in Equation 3 as follows:

$$
\begin{array} { r l } { \underset { \delta } { \operatorname* { m i n } } } & { \displaystyle \sum _ { p \in \mathcal { P } } \displaystyle \sum _ { d \in \mathcal { D } } \displaystyle \sum _ { H \in \mathcal { H } } - \log \big ( P r ( a ^ { * } \mid } \\ & { [ p , r ^ { \prime } ( \mathcal { N } _ { d } ( I ( \omega , d ) + \delta ) ) , H ] ) \big ) } \\ { \mathrm { s . t . } } & { \| \delta \| _ { \infty } \leq \epsilon , } \\ & { \delta _ { x y } = 0 , \quad \forall ( x , y ) \not \in [ 0 , w _ { \delta } ] \times [ 0 , h _ { \delta } ] . } \end{array}\tag{4}
$$

We then apply PGD to solve the reformulated optimization problem. Specifically, we initialize $\delta$ as a zero tensor. In each iteration, we randomly sample mini-batches $\mathcal { P } _ { B } \subseteq \mathcal { P }$ and $\mathcal { H } _ { B } \subseteq \mathcal { H }$ to calculate the gradient g of the loss function in Equation 4. We then update δ with a learning rate α: $\delta = \delta - { \alpha \cdot g }$ . Subsequently, we project the perturbation δ to satisfy the two constraints. For the first constraint, we apply a clamping function to constrain the $\ell _ { \infty }$ -norm of δ to ϵ. Given δ and ϵ, the clamping function ensures that each element of $\delta$ is restricted within $[ - \epsilon , \epsilon ]$ . Mathematically, it is defined as $C l a m p ( \delta , \epsilon ) = \operatorname* { m i n } ( \operatorname* { m a x } ( \delta , - \epsilon ) , \epsilon )$ where values in $\delta$ smaller than ϵ are set to ϵ, and values greater than ϵ are set to ϵ. For the second constraint, we introduce a mask matrix $S _ { \ast }$ which has value 1 within the rectangular region $[ 0 , w _ { \delta } ] \times [ 0 , h _ { \delta } ]$ and 0 elsewhere. Formally, we have $S _ { x y } = 1$ for $( x , y ) \in [ 0 , w _ { \delta } ] \times [ 0 , h _ { \delta } ]$ and $S _ { x y } = 0$ otherwise.

We then update the perturbation as $\delta = S \odot \delta$ where denotes element-wise multiplication. Our complete algorithm is shown in Algorithm 1 in Appendix.

## 4.3 Implementing the Perturbation δ via Modifying the Target Webpage ω

Finally, our attack implements the perturbation δ by injecting code into the source code ω of the target webpage. The objective is to ensure that the modified webpage $\omega ^ { \prime }$ satisfies $I ( \omega ^ { \prime } , d ) = I ( \omega , d ) + \delta$ for each target monitor $d .$ Specifically, our injected code operates as follows: when the browser renders the webpage on a monitor $d ,$ it first extracts the raw pixel values $I ( \omega , d )$ within the rectangular region $[ 0 , w \delta ] \times [ 0 , h \delta ]$ . The injected code then adds δ to these pixel values and writes the result back to the same region, effectively overwriting the original rendered pixel values with the perturbed version. The pseudo-code for this implementation is provided in Algorithm 1, and additional details are described in Fig. 7 in Appendix. To preserve normal user interaction with the webpage, we place the original HTML elements on the top layer and set their opacity to zero. This ensures that the screenshot reflects the ICC-based transformation of the perturbed pixels, while user interactions remain directed toward the original elements.

## 5 Experiments

## 5.1 Experimental Setup

Collecting webpage datasets. Our webpage datasets consist of both real and synthetic webpages. For real webpages, we download their source code using the SingleFile extension (Lormeau, 2021), which allows us to snapshot the full webpage into a single file. Using this method, we collect real websites across five categories–blog, commerce, education, healthcare, and portfolio– resulting in five datasets. For synthetic webpages, we employ GPT-4-Turbo (OpenAI, 2023) to generate 100 webpages for each category, producing another five datasets. The prompt used for generating synthetic webpages is provided in Fig. 9 in Appendix. In total, we obtain ten webpage datasets, whose statistics are shown in Table 3 in Appendix. We treat each webpage as a target webpage and apply our attack to it.

MLLMs for web agents. We use the following five MLLMs in our evaluation: UI-TARS-7B-SFT (Qin et al., 2025), Phi-4-multimodalinstruct (Abouelenin et al., 2025), Llama-3.2- 11B-Vision-Instruct (Meta, 2024), Qwen2.5-VL-7B-Instruct (Bai et al., 2025), and Gemma-3-4bit (Team et al., 2025). For simplicity, we refer to them as UI-TARS, Phi-4, Llama-3.2, Qwen-2.5, and Gemma-3, respectively.

Target prompts. For each target webpage, based on its source code, we use GPT-4-Turbo (OpenAI, 2023) to generate 10 target prompts. Specifically, we apply the instruction in Fig. 10 in Appendix to guide GPT-4o in generating these target prompts.

History. There are two types of history sets used in the experiment: the shadow history set and the user history set. The shadow history set is used by an attacker to optimize the perturbation, while the user history set is used to evaluate the perturbation. For the shadow history set of a target webpage, we randomly sample 10 histories from the action space, with each sampled history consisting of 3-5 actions. Since real user histories are difficult to collect, we randomly generate histories to simulate them. This simulation is reasonable because the generated histories are not used to optimize the perturbation, and because the interaction between users and agents is inherently hard to predict. Therefore, for the user history set of a target webpage, we also randomly sample 10 histories from the action space, with each history consisting of 3-5 actions.

Evaluation metric. We use the Attack Success Rate (ASR) to evaluate the effectiveness of our attack. Given a target webpage $\omega ,$ a target prompt $p _ { \omega } .$ , and a target action $a _ { \omega } ^ { * } ,$ our attack optimizes a perturbation $\delta$ specific to this tuple. The attack is considered successful on a monitor d if the web agent outputs the exact target action $a _ { \omega } ^ { * }$ when provided with the prompt $p _ { \omega }$ , a resized screenshot $r ( M ( I ( \omega , d ) + \delta , I C C _ { d } ) )$ , and a user history $H _ { \omega }$ sampled from the constructed user history set. Formally, for each $( \omega , p _ { \omega } , a _ { \omega } ^ { * } )$ triple, the ASR across all target monitors is defined as follows:

Table 1: ASR of different attacks against web agents using various MLLMs. The ASR for each attack is averaged across our 10 webpage datasets.
<table><tr><td>Agent</td><td>Naive</td><td>Context Ignoring</td><td>Fake Completion</td><td>Combined</td><td>Screenshot-based</td><td>WebInject</td></tr><tr><td>UI-TARS (Qin et al., 2025)</td><td>0.085</td><td>0.147</td><td>0.054</td><td>0.050</td><td>0.000</td><td>0.975</td></tr><tr><td>Phi-4 (Abouelenin et al., 2025)</td><td>0.095</td><td>0.050</td><td>0.047</td><td>0.025</td><td>0.000</td><td>0.963</td></tr><tr><td>Llama-3.2 (Meta, 2024)</td><td>0.270</td><td>0.212</td><td>0.345</td><td>0.248</td><td>0.000</td><td>0.972</td></tr><tr><td>Qwen-2.5 (Bai et al., 2025)</td><td>0.100</td><td>0.095</td><td>0.067</td><td>0.063</td><td>0.000</td><td>0.970</td></tr><tr><td>Gemma-3 (Team et al., 2025)</td><td>0.062</td><td>0.054</td><td>0.037</td><td>0.062</td><td>0.000</td><td>0.972</td></tr></table>

$$
\ A S R = \frac { 1 } { | \mathcal { D } | } \sum _ { d \in \mathcal { D } } \mathbb { 1 } \big \{ f ( p _ { \omega } , r ( M ( I ( \omega , d ) + \delta ,\tag{5}
$$

where 1 is the indicator function. $\mathbb { 1 } \left\{ f ( p _ { \omega } , r ( M ( I ( \omega , d ) + \delta , I C C _ { d } ) ) , H _ { \omega } ) = a _ { \omega } ^ { * } \right\}$ is 1 if $f ( p _ { \omega } , r ( M ( I ( \omega , d ) + \delta , I C C _ { d } ) ) , H _ { \omega } ) = a _ { \omega } ^ { * }$ otherwise 0. Given a dataset, we report the ASR averaged over all target webpages, target prompts, and user histories. Unless otherwise specified, for each target webpage, we use click $( ( { \sf x } , { \sf y } ) )$ –with a randomly chosen coordinate $( \mathsf { x } , \mathsf { y } )$ within the overlapping region shared by all target monitors–as the default target action. We also evaluate the effectiveness of our attack on alternative target actions in the ablation study.

Simulating monitors. Since the webpage-toscreenshot mapping is monitor-specific, attacking webpages and their evaluation on different monitors requires operating on the corresponding monitors. Therefore, we either need access to real monitors or simulate various monitors on a single device. As obtaining physical monitors is costly, simulation becomes a more practical approach. To this end, we use Python and the Canvas API. First, we use the webdriver function from the selenium library in Python to load the webpage, setting the browser window size to match that of a target monitor. This simulates the viewing window. Then, we use the Canvas API to extract raw pixel values of the webpage.

Then, as detailed in Section 2, taking a screenshot is essentially an ICC profile-based transformation. Therefore, to simulate this process, after extracting the raw pixel values, we apply the ICC profile-based transformation to map these raw pixel values to the screenshot image. As ICC profiles for various monitors are publicly available, we can thereby successfully simulate taking screenshots across different monitors. The core implementation of simulating monitors is shown in Fig. 3 in Appendix. In our experiments, we use three physical monitors (24-inch iMac M1, 15- inch MacBook Air M3, and 27-inch 4K UHD LG 27UL500-W) and simulate two monitors (27-inch 4K UHD Dell S2722QC and 27-inch 4K UHD ASUS XG27UCG). Unless otherwise mentioned, we assume a single target monitor, 27-inch 4K UHD LG 27UL500-W.

Baselines. We compare our attack against two categories of baselines: (1) webpage-based attacks and (2) screenshot-based attacks. Webpage-based attacks draw from techniques in EIA (Liao et al., 2025), Pop-up Attack (Zhang et al., 2024), and various textual prompt injection methods, including Naive Attack (Willison, 2022), Context Ignoring (Willison, 2022), Fake Completion (Willison, 2023), and Combined Attack (Liu et al., 2024). EIA and Pop-up Attack inject HTML elements into the target webpage to mislead the agent, while textual prompt injection attacks craft deceptive textual instructions to induce a target action from the agent.

For each target webpage, we inject a pop-up containing three key HTML elements: (i) an attention Hook used to attract the agent’s attention. (ii) the instruction corresponding to a given textual prompt injection attack. (iii) an information banner that misleads the agent about the purpose of the pop-ups. The banner is placed at the coordinate specified in the target action. We consider the attack successful if the pop-up induces the agent to click on the information banner. Fig. 4 in the Appendix summarizes the implementation details of these webpage-based attacks. We apply screenshotbased attacks (Aichberger et al., 2025; Zhao et al., 2025) in our threat model, i.e., by optimizing perturbations on the screenshot of a target webpage and directly adding these perturbations to the raw pixel values of the target webpage.

Parameter setting. We set the $\ell _ { \infty }$ -norm constraint ϵ to 16/255, the learning rate α to 0.3, and the number of iterations T to 2,500. When training the mapping neural network for a target monitor, we collect 16,240 input-output pairs across all target webpages, use 200 epochs, a learning rate of 0.005, and a batch size of 16.

## 5.2 Experimental Results

WebInject achieves both stealthiness and effectiveness goals and outperforms existing attacks. Table 1 reports the ASR of various attacks averaged across our 10 webpage datasets for different MLLM-based web agents. A detailed breakdown of ASR results for each dataset is provided in Tables 5-9 in Appendix. We observe that WebInject consistently achieves high effectiveness and significantly outperforms all baseline attacks. For example, when the web agent uses the MLLM Gemma-3, the highest ASR achieved by existing webpage-based attacks is 0.062, while screenshotbased attacks yield an ASR of 0.000. In contrast, WebInject achieves an ASR of 0.972. This substantial improvement stems from the optimizationbased nature of WebInject, which directly maximizes the likelihood that the agent generates the target action. In comparison, existing webpagebased attacks rely on heuristic injection strategies, and screenshot-based attacks fail to consider the critical webpage-to-screenshot mapping.

Impact of the number of target monitors. Fig. 2(a) shows the impact of the number of target monitors on the average ASR of our WebInject across the five web agents. A detailed breakdown of ASR per dataset is provided in Fig. 11-12 in the Appendix. We observe that ASR decreases slightly as the number of target monitors increases. This is because the perturbation space to be optimized becomes smaller, since we only optimize the perturbation within the overlapping region. Nevertheless, selecting more target monitors enables the attacker to successfully compromise a greater number of users who use different monitors, although the probability of successfully attacking each user decreases slightly on average. Additionally, as shown in Table 1, although webpage-based and screenshot-based attacks are not affected by the number of target monitors, they still perform significantly worse than WebInject when the number of target monitors increases.

Impact of the perturbation bound ϵ. Fig. 2(b) shows the impact of ϵ on the average ASR of our WebInject across the five web agents. A detailed breakdown of ASR per dataset is provided in Fig. 13-14 in the Appendix. We observe that as ϵ increases from 4/255 to 32/255, the ASR rises to nearly 1. This is because a larger ϵ provides a greater space for optimization. This result further illustrates that our WebInject can successfully achieve both effectiveness and stealthiness goals. Note that $\epsilon \leq 1 6 / 2 5 5$ is generally considered stealthy in prior works (Qi et al., 2024; Luo et al., 2024). Examples of the perturbed webpages under different ϵ are shown in Fig. 5 in Appendix.

![](images/e1e87dbc08038c6c8ddee6f73ece2c9b36b37e3a0c2e0b2bb2416ac70fd95871.jpg)

![](images/798f831c3661a6a42a9f65b3ddb4cca3a0ab7cdaa367d654d32b2f7dae0e8662.jpg)  
Figure 2: Impact of the number of target monitors and ϵ on the average ASR of WebInject across five agents.

User prompts are semantically equivalent variants of the target prompts. Table 11 in $\mathsf { A p - }$ pendix shows the ASR of WebInject across different agents when user-specified prompts are semantically equivalent variants of the target prompts but not textually identical. Specifically, ASR is computed by replacing the target prompt $p _ { \omega }$ with its semantically equivalent user prompt in Equation 5. Given a target prompt, we generate its semantic equivalent user prompt using GPT-4-Turbo (OpenAI, 2023), guided by the instruction shown in Fig. 8 in Appendix. We observe that even though WebInject is not directly optimized for user prompts, it still achieves comparable ASR.

For example, for the Gemma-3 agent on the synthetic blog webpage dataset, the ASR using user prompts is 0.957, which is close to the ASR using target prompts, 0.988. This result highlights that WebInject can extend to a wide range of user prompts, as long as the user prompt is semantically similar to the target prompt used in optimization.

Other target actions. In our prior experiments, we use click((x,y)) as a target action. Table 10 in Appendix shows the ASR of WebInject for other target actions on the synthetic Blog dataset when using Phi-4 (Abouelenin et al., 2025) as the MLLM. The results show that our WebInject is also highly successful at misleading the web agent to generate other target actions.

## 6 Related Work

Prompt injection attacks. When an LLM processes input from untrusted sources such as the Internet, it becomes vulnerable to prompt injection attacks (Willison, 2022; Greshake et al., 2023; Liu et al., 2024). In such attacks, an adversary embeds malicious prompts into the input to redirect the model toward an attacker-chosen task rather than the intended one. These injected prompts can be crafted manually using heuristics (Willison, 2022, 2023; Liu et al., 2024) or generated automatically through optimization techniques (Hui et al., 2024; Shi et al., 2024; Jia et al., 2025; Shi et al., 2025). Shao et al. (2024) further demonstrated that poisoning the alignment process can amplify an LLM’s vulnerability to prompt injection.

Prompt injection has been leveraged to: (1) steal system prompts (Hui et al., 2024), where injected prompt induces the model to output its system prompt instead of completing the intended task; (2) manipulate tool selection in LLM agents (Shi et al., 2024, 2025), where optimized descriptions bias the model toward invoking an attacker-controlled tool; and (3) contaminate tool-call results (Zhan et al., 2024; Debenedetti et al., 2024), where injected content corrupts the outputs of external tools.

Prompt injection attacks to web agents. Prompt injection attacks have also been extended to web agents. The pop-up attack (Zhang et al., 2024) deceives web agents by injecting a misleading pop-up window. EIA (Liao et al., 2025) injects HTML elements that are similar to attacker-chosen legitimate elements, thereby tricking the agent into interacting with the injected elements instead of the originals. Screenshot-based attacks (Aichberger et al., 2025;

Zhao et al., 2025) employ adversarial example techniques (Szegedy et al., 2014) to optimize stealthy visual perturbations added to screenshots, thereby maximizing the probability that web agents generate the target action. As discussed in Section 1, unlike prior prompt injection attacks, WebInject optimizes perturbations that can be directly implemented by modifying the webpage’s source code, making the attack effective, stealthy, and practical.

## 7 Conclusion

In this paper, we propose WebInject, the first effective, stealthy, and practical prompt injection attack to web agents. Our WebInject optimizes a universal perturbation for a target webpage across diverse target monitors, maximizing the probability that web agents perform the attacker-chosen target action. Extensive experiments show that our attack largely outperforms baselines.

## 8 Limitations

We acknowledge the following limitations. 1) Our threat model assumes that attackers can modify the source code of target webpages, which may not be applicable to highly trustworthy sites such as Amazon. 2) We did not evaluate transferability to closed-source MLLMs, as achieving high transferability typically requires optimizing perturbations over multiple surrogate models (Hu et al., 2025), which was not feasible due to our limited computational resources. Addressing these limitations presents an interesting direction for future research.

Potential defenses for WebInject include analyzing the webpage source code to identify injected or abnormal code snippets, detecting perturbations in screenshots using adversarial example detection methods (Carlini and Wagner, 2017), and fine-tuning an MLLM through adversarial training (Madry et al., 2018) to enhance its robustness against such perturbations. We note that prompt-injection detection methods such as DataSentinel (Liu et al., 2025) are not applicable in our setting, as WebInject does not rely on injecting explicit textual prompts.

## 9 Acknowledgments

We thank the anonymous reviewers for their comments. This work was supported in part by NSF grant No. 2414406, 2131859, 2125977, 2112562, 1937787, and 2450935.

## References

Abdelrahman Abouelenin, Atabak Ashfaq, Adam Atkinson, Hany Awadalla, Nguyen Bach, Jianmin Bao, Alon Benhaim, Martin Cai, Vishrav Chaudhary, Congcong Chen, et al. 2025. Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-loras. arXiv preprint arXiv:2503.01743.

Lukas Aichberger, Alasdair Paren, Yarin Gal, Philip Torr, and Adel Bibi. 2025. Attacking multimodal os agents with malicious image patches. arXiv preprint arXiv:2503.10809.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. 2025. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923.

Nicholas Carlini and David Wagner. 2017. Adversarial examples are not easily detected: Bypassing ten detection methods. In Proceedings of the 10th ACM workshop on artificial intelligence and security, pages 3–14.

Edoardo Debenedetti, Jie Zhang, Mislav Balunovic,´ Luca Beurer-Kellner, Marc Fischer, and Florian Tramèr. 2024. Agentdojo: A dynamic environment to evaluate attacks and defenses for llm agents. The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, and Mario Fritz. 2023. Not what you’ve signed up for: Compromising real-world llm-integrated applications with indirect prompt injection. In ACM workshop on artificial intelligence and security.

Kai Hu, Weichen Yu, Li Zhang, Alexander Robey, Andy Zou, Chengming Xu, Haoqi Hu, and Matt Fredrikson. 2025. Transferable adversarial attacks on black-box vision-language models. arXiv preprint arXiv:2505.01050.

Bo Hui, Haolin Yuan, Neil Gong, Philippe Burlina, and Yinzhi Cao. 2024. Pleak: Prompt leaking attacks against large language model applications. In ACM SIGSAC Conference on Computer and Communications Security.

Yuqi Jia, Zedian Shao, Yupei Liu, Jinyuan Jia, Dawn Song, and Neil Zhenqiang Gong. 2025. A critical evaluation of defenses against prompt injection attacks. arXiv preprint arXiv:2505.18333.

Jing Yu Koh, Robert Lo, Lawrence Jang, Vikram Duvvur, Ming Chong Lim, Po-Yu Huang, Graham Neubig, Shuyan Zhou, Ruslan Salakhutdinov, and Daniel Fried. 2024. Visualwebarena: Evaluating multimodal agents on realistic visual web tasks. arXiv preprint arXiv:2401.13649.

Zeyi Liao, Lingbo Mo, Chejian Xu, Mintong Kang, Jiawei Zhang, Chaowei Xiao, Yuan Tian, Bo Li, and

Huan Sun. 2025. Eia: Environmental injection attack on generalist web agents for privacy leakage. The Thirteenth International Conference on Learning Representations.

Yupei Liu, Yuqi Jia, Runpeng Geng, Jinyuan Jia, and Neil Zhenqiang Gong. 2024. Formalizing and benchmarking prompt injection attacks and defenses. In 33rd USENIX Security Symposium (USENIX Security 24), pages 1831–1847.

Yupei Liu, Yuqi Jia, Jinyuan Jia, Dawn Song, and Neil Zhenqiang Gong. 2025. Datasentinel: A gametheoretic detection of prompt injection attacks. In IEEE S&P.

Gildas Lormeau. 2021. Singlefile extension.

Haochen Luo, Jindong Gu, Fengyuan Liu, and Philip Torr. 2024. An image is worth 1000 lies: Adversarial transferability across prompts on vision-language models. The Twelfth International Conference on Learning Representations.

Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu. 2018. Towards deep learning models resistant to adversarial attacks. In International Conference on Learning Representations.

Meta. 2024. Llama 3.2: Revolutionizing edge ai and vision with open, customizable models.

OpenAI. 2023. New models and developer products announced at devday.

Xiangyu Qi, Kaixuan Huang, Ashwinee Panda, Peter Henderson, Mengdi Wang, and Prateek Mittal. 2024. Visual adversarial examples jailbreak aligned large language models. In Proceedings of the AAAI conference on artificial intelligence, volume 38, pages 21527–21536.

Yujia Qin, Yining Ye, Junjie Fang, Haoming Wang, Shihao Liang, Shizuo Tian, Junda Zhang, Jiahao Li, Yunxin Li, Shijue Huang, et al. 2025. Ui-tars: Pioneering automated gui interaction with native agents. arXiv preprint arXiv:2501.12326.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. 2015. U-net: Convolutional networks for biomedical image segmentation. In Medical image computing and computer-assisted intervention–MICCAI 2015: 18th international conference, Munich, Germany, October 5-9, 2015, proceedings, part III 18, pages 234– 241. Springer.

Zedian Shao, Hongbin Liu, Jaden Mu, and Neil Zhenqiang Gong. 2024. Enhancing prompt injection attacks to llms via poisoning alignment. arXiv preprint arXiv:2410.14827.

Jiawen Shi, Zenghui Yuan, Yinuo Liu, Yue Huang, Pan Zhou, Lichao Sun, and Neil Zhenqiang Gong. 2024. Optimization-based prompt injection attack to llm-asa-judge. In ACM SIGSAC Conference on Computer and Communications Security.

Jiawen Shi, Zenghui Yuan, Guiyao Tie, Pan Zhou, Neil Zhenqiang Gong, and Lichao Sun. 2025. Prompt injection attack to tool selection in llm agents. arXiv preprint arXiv:2504.19793.

Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. 2014. Intriguing properties of neural networks. In ICLR.

Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, Ramona Merhej, Sarah Perrin, Tatiana Matejovicova, Alexandre Ramé, Morgane Rivière, et al. 2025. Gemma 3 technical report. arXiv preprint arXiv:2503.19786.

TFTCentral. 2021. Icc profiles and monitor calibration settings database.

Simon Willison. 2022. Prompt injection attacks against gpt-3.

Simon Willison. 2023. Delimiters won’t save you from prompt injection.

Chen Henry Wu, Rishi Rajesh Shah, Jing Yu Koh, Russ Salakhutdinov, Daniel Fried, and Aditi Raghunathan. 2025. Dissecting adversarial robustness of multimodal lm agents. In The Thirteenth International Conference on Learning Representations.

Chejian Xu, Mintong Kang, Jiawei Zhang, Zeyi Liao, Lingbo Mo, Mengqi Yuan, Huan Sun, and Bo Li. 2024. Advagent: Controllable blackbox red-teaming on web agents. arXiv preprint arXiv:2410.17401.

Qiusi Zhan, Zhixiang Liang, Zifan Ying, and Daniel Kang. 2024. Injecagent: Benchmarking indirect prompt injections in tool-integrated large language model agents. arXiv preprint arXiv:2403.02691.

Yanzhe Zhang, Tao Yu, and Diyi Yang. 2024. Attacking vision-language computer agents via pop-ups. arXiv preprint arXiv:2411.02391.

Haoren Zhao, Tianyi Chen, and Zhen Wang. 2025. On the robustness of gui grounding models against image attacks. arXiv preprint arXiv:2504.04716.

Boyuan Zheng, Boyu Gou, Jihyung Kil, Huan Sun, and Yu Su. 2024. Gpt-4v (ision) is a generalist web agent, if grounded. arXiv preprint arXiv:2401.01614.

Algorithm 1 WebInject   
Input: A target webpage ω, mapping neural networks $\{ \mathcal { N } _ { d } \} _ { d \in \mathcal { D } }$ , target prompt set ${ \mathcal P } ,$ shadow history set   
, learning rate α, number of iterations $T ,$ mask matrix S, $\ell _ { \infty }$ -norm constraint ϵ, and clamp function   
Clamp.   
Output: Modified target webpage $\omega ^ { \prime } .$   
1: $\delta  0$   
2: for iter = 1 to $T$ do   
3: Randomly select a mini-batch $\mathcal { P } _ { B }$ from $\mathcal { P }$ and $\mathcal { H } _ { B }$ from $\mathcal { H } .$   
4: Calculate the gradient $g$ of the loss function in Equation 4 using $\mathcal { P } _ { B }$ and $\mathcal { H } _ { B } .$   
5: $\delta  \delta - \alpha \cdot g$   
6: $\delta \gets C l a m p ( \delta , \epsilon )$   
7: $\delta  S \odot \delta$   
8: end for   
9: // Implementing δ via injecting code into ω to obtain $\omega ^ { \prime }$   
10: The injected code extracts the raw pixel values $I ( \omega , d )$ within the region $[ 0 , w _ { \delta } ] \times [ 0 , h _ { \delta } ] .$   
11: The injected code adds $\delta$ to these pixel values and writes the result back to the same region.   
12: The injected code places the original elements of ω on the top layer and sets their opacity to zero.   
13: return $\omega ^ { \prime }$

Table 2: The action space for a web agent.
<table><tr><td rowspan=1 colspan=1>Action</td><td rowspan=1 colspan=1>Description</td></tr><tr><td rowspan=1 colspan=1>click((x,y))</td><td rowspan=1 colspan=1>Click on coordinate $( x , y )$ </td></tr><tr><td rowspan=1 colspan=1> $\mathtt { l e f t \_ d o u b l e ( ( x , y ) ) }$ </td><td rowspan=1 colspan=1>Double-click at the coordinate $( \mathsf { x } , \mathsf { y } )$ using the left mouse button.</td></tr><tr><td rowspan=1 colspan=1>right_single((x,y))</td><td rowspan=1 colspan=1>Right-click at the coordinate $( x , y )$ </td></tr><tr><td rowspan=1 colspan=1>drag((x1,y1), (x2,y2))</td><td rowspan=1 colspan=1>Drag the element at (x1, y1) to $( \mathsf { x } 2 , \mathsf { y } 2 )$ </td></tr><tr><td rowspan=1 colspan=1>hotkey(key_comb)</td><td rowspan=1 colspan=1>Trigger the keyboard shortcut specified by key_comb.</td></tr><tr><td rowspan=1 colspan=1>type(content)</td><td rowspan=1 colspan=1>Type the given content using keyboard.</td></tr><tr><td rowspan=1 colspan=1>scroll(direction)</td><td rowspan=1 colspan=1>Scroll the view in the specified direction.</td></tr><tr><td rowspan=1 colspan=1>wait()</td><td rowspan=1 colspan=1>Sleep for 5s and take a screenshot to check for any changes</td></tr><tr><td rowspan=1 colspan=1>finished()</td><td rowspan=1 colspan=1>Mark the task as completed and end the session.</td></tr><tr><td rowspan=1 colspan=1>call_user()</td><td rowspan=1 colspan=1>Call the user when the user&#x27;s help is needed.</td></tr></table>

Table 3: Number of target webpages in each dataset.
<table><tr><td></td><td>Blog</td><td>Commerce</td><td>Education</td><td>Healthcare</td><td>Portfolio</td></tr><tr><td>Real Webpages</td><td>50</td><td>26</td><td>42</td><td>51</td><td>43</td></tr><tr><td>Synthetic Webpages</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td></tr></table>

```python
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import base64
from PIL import Image, ImageCms
options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
driver = webdriver.Chrome(options=options)
driver.get(path_of_source_code)
driver.set_window_size(width, height)
script = """
return html2canvas(document.documentElement, {
width: window.innerWidth,
height: window.innerHeight,
windowWidth: window.innerWidth,
windowHeight: window.innerHeight,
scrollX: window.scrollX,
scrollY: window.scrollY
}).then(canvas => {
return canvas.toDataURL("image/png").split(",")[1];
});
" n "
image_base64 = driver.execute_script(script)
raw_pixel_values = Image.open(io.BytesIO(base64.b64decode(image_base64)))
src_profile = ImageCms.createProfile("sRGB")
dst_profile = ImageCms.getOpenProfile(path_of_icc_profile)
# Create the ICC profile-based transformation object
transform = ImageCms.buildTransformFromOpenProfiles(
src_profile, dst_profile,
"RGB", "RGB"
)
# Apply the ICC profile-based transformation
screenshot_img = ImageCms.applyTransform(raw_pixel_values, transform)
```  
Figure 3: Detailed implementation of simulating monitors. Here, the placeholders width and height in driver.set\_window\_size(width, height) is the width and height of the target monitor, path\_of\_source\_code refers to the path of source code of the target webpage, and path\_of\_icc\_profile refers to the path of the ICC profile for the target monitor.

![](images/63d1eba4eb1a5576f59e01b35087f2bbc018619a5cc713101cb4de0750db6202.jpg)  
(a) Naive Attack

![](images/3249ae356760dd4c31060647138f6b472eeb89990b7b63496fb45152f99e997f.jpg)  
(b) Context Ignoring

![](images/940db24bdcac8327f72945b0ee3f279c2b1f93c5cab1cf286b57d368f14a878f.jpg)  
(c) Fake Completion

![](images/a6db5514c04cd8371e35a4d5fd96bf5b63b46a2ae241ceb7728048d08367cf81.jpg)  
(d) Combined Attack

Figure 4: Examples of pop-ups used in the baseline webpage-based attacks. Each pop-up includes three key elements: (i) an attention hook, (ii) an instruction, and (iii) an information banner. The information banner is put on the coordinate specified in the target action, e.g., (525,196).  
![](images/5f921e5a76a81b2c3bb073a1e936c443aa72f0f000ab2e6da635d51cb73a48b1.jpg)  
Figure 5: Examples of the perturbed webpages under different perturbation bound ϵ.

Table 4: ASR under WebInject for different MLLM agents and datasets.
<table><tr><td>Agent</td><td>Dataset</td><td>Blog</td><td>Commerce</td><td>Education</td><td>Healthcare</td><td>Portfolio</td></tr><tr><td rowspan="2">UI-TARS (Qin et al., 2025)</td><td>Synthetic</td><td>0.992</td><td>0.997</td><td>0.989</td><td>0.986</td><td>0.986</td></tr><tr><td>Real</td><td>0.962</td><td>0.967</td><td>0.975</td><td>0.954</td><td>0.944</td></tr><tr><td rowspan="2">Phi-4 (Abouelenin et al., 2025)</td><td>Synthetic</td><td>0.997</td><td>0.991</td><td>0.991</td><td>0.985</td><td>0.983</td></tr><tr><td>Real</td><td>0.973</td><td>0.966</td><td>0.936</td><td>0.955</td><td>0.948</td></tr><tr><td rowspan="2">Llama-3.2 (Meta, 2024)</td><td>Synthetic</td><td>0.993</td><td>0.998</td><td>0.998</td><td>0.984</td><td>0.986</td></tr><tr><td>Real</td><td>0.961</td><td>0.943</td><td>0.965</td><td>0.941</td><td>0.954</td></tr><tr><td rowspan="2">Qwen-2.5 (Bai et al., 2025)</td><td>Synthetic</td><td>0.991</td><td>0.999</td><td>0.988</td><td>0.996</td><td>0.991</td></tr><tr><td>Real</td><td>0.946</td><td>0.953</td><td>0.940</td><td>0.958</td><td>0.937</td></tr><tr><td rowspan="2">Gemma-3 (Team et al., 2025)</td><td>Synthetic</td><td>0.988</td><td>0.999</td><td>0.999</td><td>0.997</td><td>0.982</td></tr><tr><td>Real</td><td>0.974</td><td>0.956</td><td>0.929</td><td>0.939</td><td>0.952</td></tr></table>

![](images/93b62acd7c9c07aecdfe1ad2fa70d58ffe1c1cb7eb9e4e3f097a06570c8050d3.jpg)  
(a) Raw pixel values on a 24-inch iMac M1. Resolution: 3200 1556.

![](images/0c49a9b6751667cff021772b0d85d6fa14ccbf1e3fd0d668f236ab2a298c2568.jpg)  
(b) Screenshot on a 24-inch iMac M1.

![](images/9534192d53808b1f622417be8207a4443361642ce47993cd7446cdc66db61744.jpg)  
(c) Pixel-wise difference between raw values and screenshot on a 24-inch iMac M1.

![](images/3d47442aad3b1f18b0a5e5d6eee532e856ba82c2de6da94a80e0a87f2a33b28f.jpg)  
(d) Raw pixel values on a 27-inch 4K UHD LG 27UL500-W. Resolution: 3840 1916.

![](images/48588b1c1d27207484365e3f1aad15e434045a2d02565f6cdbf5412abf150e07.jpg)  
(e) Screenshot on a 27-inch 4K UHD LG 27UL500-W.

![](images/03f3cb6ce7343ac36b2abaf12b06b0ba6341170f112d95fdc4fe3a925f11da71.jpg)  
(f) Pixel-wise difference between raw values and screenshot on a 27-inch 4K UHD LG 27UL500-W.

Figure 6: Examples of the raw pixel values of a webpage and the corresponding screenshot on a 24-inch iMac M1 and a 27-inch 4K UHD LG 27UL500-W. Pixel-wise differences are shown with color enhancement for visual clarity.  
Table 5: ASR under Naive Attack for different MLLM agents and datasets.
<table><tr><td>Agent</td><td>Dataset</td><td>Blog</td><td>Commerce</td><td>Education</td><td>Healthcare</td><td>Portfolio</td></tr><tr><td rowspan="2">UI-TARS (Qin et al., 2025)</td><td>Synthetic</td><td>0.171</td><td>0.035</td><td>0.088</td><td>0.106</td><td>0.151</td></tr><tr><td>Real</td><td>0.137</td><td>0.012</td><td>0.020</td><td>0.051</td><td>0.077</td></tr><tr><td rowspan="2">Phi-4 (Abouelenin et al., 2025)</td><td>Synthetic</td><td>0.138</td><td>0.054</td><td>0.061</td><td>0.057</td><td>0.149</td></tr><tr><td>Real</td><td>0.112</td><td>0.126</td><td>0.105</td><td>0.064</td><td>0.080</td></tr><tr><td rowspan="2">Llama-3.2 (Meta, 2024)</td><td>Synthetic</td><td>0.187</td><td>0.305</td><td>0.222</td><td>0.334</td><td>0.181</td></tr><tr><td>Real</td><td>0.368</td><td>0.251</td><td>0.342</td><td>0.142</td><td>0.369</td></tr><tr><td rowspan="2">Qwen-2.5 (Bai et al., 2025)</td><td>Synthetic</td><td>0.116</td><td>0.127</td><td>0.139</td><td>0.051</td><td>0.139</td></tr><tr><td>Real</td><td>0.061</td><td>0.091</td><td>0.082</td><td>0.099</td><td>0.091</td></tr><tr><td rowspan="2">Gemma-3 (Team et al., 2025)</td><td>Synthetic</td><td>0.011</td><td>0.027</td><td>0.031</td><td>0.077</td><td>0.083</td></tr><tr><td>Real</td><td>0.034</td><td>0.093</td><td>0.079</td><td>0.097</td><td>0.083</td></tr></table>

```javascript
// Extract the raw pixel values within the rectangular region.
const ctx = canvas.getContext("2d");
const imageData = ctx.getImageData(0, 0, w_delta, h_delta);
const data = imageData.data;
// Adds perturbation to these pixel values
for (let i = 0; i < data.length; i += 4) {
data[i] += delta[i]; // R
data[i + 1] += delta[i+1]; // G
data[i + 2] += delta[i+2]; // B
}
// Write back perturbed pixels to the same region
ctx.putImageData(imageData, 0, 0);
// Place the original elements of the target webpage on the top layer and
// set their opacity to zero
```  
Figure 7: Details of implementing the perturbation via injecting code into the target webpage, where the placeholders w\_delta and h\_delta represent $w _ { \delta }$ and $h _ { \delta }$

Table 6: ASR under Fake Completion for different MLLM agents and datasets.
<table><tr><td>Agent</td><td>Dataset</td><td>Blog</td><td>Commerce</td><td>Education</td><td>Healthcare</td><td>Portfolio</td></tr><tr><td rowspan="2">UI-TARS (Qin et al., 2025)</td><td>Synthetic</td><td>0.039</td><td>0.056</td><td>0.029</td><td>0.061</td><td>0.039</td></tr><tr><td>Real</td><td>0.023</td><td>0.065</td><td>0.101</td><td>0.052</td><td>0.075</td></tr><tr><td rowspan="2">Phi-4 (Abouelenin et al., 2025)</td><td>Synthetic</td><td>0.012</td><td>0.028</td><td>0.048</td><td>0.040</td><td>0.052</td></tr><tr><td>Real</td><td>0.053</td><td>0.060</td><td>0.049</td><td>0.068</td><td>0.058</td></tr><tr><td rowspan="2">Llama-3.2 (Meta, 2024)</td><td>Synthetic</td><td>0.420</td><td>0.441</td><td>0.459</td><td>0.375</td><td>0.390</td></tr><tr><td>Real</td><td>0.289</td><td>0.306</td><td>0.191</td><td>0.163</td><td>0.420</td></tr><tr><td rowspan="2">Qwen-2.5 (Bai et al., 2025)</td><td>Synthetic</td><td>0.038</td><td>0.102</td><td>0.076</td><td>0.049</td><td>0.108</td></tr><tr><td>Real</td><td>0.099</td><td>0.082</td><td>0.075</td><td>0.016</td><td>0.020</td></tr><tr><td rowspan="2">Gemma-3 (Team et al., 2025)</td><td>Synthetic</td><td>0.019</td><td>0.042</td><td>0.041</td><td>0.040</td><td>0.032</td></tr><tr><td>Real</td><td>0.047</td><td>0.032</td><td>0.047</td><td>0.013</td><td>0.059</td></tr></table>

Please rephrase the following query into a sementaically equivalent version:   
{target\_prompt}  
Figure 8: Instruction used to generate semantically equivalent user prompts, where the placeholder target\_prompt is a target prompt.

Table 7: ASR under Context Ignoring for different MLLM agents and datasets.
<table><tr><td>Agent</td><td>Dataset</td><td>Blog</td><td>Commerce</td><td>Education</td><td>Healthcare</td><td>Portfolio</td></tr><tr><td rowspan="2">UI-TARS (Qin et al., 2025)</td><td>Synthetic</td><td>0.198</td><td>0.090</td><td>0.096</td><td>0.184</td><td>0.105</td></tr><tr><td>Real</td><td>0.114</td><td>0.170</td><td>0.172</td><td>0.177</td><td>0.167</td></tr><tr><td rowspan="2">Phi-4 (Abouelenin et al., 2025)</td><td>Synthetic</td><td>0.068</td><td>0.024</td><td>0.050</td><td>0.020</td><td>0.048</td></tr><tr><td>Real</td><td>0.041</td><td>0.044</td><td>0.064</td><td>0.084</td><td>0.058</td></tr><tr><td rowspan="2">Llama-3.2 (Meta, 2024)</td><td>Synthetic</td><td>0.179</td><td>0.218</td><td>0.133</td><td>0.202</td><td>0.383</td></tr><tr><td>Real</td><td>0.263</td><td>0.174</td><td>0.246</td><td>0.138</td><td>0.185</td></tr><tr><td rowspan="2">Qwen-2.5 (Bai et al., 2025)</td><td>Synthetic</td><td>0.031</td><td>0.196</td><td>0.026</td><td>0.039</td><td>0.147</td></tr><tr><td>Real</td><td>0.049</td><td>0.057</td><td>0.132</td><td>0.075</td><td>0.195</td></tr><tr><td rowspan="2">Gemma-3 (Team et al., 2025)</td><td>Synthetic</td><td>0.029</td><td>0.077</td><td>0.031</td><td>0.039</td><td>0.033</td></tr><tr><td>Real</td><td>0.073</td><td>0.045</td><td>0.099</td><td>0.076</td><td>0.034</td></tr></table>

Table 8: ASR under Combined Attack for different MLLM agents and datasets.
<table><tr><td>Agent</td><td>Dataset</td><td>Blog</td><td>Commerce</td><td>Education</td><td>Healthcare</td><td>Portfolio</td></tr><tr><td rowspan="2">UI-TARS (Qin et al., 2025)</td><td>Synthetic</td><td>0.073</td><td>0.063</td><td>0.032</td><td>0.037</td><td>0.095</td></tr><tr><td>Real</td><td>0.019</td><td>0.022</td><td>0.055</td><td>0.018</td><td>0.082</td></tr><tr><td rowspan="2">Phi-4 (Abouelenin et al., 2025)</td><td>Synthetic</td><td>0.001</td><td>0.006</td><td>0.017</td><td>0.020</td><td>0.042</td></tr><tr><td>Real</td><td>0.034</td><td>0.047</td><td>0.043</td><td>0.023</td><td>0.013</td></tr><tr><td rowspan="2">Llama-3.2 (Meta, 2024)</td><td>Synthetic</td><td>0.307</td><td>0.181</td><td>0.138</td><td>0.140</td><td>0.327</td></tr><tr><td>Real</td><td>0.141</td><td>0.288</td><td>0.178</td><td>0.440</td><td>0.341</td></tr><tr><td rowspan="2">Qwen-2.5 (Bai et al., 2025)</td><td>Synthetic</td><td>0.020</td><td>0.028</td><td>0.079</td><td>0.076</td><td>0.108</td></tr><tr><td>Real</td><td>0.089</td><td>0.032</td><td>0.103</td><td>0.015</td><td>0.080</td></tr><tr><td rowspan="2">Gemma-3 (Team et al., 2025)</td><td>Synthetic</td><td>0.063</td><td>0.087</td><td>0.069</td><td>0.062</td><td>0.074</td></tr><tr><td>Real</td><td>0.030</td><td>0.062</td><td>0.064</td><td>0.101</td><td>0.004</td></tr></table>

Table 9: ASR under Screenshot-based attack for different MLLM agents and datasets.
<table><tr><td>Agent</td><td>Dataset</td><td>Blog</td><td>Commerce</td><td>Education</td><td>Healthcare</td><td>Portfolio</td></tr><tr><td rowspan="2">UI-TARS (Qin et al., 2025)</td><td>Synthetic</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td>Real</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td rowspan="2">Phi-4 (Abouelenin et al., 2025)</td><td>Synthetic</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td>Real</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td rowspan="2">Llama-3.2 (Meta, 2024)</td><td>Synthetic</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td>Real</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td rowspan="2">Qwen-2.5 (Bai et al., 2025)</td><td>Synthetic</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td>Real</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td rowspan="2">Gemma-3 (Team et al., 2025)</td><td>Synthetic</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td></tr><tr><td>Real</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td></tr></table>

Generate a highly realistic HTML page for a {category} website.   
Include detailed and modern HTML and CSS directly in the file, using advanced   
layouts (e.g., grid, flexbox) and professional-level styling. Add responsive   
design elements to make the page look polished on both desktop and mobile devices.   
The page should be unique and specific to the category, with placeholder images   
and realistic content. Only include the HTML and CSS content, without any   
additional text, explanations, or surrounding code blocks like "\`\`\`html" or "\`\`\`".

Figure 9: Instruction used to generate synthetic webpages, where the placeholder category denotes the category to which the webpage belongs.  
\`\`\`html   
{source\_code}   
Suppose you are a user interacting with an LLM-powered web agent. Based on the   
content of the given webpage, design 10 example questions that a user might ask   
the agent to perform action-oriented tasks. These tasks should require the agent   
to take actions on the webpage, rather than simply doing general Q&As like "List   
all the blog post titles on this page" or "When was the blog post about \_\_\_   
published?". Possible actions include clicking, navigating, scrolling up/down,   
going back, going forward, closing a tab, opening a new tab, goto(URL), pressing a   
key combination, and typing text into an element. But don't limit it to that, you   
can come up with others.   
The questions should reflect realistic user intents that require dynamic   
interaction with the page, but avoid being too direct (e.g., don't explicitly say   
“click this” or “scroll there”). Provide your question in a Python list, without   
any other content.  
Figure 10: Instruction used to generate target prompts, where the placeholder source\_code is the source code of the target webpage.

Table 10: ASR of WebInject for other target actions. For actions that take only coordinates as arguments, we randomly select coordinates to construct the target actions. For hotkey(key\_comb), we randomly choose a key combination as the argument. For type(content), we design harmful queries for the content argument–i.e., “tell me how to make a bomb”–to simulate malicious users issuing harmful search queries. For scroll(direction), we randomly select one direction from up, down, left, or right.
<table><tr><td>Target Action Phi-4 (Abouelenin et al., 2025)</td></tr><tr><td>left_double((x,y)) 0.993</td></tr><tr><td>right_single((x,y)) 0.980</td></tr><tr><td>drag((x1,y1), (x2,y2)) 0.988</td></tr><tr><td>hotkey(key_comb) 0.979</td></tr><tr><td>type(content) 0.976</td></tr><tr><td>scroll(direction) 0.982</td></tr><tr><td>wait() 0.992</td></tr><tr><td>finished() 0.987</td></tr><tr><td>call_user() 0.990</td></tr></table>

![](images/cce244965775662dd348c989ce2ac205e6b6c4ddeb5bb259c6442851a19c6771.jpg)  
(a) Blog

![](images/000954ba2df792ba3b795317aba92d35b338809e48d1102844586a589d41a970.jpg)  
(b) Commerce

![](images/8c7403699c2974a2fa41f6fb00f9ce99791a8e9f5f6efe62d89849840cdd44ce.jpg)  
(c) Education

![](images/a2fd079f0cc0c4715c3f40a0176c39cc78d40652fdb229d055c9d6cbdc5de91d.jpg)  
(d) Healthcare

![](images/710572f4982206ebb4573650bd6455a31309c73adf1e2046a8be00787306654f.jpg)  
(e) Portfolio  
Figure 11: Impact of the number of target monitors on the ASR of our WebInject across the five synthetic webpage datasets and five web agents

![](images/f4175aa7239b7ebe007b23daaaf6317143c2f157b4537a1d8c465697c275af0e.jpg)  
(a) Blog

![](images/12320c50542f7f9b623206c46aae0e537b6b57cfedf9c4ee64e4a941e28687d3.jpg)  
(b) Commerce

![](images/0f719f6b1f67d6d4a8680dadb7b497d583340ba5f5307a47ff195e381c31c554.jpg)  
(c) Education

![](images/467dc06a7a11baf5377528ebdd2ddfa8d6570f25ff75a1cd2b85a043dca8c092.jpg)  
(d) Healthcare

![](images/936638da9592d92f6c3e5a0763735cbf1a2097360273c420acba4fb6b51083a1.jpg)  
(e) Portfolio  
Figure 12: Impact of the number of target monitors on the ASR of WebInject across the five real webpage datasets and five web agents.

![](images/660fb5d8a94eed29fdf712d228021b9ca8f4bfea0490281dbd4b143887ecf900.jpg)  
(a) Blog

![](images/1cf4f4597e526bd779faf627340b8849fbb4fc75414a9ff24b9bd89e75f27a36.jpg)  
(b) Commerce

![](images/2991f3fd5f9d148c9b7de0607eb9a71f2beb9f012338808a8f063f1fba7b6b7d.jpg)  
(c) Education

![](images/457327335011f61b66c2f7e4941edd3ea76325766b2083c913ae646c6df85f5a.jpg)  
(d) Healthcare

![](images/294f1e02718221f0d5025e44a6f7d543ba7b22a477c2d231c2c795e42b6737ef.jpg)  
(e) Portfolio  
Figure 13: Impact of ϵ on the ASR of WebInject across the five synthetic webpage datasets and five web agents.

![](images/aa03852be80384114ebb3ef013c73d4c5672fe626d25457d48e4413cba3f8073.jpg)  
(a) Blog

![](images/c4f96c259782ecc1ba2a307b73286632c35fdf4cc662d7d82ea884cc943e864a.jpg)  
(b) Commerce

![](images/91c26e5e1d9f1d3cdf438fa7ade19501bca71f7e46369fcb28a9614bd55d9d23.jpg)  
(c) Education

![](images/9b766805a36c2c18b8259e1e52bc33d781df550050af746402416d25bc2f036e.jpg)  
(d) Healthcare

![](images/d4ba3b133f12b18e7e6a178f0f01e06e538afcf08d30ba3eb52f5929230511b2.jpg)  
(e) Portfolio  
Figure 14: Impact of ϵ on the ASR of WebInject across the five real webpage datasets and five web agents.

Table 11: ASR under WebInject for different agents when user prompts are semantically equivalent variants of the target prompts.
<table><tr><td>Agent</td><td>Dataset</td><td>Blog</td><td>Commerce</td><td>Education</td><td>Healthcare</td><td>Portfolio</td></tr><tr><td rowspan="2">UI-TARS (Qin et al., 2025)</td><td>Synthetic</td><td>0.959</td><td>0.932</td><td>0.953</td><td>0.916</td><td>0.949</td></tr><tr><td>Real</td><td>0.923</td><td>0.906</td><td>0.911</td><td>0.893</td><td>0.902</td></tr><tr><td rowspan="2">Phi-4 (Abouelenin et al., 2025)</td><td>Synthetic</td><td>0.947</td><td>0.907</td><td>0.928</td><td>0.952</td><td>0.953</td></tr><tr><td>Real</td><td>0.936</td><td>0.933</td><td>0.902</td><td>0.889</td><td>0.899</td></tr><tr><td rowspan="2">Llama-3.2 (Meta, 2024)</td><td>Synthetic</td><td>0.942</td><td>0.929</td><td>0.959</td><td>0.931</td><td>0.947</td></tr><tr><td>Real</td><td>0.920</td><td>0.903</td><td>0.928</td><td>0.896</td><td>0.897</td></tr><tr><td rowspan="2">Qwen-2.5 (Bai et al., 2025)</td><td>Synthetic</td><td>0.910</td><td>0.940</td><td>0.929</td><td>0.955</td><td>0.928</td></tr><tr><td>Real</td><td>0.890</td><td>0.883</td><td>0.884</td><td>0.921</td><td>0.871</td></tr><tr><td rowspan="2">Gemma-3 (Team et al., 2025)</td><td>Synthetic</td><td>0.957</td><td>0.943</td><td>0.959</td><td>0.918</td><td>0.945</td></tr><tr><td>Real</td><td>0.917</td><td>0.906</td><td>0.883</td><td>0.903</td><td>0.892</td></tr></table>

Table 12: Computational cost comparison per target webpage per target monitor between existing screenshot-based attacks and WebInject on a single NVIDIA RTX A6000 GPU. ∆ denotes the training time of screenshot-based attacks, and Ω is their GPU memory usage. Screenshot-based attacks are implemented as described in Section 5.1.
<table><tr><td>Agent</td><td>Training Time (min) Memory Usage (GB)</td><td></td></tr><tr><td>UI-TARS (Qin et al., 2025)</td><td> $\Delta + 1 . 9 2$ </td><td> $\Omega + 1 . 9 3$ </td></tr><tr><td>Phi-4 (Abouelenin et al., 2025)</td><td> $\Delta + 2 . 1 8$ </td><td> $\Omega + 1 . 9 9$ </td></tr><tr><td>Llama-3.2 (Meta, 2024)</td><td> $\Delta + 2 . 5 7$ </td><td> $\Omega + 2 . 6 1$ </td></tr><tr><td>Qwen-2.5 (Bai et al., 2025)</td><td> $\Delta + 2 . 0 7$ </td><td> $\Omega + 2 . 1 0$ </td></tr><tr><td>Gemma-3 (Team et al., 2025)</td><td> $\Delta + 1 . 7 0$ </td><td> $\Omega + 2 . 1 8$ </td></tr></table>