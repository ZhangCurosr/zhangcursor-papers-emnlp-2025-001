# AdaptThink: Reasoning Models Can Learn When to Think

Jiajie Zhang, Nianyi Lin, Lei Hou\*, Ling Feng, Juanzi Li Tsinghua University

## Abstract

Recently, large reasoning models have achieved impressive performance on various tasks by em ploying human-like deep thinking. However, the lengthy thinking process substantially increases inference overhead, making efficiency a critical bottleneck. In this work, we first demonstrate that NoThinking, which prompts the reasoning model to skip thinking and directly generate the final solution, is a better choice for relatively simple tasks in terms of both performance and efficiency. Motivated by this, we propose AdaptThink, a novel RL algorithm to teach reasoning models to choose the optimal thinking mode adaptively based on problem difficulty. Specifically, AdaptThink features two core components: (1) a constrained opti mization objective that encourages the model to choose NoThinking while maintaining the overall performance; (2) an importance sampling strategy that balances Thinking and NoThinking samples during on-policy training, thereby enabling cold start and allowing the model to explore and exploit both thinking modes throughout the training process. Our experiments indi cate that AdaptThink significantly reduces the inference costs while further enhancing performance. Notably, on three math datasets, Adapt-Think reduces the average response length of DeepSeek-R1-Distill-Qwen-1.5B by 53% and improves its accuracy by 2.4%, highlighting the promise of adaptive thinking-mode selection for optimizing the balance between reasoning quality and efficiency.

## 1 Introduction

Recent advancements in large reasoning models, such as OpenAI o1(OpenAI, 2024) and DeepSeek-R1 (DeepSeek-AI, 2025), have demonstrated remarkable capabilities in tackling complex tasks. Given a problem, these models first engage in a long chain of thought—also referred to as Thinking—where they iteratively explore different approaches, accompanied by reflection, backtracking, and self-verification. Subsequently, they produce a final solution that contains only the correct steps and the answer to present to the user. While the long-thinking process markedly enhances the model’s reasoning capacities, it also substantially increases inference overhead and latency (Qu et al., 2025; Sui et al., 2025). In particular, for some simple queries where users expect fast, near-instant responses, these models often generate excessive thinking with unnecessarily detailed steps or redundant attempts, resulting in a suboptimal user experience (Chen et al., 2024; Shen et al., 2025).

![](images/9cd2dbb5e360b0b36cfd4e478cf67db3c66afc50acd91e2bb6da3cfe0d31f172.jpg)  
Figure 1: AdaptThink enables models to adaptively select between Thinking or NoThinking mode based on problem difficulty, thereby improving reasoning efficiency while further improving overall performance.

Existing efforts to improve reasoning efficiency primarily focus on reducing the length of model responses, either through incorporating length-based rewards in reinforcement learning (RL) (Arora and Zanette, 2025; Team et al., 2025), finetuning with preference pairs that penalizes longer responses (Chen et al., 2024; Shen et al., 2025; Luo et al., 2025a), or by merging reasoning and non-reasoning models (Wu et al., 2025). Nevertheless, these methods still apply thinking to all instances, regardless of whether thinking itself is necessary for every problem. In this work, we draw inspiration from the recently introduced No-Thinking approach (Ma et al., 2025), which allows reasoning models to skip the thinking process and directly generate the final solution by prompting with a pseudo-thinking process. Specifically, we further simplify the approach by prompting the model with an empty thinking segment (i.e., “<think></think>”). Our pilot study in Section 3 indicates that NoThinking achieves comparable or even better performance than Thinking on relatively simple problems (up to high-school competition level), while significantly reducing token usage; the benefits of Thinking only become pronounced when the problem is difficult enough.

In light of this observation, we are curious: Can the reasoning model learn to select Thinking or No-Thinking mode adaptively based on the difficulty of the input problem, thereby achieving more efficient reasoning without sacrificing or even improving performance? To this end, we propose AdaptThink, a novel RL algorithm to teach reasoning models when to think. Specifically, AdaptThink features two core components: (1) a constrained optimization objective that encourages the model to choose NoThinking while ensuring overall performance does not degrade; (2) an importance sampling strategy that balances Thinking and NoThinking samples during on-policy training, thereby overcoming the challenge of cold start and allowing the model to explore and exploit both thinking modes throughout the whole training process.

Our experiments demonstrate that AdaptThink effectively enables reasoning models to adaptively select the optimal thinking mode based on problem difficulty, leading to substantial reductions in inference cost compared to prior approaches, while consistently enhancing model accuracy. For instance, on GSM8K, MATH500, and AIME2024, AdaptThink reduces the average response length of DeepSeek-R1-Distill-Qwen-1.5B by 50.9%, 63.5%, and 44.7%, and improving its accuracy by 4.1%, 1.4%, and 1.6%, respectively. The remarkable results substantiate the potential of difficultyadaptive thinking-mode selection as a promising paradigm for advancing the trade-off between reasoning performance and efficiency.

In summary, our key contributions are as follows: (1) We simplify the NoThinking approach and demonstrate its advantages over Thinking for simpler tasks in terms of both performance and efficiency; (2) We propose AdaptThink, a novel RL algorithm that empowers reasoning models to adaptively select the optimal thinking mode adaptively based on problem difficulty, thereby substantially reducing inference costs and further improving performance; (3) We conduct extensive experiments to validate the efficacy of AdaptThink.

## 2 Related Work

Large Reasoning Models. Recent frontier large reasoning models (LRMs), such as OpenAI o1 (OpenAI, 2024), DeepSeek-R1 (DeepSeek-AI, 2025), and QwQ (Qwen Team, 2025), have developed the ability to employ human-like deep thinking in problem solving by generating a long chain of thought before arriving at a final solution. Such advanced ability is typically acquired through largescale RL with verified rewards or fine-tuning on distilled reasoning traces. Despite promising performance, the lengthy thinking process introduces substantial inference costs and latency. Consequently, a variety of approaches have been proposed for more efficient reasoning.

Efficient Reasoning for LRMs. Most existing methods to improve the efficiency of LRMs focus on reducing the token usage in model responses. Some methods incorporate length-based rewards into RL to incentivize more concise responses (Arora and Zanette, 2025; Team et al., 2025) or enable precise control over response length (Aggarwal and Welleck, 2025). Other approaches finetune models with length-related preference pairs, which are obtained from best-of-N sampling (Luo et al., 2025a; Shen et al., 2025) or through postprocessing (Chen et al., 2024). Additionally, several works pursue training-free methods to decrease response length, employing techniques such as model merging (Team et al., 2025; Wu et al., 2025) or prompting (Han et al., 2024; Muennighoff et al., 2025; Fu et al., 2025; Xu et al., 2025). Nevertheless, these methods still utilize long thinking for all problems, while the recent NoThinking approach (Ma et al., 2025) allows reasoning models to bypass long thinking and directly output the final solution via prompting, achieving performance comparable to Thinking in low-tokenbudget settings. In this work, we further demonstrate that even with a sufficient token budget, No-Thinking can outperform Thinking on simple problems while using significantly fewer tokens. This observation motivates us to propose AdaptThink to teach reasoning models to adaptively select the optimal thinking mode based on problem difficulty, which is a new direction for efficient reasoning.

![](images/857a8402ec7d7a1d5c4cc2a245fab000923489c928947ce96c97a44d87a38183.jpg)

![](images/e1cd62c4063a9fa592c68351f0c39eec789a3aa070e57f308b77c18203208acf.jpg)

![](images/d985808946991e15df5b2d7bbe26b9813b94b1a292b1b4b4baef297f7e526ab0.jpg)  
Figure 2: Comparison of DeepSeek-R1-Distill-Qwen-7B using Thinking and NoThinking mode across different difficulty levels of MATH500 dataset.

## 3 Motivation

## 3.1 Preliminary

Consider a reasoning model parameterized by θ and denoted by π<sub>θ</sub>. Given a prompt $x \quad =$ $[ x _ { 1 } , \ldots , x _ { n } , < \mathtt { t h i n k } > ]$ , where $[ x _ { 1 } , \ldots , x _ { n } ]$ represents the problem and <think> is the special token to start the thinking process, the model generates a response $y = [ y _ { 1 } , \ldots , y _ { l } , < / \mathrm { t h i n k } > , y _ { l + 2 } , \ldots , y _ { m } ]$ Here, $[ y _ { 1 } , \ldots , y _ { l } ]$ corresponds to the thinking, which is a long chain of thought consisting of constant exploration, reflection, and self-verification. The token </think> marks the end of thinking. The remaining sequence, $\left[ y _ { l + 2 } , \ldots , y _ { m } \right]$ , denotes the final solution, which only includes the correct steps to solve the problem and the final answer. From the perspective of probability theory, the response y is a sample drawn from the conditional probability distribution $\pi _ { \boldsymbol { \theta } } ( \cdot | \boldsymbol { x } )$ . Since y is generated in an auto-regressive way, the conditional probability $\pi _ { \boldsymbol { \theta } } ( y | \boldsymbol { x } )$ can be decomposed as:

$$
\pi _ { \boldsymbol { \theta } } ( y | \boldsymbol { x } ) = \prod _ { t = 1 } ^ { m } \pi _ { \boldsymbol { \theta } } ( y _ { t } | \boldsymbol { x } , y _ { < t } )\tag{1}
$$

## 3.2 NoThinking is Better for Simple Problems

Current reasoning models, such as OpenAI o1 and DeepSeek-R1, apply long thinking across all problems (denoted as Thinking mode). Though enhancing models’ reasoning capabilities, the lengthy thinking process often leads to unnecessary computation overhead, especially for some simple problems that can also be solved by non-reasoning models (e.g., GPT-4o and Qwen-2.5-Instruct) without thinking. Recently, Ma et al. (2025) proposed No-Thinking method, which enables reasoning models to bypass long thinking and directly generate the final solution by prompting with a fake thinking process “Okay, I think I have finished thinking.</think>”, and found it is still effective in low-token-budget settings. In this work, we further simplify NoThinking by providing the models with an empty thinking (i.e., enforcing the first generated token $y _ { 1 } = < / \mathrm { t h i n k } > )$ . Then, we conduct a pilot study to compare Thinking and NoThinking from the perspective of problem difficulty, with a sufficient token budget (16K).

Specifically, we utilize MATH500 (Lightman et al., 2024) dataset for the pilot study since its have categorized problems into five difficulty levels. For each problem, we employ DeepSeek-R1-Distill-Qwen-7B to generate 16 responses using Thinking and NoThinking, respectively. Then we analyze the accuracy, response length, and instance-level pass rate across the five difficulty levels. As illustrated in Figure 2, although the model is trained using longthinking data, NoThinking still achieves accuracy comparable to Thinking on relatively simple problems (Level 1 to 3), and even slightly outperforms Thinking on the easiest Level-1 problems. Meanwhile, the average length of NoThinking responses is significantly shorter than Thinking ones. Additionally, compared to Nothinking, Thinking only improves the instance-level pass rate for less than half of the problems from Level 1 to 4. Overall, these findings indicates that Thinking only brings notable benefits for challenging problems, whereas NoThinking can be a better choice for simpler questions in terms of both accuracy and efficiency. This motivates us to explore efficient reasoning from a new perspective: teaching the reasoning model to adaptively select Thinking or NoThinking mode based on problem difficulty, thereby reducing inference costs while maintaining or even improving the overall performance. To this end, we propose AdaptThink, a novel RL algorithm that teaches reasoning models when to think.

## 4 AdaptThink

Our AdaptThink algorithm consists of two important components: (1) a constrained optimization objective that incentivizes the model to select No-Thinking mode, while ensuring the overall performance does not decrease; (2) an importance sampling strategy that balances Thinking and NoThinking samples during on-policy training, thereby enabling cold start and also allowing the model to explore and exploit both thinking modes throughout the entire training process. We will introduce these two components in detail as follows.

## 4.1 Constrained Optimization Objective

Considering that NoThinking mode offers a significant advantage over Thinking in reasoning efficiency, an ideal selection policy should prefer to choose NoThinking as long as the overall performance is not diminished. In other words, we should maximize the probability of generating NoThinking responses while ensuring the model’s accuracy does not decline.

Formally, consider a reasoning model $\pi _ { \theta }$ and a dataset $\mathcal { D } _ { \ell }$ . Let $\pi _ { \theta _ { \mathrm { r e f } } }$ denote the reference model, which is the initial π and remains unchanged during training. Let $R ( x , y , y ^ { * } )$ be the reward function $( \mathrm { i . e . }$ , accuracy in math solving), where $x , y ,$ and $\hat { y }$ denote the prompt, model response, and golden answer, respectively. The function returns 1 if $\mathsf { y }$ is both correct and properly formatted; otherwise, it returns 0. For simplicity, we omit yˆ and denote the function as $R ( x , y )$ . Let $\mathbb { 1 } \left( y _ { 1 } = < / \mathrm { t h i n k } > \right)$ be the indicator function, which returns 1 if the first token of y is </think> (i.e., y is a NoThinking response), otherwise returns 0. Then our optimization objective can be formulated as:

$$
\begin{array} { r l } & { \operatorname* { m a x } \mathbb { E } _ { x \sim \mathcal { D } , y \sim \pi _ { \theta } ( \cdot | x ) } \mathbb { 1 } ( y _ { 1 } = < / \mathrm { t h i n k } > ) } \\ & { ~ s . t . \mathbb { E } _ { x \sim \mathcal { D } , y \sim \pi _ { \theta } ( \cdot | x ) } R ( x , y ) \geq } \\ & { ~ \mathbb { E } _ { x \sim \mathcal { D } , y ^ { \prime } \sim \pi _ { \theta _ { \mathrm { r e f } } } ( \cdot | x ) } R ( x , y ^ { \prime } ) . } \end{array}\tag{2}
$$

To solve this constrained optimization problem, we incorporate the constraint into the objective as a penalty term, with a penalty weight $\lambda \geq 0$

$$
\begin{array} { r l r } & { \operatorname* { m a x } \mathbb { E } _ { x \sim \mathcal { D } , y \sim \pi _ { \theta } ( \cdot | x ) , y ^ { \prime } \sim \pi _ { \theta _ { \mathrm { r e f } } } ( \cdot | x ) } \mathbb { 1 } \big ( y _ { 1 } = < / \mathrm { t h i n k } > \big ) } & \\ & { \quad \quad \quad + \lambda \big ( R ( x , y ) - R ( x , y ^ { \prime } ) \big ) . } & \end{array}
$$

By dividing the both side by λ, letting $\begin{array} { r } { \delta = \frac { 1 } { \lambda } } \end{array}$ , and reorganizing the terms about $\pi _ { \theta _ { \mathrm { r e f } } }$ , we have:

$$
\begin{array} { r } { \operatorname* { m a x } \mathbb { E } _ { x \sim \mathcal { D } , y \sim \pi _ { \theta } ( \cdot | x ) } \mathbb { 1 } ( y _ { 1 } = < / \mathsf { t h i n k } > ) \cdot \delta } \\ { + R ( x , y ) - \mathbb { E } _ { y ^ { \prime } \sim \pi _ { \theta _ { \mathrm { r e f } } } ( \cdot | x ) } R ( x , y ^ { \prime } ) . } \end{array}\tag{4}
$$

In practice, $\mathbb { E } _ { y ^ { \prime } \sim \pi _ { \theta _ { \mathrm { r e f } } } ( \cdot | x ) } R ( x , y ^ { \prime } )$ can be approximated by pre-sampling before training. Specifically, we sample $K$ responses from $\pi _ { \theta _ { \mathrm { r e f } } } ( \cdot | x )$ for

each $x ,$ , and calculate their mean reward:

$$
\bar { R } _ { \mathrm { r e f } } ( x ) = \frac { 1 } { K } \sum _ { i = 1 } ^ { K } R ( x , y ^ { \prime i } ) , y ^ { \prime i } \sim \pi _ { \theta _ { \mathrm { r e f } } } ( \cdot | x ) .\tag{5}
$$

Then the optimization objective becomes:

$$
\begin{array} { r l } & { \operatorname* { m a x } \mathbb { E } _ { x \sim \mathcal { D } , y \sim \pi _ { \theta } ( \cdot | x ) } \mathbb { 1 } \left( y _ { 1 } = < / \mathrm { t h i n k } > \right) \cdot \delta } \\ & { \quad \quad \quad + R ( x , y ) - \bar { R } _ { \mathrm { r e f } } ( x ) . } \end{array}\tag{6}
$$

Since $\mathbb { 1 } ( y _ { 1 } = < / \mathrm { t h i n k } > )$ and $R ( x , y )$ are not differentiable, we employ policy gradient method to solve this optimization. Specifically, let $\pi _ { \theta _ { \mathrm { o l d } } }$ be a distribution equal to $\pi _ { \theta }$ without gradient update, and define the advantage function: $A ( x , y ) =$ $\mathbb { 1 } \left( y _ { 1 } = < / \mathrm { t h i n k } > \right) \cdot \delta + R ( x , y ) - \bar { R } _ { \mathrm { r e f } } ( x )$ . Then the objective can be converted into a PPO-style (Schulman et al., 2017) loss without KL penalty:

$$
\begin{array} { r l } & { \mathcal { L } ( \theta ) = - \mathbb { E } _ { x \sim \mathcal { D } , y \sim \pi _ { \theta _ { \mathrm { o l d } } } ( \cdot | x ) } \big [ \operatorname* { m i n } \big ( \frac { \pi _ { \theta } ( y | x ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( y | x ) } A ( x , y ) , } \\ & { \quad \quad \mathrm { c l i p } ( \frac { \pi _ { \theta } ( y | x ) } { \pi _ { \theta _ { \mathrm { o l d } } } ( y | x ) } , 1 - \epsilon , 1 + \epsilon ) A ( x , y ) \big ) \big ] . \quad ( 7 ) } \end{array}
$$

Here, clip( ) denotes the clipping function, which improves the stability of training.

## 4.2 Importance Sampling

At each step of optimizing $\mathcal { L } ( \boldsymbol { \theta } )$ using on-policy training, we sample a batch $\mathcal { D } _ { b }$ from the dataset $\mathcal { D } ,$ and then sample K responses $\{ y ^ { i } \} _ { i = 1 } ^ { K }$ from $\pi _ { \theta _ { \mathrm { o l d } } } ( \cdot | x )$ for each $x \in \mathcal { D } _ { b }$ to estimate $\mathcal { L } ( \boldsymbol { \theta } )$ . However, since the initial $\pi _ { \theta }$ naturally apply Thinking across all problems, it is impossible to get Nothinking samples from $\pi _ { \theta _ { \mathrm { o l d } } }$ from the training starts (i.e., $\pi _ { \theta _ { \mathrm { o l d } } } ( y _ { 1 } = < / \mathrm { t h i n k } > | x ) \approx 0 )$ . As a result, the model can only learn from Thinking samples and will never generate NoThinking responses.

To solve this cold-start challenge, we employ the technique of importance sampling. Specifically, we define a new distribution $\pi _ { \mathrm { I S } } ( \cdot | x )$

$$
\pi _ { \mathrm { I S } } ( y _ { t } = a | x , y _ { < t } ) = \left\{ \begin{array} { l l } { 0 . 5 , \mathrm { ~ i f ~ } t = 1 , a = < / \mathrm { t h i n k } > ; } \\ { 0 . 5 , \mathrm { ~ i f ~ } t = 1 , a = w _ { \mathrm { s t a r t } } ; } \\ { \pi _ { \theta _ { \mathrm { o l d } } } ( y _ { t } = a | x , y _ { < t } ) , \mathrm { ~ i f ~ } t > 1 . } \end{array} \right.\tag{8}
$$

Here, $w _ { \mathrm { s t a r t } }$ is a common word to start long thinking, such as $\ddot { } \cdot { A l r i g h t } ^ { \prime \prime }$ . During training, we sample responses $\{ y ^ { i } \} _ { i = 1 } ^ { \bar { K } }$ from $\pi _ { \mathrm { I S } } ( \cdot | x )$ instead of $\pi _ { \theta _ { \mathrm { o l d } } } ( \cdot | x )$ so that half of the samples in a batch are in Thinking mode and the other half are NoThinking. This allows the model to learn from both modes from the beginning of training, and finally adaptively be able to select the appropriate mode. Accordingly, our final loss function of AdaptThink becomes:

Algorithm 1 AdaptThink   
Input: policy model $\pi _ { \boldsymbol { \theta } } ;$ dataset ; hyperparameters $\overline { { K , \delta , \epsilon } }$   
Initialize: reference model $\pi _ { \theta _ { \mathrm { r e f } } }  \pi _ { \theta }$   
1: Sample K responses $\{ y ^ { \prime i } \} _ { i = 1 } ^ { K } \sim \pi _ { \theta _ { \mathrm { r e f } } } ( \cdot | x )$ and calculate ${ \bar { R } } _ { \mathrm { r e f } } ( x )$ for each $x \in \mathcal { D }$ (Equation 5)   
2: for step $\mathbf { \Psi } ) = 1 , \dotsc , M$ do   
3: Update the old policy model $\pi _ { \theta _ { \mathrm { o l d } } }  \pi _ { \theta }$ and importance sampling distribution π<sub>IS</sub> (Equation 8)   
4: Sample a batch $\dot { \mathcal { D } } _ { b }$ from   
5: Sample K responses $\{ y ^ { i } \} _ { i = 1 } ^ { K } \sim \pi _ { \mathrm { I S } } ( \cdot | x )$ for each $x \in \mathcal { D } _ { b }$ and estimate ${ \mathcal { L } } _ { \mathrm { A T } } ( \theta )$ (Equation 9. Half of $y ^ { i }$ are Thinking   
responses and the other half are NoThinking responses.)   
Update the policy model $\pi \theta$ by minimizing ${ \dot { \mathcal { L } } } _ { \mathrm { A T } } ( \theta )$   
7: end for   
Output: π<sub>θ</sub>

$$
\begin{array} { r l r } & { } & { { \mathcal L } _ { \mathrm { A T } } ( \theta ) = - \mathbb E _ { x \sim \mathcal D , y \sim \pi _ { \mathrm { I S } } ( \cdot \vert x ) } \big [ \operatorname* { m i n } \big ( \frac { \pi _ { \theta } ( y \vert x ) } { \pi _ { \mathrm { I S } } ( y \vert x ) } A ( x , y ) , } \\ & { } & { \mathrm { c l i p } ( \frac { \pi _ { \theta } ( y \vert x ) } { \pi _ { \mathrm { I S } } ( y \vert x ) } , 1 - \epsilon , 1 + \epsilon ) A ( x , y ) \big ) \big ] . \qquad ( 9 ) } \end{array}
$$

In addition to enabling cold start, importance sampling also preserves the opportunities for exploration and exploitation across both Thinking and NoThinking modes during the entire training process. This prevents $\pi _ { \theta }$ from collapsing into one thinking mode forever and completely ignoring the other, even if the latter mode may demonstrate a greater advantage in the future. Finally, we summarize our AdaptThink algorithm in Algorithm 1.

## 4.3 A New Perspective to Understand the Loss

In this subsection, we provide another perspective to understand our loss function ${ \mathcal { L } } _ { \mathrm { A T } } ( \theta )$ by comparing the advantage $A ( x , y )$ of Thinking and No-Thinking samples from π<sub>IS</sub>( x). Given a prompt x, we denote the average pass rate of Thinking and NoThinking samples as $\bar { R } _ { \mathrm { t h i n k } } ( x )$ and $\bar { R } _ { \mathrm { n o t h i n k } } ( x )$ respectively. Then their average advantages are:

$$
\begin{array} { r l } & { \bar { A } _ { \mathrm { t h i n k } } ( x ) = \bar { R } _ { \mathrm { t h i n k } } ( x ) - \bar { R } _ { \mathrm { r e f } } ( x ) , } \\ & { \bar { A } _ { \mathrm { n o t h i n k } } ( x ) = \delta + \bar { R } _ { \mathrm { n o t h i n k } } ( x ) - \bar { R } _ { \mathrm { r e f } } ( x ) . } \end{array}\tag{10}
$$

Note that the probability of choosing NoThinking $( \mathrm { i . e . , } \pi _ { \theta } ( y _ { 1 } = < / \mathrm { t h i n k } > | x ) )$ and Thinking (i.e., $\pi _ { \boldsymbol { \theta } } ( y _ { 1 } = w ^ { * } | \boldsymbol { x } ) )$ are competitive. Therefore, when optimizing $\mathcal { L } _ { \mathrm { A T } } ( \theta ) , \pi _ { \theta } ( y _ { 1 } = < / \mathrm { t h i n k } > | x )$ will improve only if $\bar { A } _ { \mathrm { n o t h i n k } } ( x ) > 0$ and $\bar { A } _ { \mathrm { n o t h i n k } } ( x ) >$ $\bar { A } _ { \mathrm { t h i n k } } ( x )$ , which give us:

$$
\begin{array} { r l } & { \bar { R } _ { \mathrm { n o t h i n k } } ( x ) + \delta > \bar { R } _ { \mathrm { r e f } } ( x ) , } \\ & { \bar { R } _ { \mathrm { n o t h i n k } } ( x ) + \delta > \bar { R } _ { \mathrm { t h i n k } } ( x ) . } \end{array}\tag{11}
$$

In other words, only when the problem is simple enough such that the accuracy gap between No-Thinking and Thinking, as well as the reference model, is smaller than δ, ${ \mathcal { L } } _ { \mathrm { A T } } ( \theta )$ will favor No-Thinking and encourage $\pi _ { \theta }$ to directly generate the final solution. For more challenging problems where NoThinking lags far behind the other two, ${ \mathcal { L } } _ { \mathrm { A T } } ( \theta )$ will prioritize performance and guide π<sub>θ</sub> to engage in Thinking more frequently. Therefore, ${ \mathcal { L } } _ { \mathrm { A T } } ( \theta )$ aligns well with our expectation for difficulty-adaptive thinking in Section 3.2.

## 5 Experiments

## 5.1 Setup

Models. We select DeepSeek-R1-Distill-Qwen-1.5B and DeepSeek-R1-Distill-Qwen-7B, two popular reasoning models that demonstrate impressive performance on math problem solving, as the initial policy models.

Dataset and Metrics. The training dataset we use is DeepScaleR (Luo et al., 2025b) dataset, which consists of 40K math problems drawn from AIME 1983-2023, AMC, Omni-Math (Gao et al., 2024), and STILL (Min et al., 2024). For evaluation, we use three math datasets with increasing difficulty: GSM8K (Cobbe et al., 2021) test set (1319 grade school math problems), MATH500 (Lightman et al., 2024) (500 high-school competition math problems), and AIME 2024 (30 Olympiadlevel math problems). For evaluation metrics, we consider both accuracy and response length. We also report the average accuracy variation and the average length reduction rate across all the test datasets. Considering the limited size of AIME 2024, we repeatedly sample 16 responses for each case and report the average results. For all models, we set the evaluation context size to 16K, and set the temperature to 0.6 as suggested in DeepSeek’s model cards.

Implementation Details. We build our code based on VeRL (Sheng et al., 2024) framework. The training context size, batch size, and the learning rate are set to 16K, 128, and 2e-6, respectively. The hyperparameters K, δ, and ϵ in AdaptThink are set to 16, 0.05, and 0.2, respectively. The comparison of using different δ is shown in Section 5.4. We train the models for 1 epoch, which is 314 steps in total. For the 1.5B model, we use one 8 H800 node and cost about 32 hours. For the 7B model, we use four 8 H800 nodes and cost about 28 hours. Finally, we select the checkpoints on 300 and 150 steps for the 1.5B and 7B models, respectively, where the models’ accuracy and response lengths achieve a good balance.

<table><tr><td>Method</td><td></td><td>GSM8K</td><td></td><td>MATH 500</td><td></td><td>AIME 2024</td><td>Average</td></tr><tr><td></td><td>Acc</td><td>Length RatioNT</td><td>Acc</td><td>Length RatioNT</td><td>Acc</td><td>Length RatioNT</td><td>∆Acc ∆Length</td></tr></table>

<table><tr><td> $\mathrm { O r i g i n a l } _ { T h i n k i n g }$ </td><td>79.0</td><td>978</td><td>0.0%</td><td>80.6</td><td>4887</td><td>0.0%</td><td>29.4</td><td>12073</td><td>0.0%</td><td></td><td></td></tr><tr><td> $\mathrm { O r i g i n a l } _ { N o T h i n k i n g }$ </td><td>69.8</td><td>280</td><td>100.0%</td><td>67.2</td><td>658</td><td>100.0%</td><td>14.0</td><td>2190</td><td>100.0%</td><td>-12.7</td><td>-79.9%</td></tr><tr><td> $\mathrm { D P O } _ { S h o r t e s t }$ </td><td>78.3</td><td>804</td><td>0.0%</td><td>82.4</td><td>3708</td><td>0.0%</td><td>30.7</td><td>10794</td><td>0.0%</td><td>+0.8</td><td>-17.5%</td></tr><tr><td>OverThink</td><td>77.2</td><td>709</td><td>0.0%</td><td>81.2</td><td>4131</td><td>0.0%</td><td>28.3</td><td>11269</td><td>0.0%</td><td>-0.8</td><td>-16.5%</td></tr><tr><td>DAST</td><td>77.2</td><td>586</td><td>0.0%</td><td>83.0</td><td>2428</td><td>0.0%</td><td>26.9</td><td>7745</td><td>0.0%</td><td>-0.6</td><td>-42.1%</td></tr><tr><td>O1-Pruner</td><td>74.8</td><td>458</td><td>0.0%</td><td>82.2</td><td>3212</td><td>0.0%</td><td>28.9</td><td>10361</td><td>0.0%</td><td>-1.0</td><td>-33.9%</td></tr><tr><td>TLMRE</td><td>80.7</td><td>863</td><td>0.0%</td><td>85.0</td><td>3007</td><td>0.0%</td><td>29.2</td><td>8982</td><td>0.0%</td><td>+2.0</td><td>-25.3%</td></tr><tr><td>ModelMerging</td><td>79.7</td><td>603</td><td>0.0%</td><td>63</td><td>2723</td><td>0.0%</td><td>18.1</td><td>10337</td><td>0.0%</td><td>-9.4</td><td>-32.3%</td></tr><tr><td>RFTMixThinking</td><td>76</td><td>1077</td><td>8.8%</td><td>72.4</td><td>4341</td><td>33.4%</td><td>25.2</td><td>11157</td><td>21.0%</td><td>-5.1</td><td>-2.9%</td></tr><tr><td>AdaptThink</td><td>83.1</td><td>480</td><td>86.9%</td><td>82.0</td><td>1782</td><td>76.8%</td><td>31.0</td><td>6679</td><td>40.4%</td><td>+2.4</td><td>-53.0%</td></tr><tr><td colspan="10">DeepSeek-R1-Distill-Qwen-7B</td><td></td><td></td></tr><tr><td>OriginalThinking</td><td>87.9</td><td>682</td><td>0.0%</td><td>90.2</td><td>3674</td><td>0.0%</td><td>53.5</td><td>10306</td><td>0.0%</td><td></td><td></td></tr><tr><td> $\mathrm { O r i g i n a l } _ { N o T h i n k i n g }$ </td><td>85.1</td><td>283</td><td>100.0%</td><td>80.6</td><td>697</td><td>100.0%</td><td>24.2</td><td>1929</td><td>100.0%</td><td>-13.9</td><td>-73.6%</td></tr><tr><td> $\mathrm { D P O } _ { S h o r t e s t }$ </td><td>85.7</td><td>402</td><td>0.0%</td><td>91.6</td><td>2499</td><td>0.0%</td><td>52.5</td><td>8699</td><td>0.0%</td><td>-0.6</td><td>-29.5%</td></tr><tr><td>OverThink</td><td>86.3</td><td>426</td><td>0.0%</td><td>89.4</td><td>2435</td><td>0.0%</td><td>53.1</td><td>8744</td><td>0.0%</td><td>-0.9</td><td>-28.8%</td></tr><tr><td>DAST</td><td>86.7</td><td>459</td><td>0.0%</td><td>89.6</td><td>2162</td><td>0.0%</td><td>45.6</td><td>7578</td><td>0.0%</td><td>-3.2</td><td>-33.4%</td></tr><tr><td>01-Pruner</td><td>87.6</td><td>428</td><td>0.0%</td><td>86.6</td><td>2534</td><td>0.0%</td><td>49.2</td><td>9719</td><td>0.0%</td><td>-2.7</td><td>-24.7%</td></tr><tr><td>TLMRE</td><td>88.9</td><td>756</td><td>0.0%</td><td>91.8</td><td>2899</td><td>0.0%</td><td>54.0</td><td>8633</td><td>0.0%</td><td>+1.0</td><td>-8.8%</td></tr><tr><td>ModelMerging</td><td>88.4</td><td>531</td><td>0.0%</td><td>72.6</td><td>2280</td><td>0.0%</td><td>36.9</td><td>8624</td><td>0.0%</td><td>-11.2</td><td>-25.5%</td></tr><tr><td> $\mathrm { R F T } _ { M i x T h i n k i n g }$ </td><td>86.2</td><td>365</td><td>66.5%</td><td>84.8</td><td>2411</td><td>64.8%</td><td>49.4</td><td>9969</td><td>10.0%</td><td>-3.7</td><td>-28.0%</td></tr><tr><td>AdaptThink</td><td>91.0</td><td>309</td><td>99.6%</td><td>92.0</td><td>1875</td><td>76.6%</td><td>55.6</td><td>8599</td><td>6.3%</td><td>+2.3</td><td>-40.1%</td></tr></table>

Table 1: Accuracy (Acc), response length (Length), and the ratio of NoThinking responses (Ratio<sub>NT</sub>) of different methods on three math benchmarks. The best and second results are bolded and underlined, respectively.

![](images/8fc283c2543cc22e17bf8295cdeced3547855c176469a7e6c18e9412579b95b2.jpg)

![](images/270cb36b4a395816f9573f8b17d3ab3b9f930c5cf1cba94e622085ce279493b8.jpg)  
Figure 3: Left: The ratio that AdaptThink-7B choose Thinking or NoThinking across different MATH levels. Right: Comparison of accuracy between AdaptThink-7B and DeepSeek-R1-Distill-Qwen-7B using Thinking and NoThinking across different MATH levels.

## 5.2 Baselines

We compare AdaptThink with the following representative methods for efficient reasoning:

$\mathbf { D P O } _ { S h o r t e s t }$ constructs preference data by sampling multiple responses for each problem in the training dataset and pairing the shortest correct response and the longest responses, then uses DPO (Rafailov et al., 2023) to finetune the model.

• OverThink (Chen et al., 2024) first constructs preference data by taking the original longthinking response for each training problem as the negative example and retaining the first two attempts that arrive at the correct answer in the thinking as the positive example, and then uses SimPO (Meng et al., 2024) to alleviate models overthinking behaviors.

• DAST (Shen et al., 2025) first constructs preference data by ranking pre-sampled responses with a length-based reward function, and then employs SimPO to finetune the model.

• O1-Pruner (Luo et al., 2025a) first estimates the reference model’s performance through presampling and then uses off-policy RL-style finetuning to encourage the model to generate shorter reasoning processes under accuracy constraints.

<table><tr><td rowspan="2">Method</td><td colspan="3">GSM8K</td><td colspan="3">MATH500</td><td colspan="3">AIME 2024</td><td colspan="2">Average</td></tr><tr><td>Acc</td><td>Length</td><td>RatioNT</td><td>Acc</td><td>Length</td><td>RatioNT</td><td>Acc</td><td>Length</td><td>RatioNT</td><td>∆Acc</td><td>∆Length</td></tr><tr><td colspan="10">DeepSeek-R1-Distill-Qwen-1.5B</td><td></td><td></td></tr><tr><td> $\mathrm { O r i g i n a l } _ { T h i n k i n g }$ </td><td>79.0</td><td>978</td><td>0.0%</td><td>80.6</td><td>4887</td><td>0.0%</td><td>29.4</td><td>12073</td><td>0.0%</td><td></td><td></td></tr><tr><td>OriginalNoThinking</td><td>69.8</td><td>280</td><td>100.0%</td><td>67.2</td><td>658</td><td>100.0%</td><td>14.0</td><td>2190</td><td>100.0%</td><td>-12.7</td><td>-79.9%</td></tr><tr><td colspan="10">AdaptThink-1.5B</td></tr><tr><td>δ=0</td><td>84.6</td><td>718</td><td>70.4%</td><td>86</td><td>2511</td><td>50.0%</td><td>34.8</td><td>9279</td><td>0.2%</td><td>+5.5</td><td>-32.8%</td></tr><tr><td>δ=0.01</td><td>82.4</td><td>638</td><td>55.6%</td><td>82.4</td><td>2473</td><td>67.0%</td><td>32.5</td><td>9165</td><td>19.8%</td><td>+2.8</td><td>-36.1%</td></tr><tr><td>δ=0.02</td><td>83.1</td><td>628</td><td>75.7%</td><td>83.4</td><td>2337</td><td>62.6%</td><td>31.3</td><td>8696</td><td>21.3%</td><td>+2.9</td><td>-38.6%</td></tr><tr><td>δ=0.05</td><td>83.1</td><td>480</td><td>86.9%</td><td>82</td><td>1782</td><td>76.8%</td><td>31</td><td>6679</td><td>40.4%</td><td>+2.4</td><td>-53.0%</td></tr><tr><td>δ=0.075</td><td>83.2</td><td>580</td><td>71.8%</td><td>80.2</td><td>1621</td><td>84.2%</td><td>29.2</td><td>6087</td><td>64.2%</td><td>+1.2</td><td>-52.4%</td></tr><tr><td>δ=0.1</td><td>82.5</td><td>358</td><td>91.7%</td><td>78.2</td><td>1272</td><td>90.4%</td><td>26.7</td><td>5301</td><td>83.5%</td><td>-0.5</td><td>-64.5%</td></tr></table>

Table 2: Performance of AdaptThink-1.5B using different δ value.

• TLMRE (Arora and Zanette, 2025) incorporates a length-based penalty in on-policy RL to incentivize the model to produce shorter responses.

• ModelMerging (Wu et al., 2025) reduces the response length of a reasoning model by weightedly averaging its weights with a non-reasoning model (i.e., Qwen-2.5-Math-1.5B/7B).

• RFT<sub>MixThinking</sub> (Reject Fine-tuning) first samples multiple responses for each training problem x using both Thinking and NoThinking, then selects (1) correct NoThinking responses if the instance-level pass rate $\bar { R } _ { \mathrm { n o t h i n k } } ( x ) \ \geq$ $\bar { R } _ { \mathrm { t h i n k } } ( x )$ and (2) correct Thinking responses if $\bar { R } _ { \mathrm { n o t h i n k } } ( x ) < \bar { R } _ { \mathrm { t h i n k } } ( x )$ , and uses these selected responses to finetune the model.

For fair comparison, we re-implement all these baselines using DeepScaleR dataset.

## 5.3 Main Results

Table 1 presents the evaluation results of different methods on GSM8K, MATH500, and AIME 2024. Compared to the original 1.5B and 7B models, AdaptThink reduces the average response length by 53.0% and 40.1%, respectively, while also improves the average accuracy by 2.4% and 2.3%, demonstrating that AdaptThink enables significantly more efficient reasoning without compromising and even enhancing model performance. Moreover, AdaptThink outperforms most baselines—all of which optimize response length within the Thinking mode—in terms of both accuracy and length reduction. It also achieves the best average results, highlighting the effictiveness of adaptive thinking-mode selection as a novel paradigm for achieving efficient reasoning.

For the methods that involve both Thinking and NoThinking modes, we additionally report the ratio of responses generated in NoThinking mode (i.e., Ratio<sub>NT</sub> in Table 1). As shown in the table, AdaptThink produces more NoThinking responses for easier test sets (i.e., GSM8K and MATH500) while employing Thinking mode more frequently for challenging test sets (i.e., AIME 2024). See Appendix A for detailed cases. A similar trend is also observed within the five difficulty levels of MATH500 (Figure 3 and 5), where AdaptThink predominantly selects the NoThinking mode for the easiest Level-1 problems, and progressively increases the use of Thinking as the difficulty level rises. Meanwhile, compared to the original models using only Thinking or NoThinking mode, Adapt-Think consistently achieves higher accuracy across most difficulty levels. These findings suggest that AdaptThink has successfully taught the model to adaptively choose an appropriate thinking mode based on problem difficulty, achieving a better balance between efficiency and performance.

## 5.4 More Analyses

Effect of δ. To show the effect of δ in our advantage function A(x, y), we implement AdaptThink with different δ values on the 1.5B model and summarize the evaluation results in Table 2. As δ increases, the proportion of NoThinking responses progressively rises, resulting in a corresponding reduction in the average response length. However, the gain in accuracy also gradually decreases at the same time. This indicates that δ serves as a control parameter for the trade-off between reasoning efficiency and accuracy improvement. Notably, even when δ = 0, the model chooses NoThinking for over 50% of problems in GSM8K and MATH500, implying that NoThinking may possess an inherent advantage over Thinking when addressing relatively straightforward problems. Furthermore, for most δ values, AdaptThink consistently achieves notable reduction in response length and also improves accuracy, which underscores its robustness.

![](images/32ee34f54a370ba78a7114b9509edefa44887623e2e137b4cd9f433bff084f93.jpg)

![](images/079711e01ce8eda785437dd9d895bf9e05a98fcb8f7ba39e4b3af28b188ddc6c.jpg)

![](images/b7785fbce72d4d2026abb7ca8083af4157362402ce8cd2a3be1315b50467064a.jpg)  
Figure 4: Comparison of average accuracy, average response length, and the ratio of NoThinking responses on the three math test sets between AdaptThink and naive GPRO at different training steps.

<table><tr><td>Method</td><td>RatioIT</td><td>Length</td></tr><tr><td colspan="3">DeepSeek-R1-Distill-Qwen-1.5B</td></tr><tr><td>Final Solutions from OriginalThinking OriginalNoThinking AdaptThink</td><td>9.5% 8.2% 7.9%</td><td>1799 665 826</td></tr><tr><td colspan="3">DeepSeek-R1-Distill-Qwen-7B</td></tr><tr><td>Final Solutions from  $\mathrm { O r i g i n a l } _ { T h i n k i n g }$ </td><td>0.7%</td><td>321</td></tr><tr><td> $\mathrm { O r i g i n a l } _ { N o T h i n k i n g }$  AdaptThink</td><td>0.9% 4.2%</td><td>341 426</td></tr></table>

Table 3: For the test cases where AdaptThink chooses NoThinking, we compare the implicit thinking ratio (Ratio<sub>IT</sub>) and average length of three scenarios.

Effect of importance sampling. To demonstrate the effect of importance sampling, we compare AdaptThink with naive GRPO that samples directly from $\pi _ { \theta _ { \mathrm { o l d } } } ( \cdot | x )$ during training. As shown in Figure 4, because $\pi _ { \theta _ { \mathrm { o l d } } } ( \cdot | x )$ is initially unable to produce NoThinking samples, GRPO can only learn from Thinking samples and focus on improving accuracy throughout the training process. The response length of GRPO decreases only to around 4,600 (by eliminating overlong responses that would be truncated and receive no reward), after which it gradually increases. In contrast, our importance sampling strategy enables AdaptThink to learn from both Thinking and NoThinking samples at each training step. As the model gradually learn to generate more NoThinking responses for simple problems, the response length eventually decreases to below 3,000 tokens.

Implicit thinking ratio. A potential concern for AdaptThink is that RL may activate thinking features (e.g., reflection) within NoThinking mode (similar to DeepSeek-R1-Zero) and produce many implicit thinking responses. To allay this concern, we examine the test cases where AdaptThink chooses the NoThinking mode. For these cases, we collect (1) NoThinking responses from AdaptThink, (2) NoThinking responses from the original reasoning model, and (3) the final solution part of Thinking responses from the original model. We compare the ratio of implicit thinking responses (denoted by Ratio<sub>IT</sub>) across these three scenarios by detecting whether some representative keywords for thinking (e.g, “Wait” and “Alternatively”) appear in the solutions. We also compare the average length of these solutions. As presented in Table 3, Adapt-Think only slightly increases the implicit thinking ratio and response length for the 7B model. To entirely eliminate such behavior, one possible approach is to assign zero reward to implicit thinking samples during RL training.

<table><tr><td rowspan="2">Method</td><td colspan="4">MMLU Acc Length RatioNT ∆Acc ∆Length</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td colspan="3">DeepSeek-R1-Distill-Qwen-1.5B</td><td></td><td></td></tr><tr><td> $\mathrm { O r i g i n a l } _ { T h i n k i n g }$   $\mathrm { O r i g i n a l } _ { N o T h i n k i n g }$  AdaptThink</td><td>35.7 20.6 42.2</td><td>1724 208 1055</td><td>0.00% 100.00% 16.43%</td><td>- 1 -15.1 -87.9% +6.5 -38.8%</td></tr><tr><td colspan="3">DeepSeek-R1-Distill-Qwen-7B</td><td></td><td></td></tr><tr><td> $\mathrm { O r i g i n a l } _ { T h i n k i n g }$ </td><td>63.4 1257</td><td>0.00%</td><td>一</td><td>一</td></tr><tr><td rowspan="2"> $\mathrm { O r i g i n a l } _ { N o T h i n k i n g }$   $A d a p t T h i n k$ </td><td>51.2 128</td><td>100.00%</td><td>-12.2</td><td>-89.8%</td></tr><tr><td>63.6 856</td><td>16.41%</td><td>+0.2</td><td>-31.9%</td></tr></table>

Table 4: The performance of AdaptThink on the out-ofdistribute test set MMLU.

Generalizability to OOD scenario. To assess the ability of AdaptThink to generalize in out-ofdistribution scenarios, we conduct an evaluation on MMLU, which contains 14K multi-choice questions and covers 57 diverse domains, distinct from our training data in question format and subjects. As shown in Table 4, AdaptThink reduces average response length by more than 30% by producing NoThinking responses for about 16% of the problems (see Figure 8 for cases), while achieving higher accuracy than the original models.

## 6 Conclusion

In this work, we first demonstrate the advantages of NoThinking over Thinking in both performance and efficiency for relatively simple tasks. Motivated by this, we propose AdaptThink, a novel RL algorithm to enable reasoning models to adaptively select the optimal thinking mode based on problem difficulty. Experiments show that AdaptThink significantly reduces inference costs and further improves model performance, highlighting the promise of adaptive thinking-mode selection for advancing the trade-off between reasoning quality and efficiency.

## 7 Limitation

We discuss several limitations of our work in this section: (1) Due to limited computational resources, we only conduct our experiments on 1.5B and 7B models. Nevertheless, these experiments still demonstrate the efficacy of AdaptThink across different model sizes. (2) Similar to most previous open-sourced works, we only train our models using mathematical datasets because they are easy to obtain and can offer accurate, verifiable rewards. Though our evaluation on MMLU shows Adapt-Think models can well generalize to OOD scenarios, we believe they can achieve better results if more training datasets with verifiable rewards for general domains are available.

## 8 Ethical Considerations

All the models and datasets used in this work are publicly published with permissible licenses.

## 9 Acknowledgement

This work is supported by National Natural Science Foundation of China (62476150), Beijing Natural Science Foundation (L243006), and Tsinghua University Initiative Scientific Research Program.

## References

Pranjal Aggarwal and Sean Welleck. 2025. L1: con trolling how long A reasoning model thinks with reinforcement learning. CoRR, abs/2503.04697.

Daman Arora and Andrea Zanette. 2025. Training language models to reason efficiently. CoRR, abs/2502.04463.

Xingyu Chen, Jiahao Xu, Tian Liang, Zhiwei He, Jianhui Pang, Dian Yu, Linfeng Song, Qiuzhi Liu, Mengfei Zhou, Zhuosheng Zhang, Rui Wang, Zhaopeng Tu, Haitao Mi, and Dong Yu. 2024. Do

NOT think that much for 2+3=? on the overthinking of o1-like llms. CoRR, abs/2412.21187.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. CoRR, abs/2110.14168.

DeepSeek-AI. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. CoRR, abs/2501.12948.

Yichao Fu, Junda Chen, Yonghao Zhuang, Zheyu Fu, Ion Stoica, and Hao Zhang. 2025. Reasoning without self-doubt: More efficient chain-of-thought through certainty probing. In ICLR 2025 Workshop on Foundation Models in the Wild.

Bofei Gao, Feifan Song, Zhe Yang, Zefan Cai, Yibo Miao, Qingxiu Dong, Lei Li, Chenghao Ma, Liang Chen, Runxin Xu, Zhengyang Tang, Benyou Wang, Daoguang Zan, Shanghaoran Quan, Ge Zhang, Lei Sha, Yichang Zhang, Xuancheng Ren, Tianyu Liu, and Baobao Chang. 2024. Omni-math: A universal olympiad level mathematic benchmark for large language models. CoRR, abs/2410.07985.

Tingxu Han, Zhenting Wang, Chunrong Fang, Shiyu Zhao, Shiqing Ma, and Zhenyu Chen. 2024. Token-budget-aware LLM reasoning. CoRR, abs/2412.18547.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. Let’s verify step by step. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. Open-Review.net.

Haotian Luo, Li Shen, Haiying He, Yibo Wang, Shiwei Liu, Wei Li, Naiqiang Tan, Xiaochun Cao, and Dacheng Tao. 2025a. O1-pruner: Lengthharmonizing fine-tuning for o1-like reasoning pruning. CoRR, abs/2501.12570.

Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Li Erran Li, Raluca Ada Popa, and Ion Stoica. 2025b. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl. Notion Blog.

Wenjie Ma, Jingxuan He, Charlie Snell, Tyler Griggs, Sewon Min, and Matei Zaharia. 2025. Reasoning models can be effective without thinking. arXiv preprint arXiv:2504.09858.

Yu Meng, Mengzhou Xia, and Danqi Chen. 2024. Simpo: Simple preference optimization with a reference-free reward. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024.

Yingqian Min, Zhipeng Chen, Jinhao Jiang, Jie Chen, Jia Deng, Yiwen Hu, Yiru Tang, Jiapeng Wang, Xiaoxue Cheng, Huatong Song, Wayne Xin Zhao, Zheng Liu, Zhongyuan Wang, and Ji-Rong Wen. 2024. Imitate, explore, and self-improve: A reproduction report on slow-thinking reasoning systems. CoRR, abs/2412.09413.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel J. Candès, and Tatsunori Hashimoto. 2025. s1: Simple test-time scaling. CoRR, abs/2501.19393.

OpenAI. 2024. Learning to reason with llms. https://openai.com/index/ learning-to-reason-with-llms/. Accessed: 2025-05-07.

Xiaoye Qu, Yafu Li, Zhaochen Su, Weigao Sun, Jianhao Yan, Dongrui Liu, Ganqu Cui, Daizong Liu, Shuxian Liang, Junxian He, Peng Li, Wei Wei, Jing Shao, Chaochao Lu, Yue Zhang, Xian-Sheng Hua, Bowen Zhou, and Yu Cheng. 2025. A survey of efficient reasoning for large reasoning models: Language, multimodality, and beyond. CoRR, abs/2503.21614.

Qwen Team. 2025. Qwq-32b-preview. https: //qwenlm.github.io/blog/qwq-32b-preview/. Accessed: 15 March 2025.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. ArXiv, abs/1707.06347.

Yi Shen, Jian Zhang, Jieyun Huang, Shuming Shi, Wenjing Zhang, Jiangze Yan, Ning Wang, Kai Wang, and Shiguo Lian. 2025. DAST: difficulty-adaptive slow-thinking for large reasoning models. CoRR, abs/2503.04472.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2024. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256.

Yang Sui, Yu-Neng Chuang, Guanchu Wang, Jiamu Zhang, Tianyi Zhang, Jiayi Yuan, Hongyi Liu, Andrew Wen, Shaochen Zhong, Hanjie Chen, and Xia Ben Hu. 2025. Stop overthinking: A survey on efficient reasoning for large language models. CoRR, abs/2503.16419.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, Chuning

Tang, Congcong Wang, Dehao Zhang, Enming Yuan, Enzhe Lu, Fengxiang Tang, Flood Sung, Guangda Wei, Guokun Lai, and 75 others. 2025. Kimi k1.5: Scaling reinforcement learning with llms. CoRR, abs/2501.12599.

Han Wu, Yuxuan Yao, Shuqi Liu, Zehua Liu, Xiaojin Fu, Xiongwei Han, Xing Li, Hui-Ling Zhen, Tao Zhong, and Mingxuan Yuan. 2025. Unlocking efficient longto-short LLM reasoning with model merging. CoRR, abs/2503.20641.

Silei Xu, Wenhao Xie, Lingxiao Zhao, and Pengcheng He. 2025. Chain of draft: Thinking faster by writing less. CoRR, abs/2502.18600.

![](images/14e5aa838e64b13abee7b1d0879f96e09fec0e9208980fb2384ffec1e8d0d85c.jpg)

![](images/4b4ad4cb3bace1ae8d677e7688451e4b33f6d84c0562d9484bba58ea8f82f07c.jpg)  
Figure 5: Left: The ratio that AdaptiveThink-1.5B choose Thinking or NoThinking across different MATH levels. Right: Comparison of accuracy between AdaptiveThink-1.5B and the DeepSeek-R1-Distill-Qwen-1.5B using Thinking and NoThinking across different MATH levels.

## A Case Study

We show several cases of AdaptThink in Figure 6, 7, 8, and 9 to illustrate its ability of adaptive thinkingmode selection. As demonstrated in Figure 6, 7, and 8, when faced with relatively simple problems, DeepSeek-R1-Distill-Qwen-7B costs a large number of tokens in thinking, which often contains many unnecessary steps and redundant attempts. In contrast, our AdaptThink-7B model adaptively chooses NoThinking mode and directly produces a concise final solution, even if the input problem differs from our training data in both the question format and the subject. On the hard hand, the case in Figure 9 indicates that, when the input problem is difficult, AdaptThink-7B is also able to automatically employ Thinking to ensure its accuracy.

g Question (GSM8K): Two cars are driving on a highway. The first car is traveling at an average speed of 60 miles per hour when the second car passes it at an average speed of 70 miles per hour. If both cars continue on the highway at the same speed, how many miles will separate them after 2 hours?

## 3 DeepSeek-R1-Distill-Qwen-7B:

<think>Okay, so I have this problem here about two cars on a highway. Let me try to understand it step by step. The first car is going at an average speed of 60 miles per hour. Then, the second car passes the first car, and it’s going at 70 miles per hour. The question is asking, if both cars keep going at the same speed, how many miles apart will they be after 2 hours. Hmm, okay. So, let me visualize this. There’s a highway, and both cars are moving in the same direction. The first car is ahead, going 60 mph, and the second car catches up to it, going 70 mph. So, the second car is faster, right? It’s going 10 mph faster than the first car. Wait, so the second car is approaching the first car at a rate of 10 miles per hour. That makes sense because 70 minus 60 is 10. So, the gap between them is closing at 10 mph. But the question is asking, after 2 hours, how far apart will they be. Hmm, wait a second. If the second car is catching up, the distance between them is decreasing. But the question says, how many miles will separate them after 2 hours. So, is it possible that the second car overtakes the first car before 2 hours, and then the distance becomes positive again? Or maybe they’re both moving in the same direction, so the second car is just gaining on the first car, but after 2 hours, the distance between them is how much? Wait, [...](omit 2679 tokens) so, I think I’m confident that the answer is 20 miles.</think> Final Answer: The number of miles that will separate them after 2 hours is 20 . The first car is traveling at an average speed of 60 miles per hour. The second car passes it at an average speed of 70 miles per hour. Both cars continue on the highway at the same speed. We need to determine how many miles will separate them after 2 hours. 1. The relative speed between the second car and the first car is the difference in their speeds: 70 mph  60 mph = 10 mph. 2. The separation between the two cars after time t hours is given by the relative speed multiplied by time: Separation = 10 mph t hours. 3. After 2 hours, the separation will be: Separation = 10 mph 2 hours = 20 miles. Thus, the number of miles that will separate them after 2 hours is 20 . (3303 tokens)

## 3 AdaptThink-7B (ours):

<think></think>To determine how many miles will separate the two cars after 2 hours, let’s break down the problem step by step. 1. Understand the Speeds: The first car is traveling at 60 miles per hour (mph). The second car is traveling at 70 mph. 2. Determine the Relative Speed: Since both cars are moving in the same direction, the relative speed between them is the difference in their speeds. Relative speed = Speed of second car  Speed of first car = 70 mph  60 mph = 10 mph. 3. Calculate the Distance Separated After 2 Hours: Distance = Speed Time. Here, the relative speed is 10 mph, and the time is 2 hours. So, Distance = 10 mph 2 hours = 20 miles. Therefore, after 2 hours, the two cars will be 20 miles apart. (222 tokens)

Figure 6: For a simple math problem from GSM8K, DeepSeek-R1-Distill-Qwen-7B costs about 3000 tokens in thinking, which contains many unnecessary steps and redundant attempts. In contrast, our AdaptThink-7B adaptively chooses NoThinking mode and directly produces a concise final solution.

g Question (MATH500): In how many ways can 8 people sit around a round table if 3 of the people – Pierre, Rosa, and Thomas – all want to sit together? (Two seatings are considered the same if one is a rotation of the other.)

## 3 DeepSeek-R1-Distill-Qwen-7B:

<think>Alright, so I have this problem where I need to figure out how many ways 8 people can sit around a round table, but there’s a condition: Pierre, Rosa, and Thomas all want to sit together. Hmm, okay. Let me think about this step by step. First, I remember that when arranging people around a round table, the number of arrangements is different from arranging them in a straight line because rotations of the same arrangement are considered identical. For a round table with n people, the number of distinct seating arrangements is (n 1)! instead of n!. That’s because fixing one person’s position accounts for the rotations, and then arranging the others relative to that fixed position. But in this case, there’s a condition: Pierre, Rosa, and Thomas must all sit together. So, how does that affect the calculation? I think the way to handle this is to treat the group of Pierre, Rosa, and Thomas as a single entity or "block." If we consider them as one block, then we’re effectively arranging this block along with the other 5 people. So, that makes a total of 6 "blocks" to arrange around the table. Wait, hold on. If we have 8 people and we’re grouping 3 of them together, how many people are left? Let me subtract: $8 - 3 = 5$ . So, yes, we have 5 individuals plus the block, totaling 6 units to arrange around the table. Now, since it’s a round table, the number of ways to arrange these 6 units is $( 6 - 1 ) ! = 5 ! = 1 2 0$ . That’s because, again, fixing one position accounts for the rotational symmetry. But wait, [...](omit 9138 tokens) Therefore, I think the answer is 720. </think> Final Answer: The number of ways is 720 . To determine the number of ways 8 people can sit around a round table with the condition that Pierre, Rosa, and Thomas must sit together, we can treat them as a single block. This reduces the problem to arranging 6 entities (the block and the other 5 people) around a circular table. Treat the group as a single block: This gives us 6 entities (the block and 5 individuals). Arrange these 6 entities around a circular table: The number of distinct arrangements of 6 entities around a circular table is (6 1)! = 5!. Arrange the people within the block: The 3 people in the block can be arranged among themselves in 3! ways. Thus, the total number of arrangements is the product of these two results: $( 6 - 1 ) ! \times 3 ! = 5 ! \times 3 ! = 1 2 0 \times 6 = 7 2 0$ Therefore, the number of ways 8 people can sit around a round table with Pierre, Rosa, and Thomas sitting together is 720 . (9812 tokens)

## 3 AdaptThink-7B (ours):

<think></think>To determine the number of ways 8 people can sit around a round table with the condition that Pierre, Rosa, and Thomas must all sit together, we can treat the group of three as a single entity or "block." Here’s the step-by-step solution: 1. Treat Pierre, Rosa, and Thomas as a single block: This reduces the problem to arranging 6 entities around a round table: the block and the other 5 individuals. 2. Arrange the 6 entities around the table: For circular arrangements, the number of ways to arrange n entities is (n 1)!. So, for 6 entities, it’s $( 6 - 1 ) ! = 5 ! = 1 2 0$ ways. 3. Arrange Pierre, Rosa, and Thomas within their block: The three people can be arranged among themselves in 3! = 6 ways. 4. Multiply the two results together: The total number of arrangements is the product of the arrangements of the blocks and the internal arrangements of the group: $5 ! \times 3 ! = 1 2 0 \times 6 = 7 2 0$ . So, the number of ways 8 people can sit around a round table with Pierre, Rosa, and Thomas all sitting together is 720 . (297 tokens)

Figure 7: For a relatively simple math problem from MATH500, DeepSeek-R1-Distill-Qwen-7B costs about 9000 tokens in thinking. In contrast, our AdaptThink-7B adaptively chooses NoThinking mode and directly produces a concise final solution, using only 297 tokens.

g Question (MMLU): A proposed mechanism for a reaction is as follows: $\mathrm { N O _ { 2 } + F _ { 2 } \to N O _ { 2 } F + F }$ (Slow step). $\mathrm { F + N O _ { 2 } \to N O _ { 2 } F }$ (Fast step). What is the order of the overall reaction? A) Zero order. B) First order. C) Second order. D) Third order. Output the correct option (one of ABCD) in " boxed{}", such as " boxed $\{ \mathbf { A } \} "$

## 3 DeepSeek-R1-Distill-Qwen-7B:

<think>Okay, so I have this chemistry problem here about reaction mechanisms and determining the overall order of the reaction. Hmm, let me try to figure this out step by step. The problem says the proposed mechanism is: 1. $\mathrm { N O _ { 2 } + F _ { 2 } \to N O _ { 2 } F + F }$ (slow step) 2. $\mathrm { F + N O _ { 2 } \to N O _ { 2 } F }$ (fast step) And the question is asking for the order of the overall reaction. The options are A. Zero order, B. First order, C. Second order, D. Third order. Alright, I remember that the overall order of a reaction is determined by the sum of the orders of each reactant in the rate law. But wait, the rate law isn’t always given. In this case, the mechanism is provided, so I need to use that to figure out the rate law. First, in a reaction mechanism, the slow step is the rate-determining step, right? That means the rate law is based on that step. So I should focus on the first step: $\mathrm { N O _ { 2 } + F _ { 2 } \to N O _ { 2 } F + F }$ . Looking at the slow step, the reactants are $\mathrm { N O _ { 2 } }$ and $\mathrm { F _ { 2 } }$ . So the rate law would be rate $= k [ \mathrm { N O _ { 2 } } ] [ \mathrm { F _ { 2 } } ] ^ { m }$ where m is the order with respect to $\mathrm { F _ { 2 } }$ But wait, in the slow step, $\mathrm { F _ { 2 } }$ is a reactant, but in the second step, there’s an F that’s produced. Hmm, I think that in the overall reaction, the intermediates (like F in this case) should be canceled out to get the rate law in terms of the initial reactants. Hold on, [...](omit 3887 tokens) Therefore, the answer is C. Second order. Final Answer C </think> The proposed mechanism for the reaction is as follows: 1. $\mathrm { N O _ { 2 } + F _ { 2 } \to N O _ { 2 } F + F }$ (slow step) 2. $\mathrm { F + N O _ { 2 }  N O _ { 2 } F }$ (fast step) The overall reaction is determined by combining the two steps: $\mathrm { N O _ { 2 } + F _ { 2 } + F + N O _ { 2 }  N O _ { 2 } F + F + N O _ { 2 } F }$ Simplifying, we get: $2 \mathrm { N O _ { 2 } } + \mathrm { F _ { 2 } } \to 2 \mathrm { N O _ { 2 } F }$ . The rate law is determined by the slow step, which is the rate-determining step. The rate law for the slow step is: $\mathrm { r a t e } = k [ \mathrm { N O _ { 2 } } ] [ \mathrm { F _ { 2 } } ]$ . Since F is an intermediate, we use the steady-state approximation to express [F] in terms of $\left[ \mathrm { N O _ { 2 } } \right]$ and $[ \mathrm { F _ { 2 } } ]$ . From the fast step, we find: $k _ { 1 } [ \mathrm { N O _ { 2 } } ] [ \mathrm { F _ { 2 } } ] = k _ { 2 } [ \mathrm { F } ] [ \mathrm { N O _ { 2 } } ]$ . Solving for $\begin{array} { r } { { \bf { [ F ] } } \colon [ { \bf F } ] = \frac { k _ { 1 } } { k _ { 2 } } [ { \bf { F } } _ { 2 } ] } \end{array}$ . Substituting this back into the rate law confirms that the rate is still first order in $\left[ \mathrm { N O _ { 2 } } \right]$ and first order in $\left[ \mathrm { F _ { 2 } } \right]$ . Therefore, the overall order of the reaction is: $1 + 1 = 2$ . Thus, the order of the overall reaction is second order. C (4709 tokens)

## 3 AdaptThink-7B (ours):

<think></think>To determine the order of the overall reaction, we analyze the given mechanism: 1. Slow Step (Rate-Determining Step): $\mathrm { N O _ { 2 } + F _ { 2 } \to N O _ { 2 } F + F }$ The rate law for this step is: $\mathrm { R a t e } = k [ \mathrm { N O _ { 2 } } ] [ \mathrm { F _ { 2 } } ]$ .. 2. Fast Step: $\mathrm { F + N O _ { 2 } \to N O _ { 2 } F }$ . Since this step is fast, it does not affect the overall rate law. The overall rate law is determined by the slow step. Combining the rate laws from the slow step, the overall reaction order is the sum of the exponents in the rate law: the order with respect to $\mathrm { N O _ { 2 } }$ is 1, and the order with respect to $\mathrm { F _ { 2 } }$ is 1. Thus, the overall reaction order is $1 + 1 = 2$ The correct answer is C . (263 tokens)

Figure 8: Even for out-of-distribution problems from MMLU, AdaptThink-7B is still able to generate NoThinking response if the problem is relatively simple.

g Question (AIME 2024): Find the largest possible real part of $\begin{array} { r } { ( 7 5 + 1 1 7 i ) z + \frac { 9 6 + 1 4 4 i } { z } } \end{array}$ where z is a complex number with $| z | = 4$

## 3 DeepSeek-R1-Distill-Qwen-7B:

<think>Alright, so I have this problem here: I need to find the largest possible real part of the expression $\begin{array} { r } { ( \bar { 7 } 5 + 1 1 7 i ) z + \frac { 9 6 + \bar { 1 } 4 4 i } { z } } \end{array}$ where z is a complex number with $| z | = 4 .$ Hmm, okay, let me try to figure this out step by step. First, I remember that for complex numbers, the real part of an expression is the sum of the real parts of each term. So, I can probably break this down into two parts: the real part of $( 7 5 + 1 1 7 i ) z$ and the real part of $\frac { 9 6 + 1 { \hat { 4 } } 4 i } { z }$ Then, $[ \ldots ] ( \mathbf { o m i t }$ 3099 tokens) I think that solidifies the answer. I don’t see any mistakes in the reasoning. Final Answer 540 </think> Given the problem of finding the largest possible real part of the expression $\begin{array} { r } { \overline { { ( 7 5 + 1 1 7 i ) } } z + \frac { 9 6 + 1 4 4 i } { z } } \end{array}$ where z is a complex number with $| z | = 4 ,$ we start by expressing z as $z = 4 e ^ { i \theta }$ . This allows us to rewrite the expression in terms of θ. First, we substitute $z = 4 e ^ { i \theta }$ and $\begin{array} { r } { \frac { 1 } { z } = \frac { 1 } { 4 } e ^ { - i \theta } } \end{array}$ into the expression: $( 7 5 + { \bar { 1 } } 1 7 i ) \cdot 4 e ^ { i \theta } + ( 9 6 + 1 4 4 i ) \cdot { \frac { 1 } { 4 } } e ^ { - i \theta }$ . This simplifies to $( 3 0 0 ^ { \sim } + 4 6 \bar { 8 } i ) e ^ { i \theta } + ( 2 4 + 3 6 i ) e ^ { - i \theta }$ . Expanding each term using Euler’s formula $e ^ { i \theta } = \cos \theta + i \sin \theta$ and $e ^ { - i \theta } = \cos \theta - i \sin \theta ,$ we separate the real and imaginary parts. After combining the real parts, we get 324 cos θ 432 sin θ. To find the maximum value of this expression, we recognize it as a linear combination of sine and cosine, which can be written in the form A cos $\theta + B \sin \theta .$ The maximum value of such an expression is given by the amplitude $R = \sqrt { A ^ { 2 } + B ^ { 2 } }$ . Calculating R: $R = { \sqrt { 3 2 4 ^ { 2 } + ( - 4 3 2 ) ^ { 2 } } } = { \sqrt { 1 0 4 9 7 6 + 1 8 6 6 2 4 } } = { \sqrt { 2 9 1 6 0 0 } } = 5 4 0$ . Thus, the largest possible real part of the given expression is 540 . (3771 tokens)

## 3 AdaptThink-7B (ours):

<think>Okay, so I have this problem here: I need to find the largest possible real part of the expression $( 7 5 + 1 1 7 i ) z + \frac { 9 6 + 1 4 4 i } { z }$ , where z is a complex number with $| z | = 4$ Hmm, okay. Let me try to figure this out step by step. First, I know that z is a complex number with magnitude 4. That means [...](omit 3003 tokens) Therefore, the largest possible real part is 540 . </think> Given the expression $\begin{array} { r } { ( 7 5 + 1 1 7 i ) z + \frac { 9 6 + 1 4 4 i } { z } } \end{array}$ where z is a complex number with $| z | = 4 ,$ , we need to find the largest possible real part. First, we express z as $4 e ^ { i { \hat { \theta } } }$ where θ is the argument of z. Then, we rewrite the expression in terms of θ: $\begin{array} { r } { E = ( \hat { 7 } 5 + 1 1 7 i ) 4 e ^ { i \theta } + \frac { 9 6 + 1 4 4 i } { A } e ^ { - i \theta } } \end{array}$ . Simplifying the constants, we $\mathrm { g e t } \colon ( 7 5 + 1 1 7 i ) 4 = 3 0 0 + 4 6 8 i$ and $\begin{array} { r } { \frac { 9 6 + 1 4 4 i } { 4 } = 2 4 + 3 6 i } \end{array}$ Thus, the expression becomes: $E = ( 3 0 0 + 4 6 8 i ) e ^ { i \theta } + ( 2 4 + 3 6 i ) e ^ { - i \theta }$ Using Euler’s formula, we expand and separate into real and imaginary parts: (300 + 468i)(cos θ + i sin θ) = (300 cos θ 468 sin θ) + i(300 sin $\theta + 4 6 8 \cos \theta )$ and $( 2 4 + 3 6 i ) ( \cos \theta - i \sin \theta ) = ( 2 4 \cos \theta + 3 6 \sin \theta ) + i ( - 2 4 \sin \theta + 3 6 \cos \theta )$ . Adding the real parts together: 300 cos θ  468 sin θ + 24 cos θ + 36 sin θ = 324 cos θ  432 sin θ. The expression 324 cos θ  432 sin θ can be written as $R \cos ( \theta + \phi )$ where $R = \sqrt { 3 2 4 ^ { 2 } + ( - 4 3 2 ) ^ { 2 } }$ . Calculating R: $3 2 4 ^ { 2 } = 1 0 4 9 7 6 , 4 3 2 ^ { 2 } = 1 8 6 6 2 4 , R = { \sqrt { 1 0 4 9 7 6 + 1 8 6 6 2 4 } } = { \sqrt { 2 9 1 6 0 0 } } = 5 4 0 .$ Thus, the largest possible real part is 540 . (3743 tokens)

Figure 9: For a chanlleging problems from AIME 2024, AdaptThink-7B is able to employs Thinking to solve it, instead of directly generates the final solution.