# TTT-Bench: A Benchmark for Evaluating Reasoning Ability with Simple and Novel Tic-Tac-Toe-style Games

Prakamya Mishra \*, Jiang Liu ∗, Jialian Wu, Xiaodong Yu, Zicheng Liu, Emad Barsoum Advanced Micro Devices, Inc. (AMD) https://prakamya-mishra.github.io/TTTBench

## Abstract

Large reasoning models (LRMs) have demonstrated impressive reasoning capabilities across a broad range of tasks including Olympiad level mathematical problems, indicating evidence of their complex reasoning abilities. While many reasoning benchmarks focus on the STEM domain, the ability of LRMs to reason correctly in broader task domains remains underexplored. In this work, we introduce TTT-Bench, a new benchmark that is designed to evaluate basic strategic, spatial, and logical reasoning abilities in LRMs through a suite of four two-player Tic-Tac-Toe-style games that humans can effortlessly solve from a young age. We propose a simple yet scalable programmatic approach for generating verifiable two-player game problems for TTT-Bench. Although these games are trivial for humans, they require reasoning about the intentions of the opponent, as well as the game board’s spatial configurations, to ensure a win. We evaluate a diverse set of state-of-the-art LRMs, and discover that the models that excel at hard math problems frequently fail at these simple reasoning games. Further testing reveals that our evaluated reasoning models score on average 41% & 5% lower on TTT-Bench compared to MATH 500 & AIME 2024 respectively, with larger models achieving higher performance using shorter reasoning traces, where most of the models strug gle on long-term strategic reasoning situations on simple and new TTT-Bench tasks.

## 1 Introduction

Recent advances in large reasoning models (LRMs) have driven significant breakthroughs across various reasoning tasks (Jaech et al., 2024; DeepSeek-AI, 2025; Team, 2025c; Zhao et al., 2024; Guan et al., 2025), including deductive, arithmetic, commonsense, relational, and symbolic reasoning (Huang and Chang, 2023). The high quality of reasoning data and improvements in reinforcement learning techniques (Luo et al., 2025a; Zhang et al., 2024a; Rafailov et al., 2023; Xu et al., 2024; Grattafiori et al., 2024) have markedly improved the performance of LRMs on previously challenging mathematics benchmarks, such as GSM8K (Cobbe et al., 2021), MATH 500 (Lightman et al., 2024), and AIME 2024 (MAA, 2024). However, despite these advancements, the ability of LRMs to reason well on a broader set of reasoning problems, in domains other than maths & STEM, remains largely unexplored.

Many of the existing reasoning benchmarks focus on mathematical or STEM-related problems (Lightman et al., 2024; He et al., 2024; Rein et al., 2024; Suzgun et al., 2023; Zhang et al., 2024b; Mao et al., 2024), mainly due to their verifiable nature and the abundance of annotated problems (Li et al., 2025; Xu et al., 2025). This narrow focus limits the evaluation of LRMs to specific domains and leaves their broader reasoning capabilities largely untested (Huang and Chang, 2023). There is a clear need for benchmarks that evaluate a diverse set of reasoning capabilities (Chang et al., 2023), including strategic, spatial, and logical reasoning. Exploring this broader range of reasoning capabilities is particularly compelling when we consider human reasoning abilities. Even young children effortlessly perform intuitive reasoning tasks, particularly in strategic two-player games. They naturally anticipate their opponent’s moves, block threats, and plan multi-step strategies, relying on an instinctive grasp of strategy and spatial relationships rather than explicit or formal reasoning.

An open question that emerges is: Whether reasoning models proficient in solving sophisticated mathematical problems can correctly reason on broader domains of problems that require basic strategic, spatial, and logical reasoning with straightforward rational and minimal requirements for specialized knowledge. Although there are a few efforts to use two-player board games to evaluate these capabilities in language models (Fontana et al., 2024; Qin et al., 2024; Costarelli et al., 2024; tse Huang et al., 2025), these benchmark games are well-documented in the literature and online, hurting their reliability due to the elevated risk of training of these LRMs on associated benchmark test data.

![](images/e4e76fb6676c2c2ecde7471a9b78d65d33a901199a18790a2293767f502aaf99.jpg)  
Figure 1: Visualization of four game types in TTT-Bench (oTTT, dTTT, cTTT, and sTTT). For each game, the illustration consists of the board position annotations (above), the current game state with Alice’s (white) and Bob’s (black), and the correct next move (represented by a check mark).

Motivated by this gap, we introduce TTT-Bench, a new benchmark specifically created to evaluate the reasoning capability of LRMs through a suite of simple and novel two-player Tic-Tac-Toe-style games (Figure 1). Although trivial for humans, these games require the following three basic reasoning capabilities: (1) Spatial reasoning: to interpret and act upon structured board, (2) Strategic planning: to anticipate an opponent’s next move and for selecting actions that maximize long-term success, and (3) Logical reasoning: essential in two-player games that require reasoning over future game trajectories. These capabilities are often cited as open challenges for LLMs in prior work, including recent benchmarks on planning and spatial tasks(Duan et al., 2024; Lin et al., 2025).

TTT-Bench is a simple and diverse benchmark consisting of four types of two-player games, namely oTTT, dTTT, cTTT, and sTTT (see Figure 1 for examples), and the questions in the benchmark test the reasoning capability of LRMs by posing the question of predicting the best move by a player. The objectives of these games are not only simple but are also introduced here for the first time (except oTTT), with no documented prior literature available online, making our benchmark uncontaminated and highly reliable. Briefly, for each task in TTT-Bench, two players play a strategic game on a configuration dependent on the task to reach a winning state that satisfies a task-dependent winning constraint, and the player who reaches the winning state first wins the game. The benchmark consists of programmatically generated board game questions along with their verifiable answers, where we employ a simple heuristic-based algorithm (Appendix 1) to identify the finite set of next best possible moves and their verdict as the answers to ensure the verifiability of the solutions.

The main contributions of this work are as follows:

• In this work, we introduce TTT-Bench benchmark to evaluate the reasoning capabilities of LRMs that are proficient in solving difficult math problems, on a broader domain of basic strategic, spatial, and logical reasoning tasks through a suite of four simple two-player Tic-Tac-Toe-style games.

• Evaluated a variety of state-of-the-art (SOTA) LRMs on TTT-Bench, and revealed a surprising finding: models skilled at difficult math problems frequently struggle with these simpler reasoning tasks.

• This work not only highlights a fundamental shortcoming in existing LRMs, but also provides a new, simple, and scalable approach for automated verifiable two-player game generation, fostering future research efforts in evaluating the reasoning capability of LRMs.

## 2 Related Work

Advances in Mathematical Reasoning with Large Language Models Recent developments in LRMs have led to significant improvements in their reasoning capabilities (OpenAI, 2025; Jaech et al., 2024; DeepSeek-AI, 2025; Team, 2025c; LG AI Research, 2025; Team et al., 2025) - exhibited by an exponential performance boost in complex reasoning tasks, especially on mathematical problem-solving benchmarks (Lightman et al., 2024; MAA, 2024). Contrary to massive reasoning models, efficiently trained small reasoning models (Luo et al., 2025b; Wen et al., 2025; Team, 2025a; Ye et al., 2025; Muennighoff et al., 2025) have also proven to achieve competitive results across these benchmarks. Rapid improvements in reasoning capabilities of LRMs can be attributed to various training methodologies (Yu et al., 2025; Shao et al., 2024), where Reinforcement learning with verifiable rewards (RLVR) (Lambert et al., 2024; DeepSeek-AI, 2025) has emerged as a predominant approach.

Evaluating Reasoning Ability of LRMs Despite these advancements, the present widely used reasoning benchmarks are predominantly based on mathematical and STEM-focused tasks, potentially limiting the assessment of these models’ reasoning ability in much broader domains. Several works have demonstrated significant shortcomings of LRMs on intuitive reasoning tasks (Valmeekam et al., 2024a,b; Topsakal and Harper, 2024; Topsakal et al., 2024), where even SOTA reasoning models fall short in planning and reasoning. Such findings underscore the necessity for the development of benchmarks that assess a broader spectrum of reasoning, encompassing intuitive and strategic thinking.

## 3 TTT-Bench

TTT-Bench is a simple and scalable two-player Tic-Tac-Toe (TTT) style game benchmark for evaluating the reasoning ability of the present SOTA LRMs. This benchmark consists of four games, namely: oTTT (ordinary TTT), dTTT (double TTT), cTTT (TTT in cube), and sTTT (TTT with squares), which are described in detail below:

• oTTT: This is the ordinary TTT game, where two players play a game on a 3x3 grid. The points on the grid are labeled top to bottom and left to right as A, B, C, D, E, F, G, H, and

I. One player plays with white stone ( ) and the other uses black stone ( ). At each turn, the player places a stone of the corresponding color onto one of the positions that has not been occupied. Whoever has three stones in a line (horizontal, vertical, or diagonal) wins.

• dTTT: This is a variant of TTT with two (double) adjacent 3x3 grids. The points on the first grid are labeled top to bottom, left to right, as A, B, C, D, E, F, G, H, and I. The points on the second grid are labeled top to bottom, left to right, as C, J, K, F, L, M, I, N, and O. Here, note that the points C, F, and I are shared by the two grids. One player plays with white stone ( ) and the other uses black stone ( ). At each turn, the player places a stone of the corresponding color onto one of the positions that has not been occupied. Whoever has three stones in a line (horizontal, vertical, or diagonal) on either grid wins.

• cTTT: This is a variant of TTT which is played on two adjacent cubes. The points A, B, C, and D form the top rectangle in the first cube, and B, I, J, and C form the top rectangle in the second cube. Points E, F, G, and H form the bottom rectangle in the first cube, and F, K, L, and G form the bottom rectangle in the second cube. A-E, B-F, C-G, D-H, I-K, and J-L are the edges. Here, note that the points B, C, G, and F are shared by the two cubes. One player plays with a white sticker ( ) and the other uses a black sticker ( ). At each turn, the player places a stone of the corresponding color onto one of the positions that has not been occupied. Whoever hasfour stickers on the same plane on either cube wins.

• sTTT: This is a variant of TTT which is played on a large grid of five intersections between horizontal and vertical lines. There are five equally spaced horizontal lines, where the distance between two neighboring horizontal lines is one. Similarly, there are five equally spaced vertical lines, and the distance between two neighboring vertical lines is one. There are twenty-five intersection points between the five horizontal and vertical lines. These twenty-five points are labeled from top to bottom, left to right, as A, B, C, D, . . . , Y. One player plays with white stone ( ) and the other uses black stone ( ). At each turn, the player places a stone of the corresponding color onto one of the twenty-five points that have not been occupied. Whoever has four stones that form either a unit square (with side length of one) or a “diagonal square” with side length equal to the square root of two wins.

Examples of these games are illustrated in $\mathrm { F i g \mathrm { - } }$ ure 1, with the corresponding exact questions listed in the Appendix A. As seen from the illustrations, such games are easy for humans to play even from a young age, where a player’s victory is based on their ability to anticipate their opponent’s move, blocking threats, and if they can form basic winning strategies – skills demonstrating the simple reasoning ability of humans.

All the problems in this benchmark are formulated as a text-based board game question that starts with the description of the game, as well as the current game state with all the moves played by both players. The final TTT-Bench task questions are formulated to predict the next best move position by the current player (Appendix A), like: Where should Bob play next?. Such question formulation ensures that our benchmark solving complexity is low, as just predicting the next move out of a small set of possible available moves in our games guarantees a small solution search space, making our benchmark questions relatively easier compared to other works (Shojaee\*† et al., 2025). Since the outcomes of these games are based on finite and deterministic logic, possible next-best moves can be easily identified. Leveraging this, we created an automated pipeline to generate new test samples for each of the four games with verifiable solutions, enabling this benchmark to not only have new and unique questions that the SOTA LRMs are not contaminated against, but also make our benchmark highly scalable in terms of the size, diversity and difficulty.

## 3.1 Automated TTT-Bench Generation

We employ a programmatic approach to generate a diverse set of questions corresponding to the four games present in the TTT-Bench. In this approach, for each game, we first generate all possible game states between two players having played N moves. We then filter out the states for further processing where 1) neither player has already won, and 2) where the player who made the $N ^ { t h }$ move has not yet established a "Winning Fork". Here, by "Winning Fork", we refer to the game state in which a player has two or more potential immediate winning states, with each potential winning state being one turn (move) short of winning the game. In this state, independent of the next move of the opponent, the current player can win with 100% probability by pursuing either of the winning moves on their next chance. Refer to Appendix A for more illustrative examples of "Winning Fork" states. This leaves us with all the game states in which there is a deterministic set of positions in the board for the player playing the $( \bar { N } + 1 ) ^ { t h }$ move, that can lead to their victory in the game.

![](images/22acf8bdcd905064101af9fe323410fa621817db92c49e7ae29b8f331c21f298.jpg)  
Figure 2: Distribution of TTT-Bench question solution types. Above: Distribution of questions and their game verdicts (Win, Blocked, or Fork) based on how the game concludes after the corresponding solution move is played. Below: Distribution of questions and their solution types (single answer or multiple next-best moves).

<table><tr><td>Verdict Category</td><td>Description</td></tr><tr><td>Win</td><td>The player wins the game by playing the  $\overline { { ( N + 1 ) ^ { t h } } }$  move.</td></tr><tr><td>Blocked</td><td>The player blocks the opponent&#x27;s immediate potential win by playing the  $( N + 1 ) ^ { t h }$  move.</td></tr><tr><td>Fork</td><td>The player establishes two or more potential winning states (with exactly one turn short of winning in all the states) by playing the  $( N + 1 ) ^ { t h }$  move.</td></tr></table>

Table 1: Description of game verdict categories in TTT-Bench games.

Our final test set consists of only those games in which N moves had been played, and the verdict of the game after the $( N + 1 ) ^ { t h }$ move falls into one of the categories listed in Table 1. These categories ensure that the solutions generated for TTT-Bench questions, for all the games, are the optimal nextbest moves. The Algorithm 1 in Appendix C describes how we generated optimal next-best move solutions for the TTT-Bench questions, where the game verdicts are either "Win", "Blocked", or "Fork". In the algorithm, we first check if there exists any winning moves at the $( N + 1 ) ^ { t h }$ chance where N moves have already been played so far, and use those moves as the solution with the verdict of "Win". If there are no winning moves, then we checked if there are moves by which the opponent can win on the $( N + 1 ) ^ { t h }$ chance, and used those as the solution with the verdict of "Blocked", since not playing those moves will ensure the opponents win. Finally, in case of no blocking moves as well, we identified moves that result in a fork for the current player that will ensure a future win, and used those moves as the solution with the verdict of "Fork".

![](images/21858d03822127f2a57754e2d0d9a3e052b3d68cab34d844ba58943ae8ff5b06.jpg)  
Figure 3: Pass@1 performance comparison of LRMs on TTT-Bench (Left) and math benchmarks: AIME 2024 & MATH 500 (Right). Exact performance numbers available in Table 3 of Appendix D

Based on this automated game question-answer generation approach explained above, we generate a diverse final test set consisting of \~100 samples from each game. For each game, we used a set of predefined values of N such that the games were simple, solvable, and had a low branching factor. For oTTT we use $N = \{ 4 , 5 \} , N = \{ 6 , 7 \}$ for dTTT, $N = \{ 5 , 6 , 7 \}$ for cTTT, and $N = \{ 6 , 7 \}$ for sTTT. As shown in Figure 2, not only is this dataset diverse across different game verdicts, but it also consists of questions with both single and multiple next-best move solutions. Overall, the final TTT-Bench benchmark consists of 412 questions from four simple two-player TTT games. We release this benchmark for future research under the Apache 2.0 License.

## 4 Main Results

## 4.1 Evaluation Setup

In this work, we evaluate a comprehensive set of recent SOTA LRMs on TTT-Bench and conduct a head-to-head comparison of their performance on TTT-Bench versus two widely used mathematics benchmarks: AIME 2024 (MAA, 2024) (Olympiad-level math) & MATH500 (Lightman et al., 2024) (high school math), to thoroughly investigate the reasoning capabilities of these models. The parameter count of the models in our evaluations ranges from 1.5 billion to 32 billion parameters, which includes models like DeepSeek-R1- Distill-Qwen-1.5B / 7B / 14B / 32B (DeepSeek-AI, 2025), S1.1-1.5B / 3B / 7B / 14B / 32B (Muennighoff et al., 2025), EXAONE-Deep-2.4B / 7.8B / 32B (LG AI Research, 2025), QwQ-32B / 32B-Preview (Team, 2025c), OpenThinker2-7B / 32B (Team, 2025b), and Light-R1-7B/14B/32B (Wen et al., 2025). To explore the reasoning capabilities of the frontier reasoning models, we also evaluated OpenAI’s O3-mini-medium as well as DeepSeek’s DeepSeek-R1 models on TTT-Bench. Our evaluation framework is based on the DeepScaler (Luo et al., 2025b) and VeRL (Sheng et al., 2024) codebases. All of our evaluations are done with AMD Instinct MI300X<sup>TM</sup> GPUs. In the results for all the benchmarks, we report Pass@1 scores that are calculated based on $k = 1 6$ responses, where we generate k responses for each test sample with a temperature of 0.6, a top-p value of 0.95, and a response length of 28k tokens, and calculate its corresponding Pass@1 as:

![](images/d622a4b28165d8ed79b54c71cf46ba5b0283b29e7105eb07f424ea8afcec9cba.jpg)  
Figure 4: Pass@1 performance analysis of LRMs over individual solution verdict category $\mathrm { ( " W i n " }$ , "Blocked", & "Fork") questions in TTT-Bench tasks. Exact performance numbers available in Table 4 of Appendix D.

$$
P a s s @ 1 = \frac { 1 } { k } \sum _ { i = 1 } ^ { k } p _ { i }\tag{1}
$$

Here, $p _ { i }$ represents the correctness of the $i ^ { t h }$ response $( p _ { i } = 1$ in the case of questions with multiple solutions if the model predicts either of the solutions correctly). The final solution predictions are extracted from the generated responses using regular expression extraction from the last boxed answer (Luo et al., 2025b). Only for frontier models, we report the Pass@1 scores based on just one response per test sample in the case of TTT-Bench, whereas we report their corresponding published results in the case of math benchmarks.

## 4.2 TTT-Bench Evaluation Results

Weak reasoning ability of LRMs on simple and intuitive tasks: In Figure 3, we report the performance of SOTA reasoning models individually on the four games in TTT-Bench as well as on AIME 2024 & MATH 500. In our analysis, we leverage the performance drop $( \Delta P a s s @ 1$ as defined below) between the TTT-Bench tasks (x) and math benchmarks $( y )$ as a metric to measure the divergence in reasoning ability of these models on different domain reasoning tasks:

<table><tr><td></td><td>oTTT</td><td>dTTT</td><td>cTTT</td><td>sTTT</td><td>MATH 500</td><td>AIME 2024</td></tr><tr><td>o3-mini-medium</td><td>79.6</td><td>83.0</td><td>52.2</td><td>75.0</td><td> $\overline { { 9 7 . 3 ^ { * } } }$ </td><td>79.6*</td></tr><tr><td>DeepSeek-R1</td><td>88.9</td><td>72.0</td><td>58.9</td><td>75.8</td><td> $9 7 . 3 ^ { * }$ </td><td>79.8*</td></tr></table>

Table 2: Frontier LRM Pass@1 performance on TTT-Bench and math benchmarks. The (\*) indicates that the results for math benchmarks we sourced from the respective model’s published results. The TTT-Bench task performances are based on a single response per test sample.

$$
\Delta P a s s @ 1 = P a s s @ 1 _ { x } - P a s s @ 1 _ { y }\tag{2}
$$

Given the simplicity of the TTT-Bench tasks, we expect Pass@1 scores of SOTA LRMs to be higher relative to their performance on math benchmarks (positive $\Delta P a s s @ 1 )$ , especially for models that do well on math benchmarks. Surprisingly, we observed the opposite – the majority of the evaluated models had lower Pass@1 scores on TTT-Bench compared to their Pass@1 scores on MATH 500 ( Avg ∆Pass@1: -41.36%) & AIME 2024 ( Avg ∆Pass@1: -4.88%). In the case of MATH 500, which consists of high school math problems, the drops are significant relative to all the TTT-Bench tasks across all the models, especially for the small models. Whereas in the case of the olympiad-level math benchmark AIME 2024, LRMs performed well on the oTTT task relative to AIME, with an $\operatorname { A v g } \ \Delta P a s s @ 1$ of +8.56%, whereas they struggled on other TTT-Bench tasks. The $\Delta P a s s @ 1$ trends also illustrate the relative task reasoning difficulties over the TTT-Bench tasks, which are consistent across ∆P ass@1 comparisons against both MATH 500 & AIME 2024: oTTT<dTTT<sTTT<cTTT. We attribute this relative task difficulty order to the following two properties of these games: 1) Spatial constraints in the tasks, and 2) the number of possibilities to choose the next best move in the tasks. In Table 2, we also report the P ass@1 performance scores on TTT-Bench of flagship frontier models like o3-mini-medium & DeepSeek-R1. Due to resource constraints, we only evaluated these models with one response per test sample of TTT-Bench and reported their respective published numbers for MATH 500 & AIME 2024 <sup>1</sup> <sup>2</sup>. From our results, we observe that even the frontier reasoning models find the TTT-Bench tasks as difficult as the standard math benchmarks. These performance trends indicate poor reasoning ability ofSOTA LRMs on TTT-Bench tasks that are simple, intuitive, and of low complexity, compared to their performance on both MATH 500 & AIME 2024 questions that are difficult and require relatively more knowledge and complex reasoning efforts.

![](images/d1813473133e25116731399afcc9b8187eca8c85a330a1f6bb5afbac329832d7.jpg)

![](images/7634e74fcd96f1d70560339772f2ea9d4f9453ca28e26a2b2eb035ca09056b0e.jpg)

![](images/31f5e29171b1e037c9a534e2ff6973151d22a93455d0058a446fc435b2b83380.jpg)

![](images/3062f847f971546089fd70d2370be6b5a0205b07d54af360c5e0778666f8b37e.jpg)

![](images/ee2275d9481d8ba3ffc0943845359af6113cd87f94a37344524ee9cfe471c145.jpg)

![](images/ca916cf9a3db6ae5be4874af35214ba286ed5d8e1f579be27f152557752e1568.jpg)

![](images/8b90cc2caeba971d8ecfdecee6b1f352c76a3b477c22d0de79228963588f4088.jpg)

![](images/6158887bde02991b781b7c045e676847f4c7cf5b5ff16c1d7bc9ab368aaa0bf1.jpg)

![](images/7e35af7c65192bf6d4ca96e56575b437cb71e533d4d33501493177a5c715826e.jpg)

![](images/881353fc96cf470d151533659a008e71ae81fc778d3efa10a715cdd82e399b17.jpg)

![](images/7fd8b6e7c0a5afe8c688e6cebbfe5a1662bdc1cfdff0df2713865676101d6b3b.jpg)

![](images/dc165a120b7807c5b4876775d7ed29ff8007fb1dea7ec3eef9f02a25730dfc91.jpg)  
Figure 5: Top: Pass@1 performance of LRMs on MATH 500, TTT-Bench & AIME 2024 against the mean length of the corresponding solution responses. Mid: Model size against the corresponding model’s mean response lengths on MATH 500, TTT-Bench & AIME 2024. Bottom: The mean response lengths used for solving the TTT-Bench task against the response lengths for solving math benchmarks for different models.

Reasoning models struggle with simple longterm strategic reasoning: Since TTT-bench is generated using an automated pipeline that leveraged a heuristic-based algorithm (Appendix C) for generating the solutions and the respective game’s final verdict ("Win", "Blocked", or "Fork"), we investigate the type of reasoning tasks these LRMs are good at by analyzing their performance over these these individual solution verdict category questions. As shown in Figure 4, we observe that across the TTT-Bench tasks oTTT, dTTT, and sTTT, the performance of LRMs over questions that have the solution with verdict "Win" is consistently higher than the questions with verdict "Blocked", with the lowest for the questions with verdict "Fork". Whereas in the case of cTTT, the performance remained equally low for all the questions, independent of the solution verdict. This consistently high performance over questions with a solution verdict "Win" indicates that almost all the LRMs are capable of doing sound short-term thinking, where the solution is straightforward. On the contrary, a consistent dip in performance over slightly complex reasoning scenarios with "Blocked" & "Fork" solution verdicts indicates that the LRMs struggle to successfully solve simple long-term strategic reasoning tasks where the models need to explore more possibilities and think strategically to either block the opponent’s win or to identify a position which can lead to a long-term assured win.

![](images/a432a63adad34044e90157cd41510997cb1eb117d13886b4945b562d00af9232.jpg)  
Figure 6: Examples of predicted solutions by SOTA LRMs for questions in TTT-Bench.

LRMs overthink on TTT-Bench, with an increase in model size resulting in improved performance and efficient use of chain-of-thought: Reasoning models are usually trained with longcontext data to enable a long chain-of-thought (COT) that has proven to evoke emergent reasoning capabilities like self-reflection and backtracking (DeepSeek-AI, 2025). Ideally, a model should use a longer response length with more thinking tokens for questions deemed difficult and requiring a complex thinking process, whereas a shorter response length is used for simple and trivial questions (Jin et al., 2024). In Figure 5, we compare 1) the task performance of the models against the mean length of the corresponding solution responses (Top), 2) the model scale against the corresponding mean response lengths (Mid), and 3) mean response lengths used for solving TTT-Bench task against the mean response lengths for solving math benchmarks for different models (Bottom). From these plots, we observe that LRMs solve MATH 500 questions with higher accuracy and shorter COT, whereas in the case of both AIME and TTT-Bench questions (especially dTTT, cTTT, & sTTT), models used longer COT with high variability and lower performance. This is counterintuitive, as the question is that TTT-Bench tasks are relatively simple, straightforward, and don’t require as much thinking as is required to solve AIME questions. Consequently, we find that the larger models achieve higher performance using shorter COT across all the benchmarks, where the order of response lengths used for solving different benchmarks is: MATH 500 < TTT-Bench < AIME 2024. A head-to-head comparison between the generated solution response lengths for solving TTT-Bench tasks vs math benchmarks reveal that all the LRMs use longer COT for questions in TTT-Bench tasks against the MATH 500 questions, whereas they generate somewhat similar length COT when compared against AIME questions (especially in the case of sTTT against AIME). These findings suggest that these reasoning models consume similar amount thinking tokens for solving TTT-Bench tasks (even though they are trivial for humans) as used for solving olympiad-level math questions, and are found to produce long COT with circular, inconclusive, and repetitive thinking process for these trivial questions (examples of solution traces are provided in Appendix B), further indicating their inability to do simple, intuitive, and straightforward tasks.

To further understand the failure modes of SOTA LRMs, in Figure 6 we illustrate a few question examples from TTT-Bench and the answers predicted by different models, with their main reasons for those predictions based on the reasoning traces. These examples shed light on the simplicity of our benchmarks as well as how these models reach incorrect or suboptimal TTT-Bench predictions. In the oTTT examples, majority of models predict position "E" as the next move, which although is a decent move to block the white ( ) stones from forming a fork $( " \mathsf { A } - \mathsf { E } - \mathsf { I } ^ { \prime \prime } \& \mathsf { \Omega } ^ { \prime \prime } \mathsf { D } - \mathsf { E } - \mathsf { F } ^ { \prime \prime } )$ , but they fail to identify a better move ("H") that can guarantee a win for the current player by creating a fork for itself $( " mathsf { B } - \mathsf { E } - \mathsf { H } ^ { \prime \prime } \& \ ^ { \prime \prime } \mathsf { G } - \mathsf { H } - \mathrm { I } ^ { \prime \prime } )$ to ensure a win. Again, in the case of dTTT, cTTT & sTTT, the models tend to predict a move that blocks a forking possibility of the opponent, rather than thinking towards a long-term win by placing a move that establishes a fork for itself and ensures a win. These examples demonstrate that most of the time, SOTA LRMs produce suboptimal solutions on TTT-Bench, and struggle at long-term strategic planning towards the optimal outcome.

## 5 Conclusion

In this work, we introduce TTT-Bench, a novel benchmark designed to assess the reasoning capabilities of LRMs on intuitive and low-complexity reasoning tasks, which are effortless for humans to solve from a young age. While these models achieve remarkable success on challenging mathematical benchmarks like MATH 500 & AIME 2024, our evaluations reveal a consistent and surprising shortfall in their performance on simple two-player games, highlighting a fundamental gap in their reasoning capabilities. Through TTT-Bench, which is generated using our proposed automated and verifiable benchmark generation framework, we demonstrate that existing LRMs tend to overthink trivial tasks, exhibit performance degradation with longer outputs, and often miss optimal strategies, even when they possess the underlying capabilities to solve significantly harder problems. This discrepancy underscores the importance of diversifying evaluation protocols for reasoning models to include intuitive and simple benchmarks beyond the conventional STEM-focused paradigms.

## 6 Limitations

While TTT-Bench offers new insights into the reasoning capabilities of LRMs, several limitations must be acknowledged:

Narrow game domain. TTT-Bench focuses exclusively on a specific class of simple two-player board games inspired by Tic-Tac-Toe. Although these games are intuitive and require basic spatial and strategic reasoning, they represent only a small slice of the broader spectrum of intuitive reasoning tasks. As such, the conclusions drawn from TTT-Bench may not directly generalize to other forms of intuitive reasoning, such as commonsense planning, physical reasoning, or visual-spatial puzzles.

Text-only formulation. All TTT-Bench tasks are presented in purely textual form, even though the reasoning involved is highly spatial. While this setup allows for scalable and consistent evaluation using language models, it may introduce unnecessary difficulty due to the lack of visual context. Some model failures might stem from difficulty in mentally visualizing board states rather than limitations in reasoning per se.

Model access and decoding constraints. Some evaluations—especially for closed-source or API models—are limited to a single sample per query, unlike open models where multiple generations are averaged. This discrepancy may affect the comparability of results and lead to under- or overestimation of certain models’ performance.

## References

Yupeng Chang, Xu Wang, Jindong Wang, Yuan Wu, Linyi Yang, Kaijie Zhu, Hao Chen, Xiaoyuan Yi, Cunxiang Wang, Yidong Wang, Wei Ye, Yue Zhang, Yi Chang, Philip S. Yu, Qiang Yang, and Xing Xie. 2023. A survey on evaluation of large language models. Preprint, arXiv:2307.03109.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias

Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. Preprint, arXiv:2110.14168.

Anthony Costarelli, Mat Allen, Roman Hauksson, Grace Sodunke, Suhas Hariharan, Carlson Cheng, Wenjie Li, Joshua Clymer, and Arjun Yadav. 2024. Gamebench: Evaluating strategic reasoning abilities of llm agents. Preprint, arXiv:2406.06613.

DeepSeek-AI. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. Preprint, arXiv:2501.12948.

Jinhao Duan, Renming Zhang, James Diffenderfer, Bhavya Kailkhura, Lichao Sun, Elias Stengel-Eskin, Mohit Bansal, Tianlong Chen, and Kaidi Xu. 2024. Gtbench: Uncovering the strategic reasoning capabilities of llms via game-theoretic evaluations. In Advances in Neural Information Processing Systems, volume 37, pages 28219–28253. Curran Associates, Inc.

Nicoló Fontana, Francesco Pierri, and Luca Maria Aiello. 2024. Nicer than humans: How do large language models behave in the prisoner’s dilemma? Preprint, arXiv:2406.13605.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, and 542 others. 2024. The llama 3 herd of models. Preprint, arXiv:2407.21783.

Xinyu Guan, Li Lyna Zhang, Yifei Liu, Ning Shang, Youran Sun, Yi Zhu, Fan Yang, and Mao Yang. 2025. rstar-math: Small llms can master math reasoning with self-evolved deep thinking. Preprint, arXiv:2501.04519.

Chaoqun He, Renjie Luo, Yuzhuo Bai, Shengding Hu, Zhen Thai, Junhao Shen, Jinyi Hu, Xu Han, Yujie Huang, Yuxiang Zhang, Jie Liu, Lei Qi, Zhiyuan Liu, and Maosong Sun. 2024. OlympiadBench: A challenging benchmark for promoting AGI with olympiad-level bilingual multimodal scientific problems. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3828–3850, Bangkok, Thailand. Association for Computational Linguistics.

Jie Huang and Kevin Chen-Chuan Chang. 2023. Towards reasoning in large language models: A survey. In Findings of the Association for Computational Linguistics: ACL 2023, pages 1049–1065, Toronto, Canada. Association for Computational Linguistics.

Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El-Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutel, Alex Carney, and 1 others. 2024. Openai o1 system card. arXiv preprint arXiv:2412.16720.

Mingyu Jin, Qinkai Yu, Dong Shu, Haiyan Zhao, Wenyue Hua, Yanda Meng, Yongfeng Zhang, and Mengnan Du. 2024. The impact of reasoning step length on large language models. In Findings of the Associationfor Computational Linguistics: ACL 2024, pages 1830–1842, Bangkok, Thailand. Association for Computational Linguistics.

Nathan Lambert, Jacob Morrison, Valentina Pyatkin, Shengyi Huang, Hamish Ivison, Faeze Brahman, Lester James V Miranda, Alisa Liu, Nouha Dziri, Shane Lyu, and 1 others. 2024. Tulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124.

LG AI Research. 2025. Exaone deep: Reasoning enhanced language models. arXiv preprint arXiv:2503.12524.

Zhong-Zhi Li, Duzhen Zhang, Ming-Liang Zhang, Jiaxin Zhang, Zengyan Liu, Yuxuan Yao, Haotian Xu, Junhao Zheng, Pei-Jie Wang, Xiuyi Chen, Yingying Zhang, Fei Yin, Jiahua Dong, Zhiwei Li, Bao-Long Bi, Ling-Rui Mei, Junfeng Fang, Zhijiang Guo, Le Song, and Cheng-Lin Liu. 2025. From system 1 to system 2: A survey of reasoning large language models. Preprint, arXiv:2502.17419.

Hunter Lightman, Vineet Kosaraju, Yuri Burda, Harrison Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. 2024. Let’s verify step by step. In The Twelfth International Conference on Learning Representations.

Bill Yuchen Lin, Ronan Le Bras, Kyle Richardson, Ashish Sabharwal, Radha Poovendran, Peter Clark, and Yejin Choi. 2025. Zebralogic: On the scaling limits of llms for logical reasoning. Preprint, arXiv:2502.01100.

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, Yansong Tang, and Dongmei Zhang. 2025a. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. Preprint, arXiv:2308.09583.

Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Li Erran Li, Raluca Ada Popa, and Ion Stoica. 2025b. Deepscaler: Surpassing o1-preview with a 1.5b model by scaling rl. https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2. Notion Blog.

Mathematical Association of America MAA. 2024. American invitational mathematics examination.

Yujun Mao, Yoon Kim, and Yilun Zhou. 2024. Champ: A competition-level dataset for fine-grained analyses of llms’ mathematical reasoning capabilities. Preprint, arXiv:2401.06961.

Niklas Muennighoff, Zitong Yang, Weijia Shi, Xiang Lisa Li, Li Fei-Fei, Hannaneh Hajishirzi, Luke Zettlemoyer, Percy Liang, Emmanuel Candès, and Tatsunori Hashimoto. 2025. s1: Simple test-time scaling. Preprint, arXiv:2501.19393.

OpenAI. 2025. Openai o3 and o4-mini system card.

Zhanyue Qin, Haochuan Wang, Deyuan Liu, Ziyang Song, Cunhang Fan, Zhao Lv, Jinlin Wu, Zhen Lei, Zhiying Tu, Dianhui Chu, Xiaoyan Yu, and Dianbo Sui. 2024. UNO arena for evaluating sequential decision-making capability of large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 7630–7645, Miami, Florida, USA. Association for Computational Linguistics.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems, volume 36, pages 53728–53741. Curran Associates, Inc.

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, and Samuel R. Bowman. 2024. GPQA: A graduate-level google-proof q&a benchmark. In First Conference on Language Modeling.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. Preprint, arXiv:2402.03300.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2024. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256.

Parshin Shojaee\*†, Iman Mirzadeh\*, Keivan Alizadeh, Maxwell Horton, Samy Bengio, and Mehrdad Farajtabar. 2025. The illusion of thinking: Understanding the strengths and limitations of reasoning models via the lens of problem complexity.

Mirac Suzgun, Nathan Scales, Nathanael Schärli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc Le, Ed Chi, Denny Zhou, and Jason Wei. 2023. Challenging BIG-bench tasks and whether chain-of-thought can solve them. In Findings ofthe Associationfor Computational Linguistics: ACL 2023, pages 13003–13051, Toronto, Canada. Association for Computational Linguistics.

Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, and 1 others. 2025. Kimi k1.5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599.

NovaSky Team. 2025a. Sky-t1: Train your own o1 preview model within \$450. https://novaskyai.github.io/posts/sky-t1. Accessed: 2025-01-09.

OpenThoughts Team. 2025b. Open Thoughts. https://open-thoughts.ai.

Qwen Team. 2025c. Qwq-32b: Embracing the power of reinforcement learning.

Oguzhan Topsakal, Colby Jacob Edell, and Jackson Bailey Harper. 2024. Evaluating large language models with grid-based game competitions: An extensible llm benchmark and leaderboard. Preprint, arXiv:2407.07796.

Oguzhan Topsakal and Jackson B. Harper. 2024. Benchmarking large language model (llm) performance for game playing via tic-tac-toe. Electronics, 13(8).

Jen tse Huang, Eric John Li, Man Ho Lam, Tian Liang, Wenxuan Wang, Youliang Yuan, Wenxiang Jiao, Xing Wang, Zhaopeng Tu, and Michael R. Lyu. 2025. How far are we on the decision-making of llms? evaluating llms’ gaming ability in multi-agent environments. Preprint, arXiv:2403.11807.

Karthik Valmeekam, Kaya Stechly, Atharva Gundawar, and Subbarao Kambhampati. 2024a. Planning in strawberry fields: Evaluating and improving the planning and scheduling capabilities of lrm o1. Preprint, arXiv:2410.02162.

Karthik Valmeekam, Kaya Stechly, and Subbarao Kambhampati. 2024b. Llms still can’t plan; can lrms? a preliminary evaluation of openai’s o1 on planbench. Preprint, arXiv:2409.13373.

Liang Wen, Yunke Cai, Fenrui Xiao, Xin He, Qi An, Zhenyu Duan, Yimin Du, Junchen Liu, Lifu Tang, Xiaowei Lv, Haosheng Zou, Yongchao Deng, Shousheng Jia, and Xiangzheng Zhang. 2025. Light-r1: Curriculum sft, dpo and rl for long cot from scratch and beyond. arXiv preprint arXiv:2503.10460.

Fengli Xu, Qianyue Hao, Zefang Zong, Jingwei Wang, Yunke Zhang, Jingyi Wang, Xiaochong Lan, Jiahui Gong, Tianjian Ouyang, Fanjin Meng, Chenyang Shao, Yuwei Yan, Qinglong Yang, Yiwen Song, Sijian Ren, Xinyuan Hu, Yu Li, Jie Feng, Chen Gao, and Yong Li. 2025. Towards large reasoning models: A survey of reinforced reasoning with large language models. Preprint, arXiv:2501.09686.

Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, and Bill Yuchen Lin. 2024. Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing. Preprint, arXiv:2406.08464.

Yixin Ye, Zhen Huang, Yang Xiao, Ethan Chern, Shijie Xia, and Pengfei Liu. 2025. Limo: Less is more for reasoning. Preprint, arXiv:2502.03387.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Tiantian Fan, Gaohong Liu, Lingjun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, Wang Zhang, Hang Zhu, and 16 others. 2025. Dapo: An open-source llm reinforcement learning system at scale. Preprint, arXiv:2503.14476.

Dan Zhang, Sining Zhoubian, Ziniu Hu, Yisong Yue, Yuxiao Dong, and Jie Tang. 2024a. rest mcts : Llm self-training via process reward guided tree search. In Advances in Neural Information Processing Systems, volume 37, pages 64735–64772. Curran Associates, Inc.

Jiaxin Zhang, Zhong-Zhi Li, Ming-Liang Zhang, Fei Yin, Cheng-Lin Liu, and Yashar Moshfeghi. 2024b. GeoEval: Benchmark for evaluating LLMs and multimodal models on geometry problem-solving. In Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 1258–1276, Bangkok, Thailand. Association for Computational Linguistics.

Yu Zhao, Huifeng Yin, Bo Zeng, Hao Wang, Tianqi Shi, Chenyang Lyu, Longyue Wang, Weihua Luo, and Kaifu Zhang. 2024. Marco-o1: Towards open reasoning models for open-ended solutions. Preprint, arXiv:2411.14405.

## A TTT-Bench Examples

Here we provide the questions from TT-Bench and the solution illustrations for the examples in Figure 1.   
For the solutions, we also provide the verdict and the states contributing to the corresponding verdict.

## oTTT test sample example in Figure 1

## Question:

Alice and Bob are playing a game on a 3x3 grid. The points on the grid are labeled top to bottom, left to right, as A,B,C,D,E,F,G,H,I. Alice plays white. Bob plays black. At each turn, the player places a stone of the corresponding color onto one of the positions that has not been occupied. Whoever has three stones in a line (horizontal, vertical, or diagonal) wins. Alice first places a white stone at A. Bob places a black stone at B. Alice places a white stone at H. Bob places a black stone at C. Where should Alice play next? Game Visualization:

Board Positions:

<table><tr><td rowspan=1 colspan=1>A</td><td rowspan=1 colspan=1>B</td><td rowspan=1 colspan=1>C</td></tr><tr><td rowspan=1 colspan=1>D</td><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>F</td></tr><tr><td rowspan=1 colspan=1>G</td><td rowspan=1 colspan=1>H</td><td rowspan=1 colspan=1>I</td></tr></table>

## Current

Board:

![](images/d110000858a94900c14c0486c056a0fac40b09bbe73a1cf040c594af65766f86.jpg)

## Solution:

G (Fork with A-D-G & G-H-I) or I (Fork with A-E-I & G-H-I)

## dTTT test sample example in Figure 1

## Question:

Alice and Bob are playing a game on two adjacent 3x3 grids. The points on the first grid are labeled top to bottom, left to right, as A,B,C,D,E,F,G,H,I. The points on the second grid are labeled top to bottom, left to right, as C,J,K,F,L,M,I,N,O. Note that points C,F, I are shared by the two grids. Alice plays white. Bob plays black. At each turn, the player places a stone of the corresponding color onto one of the positions that has not been occupied. Whoever has three stones in a line (horizontal, vertical, or diagonal) on either grid wins. Alice first places a white stone at I. Bob places a black stone at J. Then Alice at F. Bob places a black stone at L. Then Alice at B. Bob places a black stone at H. Where should Alice play next?

Game Visualization:

Board

Positions:

<table><tr><td rowspan=1 colspan=6>Grid1          Grid2..</td></tr><tr><td></td><td rowspan=1 colspan=1>A</td><td rowspan=1 colspan=1>BD</td><td rowspan=1 colspan=1>C</td><td rowspan=1 colspan=1>J</td><td rowspan=1 colspan=1>K</td></tr><tr><td></td><td rowspan=1 colspan=1>D</td><td rowspan=1 colspan=1>EO</td><td rowspan=1 colspan=1>F</td><td rowspan=1 colspan=1>L</td><td rowspan=1 colspan=1>M</td></tr><tr><td></td><td rowspan=1 colspan=1>G</td><td rowspan=1 colspan=1>HOD</td><td rowspan=1 colspan=1>I</td><td rowspan=1 colspan=1>iN</td><td rowspan=1 colspan=1>o</td></tr></table>

Current

Board:

![](images/7d46319bd22d1ec1202e45c44cee09ad0023ed88ca86bed9bc3a729c2a8ef1f5.jpg)

Solution:

C (Win with C-F-I)

## cTTT test sample example in Figure 1

## Question:

Alice and Bob are playing a game on two adjacent cubes. ABCD forms the top rectangle in the first cube, and BIJC forms the top rectangle in the second cube. EFGH forms the bottom rectangle in the first cube, and FKLG forms the bottom rectangle in the second cube. AE is an edge, BF is an edge, CG is an edge, DH is an edge, IK is an edge, and JL is an edge. Note that vertices B,C,G,F are shared by the two cubes. Alice and Bob play a game where they take turns to put stickers on the vertices of the cubes that have not been occupied. Alice plays white stickers. Bob plays black stickers. The person who has four stickers on the same plane on either cube wins. Alice first places a white stone at A. Then Bob places a black stone at B. Then Alice places a white stone at C. Then Bob places a black stone at I. Then Alice places a white stone at E. Then Bob places a black stone at K. Where should Alice play next? Game Visualization:

Board Positions:

![](images/14d858fa3ad5c52829287f89ab7fcaa7b2abb5f03da34a3482d41ab5596d96e2.jpg)

Current

Board:

![](images/dbfd20ee38c5ba115270f15737da6d032e88ab7a5967aba73fdd1e20c7623c42.jpg)

## Solution:

G (Win with A-C-G-E)

Question:   
Alice and Bob are playing a game on a board. There are 5 equally spaced   
horizontal lines where the distance between two neighboring horizontal lines   
is 1. Similarly, there are 5 equally spaced vertical lines, and the distance   
between two neighboring vertical lines is 1. There are 25 intersection points   
between the 5 horizontal lines and 5 vertical lines. These 25 points are labeled   
from top to bottom, left to right, as A, B, C, D, . . . , Y. Alice plays white.   
Bob plays black. At each turn, the player places a stone of the corresponding   
color onto one of the 25 points that have not been occupied. Whoever has four   
stones that form either a unit square (with side length of 1) or a “diagonal   
square” with side length equal to the square root of 2 wins. For example, ABGF   
is a unit square. FBHL is a diagonal square. Alice first places a white stone   
at A. Then Bob places a black stone at D. Then Alice places a white stone at B.   
Then Bob places a black stone at K. Then Alice places a white stone at H. Then   
Bob places a black stone at U. Where should Alice play next?   
Game Visualization:   
Board   
Positions:   
A B C D E   
F G H I J   
K L M N o   
P Q R s T   
U V W x Y   
Current   
Board:   
O O   
V O   
Solution:   
F (Fork with A-B-F-G & F-B-H-L) or G (Fork with A-B-G-F & A-C-H-G)

![](images/da7ca5facf99a3b70329e71434cab72494e816bac305e845b037627a3bfff4e7.jpg)

![](images/191ca6f40277b897fad7b6849070b6aab513847257aefcf196e464b66ac91644.jpg)

## B Solution Trace Examples

Here we provide the reasoning traces produced by different models for question examples in Figure 1.

LIMO Solution Trace for oTTT question in Figure 6 (Correct)   
Question:   
Alice and Bob are playing a game on a 3x3 grid. The points on the grid are   
labeled top to bottom, left to right, as A,B,C,D,E,F,G,H,I. Alice plays white.   
Bob plays black. At each turn, the player places a stone of the corresponding   
color onto one of the positions that has not been occupied. Whoever has three   
stones in a line (horizontal, vertical, or diagonal) wins. Alice first places   
a white stone at A. Bob places a black stone at B. Alice places a white stone   
at C. Bob places a black stone at G. Alice places a white stone at D. Where   
should Bob play next? Let’s think step by step and output the final answer   
within \boxed{}.   
Correct Answer: H (Forms fork with B-E-H & G-H-I)   
Game Visualization:

Positions:

<table><tr><td rowspan=1 colspan=1>A</td><td rowspan=1 colspan=1>B</td><td rowspan=1 colspan=1>C</td></tr><tr><td rowspan=1 colspan=1>D</td><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>F</td></tr><tr><td rowspan=1 colspan=1>G</td><td rowspan=1 colspan=1>H</td><td rowspan=1 colspan=1>I</td></tr></table>

![](images/bd684c067c7c6cc6c5a37054251828450dbf5e60e3f12b9f76cf472aa6ecdad5.jpg)

## Solution:

<think>

## </think>

Therefore, if Bob plays at H, Alice must play at I to prevent Bob from winning. Then, Bob plays at F, leading to a draw. If Alice makes a mistake and plays at F, Bob wins. Therefore, playing at H is a good move because it puts pressure on Alice to play correctly. If she doesn’t, Bob wins. If she does, the game is a draw.

Therefore, considering all options, Bob should play at H. This creates a threat that Alice must respond to, and if she responds incorrectly, Bob can win. If she responds correctly, the game is a draw, which is still a better outcome than losing.

Therefore, the best move for Bob is to play at H.

\*\*Final Answer\*\*

\boxed{H}

## QwQ-32B Solution Trace for dTTT question in Figure 6 (Wrong)

## Question:

Alice and Bob are playing a game on two adjacent 3x3 grid. The points on the first grid are labeled top to bottom, left to right, as A,B,C,D,E,F,G,H,I. The points on the second grid are labeled top to bottom, left to right, as C,J,K,F,L,M,I,N,O. Note that points C,F, I are shared by the two grids. Alice plays white. Bob plays black. At each turn, the player places a stone of the corresponding color onto one of the positions that has not been occupied. Whoever has three stones in a line (horizontal, vertical, or diagonal) on either grid wins. Alice first places a white stone at I. Bob places a black stone at G. Then Alice at J. Bob places a black stone at K. Then Alice at D. Bob places a black stone at A. Then Alice at M. Where should Bob play next? Let’s think step by step and output the final answer within \boxed{}.

## Correct Answer: C (Forms fork with A-B-C & G-E-C)

Game Visualization:

Board

Positions:

<table><tr><td rowspan=1 colspan=1>A</td><td rowspan=1 colspan=1>B</td><td rowspan=1 colspan=1>C</td><td rowspan=1 colspan=1>J</td><td rowspan=1 colspan=1>K</td></tr><tr><td rowspan=1 colspan=1>D</td><td rowspan=1 colspan=1>E</td><td rowspan=1 colspan=1>F</td><td rowspan=1 colspan=1>L</td><td rowspan=1 colspan=1>M</td></tr><tr><td rowspan=1 colspan=1>G</td><td rowspan=1 colspan=1>HDDODL</td><td rowspan=1 colspan=1>I</td><td rowspan=1 colspan=1>N</td><td rowspan=1 colspan=1>o</td></tr></table>

Current

Board:

![](images/68be3dae137ad88a03c9aa6c372290199adfd3763c54517355b1bb0ff0c34e4e.jpg)

## Solution:

<think>

</think>

Alice and Bob are playing a game on two adjacent 3x3 grids. The key steps and reasoning are as follows:

1. \*\*Current Positions\*\*:

\- \*\*Grid1\*\*:

\- Alice: I (bottom-right), D (middle-left)

\- Bob: G (bottom-left), A (top-left)

\- \*\*Grid2\*\*:

\- Alice: J (top-middle), M (middle-right), I (bottom-left)

\- Bob: K (top-right)

2. \*\*Alice’s Potential Threats\*\*:

\- In \*\*Grid1\*\*, Alice has D and I. To form a line, she needs a third stone in a row, column, or diagonal.

\- In \*\*Grid2\*\*, Alice has J, M, and I. These positions do not form a line yet, but she could aim for a diagonal or column.

3. \*\*Critical Blocking Moves\*\*:

\- \*\*Grid1’s Column3 (C, F, I)\*\*: Alice has I. If she takes C and F, she can form a vertical line. Bob can block this by taking F.

\- \*\*Grid1’s Row2 (D, E, F)\*\*: Alice has D. Taking F would block her from completing this row.

\- \*\*Grid2’s Diagonal (K-L-I)\*\*: If Alice takes L, she could form a diagonal with I and L. However, Bob’s K blocks this.

4. \*\*Strategic Considerations\*\*:

\- Taking F in \*\*Grid1\*\* blocks both the vertical column3 and the horizontal row2 threats.

```latex
Taking F also affects **Grid2’s column1**, preventing Alice from forming a
vertical line there.
After evaluating all possibilities, the optimal move for Bob is to place a stone
at **F**, which blocks Alice’s potential threats in both grids effectively.
\boxed{F}
```

![](images/2c5bcf8939e49f62ba469d835133b0761b7a5964c61eb194ae8e4460cf3d9794.jpg)

DeepSeek-R1-Distill-Qwen-32B Solution Trace for cTTT question in Figure 6   
(Wrong)   
Question:   
Alice and Bob are playing a game on two adjacent cubes. ABCD forms the top   
rectangle in the first cube and BIJC forms the top rectangle in the second cube.   
EFGH forms the bottom rectangle in the first cube and FKLG forms the bottom   
rectangle in the second cube. AE is an edge, BF is an edge, CG is an edge, DH   
is an edge, IK is an edge, and JL is an edge. Note that vertices B,C,G,F are   
shared by the two cubes. Alice and Bob plays a game where they take turns to put   
stickers on the vertices of the cubes that have not been occupied. Alice plays   
white stickers. Bob plays black stickers. The person who has four stickers on   
the same plane on either cube wins. Alice first places a white stone at B. Then   
Bob places a black stone at A. Then Alice places a white stone at G. Then Bob   
places a black stone at C. Then Alice places a white stone at J. Then Bob places   
a black stone at E. Then Alice places a white stone at K. Where should Bob play   
next? Let’s think step by step and output the final answer within \boxed{}.   
Correct Answer: D (Forms fork with A-D-H-E & D-C-E-F)   
Game Visualization:   
Board   
Positions:   
D C J   
A B I   
H G L   
E F K   
Current   
Board:   
  
Block opponent fork   
Solution:   
<think>   
</think>

![](images/9e9f788856cd22b8721b22d2d622e20db81da680b5990b3b82fe410a81740dae.jpg)

Alice and Bob are playing a game on two adjacent cubes. The goal is to place stickers on vertices such that four stickers of the same color form a plane on either cube. Alice goes first and places white stickers, while Bob places black stickers. The current moves are as follows:

1. Alice places a white sticker at B.

2. Bob places a black sticker at A.

3. Alice places a white sticker at G.

4. Bob places a black sticker at C.

5. Alice places a white sticker at J.

6. Bob places a black sticker at E.

7. Alice places a white sticker at K.

The vertices involved are A, B, C, D, E, F, G, H, I, J, K, L. The shared vertices are B, C, F, G. Alice’s current stickers are at B, G, J, K. Bob’s current stickers are at A, C, E. The remaining vertices are D, H, I, L, F. To determine where Bob should place his next sticker, we need to consider the potential planes where Alice could win and block her. Alice is threatening multiple planes, but Bob can only block one.

After analyzing the potential planes and considering the current state of the game, Bob should place his next sticker at L to block Alice’s potential win on the back face (C, G, L, J) and the bottom face (F, K, L, G). This move also blocks Alice +from completing these planes, preventing her from winning on her next turn.

Thus, the final answer is:

\boxed{L}

## QwQ-32B Solution Trace for sTTT question in Figure 6 (Wrong)

## Question:

Alice and Bob are playing a game on a board. There are 5 equally spaced horizontal lines where the distance between two neighboring horizontal lines is 1. Similarly, there are 5 equally spaced vertical lines, and the distance between two neighboring vertical lines is 1. There are 25 intersection points between the 5 horizontal lines and 5 vertical lines. These 25 points are labeled from top to bottom, left to right, as A, B, C, D, . . . , Y. Alice plays white. Bob plays black. At each turn, the player places a stone of the corresponding color onto one of the 25 points that have not been occupied. Whoever has four stones that form either a unit square (with side length of 1) or a “diagonal square” with side length equal to the square root of 2 wins. For example, ABGF is a unit square. FBHL is a diagonal square. Alice first places a white stone at A. Then Bob places a black stone at E. Then Alice places a white stone at B. Then Bob places a black stone at I. Then Alice places a white stone at C. Then Bob places a black stone at O. Then Alice places a white stone at W. Where should Bob play next? Let’s think step by step and output the final answer within \boxed{}.

Correct Answer: J (Forms fork with D-E-J-I & I-J-O-N)

Game Visualization:

3. \*\*Threat analysis\*\*: Alice’s top-row stones (A, B, C) could form part of a   
unit square if combined with stones below. The critical points for such squares   
include G(1,1) and H(1,2).

Positions:

<table><tr><td rowspan=1 colspan=1>A</td><td rowspan=1 colspan=1>B</td><td rowspan=1 colspan=1>C</td><td rowspan=1 colspan=1>D</td><td rowspan=1 colspan=1>E</td></tr><tr><td rowspan=1 colspan=1>F</td><td rowspan=1 colspan=1>G</td><td rowspan=1 colspan=1>H</td><td rowspan=1 colspan=1>I</td><td rowspan=1 colspan=1>J</td></tr><tr><td rowspan=1 colspan=1>K</td><td rowspan=1 colspan=1>L</td><td rowspan=1 colspan=1>M</td><td rowspan=1 colspan=1>N</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>P</td><td rowspan=1 colspan=1>Q</td><td rowspan=1 colspan=1>R</td><td rowspan=1 colspan=1>S</td><td rowspan=1 colspan=1>T</td></tr><tr><td rowspan=1 colspan=1>U</td><td rowspan=1 colspan=1>V</td><td rowspan=1 colspan=1>W</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1>Y</td></tr></table>

Current

Board:

![](images/6211a2e699e4cfe320f1578ddda5d89ba9d674dbb95481594c1e68f325e98957.jpg)

Block opponent fork

## Solution:

<think>

</think>

Alice and Bob are playing a game on a 5x5 grid where the goal is to form a unit   
square (side length 1) or a diagonal square (side length square root 2) with   
four stones. Alice’s moves so far are at A, B, C, and W, while Bob’s moves are   
at E, I, and O.

Key considerations:

4. \*\*Blocking potential squares\*\*: The unit square involving A, B, and needing F(1,0) and G(1,1) is a key threat. Since Alice has A and B, blocking G(1,1) would prevent her from completing this square in future moves.

## C Algorithm for generating TTT-Bench solutions

On the next page, we provide the detailed algorithm used for the automated generation of TTT-Bench questions and their corresponding solutions.

```julia
Algorithm 1: Algorithm for generating TTT-Bench solutions
Total number of positions: T, Number of moves played: $N = n _ { c } + n _ { o } .$ , Number of positions in winning lines: $n ,$
Number of winning lines: t, Current player positions: $P ^ { c } : \{ p _ { 1 } ^ { c } , . . . , p _ { n _ { c } } ^ { c } \}$ , Opponent positions: $P ^ { o } : \{ p _ { 1 } ^ { \breve { o } } , . . . , p _ { n _ { o } } ^ { o } \}$
Available Position: $A : \left\{ a _ { 1 } , . . . , a _ { T - N } \right\}$ , Wining States: $W : \{ ( w _ { 1 } ^ { 1 } , . . . , x _ { n } ^ { 1 } ) , . . . ( w _ { 1 } ^ { t } , . . . , x _ { n } ^ { t } ) \}$
Function check_won $( P _ { n e w } ^ { c } ) \colon$
for $( w _ { 1 } ^ { i } , . . . , x _ { n } ^ { i } )$ in W do
i $\mathbb { \tilde { \Gamma } } \mathbb { \breve { \times } } ( w _ { j } ^ { i } \in \underline { { P _ { n e w } ^ { c } } } | w _ { j } ^ { i } \in ( w _ { 1 } ^ { i } , . . . , x _ { n } ^ { i } ) )$ then
return True;
end
end
return False;
Function get_wins $( P ^ { c } , A ) { : }$
w_moves $ \{ \}$
for a in A do
$P _ { n e w } ^ { c } \gets P ^ { c } + a _ { i }$
if check_won $( P _ { n e w } ^ { c } )$ then
w_moves w_moves $+ \ a _ { i }$
end
end
return w_moves;
Function get_forks $( P ^ { c } , A )$ :
f_moves $ \{ \}$
for a in A do
$P _ { n e w } ^ { c }  P ^ { c } + a _ { i } , .$ A<sup>potential</sup> $ a _ { j } \in A ; a _ { j } \neq a _ { i } .$ , threat_count $ 0$
for $( w _ { 1 } ^ { i } , . . . , x _ { n } ^ { i } )$ in W do
W<sup>current</sup> $ w _ { j } ^ { i } | w _ { j } ^ { i } \in ( w _ { 1 } ^ { i } , . . . , x _ { n } ^ { i } )$ & $w _ { j } ^ { i } \in P _ { n e } ^ { c }$ w
$W ^ { o p p o n e n t }  w _ { j } ^ { i } | w _ { j } ^ { i } \in ( w _ { 1 } ^ { i } , . . . , x _ { n } ^ { i } ) \& w _ { j } ^ { i } \notin P _ { n e w } ^ { c } \& w _ { j } ^ { i } \notin A ^ { i }$ potential
if len(W<sup>current</sup>) $) = = n - 1$ & len(W<sup>opponent</sup>) == 0 then
threat_count threat_count + 1
end
end
if threat_count $> = 2$ then
f_moves $ f _ { - }$ moves $+ a _ { i }$
end
end
return f_moves;
Function get_solution $( P ^ { c } , P ^ { o } , A ) \colon$
w_moves get_wins $( P ^ { c } , \dot { A } )$ ▷ Check if current player has wining moves.
if len(w_moves) $> = 1$ then
return w_moves,"Win";
end
opp_w_moves get_wins $( P ^ { o } , A )$ ▷ Check if current player has blocking moves.
if len(opp_w_moves) == 1 then
return opp_w_moves,"Blocked";
end
else
return None;
end
f_moves get_forks $( P ^ { c } , A )$ ▷ Check if current player has forking moves.
if len(f_moves) >= 1 then
return f_moves,"Fork";
end
```

## D Performance Results

In Table 3 and Table 4 we report the Pass@1 performance numbers used in Figure 3 and Figure 4 respectively.

<table><tr><td>Models</td><td>oTTT</td><td>dTTT</td><td>cTTT</td><td>sTTT</td><td>AIME 2024</td><td>MATH 500</td></tr><tr><td>QwQ-32B</td><td>92.03</td><td>78.13</td><td>60.14</td><td>60.10</td><td>79.17</td><td>95.00</td></tr><tr><td>QwQ-32B-Preview</td><td>66.73</td><td>60.13</td><td>50.49</td><td>50.16</td><td>47.08</td><td>90.16</td></tr><tr><td>EXAONE-Deep-32B</td><td>89.09</td><td>71.25</td><td>51.46</td><td>60.42</td><td>70.00</td><td>94.95</td></tr><tr><td>OpenThinker2-32B</td><td>88.85</td><td>73.00</td><td>56.60</td><td>66.30</td><td>74.17</td><td>94.53</td></tr><tr><td>DeepSeek-R1-Distill-Qwen-32B</td><td>70.77</td><td>51.19</td><td>42.36</td><td>55.31</td><td>69.58</td><td>93.28</td></tr><tr><td>s1.1-32B</td><td>83.76</td><td>71.81</td><td>57.22</td><td>62.86</td><td>60.63</td><td>92.46</td></tr><tr><td>Light-R1-32B-DS</td><td>81.80</td><td>71.56</td><td>56.46</td><td>67.03</td><td>77.92</td><td>94.69</td></tr><tr><td>LIMO</td><td>78.55</td><td>64.31</td><td>50.83</td><td>55.89</td><td>58.33</td><td>91.91</td></tr><tr><td>OpenThinker-32B</td><td>82.66</td><td>66.56</td><td>52.71</td><td>59.69</td><td>68.33</td><td>93.83</td></tr><tr><td>Sky-T1-32B-Preview</td><td>59.87</td><td>48.69</td><td>42.57</td><td>42.86</td><td>33.13</td><td>86.75</td></tr><tr><td>DeepSeek-R1-Distill-Qwen-14B</td><td>68.81</td><td>50.94</td><td>45.00</td><td>52.66</td><td>66.88</td><td>93.45</td></tr><tr><td>Light-R1-14B-DS</td><td>78.92</td><td>61.50</td><td>50.14</td><td>59.06</td><td>75.42</td><td>94.04</td></tr><tr><td>s1.1-14B</td><td>64.77</td><td>51.69</td><td>43.89</td><td>42.50</td><td>38.96</td><td>88.38</td></tr><tr><td>OpenThinker2-7B</td><td>63.48</td><td>49.75</td><td>49.10</td><td>47.66</td><td>57.08</td><td>92.16</td></tr><tr><td>EXAONE-Deep-7.8B</td><td>86.64</td><td>67.94</td><td>52.99</td><td>64.64</td><td>68.96</td><td>94.54</td></tr><tr><td>AceMath-RL-Nemotron-7B</td><td>67.40</td><td>44.94</td><td>43.89</td><td>48.65</td><td>69.17</td><td>93.76</td></tr><tr><td>DeepSeek-R1-Distill-Qwen-7B</td><td>45.22</td><td>29.25</td><td>31.88</td><td>47.66</td><td>54.38</td><td>92.05</td></tr><tr><td>OpenThinker-7B</td><td>35.66</td><td>24.63</td><td>27.01</td><td>22.40</td><td>31.46</td><td>85.90</td></tr><tr><td>Light-R1-7B-DS</td><td>50.31</td><td>29.81</td><td>29.03</td><td>24.48</td><td>58.33</td><td>92.18</td></tr><tr><td>s1.1-7B</td><td>39.95</td><td>19.19</td><td>25.00</td><td>11.35</td><td>17.29</td><td>79.45</td></tr><tr><td>EXAONE-Deep-2.4B</td><td>76.84</td><td>44.50</td><td>40.35</td><td>45.68</td><td>52.29</td><td>91.60</td></tr><tr><td>s1.1-3B</td><td>9.68</td><td>5.63</td><td>6.53</td><td>1.20</td><td>5.21</td><td>61.81</td></tr><tr><td>DeepSeek-R1-Distill-Qwen-1.5B</td><td>22.92</td><td>10.06</td><td>18.19</td><td>3.49</td><td>27.50</td><td>82.58</td></tr><tr><td>DeepScaleR-1.5B-Preview</td><td>23.04</td><td>16.50</td><td>22.99</td><td>8.18</td><td>40.63</td><td>87.43</td></tr><tr><td>s1.1-1.5B</td><td>4.47</td><td>2.69</td><td>3.47</td><td>0.78</td><td>1.67</td><td>41.10</td></tr><tr><td>STILL-3-1.5B-preview</td><td>24.51</td><td>12.25</td><td>19.79</td><td>3.18</td><td>30.63</td><td>84.59</td></tr></table>

Table 3: Pass@1 performance of LRMs on TTT-Bench, AIME 2024, and MATH500.

<table><tr><td rowspan="2"></td><td colspan="3">oTTT</td><td colspan="3">dTTT</td><td colspan="3">cTTT</td><td colspan="3">sTTT</td></tr><tr><td>Models Win</td><td>Blocked</td><td>Fork</td><td>Win</td><td>Blocked</td><td>Fork</td><td>Win</td><td>Blocked</td><td>Fork</td><td>Win</td><td>Blocked</td><td>Fork</td></tr><tr><td>QwQ-32B</td><td>98.12</td><td>95.77</td><td>82.39</td><td>96.09</td><td>95.62</td><td>51.41</td><td>63.12</td><td>72.29</td><td>49.53</td><td>75.62</td><td>91.25</td><td>36.77</td></tr><tr><td>QwQ-32B-Preview</td><td>85.31</td><td>53.12</td><td>53.69</td><td>83.12</td><td>49.69</td><td>42.34</td><td>44.69</td><td>51.25</td><td>52.81</td><td>77.08</td><td>53.33</td><td>35.1</td></tr><tr><td>EXAONE-Deep-32B</td><td>99.84</td><td>94.3</td><td>67.61</td><td>95.62</td><td>90.31</td><td>37.34</td><td>42.19</td><td>66.46</td><td>44.84</td><td>96.46</td><td>66.46</td><td>39.38</td></tr><tr><td>OpenThinker2-32B</td><td>99.38</td><td>95.77</td><td>66.19</td><td>95.47</td><td>95.31</td><td>39.38</td><td>49.69</td><td>68.75</td><td>50.94</td><td>92.29</td><td>86.25</td><td>43.33</td></tr><tr><td>DeepSeek-R1-Distill-Qwen-32B</td><td>87.5</td><td>67.46</td><td>51.99</td><td>70.94</td><td>55.31</td><td>29.38</td><td>43.75</td><td>45.0</td><td>39.69</td><td>87.92</td><td>69.79</td><td>31.77</td></tr><tr><td>s1.1-32B</td><td>97.03</td><td>82.72</td><td>65.62</td><td>94.38</td><td>83.12</td><td>43.59</td><td>52.19</td><td>63.96</td><td>54.69</td><td>86.88</td><td>78.12</td><td>43.23</td></tr><tr><td>Light-R1-32B-DS</td><td>95.78</td><td>81.43</td><td>61.93</td><td>95.0</td><td>85.62</td><td>41.09</td><td>55.62</td><td>66.25</td><td>49.53</td><td>88.75</td><td>89.38</td><td>45.0</td></tr><tr><td>LIMO</td><td>93.75</td><td>75.0</td><td>61.36</td><td>86.25</td><td>63.75</td><td>42.66</td><td>36.56</td><td>59.17</td><td>51.72</td><td>82.71</td><td>71.46</td><td>34.69</td></tr><tr><td>OpenThinker-32B</td><td>95.47</td><td>87.5</td><td>58.81</td><td>92.34</td><td>82.81</td><td>32.66</td><td>42.5</td><td>66.25</td><td>47.66</td><td>90.21</td><td>75.42</td><td>36.56</td></tr><tr><td>Sky-T1-32B-Preview</td><td>76.88</td><td>46.32</td><td>51.42</td><td>65.47</td><td>51.25</td><td>30.63</td><td>33.12</td><td>44.79</td><td>45.62</td><td>62.71</td><td>44.79</td><td>31.98</td></tr><tr><td>DeepSeek-R1-Distill-Qwen-14B</td><td>85.0</td><td>65.26</td><td>49.72</td><td>75.31</td><td>52.19</td><td>25.94</td><td>29.69</td><td>51.46</td><td>47.81</td><td>82.5</td><td>62.92</td><td>32.6</td></tr><tr><td>Light-R1-14B-DS</td><td>94.06</td><td>78.86</td><td>59.38</td><td>91.56</td><td>70.62</td><td>26.88</td><td>42.19</td><td>56.46</td><td>49.38</td><td>91.25</td><td>73.96</td><td>35.52</td></tr><tr><td>s1.1-14B</td><td>83.44</td><td>60.48</td><td>42.33</td><td>76.88</td><td>46.56</td><td>29.06</td><td>26.25</td><td>49.58</td><td>48.44</td><td>65.0</td><td>51.04</td><td>26.98</td></tr><tr><td>OpenThinker2-7B</td><td>83.28</td><td>57.17</td><td>41.48</td><td>73.75</td><td>35.0</td><td>33.12</td><td>31.87</td><td>54.17</td><td>53.91</td><td>77.29</td><td>47.92</td><td>32.71</td></tr><tr><td>EXAONE-Deep-7.8B</td><td>96.09</td><td>94.85</td><td>64.2</td><td>90.0</td><td>77.19</td><td>41.25</td><td>38.75</td><td>57.92</td><td>56.41</td><td>95.0</td><td>86.25</td><td>38.65</td></tr><tr><td>AceMath-RL-Nemotron-7B</td><td>87.5</td><td>59.93</td><td>47.16</td><td>71.09</td><td>36.25</td><td>23.12</td><td>44.06</td><td>38.12</td><td>48.12</td><td>82.92</td><td>65.0</td><td>23.33</td></tr><tr><td>DeepSeek-R1-Distill-Qwen-7B</td><td>57.66</td><td>34.56</td><td>40.91</td><td>44.38</td><td>19.38</td><td>19.06</td><td>27.81</td><td>30.0</td><td>35.31</td><td>57.92</td><td>28.96</td><td>14.17</td></tr><tr><td>OpenThinker-7B</td><td>43.44</td><td>31.62</td><td>30.4</td><td>35.47</td><td>15.31</td><td>18.44</td><td>17.81</td><td>21.25</td><td>35.94</td><td>31.46</td><td>21.04</td><td>18.54</td></tr><tr><td>Light-R1-7B-DS</td><td>71.72</td><td>36.76</td><td>35.51</td><td>47.34</td><td>21.25</td><td>16.56</td><td>24.06</td><td>27.92</td><td>32.34</td><td>52.71</td><td>24.17</td><td>10.52</td></tr><tr><td>s1.1-7B</td><td>50.16</td><td>36.58</td><td>29.83</td><td>25.16</td><td>15.0</td><td>15.31</td><td>17.19</td><td>21.04</td><td>31.87</td><td>15.83</td><td>9.58</td><td>10.0</td></tr><tr><td>EXAONE-Deep-2.4B</td><td>77.97</td><td>97.79</td><td>53.41</td><td>57.5</td><td>56.25</td><td>25.62</td><td>24.06</td><td>53.33</td><td>38.75</td><td>54.58</td><td>86.67</td><td>20.73</td></tr><tr><td>s1.1-3B</td><td>9.53</td><td>7.72</td><td>13.07</td><td>5.62</td><td>3.12</td><td>6.88</td><td>5.31</td><td>2.92</td><td>9.84</td><td>1.46</td><td>0.62</td><td>1.35</td></tr><tr><td>DeepSeek-R1-Distill-Qwen-1.5B</td><td>28.59</td><td>18.01</td><td>21.31</td><td>10.0</td><td>6.88</td><td>11.72</td><td>15.0</td><td>14.37</td><td>22.66</td><td>5.83</td><td>3.96</td><td>2.08</td></tr><tr><td>DeepScaleR-1.5B-Preview</td><td>40.62</td><td>8.46</td><td>15.62</td><td>19.69</td><td>11.56</td><td>15.78</td><td>22.19</td><td>17.08</td><td>27.81</td><td>14.17</td><td>6.04</td><td>6.25</td></tr><tr><td>s1.1-1.5B</td><td>3.44</td><td>4.41</td><td>6.25</td><td>3.28</td><td>0.62</td><td>3.12</td><td>12.81</td><td>13.12</td><td>28.28</td><td>1.04</td><td>1.46</td><td>0.31</td></tr><tr><td>STILL-3-1.5B-preview</td><td>31.09</td><td>16.54</td><td>24.43</td><td>15.00</td><td>7.19</td><td>12.03</td><td>12.81</td><td>13.12</td><td>28.28</td><td>6.04</td><td>1.88</td><td>2.40</td></tr></table>

Table 4: Pass@1 performance of LRMs over individual solution verdict category ("Win", "Blocked", & "Fork") questions in TTT-Bench tasks.