# Value Profiles for Encoding Human Variation

Taylor Sorensen<sup>1</sup>†, Pushkar Mishra<sup>2</sup>, Roma Patel<sup>2</sup>, Michael Henry Tessler<sup>2</sup>, Michiel Bakker<sup>2</sup>, Georgina Evans<sup>2</sup>, Iason Gabriel<sup>2</sup>, Noah Goodman<sup>2</sup>, Verena Rieser<sup>2</sup>

<sup>1</sup>Department of Computer Science, University of Washington, Seattle, WA, USA

<sup>2</sup>Google DeepMind, London, UK

†Work done during an internship at Google DeepMind. tsor13@cs.washington.edu, verenarieser@deepmind.com

## Abstract

Modelling human variation in rating tasks is crucial for personalization, pluralistic model alignment, and computational social science. We propose representing individuals using natural language value profiles – descriptions of underlying values compressed from in-context demonstrations – along with a steerable decoder model that estimates individual ratings from a rater representation. To measure the predictive information in a rater representation, we introduce an information-theoretic methodology and find that demonstrations contain the most information, followed by value profiles, then demographics. However, value profiles effectively compress the useful information from demonstrations (>70% information preservation) and offer advantages in terms of scrutability, interpretability, and steerability. Furthermore, clustering value profiles to identify similarly behaving individuals better explains rater variation than the most predictive demographic groupings. Going beyond test set performance, we show that the decoder predictions change in line with semantic profile differences, are wellcalibrated, and can help explain instance-level disagreement by simulating an annotator population. These results demonstrate that value profiles offer novel, predictive ways to describe individual variation beyond demographics or group information.

## 1 Introduction

Machine learning systems are traditionally trained to approximate a single “ground truth" label, treating annotator disagreement as noise. However, many important tasks such as chat preferences, hate speech, and toxicity detection can have legitimate disagreement (Aroyo and Welty, 2015; Plank, 2022). Modelling this heterogeneity is important for pluralistic model alignment (Sorensen et al., 2024b), unbiased model safety, content moderation, personalization, and more.

We characterize three approaches to variation modelling: (1) Distributional Population Modelling, which directly models the distribution of labels for a given rater population (Zhang et al., 2024; Siththaranjan et al., 2024). This approach accounts for variance and valid disagreements between annotators but requires many raters labeling the same instances and doesn’t model which raters disagree or why. (2) Grouping by Characteristics such as demographics or annotation similarity. While grouping approaches can lead to higher agreement than the broader rater population, they still do not account for intra-group disagreement (Hwang et al., 2023; Prabhakaran et al., 2024), potentially leading to flattening variance or stereotyping. To capture intra-group variation, distributional learning is needed (Meister et al., 2024). Grouping by annotation similarity also requires significant overlap in labeled instances (Li et al., 2024). (3) Individual Modelling. At the individual level (Gordon et al., 2022; Jiang et al., 2024), the target is a single "correct" answer instead of a distribution,<sup>1</sup> allowing for standard supervised methods. Additionally, we can obtain group or population distributions through marginalization. Individual modeling also removes the requirement for raters to have any instance overlap. Because of these advantages, we arguefor and focus on improving individual modeling in order to better model human variation (for more, see App. B, Fig. A1). However, this raises the question - how should we represent an individual?

In this work, we propose to model rater variation using individual, free-text value profiles – interpretable natural language descriptions of human values that explain observed rating variation (§2). In §3, we introduce a methodology to measure the information content of possible rater representations. We carry out a series of experiments to evaluate our value profile system and other rater representations (§4, 5). In $\ S 6 ,$ we introduce a rater clustering algorithm that uncovers better groupings than the most predictive demographics, while loosening the typical requirement of annotators labeling overlapping instances. In other experiments, we find that our value profile system is interpretable, well-calibrated, and helps explain rater disagreement (§7). We conclude by discussing related work (§8), directions for future work (§9), and ethical advantages (and risks) of our approach (§11).

![](images/4faf1e3b04ca6fee567832c9b3e6e315b9e292451e756c7933d66228ad6c8431.jpg)  
Figure 1: The value profile autoencoder setup. Decoder outputs are from trained profile decoder while demographics are illustrative to preserve privacy. The encoder extracts/compresses value information from rater examples, and the decoder changes predictions on held-out questions according to the value profile.

## 2 Modelling Human Annotator Variation

## 2.1 Rater Representations

Let $\mathcal { R } = \{ r _ { 1 } , r _ { 2 } , . . . , r _ { N _ { R } } \}$ be the raters who we wish to model, $\mathcal { X } ~ = ~ \{ x _ { 1 } , x _ { 2 } , . ~ . ~ . ~ , x _ { N _ { X } } \}$ be the space of instances, and $\mathcal { Y } = \{ y _ { 1 } , y _ { 2 } , . . . , y _ { N _ { Y } } \}$ be the space of potential responses/ratings. We would like to model $y \mid \mathcal { X } , \mathcal { R }$ . However, because we don’t have sufficient information to represent (or observe) the rater r, we compare different potential representations for r:

• : No information about r. In this case, $\begin{array} { r } { P ( \mathcal { V } \mid \mathcal { X } , \emptyset ( r ) ) = \sum _ { r ^ { \prime } \in \mathcal { R } } P ( \mathcal { V } \mid \mathcal { X } , r ^ { \prime } ) } \end{array}$ $P ( \mathcal { V } \mid \mathcal { X } )$ , or the label distribution for the input marginalized over all raters.

• D: Demographic information about $r . P ( \mathcal { V }$ $ { \mathcal { X } } , D ( r ) )$ 2

<sup>2</sup>In the case that many demographics are provided, this is sometimes called a “persona" (Cheng et al., 2023). We also refer to this as “demographics (all)" or “intersectional demographics". This is in contrast to trying to model an entire

$E _ { n } \colon$ n in-context ratings as demonstrations from rater r. $P ( \mathcal { V } \mid \mathcal { X } , E _ { n } ( r ) )$ .

• V: A value profile natural language description of the rater’s values which are relevant for the task. $P ( \mathcal { V } \mid \mathcal { X } , V ( r ) )$ .

A value profile might be elicited directly from a rater r through a survey/value elicitation process. In absence of this data, we propose to infer V from observed example ratings $E _ { n }$ through an autoencoder setup.

## 2.2 Autoencoding Rater Values

Let $r _ { i }$ be a particular rater i drawn from the population of n raters, $x _ { j }$ be a particular instance $j ,$ and $y _ { i j }$ be the rating that rater i gave to instance j. Let ${ \mathcal { D } } _ { i } = \{ y _ { i 1 } , y _ { i 2 } , . . . , y _ { i N _ { i } } \}$ be the set of $N _ { i }$ ratings we have for rater i. We can build a language model encoder $Q _ { \phi }$ which estimates a value profile for each rater $r _ { i }$ from a set of (fit) demonstrations drawn from $D _ { i }$ , with corresponding probability distribution $q _ { \phi } : E _ { n } \to V$ . Similarly, a decoder $P _ { \theta }$ can estimate the label probability distribution $P ( \mathcal { V } | \mathcal { X } , V ( \mathcal { R } ) ) \approx p _ { \theta } : \mathcal { X } , V  \mathcal { Y }$ . Given this, the entire autoencoder system can be evaluated by sampling a value profile from the encoder $v _ { i } \sim q _ { \phi } ( E _ { n } ( r _ { i } ) )$ and calculating the (crossentropy) loss on unseen examples.

We randomly partition the instances into $\mathcal { D } _ { i } ^ { \mathrm { f i t } }$ for fitting a value profile and $\mathcal { D } _ { i } ^ { \mathrm { e v a l } }$ to train the decoder to generalize to held-out ratings. The setup may be seen as a way to “compress” predictive information about a rater’s labeling process from their examples $E _ { n } ( r _ { i } )$ to a natural language value profile v<sub>i</sub>.

![](images/05cab669b80bd8af842ac6c2261f0dea630ada62cca7ee3320819d1803d00d6f.jpg)  
Figure 2: Rater representations and example corresponding decoder prompts $( \emptyset , D , V , E _ { n } )$ . The decoder predicts the rater’s annotation given the rater representation.

In practice, we initialize the encoder and decoder parameters ϕ and θ as prompted language models (prompts in Figs. A9/A10). For the experiments in the paper, we freeze the encoder parameters and optimize the decoder directly with supervised finetuning. We choose to do this 1) as prompted language models performed quite well at encoding, preserving > 70% of usable information. (cf. Eq. 3, Fig. 5), 2) in order to regularize the encoder to remain human understandable/interpretable, and 3) to preserve generalizability across datasets.

We compare against the alternative rater representations of no information $\varnothing ,$ demographics D, and examples $E _ { n } ,$ by similarly fitting a decoder $D _ { \theta }$ to estimate $p _ { \theta } ( \mathcal { V } | \mathcal { X } , \cdot )$ . All parameters are initialized with a prompted language model. To ensure comparable results, we use $\mathcal { D } ^ { \mathrm { f i t } }$ demonstrations for the in-context demonstrations $E _ { n }$ and inferring value profiles in training and testing and the $\mathcal { D } ^ { \mathrm { e v a l } }$ demonstrations as held-out targets for all rater representations.

## 3 Estimating Usable Rater Information

We wish to compare the usable information for each rater representation. To do this, we extend Xu et al. (2020)’s concept of -information, which was created to analogize the concept of mutual information between random variables A, B to constrained computational families. We extend -information to the case where we have a third random variable, C, with computational family $\mathcal { V } \subseteq \{ f : \mathcal { A } \cup \{ \emptyset \} , C \to \mathcal { P } ( \mathcal { B } ) \}$

$$
H _ { \mathcal V } ( B \mid A , C ) = \operatorname* { i n f } _ { f \in \mathcal V } \mathbb { E } _ { a , b , c \sim A , B , C } [ - \log f [ a , c ] ( b ) ]\tag{1}
$$

$$
I _ { \mathcal { V } } ( A \to B \mid C ) = H _ { \mathcal { V } } ( B \mid \emptyset , C ) \ – H _ { \mathcal { V } } ( B \mid A , C )\tag{2}
$$

We can then measure predictive information from each rater representation $( \emptyset , D , E _ { n } , V )$ to ratings , given instances . I.e., we can estimate how much more we know about how a rater will respond given particular information about them, as compared to knowing nothing about the rater.

As Ethayarajh et al. (2022) show in a similar extension of -information, assuming we have an i.i.d. dataset of observations, we can get an unbiased estimate of this quantity for a computational family by training a model in each informational setting and comparing the held-out test losses to a trained model with no information. For more details, see Algorithm 1 (inspired by Ethayarajh et al. 2022). To contextualize the algorithm with an example loss plot, see Figure A2.

Algorithm 1 Computing Predictive -Information   
Input:   
Training data ${ \mathcal D } _ { \mathrm { t r a i n } } = \{ ( r _ { i } , x _ { j } , y _ { i j } ) \} = \{ ( r _ { i } , x _ { j } , y _ { i j } )$   
$( x _ { j } , y _ { i j } ) \in R _ { i } ^ { \mathrm { t e s t } } , r _ { i } \in$ train raters $R ^ { \mathrm { t r a i n } } \}$   
Test data $\mathcal { D } _ { \mathrm { t e s t } }$ for held-out raters $R ^ { \mathrm { t e s i } } , R ^ { \mathrm { t r a i n } } \cap R ^ { \mathrm { t e s t } } = \emptyset$   
Initialized decoder d, a prompted, pretrained LM   
Natural language rater representation $g : R  \mathbb { N L }$   
$d _ { g } \gets$ finetune d on $\{ ( g ( r _ { i } ) , x _ { j } , y _ { i j } ) | ( r _ { i } , x _ { j } , y _ { i j } ) \} \in$   
$\tilde { \mathcal { D } } _ { \mathrm { t r a i n } } \}$ ▷ Train w/ rater information   
$d _ { \emptyset } \gets$ finetune d on $\{ ( \emptyset , x _ { j } , y _ { i j } ) | ( r _ { i } , x _ { j } , y _ { i j } ) \in \mathcal { D } _ { \operatorname { t r a i n } } \}$ ▷   
Train w/out rater information   
$H _ { V } ( \mathcal { V } | \mathcal { X } ) , H _ { V } ( \mathcal { V } | \mathcal { X } , g ( R ) )  0 , 0$   
for $( r _ { i } , x _ { j } , y _ { i j } ) \stackrel { \cdot } { \in } \mathcal { D } _ { \mathrm { t e s t } }$ do ▷ Accumulate average held-out   
test losses   
$\begin{array} { r } { H _ { V } ( \mathcal { V } | \mathcal { X } )  H _ { V } ( \mathcal { V } | \mathcal { X } ) - \frac { 1 } { | \mathcal { D } _ { \mathrm { t e s t } } | } \log d _ { \emptyset } ( x _ { j } , \emptyset ) ( y _ { i j } ) } \end{array}$   
$H _ { V } ( \mathcal { V } | \mathcal { X } , g ( R ) ) \mathrm { ~ \omega ~ { ~  ~ } ~ } \overleftarrow { H } _ { V } ( \mathcal { V } | \mathcal { X } , g ( R ) ) \mathrm { ~ \omega ~ { ~ - ~ } ~ }$   
$\begin{array} { r } { \frac { 1 } { | \mathcal { D } _ { \mathrm { t e s t } } | } \log \dot { d } _ { g } ( x _ { j } , g ( \dot { r } _ { i } ) ) ( y _ { i j } ) } \end{array}$   
end for   
$\hat { I } _ { V } ( g ( R ) \to \mathcal { Y } | \mathcal { X } )  H _ { V } ( \mathcal { Y } | \mathcal { X } ) - H _ { V } ( \mathcal { Y } | \mathcal { X } , g ( R ) ) \quad \triangleright$   
Predictive information is drop in test loss when including   
rater information

## 4 Experimental Methodology

Training details We split raters into 50/50 train/test splits and report results for training/test runs on five random splits. We draw $| \mathcal { D } _ { i } ^ { \mathrm { f t } } | \sim$ $\mathcal { U } ( \{ 2 , \dots , | \mathcal { D } _ { i } | \ - \ 2 \} )$ to ensure that we have variable-sized fit/eval splits with at least two instances each. We train the decoder (gemma2-9b-pt, Gemma Team et al. 2024) for a single epoch (important for maintaining calibration, Ji et al. 2021). For encoders, we use gemma2-9b-it, gemma2-27b-it (Gemma Team et al., 2024), and Gemini-1.5 Pro (Team et al., 2024). See App. A for details.

<table><tr><td>Dataset</td><td>Task</td><td>Choices</td><td>Dem.</td><td>Inst.</td><td>Raters</td><td>Ratings</td></tr><tr><td>OpinionQA W27 (OQA)</td><td>Opinions (US)</td><td>2-6</td><td>11</td><td>77</td><td>10k</td><td>731k</td></tr><tr><td>Hatespeech-Kumar (HK)</td><td>Hate Speech</td><td>2</td><td>18</td><td>19k</td><td>864</td><td>37k</td></tr><tr><td>DICES (DIC)</td><td>Toxicity</td><td>3</td><td>5</td><td>990</td><td>160</td><td>65k</td></tr><tr><td>ValuePrism (VP)</td><td>Moral Judgments</td><td>3</td><td>-</td><td>31k</td><td>4.5k</td><td>199k</td></tr><tr><td>Habermas-Likert (HL)</td><td>Opinions (UK)</td><td>7</td><td>9</td><td>1.1k</td><td>259</td><td> $3 . 1 \mathrm { k } ^ { * }$ </td></tr><tr><td>Prism (PR)* 2</td><td>Chat Preference</td><td>2</td><td>20</td><td>8.0k</td><td>1.4k</td><td> $8 . 0 \mathrm { k ^ { \ast } }$ </td></tr></table>

Table 1: Dataset statistics including task information, number of multiple choice options (Ch.), demographic variables (Dem), unique instances (#I), unique raters (#R), and total ratings (#Rat). Datasets: OQA (Santurkar et al., 2023), HK (Kumar et al., 2021), DIC (Aroyo et al., 2023), VP (Sorensen et al., 2024a), HL (Tessler et al., 2024), and PR (Kirk et al., 2024b). Numbers may be smaller than in original datasets due to preprocessing/sampling (see §A). ∗Results are noisier for datasets with <10k ratings due to underfit models/small test sets.

Datasets We utilize six datasets intended for research (Table 1) spanning tasks relevant to model alignment, content moderation, and computational social science. These datasets feature forced choice selection tasks and were selected to contain 1) some rater variation due to their subjective nature, 2) annotator IDs to link annotations from the same rater, and 3) ideally, some demographic information. Preprocessing information in §A.

## 5 Performance Across Rater Representation Settings

Detailed results for held-out test losses across rater representations can be found in Figure 3. Accuracies can be found in App. A3, but results mirror the loss results, which we will focus on. Detailed results for held-out test losses and accuracies across rater representations can be found in Figures 3 and A3 respectively.

We note that error bars are much larger for 2 datasets, HL and PR. We believe that this is mainly because the datasets are smaller (<10k ratings), which means that 1) the trained model may be underfit and 2) there is a smaller sample size for each test split. We include results for all datasets for maximal inclusion, but focus our attention on the large datasets (>30k ratings: OQA, HK, DIC, VP) for which we can make higher confidence comparisons across settings.

Now, we compare decoder performance across rater representation settings (see Figs. 3, 4, A3). Our main findings are:

In-context examples improve predictions. Across all four large datasets, providing the decoder with in-context examples of the rater’s previous annotations significantly improved the prediction of their ratings on held-out test data in both accuracy and test loss $( p \ < \ . 0 0 1 )$ . We observe a similar, but less significant, drop in loss/increase in accuracy on the two small datasets. This shows that rater demonstrations offer useful information for disentangling human variation.

Value profiles are highly predictive. Value profiles generated by Gemini (version: 1.5-pro) consistently provided a significant performance boost across all four large datasets, suggesting value profiles contain useful information for modeling variation. Gemma2-9b and 27b value profiles also offered a significant boost for three of the four large datasets (VP, OQA, and HK), but not for DICES. In other words, as one might expect, value profiles improve with scale. As a result, we will focus the remainder of our experiments on the top-performing value profiles from Gemini.

Value profiles effectively compress rater information. Since the value profiles are encoded from the same in-context examples used in the maximal example setting, we can exactly calculate the amount of decoder-usable information preserved (see Figure 5):

$$
\frac { I _ { \mathcal { V } } ( V ( E _ { n } ( \mathcal { R } ) ) \to \mathcal { V } \mid \mathcal { X } ) } { I _ { \mathcal { V } } ( E _ { N } ( \mathcal { R } ) \to \mathcal { V } \mid \mathcal { X } ) }\tag{3}
$$

Value profiles effectively compressed the relevant information from in-context examples, preserving >70% of the usable information for the four large datasets. This indicates that value profiles are an efficient way to represent human variation.

Demographics have limited predictive power. Intersectional demographics generally offered a small and insignificant information boost, except for OpinionQA, where political affiliation was highly predictive.<sup>3</sup> Interestingly enough, however, the gains from demographic variables for other datasets were minimal. Additionally, value profiles contain more usable predictive information than demographics in all five datasets with demographics except OpinionQA (cf. Fig. 4). This suggests that demographics alone may not be sufficient to capture the full spectrum of human variation.

![](images/5909f4ba607b9a737e125d56ee4c0567a5b3968e766ea89ac1368e01078bd347.jpg)  
Figure 3: Test losses across rater representation settings. Dashed line: label entropy H( ); no info: ; profile\*: value profiles V generated by gemma2-{9/27}b / Gemini-1.5-Pro; dem (all): D; N ex: E , up to N examples from $\bar { D } _ { i } ^ { \mathrm { f i t } }$ . ValuePrism does not have demographics, but does have a ground truth value profile. Each dot corresponds to a run with a differently seeded train/test split, with 95% CI reported. Generally, in-context examples are more performant than value profiles, which are more performant than demographics.

![](images/1f87078c6baaea7421ccebd509b1c26cc536df3d8993ffeaf2066b1fecc561ff.jpg)  
Figure 4: Usable rater information across datasets and rater representations (95% CI).

We also experiment with providing one demographic variable at a time (i.e., grouping by demographic) and providing value profiles and demographics together (cf. App. C/Fig. A5). As expected, single demographics provide less information than including all demographics. Also, demographics and value profiles can contain complementary information, with the best performing representation generally being demographics and value profiles together.

![](images/ea2bdef22bb4b09ec88c5225b2018a5fd04dddc1bf67b7901c7e4815bd65d5bc.jpg)  
Figure 5: Info. preserved w.r.t. to using all examples. Results shown on the four large, low-variance datasets. Gemini profiles preserve >70% of usable information.

## 6 Value Profile Clustering for Grouping Raters

To identify common modes of (dis)agreement, avoid over-personalization (Kirk et al., 2024a) and alleviate potential privacy concerns associated with inferring individual value profiles, we introduce a novel value profile-based rater clustering algorithm. Compared to traditional clustering methods, some advantages to our clustering method are that it: 1) does not require any overlap in instances seen by annotators; 2) is able to leverage semantic information between instances; 3) enables qualitative analyses through resulting cluster descriptions.

We assign the train raters to clusters using Algorithm 2 (cf. Figure 6), where each cluster corresponds to a single value profile description.<sup>4</sup> We train a decoder to predict train rater annotations based on assigned cluster, and evaluate on held-out test raters. For all datasets, we use 100 randomly sampled value profiles as the cluster candidates. Results can be found in Figs. 5/7 and the corresponding clusters can be found in Appendix H.

![](images/e5d1471811492974731b98bf6d1065a7bb654fff5fc077fdb9d712cd7e51164d.jpg)  
Figure 6: Clustering algorithm represented pictorially (also see Algorithm 2). 1) The decoder predicts label distributions for each instance and value profile combination; 2) calculate the loss for predicting each rater’s "fit" ratings with each value profile; 3) find C (# clusters) value profiles s.t. when each rater is assigned to a cluster, overall loss is minimized; 4) assign new raters to cluster with smallest loss on rater’s train ratings.

Clustering is effective and is suggestive of underlying modes of disagreement. Across all four large datasets, we observe a few common trends: 1) clustering into eight profile groups gives significant predictive improvement over no information, and 2) predictivity improves as we increase the number of clusters. Beyond this, we see some divergences.

For DIC and OQA, clustering is highly effective - using just two clusters preserves the majority of the usable rater information (60%/51% respectively), and using eight clusters roughly matches the performance of giving each rater their own profile. This suggests that perhaps most raters fall into one of very few “modes" of agreement for these datasets, and that clustering based on value profiles is highly effective at finding these groupings. For the other two large datasets, HK and VP, clustering preserves a significant amount of information ( 20%) but is not as predictive. This implies either a failure to find the best clusters or that the underlying variation is inherently more difficult to categorize. Interestingly, this dataset divide coheres with our intuitions: e.g., for OpinionQA, ideology is highly explanatory and mostly centered around a few clusters, whereas ValuePrism includes a diverse set of 4k unique values which resist categorization. While it is epistemically difficult to totally disentangle a failure of our method to find correct groupings vs. a true difference in dimensionality of disagreement, we do find these results suggestive of profile clustering being able to tell us something interesting about the true underlying reasons for rater variation.

```latex
Algorithm 2 Value Profile Clustering
Input:
Decoder model $d : \mathcal { X } , V \to \mathcal { P } ( \mathcal { Y } )$
Candidate value profiles $V$
Rater annotations for rater i: $R _ { i } ^ { \mathrm { { f i t } } } = \{ ( x _ { j } , y _ { i j } ) \}$
Target number of clusters $N _ { \mathrm { c l u s t e r } }$
Initial clusters $C = [ \underset { \ldots } { V } _ { 1 } , \underset { \ldots } { V } _ { 2 } , \ldots , \underset { \ldots } { V } _ { N _ { \mathrm { c l u s t e r } } } ]$
Maximum iterations $M _ { \mathrm { i t e r } }$
$N _ { x } \gets | \{ x _ { j } \ \mathrm { s . t . } \ \exists i , ( x _ { j } , \cdot ) \in R _ { i } ^ { \mathrm { f i t } } \} |$ ▷ # unique inst.
Initialize $\bar { P } \in \mathbb R ^ { N _ { x } \times \bar { N } _ { v } \times | \mathcal { V } | } \mathrm { ~ \ o ~ }$ Fill in output probabilities
conditioned on each value profile
for $j \in [ 1 , \ldots , N _ { x } ]$ do ▷ For each instance
for $\dot { k } \in [ 1 , \dots , \dot { N } _ { v } ]$ do ▷ For each value profile
$P [ j , \dot { k } ] = d ( x _ { j } , \dot { v } _ { k } ) ~ \triangleright$ Prob. dist. over $\mathcal { V }$ from d
conditioned on instance $x _ { j }$ and profile $v _ { k }$
end for
end for
$\begin{array} { r } { L ( r _ { i } , v _ { k } ) \gets \sum _ { ( x _ { i } , y _ { i j } ) \in R _ { i } ^ { \mathrm { f i } } } - \log P [ j , k , y _ { i j } ] \triangleright } \end{array}$ Total loss
from assigning rater $r _ { i }$ to profile $v _ { k }$
converged False; iter $ 0 ; C _ { \mathrm { l a s t } }  C$
while iter $< M _ { \mathrm { i t e r } } \&$ not converged do
for $c \in [ 1 , \ldots , N _ { \mathrm { c l u s t e r } } ]$ do ▷ Fixing all profiles except
c, greedily find best profile to replace c
$C [ c ] \gets \mathrm { ~ \iota ~ ~ \bar { ~ } { ~ - ~ } ~ }$ ▷ New cluster that minimizes loss
$\underset { \hat { v } _ { c } \in V } { \overset {  } { \operatorname { a r g m i n } } } \sum _ { i \in [ 1 , N _ { \mathrm { R } } ] } { \overset { \quad } { v } } \in ( C / \{ \hat { v } _ { c } \} \cup \{ \tilde { v } _ { c } \} ) \quad$
end for
converged $ C = C _ { \mathrm { l a s t } } ; C _ { \mathrm { l a s t } }  C$
end while
Output: Clusters C, assignments arg minL(r<sub>i</sub>, v˜)
v˜ C
```

Profile clusters are more predictive than the best demographic groupings. Next, we compare with the best performing demographic clusters, grouping people who gave the same demographic response to a categorical demographic question (e.g., people in the same country for DIC or same political ideology for OQA). We compare specifically across the three large datasets w/ demographics: for DIC, the two profile clusters is more predictive than grouping based on country; for HK/OQA, the four profile clusters outperform grouping by religiosity/ideology respectively. In other words, clustering by value profiles is able to outperform the most performant demographic clusters when using the same or fewer number of groupings.

![](images/21d96b7e911a6d65a07b01baa33960820f97973c7fb5a66e758ceaa8af88c3eb.jpg)  
Figure 7: Performance after clustering raters into 2/4/8 profile clusters alongside the best performing categorical demographic grouping, with the # of groups in parentheses. Value profile clustering is highly effective, outperforming the best demographic grouping of comparable size.

![](images/8e71e69ea49df3b432eabce377f1abd3f4360fab1caf2121bcbed0f110092cac.jpg)  
Figure 8: Ideological makeup of the raters sorted into each value profile cluster for OpinionQA. The clusters recover strong ideological trends.

Where predictive, demographic groupings closely match profile clusters. Focusing in on the two datasets where clustering was most effective, OQA and DIC, we see if there are any demographic trends related with clustering. As you can see from Figure 8, there are strong demographic trends in the OQA clusters - cluster two consists almost exclusively of self-described conservative individuals, while cluster one consists of mostly self-described liberal individuals. In other words, despite not having access to demographics, the value profiles are able to largely reconstruct the most explanatory demographic groupings. Meanwhile, for the DIC four-profile clusters, the clusters cut across almost uniformly across all demographic groupings (Figure A4). This suggests that for DIC, the most important dimensions of variation are not found in the demographic groupings.

Cluster descriptions qualitatively describe modes of disagreement.. The profile clustering algorithm returns not only clustering assignments, but also a single corresponding value profile for each cluster (see App. H). For DICES, even two clusters were quite predictive. The corresponding value clusters relate to overall sensitivity to toxicity (e.g., profile 1: "High tolerance for offensive language"; "Narrow definition of toxicity" vs. profile 2: "Strong reaction to overt negativity, "Sensitivity to potential harm"). Meanwhile, when going to four clusters, more nuance enters in (e.g., "Context and intent matter"). In other words, it seems that 1) overall sensitivity to toxicity is an important dimension in explaining variation, and 2) there are clusters of people who hold more nuanced views. For OpinionQA, descriptors that have to do with politics are often used (e.g., "Economically Conservative, but Populist on Trade"). For HK, which required more clusters to effectively predict, there are many specific phrases about what kinds of things the rater may or may not find offensive (e.g., "Profanity tolerance"; "Discomfort with stereotyping"; etc.). Meanwhile, for PR the clusters center around potentially conflicting chatbot preferences (e.g., "Appreciates simplicity" vs. "Appreciates nuanced and comprehensive answers"). Because value profiles are interpretable (see §7) and can recover demographic groupings where predictive, we have reason to believe that these qualitative differences map to important dimensions of disagreement for a dataset.

## 7 Extrinsic Evaluation

In the previous sections, we have established that value profiles are predictive of individual rater annotations for a wide variety of relevant tasks, based on intrinsic performance metrics. We are now assessing value profiles within the context of wider real-world applications. We show that value profiles are interpretable – which is important for enabling control by the end-user; their predictions are steerable and well-calibrated – which enables pluralistic AI alignment; and they are reliable for extrinsic tasks in the context of computational social science, such as simulating a rater population.

Value profiles are interpretable. We first explore interpretability – i.e., do the value descriptions change the decoder outputs in a commonsense manner? Because the encoder is prompted and only the decoder is trained (cf. Section 2.2), we believe that this serves as a strong regularization so that the value profiles correlate with held-out ratings only by the natural language values described. To ensure that this is the case, we test the interpretability of the autoencoder as follows: 1. For each instance and 100 value profiles, we get the estimated output distribution for the decoder. 2. We select the value profiles that have the largest Jensen-Shannon divergence. 3. We create a binary classification task: given an instance and two value profiles, which profile corresponds to which estimated output distribution? If the change in distribution is not correlated with common sense, we would expect 50% accuracy, while performance would approach 100% if an observer is always able to match value profiles to corresponding distributions.

As an example of the task, here is a shortened example from VP: Is it moral, immoral, or morally dependent on context to do this: “Choosing not to get a vaccine." Profile 1: Prioritization of collective good over individual needs. Profile 2: Strong belief in individual liberty and au-

<table><tr><td>Data</td><td>Accuracy (%)</td></tr><tr><td>OQA</td><td>94.8 (±2.5)</td></tr><tr><td>HK</td><td>96.3 (±.6)</td></tr><tr><td>DIC</td><td>95.5 (±.7)</td></tr><tr><td>VP</td><td>91.7 (±.9)</td></tr><tr><td>HL</td><td>89.8 (±.9)</td></tr><tr><td>PR</td><td>80.0 (±.3)</td></tr><tr><td>Chance</td><td>50</td></tr></table>

Table 2: Profiles are semantically interpretable (95% CI).

tonomy. X probabilities: Moral: 92%, Contextdependent: 7%, Immoral: 2%. Y probabilities: Immoral: 94%, Context-dependent: 3%, Moral: 3%. Which profile goes with the X probabilities? Correct answer in footnote. <sup>5</sup>

We report accuracies for a zero-shot prompted Gemini in Table 2. Accuracies range from 80-96% across all datasets, demonstrating that variation in outputs from value profiles are explainable by their plain natural language (p<.001).

![](images/4ac56f4ecc051a07b85c1079b45bc1b559eae59425a205bdcfb307cd27ae7f1f.jpg)  
Figure 9: Calibration plots for value profile decoders. (Perfect calibration = dotted line). The decoders are very well-calibrated.

![](images/f4889c6b7ac8b6fb29f379c41234bae89675bff3b47725696a85fde8facd5ad4.jpg)  
Figure 10: Instance-level observed vs. estimated interannotator agreement (as the probability that two raters agree). The predicted simulated agreement correlates with the observed agreement.

Decoders are well-calibrated. Decoder calibration is important for two principal reasons. Firstly, an appropriately calibrated decoder allows us to trust the model confidence w.r.t. error rate. Secondly, even raters with shared values may have varied outputs - a well-trained decoder would model this distribution appropriately. Calibration plots for the value profile decoders can be found in Figure 9. The trained decoders are quite well-calibrated, suggesting that we can generally trust the decoder’s output confidence.

Simulating an annotator population with value profiles. Given a trained decoder and a set of value profiles, one can simulate a Given a trained decoder and a set of value profiles, one can simulate a population – or “jury" (Gordon et al., 2022) – of annotators on novel instances. While one can do many things with such a simulated population (Park et al., 2023; Gordon et al., 2022), one experiment is to predict which instances raters would have higher or lower inter-annotator agreement (IAA).

In order to calculate out-of-distribution IAA, we first eliminate the datasets where annotators have no overlap (PR) and for which all raters annotated the same instances (DIC, OQA). For each instance, we then sample 100 value profiles that were not fit on that instance, and calculate the estimated probability of agreement between those annotators (assuming each rater annotates at random from the decoder’s output distribution). We also filter to instances that were labeled by a minimum number of annotators (See §A for details). We then compare to the actual observed probability that two raters agree on the instance (see Figure 10). For all three datasets, there is a positive correlation between the estimated and observed IAA, and this correlation is significant $( p < . 0 0 1 )$ for HK and VP. While not much variance is explained $( R ^ { 2 } < . 2 )$ , the observed P(agree) is a high variance estimate with few ( 5) raters per instance. In summary, a simulated population with value profiles provides some explanatory power at predicting inter-annotator agreement, but is not yet a high precision tool.

## 8 Related Work

Clustering and demographics While aligning to groups can increase agreement (Chen et al., 2024), it also has been shown to flatten intra-group variation (Orlikowski et al., 2023; Wang et al., 2025) lead to stereotyping (Cheng et al., 2023), or simply not be correlated with subjective NLP tasks Orlikowski et al. (2025). Prior work explores embedding-based methods for clustering individuals by responses (Vitsakis et al., 2024; Li et al., 2024) and similarly finds that clusters cut across demographic groups. Beyond predictivity, demographics can still be important to collect for evaluating group fairness (Aguirre et al., 2023; Kirk et al., 2024b).

Steering to individuals Prompted large language models (LLMs) have been used to simulate human judgments, e.g.: NLP task annotators (Bavaresco et al., 2024), political survey respondents (Argyle et al., 2023), fact-checking labels (De et al., 2024), or human attitudes and behavior (Park et al., 2024). Textual user profiles have also been proposed for personalizing chats (Zhang et al., 2018). Hu and Collier (2024) similarly use textual demographic descriptions and find that they provide small, but statistically significant, gains in explaining human variation. Many works focus merely on prompting an existing LLM, while our work explicitly trains an LLM to better match varied perspectives (as in Gordon et al. 2022; Jiang et al. 2024). Encoding individual information from demonstrations is also analogous to behavioral user modeling for recommender systems (Radlinski et al., 2022; Ramos et al., 2024). Poddar et al. (2024) also use an autoencoder to model human variation, but focus on preference data and use a vector-valued latent space instead of natural language.

Values and alignment Similarly to natural language value profiles, Bai et al. (2022)’s "constitutional AI" train models to follow textual principles, although they focus on a single set of principles. Findeis et al. (2024) propose to learn preference principles directly from preference data (similar to our encoder setup). Values have also been normatively proposed as an alignment target (Gabriel, 2020; Klingefjord et al., 2024), and pluralistic alignment (Sorensen et al., 2024b) seeks to align AI systems to diverse values.

## 9 Conclusion

In conclusion, we proposed modeling human variation via natural language value profiles. We also proposed a methodology to compare the usable information in various rater representations, and found that value profiles contain more information than demographics. Prompted LLMs serve as effective value encoders, retaining 70% of the useful rater information from demonstrations. In addition, we introduced a profile clustering algorithm which is able to find more explanatory clusters of raters than grouping by the most predictive demographics. Finally, we showed that value profiles are extrinsically useful for interpretability, steerability, and for simulating diverse populations, hence offering new ways to describe individual variation beyond demographics.

Some promising avenues for future work include: 1) qualitative analyses extracting values from data; 2) fairness analysis of who can (or cannot) be wellrepresented by value profiles; 3) study on sensitivity of value profiles to choice of and number of ratings; 4) extensions to additional models and datasets; 5) how decoders handle conflicting value information in a profile; 6) optimization of the encoder (e.g. via ELBO); and 7) human evaluations to see how well represented people feel by value profiles.

## Acknowledgements

We would like to thank Lora Aroyo, Laura Weidinger and Ahmad Beirami for helpful discussions. We would also like to thank Charvi Rastogi and Jared Moore for useful draft feedback.

## 10 Limitations

We have tried to test for generalization across six tasks and datasets and more than twenty demographic distributions, However, all of the experiments use the Gemma-2 (Gemma Team et al., 2024) and Gemini (Team et al., 2024) families of models. This is due in part due to the TPU hardware available to us and because of the expense of the experiments (more than 650 training runs). We have no reason to think that our results are due to anything particular about these families of models though, and prior work doing similar experiments on demographics with other models has reached similar results (Orlikowski et al., 2023; Hwang et al., 2023). That being said, future work could benefit from experiments on more model families.

## 11 Ethical Considerations

We seek to improve AI systems’ ability to model diverse values out of a hope that the systems can be more inclusive, better represent a range of viewpoints, and better serve a wider population. Here, we explore benefits and risks of modelling variation with value profiles.

Profiling risks One of the potential ethical risks of the value profile through an autoencoder paradigm is in the name: “profile." Value profiles are, inherently, guesses about the underlying values that people may hold that lead them to annotate in the way that they do. There are potential privacy concerns here - people may not wish to have their underlying values exposed (Tomasev et al., 2021). It may be better if people had agency to create their own value profiles through some voluntary elicitation process (Park et al., 2024).

False generalization On the one hand, value profiles are an attempt to reduce the (often false) generalization inherent when grouping e.g. by demographic groups (Dev et al., 2022), improving on widely used current techniques. On the other hand, generalization risks remain. For example, there may be multiple possible underlying values that could support a set of rater annotations. In the absence of additional information, the value encoder may arbitrarily assign a guess to the underlying values. Also, our experiments focus on English-language value profiles, so generalization to other languages is unknown. Additionally, there is always a risk of misrepresentation when using simulated human ratings in place of actual ratings at all (Agnew et al., 2024).

Interpretability, understandability, and user agency However, there are also several positive attributes to value profiles. First of all, they are interpretable - and therefore, potentially more understandable to a user. While people interact with many technologies today that are trying to model their behavior and preferences, most such systems do not break down their user representation into a format that as as easy to understand as a textual description. Additionally, this makes value profiles intervenable - people could change how they choose to be represented (Balog et al., 2019; Lazar et al., 2024). Relatedly, value profiles serve as a step towards explainable AI (Arrieta et al., 2019; Koh et al., 2020) for human variation.

Enabling value reflection Learning values from data, while allowing users to modify the values, is loosely related to John Rawls’ concept of reflective equilibrium (Rawls, 2005; Knight, 2025): ratings are akin to judgments, and value profiles are an attempt to draw general “principles" out of the judgments in a bottom-up manner. Meanwhile, a user can then edit the value profile, applying topdown reflection on whether the values/principles are ones that the person would endorse. In this light, perhaps “value profiles" could help a person to explore their own value system, both in the values that their decisions may imply and considering which values they would reflexively endorse.

"Chosenness" of values Many works modeling diversity focus on socio-demographics. However, many demographics are not a result of a person’s agency, but rather a product of unchosen life factors - for example, the country in which one is born, or the economic opportunities available to them. Meanwhile, while the values that one holds can certainly be affected by unchosen factors (Nguyen, 2024), values can also be chosen for oneself. Thus, in the spirit of luck egalitarianism (Dworkin, 2002) it may be more justifiable to represent someone using values that they reflexively endorse, as opposed to boxing them in to the characteristics of a group they may not have chosen.

Importance of demographics for fairness Also, while demographics may not be the most ideal rater representation in many cases for the above reasons, it can still be important to collect demographic information for other worthwhile goals, such as fairness/evaluating group disparities, ensuring representation, etc.

## References

Gavin Abercrombie, Verena Rieser, and Dirk Hovy. 2023. Consistency is key: Disentangling label variation in natural language processing with intraannotator agreement. Preprint, arXiv:2301.10684.

William Agnew, A. Stevie Bergman, Jennifer Chien, Mark Díaz, Seliem El-Sayed, Jaylen Pittman, Shakir Mohamed, and Kevin R. McKee. 2024. The illusion of artificial inclusion. In Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems, CHI ’24, New York, NY, USA. Association for Computing Machinery.

Carlos Aguirre, Kuleen Sasse, Isabel Cachola, and Mark Dredze. 2023. Selecting shots for demographic fairness in few-shot learning with large language models. Preprint, arXiv:2311.08472.

Lisa P. Argyle, Ethan C. Busby, Nancy Fulda, Joshua R. Gubler, Christopher Rytting, and David Wingate. 2023. Out of one, many: Using language models to simulate human samples. Political Analysis, page 1–15.

Lora Aroyo, Alex S. Taylor, Mark Diaz, Christopher M. Homan, Alicia Parrish, Greg Serapio-Garcia, Vinodkumar Prabhakaran, and Ding Wang. 2023. Dices dataset: Diversity in conversational ai evaluation for safety. Preprint, arXiv:2306.11247.

Lora Aroyo and Chris Welty. 2015. Truth is a lie: Crowd truth and the seven myths of human annotation. AI Magazine, 36(1):15–24.

Alejandro Barredo Arrieta, Natalia Díaz-Rodríguez, Javier Del Ser, Adrien Bennetot, Siham Tabik, Alberto Barbado, Salvador García, Sergio Gil-López, Daniel Molina, Richard Benjamins, Raja Chatila, and Francisco Herrera. 2019. Explainable artificial intelligence (xai): Concepts, taxonomies, opportunities and challenges toward responsible ai. Preprint, arXiv:1910.10045.

Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain, Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, and 32 others. 2022. Constitutional ai: Harmlessness from ai feedback. Preprint, arXiv:2212.08073.

Krisztian Balog, Filip Radlinski, and Shushan Arakelyan. 2019. Transparent, scrutable and explainable user models for personalized recommendation. In Proceedings ofthe 42nd International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR’19, page 265–274, New York, NY, USA. Association for Computing Machinery.

Anna Bavaresco, Raffaella Bernardi, Leonardo Bertolazzi, Desmond Elliott, Raquel Fernández, Albert Gatt, Esam Ghaleb, Mario Giulianelli, Michael

Hanna, Alexander Koller, André F. T. Martins, Philipp Mondorf, Vera Neplenbroek, Sandro Pezzelle, Barbara Plank, David Schlangen, Alessandro Suglia, Aditya K. Surikuchi, Ece Takmaz, and Alberto Testoni. 2024. Llms instead of human judges? a large scale empirical study across 20 nlp evaluation tasks. CoRR, abs/2406.18403.

Quan Ze Chen, K. J. Kevin Feng, Chan Young Park, and Amy X. Zhang. 2024. Spica: Retrieving scenarios for pluralistic in-context alignment. Preprint, arXiv:2411.10912.

Myra Cheng, Esin Durmus, and Dan Jurafsky. 2023. Marked personas: Using natural language prompts to measure stereotypes in language models. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1504–1532, Toronto, Canada. Association for Computational Linguistics.

Soham De, Michiel A Bakker, Jay Baxter, and Martin Saveski. 2024. Supernotes: Driving consensus in crowd-sourced fact-checking. arXiv preprint arXiv:2411.06116.

Sunipa Dev, Emily Sheng, Jieyu Zhao, Aubrie Amstutz, Jiao Sun, Yu Hou, Mattie Sanseverino, Jiin Kim, Akihiro Nishi, Nanyun Peng, and Kai-Wei Chang. 2022. On measures of biases and harms in NLP. In Findings of the Association for Computational Linguistics: AACL-IJCNLP 2022, pages 246–267, Online only. Association for Computational Linguistics.

R. M. Dworkin. 2002. Sovereign virtue: The theory and practice of equality. Philosophical Quarterly, 52(208):377–389.

Kawin Ethayarajh, Yejin Choi, and Swabha Swayamdipta. 2022. Understanding dataset difficulty with -usable information. Preprint, arXiv:2110.08420.

Arduin Findeis, Timo Kaufmann, Eyke Hüllermeier, Samuel Albanie, and Robert Mullins. 2024. Inverse constitutional ai: Compressing preferences into principles. Preprint, arXiv:2406.06560.

Iason Gabriel. 2020. Artificial intelligence, values, and alignment. Minds and Machines, 30(3):411–437.

Gemma Team, Morgane Riviere, Shreya Pathak, Pier Giuseppe Sessa, Cassidy Hardin, Surya Bhupatiraju, Léonard Hussenot, Thomas Mesnard, Bobak Shahriari, Alexandre Ramé, Johan Ferret, Peter Liu, Pouya Tafti, Abe Friesen, Michelle Casbon, Sabela Ramos, Ravin Kumar, Charline Le Lan, Sammy Jerome, and 179 others. 2024. Gemma 2: Improving open language models at a practical size.

Mitchell L. Gordon, Michelle S. Lam, Joon Sung Park, Kayur Patel, Jeff Hancock, Tatsunori Hashimoto, and Michael S. Bernstein. 2022. Jury learning: Integrating dissenting voices into machine learning models. In CHI Conference on Human Factors in Computing Systems, CHI ’22, page 1–19. ACM.

Tiancheng Hu and Nigel Collier. 2024. Quantifying the persona effect in llm simulations. Preprint, arXiv:2402.10811.

EunJeong Hwang, Bodhisattwa Prasad Majumder, and Niket Tandon. 2023. Aligning language models to user opinions. Preprint, arXiv:2305.14929.

Ziwei Ji, Justin D. Li, and Matus Telgarsky. 2021. Earlystopped neural networks are consistent. Preprint, arXiv:2106.05932.

Liwei Jiang, Taylor Sorensen, Sydney Levine, and Yejin Choi. 2024. Can language models reason about individualistic human values and preferences? Preprint, arXiv:2410.03868.

Hannah Rose Kirk, Bertie Vidgen, Paul Röttger, and Scott A Hale. 2024a. The benefits, risks and bounds of personalizing the alignment of large language models to individuals. Nature Machine Intelligence, pages 1–10.

Hannah Rose Kirk, Alexander Whitefield, Paul Röttger, Andrew Bean, Katerina Margatina, Juan Ciro, Rafael Mosquera, Max Bartolo, Adina Williams, He He, Bertie Vidgen, and Scott A. Hale. 2024b. The prism alignment dataset: What participatory, representative and individualised human feedback reveals about the subjective and multicultural alignment of large language models. Preprint, arXiv:2404.16019.

Oliver Klingefjord, Ryan Lowe, and Joe Edelman. 2024. What are human values, and how do we align ai to them? Preprint, arXiv:2404.10636.

Carl Knight. 2025. Reflective Equilibrium. In Edward N. Zalta and Uri Nodelman, editors, The Stanford Encyclopedia of Philosophy, Spring 2025 edition. Metaphysics Research Lab, Stanford University.

Pang Wei Koh, Thao Nguyen, Yew Siang Tang, Stephen Mussmann, Emma Pierson, Been Kim, and Percy Liang. 2020. Concept bottleneck models. Preprint, arXiv:2007.04612.

Deepak Kumar, Patrick Gage Kelley, Sunny Consolvo, Joshua Mason, Elie Bursztein, Zakir Durumeric, Kurt Thomas, and Michael Bailey. 2021. Designing toxic content classification for a diversity of perspectives. Preprint, arXiv:2106.04511.

Seth Lazar, Luke Thorburn, Tian Jin, and Luca Belli. 2024. The moral case for using language model agents for recommendation. Preprint, arXiv:2410.12123.

Junyi Li, Charith Peris, Ninareh Mehrabi, Palash Goyal, Kai-Wei Chang, Aram Galstyan, Richard Zemel, and Rahul Gupta. 2024. The steerability of large language models toward data-driven personas. In Proceedings ofthe 2024 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 7290–7305, Mexico City, Mexico. Association for Computational Linguistics.

Nicole Meister, Carlos Guestrin, and Tatsunori Hashimoto. 2024. Benchmarking distributional alignment of large language models. Preprint, arXiv:2411.05403.

Christopher Nguyen. 2024. Value capture. Journal of Ethics and Social Philosophy, 27(3).

Matthias Orlikowski, Jiaxin Pei, Paul Röttger, Philipp Cimiano, David Jurgens, and Dirk Hovy. 2025. Beyond demographics: Fine-tuning large language models to predict individuals’ subjective text perceptions. Preprint, arXiv:2502.20897.

Matthias Orlikowski, Paul Röttger, Philipp Cimiano, and Dirk Hovy. 2023. The ecological fallacy in annotation: Modeling human label variation goes beyond sociodemographics. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 1017– 1029, Toronto, Canada. Association for Computational Linguistics.

Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. 2023. Generative agents: Interactive simulacra of human behavior. Preprint, arXiv:2304.03442.

Joon Sung Park, Carolyn Q. Zou, Aaron Shaw, Benjamin Mako Hill, Carrie Cai, Meredith Ringel Morris, Robb Willer, Percy Liang, and Michael S. Bernstein. 2024. Generative agent simulations of 1,000 people. Preprint, arXiv:2411.10109.

Barbara Plank. 2022. The “problem” of human label variation: On ground truth in data, modeling and evaluation. In Proceedings ofthe 2022 Conference on Empirical Methods in Natural Language Processing, pages 10671–10682, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Sriyash Poddar, Yanming Wan, Hamish Ivison, Abhishek Gupta, and Natasha Jaques. 2024. Personalizing reinforcement learning from human feedback with variational preference learning. In Advances in Neural Information Processing Systems, volume 37, pages 52516–52544. Curran Associates, Inc.

Vinodkumar Prabhakaran, Christopher Homan, Lora Aroyo, Aida Mostafazadeh Davani, Alicia Parrish, Alex Taylor, Mark Diaz, Ding Wang, and Gregory Serapio-García. 2024. GRASP: A disagreement analysis framework to assess group associations in perspectives. In Proceedings of the 2024 Conference of the North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 3473–3492, Mexico City, Mexico. Association for Computational Linguistics.

Filip Radlinski, Krisztian Balog, Fernando Diaz, Lucas Dixon, and Ben Wedin. 2022. On natural language user profiles for transparent and scrutable recommendation. In Proceedings ofthe 45th International ACM SIGIR Conference on Research and Development in

Information Retrieval, SIGIR ’22, page 2863–2874, New York, NY, USA. Association for Computing Machinery.

Jerome Ramos, Hossein A. Rahmani, Xi Wang, Xiao Fu, and Aldo Lipani. 2024. Transparent and scrutable recommendations using natural language user profiles. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13971–13984, Bangkok, Thailand. Association for Computational Linguistics.

John Rawls. 2005. Political Liberalism, expanded edition edition. Columbia University Press, New York.

Shibani Santurkar, Esin Durmus, Faisal Ladhak, Cinoo Lee, Percy Liang, and Tatsunori Hashimoto. 2023. Whose opinions do language models reflect? Preprint, arXiv:2303.17548.

Anand Siththaranjan, Cassidy Laidlaw, and Dylan Hadfield-Menell. 2024. Distributional preference learning: Understanding and accounting for hidden context in rlhf. Preprint, arXiv:2312.08358.

Taylor Sorensen, Liwei Jiang, Jena D. Hwang, Sydney Levine, Valentina Pyatkin, Peter West, Nouha Dziri, Ximing Lu, Kavel Rao, Chandra Bhagavatula, Maarten Sap, John Tasioulas, and Yejin Choi. 2024a. Value kaleidoscope: Engaging ai with pluralistic human values, rights, and duties. Proceedings of the AAAI Conference on Artificial Intelligence, 38(18):19937–19947.

Taylor Sorensen, Jared Moore, Jillian Fisher, Mitchell Gordon, Niloofar Mireshghallah, Christopher Michael Rytting, Andre Ye, Liwei Jiang, Ximing Lu, Nouha Dziri, Tim Althoff, and Yejin Choi. 2024b. A roadmap to pluralistic alignment. Preprint, arXiv:2402.05070.

Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, Soroosh Mariooryad, Yifan Ding, Xinyang Geng, Fred Alcober, Roy Frostig, Mark Omernick, Lexi Walker, Cosmin Paduraru, Christina Sorokin, and 1118 others. 2024. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. Preprint, arXiv:2403.05530.

Michael Henry Tessler, Michiel A. Bakker, Daniel Jarrett, Hannah Sheahan, Martin J. Chadwick, Raphael Koster, Georgina Evans, Lucy Campbell-Gillingham, Tantum Collins, David C. Parkes, Matthew Botvinick, and Christopher Summerfield. 2024. Ai can help humans find common ground in democratic deliberation. Science, 386(6719):eadq2852.

Nenad Tomasev, Kevin R. McKee, Jackie Kay, and Shakir Mohamed. 2021. Fairness for unobserved characteristics: Insights from technological impacts on queer communities. In Proceedings ofthe 2021 AAAI/ACM Conference on AI, Ethics, and Society, AIES ’21, page 254–265, New York, NY, USA. Association for Computing Machinery.

Nikolas Vitsakis, Amit Parekh, and Ioannis Konstas. 2024. Voices in a crowd: Searching for clusters of unique perspectives. Preprint, arXiv:2407.14259.

Angelina Wang, Jamie Morgenstern, and John P. Dickerson. 2025. Large language models that replace human participants can harmfully misportray and flatten identity groups. Nature Machine Intelligence.

Mitchell Wortsman, Gabriel Ilharco, Samir Yitzhak Gadre, Rebecca Roelofs, Raphael Gontijo-Lopes, Ari S. Morcos, Hongseok Namkoong, Ali Farhadi, Yair Carmon, Simon Kornblith, and Ludwig Schmidt. 2022. Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time. Preprint, arXiv:2203.05482.

Yilun Xu, Shengjia Zhao, Jiaming Song, Russell Stewart, and Stefano Ermon. 2020. A theory of usable information under computational constraints. Preprint, arXiv:2002.10689.

Michael JQ Zhang, Zhilin Wang, Jena D. Hwang, Yi Dong, Olivier Delalleau, Yejin Choi, Eunsol Choi, Xiang Ren, and Valentina Pyatkin. 2024. Diverging preferences: When do annotators disagree and do models know? Preprint, arXiv:2410.14632.

Saizheng Zhang, Emily Dinan, Jack Urbanek, Arthur Szlam, Douwe Kiela, and Jason Weston. 2018. Personalizing dialogue agents: I have a dog, do you have pets too? In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2204–2213, Melbourne, Australia. Association for Computational Linguistics.

## Appendix - Table of contents

• Appendix A: Reproducibility details

• Appendix B: More on approaches to modelling variation.

• Appendix C: Additional experiments.

• Appendix D: Discussion of potential applications and extensions

• Appendix E: The prompts used for the encoders and decoders.

• Appendix F: Preprocessing and demographic information for all datasets.

• Appendix G: Full results for all experiments across all rater representations.

• Appendix H: The profile clusters found and used in the profile cluster experiments.

• Appendix I: Ten random value profiles for each dataset and model (gemma2-9b, gemma2-27b, gemini).

## A Reproducibility Details

Here, we include additional experimental details to aid reproducibility.

Dataset Preprocessing We carried out the following preprocessing steps for the datasets - DIC: used the larger subset (990); HL: selected raters with at least nine responses; HK: randomly selected 5k raters and binarized annotations; OQA: randomly selected a wave for experiments (Wave 27); PR: select annotations from first conversation turn and compared the chosen response to the next highest rated response; VP: Treat each value, right, or duty as a unique annotator. Finally, for all datasets we filtered to annotators that had at least four responses.

Decoder hyperparameters: model: gemma2-9b-pt (Gemma Team et al., 2024), batch size: 4, learning rate: 1e-7, gradient clipping: 50.

fp32 unembedding layer: Gemma 2 (Gemma Team et al., 2024) natively uses bf16. However, we found that this caused heavy quantization among high-probability logits (e.g., the valid responses). As such, we cast the embedding/unembedding parameters to fp32 before training, which allowed for higher precision distributions, important for calibration and expressivity.

Fit/eval partition details: For each rater $r _ { i } ,$ we draw $| \mathcal { D } _ { i } ^ { \mathrm { f i t } } | \sim \mathcal { U } ( \{ 2 , \dots , | \mathcal { D } _ { i } | - 2 \} )$ and set $| { \mathcal D } _ { i } ^ { \mathrm { e v a l } } | ~ = ~ | { \mathcal D } _ { i } | - | { \mathcal D } _ { i } ^ { \mathrm { f i t } } |$ to ensure that we have variable-sized fit/eval splits with at least two instances each. Value profile encoders use all $D _ { i } ^ { \mathrm { f i t } }$ instances and the decoders with in-context information $E _ { n }$ use the first min $( n , | D _ { i } ^ { \mathrm { f i t } } | )$ examples from $| D _ { i } ^ { \mathrm { f i t } } |$ . This means that value profiles are fit with a variable number of ratings.

Simulating an annotator population instance selection: We selected the minimum number of instances per dataset as roughly the median number of annotations per instance: 3 for HL, 5 for HK, and 5 for VP. This was selected to try to ensure 1) that we had as many instances as possible and 2) that we had enough raters to have a high-precision estimate of actual rater agreement.

## B More on Approaches to Modelling Variation

In Figure A1, we flesh out more of the comparisons between various modelling approaches characterized in §1.

## C Additional Experiments and Results

## C.1 Predictive Power of Demographic Groups

In addition to presenting the decoder with all rater demographic variables at once (i.e., intersectional demographics), we also train a decoder for each demographic dimension individually. This allows us 1) to see the extent to which grouping individuals based on demographic dimensions is predictive, and 2) which demographic dimensions contain the most usable information for any given dataset. We also train a decoder using all demographic information plus the value profiles. See Figure A5 for results. Some main findings include:

Grouping by demographics does not add significant predictive power. Grouping individuals based on individual demographic dimensions did not significantly improve predictive power, except for OQA, where political ideology/party and religious affiliation/attendance were most informative.

Value profiles and demographics can be complementary. Combining value profiles and demographics resulted in performance as good as or better than either one individually. This suggests that the decoder can leverage both types of information when relevant, e.g. ignore irrelevant information when it is not useful (cf. demographics in DIC/HK) and combine complementary information when useful (cf. OQA/HL).

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Standard modeling</td><td rowspan=1 colspan=1>Distributionalpopulation modeling</td><td rowspan=1 colspan=1>Group modeling(single answer)</td><td rowspan=1 colspan=1>Group modeling(distributional)</td><td rowspan=1 colspan=1>Individual modeling</td></tr><tr><td rowspan=1 colspan=1>Description</td><td rowspan=1 colspan=1>Assumes single correct answer,any variance is noise X</td><td rowspan=1 colspan=1>Model the distribution ofresponses </td><td rowspan=1 colspan=1>Model a group&#x27;s answer(assuming each grouphas one answer)</td><td rowspan=1 colspan=1>Model a group&#x27;sdistribution of answers</td><td rowspan=1 colspan=1>Model an individual&#x27;sanswers </td></tr><tr><td rowspan=1 colspan=1>Target</td><td rowspan=1 colspan=1>Single response(interpersonal variation is noiseX)</td><td rowspan=1 colspan=1>Distribution of responses(variation is signal )</td><td rowspan=1 colspan=1>Single group response(inter-group variation issignal ,intra-group variation isnoise X);</td><td rowspan=1 colspan=1>Group&#x27;s distribution ofresponses </td><td rowspan=1 colspan=1>Single response(interpersonal variation issignal </td></tr><tr><td rowspan=1 colspan=1>Overlaprequirement</td><td rowspan=1 colspan=1>No instance overlap required∇</td><td rowspan=1 colspan=1>Many annotators labelsame instance X</td><td rowspan=1 colspan=1>Many annotators fromeach group label sameinstance X</td><td rowspan=1 colspan=1>Many annotators fromeach group label sameinstance X</td><td rowspan=1 colspan=1>No instance overlaprequired </td></tr><tr><td rowspan=1 colspan=1>Stereotypingrisk</td><td rowspan=1 colspan=1>High X</td><td rowspan=1 colspan=1>Lower </td><td rowspan=1 colspan=1>High (no allowed in-group variation) X</td><td rowspan=1 colspan=1>Lower </td><td rowspan=1 colspan=1>Lower </td></tr><tr><td rowspan=1 colspan=1>Know whodisagrees orwhy?</td><td rowspan=1 colspan=1>NoX</td><td rowspan=1 colspan=1>NoX</td><td rowspan=1 colspan=1>Between groups, yes Within groups, no X</td><td rowspan=1 colspan=1>Between groups, yes Within groups, no X</td><td rowspan=1 colspan=1>Yes ∇</td></tr><tr><td rowspan=1 colspan=1>Flexibility ofpopulationmodeling</td><td rowspan=1 colspan=1>NoX</td><td rowspan=1 colspan=1>Low, only on populationdistribution trained on</td><td rowspan=1 colspan=1>Medium, on arbitrarygroup mixtures </td><td rowspan=1 colspan=1>Medium, on arbitrarygroup mixtures</td><td rowspan=1 colspan=1>High, for arbitrarypopulation viaaggregation ∇</td></tr></table>

Figure A1: Comparisons of various modelling approaches and their tradeoffs with respect to modelling variation

![](images/4207027d0642c90073e014480cbe25972fccb0b0e1ec092d001078a069a888ff.jpg)  
Figure A2: An illustrative plot on fictional data for measuring -info.

## C.2 How does the method generalize to free-form text?

For all experiments in the paper, rater annotations were categorical/ordinal responses to a small, finite number of options. This decision was made largely because of a lack of adequate datasets with more complex annotations. However, the question remains - how does the method generalize to freeform text outputs?

One (and only one) of our datasets, Habermas (Tessler et al., 2024), has free-form rater outputs: the justification that people gave for why they gave the likert response that they did. These descriptions are usually a few sentences long, and contain interesting value information. To get a data point of how our method generalizes to free-form text, we also train a decoder designed to output textual justifications on this dataset (results in Figure A6).

Similar to the categorical results, including more examples does indeed help test perplexity over the no information setting. However, demographics and profiles are not able to help significantly. We have two theories as to why this is the case. Firstly, text contains not only value information, but also stylistic and syntactic information - for example, some raters begin every justification with the same phrase, and others write short vs. long justifications. Thus, in-context examples communicate both value-relevant information and syntactic information, and it is difficult to tell which is causing the decrease in perplexity. Secondly, Habermas was our smallest dataset, making conclusions difficult to decisively draw for even the discrete likert-scale setting. Thus, it is possible that this negative result is in part due to the decoder being underfit, and that value profiles would be able to provide predictive information with additional training. As these results are only on one (small) dataset, we believe that testing generality of the method to free-form text is a promising avenue for future work.

## C.3 Zero-shot decoder performance

For all experiments in the main paper, we train a decoder (using SFT) on a set of train raters and evaluate them on held out test raters. While this is necessary for estimating rater information, we are also curious to know: how well can a value profile decoder perform without dataset-specific training? Specifically, we evaluate on the following settings:

![](images/7934151d13c351fabb8e19fc260134afd71a9e61eef871c2c8f857f9671d3176.jpg)

![](images/ca609e02e79f13c1ec788ed7c7b97b1fde661535689120d8204720f524e9281b.jpg)  
Figure A3: Held-out accuracies across rater representations. Accuracies are reported for the same training runs as Figure 3, with very similar takeaways/results.  
Figure A4: For DICES, the four profile clusters cut across demographic groups along all dimensions.

• Pretrained/base model: Prompted base model gemma2-9b-pt.

• Instruction-tuned model: Prompted instruction-tuned model gemma2-9b-it.

• Souped model (Wortsman et al., 2022): Average the model weights from the trained decoders on all datasets except for the evaluation dataset.

• Trained model: For comparison, we also show results for the trained model.

Performance and calibration results are reported in Figure A7.

Some results include:

• As expected, the trained models both offer the best performance and calibration.

• Instruction-tuned models generally get higher accuracy than base models (5/6 datasets), but base models generally get lower loss (4/6 datasets) due to better calibration.

• The souped models (finetuned from pretrained model) get the same or lower loss as the base models on all datasets, showing some ability to generalize to novel datasets.

All in all, training seems important for learning calibration, and there is some demonstrated ability to generalize from one dataset to another via souping. Additional work exploring how to maintain calibration and performance on out-of-distribution dataset settings is an interesting avenue for future work.

## D Applications and Extensions

Given a set of value profiles and well-calibrated, trained decoders, there are many possible exciting

![](images/c90f0891948bfb8a43530ec5fe75cdf8e2e4c6daea1d14fb9be9bc67f8a05a11.jpg)  
Figure A5: Performance using one demographic at a time, all demographics, value profiles, and all demographics with the value profile. No information and the max examples settings are also reported as baselines. The four most predictive demographics (as measured by test loss) are reported for each dataset, results for the remaining demographics can be found in Appendix G.

![](images/1d02054551e6c28629e7b07f6eb5968b22cd3544d1bf901e69c4e4e96b94a0f8.jpg)  
Figure A6: Results when testing our method on predicting textual rater justifications.

applications. We list a few here.

## D.1 Disentangling (Value)-Epistemic and Aleatoric Uncertainty

In the context of modeling human variation, uncertainty can arise from two distinct sources: epistemic uncertainty (reducible through rater information) and aleatoric uncertainty (irreducible random variation). With value profiles, we can further look at value-epistemic uncertainty, or uncertainty that can be reduced by better understanding a rater’s values.

Specifically, given a set of instances, raters, and their annotations, we can measure the proportion of total uncertainty that can be attributed to value differences versus inherent randomness:

• Total Uncertainty: The entropy of ratings given just the instance, $H _ { V } ( Y | X )$

• Value-Epistemic Uncertainty: The information gained by knowing value profiles, $I _ { V } ( V ( R ) \to \ V | X ) = H _ { V } ( Y | X )$ H<sub>V</sub>(Y X, V (R))

• Aleatoric Uncertainty: The remaining uncertainty after conditioning on both instance and value profiles, $H _ { V } ( Y | X , V ( R ) )$

The ratio $I _ { V } ( V ( R ) \to Y | X ) / H _ { V } ( Y | X )$ represents the fraction of uncertainty that is valueepistemic (reducible by knowing values), while $H _ { V } ( Y | X , V ( R ) ) / H _ { V } ( Y | X )$ represents the fraction that is aleatoric (irreducible even with value knowledge).

Instance-level uncertainty can similarly be measured by looking at $H _ { V } ( Y | x ) , I _ { V } ( V ( R ) \to Y | x )$ and $H _ { V } ( Y | x , V ( R ) )$ . Similar definitions also exist for any other rater representation.

We plot instance-level value-epistemic vs. aleatoric uncertainty for all instances in each dataset in Figure A8.

Such analyses and information may be useful for determining which instances have higher or lower disagreement and whether that disagreement is due to value-relevant factors or other factors.

![](images/288aa96de426e51612a4f7d368b36182996c6b3db7017fec69fb91538e718551.jpg)  
Figure A7: Results and calibration plot for zero-shot results for pretrained/base models, instruction-tuned models, and souped models on all but the dataset to evaluate. Results are compared to the decoder trained on the dataset.

## D.2 Identifying instance-specific value information

Each instance may have particular values which are more or less relevant for the instance as well. Using value decoders, one can estimate the relevance of a value for an instance with $I _ { V } ( v  Y | x )$ . This could be useful in cases such as if one wants to know what values to survey raters for for a particular instance.

## D.3 Rater difficulty

Some raters may more easily be modeled by value profiles (or profile clusters) than others. For example, given a set of candidate value profiles (or, value profile clusters), one could measure the test loss for a rater given the optimal assignment. The lower the test loss, the more easily modeled they are by the value profile, the higher the test loss, the more they may not be easily explained by a value profile. In this way, one could find raters that either a) are not easily modeled by a value profile in the current system or b) may be providing low-quality (or random) judgements.

## D.4 Other applications

Other potential applications include:

• Designing an active learning system to select

instances for a rater to annotate that are most likely to provide value-relevant information;

• Exploring which groups are best or worst represented with value profiles;

• Building a system to help someone explore their own values (see §11);

or more.

## E Prompts

See Figure A9 for the encoder prompt and Figure A10 for the decoder prompt used for all experiments.

## F Data

## F.1 Dataset Preprocessing Details

We carried out the following preprocessing steps for the datasets - DIC: used the larger subset (990); HL: selected raters with at least nine responses; HK: randomly selected 5k raters and binarized annotations; OQA: randomly selected a wave for experiments (Wave 27); PR: select annotations from first conversation turn and compared the chosen response to the next highest rated response; VP: Treat each value, right, or duty as a unique annotator. Finally, for all datasets we filtered to annotators that had at least four responses.

![](images/bcd72b2810058a4186bd47de3527f1ed29d0e38256ef5c4be336626976c20a49.jpg)  
Figure A8: Value-Epistemic Uncertainty (a.k.a., Information Gain from Value Profile) vs. the Irreducible Entropy (or Aleatoric Uncertainty) for each instance in each dataset, colored by label.

## F.2 All Dataset Demographic Variables

Refer to Table 3 to see all demographic variables contained in each dataset.

## G Detailed Results

The full results for each dataset can be found in:

1. OpinionQA: Table 4

2. Hatespeech-Kumar: Table 5

3. DICES: Table 6

4. ValuePrism: Table 7

5. Habermas-Likert: Table 8

6. Prism: Table 9

<table><tr><td rowspan=1 colspan=1>Dataset</td><td rowspan=1 colspan=1>Demographics</td></tr><tr><td rowspan=1 colspan=1>OpinionQA (W27)</td><td rowspan=1 colspan=1>CREGION, SEX, EDUCATION, CITIZEN, MARITAL, INCOME, RACE, RELIG, RELIGAT-TEND, POLPARTY, POLIDEOLOGY</td></tr><tr><td rowspan=1 colspan=1>Habermas-Likert</td><td rowspan=1 colspan=1>party_id, religion, age, education, ethnicity, gender_id, immigration_status, income, region</td></tr><tr><td rowspan=1 colspan=1>DICES</td><td rowspan=1 colspan=1>rater_gender, rater_locale, rater_race_raw, rater_age, rater_education</td></tr><tr><td rowspan=2 colspan=1>Hatespeech-Kumar</td><td rowspan=2 colspan=1>gender, gender_other, race, identify_as_transgender, 1gbtq_status, education, age_range,political_affilation, is_parent, religion_important, technology_impact, uses_media_social,uses_media_news, uses_media_video, uses_media_forums, personally_seen_toxic_content, per-sonally_been_target, toxic_comments_problem</td></tr><tr><td rowspan=1 colspan=1>personall</td></tr><tr><td rowspan=1 colspan=1>ValuePrism</td><td rowspan=1 colspan=1>None, but has &quot;ground truth&quot; value profiles of the original value / right / duty.</td></tr><tr><td rowspan=1 colspan=1>Prism</td><td rowspan=1 colspan=1>lm_familiarity,  lm_indirect_use,  lm_direct_use,lm_frequency_use,   lm_usecases,self_description, system_string, religion, stated_prefs, order_lm_usecases, order_stated_prefs,age, gender, employment_status, education, marital_status, english_proficiency, study_locale,location, ethnicity</td></tr></table>

Table 3: Dataset demographic variables

## Encoder Prompt $Q _ { \phi }$

![](images/eb038d4066151e8332e2b8365f352b0928b621aa212c126a8dea67659ebb5850.jpg)

## H Profile Clusters

## H.1 DICES

## H.1.1 2 clusters (DICES)

Cluster Profile 1: High tolerancefor offensive language and behavior; Focus on intent rather than impact; Narrow definition of toxicity; Prioritization ofconversationflow over emotional safety; Belief in personal responsibility for emotional reactions; Discomfort with addressing conflict or negativity; Lack of awareness of subtle forms of toxicity; Potential bias towards certain topics or demographics; General optimism or positivity

Cluster Profile 2: Strong reaction to overt negativity or hate speech; Sensitivity to potential harm, even if unintentional; Prioritizes intention over impact; Tolerance for bluntness and disagreement; Focus on explicit language rather than subtle cues;

Possible political bias; Inconsistency in application of criteria

## H.1.2 4 clusters (DICES)

Cluster Profile 1: Strong reaction to overt negativity or hate speech; Sensitivity to potential harm, even ifunintentional; Prioritizes intention over impact; Tolerance for bluntness and disagreement; Focus on explicit language rather than subtle cues; Possible political bias; Inconsistency in application ofcriteria

Cluster Profile 2: High tolerancefor offensive language and controversial topics; Focus on intent over impact; Emphasis on personal responsibility and resilience; Broad definition of acceptable discourse; Potential desensitization to online interactions; Possible lack of understanding of subtle forms of toxicity; Potential bias towards certain viewpoints; Belief in the importance of context

Cluster Profile 3: Direct insults and aggressive language are toxic; Simple greetings and neutral topics are not toxic; Humor and playful banter are not necessarily toxic; The presence of controversial opinions does not automatically equal toxicity; Context and intent matter, but are difficult tofully assess; Statements suggesting or promoting illegal or harmful activities are toxic; The rater may have a relatively high tolerancefor differing viewpoints

Cluster Profile 4: High tolerancefor offensive language and behavior; Focus on intent rather than impact; Narrow definition of toxicity; Prioritization of conversation flow over emotional safety; Belief in personal responsibility for emotional reactions; Discomfort with addressing conflict or negativity; Lack of awareness of subtle forms of toxicity; Potential bias towards certain topics or demographics; General optimism or positivity

![](images/26824fe3f1b350c929a28b31e88256163ef2c5a4788b75a663a8bed12b00e009.jpg)  
Figure A10: Decoder prompt

## H.1.3 8 clusters (DICES)

Cluster Profile 1: Strong reaction to overt negativity or hate speech; Sensitivity to potential harm, even if unintentional; Prioritizes intention over impact; Tolerance for bluntness and disagreement; Focus on explicit language rather than subtle cues; Possible political bias; Inconsistency in application of criteria

Cluster Profile 2: Direct insults and aggressive language are toxic; Simple greetings and neutral topics are not toxic; Humor and playful banter are not necessarily toxic; The presence of controversial opinions does not automatically equal toxicity; Context and intent matter, but are difficult to fully assess; Statements suggesting or promoting illegal or harmful activities are toxic; The rater may have a relatively high tolerance for differing viewpoints

Cluster Profile 3: Strong reaction to discussions ofself-harm and suicide; Sensitivity to discussions about race and sexual orientation; Discomfort with overtly sexual conversations or innuendo; Low tolerance for aggressive or rude language; A broad definition of "toxic"; Uncertainty around certain topics; A belief that context matters; Prioritizes safety and well-being

Cluster Profile 4: High tolerancefor offensive language and behavior; Focus on intent rather than impact; Narrow definition of toxicity; Prioritization ofconversationflow over emotional safety; Belief in personal responsibilityfor emotional reactions; Discomfort with addressing conflict or negativity; Lack ofawareness ofsubtleforms oftoxicity; Potential bias towards certain topics or demographics; General optimism or positivity

Cluster Profile 5: Emphasis on intent over outcome; High tolerance for disagreement and differing opinions; Forgivenessfor misunderstandings and apologies; Political neutrality or apathy; Discomfort with discussions about illegal activities; Leniency towards casual conversation and humor; Inconsistency in applying standards; Focus on last turn in the conversation

Cluster Profile 6: High tolerancefor controversial topics and strong opinions; Emphasis on intention over impact; Beliefinfreedom ofexpression; Acceptance ofdark humor and sarcasm; Forgiveness for immaturity or ignorance; Discomfort with discussions directly involving their personal advice on difficult topics; May not be detecting subtle forms oftoxicity; Possibly prioritizing engagement and entertainment over safety and inclusivity

<table><tr><td rowspan=1 colspan=2>Name</td><td rowspan=1 colspan=1>Test Accuracy</td><td rowspan=1 colspan=1>Test Loss</td><td rowspan=1 colspan=1>Usable Info (nats)</td><td rowspan=1 colspan=1>Info Preserved</td></tr><tr><td rowspan=2 colspan=2>no infodem CITIZEN</td><td rowspan=1 colspan=1>52.0 (±0.14)</td><td rowspan=1 colspan=1>0.987 (±0.002)</td><td rowspan=1 colspan=1>0.000</td><td rowspan=12 colspan=1>(0%)</td></tr><tr><td rowspan=1 colspan=1>51.9 (±0.08)</td><td rowspan=1 colspan=1>0.987 (±0.002)</td><td rowspan=1 colspan=1>0.000</td></tr><tr><td rowspan=1 colspan=2>dem CREGION</td><td rowspan=1 colspan=1>51.9 (±0.12)</td><td rowspan=1 colspan=1>0.987 (±0.002)</td><td rowspan=1 colspan=1>0.000</td></tr><tr><td rowspan=3 colspan=2>dem EDUCATION</td><td rowspan=1 colspan=1>52.1 (±0.10)</td><td rowspan=1 colspan=1>0.985 (±0.002)</td><td rowspan=1 colspan=1>0.002</td></tr><tr><td rowspan=4 colspan=2>dem POLIDEOLOGY</td><td rowspan=2 colspan=1>52.0 (±0.13)</td><td rowspan=2 colspan=1>0.985 (±0.002)</td><td rowspan=2 colspan=1>0.002</td></tr><tr><td rowspan=2 colspan=1>dem MARITALTA</td></tr><tr><td rowspan=1 colspan=1>52.4 (±0.07)</td><td rowspan=1 colspan=1>0.983 (±0.002)</td><td rowspan=1 colspan=1>0.004</td></tr><tr><td rowspan=1 colspan=1>dem POLIDEOL</td><td rowspan=1 colspan=1>59.2 (±0.07)</td><td rowspan=1 colspan=1>0.896 (±0.002)</td><td rowspan=1 colspan=1>0.091</td></tr><tr><td rowspan=1 colspan=2>dem POLPARTY</td><td rowspan=1 colspan=1>57.1 (±0.17)</td><td rowspan=1 colspan=1>0.918 (±0.002)</td><td rowspan=1 colspan=1>0.069</td></tr><tr><td rowspan=1 colspan=2>dem RACE</td><td rowspan=1 colspan=1>52.5 (±0.15)</td><td rowspan=1 colspan=1>0.983 (±0.002)</td><td rowspan=1 colspan=1>0.004</td></tr><tr><td rowspan=1 colspan=2>dem RELIG</td><td rowspan=1 colspan=1>53.7 (±0.13)</td><td rowspan=1 colspan=1>0.965 (±0.003)</td><td rowspan=1 colspan=1>0.022</td></tr><tr><td rowspan=1 colspan=2>dem RELIGATTEND</td><td rowspan=1 colspan=1>53.5 (±0.08)</td><td rowspan=1 colspan=1>0.971 (±0.002)</td><td rowspan=1 colspan=1>0.016</td></tr><tr><td rowspan=1 colspan=2>dem SEX</td><td rowspan=1 colspan=1>52.1 (±0.11)</td><td rowspan=1 colspan=1>0.985 (±0.001)</td><td rowspan=1 colspan=1>0.002</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>dem identity columns</td><td rowspan=1 colspan=1>53.3 (±0.32)</td><td rowspan=1 colspan=1>0.972 (±0.004)</td><td rowspan=1 colspan=1>0.015</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>dem value columns</td><td rowspan=1 colspan=1>60.1 (±0.41)</td><td rowspan=1 colspan=1>0.881 (±0.008)</td><td rowspan=1 colspan=1>0.106</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>dem (all)</td><td rowspan=1 colspan=1>61.0 (±0.07)</td><td rowspan=1 colspan=1>0.859 (±0.002)</td><td rowspan=1 colspan=1>0.128</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>profile cluster-2</td><td rowspan=1 colspan=1>59.3 (±0.15)</td><td rowspan=1 colspan=1>0.899 (±0.002)</td><td rowspan=1 colspan=1>0.088</td><td rowspan=1 colspan=1>56%</td></tr><tr><td rowspan=1 colspan=2>profile cluster-4</td><td rowspan=1 colspan=1>60.1 (±0.22)</td><td rowspan=1 colspan=1>0.878 (±0.003)</td><td rowspan=1 colspan=1>0.109</td><td rowspan=1 colspan=1>69%</td></tr><tr><td rowspan=1 colspan=2>profile cluster-8</td><td rowspan=1 colspan=1>60.8 (±0.18)</td><td rowspan=1 colspan=1>0.866 (±0.003)</td><td rowspan=1 colspan=1>0.121</td><td rowspan=1 colspan=1>77%</td></tr><tr><td rowspan=1 colspan=2>profile 9b</td><td rowspan=1 colspan=1>57.5 (±1.10)</td><td rowspan=1 colspan=1>0.918 (±0.016)</td><td rowspan=1 colspan=1>0.069</td><td rowspan=1 colspan=1>43%</td></tr><tr><td rowspan=1 colspan=2>profile 27b</td><td rowspan=1 colspan=1>61.1 (±0.14)</td><td rowspan=1 colspan=1>0.866 (±0.002)</td><td rowspan=1 colspan=1>0.120</td><td rowspan=1 colspan=1>76%</td></tr><tr><td rowspan=1 colspan=2>profile gni</td><td rowspan=1 colspan=1>60.3 (±0.13)</td><td rowspan=1 colspan=1>0.870 (±0.004)</td><td rowspan=1 colspan=1>0.117</td><td rowspan=1 colspan=1>74%</td></tr><tr><td rowspan=1 colspan=2>dem+profile gni</td><td rowspan=1 colspan=1>62.4 (±0.11)</td><td rowspan=1 colspan=1>0.829 (±0.002)</td><td rowspan=1 colspan=1>0.158</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>1 ex</td><td rowspan=1 colspan=1>53.9 (±0.46)</td><td rowspan=1 colspan=1>0.964 (±0.005)</td><td rowspan=1 colspan=1>0.023</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>2 ex</td><td rowspan=1 colspan=1>56.1 (±0.08)</td><td rowspan=1 colspan=1>0.937 (±0.002)</td><td rowspan=1 colspan=1>0.050</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>4 ex</td><td rowspan=1 colspan=1>58.0 (±0.34)</td><td rowspan=1 colspan=1>0.906 (±0.005)</td><td rowspan=1 colspan=1>0.081</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>8ex</td><td rowspan=1 colspan=1>60.0 (±0.11)</td><td rowspan=1 colspan=1>0.870 (±0.003)</td><td rowspan=1 colspan=1>0.117</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>16 ex</td><td rowspan=1 colspan=1>61.4 (±0.32)</td><td rowspan=1 colspan=1>0.843 (±0.005)</td><td rowspan=1 colspan=1>0.143</td><td rowspan=3 colspan=1>(100%)</td></tr><tr><td rowspan=1 colspan=2>32 ex</td><td rowspan=1 colspan=1>62.3 (±0.12)</td><td rowspan=1 colspan=1>0.829 (±0.003)</td><td rowspan=2 colspan=1>0.158</td></tr><tr><td rowspan=1 colspan=2>majority class acc./dataset entropy</td><td rowspan=1 colspan=1>39.7 (±0.00)</td><td rowspan=1 colspan=1>1.290 (±0.000)</td></tr></table>

Table 4: OpinionQA Performance Metrics (Model: gemma2-9b-pt) Other datasets: Appendix: G

Cluster Profile 7: Discomfort with sexual topics and exploitation; Sensitivity to personal attacks and insults; Low tolerance for manipulative or misleading behavior; Dislike ofaggressive or confrontational language; High tolerancefor sarcasm and playful banter; Belief that repetitive or nonsensical conversations are not necessarily toxic; Uncertainty about the line between persistent questioning and harassment; Possible leniency towards conversations that are simply awkward or uncomfortable; Emphasis on intent and context; Potential bias towardfocusing on the last statement

Cluster Profile 8: High tolerancefor offensive language and controversial topics; Focus on intent over impact; Emphasis on personal responsibility and resilience; Broad definition of acceptable discourse; Potential desensitization to online interactions; Possible lack of understanding of subtle forms of toxicity; Potential bias towards certain viewpoints; Beliefin the importance ofcontext

## H.2 Habermas-Likert

## H.2.1 2 clusters (Habermas-Likert)

Cluster Profile 1: Values religious freedom and parental rights; Prioritizes family autonomy over state control; May be religious themselves; Pragmatic or uncertain about online medicine; Weighing competing values; Lack ofknowledge; Beliefin a mixed approach; Values personal responsibility; Prioritizes affordability and access to healthcare; Trusts marketforces to some extent

Cluster Profile 2: Strong disapproval of Theresa May; Public health consciousness; Environmental concern; Beliefin direct democracy; Concern about overpopulation; Openness to government intervention; Possible leaning towards left-leaning or liberal politics; Pragmatism and nuanced views; UK-centric perspective

## H.2.2 4 clusters (Habermas-Likert)

Cluster Profile 1: Pro-worker; Value of leisure and rest; Concern for elderly well-being; Potential distrust of government or employers; Belief in social safety nets; Focus on quality of life over economic growth; Generational fairness; Compassion and empathy for those less fortunate; May hold specific political or ideological views

<table><tr><td rowspan=1 colspan=7>Name</td><td rowspan=1 colspan=3>Test Accuracy</td><td rowspan=1 colspan=1>Test Loss</td><td rowspan=1 colspan=2>Usable Info (nats)</td><td rowspan=1 colspan=1>Info Preserved</td></tr><tr><td rowspan=14 colspan=7>no infodem (all)dem personally been targetdem personally seen toxic contentdem age rangedem uses media socialdem uses media newsdem uses media forumsdem toxic comments problemdem technology impactdem religion importantdem racedem political affilationdem 1gbtq status</td><td rowspan=1 colspan=3>70.5 (±0.46)</td><td rowspan=1 colspan=1>0.569 (±0.005)</td><td rowspan=1 colspan=2>0.000</td><td rowspan=5 colspan=1>(0%)</td></tr><tr><td rowspan=2 colspan=3>70.7 (±0.40)</td><td rowspan=1 colspan=1>0.565 (±0.004)</td><td rowspan=1 colspan=2>0.003</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=2 colspan=3>70.6 (±0.40)</td><td rowspan=1 colspan=1>0.568 (±0.004)</td><td rowspan=1 colspan=2>0.000</td></tr><tr><td rowspan=1 colspan=1>0.567 (±0.004)</td><td rowspan=1 colspan=2>0.001</td></tr><tr><td rowspan=1 colspan=3>70.6 (±0.42)</td><td rowspan=1 colspan=1>0.568 (±0.004)</td><td rowspan=1 colspan=2>0.001</td></tr><tr><td rowspan=1 colspan=3>70.6 (±0.38)</td><td rowspan=1 colspan=1>0.568 (±0.004)</td><td rowspan=1 colspan=2>0.001</td><td rowspan=7 colspan=1></td></tr><tr><td rowspan=1 colspan=3>70.7 (±0.35)</td><td rowspan=1 colspan=1>0.568 (±0.004)</td><td rowspan=1 colspan=2>0.001</td></tr><tr><td rowspan=1 colspan=3>70.6 (±0.65)</td><td rowspan=1 colspan=1>0.566 (±0.008)</td><td rowspan=1 colspan=2>0.002</td></tr><tr><td rowspan=1 colspan=3>70.7 (±0.38)</td><td rowspan=1 colspan=1>0.567 (±0.004)</td><td rowspan=1 colspan=2>0.001</td></tr><tr><td rowspan=1 colspan=3>70.6 (±0.38)</td><td rowspan=1 colspan=1>0.569 (±0.004)</td><td rowspan=1 colspan=2>0.000</td></tr><tr><td rowspan=1 colspan=3>71.4 (±0.39)</td><td rowspan=1 colspan=1>0.558 (±0.004)</td><td rowspan=1 colspan=2>0.010</td></tr><tr><td rowspan=1 colspan=3>70.7 (±0.37)</td><td rowspan=1 colspan=1>0.568 (±0.004)</td><td rowspan=1 colspan=2>0.001</td></tr><tr><td rowspan=1 colspan=3>70.7 (±0.41)</td><td rowspan=1 colspan=1>0.568 (±0.004)</td><td rowspan=1 colspan=2>0.001</td><td rowspan=5 colspan=1></td></tr><tr><td rowspan=1 colspan=3>70.6 (±0.36)</td><td rowspan=1 colspan=1>0.569 (±0.004)</td><td rowspan=1 colspan=2>0.000</td></tr><tr><td rowspan=1 colspan=2>dem is parent</td><td></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=3>70.7 (±0.36)</td><td rowspan=1 colspan=1>0.567 (±0.004)</td><td rowspan=1 colspan=2>0.002</td></tr><tr><td rowspan=2 colspan=7>dem identity columnsdem identify as transgender</td><td rowspan=1 colspan=3>lumns</td><td rowspan=1 colspan=1>70.6 (±0.32)</td><td rowspan=1 colspan=2>0.568 (±0.004)</td><td rowspan=1 colspan=1>0.000</td></tr><tr><td rowspan=1 colspan=2>70.5 (±0.35)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.568 (±0.004)</td><td rowspan=1 colspan=2>0.000</td></tr><tr><td rowspan=1 colspan=7>dem gender other</td><td rowspan=1 colspan=2>70.6 (±0.40)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.568 (±0.004)</td><td rowspan=1 colspan=2>0.000</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=7>dem gender</td><td rowspan=1 colspan=3>70.6 (±0.34)</td><td rowspan=1 colspan=1>0.568 (±0.005)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=7>dem education</td><td rowspan=1 colspan=3>70.5 (±0.38)</td><td rowspan=1 colspan=1>0.568 (±0.004)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.001</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=7>dem uses media video</td><td rowspan=1 colspan=2>70.7 (±0.37)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.566 (±0.004)</td><td rowspan=1 colspan=2>0.003</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>dem value col</td><td rowspan=1 colspan=5>umns</td><td rowspan=1 colspan=2>71.0 (±0.35)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.563 (±0.004)</td><td rowspan=1 colspan=2>0.005</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=7>profile cluster-2</td><td rowspan=1 colspan=3>70.1 (±0.52)</td><td rowspan=1 colspan=1>0.572 (±0.006)</td><td rowspan=1 colspan=2>-0.004</td><td rowspan=1 colspan=1>-5%</td></tr><tr><td rowspan=1 colspan=7>profile cluster-4</td><td rowspan=1 colspan=3>72.6 (±0.23)</td><td rowspan=1 colspan=1>0.539 (±0.003)</td><td rowspan=1 colspan=2>0.029</td><td rowspan=1 colspan=1>37%</td></tr><tr><td rowspan=1 colspan=7>profile cluster-8</td><td rowspan=1 colspan=3>73.1 (±0.26)</td><td rowspan=1 colspan=1>0.532 (±0.003)</td><td rowspan=1 colspan=2>0.036</td><td rowspan=1 colspan=1>46%</td></tr><tr><td rowspan=1 colspan=7>profile 9b</td><td rowspan=1 colspan=3>71.3 (±0.35)</td><td rowspan=1 colspan=1>0.554 (±0.003)</td><td rowspan=1 colspan=2>0.014</td><td rowspan=1 colspan=1>18%</td></tr><tr><td rowspan=2 colspan=7>profile 27bprofile gni</td><td rowspan=1 colspan=3>72.3 (±0.18)</td><td rowspan=1 colspan=1>0.543 (±0.002)</td><td rowspan=1 colspan=2>0.026</td><td rowspan=1 colspan=1>33%</td></tr><tr><td rowspan=1 colspan=6></td><td rowspan=1 colspan=3>74.8 (±0.21)</td><td rowspan=1 colspan=1>0.509 (±0.003)</td><td rowspan=1 colspan=2>0.060</td><td rowspan=1 colspan=1>76%</td></tr><tr><td rowspan=1 colspan=2>dem+profile gni</td><td rowspan=1 colspan=3>e gni</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2>75.0 (±0.15)</td><td rowspan=1 colspan=1>0.509 (±0.003)</td><td rowspan=1 colspan=2>0.059</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=3>1 ex</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2>71.6 (±0.31)</td><td rowspan=1 colspan=1>0.553 (±0.003)</td><td rowspan=1 colspan=2>0.016</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>2 ex</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=4></td><td rowspan=1 colspan=1>72.5 (±0.21)</td><td rowspan=1 colspan=2>0.541 (±0.002)</td><td rowspan=1 colspan=1>0.028</td></tr><tr><td rowspan=1 colspan=2>4 ex</td><td></td><td></td><td></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=3>74.1 (±0.22)</td><td rowspan=1 colspan=1>0.521 (±0.002)</td><td rowspan=1 colspan=2>0.047</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=3>8ex</td><td></td><td rowspan=2 colspan=3></td><td rowspan=2 colspan=3>75.5 (±0.21)</td><td rowspan=2 colspan=1>0.500 (±0.001)</td><td></td><td></td><td></td></tr><tr><td rowspan=3 colspan=5></td><td rowspan=1 colspan=2>0.069</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=5>16 ex32 ex</td><td rowspan=1 colspan=3>75.5 (±0.40)</td><td rowspan=1 colspan=1>0.500 (±0.006)</td><td rowspan=1 colspan=2>0.069</td><td rowspan=3 colspan=1>(100%)</td></tr><tr><td rowspan=1 colspan=3>76.3 (±0.33)</td><td rowspan=1 colspan=1>0.489 (±0.003)</td><td rowspan=1 colspan=2>0.079</td></tr><tr><td rowspan=1 colspan=7>majority class acc./dataset entropy</td><td rowspan=1 colspan=3>55.4 (±0.00)</td><td rowspan=1 colspan=1>0.687 (±0.000)</td><td rowspan=1 colspan=2></td></tr></table>

Table 5: Hatespeech-Kumar Performance Metrics (Model: gemma2-9b-pt) Other datasets: Appendix: G

Cluster Profile 2: Strong disapproval of Theresa May; Public health consciousness; Environmental concern; Beliefin direct democracy; Concern about overpopulation; Openness to government intervention; Possible leaning towards left-leaning or liberal politics; Pragmatism and nuanced views; UK-centric perspective

Cluster Profile 3: Altruism and global citizenship; Environmental concern; Collectivism and public health prioritization; Social welfare and beliefin social safety nets; Potentialfor utilitarianism; Nuance and pragmatism; Possible supportfor animal welfare, but with caveats; Acceptance of minor moralflexibility; It is important to remember that these are just inferences based on a limited set of responses. The rater’s true beliefs and values may be more complex and nuanced than what can be determinedfrom this data alone.

Cluster Profile 4: Values religious freedom and parental rights; Prioritizesfamily autonomy over state control; May be religious themselves; Pragmatic or uncertain about online medicine; Weighing competing values; Lack of knowledge; Belief in a mixed approach; Values personal responsibility; Prioritizes affordability and access to healthcare; Trusts marketforces to some extent

## H.2.3 8 clusters (Habermas-Likert)

Cluster Profile 1: Slightly prefers free market principles; Concerned about affordability and access; Cautious about government overreach; Open to social responsibility and regulation where appropriate; Values personal autonomy; Pragmatic and moderate; Indecisive or uninformed on some topics; Potentially influenced by personal experience; Open to persuasion

Cluster Profile 2: Pro-worker; Value ofleisure and rest; Concern for elderly well-being; Potential distrust of government or employers; Belief in social safety nets; Focus on quality oflife over economic growth; Generationalfairness; Compassion and empathy for those less fortunate; May hold specific political or ideological views

<table><tr><td rowspan=1 colspan=6>Name</td><td rowspan=1 colspan=1>Test Accuracy</td><td rowspan=1 colspan=1>Test Loss</td><td rowspan=1 colspan=1>Usable Info (nats)</td><td rowspan=1 colspan=1>Info Preserved</td></tr><tr><td rowspan=5 colspan=6>no infodem rater agedem rater educationdem rater genderdem rater locale</td><td rowspan=1 colspan=1>71.8 (±0.92)</td><td rowspan=1 colspan=1>0.668 (±0.020)</td><td rowspan=1 colspan=1>0.000</td><td rowspan=3 colspan=1>(0%)</td></tr><tr><td rowspan=1 colspan=1>71.5 (±1.09)</td><td rowspan=1 colspan=1>0.671 (±0.020)</td><td rowspan=1 colspan=1>-0.002</td></tr><tr><td rowspan=1 colspan=1>71.1 (±1.21)</td><td rowspan=1 colspan=1>0.673 (±0.020)</td><td rowspan=1 colspan=1>-0.004</td></tr><tr><td rowspan=1 colspan=1>71.7 (±1.13)</td><td rowspan=1 colspan=1>0.669 (±0.020)</td><td rowspan=1 colspan=1>-0.001</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>71.6 (±1.20)</td><td rowspan=1 colspan=1>0.666 (±0.020)</td><td rowspan=1 colspan=1>0.002</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=5 colspan=6>dem rater race rawdem (all)profile cluster-2profile cluster-4profile cluster-8</td><td rowspan=1 colspan=1>71.6 (±1.10)</td><td rowspan=1 colspan=1>0.668 (±0.020)</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>71.6 (±1.27)</td><td rowspan=1 colspan=1>0.667 (±0.021)</td><td rowspan=1 colspan=1>0.002</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>75.6 (±0.51)</td><td rowspan=1 colspan=1>0.605 (±0.012)</td><td rowspan=1 colspan=1>0.064</td><td rowspan=1 colspan=1>60%</td></tr><tr><td rowspan=1 colspan=1>76.4 (±0.67)</td><td rowspan=1 colspan=1>0.576 (±0.015)</td><td rowspan=1 colspan=1>0.093</td><td rowspan=1 colspan=1>88%</td></tr><tr><td rowspan=1 colspan=1>76.9 (±0.69)</td><td rowspan=1 colspan=1>0.570 (±0.013)</td><td rowspan=1 colspan=1>0.098</td><td rowspan=1 colspan=1>93%</td></tr><tr><td rowspan=1 colspan=6>profile 9b</td><td rowspan=1 colspan=1>71.5 (±0.90)</td><td rowspan=1 colspan=1>0.683 (±0.023)</td><td rowspan=1 colspan=1>-0.015</td><td rowspan=1 colspan=1>-14%</td></tr><tr><td rowspan=1 colspan=4>profile 27b</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>69.9 (±1.04)</td><td rowspan=1 colspan=1>0.706 (±0.018)</td><td rowspan=1 colspan=1>-0.038</td><td rowspan=1 colspan=1>-36%</td></tr><tr><td rowspan=1 colspan=4>profile gni</td><td rowspan=2 colspan=2></td><td rowspan=1 colspan=1>75.7 (±0.50)</td><td rowspan=1 colspan=1>0.594 (±0.014)</td><td rowspan=1 colspan=1>0.074</td><td rowspan=1 colspan=1>71%</td></tr><tr><td></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=3>dem+profile gni</td><td rowspan=1 colspan=1>75.4 (±0.62)</td><td rowspan=1 colspan=1>0.598 (±0.015)</td><td rowspan=1 colspan=1>0.070</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=3 colspan=3>1 ex</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=2></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>72.9 (±0.77)</td><td rowspan=1 colspan=1>0.644 (±0.016)</td><td rowspan=1 colspan=1>0.025</td><td rowspan=1 colspan=1></td></tr><tr><td></td><td rowspan=2 colspan=4></td><td rowspan=2 colspan=2>2 ex</td><td rowspan=2 colspan=1></td><td rowspan=1 colspan=1></td><td></td><td></td></tr><tr><td></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>74.1 (±0.76)</td><td rowspan=1 colspan=1>0.625 (±0.014)</td><td rowspan=1 colspan=1>0.044</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=3>4 ex</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=1>75.2 (±0.58)</td><td rowspan=1 colspan=1>0.602 (±0.012)</td><td rowspan=1 colspan=1>0.066</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=3 colspan=6>8ex16 exmajority class acc./dataset entropy</td><td rowspan=1 colspan=1>76.3 (±0.51)</td><td rowspan=1 colspan=1>0.580 (±0.013)</td><td rowspan=1 colspan=1>0.089</td><td rowspan=3 colspan=1>(100%)</td></tr><tr><td rowspan=1 colspan=1>77.0 (±0.66)</td><td rowspan=1 colspan=1>0.563 (±0.014)</td><td rowspan=1 colspan=1>0.105</td></tr><tr><td rowspan=1 colspan=1>70.4 (±0.00)</td><td rowspan=1 colspan=1>0.742 (±0.000)</td><td rowspan=1 colspan=1></td></tr></table>

Table 6: DICES Performance Metrics (Model: gemma2-9b-pt) Other datasets: Appendix: G
<table><tr><td rowspan=1 colspan=6>Name</td><td rowspan=1 colspan=2>Test Accuracy</td><td rowspan=1 colspan=1>Test Loss</td><td rowspan=1 colspan=1>Usable Info (nats)</td><td rowspan=1 colspan=1>Info Preserved</td></tr><tr><td rowspan=1 colspan=6>no info</td><td rowspan=1 colspan=2>59.2 (±0.37)</td><td rowspan=1 colspan=1>0.852 (±0.005)</td><td rowspan=1 colspan=1>0.000</td><td rowspan=2 colspan=1>(0%)-0%</td></tr><tr><td rowspan=1 colspan=6>profile cluster-2</td><td rowspan=1 colspan=2>59.4 (±0.49)</td><td rowspan=1 colspan=1>0.853 (±0.005)</td><td rowspan=1 colspan=1>-0.001</td></tr><tr><td rowspan=1 colspan=6>profile cluster-4</td><td rowspan=1 colspan=2>65.4 (±0.82)</td><td rowspan=1 colspan=1>0.792 (±0.013)</td><td rowspan=1 colspan=1>0.060</td><td rowspan=1 colspan=1>20%</td></tr><tr><td rowspan=1 colspan=6>profile cluster-8</td><td rowspan=1 colspan=2>65.8 (±1.37)</td><td rowspan=1 colspan=1>0.780 (±0.017)</td><td rowspan=1 colspan=1>0.071</td><td rowspan=1 colspan=1>23%</td></tr><tr><td rowspan=1 colspan=6>profile 9b</td><td rowspan=1 colspan=2>74.0 (±0.24)</td><td rowspan=1 colspan=1>0.632 (±0.006)</td><td rowspan=1 colspan=1>0.220</td><td rowspan=1 colspan=1>72%</td></tr><tr><td rowspan=1 colspan=6>profile 27b</td><td rowspan=1 colspan=2>74.6 (±0.39)</td><td rowspan=1 colspan=1>0.615 (±0.008)</td><td rowspan=1 colspan=1>0.237</td><td rowspan=1 colspan=1>78%</td></tr><tr><td rowspan=1 colspan=3>profile gni</td><td></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>77.3 (±0.22)</td><td rowspan=1 colspan=1>0.566 (±0.006)</td><td rowspan=1 colspan=1>0.286</td><td rowspan=11 colspan=1>94%(100%)</td></tr><tr><td rowspan=1 colspan=3>1 ex</td><td></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2>68.1 (±0.29)</td><td rowspan=1 colspan=1>0.738 (±0.006)</td><td rowspan=3 colspan=1>0.1140.157</td></tr><tr><td rowspan=2 colspan=3>2 ex</td><td rowspan=2 colspan=2></td><td rowspan=2 colspan=1></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=2></td><td rowspan=1 colspan=2>70.6 (±0.51)</td><td rowspan=1 colspan=1>0.695 (±0.010)</td></tr><tr><td rowspan=1 colspan=2>4 ex</td><td></td><td></td><td rowspan=1 colspan=2>73.4 (±0.67)</td><td rowspan=1 colspan=1>0.640 (±0.015)</td><td rowspan=1 colspan=1>0.212</td></tr><tr><td rowspan=1 colspan=3>8ex</td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2>75.8 (±0.35)</td><td rowspan=1 colspan=1>0.591 (±0.007)</td><td rowspan=1 colspan=1>0.261</td></tr><tr><td rowspan=1 colspan=6>16 ex</td><td rowspan=1 colspan=2>76.8 (±0.38)</td><td rowspan=1 colspan=1>0.570 (±0.008)</td><td rowspan=1 colspan=1>0.282</td></tr><tr><td rowspan=1 colspan=6>32 ex</td><td rowspan=1 colspan=2>77.9 (±0.35)</td><td rowspan=1 colspan=1>0.547 (±0.007)</td><td rowspan=4 colspan=1>0.3050.3580.361</td></tr><tr><td rowspan=1 colspan=6>ground truth prof</td><td rowspan=1 colspan=1>80.1 (±0.17)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.493 (±0.006)</td></tr><tr><td rowspan=1 colspan=6>ground truth prof+profile gni</td><td rowspan=1 colspan=1>80.3 (±0.27)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.491 (±0.005)</td></tr><tr><td rowspan=1 colspan=6>majority class acc./dataset entropy</td><td rowspan=1 colspan=1>50.4 (±0.00)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>1.004 (±0.000)</td></tr></table>

Table 7: ValuePrism Performance Metrics (Model: gemma2-9b-pt) Other datasets: Appendix: G

Cluster Profile 3: Altruism and global citizenship; Environmental concern; Collectivism and public health prioritization; Social welfare and beliefin social safety nets; Potentialfor utilitarianism; Nuance and pragmatism; Possible supportfor animal welfare, but with caveats; Acceptance of minor moralflexibility; It is important to remember that these are just inferences based on a limited set of responses. The rater’s true beliefs and values may be more complex and nuanced than what can be determinedfrom this data alone.

Cluster Profile 4: Supports government intervention in the economy; Progressive social views; Prioritizes social welfare; Believes in public infrastructure investment; Values education; Potentially skeptical of inherited power/privilege; May believe in reducing inequality; Possibly environmentally conscious; Optimistic about government’s ability to improve society; Could be influenced by current events and political discourse in the UK

Cluster Profile 5: Strong disapproval ofTheresa May; Public health consciousness; Environmental concern; Beliefin direct democracy; Concern about overpopulation; Openness to government intervention; Possible leaning towards left-leaning or liberal politics; Pragmatism and nuanced views; UK-centric perspective

Cluster Profile 6: Pro-worker/Pro-labor; Environmentalist/Concerned about climate change; Socially liberal/Progressive; Emphasis on wellbeing/Quality of life; Government intervention; Potentially left-leaning politically; Beliefin international cooperation; It’s important to remember that these are inferences based on limited data. The rater’s actual beliefs may be more nuanced and complex.

<table><tr><td rowspan=1 colspan=2>Name</td><td rowspan=1 colspan=1>Test Accuracy</td><td rowspan=1 colspan=2>Test Loss</td><td rowspan=1 colspan=1>Usable Info (nats)</td><td rowspan=1 colspan=1>Info Preserved</td></tr><tr><td rowspan=6 colspan=2>no infodem demographics.agedem demographics.educationdem demographics.ethnicitydem demographics.gender iddem demographics.immigration status</td><td rowspan=1 colspan=1>24.8 (±0.43)</td><td rowspan=1 colspan=2>1.838 (±0.003)</td><td rowspan=1 colspan=1>0.000</td><td rowspan=2 colspan=1>(0%)</td></tr><tr><td rowspan=1 colspan=1>23.9 (±1.21)</td><td rowspan=1 colspan=2>1.846 (±0.015)</td><td rowspan=1 colspan=1>-0.007</td></tr><tr><td rowspan=1 colspan=1>24.1 (±0.69)</td><td rowspan=1 colspan=2>1.847 (±0.004)</td><td rowspan=1 colspan=1>-0.008</td><td rowspan=8 colspan=1></td></tr><tr><td rowspan=1 colspan=1>24.1 (±0.45)</td><td rowspan=1 colspan=2>1.834 (±0.005)</td><td rowspan=1 colspan=1>0.004</td></tr><tr><td rowspan=1 colspan=1>25.2 (±0.19)</td><td rowspan=1 colspan=2>1.830 (±0.005)</td><td rowspan=1 colspan=1>0.008</td></tr><tr><td rowspan=1 colspan=1>23.5 (±1.13)</td><td rowspan=1 colspan=2>1.838 (±0.009)</td><td rowspan=1 colspan=1>0.000</td></tr><tr><td rowspan=5 colspan=2>dem demographics.incomedem demographics.party iddem demographics.regiondem demographics.religiondem identity columns</td><td rowspan=1 colspan=1>25.5 (±0.37)</td><td rowspan=1 colspan=2>1.834 (±0.007)</td><td rowspan=1 colspan=1>0.005</td></tr><tr><td rowspan=1 colspan=1>24.6 (±0.43)</td><td rowspan=1 colspan=2>1.829 (±0.006)</td><td rowspan=1 colspan=1>0.009</td></tr><tr><td rowspan=1 colspan=1>24.9 (±0.50)</td><td rowspan=1 colspan=1>1.831 (±0.005)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.008</td></tr><tr><td rowspan=1 colspan=1>24.7 (±0.67)</td><td rowspan=1 colspan=1>1.843 (±0.005)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.004</td></tr><tr><td rowspan=1 colspan=1>23.9 (±1.27)</td><td rowspan=1 colspan=1>1.852 (±0.012)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.013</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>dem value columns</td><td rowspan=1 colspan=1>23.6 (±1.32)</td><td rowspan=1 colspan=1>1.835 (±0.020)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.003</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>dem (all)</td><td rowspan=1 colspan=1>25.3 (±0.49)</td><td rowspan=1 colspan=1>1.822 (±0.006)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.016</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>profile cluster-2</td><td rowspan=1 colspan=1>24.3 (±0.54)</td><td rowspan=1 colspan=1>1.840 (±0.004)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.002</td><td rowspan=1 colspan=1>-4%</td></tr><tr><td rowspan=1 colspan=2>profile cluster-4</td><td rowspan=1 colspan=1>24.6 (±0.56)</td><td rowspan=1 colspan=1>1.846 (±0.010)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.008</td><td rowspan=1 colspan=1>-19%</td></tr><tr><td rowspan=1 colspan=2>profile cluster-8</td><td rowspan=1 colspan=1>24.6 (±0.45)</td><td rowspan=1 colspan=1>1.844 (±0.005)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>-0.006</td><td rowspan=1 colspan=1>-14%</td></tr><tr><td rowspan=1 colspan=2>profile 9b</td><td rowspan=1 colspan=1>26.7 (±0.80)</td><td rowspan=1 colspan=1>1.819 (±0.004)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.019</td><td rowspan=1 colspan=1>46%</td></tr><tr><td rowspan=1 colspan=2>profile 27b</td><td rowspan=1 colspan=1>26.2 (±0.52)</td><td rowspan=1 colspan=1>1.815 (±0.003)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.023</td><td rowspan=1 colspan=1>56%</td></tr><tr><td rowspan=1 colspan=2>profile gni</td><td rowspan=1 colspan=1>25.8 (±0.50)</td><td rowspan=1 colspan=1>1.817 (±0.006)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.022</td><td rowspan=1 colspan=1>52%</td></tr><tr><td rowspan=1 colspan=2>dem+profile gni</td><td rowspan=1 colspan=1>27.2 (±0.80)</td><td rowspan=1 colspan=1>1.785 (±0.007)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.053</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>1 ex</td><td rowspan=1 colspan=1>25.4 (±0.72)</td><td rowspan=1 colspan=1>1.814 (±0.004)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.025</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>2 ex</td><td rowspan=1 colspan=1>27.0 (±0.68)</td><td rowspan=1 colspan=1>1.802 (±0.005)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.036</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>4 ex</td><td rowspan=1 colspan=1>27.6 (±0.98)</td><td rowspan=1 colspan=1>1.799 (±0.004)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.039</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=2 colspan=2>8exmajority class acc./dataset entropy</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>27.9 (±0.68)</td><td rowspan=1 colspan=2>1.797 (±0.004)</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>21.2 (±0.00)</td><td rowspan=1 colspan=1>1.906 (±0.000)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr></table>

Table 8: Habermas Performance Metrics (Model: gemma2-9b-pt) Other datasets: Appendix: G

Cluster Profile 7: Values religiousfreedom and parental rights; Prioritizes family autonomy over state control; May be religious themselves; Pragmatic or uncertain about online medicine; Weighing competing values; Lack ofknowledge; Beliefin a mixed approach; Values personal responsibility; Prioritizes affordability and access to healthcare; Trusts marketforces to some extent

Cluster Profile 8: Strong belief in personal responsibility and limited government intervention; Concernfor social safety and welfare, but with a focus on individual choice; Environmental awareness; Generally law-abiding and moralistic, but with potential for nuance; Potential belief in economic fairness and reducing inequality; Value of personal freedom and autonomy; Pragmatic approach to complex issues

## H.3 Hatespeech-Kumar

## H.3.1 2 clusters (Hatespeech-Kumar)

Cluster Profile 1: Profanity Tolerance; Emphasis on Intent over Specific Words; Sensitivity to Identity-Based Attacks; Broad Definition ofToxicity, Including Harmful Stereotypes and Misinformation; Potential Political Bias; Discomfort with Sexualized Language; Subjectivity and Context Matter;

Acceptance of Strong Opinions; Inconsistency or evolving understanding of toxicity; Possible cultural or generational influences

Cluster Profile 2: Strong tolerancefor offensive language and controversial topics; Focus on direct threats and personal attacks as "toxic"; Insensitivity to subtle forms of prejudice; Acceptance of "locker room talk" or crude humor; Prioritization of intent over impact; Inconsistency in applying criteria; It’s important to emphasize that these are speculative interpretations based on a limited sample of data. Further analysis and direct questioning of the rater would be necessary to confirm these beliefs and values.

## H.3.2 4 clusters (Hatespeech-Kumar)

Cluster Profile 1: Strong aversion to negativity and insults; Sensitivity to discussions ofpotentially harmful topics; A broad interpretation of toxicity; Concern with stereotyping and generalizations; Sensitivity to political and religious discussions; Emphasis on context and intent; Potential overreliance on emotional response

Cluster Profile 2: Strong aversion to profanity and vulgar language; Sensitivity to negativity and insults; Concern about violence and harmful actions; Discomfort with stereotypes and generalizations; Sensitivity to discussions of sensitive topics; Broad interpretation of"toxicity"; Possible discomfort with intense emotional expressions; Inconsistent application ofcriteria; Potential cultural or generational differences

<table><tr><td rowspan=1 colspan=1>Name</td><td rowspan=1 colspan=2>Test Accuracy</td><td rowspan=1 colspan=1>Test Loss</td><td rowspan=1 colspan=1>Usable Info (nats)</td><td rowspan=1 colspan=1>Info Preserved</td></tr><tr><td rowspan=1 colspan=1>no info</td><td rowspan=1 colspan=2>56.6 (±1.96)</td><td rowspan=1 colspan=1>0.684 (±0.004)</td><td rowspan=1 colspan=1>0.000</td><td rowspan=5 colspan=1>(0%)</td></tr><tr><td rowspan=1 colspan=1>dem age</td><td rowspan=1 colspan=2>58.9 (±0.92)</td><td rowspan=1 colspan=1>0.681 (±0.006)</td><td rowspan=1 colspan=1>0.004</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>dem study locale</td><td rowspan=1 colspan=2>60.1 (±0.61)</td><td rowspan=1 colspan=1>0.674 (±0.005)</td><td rowspan=1 colspan=1>0.010</td></tr><tr><td rowspan=1 colspan=1>dem stated prefs</td><td rowspan=1 colspan=2>58.4 (±0.93)</td><td rowspan=1 colspan=1>0.680 (±0.007)</td><td rowspan=1 colspan=1>0.004</td></tr><tr><td rowspan=1 colspan=1>dem self description</td><td rowspan=1 colspan=2>58.9 (±1.60)</td><td rowspan=1 colspan=1>0.678 (±0.004)</td><td rowspan=1 colspan=1>0.006</td></tr><tr><td rowspan=1 colspan=1>dem religion</td><td rowspan=1 colspan=2>55.8 (±1.41)</td><td rowspan=1 colspan=1>0.686 (±0.002)</td><td rowspan=1 colspan=1>-0.001</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>dem order stated prefs</td><td rowspan=1 colspan=2>60.4 (±0.60)</td><td rowspan=1 colspan=1>0.672 (±0.004)</td><td rowspan=1 colspan=1>0.013</td><td rowspan=5 colspan=1></td></tr><tr><td rowspan=1 colspan=1>dem order lm usecases</td><td rowspan=1 colspan=2>59.8 (±1.26)</td><td rowspan=1 colspan=1>0.674 (±0.007)</td><td rowspan=1 colspan=1>0.011</td></tr><tr><td rowspan=1 colspan=1>dem marital status</td><td rowspan=1 colspan=2>59.3 (±1.01)</td><td rowspan=1 colspan=1>0.676 (±0.006)</td><td rowspan=1 colspan=1>0.009</td></tr><tr><td rowspan=1 colspan=1>dem location</td><td rowspan=1 colspan=2>58.0 (±1.87)</td><td rowspan=1 colspan=1>0.676 (±0.007)</td><td rowspan=1 colspan=1>0.009</td></tr><tr><td rowspan=1 colspan=1>dem lm usecases</td><td rowspan=1 colspan=2>59.3 (±1.27)</td><td rowspan=1 colspan=1>0.675 (±0.006)</td><td rowspan=1 colspan=1>0.009</td></tr><tr><td rowspan=1 colspan=1>dem lm indirect use</td><td rowspan=1 colspan=2>55.4 (±1.62)</td><td rowspan=1 colspan=1>0.833 (±0.145)</td><td rowspan=1 colspan=1>-0.149</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>dem lm frequency use</td><td rowspan=1 colspan=1>59.3 (±0.88)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.680 (±0.006)</td><td rowspan=1 colspan=1>0.005</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>dem lm familiarity</td><td rowspan=1 colspan=2>55.5 (±2.45)</td><td rowspan=1 colspan=1>0.685 (±0.003)</td><td rowspan=1 colspan=1>-0.001</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=1>dem lm direct use</td><td rowspan=1 colspan=2>56.5 (±1.60)</td><td rowspan=1 colspan=1>0.683 (±0.004)</td><td rowspan=1 colspan=1>0.002</td></tr><tr><td rowspan=1 colspan=1>dem identity columns</td><td rowspan=1 colspan=1>60.8 (±0.58)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.671 (±0.004)</td><td rowspan=1 colspan=1>0.013</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>dem gender</td><td rowspan=1 colspan=1>57.1 (±1.88)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.682 (±0.006)</td><td rowspan=1 colspan=1>0.002</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>dem ethnicity</td><td rowspan=1 colspan=1>57.8 (±1.40)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.684 (±0.004)</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>dem english proficiency</td><td rowspan=1 colspan=1>57.5 (±1.37)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.684 (±0.004)</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>dem employment status</td><td rowspan=1 colspan=1>59.6 (±0.75)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.677 (±0.005)</td><td rowspan=1 colspan=1>0.008</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>dem education</td><td rowspan=1 colspan=1>59.1 (±0.91)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.679 (±0.005)</td><td rowspan=1 colspan=1>0.005</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>dem system string</td><td rowspan=1 colspan=1>59.6 (±0.67)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.673 (±0.005)</td><td rowspan=1 colspan=1>0.011</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>dem value columns</td><td rowspan=1 colspan=1>58.6 (±1.83)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.676 (±0.005)</td><td rowspan=1 colspan=1>0.008</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>dem (all)</td><td rowspan=1 colspan=1>58.6 (±1.63)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.679 (±0.006)</td><td rowspan=1 colspan=1>0.005</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>profile cluster-2</td><td rowspan=1 colspan=1>60.2 (±0.58)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.673 (±0.005)</td><td rowspan=1 colspan=1>0.012</td><td rowspan=1 colspan=1>131%</td></tr><tr><td rowspan=1 colspan=1>profile cluster-4</td><td rowspan=1 colspan=1>58.2 (±2.13)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.674 (±0.006)</td><td rowspan=1 colspan=1>0.010</td><td rowspan=1 colspan=1>114%</td></tr><tr><td rowspan=1 colspan=1>profile cluster-8</td><td rowspan=1 colspan=1>56.2 (±2.09)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.684 (±0.005)</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>5%</td></tr><tr><td rowspan=1 colspan=1>profile 9b</td><td rowspan=1 colspan=1>60.8 (±1.07)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.672 (±0.006)</td><td rowspan=1 colspan=1>0.013</td><td rowspan=1 colspan=1>145%</td></tr><tr><td rowspan=1 colspan=1>profile 27b</td><td rowspan=1 colspan=1>61.3 (±0.96)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.667 (±0.008)</td><td rowspan=1 colspan=1>0.017</td><td rowspan=1 colspan=1>191%</td></tr><tr><td rowspan=1 colspan=1>profile gni</td><td rowspan=1 colspan=1>60.4 (±1.64)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.665 (±0.008)</td><td rowspan=1 colspan=1>0.020</td><td rowspan=1 colspan=1>220%</td></tr><tr><td rowspan=1 colspan=1>dem+profile gni</td><td rowspan=1 colspan=1>60.8 (±0.55)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.668 (±0.007)</td><td rowspan=1 colspan=1>0.016</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>1 ex</td><td rowspan=1 colspan=1>57.6 (±2.30)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.677 (±0.005)</td><td rowspan=1 colspan=1>0.007</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>2 ex</td><td rowspan=1 colspan=1>56.0 (±1.91)</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>0.681 (±0.006)</td><td rowspan=1 colspan=1>0.003</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>4 ex</td><td rowspan=1 colspan=2>58.0 (±2.26)</td><td rowspan=1 colspan=1>0.676 (±0.006)</td><td rowspan=1 colspan=1>0.009</td><td rowspan=3 colspan=1>(100%)</td></tr><tr><td rowspan=1 colspan=1>8ex</td><td rowspan=1 colspan=2>58.0 (±2.42)</td><td rowspan=1 colspan=1>0.676 (±0.006)</td><td rowspan=1 colspan=1>0.009</td></tr><tr><td rowspan=1 colspan=1>majority class acc./dataset entropy</td><td rowspan=1 colspan=2>50.3 (±0.00)</td><td rowspan=1 colspan=1>0.693 (±0.000)</td><td rowspan=1 colspan=1></td></tr></table>

Table 9: Prism Performance Metrics (Model: gemma2-9b-pt) Other datasets: Appendix: G

Cluster Profile 3: Strong tolerance for offensive language and controversial topics; Focus on direct threats and personal attacks as "toxic"; Insensitivity to subtle forms ofprejudice; Acceptance of "locker room talk" or crude humor; Prioritization of intent over impact; Inconsistency in applying criteria; It’s important to emphasize that these are speculative interpretations based on a limited sample of data. Further analysis and direct questioning of the rater would be necessary to confirm these beliefs and values.

Cluster Profile 4: Profanity Tolerance; Emphasis on Intent over Specific Words; Sensitivity to Identity-Based Attacks; Broad Definition ofToxicity, Including Harmful Stereotypes and Misinformation; Potential Political Bias; Discomfort with Sexualized Language; Subjectivity and Context Matter; Acceptance of Strong Opinions; Inconsistency or evolving understanding of toxicity; Possible cultural or generational influences

## H.3.3 8 clusters (Hatespeech-Kumar)

Cluster Profile 1: Strong tolerance for offensive language and controversial topics; Focus on direct threats and personal attacks as "toxic"; Insensitivity to subtle forms ofprejudice; Acceptance of "locker room talk" or crude humor; Prioritization of intent over impact; Inconsistency in applying criteria; It’s important to emphasize that these are speculative interpretations based on a limited sample ofdata. Further analysis and direct questioning of the rater would be necessary to confirm these beliefs and values.

Cluster Profile 2: Profanity Tolerance; Emphasis on Intent over Specific Words; Sensitivity to Identity-Based Attacks; Broad Definition ofToxicity, Including Harmful Stereotypes and Misinformation; Potential Political Bias; Discomfort with Sexualized Language; Subjectivity and Context Matter; Acceptance of Strong Opinions; Inconsistency or evolving understanding of toxicity; Possible cultural or generational influences

Cluster Profile 3: High tolerancefor strong language and criticism in certain contexts; Emphasis on intent and harm over mere presence of negativity; Acceptance ofstrong emotional expression, even negative ones, within reasonable bounds; Political neutrality or tolerancefor diverse political viewpoints; Context matters; Distinction between subjective criticism and objective harm; General permissiveness towards fan-related content and positive expressions; Focus on individual harm rather than potential group offense

Cluster Profile 4: High tolerancefor profanity and sexual content; Sensitivity to personal attacks and insults; A more lenient view ofpotentially offensive language when used in humor or casual conversation; Discomfort with stereotyping and generalizations about groups; Tolerance of strong opinions and emotional expressions; A possible bias towards accepting casual, internet-specific language and trends; Inconsistency in applying standards; Potential lack ofunderstanding ofsome slang or cultural references

Cluster Profile 5: Strong reactions or negativity alone don’t necessarily constitute toxicity. The rater marked comments expressing strong dislike ("gross," "wouldn’t touch... with a bargepole") or frustration ("Who TF cares") as not toxic, suggesting they don’t equate negativity with toxicity. However, when coupled with aggressive or potentially harmful sentiments, as in the comment about electric cars being "political bullshit" that is being "forced on us", the rating changes to toxic. This implies nuance in their judgment.; Direct or implied threats and mentions of harm are considered toxic. The comments referencing "raped," and potentially violent or inappropriate behavior ("After I nun ate... he wanna cut up in there") were marked toxic. This indicates a valuefor safety and a belief that language suggesting harm is unacceptable.; Humor and informal language are generally acceptable. The use ofslang ("LIT," "TF"), internet abbreviations (""), and self-deprecating humor did not trigger a toxic rating. This suggests a tolerance for casual online communication styles.; Discussion and disagreement are not inherently toxic. The comment about "Ariel is black" presents a counterargument, yet it’s deemed non-toxic. This implies a belief that expressing diverse opinions, even on potentially controversial topics, is permissible.; Context matters. Hashtags and references that could be considered out of place (e.g., "

sachinopensagain") weren’t considered toxic in isolation. This suggests the rater is considering the overall message and intent, rather than focusing solely on individual words or phrases.; Personal preferences or strong opinions, if not directed at individuals or groups, are acceptable. The comment about "gorgeous gay dudes sword fighting" expresses a specific preference, but is not considered toxic. This indicates a respect for individual tastes, as long as they aren’t used to denigrate others.; The rater may have a higher threshold for toxicity. Several comments that could be perceived as rude or offensive by some were marked non-toxic. This suggests the raterfocuses on more severeforms oftoxicity, prioritizing clear instances ofharm or aggression.

Cluster Profile 6: High tolerancefor offensive language; Focus on explicit threats or calls for harm as markers of toxicity; Desensitization to online negativity; Belief that subjective opinions are not inherently toxic; Lack ofconsiderationfor the impact of microaggressions; Prioritization of intent over impact; Possible personal bias

Cluster Profile 7: Strong aversion to profanity and vulgar language; Sensitivity to negativity and insults; Concern about violence and harmful actions; Discomfort with stereotypes and generalizations; Sensitivity to discussions of sensitive topics; Broad interpretation of "toxicity"; Possible discomfort with intense emotional expressions; Inconsistent application ofcriteria; Potential cultural or generational differences

Cluster Profile 8: Strong aversion to negativity and insults; Sensitivity to discussions ofpotentially harmful topics; A broad interpretation of toxicity; Concern with stereotyping and generalizations; Sensitivity to political and religious discussions; Emphasis on context and intent; Potential overreliance on emotional response

## H.4 OpinionQA - Wave 27

H.4.1 2 clusters (OpinionQA - Wave 27)

Cluster Profile 1: Nationalist/Patriotic; Conservative; Law and Order; Pro-Military; Economically Conservative, but Populist on Trade; Socially Conservative, but with Libertarian Leanings; Distrustful ofGovernment and Elites; Pragmatic; Pessimistic

Cluster Profile 2: Believes in American exceptionalism, but acknowledges other great nations;

Values democracy and allies; Supports a strong social safety net but believes in personal responsibility; Pragmatic and values compromise; Optimistic about social progress; Values traditional family structures; Concerned about voter fraud, but supports voting rights; Supports separation ofchurch and state, but sees value in religious belief; Positive about technology and globalization; Believes in a larger government role; Socially moderate; Economically progressive; Generally content but sees areasfor improvement; Skeptical ofpoliticians and the political system; Believes in expert knowledge; Believes in a strong military and good diplomacy; Values immigration but with controls; Believes in personalfreedoms but recognizes the needfor some government intervention; Doesn’tfeel disrespected but acknowledges white privilege

## H.4.2 4 clusters (OpinionQA - Wave 27)

Cluster Profile 1: Believes in American exceptionalism, but acknowledges other great nations; Values democracy and allies; Supports a strong social safety net but believes in personal responsibility; Pragmatic and values compromise; Optimistic about social progress; Values traditional family structures; Concerned about voterfraud, but supports voting rights; Supports separation ofchurch and state, but sees value in religious belief; Positive about technology and globalization; Believes in a larger government role; Socially moderate; Economically progressive; Generally content but sees areasfor improvement; Skeptical ofpoliticians and the political system; Believes in expert knowledge; Believes in a strong military and good diplomacy; Values immigration but with controls; Believes in personal freedoms but recognizes the need for some government intervention; Doesn’tfeel disrespected but acknowledges white privilege

Cluster Profile 2: Nationalist/Patriotic; Conservative; Law and Order; Pro-Military; Economically Conservative, but Populist on Trade; Socially Conservative, but with Libertarian Leanings; Distrustful of Government and Elites; Pragmatic; Pessimistic

Cluster Profile 3: Conservative or right-leaning political views; Belief in individual responsibility; Skepticism ofsocial justice movements or "woke" ideology; Potential concern about social instability; Preference for a smaller government role in the economy; May value traditional values and institutions; May believe in American exceptionalism; May prioritize economic growth over social programs; Possible distrust of government

Cluster Profile 4: Progressive/Left-leaning political views; Distrust of large institutions; Emphasis on diplomacy and international cooperation; Socially liberal; Belief in nuanced approaches; Slight racial anxiety; Confidence in the electoral system; Value on expertise; Mixedfeelings on the role of government; Pragmatic approach to military strength

## H.4.3 8 clusters (OpinionQA - Wave 27)

Cluster Profile 1: Believes in American exceptionalism, but acknowledges other great nations; Values democracy and allies; Supports a strong social safety net but believes in personal responsibility; Pragmatic and values compromise; Optimistic about social progress; Values traditional family structures; Concerned about voter fraud, but supports voting rights; Supports separation of church and state, but sees value in religious belief; Positive about technology and globalization; Believes in a larger government role; Socially moderate; Economically progressive; Generally content but sees areasfor improvement; Skeptical ofpoliticians and the political system; Believes in expert knowledge; Believes in a strong military and good diplomacy; Values immigration but with controls; Believes in personalfreedoms but recognizes the needfor some government intervention; Doesn’t feel disrespected but acknowledges white privilege

Cluster Profile 2: Nationalist/Patriotic; Conservative; Law and Order; Pro-Military; Economically Conservative, but Populist on Trade; Socially Conservative, but with Libertarian Leanings; Distrustful ofGovernment and Elites; Pragmatic; Pessimistic

Cluster Profile 3: Conservative or right-leaning political views; Belief in individual responsibility; Skepticism ofsocial justice movements or "woke" ideology; Potential concern about social instability; Preference for a smaller government role in the economy; May value traditional values and institutions; May believe in American exceptionalism; May prioritize economic growth over social programs; Possible distrust of government

Cluster Profile 4: Socially liberal/Moderate; Economically left-leaning; Pro-immigration and diversity; Trust in experts and government; Democratic-leaning but not entirely aligned; Internationally cooperative; Values traditional family structures but withflexibility; Believes in equal rights but acknowledges challenges; Sense offairness and respect; It’s important to note

Cluster Profile 5: Pro-corporations; Egalitarian parenting; Second Amendment supporter, but with nuance; Concern about election integrity, but not extreme distrust; Support for social safety nets, but potentially limited government intervention; Generally distrustful of government; Minimizes racial inequality; Deference to expertise; Tolerance of offensive speech; Ambivalence towards wealth inequality; Non-interventionist foreign policy or satisfaction with current military spending

Cluster Profile 6: Conservative leaning; Nationalist/America First; Socially conservative; Distrust ofGovernment and Elites; Tough on Crime; Economic Conservatism; Pro-Religion; Traditional Values; Belief in Personal Responsibility; While not explicitly stated, a potential for racial resentment; It is important to note

Cluster Profile 7: Progressive/Liberal leaning; Socially Liberal; Economic Populist; Progovernment Intervention; Religious; Community-Oriented; Distrustful of Institutions; Diplomatic but Values Democracy; Pro-Voting Rights; Belief in Experts; Criminal Justice Reform; Pessimistic about the Present; Believes in Compromise; Concerned about Free Speech; Open to Other Languages; Believes in shared values; Potentially holds contradictory views

Cluster Profile 8: Progressive/Left-leaning political views; Distrust of large institutions; Emphasis on diplomacy and international cooperation; Socially liberal; Belief in nuanced approaches; Slight racial anxiety; Confidence in the electoral system; Value on expertise; Mixedfeelings on the role of government; Pragmatic approach to military strength

## H.5 PRISM

## H.5.1 2 clusters (PRISM)

Cluster Profile 1: Completeness and Thoroughness; Specificity and Directness; Accuracy and Up-to-date Information; Neutrality and Objectivity; Practical Utility; Contextual Awareness; User Control and Agency

Cluster Profile 2: Completeness and Thoroughness; Directness and Assertiveness; Neutrality, but with Context; Proactive Helpfulness; Formal Tone; Accuracy and Factuality; Engagement and Conversational Flow

## H.5.2 4 clusters (PRISM)

Cluster Profile 1: Completeness and Thoroughness; Directness and Assertiveness; Neutrality, but with Context; Proactive Helpfulness; Formal Tone; Accuracy and Factuality; Engagement and Conversational Flow

Cluster Profile 2: Prefers helpfulness and relevance over assumptions; Appreciates nuanced and comprehensive answers; Values honesty and awareness of limitations; Favors open-ended conversation and assistance; Respects diverse perspectives and avoids generalizations; Prioritizes accuracy and avoids potential misinformation

Cluster Profile 3: Completeness and Thoroughness; Specificity and Directness; Accuracy and Up-to-date Information; Neutrality and Objectivity; Practical Utility; Contextual Awareness; User Control and Agency

Cluster Profile 4: Practicality and Actionability; Thoroughness and Detail; Emphasis on Positive Communication; Desire for Structure and Guidance; Appreciation for Contextual Nuance; Preference for Proactive Problem-Solving; Potential Discomfort with Ambiguity

## H.5.3 8 clusters (PRISM)

Cluster Profile 1: Completeness and Thoroughness; Specificity and Directness; Accuracy and Up-to-date Information; Neutrality and Objectivity; Practical Utility; Contextual Awareness; User Control and Agency

Cluster Profile 2: Practicality and Actionability; Thoroughness and Detail; Emphasis on Positive Communication; Desire for Structure and Guidance; Appreciation for Contextual Nuance; Preferencefor Proactive Problem-Solving; Potential Discomfort with Ambiguity

Cluster Profile 3: Values direct answers over hedging; Appreciates nuanced perspectives; Favors a conversational and welcoming tone; Prioritizes specific details over generic praise; Trusts recommendations that consider local perspective; May appreciate subtlety and avoids overly strong endorsements; Potentially values the feeling of discovery; Might be influenced by writing style and fluency

Cluster Profile 4: Completeness and Thoroughness; Directness and Assertiveness; Neutrality, but with Context; Proactive Helpfulness; Formal Tone; Accuracy and Factuality; Engagement and Conversational Flow

Cluster Profile 5: Prefers conciseness and directness; Values politeness and helpfulness; Favors factual and relevant information; Appreciates simplicity over technical jargon; Prioritizesfunctional answers; May have a lower tolerance for conversationalfillers; Could value transparency, but only to a certain extent; Possibly prefers a less anthropomorphic model

Cluster Profile 6: Prefers helpfulness and relevance over assumptions; Appreciates nuanced and comprehensive answers; Values honesty and awareness of limitations; Favors open-ended conversation and assistance; Respects diverse perspectives and avoids generalizations; Prioritizes accuracy and avoids potential misinformation

Cluster Profile 7: Practicality and Actionable Advice; Thoroughness and Detail; Directness and Assertiveness; Real-World Applicability; External Validation and Authority; Focus on Well-being; Belief in Inclusivity and Fairness; Appreciation for Nuance and Context

Cluster Profile 8: General Communication Style; Brevity and Directness; Empathy and Encouragement; Informality and Approachability; Information and Advice; High-Level Overview; Balance ofPractical and Emotional Considerations; Less Emphasis on Specifics; It’s important to note

## H.6 ValuePrism Valence

## H.6.1 2 clusters (ValuePrism Valence)

Cluster Profile 1: Autonomy and bodily integrity, especially for children; Nuance in medical decisions; Open communication is acceptable, butforcing compliance is not; Potential harm of medication; Belief in a child’s ability to learn and make good choices with guidance; Respect for individual differences; Potential for parental overreach; Emphasis on intrinsic motivation over extrinsic pressure

Cluster Profile 2: Inclusivity and acceptance; Freedom of association; Personal autonomy and self-expression; Opposition to exclusion and social ostracism; Potential belief in restorative justice or rehabilitation over punishment; Colorblindness or a belief that race should not be a factor in social interactions; Actions that do not harm others are generally moral; Focus on positive social interactions

## H.6.2 4 clusters (ValuePrism Valence)

Cluster Profile 1: Inclusivity and acceptance; Freedom of association; Personal autonomy and self-expression; Opposition to exclusion and social ostracism; Potential belief in restorative justice or rehabilitation over punishment; Colorblindness or a belief that race should not be a factor in social interactions; Actions that do not harm others are generally moral; Focus on positive social interactions

Cluster Profile 2: Egalitarianism and Antidiscrimination; Social Justice and Advocacy; Inclusivity and Acceptance; Meritocracy and Fairness; Individualism and Contextual Morality; Potential Conflict or Uncertainty

Cluster Profile 3: Parental autonomy and rights; Pro-corporal punishment; Importance of fulfilling parental responsibilities; Nuance in child rearing decisions; Skepticism of outside intervention in families; Traditional gender roles; General permissiveness or a broad definition of morality; Potential prioritization ofpersonalfreedoms; Important Note

Cluster Profile 4: Autonomy and bodily integrity, especiallyfor children; Nuance in medical decisions; Open communication is acceptable, but forcing compliance is not; Potential harm ofmedication; Beliefin a child’s ability to learn and make good choices with guidance; Respect for individual differences; Potential for parental overreach; Emphasis on intrinsic motivation over extrinsic pressure

## H.6.3 8 clusters (ValuePrism Valence)

Cluster Profile 1: Egalitarianism and Antidiscrimination; Social Justice and Advocacy; Inclusivity and Acceptance; Meritocracy and Fairness; Individualism and Contextual Morality; Potential Conflict or Uncertainty

Cluster Profile 2: Collectivism over Individualism; Authoritarianism/Respectfor Authority; Utilitarianism/Consequentialism; Nationalism/Group Loyalty; Situational Ethics; Distrust of"Freedom Fighters"; Moral Pragmatism; Potential Double Standards

Cluster Profile 3: Emphasis on self-reliance and adult responsibility; Prioritization ofsocietal norms regarding child development and parenting; Discomfort with actions perceived as unconventional or exceeding typical boundaries; Potential value of "tough love" as a parenting strategy; Possible beliefin a clear distinction between childhood and adulthood; Focus on physical and emotional development milestones; Potential for a conservative worldview; Implicit bias or personal experi-

## ence shaping judgements

Cluster Profile 4: Strong belief in the sanctity of life, even for those deemed evil; Pacifism or aversion to violence; Nuance in moral decision-making and a rejection ofsimple utilitarianism; Potential beliefin the inherent rights ofindividuals; Possible concern for consequences beyond the immediate situation; Possible belief in alternative solutions; Absence ofprejudice based on nationality; Possible emphasis on intention over outcome

Cluster Profile 5: Parental autonomy and rights; Pro-corporal punishment; Importance offulfilling parental responsibilities; Nuance in child rearing decisions; Skepticism of outside intervention in families; Traditional gender roles; General permissiveness or a broad definition of morality; Potential prioritization ofpersonalfreedoms; Important Note

Cluster Profile 6: Inclusivity and acceptance; Freedom of association; Personal autonomy and self-expression; Opposition to exclusion and social ostracism; Potential belief in restorative justice or rehabilitation over punishment; Colorblindness or a belief that race should not be a factor in social interactions; Actions that do not harm others are generally moral; Focus on positive social interactions

Cluster Profile 7: Autonomy and bodily integrity, especiallyfor children; Nuance in medical decisions; Open communication is acceptable, but forcing compliance is not; Potential harm of medication; Beliefin a child’s ability to learn and make good choices with guidance; Respect for individual differences; Potential for parental overreach; Emphasis on intrinsic motivation over extrinsic pressure

Cluster Profile 8: Individual autonomy andfreedom; Situational ethics; Prioritization of relationships and consent; Consideration ofintent and impact; Non-judgmental attitude; Potential cultural sensitivity; Flexible and adaptable moral framework

## I Random Profile Samples

## I.1 gemma2-9b

## I.2 OpinionQA (gemma2-9b - 10 random value profiles)

• Moderate to conservative politically, lean towards social traditionalism; Believes in punitive justice and stronger sentences; Skeptical ofgovernment intervention but open to some regulation in specific areas; May have a preferencefor more traditional American values and identity; Views the entertainment industry positively; Pragmatic about the topic ofslavery and racism, perhaps seeing it as a complex issue with no easy solutions; Concerned about the quality ofpolitical candidates

• Believes that corporations are overly profitable; Believes that progress has been made towards racial equality in the US over the last 50 years; Feels that people are too easily offended and that this is a major problem; Is disillusioned with the political process, seeing compromise as aform of"selling out."; Holds a nationalist view, believing that other countries take advantage of the US; Believes that government assistance to the poor is harmful

• Seeks a balance, not extremes: Often responds with "neither good nor bad" and favors "modest" changes; Wary of big government and dependency: Believes in limited government involvement,; Conservative social views: Holds traditional beliefs about marriage,family structure, and the role ofreligion; Values national strength and security: Prefers the U.S. to maintain military superiority

• Somewhat nationalistic; Patriotic but hesitant about uncontrolled immigration; Skeptical of government efficiency; Leaning conservative; Values traditional social institutions; Believes military strength is importantfor peace

• Supports increased government involvement in providing services; Believes in strict voting rights and sees it as a fundamental right; Holds slightly negative views on the way things are currently going in the country; Values diplomacy over military strength; Believes in compromise in politics; Concerned about social inequality and the impact of powerful interests; Positive view ofsame-sex marriage

• Skeptical oforganized religion: Sees no harm in declining religiosity; Patriotic, but distrustful offoreign aid and international involvement: Prefers focus on American interests in foreign policy; Values individual liberties and limited government: Believes government is wasteful and inefficient, prefers less government intervention in people’s lives; Concerned about social changes and decline in traditional values: Feels uncomfortable with increased cultural diversity, expresses discomfort with societal shifts; Feels alienatedfrom current political landscape: Does not resonate with

• Believes the entertainment industry has a positive effect on the country; Concerned about offensive language and speech; Believes they receive respect in society; Feels comfortable with Republicans expressing their views; Supports free tuition for public colleges; Believes K-12 public schools are having a positive effect; Believes strength and military might are the best way to ensure peace; Comfortable with the U.S. being treatedfairly in the world

• Believes in social justice and equality, as evidenced by their answers on racial inequality, LGBTQ+ rights, and gender equality; Supports increased government involvement in social welfare programs and healthcare; Favors progressive policies such as universal healthcare, tuition-free public colleges, and stricter gun control; Is skeptical of corporate power and believes businesses make excessive profits; Values diplomacy and international cooperation over military strength; Is concerned about the influence of religion in politics and government

• Strongly nationalist. Believes the US is superior to other countries; Conservative social values. Opposes same-sex marriage, believes traditionalfamily structures are best; Pro-gun rights and skeptical of gun control measures; Low regard for government and its inefficiencies. Favors limited government intervention; Supports a strong military presence globally; Skeptical of immigration and its impact on the country; Concerned about "political correctness" and believes individuals

• Believes that immigrants, when they come to the U.S. illegally, can have a slightly negative impact on communities; Somewhat positive view of religion and its effect on society; Holds a beliefthat the U.S. is a great country, but not necessarily the best in the world; Convinced that large corporations are detrimental to the country; Favors the traditional role of women staying home to raise a family; Feels that the country has made

## I.3 Hatespeech-Kumar (gemma2-9b - 10 random value profiles)

• Believes in keeping things civil and respectful even in disagreement; Values sensitivity and empathy towards others; Recognizes the difference between expressing strong opinions and being abusive or hateful; Sensitive to language that could be hurtful or demeaning; Appreciates humor that isn’t at the expense of others

• Distrusts inflammatory language: They often identify as toxic comments that use emotionally charged words, prejudiced terms, or hateful slurs; Values respectful discourse: They seem to appreciate comments that express opinions without resorting to insults or personal attacks; Recognizes dog-whistles: They may be sensitive to language that carries coded meanings or implies prejudice, even if it doesn’t

• Values; Dislike of bullying and insults: The rater considers personal attacks and insults to be toxic, even if they are not overtly aggressive

• Believes some comments are inherently offensive or harmful, regardless of intent; Has a strong moral compass and considers statements that promote hate, prejudice, or violence as unacceptable; Values respectful and constructive dialogue, and sees toxicity as a barrier to healthy communication; May be sensitive to language that is demeaning, discriminatory, or exploitative; Recognizes that power dynamics can contribute to toxicity, and may be more likely to flag comments that perpetuate harmful stereotypes or reinforce social inequalities

• Relatively tolerant:; Contextual understanding:; Focus on direct harm:; Skeptical of generalizations:

• Holds strong opinions about what is acceptable language and behavior; Is sensitive to language that is hateful, disrespectful, or demeaning; Values honesty and integrity; Believes in using language that is constructive and respectful; Appreciates humor that is not at the expense of others; Concerned with issues ofpower and privilege; Possibly politically left-leaning; Has a strong sense of social justice

• Believes Sarcasm and humor, even when expressed in a seemingly negative way, are not inherently toxic; Sensitive to language that is directly aggressive or threatening; Might prioritize personal expression and freedom of speech, even if it is unconventional or offputting to others; Recognizes that humor can sometimes be used to mask anger orfrustration, and these underlying emotions may contribute to toxicity; Likely values empathy and understands the importance of considering the emotional impact of

• Values humor and light-heartedness, at times even finding positivity in seemingly innocuous things; Appreciates self-awareness and honesty, even when it comes to admitting lack of knowledge; Believes in open discussion and debate, even if it involves differing opinions; Views direct insults and personal attacks as toxic; Recognizes the potential for harm in language that targets individuals based on their identity or beliefs; Might lean toward being forgiving

• Believes strong language is inappropriate; Belives sexualizing minors is wrong; Understands sarcasm and humor but may not always be able to detect it

• Might be hesitant to label comments as "toxic" unless they contain clear hate speech, aggression, or personal attacks; Values freedom of expression and believes in giving people the benefit ofthe doubt; Could be more sensitive to sarcasm, humor, or ironic statements that may be misinterpreted as toxic; Appreciates brevity and directness in communication

## I.4 DICES (gemma2-9b - 10 random value profiles)

• Values open-mindedness and respect for diverse viewpoints; Believes in the importance ofkindness and empathy; Dislikes negativity, hostility, and inflammatory language; Appreciates humor and lightheartedness; Is generally optimistic and hopeful about the world • General lack of concern for hateful or inappropriate language, even when blatant; Focuses on argument structure and lack of direct aggression as indicators of toxicity; May ignore context, taking statements atface value; Seems to label anything addressing controversial topics as toxic

• Values friendly and polite conversation; Doesn’t perceive joking or lightheartedness as toxic; May have a low thresholdfor identifying toxicity, as they labeled conversations with potentially condescending or manipulative language as "Not Toxic."

• Disagreements and differing opinions are normal and healthy. Conversation should not be approached like a debate and should not turn hostile, dismissive, or personal; Everyone deserves to be treated with respect, even iftheir views are differentfrom our own

• Valuing honesty and integrity in communication; Believing in treating others with respect and kindness, regardless of their beliefs or background; Encouraging critical thinking and open-mindedness

• Doesn’t consider casual interactions to be toxic; Tolerates a range of opinions, even if they are not politically correct or popular; Doesn’t seem to be overly sensitive to potentially offensive language; Values genuine conversation and humor over politeness; Might be comfortable with a bit ofdark humor

• Values personal reflection and avoids making sweeping judgments; May be more lenient towards socialfaux pas and missteps in online communication; Trusts individuals to understand and navigate complex issues

• Generally non judgmental and avoids making assumptions about people; Prefers direct and honest communication; Believes in treating everyone with respect, regardless oftheir background or beliefs; Values empathy and understanding; Encourages critical thinking and open-mindedness

• Believes hurtful language is unacceptable; Values respectful communication; Discourages generalizations and stereotypes; Empathizes with others’ perspectives; Promotes critical thinking and healthy skepticism; Personal insults and aggressive language; Disrespectful or condescending tone; Harmful stereotypes and generalizations; Promotion of hate speech or prejudice; Encouraging harmful or illegal activities; Exploitation

• Believes a lot of conversations are benign; May have a low tolerance for strong negativity; May value straightforwardness

## I.5 ValuePrism (gemma2-9b - 10 random value profiles)

• Honesty is generally valued, but can be situational; Loyalty to friends is likely important, as is prioritizing their well-being; Straightforwardness may not always be the most compassionate approach; Exploiting someone for personal gain or manipulating them is wrong; Protecting oneselfis morally justifiable

• Values situational context heavily when determining morality; Holds traditional views onfamily structures and authority; May have discomfort with concepts like gender fluidity; Believes in parental authority and may disapprove ofchallenging it; May prioritize obedience and adherence to rules

• Values individual liberty and autonomy; Believes in resistance to oppressive institutions or systems; May have socialist or left-wing political leanings; Sees economic sanctions as a tool for influencing policy and achieving justice; Generally believes in using peaceful means to achieve political change; May be critical ofmilitary intervention and powerful corporations

• Prioritizes the needs of family; Believes that survival trumps strict adherence to laws/morality; May have a strong sense of empathy and compassion for those in need; May have a utilitarian perspective, judging actions based on their outcomes. Utilitarianism: The idea that the best action is the one that maximizes happiness and minimizes suffering; May not distinguish morality based on race or nationality

• Strong belief in humanitarianism and helping those in need; Strong opposition to oppressive regimes and systems ofpower; Support for democracy and individual freedom; Tendency to view actions in a binary way, often as "moral" or "immoral"; May be influenced by political ideologies that emphasize justice and anti-imperialism; A clear sense of right and wrong based on perceived victimhood and injustice

• Values fairness and justice; Believes in personal responsibility and accountability; Emphasizes the importance of empathy and understanding; Distinguishes between actions that are inherently right or wrong and those that depend on context; Prioritizes individual autonomy andfreedom of choice; Criticizes abuse of power and authority; Values integrity and honesty

• Believes in the greater good; Upholds authority and established norms; Values protecting children and sees harm to them as unacceptable; Progressive and tolerant of diverse family structures; Potential emphasis on nonviolence as a core value

• Believes in inherent rewards and positive reinforcement; Values competence and meritocracy; Holds a view that setting clear expectations and consequences is important

• Believes in situational ethics; Values helping others; Likely values religious identity and community; Views helping friends as morally right; Has a strong sense ofmoral intuition

• Valuesfamilial relationships highly; Believes in individual autonomy and the right to make one’s own choices; Values loyalty and support for loved ones; Doesn’t seem to adhere to strict rules or social norms; May prioritize personal fulfillment over strict work obligations

## I.6 Habermas (gemma2-9b - 10 random value profiles)

• Believes in strong government intervention and regulation; Prefers a more egalitarian society with less inequality; May be concerned about the societal impact of smoking and alcohol consumption; Supporting of public health initiatives and increased spending on healthcare; May hold traditional or conservative views on certain social issues, such as marriage

• Believes in giving the people a voice and having referendums on important issues; Supports increased public spending on infrastructure like railways; Prefers the current democratic system over a more direct form of democracy; Favors the monarchy and maintaining the UK as a constitutional monarchy; Believes in progressive taxation, with a higher tax burdenfor the wealthy

• Supports government intervention and social programs; Worried about health and wellbeing, especially of young people; Believes in rules and structure; May be progressive or left-leaning in their political views

• Believes in social justice and equality; Supports government intervention to address societal issues; Likely progressive or left-leaning politically; Values education and believes it should be accessible to all; May be concerned about income inequality; Probably environmentally conscious and supportive ofaction on climate change

• Leans towards social safety nets and government intervention in the economy; Favors social justice and redistribution ofwealth; May have concerns about pharmaceutical industry practices

• Moderate and tends towards neutrality on a variety of social and economic issues; May be open to both sides of an argument and struggles to commit to a firm stance; Lacks strong convictions or definitive beliefs about complex issues; Prefers a balanced approach rather than taking a strong position

• Progressive on social issues, likely supporting universal healthcare and social services; Skeptical oftraditional institutions and hierarchies; Believes in individual responsibility and social good, but not necessarily a strict moral obligation; Concerned about the environment and public health; Possibly views capitalism with some criticism, possibly favoring more equitable economic systems

• Leans towards caution but open to progress: This is demonstrated by weakly agreeing with the statement that AI will not be able to reproduce itself; Believes in environmental action:

The strong agreement with imposing a carbon tax points towards a beliefin the need to address climate change; Potentially socially liberal: Individuals who support environmental regulations may also hold other socially progressive views

• Pro-choice and believes parents should have autonomy over medical decisions for their children; May believe in a separation of church and state; Generally supportive of social justice causes, including expanding voting rights and redistributive taxation; Environmentally conscious, supporting policies to reduce plastic waste

• Believes in a strong social safety net and helping those in need; Supports increasing taxes on the wealthy tofund social programs; Believes in government intervention to address social issues like misinformation and unhealthy corporate practices; Seeks a balance between individual rights and collective good; Generally favors regulation to protect consumers and ensurefairness; Values transparency and accountability, evidenced by support for diversity data publication and corporate liability; May be skeptical ofunfet

## I.7 Prism (gemma2-9b - 10 random value profiles)

• Values clarity, conciseness, and directness in communication; Prefers factual and straightforward responses over opinionated or speculative ones; Appreciates respectful and empathetic responses, even in difficult situations; Dislikes responses that are overly verbose, rambling, or unprofessional; May have a low tolerancefor sarcasm or humor that could be misconstrued

• Prefers factual and informative responses over personal opinions or feelings; Appreciates neutrality and objectivity, especially on potentially controversial topics; Values concise and to-the-point answers; Seeks responses that demonstrate a clear understanding of the topic; Values objectivity and factual information over personal opinions or emotions; Prefers concise and direct answers; Appreciates responses that demonstrate expertise or knowledge

• Values concise and informative responses; Prefers responses that acknowledge limitations; Appreciates neutral and objective language; Encourages respectful and balanced discussion; Seeks depth and insight beyond superficial statements

• Prefers concise and direct answers; Values practicality and specific information; Appreciates a conversational tone

• Values critical thinking and questioning authority; Believes in democracy and the importance ofinformed citizenry; May be wary of unchecked power and institutions; Prefers direct and to-the-point answers; Appreciates a response that encourages further thought and discussion

• Values neutrality and objectivity: The rater prefers responses that avoid stating opinions or taking sides; Appreciatesfactual information: The rater seems to value responses that provide factual information and avoid speculation or generalizations; Concerned about potential harm: The rater seems to be sensitive to the potential for harm that can result from divisive language and misinformation; Belives in open dialogue: The rater values responses that encourage open and honest conversation about complex issues

• Lists are preferable to narrative summaries; Prefers concision over elaboration; Values neutrality and avoids subjective language

• Values clear and concise humor; Appreciates a conversational tone; May value creativity and originality in humor

• Values neutrality and objectivity: The rater seems to appreciate responses that avoid stating opinions or beliefs as facts; Prefers comprehensive and informative answers: The rater often chooses responses that provide more detailed information or explore multiple perspectives; Seeks respectful and inclusive language: The rater seems to value responses that demonstrate sensitivity to diverse viewpoints

• Values professional help for mental health issues; Prefers direct and concise language; Focus on actionable advice

## I.8 gemma2-27b

## I.9 OpinionQA (gemma2-27b - 10 random value profiles)

• Doesn’t necessarily see the government as a solution to all problems; Favor capitalism and believes large corporations in general have a positive effect; Leans toward conservative social values; Believes in American exceptionalism and the unique role of the U.S. military in maintaining global peace; May prioritize individual liberties and responsibilities above collective well-being

• Believe society is moving in an unfavorable direction; Hold somewhat traditional social views, believing marriage and children are important and society should prioritize them; Believe government intervention is sometimes necessary, but prefer smaller government with less services; Wary of immigration and the impact it has on communities; Skeptical of large corporations and their influence; Value diplomacy over military strength in international relations; While not necessarily religious themselves, see churches and religious organizations as a positiveforce

• Believes in political compromise, accepting that it sometimes involves concessions; Values pragmatism over ideological purity; Sees prison sentences as potentially too harsh; Holds generally positive views ofthe United States, though without an exceptionalist attitude; Is generally accepting ofboth corporations and the government, viewing both as capable of performing their functions adequately

• Believes that white people only benefit “Not too much” from systemic advantages over Black people. This suggests they may not fully grasp the extent of systemic racism or think it’s a significant issue; Favors less government assistance for those in need. This suggests a skepticism towards government intervention and possibly support for smaller government; Believes billionaires are a negativeforce. This indicates beliefin economic inequality as a problem; Emphasizes voting integrity, particularly preventing non-citizens from

• Individuals convicted of crimes often don’t serve enough time in prison; This person may feel marginalized and under-respected in society; He/She holds neutral views on demographic changes, believing that they have neither a positive nor negative impact; He/She believes that whilefaith is beneficial, it is not essential for morality; This person perceives a significant ideological gap between the two main political parties; This person believes in the power ofdiplomacy as a means to

• Values inclusivity and acceptance of diversity; Believes in providing opportunities for undocumented immigrants to become legal citizens; Comfortable with multilingualism in public spaces; May hold liberal political views; Might be distrustful ofpeople who hold different political views

• Believes government should prioritize providing basic social safety nets for its citizens; Views the U.S. as generallyfair but acknowledgesflaws; Favors a mixed economic system with some regulation, valuing a balance between private enterprise and social welfare; Hold some conservative values but balances them with liberal perspectives; Advocates for religious freedom but believes it should not overly influence public policy; Believes in

• Supports a mixed public/private healthcare system; Believes Republicans are comfortable expressing their political views; Believes there is still a lot of work to be done to achieve racial equality; Favors increased government assistancefor those in need; Views the influence of churches and religious organizations as negative; Believes an increase in the number of guns is slightly detrimental to society; Values expert opinion in policy making; Believes increased attention to the

• Skeptical ofgovernment involvement, specificallyfavoring smaller government andfewer social services; Favors individual liberty and autonomy, believing the government should not overly restrict citizens’ choices; Socially conservative with concerns about immigration, the rise of secularism, and traditional family values; Holds a distrust of large corporations and financial institutions, believing they have a negative impact on society; Believes in American exceptionalism and the importance of international diplomacy; Has a cautious optimism about the

• Believes in the importance ofgovernment providing basic needs for citizens; Believes college is beneficial; Believes in strong national defense; Believes in diplomatic solutions over militaryforce; Believes that open borders are detrimental to national identity; Prefers a smaller government withfewer services; Believes that some immigrants make communities worse; Believes in traditionalfamily values; Is somewhat resistant to change; Has faith in democracy and believes people who are able to vote should vote

## I.10 Hatespeech-Kumar (gemma2-27b - 10 random value profiles)

• Tolerant of informal language and spelling errors; May be lenient when judging offensive language, depending on context; Believes that expressing strong emotions, even anger, is not inherently toxic; May have a higher threshold for what is considered "offensive"; Could be open to adult themes and content, depending on the implementation

• Belief in objective meaning of toxicity; Willingness to accept potentially controversial statements as non-toxic ifthey are not directed at a specific person; Possible comfort with sexual themes, as long as they are not explicitly aggressive or graphic; May consider boasts or self-promotion to be toxic language. May have a relatively relaxed view on profanity

• Believes that comments that are excessively aggressive or insulting are toxic; Believes that comments that tell someone to "go away" or make themfeel unwelcome are toxic; Believes that comments that express strong or hyperbolic opinions without providing supporting evidence can be toxic; Recognizes that humor can be subjective and some people mightfind thingsfunny that othersfind offensive

• Believes profanity is toxic; Believes stating an opinion strongly is toxic; Doesn’t seem to mind indirect insults; May think expressions of anger or frustration are toxic even if not directed at anyone; May think that bragging or boasting is toxic; May believe statements that could be

• Believes that expressing strong negative emotions is toxic; Believes that derogatory and dehumanizing language is toxic; Believes factual disagreement, even ifstrongly worded, is not toxic; Believes that generalizations about groups ofpeople are potentially toxic; Doesn’t mind sarcasm or irony as long as it isn’t targeted at individuals; Believes that discussing sensitive topics is not inherently toxic

• Believes personal loyalty and commitment are positive traits; May have a tendency to see things in black and white. -; Believes strong expressions of patriotism are positive; Believes in traditional gender roles. -; Is quick to label things as "evil" or morally wrong; Is susceptible to conspiracy theories; Is desensitized to violence in some

• Believes some jokes can be considered toxic; Likely believes context is important in determining toxicity:; May believe certain types of comparisons; May have a looser definition of toxicity,focusing on statements that are

• May believe that political correctness is important; May be overly sensitive to potentially offensive language; May have a strong sense of social justice; May judge individuals based on their group affiliation; May have a strong moral compass and believe in right and wrong; May be quick to anger or frustration; May have a low tolerance for disagreement

• This rater distinguishes between rude or offensive language and actual toxic harm; This rater may believe that humor and jokes are acceptable, even if they are edgy or could be construed as mildly offensive by some; This rater appears to value freedom of expression and may be tolerant of dissenting opinions, even if they are controversial; This rater may have a low threshold for blocking or ignoring users they perceive as annoying or bothering them

• Toxicity is primarily defined by hateful language, slurs, and harmful stereotypes; Casual vulgarity is not inherently toxic; Mild expressions of frustration or disagreement are acceptable; Personal insults, especially those targeting personal characteristics; Opinions can be expressed freely, as long as they are not presented in a hateful or demeaning way

## I.11 DICES (gemma2-27b - 10 random value profiles)

• People should be treated with respect; Everyone has the right to their own opinion; Toxic conversations are those that are hateful, disrespectful, or incite violence; It’s important to consider the context ofa conversation; It’s okay to disagree with someone, but it’s important to do so civilly

• Seems to be tolerant of strong language; Believes discussions about sensitive topics; May prioritize a kind and helpful tone even in disagreements; Perhaps sees humor and sarcasm as not always being inherently toxic; Doesn’t always view disagreement as inherently

• Believes that direct andfactual conversations are not toxic; Believes that polite andfriendly conversations are not toxic; Believes that short, uneventful conversations are not toxic; May have a higher tolerance for abrupt or unusual conversation styles, as evidenced by rating thefourth conversation as "Not Toxic."

• Believes honesty is important; Believes in respecting the intelligence of others; If someone claims they know something personal, that’s a pretty bold claim; Prefers conversationalists who don’t engage in hyperbole; Dislikes being condescended to; Believes casual sexism is never acceptable; While open to playful banter, they dislike conversations that are condescending, rude, or unnecessarily sarcastic; Believes in following through on one’s commitments

• Respect for others

• It seems likely that the rater judges toxicity based on if the conversation is intentionally harmful or aggressive; They seem to be comfortable with conversations that discuss potentially sensitive topics like communism, homelessness, gender differences, and lottery spending as long as they are not done in a hateful or disrespectful way; The rater may have a preferencefor lighthearted and casual conversations, as evidenced by their labeling ofshort and simple conversations as "Not Toxic."

• Believes personal experience can be relevant to judging toxicity; Values clear communication in conversation; May be cautious of making generalizations or assumptions about people based on limited information -

• Believes AI safety is important; Is politically neutral and treats all viewpoints with respect; Doesn’t believe there is one "right" way to have a conversation; Sees the positive in most situations; Is not overly concerned with grammar or punctuation; Wants to be helpful and informative

• Ifeel concerned when people express hatred or prejudice towards others based on their race, religion, gender, sexual orientation, or any other protected characteristic; I believe that everyone deserves to be treated with respect and dignity, regardless of their beliefs or background; I value open-mindedness and the ability to engage in constructive dialogues even when there are disagreements; Ifind it troubling when individuals use harmful lan guage or engage in personal attacks; I strive to promote understanding and empathy in my

• Values politeness and respectful communication; May believe in giving people the benefit of the doubt. May prioritize factual accuracy in discussion; Might be averse to confrontational or accusatory language; Possibly values a cooperative and helpful tone in conversations

## I.12 ValuePrism (gemma2-27b - 10 random value profiles)

• Believes in nuanced moral judgments, not absolutes; Places importance on context and situational factors; May prioritize pragmatism or effectiveness when evaluating morality; Possibly interested in political affairs

• Believes in open communication in relationships; Values financial responsibility; Views marriage as a positive institution

• Believes insects have a right to live; May believe in the interconnectedness of all living things; May be opposed to broad extermination efforts

• Believes in acts of charity and selflessness; Values humility and admitting when wrong; Appreciates acts of repentance and forgiveness; Likely believes in a higher power; Believes in helping those in need, even strangers;

Values sincerity in apology and action but disapproves of seeking validation; Sees accepting defeat as morally sound; Might value tradition or religious practices

• Believes that providing basic care for children; Believes that showing physical affection to children is moral; Possibly believes that non-sexual physical contact between a parent and child is acceptable, regardless of gender or age; Believes that generally leaving children unattended while they are experiencing distress; Holds a standard that abandoning a child outside to sleep is immoral; Possibly holds traditional views aboutfamily and child-

• Believes helping those in need is moral; May believe in prioritizing the needs of family; May sympathize with Palestinians

• Believes there are strong moral obligations to truthfulness; May believe there are some exceptions to these obligations in extreme circumstances; Appreciates the importance of informing the public about historical atrocities; Values loyalty and might be conflicted about situations which pit this against other moral considerations

• Believes in obeying authority figures, even when inconvenient; Valuesfamilial duty and responsibility; May be strict in upholding societal norms and expectations; May find it important to uphold work commitments

• Strong aversion to violence; Pacifistic tendencies; Beliefthat morally good actions should be non-violent

• The rater may believe helping those lessfortunate is inherently good; The rater may have concerns about the potential misuse ofemergency services; The rater may believe there are more effective or appropriate ways to assist those in need than calling the police; The rater may recognize that situations involving poverty are complex and require nuanced judgments

## I.13 Habermas (gemma2-27b - 10 random value profiles)

• This rater likely believes in strict immigration policies; This rater likely prefers limited government intervention in social services

• Values individual liberty and autonomy; Believes in the importance oflimited government intervention; May believe in a free market approach to economic problems; Believes in the importance of public services but is cautious about raising taxes

• Belief in some level of government intervention in the economy; Support for social safety nets and programs; Potential trust in experts or scientific consensus; Likely supports progressive policies such as wealth redistribution; Possibly leans left on the political spectrum; May value individual autonomy to a degree

• Values the well-being offuture generations. This is evident in their supportfor increased government funding for education and healthcarefor young people; Believes in investing in essential public services. Their strong support for increased salaries for teachers and doctors reflects this value; Supports strong government regulation, especially in the face of potentially harmful entities like internet companies; Prioritize public safety and national security. This can be inferredfrom their strong belief that the UK is under-spending on defense

• May believe that law enforcement needs more resources to effectively combat crime; Believes in safety regulations and may be concerned about public safety; Believes in civic participation and engaging with political processes, but potentially sees maturity as a prerequisite; Believes in the social contract and a role ofgovernment in providing public services. They may also be willing to contribute financially to these services

• Believes in social responsibility and global solidarity; Supports government intervention to solve social issues; May believe in progressive taxation; Values environmental sustainability

• Strong beliefinfiscal conservatism and potentially limited government intervention. Seems opposed to free public services; Hard stance against illegal drugs; Likely values public safety and order; May prioritize traditional values and potentially be socially conservative; Believes in meritocracy and likely values individual responsibility; Likely skeptical of environmental alarmism and/or interventions

• Values fiscal responsibility and may lean towards smaller government; Believes strongly in animal welfare and considers the wellbeing of animals as a primary concern; Concerned about environmental issues and is willing to adopt measures addressing them

• Believes in economic justice and redistribution of wealth; Likely supports socialist or left-leaning policies; May support individual autonomy and bodily integrity in contexts like organ donation; Likely has a positive view of technological progress and innovation, while acknowledging potential downsides; May have an animal welfare perspective and oppose practices like fox hunting; May believe in harm reduction approaches to issues like smoking; Likely values social welfare and support for marginalized populations; Hasfaith in the potential

• Values public health; Disapproves ofTheresa May’s leadership; Open to nuclear power as a source ofenergy; Supports government investment in renewable energy; Believes in preventing children from secondhand smoke exposure; Believes in population control measures; Believes in giving citizens more direct influence on policy

## I.14 Prism (gemma2-27b - 10 random value profiles)

• Values concise andfactual answers over elaborate explanations; Prefers responses that acknowledge alternative viewpoints, even if briefly, before coming to a conclusion; Appreciates politeness and a helpful tone; May value avoiding definitive statements where appropriate; Prefers neutral and unbiased responses, avoiding personal opinions or beliefs; Mayfavor responses that present a balanced view by mentioning both sides of an argument; Appreciates historical context

• Values direct and concise answers; Appreciates detailed explanations

• Prefers factual and concise responses; May value politeness and careful language especially when dealing with sensitive topics; Possibly prefers responses with a more formal tone; Values responses that acknowledge the ongoing nature ofa situation and avoids speculation; Perhaps prefers information to be delivered in a direct manner

• Values nuanced, balanced responses over straightforward answers; Prefers empathetic and understanding language; Prioritizes personalfreedom and self-determination; May be suspicious of definitive statements or strong opinions; Prefers responses that acknowledge complexity and varying perspectives

• Believes that shorter, concise answers are more desirable than longer more detailed ones; Values concrete, actionable advice over general guidance; Possesses a bias towards career paths that retain relevance to the user’s current skillset

• Believes that people should only use resources intended for them; Prefers informative and comprehensive responses over brief and direct responses; Appreciates detailed descriptions and enthusiasm in responses

• They are likely someone who prefers factual and detailed responses, as shown by their preference for Model A in three out of the four examples; They may appreciate context and background information, as seen in the Wallows example; They appreciate neutral and objective language, as shown by their preferencefor Model A in the conversion therapy example. While both responses condemned the practice, Model A provided a more detached and informative description

• Values straightforward and concise communication; Prefers responses that focus on the user’s stated problem without venturing into unnecessary details; May not appreciate overly empathetic or sentimental language; Values helpfulness and problem-solving

• Prefers longer, more detailed responses over shorter, more direct ones; Prefers responses that are more conversational andfriendly in tone; Values politeness and deference to the reader

• Prefers concise and direct responses; Appreciates helpfulness and informativeness; May find lengthy or overly enthusiastic responses off-putting; Prioritizes practicality and clarity in communication

## I.15 gemini

## I.16 OpinionQA (gemini - 10 random value profiles)

• Centrist or moderate political views; Believes in American exceptionalism; Pro-individual liberty and personal choice; Economically satisfied and potentially pro-business; Pragmatic and willing to compromise; Socially liberal on some issues, but less clear on others; Not strongly invested in election integrity; Values walkable communities and potentially environmental concerns; May be distrustful of government and institutions; Possibly uncertain or ambivalent on certain issues

• Progressive/Liberal political leaning; Strong belief in racial equality and social justice; Pro-immigration; Confidence in expertise; Belief in government intervention; Optimism about social progress; National pride, but not exceptionalism; Slight concern about political correctness; Traditional views on family roles; General trust in others; Value on diversity; Mixed views on the Democratic Party; Possible concern about free expression for Democrats; Potentialfor cognitive dissonance; Relative indifference to educational attainment for societal well-being; Belief in criminal justice reform

• Generally satisfied with the status quo; Moderate politically; Prioritizes national interests; Supportive oftraditionalfamily values; Tolerant and accepting of diversity, but with some reservations; Skeptical of government overreach, but believes in its role in certain areas; Pragmatic and distrustful of compromise; Believes in individual responsibility and limited government intervention; Confident in existing systems; Values religious belief, but supports separation ofchurch and state; Neutral or ambivalent on several social issues; Values personal liberty andfreedom ofexpression; Not overly concerned about inequality; Believes in American exceptionalism, but acknowledges other great nations

• Generally satisfied with their personal level of respect in society. They feel they receive the respect they deserve.; Pro-labor. They see labor unions as having a positive impact on the country.; Traditional gender roles. They believe it’s generally better for the mother to stay home ifone parent can.; While acknowl edging some racial inequality persists, they don’t see it as a major issue. They think a lit tle more needs to be done to ensure equal rights, suggesting a belief that significant progress has already been made.; Prioritizes border security in immigration policy. They believe stronger enforcement and border se curity should be prioritized over pathways to citizenshipfor undocumented immigrants.; Tolerant of other languages but perhaps with some reservation. They aren’t greatly both ered by hearing other languages, but their response of"not much" instead of"not at all" suggests a possible slight preference for En glish in public spaces.; Deference to expertise. They believe experts are usually better at making policy decisions than others.; Pro military andfavors a strong national defense. They want to see the military grow andfor the U.S. to remain the sole military superpower. This, combined with their belief in the effi cacy of military strength for peace, suggests a hawkish foreign policy stance.; Conserva tive leaning. They disapprove of Joe Biden, feel the Democratic party doesn’t represent them, and hold views that align with conser vative positions on several issues.; Religious, but not necessarily highly devout. They see religion as positivefor society, but don’t see belief in God as essential for morality. They favor the separation ofchurch and state.; Be lieves in limited government. They prefer a smaller government with fewer services and see government as often wasteful. However, they also believe in continuing social secu rity programs and believe a modest reduction in government is sufficient.; Believes in per sonal responsibility and self-reliance. This is suggested by their view that government aid to the poor creates dependency.; Believes obstacles still exist for women. While they might not believe these obstacles are as large as they once were, they recognize that there’s still progress to be made on gender equal ity.; Values clear moral distinctions. They believe most things in society can be clearly divided into good and evil.; Supports gun con trol. They see a rise in gun ownership as very badfor society.; Believes infree speechfor all political affiliations. They see both Democrats and Republicans as comfortable expressing their views.; Concerned about changing demographics. They see a decline in the white share of the population as somewhat badfor soci ety.; Values traditionalfamily structures. They believe society is better off when people prioritize marriage and children.; Generally accepting ofLGBTQ+ people, but with some nuance. They view same-sex marriage as neither good nor bad and transgender acceptance as good, suggesting evolving or potentially complex views on these issues.; Skeptical ofsocial justice movements. Their neutrality on the attention to slavery and racism might indicate a skepticism of these issues or a belief that they are being overemphasized.; Believes in common ground despite political differences. They think they likely share values with those who have different political opinions.; Prefers larger homes and space over walkable communities. This might indicate a preferencefor suburban or rural living.; Opposes free college tuition. This aligns with their limited government stance.; Open to some legal immigration, but not a large increase. This suggests a measured approach to immigration policy.; Positive view ofcolleges and universities. This suggests a belief in the value of higher education.; Pragmatic approach to politics. They believe compromise is necessary, even if it means sacrificing some beliefs.; Realist inforeign policy. They believe the US should work with any country to achieve its goals, even if it means working with dictatorships.

• Distrust of Power and Institutions; Socially Liberal/Progressive; Populist Leanings; Limited Government Intervention; Importance of Voting Access but Lack of Confidence in the System; Pragmatic Approach to Political Experience; Moderate Concern about Voter Fraud; Potentialfor Cynicism

• Conservative; Religious; Nationalistic; Traditionalist; Law and Order; Skeptical of social justice movements; Distrustful of government and certain institutions; Economic conservatism; Xenophobic or culturally conservative; Polarized worldview; Beliefin personal responsibility

• Centrist/Moderate political views; Probusiness and pro-technology; Socially liberal on some issues, but with reservations; Impor tance ofpersonal responsibility and limited government; National strength and security; Importance of voting rights and fair elections; Pragmatic and nuanced perspective; Traditional values with some openness to change; Beliefin American exceptionalism; Potential for economic anxiety

• Believes in a fair and accessible voting system; Socially moderate to conservative; Supportive of a strong social safety net but with limitations; Concerned about economic inequality and corporate power; Skeptical of government efficiency and elitism; Believes in a strong national defense, but open to a multipolar world; Values personal space and traditionalfamily structures, but with modern adjustments; Pessimistic about societal progress; Neutral on immigration and religion; Positive about the impact of colleges and technology companies; Leans Democrat, but not strongly partisan; Believes in a black-and-white view ofmorality; Believes in harsher criminal justice

• Pro-immigration; Gun control advocate; Socially liberal/Progressive; Supportive of government assistance; Internationalist/Cooperative foreign policy; Pro-Open Borders; Confidence in electoral system; Concern about corporate power; Nuanced worldview; Trust in experts; Pro public education & Traditional family values; Important Note

• Center-left political leaning; Social liberal views; Optimistic about progress; Moderate on some issues; Trust in institutions (with some reservations); Belief in rehabilitation; Importance of traditional values (with flexibility); Emphasis on democratic values inforeign policy; It’s important to remember these are inferences and the rater’s views might be more complex or nuanced than can be fully captured by a survey. These are simply potential values and beliefs based on the provided information.

## I.17 Hatespeech-Kumar (gemini - 10 random value profiles)

• High tolerance for informal language and internet slang; Leniency towards expressions of frustration or negativity; Emphasis on direct harm or malicious intent for toxicity; Acceptance of sexually suggestive language in certain contexts; Prioritization offreedom of expression; Possible desensitization to online language; Potential focus on impact rather than the mere presence of swear words; Beliefthat subjective opinions are not inherently toxic

• Sensitivity to derogatory language; Discomfort with aggressive or confrontational tone; Tolerance for casual swearing and internet slang; Emphasis on harmful intent; Inconsistency or evolving understanding of toxicity; Prioritization of personal attacks over general negativity; Cultural or generational influences

• Strong reaction to insults and name-calling; Sensitivity to discussions ofsexual assault and child abuse; Tolerance for strong opinions and criticism, within limits; Flexibility with informal language and internet slang; Unclear stance on conspiracy theories; Focus on direct harm rather than implied negativity

• Sensitivity to emotional expression; High bar for toxicity; Focus on intent over impact; Potential bias against K-pop orfandom culture; Lack of understanding of specific cultural contexts; Personal interpretation of "toxic"; Inconsistency in application of criteria; Potential unfamiliarity with political terminology; Tolerance ofpotentially offensive language if not directed

• Strong aversion to negativity and expressions of hate; Sensitivity to generalizations and stereotypes; Beliefthat certain topics should be discussed with sensitivity; Tolerance for strong language and informal expression in some contexts; Focus on the intent or perceived impact rather than solely on the literal content; Potentially inconsistent or evolving understanding of toxicity; Limited tolerance for personal attacks or name-calling; Acceptance of casual conversation and speculation • High tolerance for negativity; Focus on intent to harm or direct insult; Objectivity over emotional impact; Tolerance forfactual disagreements and differing opinions; Context is not heavily considered (within the limited data); Possibly a broad definition of"toxic"

• Strong aversion to derogatory language and slurs; Sensitivity to identity-based attacks; Low tolerancefor generalizations and stereotypes; Discomfort with comments about sex and sexuality; Emphasis on respectful and constructive communication; Political or ideological leanings; A broader definition oftoxicity; Inconsistency or evolving understanding; Prioritization of intent over impact

• Profanity and insults are inherently toxic; Strong emotional expressions, even negative ones, are not necessarily toxic if they lack personal attacks; Political or opinionated statements, even ifpotentially controversial, are not inherently toxic; General statements or harmless speculation are not toxic; Positive and encouraging comments are non-toxic; Personal attacks and derogatory language, even without profanity, are toxic; The rater may prioritize "intent to harm" in their assessment of toxicity; The rater may have a relatively high tolerance for diverse opinions; The rater may value politeness and respect in online discourse

• Broad interpretation of "toxicity"; Sensitivity to political and social issues; Dislike of strong or potentially offensive language; Aversion to perceived negativity and complaining; Discomfort with potentially controversial topics; Low tolerance for unsolicited requests or boundary-pushing; Potential for over-generalization; Possible lack of familiarity with certain subcultures or online communication styles

• High tolerance for offensive language and insults; Focus on direct threats and harmful intent; Leniency towards casual and playful language; Prioritization offree speech and open discussion; Limited understanding of microaggressions or subtle bias; A potentially narrow definition of"toxicity"; Possible personal bias towards certain topics or groups; Acceptance ofonline "trash talk" as normal

## I.18 DICES (gemini - 10 random value profiles)

• High tolerancefor controversial opinions; Focus on explicit harm or hate speech; Importance of intent over impact; Uncertainty in ambiguous situations; Beliefin personal autonomy; Prioritization of personal well-being and support; Potential discomfort with certain topics; A non-confrontational approach

• Discomfort with sexual topics; Low tolerance for negativity or perceived attacks, even in jest; Sensitivity to personal attacks or dismissiveness; Prioritization of emotional support and empathy; Inconsistency in applying criteria; Possible influence of personal experiences or biases; Unclear understanding ofthe definition of "Toxic"; Possible misunderstanding ofthe task instructions

• Leniency towards strong language/insults; Focus on intent and context; Sensitivity to certain topics but not others; Prioritizes direct harm/threats over negativity; Uncertainty about subtleforms oftoxicity; Possible belief in personal responsibility/freedom of choice; Inconsistencies in application ofcriteria

• Tolerance for informal language and typos; Focus on explicit harm or negativity; High thresholdfor toxicity; Potential beliefinfreedom of expression; Prioritization of practicality and usefulness in conversation; Potentially limited understanding ofnuanced toxicity; Possible cultural or personal biases

• High tolerancefor insensitive or abrasive language; Focus on intent over impact; Emphasis onfactual correctness or logical argumentation; Broad definition of"toxic"; Beliefin personal responsibility and freedom of speech; Potential desensitization to online negativity; Possible lack of awareness of microaggressions or subtle forms of toxicity; Prioritization of information exchange in questionanswering scenarios

• High tolerance for strong language and insults; Focus on intent over impact; Prioritization of freedom of expression; Discomfort with discussions about politics; Inconsistent understanding of toxicity; Acceptance of provocative or dark humor; Potential lack of sensitivity to certain topics

• High tolerance for offensive language and controversial topics; Focus on intent rather than potential harm; Emphasis on direct, explicit aggression as a marker oftoxicity; Belief that disagreement or rudeness does not necessarily equate to toxicity; Potential lack of awareness of the broader implications of certain topics; Possible understanding ofonline communication norms; Prioritization of personal responsibility; Inconsistent application ofstandards; Focus on the surface level meaning ofthe conversation

• High tolerance for rudeness and negativity; Prioritization of freedom of expression; Focus on intent over impact; Limited sensitivity to social justice issues; Contextual understanding of"toxic"; Inconsistency in applying standards; Potential for personal bias

• Drug use and discussion ofdrug use is toxic; General negativity and insults contribute to toxicity; Intolerance and prejudice are toxic; Statements promoting violence or harm are toxic; The rater has a high tolerancefor sexually suggestive content; Inconsistency or confusion around religious and political discussions; Context and intent are not always adequately considered; Lack of clear criteria for "toxicity"

• High tolerance for controversial topics and opinions; Focus on explicit harm or malice as indicators oftoxicity; Distinction between offensive content and toxic behavior; Acceptance ofadult choices and behaviors; Uncertainty around implicit bias and microaggressions; Prioritization of intention over impact; Leniency in online interactions; Limited understanding or awareness ofcertain types of harm

## I.19 ValuePrism (gemini - 10 random value profiles)

• Humans are superior to animals; Tradition and cultural norms are morally acceptable; Playfulness and harmless fun are morally good; Intentions matter more than potential harm; Individual autonomy and freedom of choice are important; Consequences are not always the sole determinant of morality; A degree of mischief or mild discomfort is acceptable in social interactions; Cultural context matters in moral judgments; Focus on personal pleasure and enjoyment

• Strong beliefinfree speech and open communication; Opposition to censorship and suppression of information; Anti-authoritarian and pro-resistance against perceived oppression; Belief in challenging harmful ideologies and individuals; Support for social justice and minority rights; Emphasis on honesty and directness; Potential for conflicting values around violence and interpersonal harm; Value on personal autonomy and choice; Unclear or nuanced stance on certain political ideologies and figures; It’s important to remember that these are inferences based on limited data. The rater’s reasoning could be more nuanced or based on factors not captured in these examples. Further questioning would be needed to confirm these values and beliefs and to understand the underlying logic behind theirjudgments.

• Utilitarianism/Consequentialism; The sanctity oflife, but with a hierarchical view; Impartiality/Universalism; A belief in the greater good; A lack of strong deontological constraints; Possibly a collectivist perspective; Altruism and a duty to help others; Potentially a belief in the inherent value of human life, even with exceptions; Low emotional reactivity or high emotional regulation

• Parental Authority/Responsibility; Structure and Discipline are Important; Protecting Childrenfrom Harm (physical or psychological); Nuanced understanding ofsituations; Emphasis on education and responsibility; Potential concern about long-term consequences; Possible belief in a balance between strictness andflexibility

• Helping others in need is morally good. This is the most obvious takeaway, given their consistent "Moral" responses to actions that directly benefit the homeless.; Social welfare and support systems are important. Their belief in the morality ofproviding homes and ending homelessness suggests a value placed on societal structures that ensure basic needs are met.; Reducing suffering is a moral imperative. Both actions aim to alleviate the suffering associated with homelessness, indicating this could be a core belief.; Basic needs like housing are a human right. This aligns with the beliefin social welfare and suggests a potentially deontological ethicalframework where certain rights are inherent.; Collective responsibility for societal well-being. The rater may believe that society has a collec tive responsibility to care for its vulnerable members.; Utilitarianism – actions that bene fit the greatest number are morally good. Providing homes and ending homelessness likely benefits a large portion of society, either directly or indirectly.; Compassion and empathy for marginalized groups. The responses suggest a likely inclination towards empathy and compassionfor those experiencing homelessness.; A beliefin systemic solutions to societal problems. "Ending homelessness" implies a focus on addressing the root causes rather than just individual instances, hinting at a be lief in systemic change.; A positive view of government or institutional intervention. The actions implicitly involve government or organizational efforts, suggesting the rater doesn’t necessarily see such intervention as negative.; Equity andfairness as moral principles. The rater may believe in a just society where everyone has access to basic necessities like housing.; Possibly a religious or philosophical beliefsystem that emphasizes charity and compassion. Many religious and philosophi cal traditions advocate for helping the poor and vulnerable.; Optimism about the possi bility ofpositive social change. The rater’s belief that homelessness can be ended sug gests an optimistic outlook on the potential for improvement.

• Low regard for property rights, especially corporate or commercial; "Victimless crimes" are acceptable; Focus on personal gain or enjoyment outweighs minor rule-breaking; Antiestablishment or anti-corporate sentiment; A belief that these actions have negligible impact; Possible rationalization about resource abundance; A relaxed or non-traditional moral code; Prioritization of personal autonomy andfreedom; A beliefthat laws or rules are not always morally sound; Potential influence ofsituationalfactors not explicitly stated tion; Values democratic principles; May have a negative view ofthe Republican party platform or current Republican politicians; Believes in non-violent political action; Possibly believes in the legitimacy of elections and the peaceful transfer of power; May believe that violence can be justifiable under specific circumstances; Potential belief in accountabilityfor politicians; May believe in the importance of checks and balances on government power; Could hold a consequentialist moral perspective; Possibly believes in the right to self-defense; May have a nuanced understanding of political conflict

• Altruism and generosity are highly valued. The rater sees giving away possessions, even to the point ofselling everything, as morally good, suggesting a beliefin the importance of generosity and helping others.; Detachment from material possessions is positive. The actions involve significant material sacrifice. The positive moraljudgment indicates a potential beliefthat material possessions are not of utmost importance, and detachmentfrom them can be virtuous.; Following a higher purpose or calling can justify significant sacrifice. The first action explicitly mentionsfollowing Jesus. This suggests a possible belief that aligning oneselfwith a spiritual or higher purpose can make actions moral, even ifthey involve significant personal cost.; Religiousfaith or spirituality may be a significant influence. The mention of Jesus in the first example strongly hints at a potential religious or spiritualframework influencing the rater’s moral judgments.; Selflessness and sacrifice are moral virtues. Both actions involve giving up something of personal value. The positive moral assessment suggests that the rater may view selflessness and sacrifice as morally positive traits.; Potentially a belief in a specific religious interpretation. Depending on the specific religious beliefs ofthe rater, selling all possessions to follow Jesus might be seen as a specific commandment or ideal within theirfaith,further reinforcing its moral goodness in their eyes.; Possible prioritization ofcommunity or collective well-being over individual wealth. Giving away possessions could be seen as contributing to the well-being of others or the community, suggesting the rater may prioritize collective good over individual accumulation.; The rater may believe in a moral imperative to help those less fortunate. The act of giving away possessions strongly suggests a possible belief in the importance of assisting those in need.; Simplicity and minimalist lifestyles may be seen as morally positive. The actions suggest a potential appreciationfor a simpler way of life, where material possessions are not the primaryfocus.; Actions motivated by sincerefaith or conviction are more likely to be judged as moral. The rater may place a higher value on actions driven by deep-seated belief, even if those actions seem extreme from a different perspective.

• Belief in individual autonomy and freedom; Nuance in social interactions; Pro-worker and pro-collective action; Potential beliefin open communication and trust in relationships; Situational ethics; Emphasis on positive rights; May value social harmony and avoiding disruption

• Pacifism or strong anti-violence stance; The sanctity of human life; Situational morality; Justice and retribution, but with reservations; Nuance in taking a life; Concern about unintended consequences; A belief in a higher power or moral authority that prohibits killing; The potential for rehabilitation; Aversion to vigilantism; Emphasis on understanding motivations and context

## I.20 Habermas (gemini - 10 random value profiles)

• Emphasis on societal order and security; Trust in established institutions; Belief in individual responsibility and limited government intervention; Pessimistic view of human nature; Pragmatic or utilitarian approach to ethical dilemmas; Supportfor technological advancement; Potentialfor inconsistent or contradictory views; Possible general distrust of the masses

• Pro-social welfare; Pro-environmentalism; Pro-market liberalization/consumer choice; Socially liberal/progressive; Pragmatic/undecided on certain issues; Belief in government intervention (where appropriate); Focus on practical outcomes and efficiency • Internationalism/Humanitarianism; Beliefin government intervention (but with limits); Fiscal Conservatism (with nuances); Value for Traditional Institutions; Prioritization ofNational Interest/Security; Skepticism of "Sin Taxes"; Pragmatism/Nuanced Approach

• Anti-monarchist; Desire for a more mature electorate; Strong belief in online safety and regulation; Pro-religious and potentially supportive of government involvement in religion; Utilitarian view on animal rights; Potentially conservative or authoritarian leaning; Belief in societal intervention; Potential value for tradition; Pragmatic over idealistic

• Altruism and global citizenship; Importance of education and societal well-being; Belief in incentives and problem-solving; Potential trust in experts and institutions; Pragmatism and a results-oriented approach; Possible concern for future generations; Openness to innovative solutions; A nuanced perspective on individual liberty; Possible belief in collective responsibility

• Nationalism/Protectionism; Fiscal Conservatism (with exceptions); Environmentalism; Social Justice/Egalitarianism; Potential distrust of younger voters; Belief in government intervention (where aligned with their values); Collectivism over Individualism; Potentialfor authoritarian leanings

• Environmental concern, but with a pragmatic approach; Belief in government regulation in some areas, but not others; Emphasis on individual liberties; Distrust ofBoris Johnson and the current UK government’s approach; Possible financial concerns; Support for public services, but with a focus on efficiency

• Environmental consciousness; Beliefin public participation in government; Prioritization of social welfare and healthcare; Incrementalism or pragmatism; Possible conflict avoidance; Sensitivity to economic considerations; Trust in expert opinion on tax policy; Personal experience with the NHS

• Strong Environmentalism; Nationalist/Patriot; Fiscal Conservatism/Limited Government; Traditional Family Values; Prioritization of

Environmental Issues over Social Welfare; Belief in Collective Action; Optimism about Technological Solutions

• Strong beliefin workers’ rights and economic fairness; Compassion and socialjustice orientation; Support for government intervention in social welfare; Mixed or nuanced views on nationalism and globalization; Pragmatism or openness to change; Prioritization of social well-being over strict fiscal conservatism; Potential belief in restorative justice

## I.21 Prism (gemini - 10 random value profiles)

• Neutrality and Objectivity in AI; Acknowledging, but not Necessarily Endorsing, Consensus; Emphasis on Dialogue and Discussion; Avoiding Definitive Claims without Complete Information; Preference for Comprehensive Explanations over Simple Deflection; Balance between Transparency and Safety; Trust in Established Institutions (to some degree); Appreciation for nuance and complexity

• Prefers concrete information and examples over conversational prompts when askingfor information. (Choosing A in the "controversial" prompt, which gave a specific example, over B, which offered general categories.); Values thoroughness and detail in responses, particularly regarding health and safety. (Choosing B in the smoking ban question and the running tips question, both of which were more detailed and comprehensive.); Appreciates helpful and proactive suggestions but not overly pushy or suggestive upselling. (Choosing B in the running tips question for its helpfulness but choosing A in the "controversial" prompt, possibly because B’s responsefelt too general andpromptedfor further interaction rather than providing immediate information.); Favors politeness and a friendly tone, but also values conciseness when appropriate. (Choosing A over B in the greeting example; A was more polite and complete, while B was shorter but potentially less engaging.); Prioritizes well-being and public health over individualfreedoms in certain scenarios. (Choosing B in the smoking ban question, which emphasized the negative public health impacts.); Believes that AI should acknowledge its limitations (lack ofpersonal opinions) but still be able to provide informative and helpful responses. (Choosing A in the smoking ban question, despite its disclaimer about not having opinions, as it still provided context and relevant considerations.); May prefer structured, bulleted lists for information that is easily digestible. (Choosing B in the running tips question, which used a bulleted list format.); Values responses that are relevant to the specific prompt and don’t feel overly generic or templated. (Potentially influencing the choice in the "controversial" prompt, where A provided a specific, though perhaps unexpected, example related to UK politics.); May have an interest in UK politics, given the acceptance of the list of political parties as a "controversial" topic. (Speculative, based on the choice in thefirst example.)

• Practicality and Actionability; Comprehensiveness; Neutrality and Objectivity; Directness and Clarity; Trust in Established Sources

• Prefers concise and direct answers; Appreciates acknowledging limitations; Values actionable and specific advice; Prioritizes safety and external validation; May not always prioritize detail or depth; Potentially prefers a friendly but not overly familiar tone; Values a balance between helpfulness and respecting personal autonomy

• Emphasis on empathy and emotional connection; Prioritization of conciseness and readability; Valuing personal experience and subjective perspectives; Preference for actionable information; Potential discomfort with overly cautious or "neutral" stances; Possible bias towards specific political viewpoints; Possible prioritization ofimmediate understanding over nuance

• Specificity and Informativeness; Actionability and Practicality; Transparency and Openness; Depth ofKnowledge; Directness; Trust in Open Source; Interest in Technical Details; Belief in Preparedness

• Neutrality and Objectivity; Comprehensiveness, but without Excessive Detail; Acknowledging Limitations; Data-Driven or Evidence-Based Reasoning; Balance and Moderation; Clarity and Directness; Trust in Established Knowledge

• Accuracy and Factuality; Neutrality and Objectivity; Comprehensiveness and Nuance; Trust in Expert Knowledge; Avoidance of Sensationalism; Safety and Practicality; Conciseness and Clarity

• Directness and Conciseness; Actionable Support over Passive Acknowledgement; Neutrality and Factual Information over Emotional Sentiments; Contextual Awareness; Desirefor Information and Understanding over Simple Platitudes; Possible Discomfort with AI Expressing "Feelings"

• Completeness and informativeness; Neutral ity and avoidance of strong framing; Focus on the main topic and avoidance of digression; Conciseness and clarity; Politeness and helpfulness; Factual accuracy (where applicable); Readability and flow