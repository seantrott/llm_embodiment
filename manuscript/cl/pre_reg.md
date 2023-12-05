# Registration Metadata

## TODO

- [ ] commit to VLMs for Q4


## Questions

- Which models for Q4? Hard to match on everything else. Just stick to exploratory?
- issue w/ using softmax p for q3: implicit comparison not available to human? Or is it in the aggregate?
- Random effects on human expt?
- Do Other MLLMs for Q3?
- Exploratory analyses

This metadata applies only to the registration you are creating, and will not be applied to your project.

## Title *

Do MLLMs show analogous effects to embodied simulation in humans?

## Description *

Embodied theories of human language comprehension argue that comprehenders understand language---in part---by simulating the embodied experiences that languages describe (Bergen, 2012). An common paradigm used to provide evidence for this theory uses a sentence-picture verification task, where the image either matches or does not match a visual feature that is implied (but not explicitly stated) by the sentence (Stanfield & Zwaan, 2001; Zwaan & Pecher, 2012). Participants' verification that an image shows the same object as the one described in the sentence is facilitated when implied visual features are matched, even though this information is irrelevant to the task. One interpretation of these results is that participants are automatically simulating visual features of objects described in text, and so it is easier to recognise a representation of the object that matches these simulated features than one which does not.

There are ongoing theoretical debates about how to interpret these kinds of results, including whether they imply that sensorimotor simulation plays a causal role in language comprehension or whether it is epiphenomenal (Mahon & Caramazza, 2008); and what level of integration between linguistic and sensorimotor processing the results imply (Meteyard et al., 2012). 

We address both of these debates using Multilingual Large Language Models (MLLMs). MLLMs learn relationships between distributional features of language and other modalities through being presented with linguistic and modal input, and learning to predict which features of one modality co-occur with features of the other [VisualBERT, CLIP]. MLLMs therefore constitute an explicit computational model of how input from multiple modalities can be integrated. We ask four interrelated questions about whether MLLMs show the same kinds of effects that human comprehenders do on sentence-picture verification tasks:

1) Are MLLMs sensitive to whether or not sensorimotor features of objects implied by sentences are explicitly present in modal representations?

	Evidence the MLLMs display this effect would suggest that multimodal distributional information is sufficient to generate the sensitivity to relationships between linguistic and sensorimotor representations which has been taken as evidence of embodied simulation in humans. This could either be interpreted as a deflationary account of these experiments (i.e. they cannot be evidence for embodied simulation if MLLMs show this effect) or as evidence that MLLMs provide a potential mechanism for embodied simulation in humans. Note that humans and MLLMs could produce the same effect in distinct ways, but the experiments themselves would not show that a mechanism beyond MLLM-style associative learning is neccessary.

2) Are the LLM text-encoders of MLLMs sensitive to matches between implicit and explicit sensorimotor features in a text-based adaptation of the sentence-picture verification task?

	If MLLMs *do* show evidence of the SPV match effect, there are different mechanisms by which they might do this. It could be that purely linguistic information, learned by the MLLM's text encoder, already contains sufficient information to be sensitive to implied sensorimotor features. Alternatively, this sensitivity might emerge from learning a shared embedding space between linguistic and sensorimotor features. We test this using a text-based adaptation of the task, where we use explicit linguistic descriptions of the implicit sensorimotor features, and compare representations from the text-encoder to the shared embedding space.

3) To what extent do MLLMs representations of task stimuli explain away the effects of implied sensorimotor feature match in humans?

	Evidence that MLLMs exhibit higher probabilities for matching vs non-matching stimuli would suggest that they are *sensitive* to cross-modal implicit feature matches. However, we are also interested in testing the stronger claim that they are *sufficiently sensitive* to explain human behavior at the task. This would provide stronger evidence that MLLMs provide a plausible mechanistic account of embodied simulation in human comprehenders. We test this using nested model comparisons: asking if condition (match vs non-match) continues to explain variance in human responses when MLLM predictions have already been accounted for.

4) Do architectures with closer integration between modalities show stronger & more humanlike effects?

	There are a variety of different MLLM architectures, which implement different levels of integration between language and other modalities. Dual Encoder models (Radford et al., 2021) comprise distinct modality-specific encoders that process inputs in parallel (without any interaction) and learn to project modality-specific representations to a shared embedding space. Fusion models (Li et al. 2019), in contrast, integrate modalities more tightly, by using a partially shared mechanism for encoding, or using cross-attention to allow encoding of one modality to be influenced by the other. Finally end-to-end architectures ([GPT-4?]) use a single encoding mechanism and a single stream of tokens that represent multimodal input, meaning multimodal processing is potentially as integrated as unimodal processing (depending on how the model learns to represent and direct attention across multi-modal inputs). 

	A priori, we expect that more integrated models will do better at a task that requires sensitivity to nuanced relationships between multimodal inputs. We test this using an interaction between model-type and match-effect (are matches significantly more probable for more integrated model types).

	Secondly, these different architectures provide loose analogies for proposed mechanistic models of embodied simulation in humans (Metayard, 2012). Dual Encoder theories can be roughly aligned with Secondary Embodiment theories (Mahon & Caramazza, 2008; Patterson et al., 2007), where modality-specific inputs are processed independently and used to inform a higher-level non-modality-specific representation. Fusion models loosely operationalize weak embodiment theories (Barsalou, 1999; Pulvermuller, 1999), where processing input from one modality is partly dependent on one's representations of another. End-to-End models have the potential to implement the strongest kind of multimodal interaction (Strong embodiment: Gallese & Lakoff, 2005), where linguistic inputs are processed using exactly the same neural resources as sensorimotor inputs. However, it is an empirical question whether models actually learn to do this. We test the plausibility of MLLMs as operationalizations of the mechanism of embodied simulation in humans by asking what proportion of the variance in human behavioral data they can account for.

Finally, we hope that this work will help to address a growing debate about the epistemological status of MLLMs themselves. An important criticism of pure-text LLMs is that their representations are not grounded (Harnad, 1990; Bender & Koller, 2021): LLMs learn relationships with linguistic form, but not how to relate language to the external world. MLLMs provide a potential solution to this problem by learning to integrate linguistic input with sensory input from other modalities. Indeed, embodied simulation theories argue that this is a crucial part of how human comprehenders ground language (Barsalou, 1999; Zwaan, 2003). Evidence that MLLMs show the same kinds of sensitivity and integration between modalities that has been used as evidence for embodied simulation in humans would suggest that MLLM representations are grounded in a partially similar way to human comprehenders.

On the other hand, some theorists have recently argued that even sensorimotor grounding would be insufficient to solve the grounding problem for MLLMs (Mollo & Miliere, 2023). On these accounts, MLLMs would need to connect their output to real events in the world (e.g. through having a body, goals, short feedback loops, & skin-in-the-game) [read recent Thompson on this.].

# TODO
- recognize that we did this before (COLING)
- Mention that we did this before, and this pre-reg is for new data
- Mention that we'll use Z&P for human data
- Commit to models
- Mention 5 modalities
- shift notes from below up here.


References
----------

Bergen, B. K. (2012). Louder than words: The new science of how the mind makes meaning. Basic Books.

Li, L. H., Yatskar, M., Yin, D., Hsieh, C.-J., & Chang, K.-W. (2019). VisualBERT: A Simple and Performant Baseline for Vision and Language (arXiv:1908.03557). arXiv. http://arxiv.org/abs/1908.03557

Mahon, B. Z., & Caramazza, A. (2008). A critical look at the embodied cognition hypothesis and a new proposal for grounding conceptual content. Journal of physiology-Paris, 102(1-3), 59-70.

Meteyard, L., Cuadrado, S. R., Bahrami, B., & Vigliocco, G. (2012). Coming of age: A review of embodiment and the neuroscience of semantics. Cortex, 48(7), 788-804.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020). arXiv. http://arxiv.org/abs/2103.00020

Stanfield, R. A., & Zwaan, R. A. (2001). The effect of implied orientation derived from verbal context on picture recognition. Psychological science, 12(2), 153-156.

Zwaan, R. A., & Pecher, D. (2012). Revisiting mental simulation in language comprehension: Six replication attempts. PloS one, 7(12), e51382.
Contributors


# Study Information

## Hypotheses*

*List specific, concise, and testable hypotheses. Please state if the hypotheses are directional or non-directional. If directional, state the direction. A predicted effect is also appropriate here. If a specific interaction or moderation is important to your research, you can list that as a separate hypothesis.*

**Note:** We plan to test 5 different sensorimotor features (shape, color, size, orientation in vision, and volume in audition). We treat each hypothesis within each feature as separate, because there are independent reasons why models may be able to learn some of these modalities and not others. We therefore a) do not control for multiple comparisons across these 5 modalities, and b) do not take partial evidence that MLLMs show effects on *some* modalities as evidence that MLLMs perform embodied simulation in general.

Secondly, we plan to assess multiple MLLM models for Q4. In order to avoid controlling for multiple comparisons on Qs 1-3 (as different MLLMs would essentially be tests of the same hypotheses here), we make predictions for only 1 MLLM for Q3: ImageBind ([ref]). We select ImageBind partly for convenience (it encodes both visual and audio inputs, allowing us to use a single model for all 5 features), and because it performed well in prior pilot work.

Here p(match) and p(mismatch) refer to the softmax probability of the dot product of vectors representing a sentence that implies a sensorimotor feature and another stimulus (e.g. an image, sound, or sentence) which either does or does not contain the feature.

1) Are MLLMs sensitive to whether or not sensorimotor features of objects implied by sentences are explicitly present in modal representations?

	H1A: MLLMs are sensitive to sensorimotor features of objects that are implied in sentences.

		- Prediction: p(match) > p(mismatch) in ImageBind representations of SPV stimuli

2) Are the LLM text-encoders of MLLMs sensitive to matches between implicit and explicit sensorimotor features in a text-based adaptation of the sentence-picture verification task?

	H2A: Text-only LLMs are sensitive to matches between implicit and explicit textual descriptions of sensorimotor features

		- Prediction: BaseLLM from ImageBind (BERT-base) will show the effect p(match) > p(mismatch)

	H2B: Learning to project to shared embedding space drives sensitivity to sensorimotor features

		- Prediction: ImageBind shared space will show the effect p(match) > p(mismatch), but not the BaseLLM.
		- Prediction: An interaction between match:model (base BERT vs shared ImageBind) will show a postive interaction effect of shared:match.


3) To what extent do MLLMs representations of task stimuli explain away the effects of implied sensorimotor feature match in humans?

	H3A: Human comprehenders process SPV stimuli differently from MLLMs

		- Prediction: There will be a residual effect of match condition when controlling for the effect of (M)LLM predictions p(stimulus-pair) on human behavior (accuracy & rt)

4) Do architectures with closer integration between modalities show stronger & more humanlike effects?

	H4A: MLLM architectures with closer integration will show a stronger effect of match vs mismatch.

		- Prediction: End-to-end > Fusion > Dual Encoder

	H4B: MLLM architectures with closer integration will explain a higher % of human data.

		- Prediction: End-to-end > Fusion > Dual Encoder


## Design Plan 

### Study type*
*Please check one of the following statements:*

- Experiment - A researcher randomly assigns treatments to study subjects, this includes field or lab experiments. This is also known as an intervention experiment and includes randomized controlled trials. [X]
- Observational Study - Data is collected from study subjects that are not randomly assigned to a treatment. This includes surveys, “natural experiments,” and regression discontinuity designs.
- Meta-Analysis - A systematic review of published studies.
- Other

### Blinding*
*Blinding describes who is aware of the experimental manipulations within a study. Mark all that apply.*

- No blinding is involved in this study. [X]
- For studies that involve human subjects, they will not know the treatment group to which they have been assigned.
- Personnel who interact directly with the study subjects (either human or non-human subjects) will not be aware of the assigned treatments. (Commonly known as “double blind”)
- Personnel who analyze the data collected from the study are not aware of the treatment applied to any given group.

*Is there any additional blinding in this study?*
- Blinding (Other)

### Study design*
*Describe your study design. The key is to be as detailed as is necessary given the specific parameters of the design. There may be some overlap between this question and the following questions. That is OK, as long as sufficient detail is given in one of the areas to provide all of the requested information. Examples include two-group, factorial, randomized block, and repeated measures. Is it a between (unpaired), within-subject (paired), or mixed design? Describe any counterbalancing required.*

Our materials are organized into quadruplets of two sentences and two other stimuli. The sentences both describe the same object explicitly, but each implies a different sensorimotor feature. For example "He hammered the nail into the [wall/floor]" implies that the nail is [horizontal/vertical]. The other stimuli explicitly represent these implied properties (e.g. an image of a [horizontal/vertical] nail).

We elicit responses from (M)LLMs by encoding quadruplets of stimuli and finding the dot product between each pair of implicit and explicit representations. We use the implicit sentence-wise softmax pro

### Randomization
*If you are doing a randomized study, state how you will randomize, and at what level. Typical randomization techniques include: simple, block, stratified, and adaptive covariate randomization. If randomization is required for the study, the method should be specified here, not simply the source of random numbers.*

## Sampling Plan 

### Existing Data*
*Preregistration is designed to make clear the distinction between confirmatory tests, specified prior to seeing the data, and exploratory analyses conducted after observing the data. Therefore, creating a research plan in which existing data will be used presents unique challenges. Please select the description that best describes your situation. See https://cos.io/prereg for more information.*

- Registration prior to creation of data 
- Registration prior to any human observation of the data 
- Registration prior to accessing the data 
- Registration prior to analysis of the data 
- Registration following analysis of the data 

*Explanation of existing data*
*If you indicate that you will be using some data that already exist in this study, please describe the steps you have taken to assure that you are unaware of any patterns or summary statistics in the data. This may include an explanation of how access to the data has been limited, who has observed the data, or how you have avoided observing any analysis of the specific data you will use in your study.*

### Data collection procedures*
*Please describe the process by which you will collect your data and your inclusion and exclusion criteria. If you are using human subjects, this should include the population from which you obtain subjects, recruitment efforts, payment for participation, how subjects will be selected for eligibility from the initial pool, and your study timeline. For studies that don't include human subjects, include information about how you will collect samples, duration of data gathering efforts, source or location of samples, or batch numbers you will use.*

### Sample size*
*Describe the sample size of your study. How many units will be analyzed in the study? This could be the number of people, birds, classrooms, plots, or countries included. If the units are not individuals, then describe the size requirements for each unit. If you are using a clustered or multilevel design, describe how many units are you collecting at each level of the analysis. This might be the number of samples or a range, minimum, or maximum.*

### Sample size rationale
*This could include a power analysis or an arbitrary constraint such as time, money, or personnel.*

### Stopping rule
*If your data collection procedures do not give you full control over your exact sample size, specify how you will decide when to terminate your data collection. If you are using sequential analysis, include your pre-specified thresholds.*

## Variables 

### Manipulated variables
*Precisely define all variables you plan to manipulate and the levels or treatment arms of each variable. This is not applicable to any observational study.*

### Measured variables*
*Precisely define each variable that you will measure. This will include outcome measures, as well as any measured predictors or covariates.*

### Indices
*If applicable, please define how measures will be combined into an index (or even a mean) and what measures will be used. Include either a formula or a precise description of the method. If you are using a more complicated statistical method to combine measures (e.g. a factor analysis), please note that here but describe the exact method in the analysis plan section.*

## Analysis Plan 

### Statistical models*
*What statistical model will you use to test each hypothesis? Please include the type of model (e.g. ANOVA, RMANOVA, MANOVA, multiple regression, SEM, etc) and the specification of the model. This includes each variable that will be included, all interactions, subgroup analyses, pairwise or complex contrasts, and any follow-up tests from omnibus tests. If you plan on using any positive controls, negative controls, or manipulation checks you may mention that here. Provide enough detail so that another person could run the same analysis with the information provided. Remember that in your final article any test not included here must be noted as exploratory and that you must report the results of all tests.*

### Transformations
*If you plan on transforming, centering, recoding the data, or requiring a coding scheme for categorical variables, please describe that process.*

### Inference criteria
*What criteria will you use to make inferences? Please describe the information you’ll use (e.g. specify the p-values, Bayes factors, specific model fit indices), as well as cut-off criterion, where appropriate. Will you be using one or two tailed tests for each of your analyses? If you are comparing multiple conditions or testing multiple hypotheses, will you account for this?*

### Data exclusion
*How will you determine which data points or samples if any to exclude from your analyses? How will outliers be handled? Will you use any awareness check?*

### Missing data
*How will you deal with incomplete or missing data?*

### Exploratory analysis
*If you plan to explore your data to look for unspecified differences or relationships, you may include those plans here. If you list an exploratory test here, you are not obligated to report its results. But if you do report it you are obligated to describe it as an exploratory result.*

## Other
*If there is any additional information that you feel needs to be included in your preregistration, please enter it here. Literature cited, disclosures of any related work such as replications or work that uses the same data, or other helpful context would be appropriate here.*
