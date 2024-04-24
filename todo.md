- [X] Single file
- [X] consistent format of data
- [X] test
- [X] push

- [X] bash script to run all

- [X] test loading into R
- [X] single plot

- [X] write
    - [X] methods
    - [X] results

- [X] send to Sean

- [X] image preprocessing

- [X] try different models  
    - [X] CLIP models
    - [X] image_bind models


----

Z&P
- [X] E1
- [X] E2

WB
- [X] E1
    - [X] run model on images
    - [X] add as option to scripts
- [X] E2
    - [X] check stims
    - [X] create items sheet
        - [X] get sentences from human data
    - [X] organize sound files
    - [X] code to run
    - [X] add as option to scripts

- [X] Replicate human results
    - [X] tidy data
    - [X] OG analysis
    - [X] LME analysis

    - [X] E2

- [X] Email Diane

- [X] Text study
    - [X] Check we can access text encoder models
        - [X] using code from open_clip (e.g. self.transform)
        - [X] using raw HFTextEncoder (check create_model_and_transforms)
        - [X] tidy up
        - [X] text in pipeline that we could use for csv
        - [X] test
            - [X] reproduce coling
            - [X] audio pipeline
            - [X] text pipeline
    - [X] Generate stims for text study
    - [-] Test manipulation check stims


- [X] Architectures
    - [X] Decide on VisualBERT models
    - [X] Check we can access them all
    - [X] figure out how to get consistent metric from models.
    - [X] BenchmarkLMM
    - [X] write script
        - [X] use softmax
        - [X] model handlers


- [ ] MLLM analysis
    - [X] get ids for each item
    - [X] random intercepts by item
    - [X] try interaction for model sensitivity?

    - [ ] Format
        - [ ] Try other expts as pngs?


- [ ] prereg
    - [X] intro
        - [X] change refs to architetures in intro
        - [X] mark architecture q as exploratory
        - [X] Sean: show not necessary vs not show necessary
        - [X] Ben: sufficiently sensitive


    - [X] hypotheses
        - [X] cut thing about 5 modalitie
        - [X] softmax -> cosine (or don't mention here)

        - [X] H1: sensitivity
            - [X] H1A: aggregate
            - [X] H1B: WB1
            - [X] H1C: WB2
        - [X] H2: LLM sensitive
            - [X] H1A: aggregate
            - [X] H2B-F: individual features
        - [X] H3: Shared space more sensitive
            - [X] H3A: aggregate
            - [X] H3B-F: individual features
        - [X] H4: Image to explicit feature sensitivity
            - [X] H4A: aggregate
            - [X] H4B-F: individual features
        - [X] H5: Human baselines vs text
            - [X] H5A: aggregate
            - [X] H5B-E: individual features
        - [X] H6: Human baselines vs MLLM
            - [X] H6A: aggregate
            - [X] H6B-E: individual features
        - [X] H7: Architecture
            - [X] 
        
    - [X] study design
        - [X] elaborate on H2 & H4

    - Sampling
        - [X] update human exclusion criteria
        - [X] update sample size for W&B E2

    - Analysis plan
        - [X] rewrite to use cosine
        - [X] add random intercepts
        - [X] update to follow H structure
        - [X] random effects & convergence
        - [X] update inference criteria

    - Exploratory
        - [X] architectures
        - [X] softmax

    - [X] Email
        - [X] Questions
            - [X] Q4/H4?
            - [X] H7 Exploratory or not?
                - [X] stats for H7

- [X] Baselines
    - [X] write analysis code
        - [X] merge data

        - item (1), sent_condition (H), media_condition (H), match (match)
            - [X] items
            - [X] human data
            - [X] model data
            - [X] merge
        - [X] Orientation
        - [X] Shape
        - [X] size
        - [X] distance

    - [X] say we can't do E3

- [X] Architectures
    - [X] justify CLIP 32 choice
    - [X] Write Q7 analyses
    - [X] Update pre-reg:
        - [X] softmax
        - [X] AIC

- [X] Exploratory
    - [X] baseline w/ media_explicit
    - [X] Winter E1 LM vs blah

- [X] Stimuli
    - [X] fix stims

- [ ] Upload files
    - [X] stimuli
    - [X] replications
    - [X] analysis plan

    - [X] reread & send round

- Run
    - [ ] Audio target length setting?
        - [ ] https://github.com/facebookresearch/ImageBind/blob/main/imagebind/data.py
    - [ ] softmax analyses
    - [ ] hardcore manipulation check for sounds & size
        - -> loud/quiet. A "loud/quiet" handgun sound.
        - -> large/small. A [large/small] apple.
        - vs a neighbour.

    - [ ] item level effects


# Revisions

- [ ] Swap GPT-2 ImageBind TE
    - [ ] write code
    - [ ] run
- [ ] Extra models for E1
    - issues:
        - volume study
        - cosine measure from other models
    - [ ] run CLIPs

- [ ] emphasize pre-reg
- [ ] table for E3?
- [ ] figure for E4?

