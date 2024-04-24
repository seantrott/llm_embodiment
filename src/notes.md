Architectures
-------------

To my great surprise, running lots of different vision language models with different architectures turned out to be harder than I was expecting.

There are lots of differences between the models, and some of the older ones (e.g. VisualBERT)
use CNNs which don't come pre-packaged with huggingface. The different vision encoders also
introduce other sources of variance which makes matching the text encoder on BERT seem kind
of pointless.

I have picked 4 that are on HF, and I can get results for, that represent a bit of a spread in terms of architecture, but I think we'll have to pitch this as exploratory work, because it doesn't seem like a very rigorous test of the architectures themselves.

1. [CLIP](https://arxiv.org/abs/2103.00020)
    - Text-encoder: GPT2
    - Vision-encoder: ViT or CNN
    - Integration: Dual Encoder

This is an easy one which we already have running. We could use a few different CLIPs that we already have code for.

2. [BLIP](https://arxiv.org/pdf/2201.12086.pdf)
    - text-encoder: BERT
    - vision-encoder: ViT
    - integration: Fusion, Dual Stream

This is a salesforce model that is easy to run but doesn't seem to produce great results. Dual Stream Fusion means that the text and vision encoders are separate but they can use cross-attention to see each other during processing.

3.[BridgeTower](https://arxiv.org/pdf/2206.08657.pd)
    - text-encoder: RoBERTa
    - vision-encoder: ViT
    - integration: Fusion, Dual Encoder/Single Stream

BridgeTower seems to work a little better than ViT. It has 6 layers of independent processing and 6 layers of joing multi-modal processing, which makes it a nice step between dual encoder and single stream.

4. [ViLT](https://arxiv.org/pdf/2102.03334.pdf)
    - text-encoder: Linear Projection
    - vision-encoder: Linear Projection
    - integration: Fusion, Single Stream

This is quite a different model that uses very light preprocessing of text and image stimuli and a relatively large multimodal stack. They also have a nice taxonomy of architectures in the paper that positions the model in this way. It seems to work very well from my limited testing.

I'm planning to write these up as an exploratory hypothesis that we'll test, leaving space to try other models if we have time, and essentially pitching this as exploratory work where we can say something suggestive about model architectures without having to argue that we've tested this very rigorously.

We could also include GPT-4V in this exploratory analysis, and maybe flamingo.

# Cosine vs softmax, redux

After chatting about this with Ben on Monday we realised we should be using item_id as a random intercept. When we do this, we see the same sensitivity to features in cosine as we do in softmax, even when we add the intercept for softmax.

The graphs look less impressive, but I guess I'm now leaning toward using cosine as the main analysis, and softmax as an exploratory analysis.

Theory
------

On the other hand, some theorists have recently argued that even sensorimotor grounding would be insufficient to solve the grounding problem for MLLMs (Mollo & Miliere, 2023). On these accounts, MLLMs would need to connect their output to real events in the world (e.g. through having a body, goals, short feedback loops, & skin-in-the-game) [read recent Thompson on this.].