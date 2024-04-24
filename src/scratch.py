from src import utils
data_handler = utils.DataHandler("ViT-B-32", "connell2007")
results = data_handler.analyze_data()

"""
Test audio
"""

results

ib_model = utils.ImageBindHandler("imagebind")

text_list = ["a gunshot", "a baby crying"]
audio_list = ["data/winter2012/e2/images/CRITICAL_01_NEAR.wav",
              "data/winter2012/e2/images/CRITICAL_02_NEAR.wav"]

text_features = ib_model.encode_media(text_list, modality="text")
audio_features = ib_model.encode_media(audio_list, modality="audio")

utils.Util.calculate_cosine_similarity(text_features, audio_features)
utils.Util.calculate_softmax_probability(text_features, audio_features)


"""
Test text2text
"""


ib_model = utils.ImageBindHandler("imagebind")
oc = utils.OpenClipHandler("ViT-H-14")
oc_no_proj = utils.OpenClipHandler("ViT-H-14-no-proj")

obj = "fire extinguisher"
text_list_a = [f"A {obj} that is far away in the distance.",
               f"A {obj} that is very close, right in front of you."]
text_list_b = [f"A {obj} that looks relatively small.",
               f"A {obj} that looks relatively large."]

text_features_a = oc_no_proj.encode_media(text_list_a, modality="text")
text_features_b = oc_no_proj.encode_media(text_list_b, modality="text")
utils.Util.calculate_cosine_similarity(text_features_a, text_features_b)

obj = "blender"
text_list_a = [f"You can't hear over the sound of the {obj}.",
               f"You can barely hear the sound of the {obj}."]
text_list_b = [f"The sound of the {obj} is relatively loud.",
               f"The sound of the {obj} is relatively quiet."]
text_features_a = oc_no_proj.encode_media(text_list_a, modality="text")
text_features_b = oc_no_proj.encode_media(text_list_b, modality="text")
utils.Util.calculate_cosine_similarity(text_features_a, text_features_b)

obj = "apple pie"
text_list_a = [f"She took the intact {obj} out of the oven.",
               f"She cut a single piece of the {obj}."]
text_list_b = [f"A whole {obj}.",
               f"A slice of {obj}."]
text_features_a = oc_no_proj.encode_media(text_list_a, modality="text")
text_features_b = oc_no_proj.encode_media(text_list_b, modality="text")
utils.Util.calculate_cosine_similarity(text_features_a, text_features_b)

text_features_a = oc.encode_media(text_list_a, modality="text")
text_features_b = oc.encode_media(text_list_b, modality="text")

utils.Util.calculate_cosine_similarity(text_features_a, text_features_b)
utils.Util.calculate_softmax_probability(text_features_a, text_features_b)

text_features_a = ib_model.encode_media(text_list_a, modality="text")
text_features_b = ib_model.encode_media(text_list_b, modality="text")

utils.Util.calculate_cosine_similarity(text_features_a, text_features_b)
utils.Util.calculate_softmax_probability(text_features_a, text_features_b)

"""
ViLT
"""

from src.utils import ViltHandler, BridgetowerHandler, OpenClipHandler

image_paths = ["data/pecher2006/images_processed/apple.png",
               "data/pecher2006/images_processed/applepie.png"]
texts = ["An apple", "An apple pie"]
modalities = ["text", "vision"]

vilt_handler = ViltHandler("vilt")
vilt_handler.compare_pair(texts, image_paths, modalities)


bt_handler = BridgetowerHandler("bridgetower")
bt_handler.compare_pair(texts, image_paths, modalities)

b32 = OpenClipHandler("ViT-B-32")
b32.compare_pair(texts, image_paths, modalities)


# oc, _, oc_preprocess = open_clip.create_model_and_transforms(
#     "ViT-H-14", pretrained='laion2b_s32b_b79k')
# oc_tokenizer = open_clip.get_tokenizer("ViT-H-14")

# oc_no_proj, _,  oc_preprocess = open_clip.create_model_and_transforms(
#     "ViT-H-14", pretrained='laion2b_s32b_b79k')
# oc_no_proj.text_projection = None

# imagebind = imagebind_model.imagebind_huge(pretrained=True)

# def oc_encode_text(model, text_list, tokenizer=oc_tokenizer):
#     text_inputs = tokenizer(text_list)
#     with torch.no_grad():
#         text_features = model.encode_text(text_inputs)
#     return text_features

# def ib_encode_text(model, text_list, device="cpu"):
#     inputs = {
#         ModalityType.TEXT: data.load_and_transform_text(text_list, device),
#     }
#     with torch.no_grad():
#         embeddings = model(inputs)
#     return embeddings[ModalityType.TEXT]


# text_list = ['There is a table in a old house',
#              'The table appears very large.']

# oc_embeddings = oc_encode_text(oc, text_list)
# oc_no_proj_embeddings = oc_encode_text(oc_no_proj, text_list)
# encoded_ib = ib_encode_text(imagebind, text_list)

# oc_embeddings.shape
# oc_no_proj_embeddings.shape

# # cosine similarities
# a, b = oc_embeddings.detach().numpy()
# cosine(a, b)

# a, b = oc_no_proj_embeddings.detach().numpy()
# cosine(a, b)

# a, b = encoded_ib.detach().numpy()
# cosine(a, b)

# def compare_text_pairs(model, text_list_a, text_list_b, encoding_fn):
#     """
#     Pairwise comparison of text pairs a1,b1, a2,b2 etc
#     """
#     a = encoding_fn(model, text_list_a)
#     b = encoding_fn(model, text_list_b)
#     results = 1 - cdist(
#         a.detach().numpy(),
#         b.detach().numpy(),
#         metric='cosine'
#     )
#     return results

# text_list_c = ["There was apple pie on the table",
#              "There was apple pie on the plate."]

# text_list_a = ["She placed the apple pie in a baking dish.",
#                "She served a single piece of apple pie on a plate."]
# text_list_b = ["A whole apple pie.",
#                "A slice of apple pie."]

# compare_text_pairs(oc_no_proj, text_list_a, text_list_b, oc_encode_text)
# compare_text_pairs(oc, text_list_a, text_list_b, oc_encode_text)
# compare_text_pairs(imagebind, text_list_a, text_list_b, ib_encode_text)


# results = torch.softmax(
#     embeddings[ModalityType.TEXT] @ embeddings[modality].T, dim=-1)


"""
vilt_processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-coco")
vilt = ViltForImageAndTextRetrieval.from_pretrained("dandelin/vilt-b32-finetuned-coco")

def compare_pairs(model, processor, images, texts, return_loss=False):
    Compare pairs of images and texts using a model.
    # forward pass
    scores = []
    for text in texts:
        scores.append([])
        for i, image in enumerate(images):
            # prepare inputs
            encoding = processor(image, text, return_tensors="pt")
            if return_loss:
                encoding["return_loss"] = True

            outputs = model(**encoding)
            if hasattr(outputs, "loss") and outputs.loss is not None:
                scores[-1].append(outputs.loss.item())
            elif hasattr(outputs, "logits"):
                scores[-1].append(outputs.logits[0, :].item())
            elif hasattr(outputs, "logits_per_image"):
                scores[-1].append(outputs.logits_per_image[0, :].item())

    # softmax scores by text
    print(scores)
    scores = torch.softmax(torch.tensor(scores), dim=0).tolist()
    # cosine similarity between image embeddings and text embeddings
    return scores

scores = compare_pairs(vilt, vilt_processor, images, texts)
scores

"""
import torch
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2", output_hidden_states=True)

tokenizer.pad_token = tokenizer.eos_token
text_list = ["cold cold hard cash", "this is a dog"]

inputs = tokenizer(
    text_list, return_tensors='pt', padding=True)

with torch.no_grad():
    outputs = model(**inputs)
    hidden_states = outputs.hidden_states

layer = hidden_states[-1]
layer_embeddings = layer.mean(dim=1).squeeze().numpy()

cosine(layer_embeddings[0], layer_embeddings[1])

"""
CLAP
"""

from datasets import load_dataset
import torch
from transformers import AutoProcessor, ClapModel
from sklearn.metrics.pairwise import cosine_similarity

dataset = load_dataset("ashraq/esc50")
audio_sample = dataset["train"]["audio"][0]["array"]

model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")

input_text = ["Sound of a dog", "Sound of vaccum cleaner"]

inputs = processor(text=input_text, audios=audio_sample, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)
logits_per_audio = outputs.logits_per_audio  # this is the audio-text similarity score
probs = logits_per_audio.softmax(dim=-1)  # we can take the softmax to get the label probabilities

t1 = outputs.text_embeds[0].reshape(1, -1)
t2 = outputs.text_embeds[1].reshape(1, -1)

a1 = outputs.audio_embeds[0].reshape(1, -1)

cosine_similarity(t1, t2)
cosine_similarity(t1, a1)
cosine_similarity(t2, a1)


"""
My data
"""


import torch
import torchaudio
from transformers import AutoProcessor, ClapModel

audio_files = ["data/winter2012/e2/images/CRITICAL_01_NEAR.wav",
              "data/winter2012/e2/images/CRITICAL_02_NEAR.wav"]
audio_sample = [torchaudio.load(audio_file)[0].squeeze() for audio_file in audio_files]

model = ClapModel.from_pretrained("laion/clap-htsat-unfused")
processor = AutoProcessor.from_pretrained("laion/clap-htsat-unfused")

input_text = ["a gunshot", "a baby crying"]


inputs = processor(audios=audio_sample[0],
                   return_tensors="pt", padding=True)

with torch.no_grad():
    audio_outputs = model.get_audio_features(**inputs)

inputs = processor(text=input_text, return_tensors="pt", padding=True)

with torch.no_grad():
    text_features = model.get_text_features(**inputs)

t1 = text_features[0].reshape(1, -1)
t2 = text_features[1].reshape(1, -1)

a1 = audio_outputs[0].reshape(1, -1)
a2 = audio_outputs[1].reshape(1, -1)

cosine_similarity(t1, t2)
cosine_similarity(t1, a1)
cosine_similarity(t2, a1)


"""
