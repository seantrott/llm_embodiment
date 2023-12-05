from open_clip.transformer import text_global_pool
import os

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageOps, ImageDraw
from scipy.spatial.distance import cdist, cosine
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import clip
import open_clip

"""
Setup Models
"""

def setup_model(model_name):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    if model_name == "ViT-B-32":
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai')

    elif model_name == "ViT-L-14-336":
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai')

    elif model_name == "ViT-H-14":
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='laion2b_s32b_b79k')

    elif model_name == "ViT-g-14":
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='laion2b_s34b_b88k')

    elif model_name == "ViT-bigG-14":
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='laion2b_s39b_b160k')

    elif model_name == "ViT-L-14":
        model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai')
    
    elif model_name == "imagebind":
        model = imagebind_model.imagebind_huge(pretrained=True)
        preprocess = None

    else:
        raise ValueError("Model not implemented")
    
    if isinstance(model, open_clip.model.CLIP):
        tokenizer = open_clip.get_tokenizer(model_name)
    else:
        tokenizer = None
    
    model.eval()
    model.to(device)
    return model, preprocess, tokenizer, device




def analyze_data(model, preprocess, tokenizer, device, csv_path, img_folder,
                 use_cosine=False, modality="vision"):
    df = pd.read_csv(csv_path)
    all_results = []

    # index, item = next(df.iterrows())
    # item = df.iloc[18]
    for index, item in tqdm(df.iterrows(), total=len(df)):
            
        text_list = [
                        item['sentence_a'].strip(), item['sentence_b'].strip()]
        image_paths = [os.path.join(img_folder, item['image_a']),
                       os.path.join(img_folder, item['image_b'])]
        # image_paths = [path.replace(".jpg", ".png") for path in image_paths]
        
        if isinstance(model, open_clip.model.CLIP):

            image_inputs = [preprocess(Image.open(path)).unsqueeze(0).to(device) for path in image_paths]
            text_inputs = tokenizer(text_list)

            with torch.no_grad():
                text_features = model.encode_text(text_inputs)
                image_features = torch.stack([model.encode_image(img_input) for img_input in image_inputs]).squeeze()
                
                if use_cosine:
                    # Calculate the cosine similarity
                    results = 1 - cdist(
                        text_features.detach().numpy(),
                        image_features.detach().numpy(),
                        metric='cosine'
                    )
                else:
                    # Calculate the softmax probability
                    results = torch.softmax(
                        text_features @ image_features.T, dim=-1)


        elif isinstance(model, clip.model.CLIP):

            text_inputs = clip.tokenize(text_list).to(device)
            image_inputs = [preprocess(Image.open(path)).unsqueeze(0).to(device) for path in image_paths]
            with torch.no_grad():
                text_features = model.encode_text(text_inputs)
                image_features = [model.encode_image(img_input) for img_input in image_inputs]
                
                # Calculate the similarity
                if use_cosine:
                    # Calculate the cosine distance
                    results = 1 - cdist(
                        text_features.detach().numpy(),
                        torch.stack(image_features).squeeze().detach().numpy(),
                        metric='cosine'
                    )
                else:
                    # Calculate the softmax probability
                    results = torch.softmax(
                        text_features @ torch.stack(image_features).squeeze().T, dim=-1)

        elif isinstance(model, imagebind_model.ImageBindModel):

            # Transform raw data
            stimuli = {
                "image": data.load_and_transform_vision_data,
                "audio": data.load_and_transform_audio_data
            }[modality](image_paths, device)

            inputs = {
                ModalityType.TEXT: data.load_and_transform_text(text_list, device),
                modality: stimuli,
            }

            with torch.no_grad():
                embeddings = model(inputs)

                if use_cosine:
                    # Calculate the cosine distance
                    results = 1 - cdist(
                        embeddings[ModalityType.TEXT].detach().numpy(),
                        embeddings[modality].detach().numpy(),
                        metric='cosine'
                    )
                else:
                    # Calculate the softmax probability
                    results = torch.softmax(
                        embeddings[ModalityType.TEXT] @ embeddings[modality].T, dim=-1)
        else:
            raise ValueError("Model must be either 'clip' or 'imagebind'")

        all_results.append({
            'match_a': results[0][0].item(),
            'mismatch_a': results[0][1].item(),
            'match_b': results[1][1].item(),
            'mismatch_b': results[1][0].item(),
            'object': item['object']
        })

    return pd.DataFrame(all_results)


def format_results(df, model_name, dataset):
    melted_df = pd.melt(df.drop(columns=['object']))
    melted_df['sentence'] = melted_df['variable'].apply(
        lambda x: x.split('_')[-1])
    melted_df['match'] = melted_df['variable'].apply(
        lambda x: x.split('_')[0])
    melted_df = melted_df.rename(
        columns={'value': 'probability'}).drop(columns=['variable'])
    melted_df = melted_df[["sentence", "match", "probability"]]
    melted_df["model"] = model_name
    melted_df["dataset"] = dataset
    return melted_df


oc, _, oc_preprocess = open_clip.create_model_and_transforms(
    "ViT-H-14", pretrained='laion2b_s32b_b79k')
oc_tokenizer = open_clip.get_tokenizer("ViT-H-14")

oc_no_proj, _,  oc_preprocess = open_clip.create_model_and_transforms(
    "ViT-H-14", pretrained='laion2b_s32b_b79k')
oc_no_proj.text_projection = None

imagebind = imagebind_model.imagebind_huge(pretrained=True)

def oc_encode_text(model, text_list, tokenizer=oc_tokenizer):
    text_inputs = tokenizer(text_list)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    return text_features

def ib_encode_text(model, text_list, device="cpu"):
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    }
    with torch.no_grad():
        embeddings = model(inputs)
    return embeddings[ModalityType.TEXT]


text_list = ['There is a table in a old house',
             'The table appears very large.']

oc_embeddings = oc_encode_text(oc, text_list)
oc_no_proj_embeddings = oc_encode_text(oc_no_proj, text_list)
encoded_ib = ib_encode_text(imagebind, text_list)

oc_embeddings.shape
oc_no_proj_embeddings.shape

# cosine similarities
a, b = oc_embeddings.detach().numpy()
cosine(a, b)

a, b = oc_no_proj_embeddings.detach().numpy()
cosine(a, b)

a, b = encoded_ib.detach().numpy()
cosine(a, b)

def compare_text_pairs(model, text_list_a, text_list_b, encoding_fn):
    """
    Pairwise comparison of text pairs a1,b1, a2,b2 etc
    """
    a = encoding_fn(model, text_list_a)
    b = encoding_fn(model, text_list_b)
    results = 1 - cdist(
        a.detach().numpy(),
        b.detach().numpy(),
        metric='cosine'
    )
    return results

text_list_c = ["There was apple pie on the table",
             "There was apple pie on the plate."]

text_list_a = ["She placed the apple pie in a baking dish.",
               "She served a single piece of apple pie on a plate."]
text_list_b = ["A whole apple pie.",
               "A slice of apple pie."]

compare_text_pairs(oc_no_proj, text_list_a, text_list_b, oc_encode_text)
compare_text_pairs(oc, text_list_a, text_list_b, oc_encode_text)
compare_text_pairs(imagebind, text_list_a, text_list_b, ib_encode_text)






results = torch.softmax(
    embeddings[ModalityType.TEXT] @ embeddings[modality].T, dim=-1)



def results_summary(df):
    summary = df[["match", "probability"]].groupby(["match"]).mean()
    return summary


def ttest(df):
    from scipy.stats import ttest_ind
    match = df[df["match"] == "match"]["probability"]
    mismatch = df[df["match"] == "mismatch"]["probability"]
    t, p = ttest_ind(match, mismatch)
    return t, p


def plot_results(df, save_path=None):
    sns.pointplot(data=df, x="match",
                  y="probability", hue="sentence")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
