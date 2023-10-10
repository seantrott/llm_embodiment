import os

import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import clip

def setup_model(model_name):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    if model_name == "imagebind":
        model = imagebind_model.imagebind_huge(pretrained=True)
        preprocess = None
    elif model_name == "clip":
        model, preprocess = clip.load("ViT-B/32", device=device)
    else:
        raise ValueError("Model must be either 'clip' or 'imagebind'")
    
    model.eval()
    model.to(device)
    return model, preprocess, device


def analyze_data(model, preprocess, device, csv_path, img_folder):
    df = pd.read_csv(csv_path)
    all_results = []

    for index, item in tqdm(df.iterrows(), total=len(df)):
            
        text_list = [
                        item['sentence_a'].strip(), item['sentence_b'].strip()]
        image_paths = [os.path.join(img_folder, item['image_a']),
                       os.path.join(img_folder, item['image_b'])]
        
        if isinstance(model, clip.model.CLIP):

            text_inputs = clip.tokenize(text_list).to(device)
            image_inputs = [preprocess(Image.open(path)).unsqueeze(0).to(device) for path in image_paths]
            with torch.no_grad():
                text_features = model.encode_text(text_inputs)
                image_features = [model.encode_image(img_input) for img_input in image_inputs]
                # Calculate the similarity
                results = torch.softmax(
                    text_features @ torch.stack(image_features).squeeze().T, dim=-1)

        elif isinstance(model, imagebind_model.ImageBindModel):

            inputs = {
                ModalityType.TEXT: data.load_and_transform_text(text_list, device),
                ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
            }

            with torch.no_grad():
                embeddings = model(inputs)

            results = torch.softmax(
                embeddings[ModalityType.TEXT] @ embeddings[ModalityType.VISION].T, dim=-1)
            
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
