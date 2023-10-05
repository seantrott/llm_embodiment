# -*- coding: utf-8 -*-
import os
import requests
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType


def setup_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = imagebind_model.imagebind_huge(pretrained=True)
    model.eval()
    model.to(device)
    return model, device


def analyze_data(model, device, csv_path, img_folder):
    df = pd.read_csv(csv_path)
    all_results = []

    for index, item in tqdm(df.iterrows(), total=len(df)):
        text_list = [item['sentence_h'].strip(), item['sentence_v'].strip()]
        image_paths = [os.path.join(img_folder, item['image_h']),
                        os.path.join(img_folder, item['image_v'])]

        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(text_list, device),
            ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
        }

        with torch.no_grad():
            embeddings = model(inputs)

        results = torch.softmax(
            embeddings[ModalityType.TEXT] @ embeddings[ModalityType.VISION].T, dim=-1)

        all_results.append({
            'match_h': results[0][0].item(),
            'mismatch_h': results[0][1].item(),
            'match_v': results[1][0].item(),
            'mismatch_v': results[1][1].item(),
            'object': item['object']
        })

    return pd.DataFrame(all_results)


def format_results(df):
    melted_df = pd.melt(df.drop(columns=['object']))
    melted_df['sentence'] = melted_df['variable'].apply(
        lambda x: x.split('_')[-1])
    melted_df['match'] = melted_df['variable'].apply(
        lambda x: x.split('_')[0])
    melted_df = melted_df.rename(
        columns={'value': 'probability'}).drop(columns=['variable'])
    melted_df = melted_df[["sentence", "match", "probability"]]
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


def main():
    model, device = setup_model()
    csv_path = "muraki2021/data/processed/items.csv"
    img_folder = "muraki2021/data/raw/images"
    img_save_path = "muraki2021/data/results/muraki2021.png"
    data_save_path = "muraki2021/data/results/muraki2021.csv"

    results_raw = analyze_data(model, device, csv_path, img_folder)
    results = format_results(results_raw)
    summary = results_summary(results)
    t, p = ttest(results)
    print(summary)
    print(f"t = {t}, p = {p}")
    plot_results(results, img_save_path)
    results.to_csv(data_save_path)


if __name__ == "__main__":
    main()
