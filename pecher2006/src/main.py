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
        text_list = [item['shape_a'].strip(), item['shape_b'].strip()]
        image_paths = [os.path.join(img_folder, item['picture shape a']),
                       os.path.join(img_folder, item['picture shape b'])]

        inputs = {
            ModalityType.TEXT: data.load_and_transform_text(text_list, device),
            ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
        }

        with torch.no_grad():
            embeddings = model(inputs)

        results = torch.softmax(
            embeddings[ModalityType.TEXT] @ embeddings[ModalityType.VISION].T, dim=-1)
        all_results.append({
            'match_a': results[0][0].item(),
            'mismatch_a': results[0][1].item(),
            'match_b': results[1][0].item(),
            'mismatch_b': results[1][1].item(),
            'object': item['object']
        })

    return pd.DataFrame(all_results)


def plot_results(df, save_path=None):
    melted_df = pd.melt(df.drop(columns=['object']))
    melted_df['Sentence'] = melted_df['variable'].apply(
        lambda x: x.split('_')[-1])
    melted_df['Match/Mismatch'] = melted_df['variable'].apply(
        lambda x: x.split('_')[0])
    melted_df = melted_df.rename(
        columns={'value': 'Probability'}).drop(columns=['variable'])

    sns.pointplot(data=melted_df, x="Match/Mismatch",
                  y="Probability", hue="Sentence")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def main():
    model, device = setup_model()
    csv_path = "pecher2006/data/raw/items.csv"
    img_folder = "pecher2006/data/raw/images"
    img_save_path = "pecher2006/data/results/pecher2006.png"
    data_save_path = "pecher2006/data/results/pecher2006.csv"

    df = analyze_data(model, device, csv_path, img_folder)
    plot_results(df, img_save_path)
    df.to_csv(data_save_path)


if __name__ == "__main__":
    main()
