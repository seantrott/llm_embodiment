# -*- coding: utf-8 -*-
"""imagebind_test.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10rIMM7AN-UbItziAynvjhUDzSvLql9qL

# Setup
"""

"""
git init
git remote add origin https://github.com/facebookresearch/ImageBind.git
git pull origin main
pip install .
pip install soundfile
"""
import requests
from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
from io import BytesIO

from PIL import Image

"""
Setup model
"""

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Instantiate model
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)


"""We see the same pattern, but with a much smaller effect. Img 1 more similar to 1, 2 to 2.

# Pecher et al. (2006)

**Note**: I uploaded the images all locally to get this started quickly, but if there's a better way to access them through GitHub that'd be ideal. Also nots ure I got everything applied correctly.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/pecher2006/items.csv")
df.head(2)

all_results = []
for index, item in df.iterrows():

    sentence_a = item['shape_a'].strip()
    sentence_b = item['shape_b'].strip()
    # print(sentence_a)
    # print(sentence_b)

    picture_a = item['picture shape a']
    picture_b = item['picture shape b']

    path1 = "data/pecher2006/Images/" + picture_a
    # img1 = Image.open(path1)
    # img1.show()

    path2 = "data/pecher2006/Images/" + picture_b
    # img2 = Image.open(path2)
    # img2.show()

    text_list = [sentence_a, sentence_b]
    image_paths = [path1, path2]

    # Load data
    inputs = {
        ModalityType.TEXT: data.load_and_transform_text(text_list, device),
        ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
    }

    with torch.no_grad():
        embeddings = model(inputs)

    results = torch.softmax(embeddings[ModalityType.TEXT] @ embeddings[ModalityType.VISION].T, dim=-1)

    all_results.append({
        'match_a': results[0][0].item(),
        'mismatch_a': results[0][1].item(),
        'match_b': results[1][0].item(),
        'mismatch_b': results[1][1].item(),
        'object': item['object']
    })

df = pd.DataFrame(all_results)
df.shape


melted_df = pd.melt(df.drop(columns=['object']))
melted_df['Sentence'] = melted_df['variable'].apply(lambda x: x.split('_')[-1])
melted_df.head(2)

melted_df['Match/Mismatch'] = melted_df['variable'].apply(lambda x: x.split('_')[0])
melted_df.head(2)

melted_df = melted_df.rename(columns={'value': 'Probability'})
# Drop the 'variable' column
melted_df = melted_df.drop(columns=['variable'])
# Reorder columns
melted_df = melted_df[['Match/Mismatch', 'Probability', 'Sentence']]
melted_df.head(2)

melted_df['Probability'].values

melted_df.groupby(['Sentence', 'Match/Mismatch']).mean()

# sns.stripplot(data = melted_df, x = "Match/Mismatch", y = "Probability", alpha = .3, hue = "Item")
sns.pointplot(data = melted_df, x = "Match/Mismatch", y = "Probability", hue = "Sentence")
plt.show()
# show plot


df["match_effect"] = ((df["match_a"] - df["mismatch_a"]) + (df["match_b"] - df["mismatch_b"])) / 2
df["img_effect"] = ((df["match_b"] - df["match_a"]) +
                    (df["mismatch_a"] - df["mismatch_b"])) / 2
df.sort_values(by="match_effect", ascending=False).head(10)
df.sort_values(by="match_effect").head(10)


