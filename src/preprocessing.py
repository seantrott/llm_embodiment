
import pandas as pd
import os
from PIL import Image, ImageOps
import re
import tiktoken


"""
Preprocessing
"""


def preprocess_image(input_folder, output_folder, target_size=(224, 224), padding_color=(255, 255, 255)):
    """
    Preprocesses images by resizing them to a target size and padding if necessary.
    Saves the processed images in the output folder.
    
    Args:
        input_folder (str): Path to the folder containing original images.
        output_folder (str): Path to the folder to save preprocessed images.
        target_size (int, optional): Target size for the image. Default is 224.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        # Add more formats if needed
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(input_folder, filename)

            with Image.open(image_path) as img:

                # Convert to RGB if necessary
                if img.mode == 'RGBA':
                    img = Image.alpha_composite(
                        Image.new("RGBA", img.size, padding_color), img)
                    img = img.convert("RGB")

                img_aspect = img.width / img.height
                target_aspect = target_size[0] / target_size[1]

                # Resize image
                if img_aspect > target_aspect:
                    new_width = target_size[0]
                    new_height = int(target_size[0] / img_aspect)
                else:
                    new_height = target_size[1]
                    new_width = int(target_size[1] * img_aspect)
                img = img.resize((new_width, new_height), Image.LANCZOS)

                # Pad image
                left_padding = (target_size[0] - new_width) // 2
                right_padding = target_size[0] - new_width - left_padding
                top_padding = (target_size[1] - new_height) // 2
                bottom_padding = target_size[1] - new_height - top_padding

                padded_img = Image.new(
                    "RGB", target_size, color=(255, 255, 255))
                padded_img.paste(img, (left_padding, top_padding))

                # Save the preprocessed image
                filename = filename.split('.')[0] + '.png'
                save_path = os.path.join(output_folder, filename)
                padded_img.save(save_path)

# preprocess_image("data/connell2007/images", "data/connell2007/images_processed")
# preprocess_image("data/pecher2006/images", "data/pecher2006/images_processed")
# preprocess_image("data/muraki2021/images", "data/muraki2021/images_processed")


def expand2square(pil_img, background_color=(255, 255, 255)):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


"""
Muraki
"""
df = pd.read_csv("muraki2021/data/raw/items.csv")
df
item_names = list(set(df["image"].tolist()))

fnames = os.listdir("muraki2021/data/raw/images")
len(fnames)

# Check all item_names are in fnames
for fname in item_names:
    if fname not in fnames:
        print(fname)

# delete files in images that are not in item_names
for fname in fnames:
    if fname not in item_names:
        os.remove(os.path.join("muraki2021/data/raw/images", fname))


# convert all pics to jpgs
def convert_pic_to_png(pic_path):
    img = Image.open(pic_path)
    img = img.convert('RGB')
    img.save(pic_path.replace('.bmp', '.png'))

# convert images in muraki2021/data/raw/images to jpg
dir_path = "muraki2021/data/raw/images"
for file in os.listdir(dir_path):
    if file.endswith('.bmp'):
        convert_pic_to_jpg(os.path.join(dir_path, file))

# delete bmps
for file in os.listdir(dir_path):
    if file.endswith('.bmp'):
        os.remove(os.path.join(dir_path, file))

# change file extensions in items.csv
df["image"] = df["image"].apply(lambda x: x.replace(".bmp", ".png"))


"""Reorganize"""
import re
df = df.sort_values("image")

df["object"] = df["image"].apply(lambda x: re.search("[a-z]+", x).group(0))
df["img_o"] = df["image"].apply(lambda x: re.search("[HV]", x).group(0))
df["match"] = df["match"].apply(lambda x: x.strip())

# H if img_o=H & match=Match
reverse = {"H": "V", "V": "H"}
df["sentence_o"] = df.apply(lambda x: x["img_o"] if x["match"] == "Match" else reverse[x["img_o"]], axis=1)

df_h = df[df["sentence_o"] == "H"][["sentence", "object"]]
df_h = df_h.drop_duplicates()
df_h = df_h.rename(columns={"sentence": "sentence_h"})

df_v = df[df["sentence_o"] == "V"][["sentence", "object"]]
df_v = df_v.drop_duplicates()
df_v = df_v.rename(columns={"sentence": "sentence_v"})

df_merge = pd.merge(df_h, df_v, on=["object"])
df_merge["image_h"] = df_merge["object"] + "H.jpg"
df_merge["image_v"] = df_merge["object"] + "V.jpg"

df_merge.to_csv("muraki2021/data/processed/items.csv", index=False)


"""
Winter & Bergen (2012)
"""

"""
E1
---
"""

# convert images in muraki2021/data/raw/images to jpg
dir_path = "data/winter2012/e1/images"
for file in os.listdir(dir_path):
    if file.endswith('.bmp'):
        convert_pic_to_png(os.path.join(dir_path, file))

# delete bmps
for file in os.listdir(dir_path):
    if file.endswith('.bmp'):
        os.remove(os.path.join(dir_path, file))

# pad small images to be the same size as larger paired image
# all images are paired e.g.
# data/winter2012/e1/images/1_EXPL_BIG.jpg data/winter2012/e1/images/1_EXPL_SMALL.jpg data/winter2012/e1/images/1_LM_BIG.jpg data/winter2012/e1/images/1_LM_SMALL.jpg


def pad_image_to_size(input_image_path, output_image_path, desired_size):
    # Open the image
    image = Image.open(input_image_path)

    # Calculate padding
    original_size = image.size
    delta_width = desired_size[0] - original_size[0]
    delta_height = desired_size[1] - original_size[1]
    padding = (delta_width // 2, delta_height // 2)

    # Add padding
    new_image = Image.new("RGB", desired_size, "white")
    new_image.paste(image, padding)

    # Save the padded image
    new_image.save(output_image_path, quality=100)

# get all image names
fnames = os.listdir("data/winter2012/e1/images")
fnames = [fname.replace(".png", "") for fname in fnames if fname.endswith(".png")]
fnames = [fname.replace("_BIG", "") for fname in fnames]
fnames = [fname.replace("_SMALL", "") for fname in fnames]
fnames = list(set(fnames))

# get all image sizes
sizes = {}
for fname in fnames:
    img = Image.open(os.path.join("data/winter2012/e1/images", fname + "_BIG.png"))
    sizes[fname] = img.size

# pad small images
for fname in fnames:
    big_size = sizes[fname]
    pad_image_to_size(
        input_image_path=os.path.join("data/winter2012/e1/images", fname + "_SMALL.png"),
        output_image_path=os.path.join("data/winter2012/e1/images", fname + "_SMALL.png"),
        desired_size=big_size)
    

"""
E2
---
"""

# read items sheet
df = pd.read_csv("data/winter2012/e2/human_data.csv")

# Get all unique item, near, far rows
df = df[["ITEM", "NEAR", "FAR"]]
df = df.drop_duplicates()

# drop rows where item startswith NO
df = df[~df["ITEM"].str.startswith("NO")]

# sort by item as a numeric
df["ITEM"] = df["ITEM"].apply(lambda x: int(x))
df = df.sort_values("ITEM")
df.reset_index(inplace=True, drop=True)

# save to items.csv
df.to_csv("data/winter2012/e2/items.csv", index=False)

"""
Descriptive stats
"""

# Get the mean and std of sentence lengths in each dataset

def get_mean_std_sentence_length(dataset, colname):
    df = pd.read_csv(f"data/{dataset}/items.csv")

    # Get no. tokens for gpt-3 text-davinci-002
    encoding = tiktoken.get_encoding("p50k_base")
    df["sentence_length"] = df[colname].apply(lambda x: len(encoding.encode(x)))

    return df["sentence_length"].mean(), df["sentence_length"].std()


get_mean_std_sentence_length("connell2007", "sentence_a")
get_mean_std_sentence_length("muraki2021", "sentence_a")
get_mean_std_sentence_length("pecher2006", "sentence_a")
