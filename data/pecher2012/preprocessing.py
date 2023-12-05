

import pandas as pd
import numpy as np
import os
from PIL import Image
import re
from fuzzywuzzy import fuzz


def convert_pic_to_jpg(pic_path):
    img = Image.open(pic_path)
    img.save(pic_path.replace('.pic', '.jpg'))


def convert_all_pics_to_bmps(dir_path):
    for file in os.listdir(dir_path):
        if file.endswith('.pic'):
            convert_pic_to_bmp(os.path.join(dir_path, file))


dir_path = "pecher2012/Orientation"
convert_all_pics_to_bmps(dir_path)

pic_path = "pecher2012/Orientation/squirl.pic"
img = Image.open(pic_path)

paxman_fnames = os.listdir(
    "orientation/Sentence Picture Verification Task Stimuli")
paxman_fnames = [fname.replace(".bmp", "") for fname in paxman_fnames]


df = pd.read_csv("pecher2012/data/raw/items.csv")
# change bmp to jpg in match and mismatch cols
df["match"] = df["match"].apply(lambda x: x.replace(".bmp", ".jpg"))
df["mismatch"] = df["mismatch"].apply(lambda x: x.replace(".bmp", ".jpg"))
item_names = set(list(df["match"]) + list(df["mismatch"]))
len(item_names)

fnames = os.listdir("pecher2012/data/raw/images")
len(fnames)

# Check all item_names are in fnames
for fname in item_names:
    if fname not in fnames:
        print(fname)


# delete files in images that are not in item_names
for fname in fnames:
    if fname not in item_names:
        os.remove(os.path.join("pecher2012/data/raw/images", fname))


"""
Preprocess data
"""

def extract_condition_e1(colname):
    match, condition, obj = re.split(" ", colname)
    return match, condition, obj

def extract_condition_e2(colname):
    match, obj = re.split(" ", colname)
    if "_" in obj:
        obj, condition = obj.split("_")
    else:
        condition = "canonical"
    return match, condition, obj


def melt_table(table, n_trials=24, extract_fn=extract_condition_e1):

    match_cols = [c for c in table.columns if re.search(
        r'match ', str(c), re.IGNORECASE)]
    match_cols = match_cols[:n_trials]
    assert len(match_cols) == n_trials

    # Get index of first col that matches each match col
    match_col_indices = [np.where(table.columns == c)[0][0] for c in match_cols]
    rt_col_indices = [c + 1 for c in match_col_indices]

    # Create mini tables
    mini_tables = []
    for i, c in enumerate(match_cols):
        mini_table = table.iloc[:, [match_col_indices[i],
                                 rt_col_indices[i]]]
        mini_table.columns = ['response', 'rt']
        mini_table.reset_index(inplace=True, drop=True)
        
        # parse condition into match, condition, object
        match, condition, obj = extract_fn(c)
        
        mini_table = mini_table.assign(
            ppt_idx=mini_table.index + 1,
            match=match,
            condition_a=condition,
            object=obj,
            accuracy=mini_table.apply(
                lambda row: 1 if row['response'] == 2 else 0, axis=1)
        )
        mini_tables.append(mini_table)

    # Concat mini tables
    df_concat = pd.concat(mini_tables)

    return df_concat

def preprocess_zp_data(df):

    # Remove rts < 300 and > 3000
    df = df[(df['rt'] > 0.3) & (df['rt'] < 3)]

    rt_stats = df.groupby('ppt_id')['rt'].agg(['mean', 'std']).rename(columns={'mean': 'rt_mean', 'std': 'rt_std'})

    # Merge these stats back into the original DataFrame
    df = df.merge(rt_stats, on='ppt_id', how='left')

    # Filter out rows where RT is more than 2 SDs from the participant mean
    df = df[(df['rt'] - df['rt_mean']).abs() <= 2 * df['rt_std']]

    # Optionally, drop the extra columns if not needed
    df = df.drop(columns=['rt_mean', 'rt_std'])
    
    return df

def format_zp_data(filepath, preprocess=False, sheet_name="median & acc",
                   extract_fn=extract_condition_e1):
    # Read the specific sheet
    df = pd.read_excel(filepath, sheet_name=sheet_name)

    # Find rows where col0 contains match (ignorecase)
    boundaries = df[df.iloc[:, 0].str.contains(
        'match', case=False, na=False)].index.to_list()
    table_indices = zip(boundaries, boundaries[1:] + [None])

    # Find excluded col (using common value 'eq list(s)'. Often it's unnamed)
    excluded_col = np.where(df.apply(lambda x: x.dtype == object and x.str.contains(
        "eq list(s)?", na=False, regex=True)).any(axis=0))[0][0]

    # find L1 col
    l1_col = np.where(df.eq("L1").any(axis=0))[0][0]

    # Split and save each table as CSV
    tables = []
    for i, (start, end) in enumerate(table_indices):
        table = df[start:end]

        # Set header to first row
        table.columns = table.iloc[0]
        table = table.drop(table.index[0])

        # Get the exclusion data
        table['excluded'] = table.iloc[:, excluded_col]
        # Remove rows where excluded is not nan or ""
        table = table[-table['excluded'].apply(
            lambda x: isinstance(x, str) and len(x.strip()) > 0)]

        # Drop excluded column
        table = table.drop(columns=['excluded'])

        # Remove rows where col0 is not numeric or is nan
        table = table[table.iloc[:, 0].apply(
            lambda x: isinstance(x, (int, float)) and not np.isnan(x))]
        
        # remove rows where fuzzy match of L1 to english < 50
        def en_fuzz_match(x):
            return fuzz.ratio(x.lower(), 'english') > 70 or x.lower() in ["en", "eng"]
        
        # print langs that will get excluded from fuzzy match
        langs = table.iloc[:, l1_col].unique()
        langs = [lang for lang in langs if not en_fuzz_match(lang)]
        print(f"Excluding {langs} from fuzzy match")
        # apply fuzzy match
        table = table[table.iloc[:, l1_col].apply(en_fuzz_match)]

        # Melt table
        table = melt_table(table, extract_fn=extract_fn)
        table["list"] = i + 1
        # set ppt_id to list_ppt_idx
        table["ppt_id"] = table["list"].astype(str) + "_" + table["ppt_idx"].astype(str)
        tables.append(table)
    
    # Concatenate tables
    df_concat = pd.concat(tables)
    df_concat.reset_index(inplace=True)

    # Confirm 24 trials per ppt
    assert (df_concat.groupby('ppt_id').count()["object"] == 24).all()

    # Preprocess data
    if preprocess:
        df_concat = preprocess_zp_data(df_concat)

    # Write to CSV
    df_concat.to_csv(filepath.replace('.xlsx', '.csv'))


filepath = "data/pecher2012/human_data/Experiment_1a.xlsx"
format_zp_data('data/pecher2012/human_data/Experiment_1a.xlsx')

filepath = "data/pecher2012/human_data/Experiment_1b.xlsx"
format_zp_data('data/pecher2012/human_data/Experiment_1b.xlsx')

filepath = "data/pecher2012/human_data/Experiment_2a.xlsx"
format_zp_data('data/pecher2012/human_data/Experiment_2a.xlsx',
               sheet_name="rt correct median", extract_fn=extract_condition_e2)

filepath = "data/pecher2012/human_data/Experiment_2b.xlsx"
format_zp_data('data/pecher2012/human_data/Experiment_2b.xlsx',
               sheet_name="rt correct median", extract_fn=extract_condition_e2)

