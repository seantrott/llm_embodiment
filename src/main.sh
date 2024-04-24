#!/bin/bash

# Arrays of datasets and models
# DATASETS=("pecher2006" "muraki2021" "connell2007" "winter2012/e1" "winter2012/e2")
DATASETS=("pecher2006")
# MODELS=("imagebind" "ViT-B-32" "ViT-L-14-336" "ViT-H-14" "gpt2-large")
# MODELS=("ViT-B-32" "ViT-L-14-336" "ViT-H-14" "bridgetower" "vilt")
MODELS=("gpt2-large")

# Iterate over each dataset and model combination
for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "Running analysis with --dataset: $dataset --model: $model --metric"
        # python3 src/main.py --dataset "$dataset" --model "$model" --metric "cosine"  --columns sentence media --modalities text vision
        # python3 src/main.py --dataset "$dataset" --model "$model" --metric "cosine"  --columns explicit media --modalities text vision
        # python3 src/main.py --dataset "$dataset" --model "$model" --metric "cosine"  --columns sentence media --modalities text audio
        # python3 src/main.py --dataset "$dataset" --model "$model" --metric "cosine"  --columns explicit media --modalities text audio
        python3 src/main.py --dataset "$dataset" --model "$model" --metric "cosine" --columns sentence explicit --modalities text text
        #python3 src/main.py --dataset "$dataset" --model "$model" --metric "softmax"  --columns sentence media --modalities text vision
        echo "----------------------------------------------------------"
    done
done