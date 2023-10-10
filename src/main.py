# -*- coding: utf-8 -*-
import os
import argparse

from utils import (setup_model, analyze_data, format_results,
                   results_summary, ttest, plot_results)


def main(args):

    # Parse arguments
    model_name = args.model
    dataset = args.dataset

    # Set up paths
    model, preprocess, device = setup_model(model_name)
    csv_path = f"data/{dataset}/items.csv"
    img_folder = f"data/{dataset}/images"
    img_save_path = f"results/{dataset}/{dataset}_{model_name}.png"
    data_save_path = f"results/{dataset}/{dataset}_{model_name}.csv"

    # Create folders for results
    os.makedirs(f"results/{dataset}", exist_ok=True)

    # Run analysis
    results_raw = analyze_data(model, preprocess, device, csv_path, img_folder)
    results = format_results(results_raw, model_name, dataset)
    summary = results_summary(results)

    # Print and save results
    t, p = ttest(results)
    print(summary)
    print(f"t = {t}, p = {p}")
    plot_results(results, img_save_path)
    results.to_csv(data_save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process LLM datasets.')
    parser.add_argument('--dataset', type=str, required=True, choices=['connell2007', 'muraki2021', 'pecher2006'],
                        help='Name of the dataset to process')
    parser.add_argument('--model', type=str, required=True, choices=['clip', 'imagebind'],
                        help='Model to use for analysis (clip or imagebind)')
    args = parser.parse_args()
    main(args)