# -*- coding: utf-8 -*-
import os
import argparse
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", module="torchvision.transforms._functional_video")
warnings.filterwarnings("ignore", module="torchvision.transforms._transforms_video")

# from src import utils
import utils


def main(args):

    # Parse arguments
    model_name = args.model
    dataset = args.dataset
    metric = args.metric
    columns = args.columns
    print(columns)
    modalities = args.modalities
    print(modalities)

    # Build save paths
    column_stub = "_".join(columns)
    result_stub = f"{dataset.replace('/', '_')}_{model_name}_{metric}_{column_stub}"
    img_save_path = f"results/{dataset}/{result_stub}.png"
    data_save_path = f"results/{dataset}/{result_stub}.csv"

    # Create folders for results
    os.makedirs(f"results/{dataset}", exist_ok=True)

    # Run analysis
    data_handler = utils.DataHandler(
        model_name=model_name,
        dataset=dataset,
        columns=columns,
        modalities=modalities,
        metric=metric
    )
    results_raw = data_handler.analyze_data()
    results = data_handler.format_results(results_raw, model_name, dataset)
    summary = data_handler.results_summary(results)
    t, p = data_handler.ttest(results)

    # Print and save results
    print(summary)
    print(f"t = {t}, p = {p}")
    data_handler.plot_results(results, img_save_path)
    results.to_csv(data_save_path, index=False)
    print(f"Results saved to {data_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process LLM datasets.')

    parser.add_argument('--dataset', type=str, required=True,
                        choices=[
                            'connell2007', 'muraki2021', 'pecher2006',
                            'winter2012/e1', 'winter2012/e2'
                        ],
                        help='Name of the dataset to process')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Model to use for analysis')
    
    parser.add_argument('--metric', type=str, default='cosine',
                        choices=['cosine', 'softmax', 'logits'],
                        help='Metric to use for analysis')
    
    parser.add_argument('--columns', type=str, nargs='+',
                        default=['sentence', 'media'],
                        help='Columns to use for analysis')
    
    parser.add_argument('--modalities', type=str, nargs='+',
                        default=['text', 'vision'],
                        help='Modalities to use for analysis')

    args = parser.parse_args()
    main(args)
