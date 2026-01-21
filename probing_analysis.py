import os
import seaborn as sns
import numpy as np
import pandas as pd
from utils.constants import WEIGHT_DECAYS, SIZE_LIMITS
from utils.logger import get_logger
from tqdm import tqdm
import matplotlib.pyplot as plt

def summarize_best_test_auc(base_dir, weight_decays = WEIGHT_DECAYS):
    """
    For each LLM/model and dataset combination, load auc_table_val.csv, select the best 
    weight_decay/layer/token (row) based on validation AUC, and select that specific 
    configuration from auc_table_test.csv. Create a summary table with the best test AUC
    based on validation for all combinations.
    
    Returns one row per (model, dataset) with the best weight_decay, best_layer, best_token 
    for each size limit (15 rows total: 3 LLMs × 5 datasets).
    """
    size_limits = [10000] + SIZE_LIMITS
    last_summary_df = pd.DataFrame()
    
    for size in tqdm(size_limits, desc="Processing size limits"):
        summary = []
        for llm in tqdm(os.listdir(base_dir), desc=f"Processing LLMs (size={size})", leave=False):
            llm_path = os.path.join(base_dir, llm)
            if not os.path.isdir(llm_path):
                continue
            for model in os.listdir(llm_path):
                model_path = os.path.join(llm_path, model)
                if not os.path.isdir(model_path):
                    continue
                for dataset in tqdm(os.listdir(model_path), desc="Processing datasets", leave=False):
                    dataset_path = os.path.join(model_path, dataset)
                    if not os.path.isdir(dataset_path):
                        continue
                    
                    logger.info(f"Processing model: {model}, dataset: {dataset}, size limit: {size}")
                    # Look for auc_table_val.csv and auc_table_test.csv in the expected subdir
                    output_dir = os.path.join(dataset_path, "output", f"size_limit_{size}")
                    val_csv = os.path.join(output_dir, "auc_table_val.csv")
                    test_csv = os.path.join(output_dir, "auc_table_test.csv")
                    if not (os.path.exists(val_csv) and os.path.exists(test_csv)):
                        continue
                    
                    val_df = pd.read_csv(val_csv)
                    test_df = pd.read_csv(test_csv)
                    
                    # Filter for allowed weight decay values
                    val_df_filtered = val_df[val_df['weight_decay'].isin(weight_decays)]
                    if val_df_filtered.empty:
                        continue
                    
                    # Find the best row across all weight_decays (highest validation AUC)
                    best_val_idx = val_df_filtered['auc'].idxmax()
                    best_row = val_df_filtered.loc[best_val_idx]
                    
                    # Use the same wd/layer/token to select from test_df
                    match = (test_df['weight_decay'] == best_row['weight_decay']) & \
                            (test_df['layer'] == best_row['layer']) & \
                            (test_df['token'] == best_row['token'])
                    test_row = test_df[match]
                    
                    if not test_row.empty:
                        test_auc = test_row.iloc[0]['auc']
                    else:
                        test_auc = -1
                    
                    summary.append({
                        'LLM': llm,
                        'model': model,
                        'dataset': dataset,
                        'weight_decay': best_row['weight_decay'],
                        'best_layer': best_row['layer'],
                        'best_token': best_row['token'],
                        'val_auc': best_row['auc'],
                        'test_auc': test_auc
                    })
        
        # Save summary for this size limit
        if summary:
            summary_df = pd.DataFrame(summary)
            summary_df = summary_df.drop(columns=['LLM'])

            summary_df.to_csv(f"./probing_baselines/summary_table_SIZE_LIMIT_{size}.csv", index=False)
            logger.info(f"Saved summary table for size_limit={size} with {len(summary_df)} rows")
            last_summary_df = summary_df
        else:
            logger.warning(f"No data found for size_limit={size}")
    
    return last_summary_df


def plot_layer_token_heatmaps(base_dir):
    """
    For each LLM/model/dataset, plot a heatmap of best test AUCs for each (layer, token) combination,
    where the best test AUC is selected by choosing the weight_decay that gives the best val AUC for that (layer, token).
    Save each heatmap in the output_dir as a PDF, with a red square around the best (layer, token) combination.
    """

    output_dir = os.path.join(base_dir, "heatmaps")
    os.makedirs(output_dir, exist_ok=True)
    # Set larger font sizes globally
    plt.rcParams.update({
        'font.size': 35,
        # 'axes.titlesize': -1,
        'axes.labelsize': 35,
        'xtick.labelsize': 32,
        'ytick.labelsize': 32,
        'legend.fontsize': 25,
        'figure.titlesize': 18
    })
    
    for llm in os.listdir(base_dir):
        llm_path = os.path.join(base_dir, llm)
        if not os.path.isdir(llm_path):
            continue
        for model in os.listdir(llm_path):
            model_path = os.path.join(llm_path, model)
            if not os.path.isdir(model_path):
                continue
            for dataset in os.listdir(model_path):
                dataset_path = os.path.join(model_path, dataset)
                if not os.path.isdir(dataset_path):
                    continue
                output_subdir = os.path.join(dataset_path, "output", "size_limit_10000")
                val_csv = os.path.join(output_subdir, "auc_table_val.csv")
                test_csv = os.path.join(output_subdir, "auc_table_test.csv")
                if not (os.path.exists(val_csv) and os.path.exists(test_csv)):
                    continue
                val_df = pd.read_csv(val_csv)
                test_df = pd.read_csv(test_csv)
                # Get all unique layers and tokens
                layers = sorted(val_df['layer'].unique())
                # Use explicit token order instead of sorting
                token_order = [0, 1, 2, -3, -2, -1]
                # Format tokens with square brackets for display
                token_labels = [f"[{token}]" for token in token_order]
                
                heatmap = np.full((len(layers), len(token_order)), np.nan)
                heatmap_val = np.full((len(layers), len(token_order)), np.nan)
                for i, layer in enumerate(layers):
                    for j, token in enumerate(token_order):
                        # For this (layer, token), find the best weight_decay by val auc
                        subset = val_df[(val_df['layer'] == layer) & (val_df['token'] == token)]
                        if subset.empty:
                            continue
                        best_idx = subset['auc'].idxmax()
                        best_row = subset.loc[best_idx]
                        best_wd = best_row['weight_decay']
                        # Get the corresponding test auc
                        test_match = (test_df['layer'] == layer) & (test_df['token'] == token) & (test_df['weight_decay'] == best_wd)
                        test_row = test_df[test_match]
                        if not test_row.empty:
                            heatmap[i, j] = test_row.iloc[0]['auc']
                            heatmap_val[i, j] = best_row['auc']
                # Find the position of the best test AUC
                best_i, best_j = np.unravel_index(np.nanargmax(heatmap), heatmap.shape)
                # best_i, best_j = np.unravel_index(np.nanargmax(heatmap_val), heatmap_val.shape)
                best_auc = heatmap[best_i, best_j]
                best_auc_val = heatmap_val[best_i, best_j]
                # Plot heatmap with numeric annotations
                plt.figure(figsize=(max(10, len(token_order)//2), max(8, len(layers)//2)))
                ax = sns.heatmap(heatmap, annot=False, fmt=".2f", xticklabels=token_labels, yticklabels=layers, cmap="viridis", annot_kws={"size": 10})
                ax.set_xlabel("Tokens")
                ax.set_ylabel("Layers")
                layer_indices = [5, 10, 15, 20, 25]
                ax.set_yticks([i + 0.5 for i in layer_indices])
                
                ax.set_yticklabels(layer_indices)
                
                # Add red rectangle around the best (layer, token) combination
                rect = plt.Rectangle((best_j, best_i), 1, 1, fill=False, edgecolor='red', lw=3)
                ax.add_patch(rect)
                
                # Add legend for the red rectangle in the top left
                # Create a custom legend element for the red rectangle
                from matplotlib.patches import Patch
                from matplotlib.lines import Line2D
                legend_elements = [Line2D([0], [0], marker='s', color='w', markerfacecolor='w', 
                                         markeredgecolor='red', markersize=15, lw=0,
                                         label=f'Best layer/token: {layers[best_i]}/[{token_order[best_j]}]')]
                ax.legend(handles=legend_elements, loc='upper left', framealpha=0.7)
                
                # Map LLM names to acronyms
                llm_acronyms = {
                    'mistralai': 'Mis-7B',
                    'meta-llama': 'LlaMa-8B',
                    'Qwen': 'Qwen-7B'
                }
                llm_acronym = llm_acronyms[llm]

                dataset_acronyms = {
                    'hotpotqa': 'HotpotQA',
                    'triviaqa': 'TriviaQA',
                    'imdb': 'IMDB',
                    'hotpotqa_with_context': 'HotpotQA-Wc',
                    'movies' : 'Movies'
                }
                dataset_acronym = dataset_acronyms[dataset]
                # Add LLM acronym and dataset names on top of the heatmap
                plt.title(f"{llm_acronym} - {dataset_acronym}")
                
                plt.tight_layout()
                # Save plot as PNG instead of PDF
                fname = f"heatmap_{llm}_{model}_{dataset}.png"
                fpath = os.path.join(output_dir, fname)
                plt.savefig(fpath, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved heatmap to {fpath}")


if __name__ == "__main__":
    base_dir = "./probing_baselines"
    logger = get_logger(f"probing_analysis.log")
    logger.info(f"Base directory: {base_dir}")
    
    summary_df = summarize_best_test_auc(
        base_dir=base_dir,
        weight_decays=WEIGHT_DECAYS
    )
    
    plot_layer_token_heatmaps(
        base_dir=base_dir
    )



