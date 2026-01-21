
import os
import pandas as pd
import wandb
import numpy as np
import re
import argparse


def get_results_15_15(link, probe_model):
    try:
        wandb.login()
    except wandb.errors.CommError as e:
        print("Error: Could not authenticate with wandb. Please ensure you are logged in.")
        raise e

    api = wandb.Api()
    match = re.search(r'wandb\.ai/([^/]+/[^/]+/sweeps/[^/?]+)', link)
    if not match:
        raise ValueError(f"Invalid wandb sweep link: {link}")
    sweep_path = match.group(1)

    sweep = api.sweep(sweep_path)
    
    search_params = {
    key: val
    for key, val in sweep.config['parameters'].items()
    if 'values' in val and len(val['values']) > 1 and key != 'probe_model' and key != 'seed'
    }
    



    runs = sweep.runs

    # Filter runs by probe model
    filtered_runs = [run for run in runs if run.config.get('probe_model') == probe_model]

    # Find the best run based on best_val_AUC
    best_run = max(filtered_runs, key=lambda run: run.summary.get('best_val_AUC', float('-inf')))
    

    # Find all runs with the same hyperparameters
    best_hyperparams = {k: v for k, v in best_run.config.items() if k != 'seed'}

    filtered_best_hyperparams = {k: v for k, v in best_hyperparams.items() if k in search_params}


    if filtered_best_hyperparams:
        matching_runs = [run for run in filtered_runs if all(run.config.get(k) == v for k, v in filtered_best_hyperparams.items())]
    else:
        matching_runs = filtered_runs


    # Determine the best epoch for each run
    best_epochs = {}

    for run in matching_runs:
        run_id = run.id  # or another unique identifier
        history = run.history(keys=['best_val_AUC'])
        
        best_epoch = None
        best_val_auc = float('-inf')
        
        for epoch, row in history.iterrows():
            if row['best_val_AUC'] > best_val_auc:
                best_val_auc = row['best_val_AUC']
                best_epoch = epoch

        best_epochs[run_id] = best_epoch

    # Scrape metrics for the best epoch of each run
    all_metrics = {}

    for run in matching_runs:
        run_id = run.id  # or other unique identifier
        history = run.history()  # get all keys
        best_epoch = best_epochs[run_id]  # get best epoch for this run

        for metric in history.columns:
            value = history.loc[best_epoch, metric]
            all_metrics.setdefault(metric, []).append(value)

    # Calculate mean and std of the metrics
    mean_metrics = {}
    std_metrics = {}

    for k, v in all_metrics.items():
        try:
            mean_metrics[k] = np.mean(v)
            std_metrics[k]  = np.std(v)
        except Exception as e:
            print(f"Skipping metric '{k}' -> {e}")
            continue


  # Save results to a CSV file in the "Results" folder
    os.makedirs("Results", exist_ok=True)
    df = pd.DataFrame({
        "metric": list(mean_metrics.keys()),
        "mean": [mean_metrics[k] for k in mean_metrics],
        "std": [std_metrics[k] for k in mean_metrics]
    })
    df.to_csv(f"Results/{probe_model}_15_15.csv", index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Scrape results from Wandb")
    parser.add_argument("--link", type=str, required=True, help="Wandb link to the sweep")
    parser.add_argument("--probe_model", type=str, required=True, help="Probe model to use")
    args = parser.parse_args()

    get_results_15_15(args.link, args.probe_model)

    
    
    
    
    
    