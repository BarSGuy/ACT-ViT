import gc
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import os
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score
from pathlib import Path
from utils.args import parse_args_token_layers_scores
from utils.dataset_preprocess import *
from utils.logger import get_logger
import time
from datetime import datetime, timedelta
import numpy as np
import os
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from utils.constants import MAX_DATASET_SIZES, TRAIN_DATASET_SIZES, MOVIES_TEST_DATASET_SIZE, SIZE_LIMITS, WEIGHT_DECAYS
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm



## ------------------------------------------------------------- ##
def clean_memory():
    """Helper function to clean both CPU and GPU memory if available"""
    # Clean CPU memory
    gc.collect()
    # Clean GPU memory if using PyTorch with CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_results(path, logger,
             clfs=None, count_table_train=None, count_table_val=None, count_table_test=None, 
             indices_train=None, indices_val=None, indices_test=None,
             all_preds_train=None, all_preds_val=None, all_preds_test=None, 
             auc_table_train=None, auc_table_val=None, auc_table_test=None,
             labels_train=None, labels_val=None, labels_test=None,
             total_time=None):
    """
    Save all results to the specified path
    
    Args:
        path: Path where results will be saved
        clfs: Dictionary of trained classifiers
        count_table_*: Count tables for each split
        all_preds_*: Prediction tables for each split
        auc_table_*: AUC score tables for each split
        *_y: Labels for each split
        logger: Logger for status messages
    """
    # Create dictionary to store all data that needs to be saved
    save_data = {
        'clfs': clfs,
        'count_table_train': count_table_train,
        'count_table_val': count_table_val,
        'count_table_test': count_table_test,
        'indices_train': indices_train,
        'indices_val': indices_val,
        'indices_test': indices_test,
        'all_preds_train': all_preds_train, 
        'all_preds_val': all_preds_val,
        'all_preds_test': all_preds_test,
        'auc_table_train': auc_table_train,
        'auc_table_val': auc_table_val,
        'auc_table_test': auc_table_test,
        'labels_train': labels_train,
        'labels_val': labels_val,
        'labels_test': labels_test,
    }

    with open(path / 'total_time.txt', 'w') as f:
        f.write(f"{total_time}")
    # Save all tables to pickle files
    for name, data in save_data.items():
        save_file = path / f"{name}.pkl"
        with open(save_file, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved {name} to {save_file}")
    
    # Also save AUC tables as CSV for easier analysis
    for split in ['train', 'val', 'test']:
        auc_table = save_data[f'auc_table_{split}']
        
        # Convert to DataFrame format
        auc_df_data = []
        for layer in auc_table:
            for token in auc_table[layer]:
                for wd in auc_table[layer][token]:
                    auc_df_data.append({
                        'layer': layer,
                        'token': token,
                        'weight_decay': wd,
                        'auc': auc_table[layer][token][wd]
                    })
        
        auc_df = pd.DataFrame(auc_df_data)
        csv_file = path / f"auc_table_{split}.csv"
        auc_df.to_csv(csv_file, index=False)
        logger.info(f"Saved {split} AUC table to {csv_file}")
    
    # Save count tables as CSV for easier analysis
    for split in ['train', 'val', 'test']:
        count_table = save_data[f'count_table_{split}']
        
        # Convert to DataFrame format
        count_df_data = []
        for layer in count_table:
            for token in count_table[layer]:
                count_df_data.append({
                    'layer': layer,
                    'token': token,
                    'count': count_table[layer][token]
                })
        
        count_df = pd.DataFrame(count_df_data)
        csv_file = path / f"count_table_{split}.csv"
        count_df.to_csv(csv_file, index=False)
        logger.info(f"Saved {split} count table to {csv_file}")
    
    return save_data


def prepare_data_for_clf(data, y, layer, token):
    """
    Prepare training data for a specific layer and token.
    
    Args:
        train_data: List of training data samples
        train_y: List of training labels
        layer: Layer index to extract features from
        token: Token index to extract features from
        logger: Logger for logging messages
        
    Returns:
        tuple: (X_train, y_train) numpy arrays ready for model training
    """
    train = []
    ys = []
    indices = []
    for i, a in enumerate(data):
        try:
            train.append(a[layer][token])
            ys.append(y[i])
            indices.append(i)
        except IndexError as e:
            logger.info(f"IndexError: {e} for sample {i} in layer {layer} and token {token}. Skipping this sample, probably the answer has less tokens than the inspected token.")

    X_train = torch.stack(train).to(torch.float32).cpu().numpy()
    y_train = torch.tensor(ys).cpu().numpy().flatten()
    indices_train = torch.tensor(indices).cpu().numpy().flatten()
    return X_train, y_train, indices_train


def initialize_dicts(num_layers, tokens_to_loop, wds):
    """
    Initialize count tables for training, validation, and test sets.
    
    Args:
        num_layers: Number of layers in the model
        tokens_to_loop: tokens to loop over
        wds: List of weight decay values to use
        
    Returns:
        Tuple of (count_table_train, count_table_val, count_table_test)
    """
    clfs = {}

    all_preds_train = {}
    all_preds_val = {}
    all_preds_test = {}
    
    auc_table_train = {}
    auc_table_val = {}
    auc_table_test = {}

    count_table_train = {}
    count_table_val = {}
    count_table_test = {}

    indices_train = {}
    indices_val = {}
    indices_test = {}

    labels_train = {}
    labels_val = {}
    labels_test = {}
    
    for layer in range(num_layers):
        clfs[layer] = {}

        count_table_train[layer] = {}
        count_table_val[layer] = {}
        count_table_test[layer] = {}

        indices_train[layer] = {}
        indices_val[layer] = {}
        indices_test[layer] = {}

        labels_train[layer] = {}
        labels_val[layer] = {}
        labels_test[layer] = {}
        
        all_preds_train[layer] = {}
        all_preds_val[layer] = {}
        all_preds_test[layer] = {}
        
        auc_table_train[layer] = {}
        auc_table_val[layer] = {}
        auc_table_test[layer] = {}
        
        for token in tokens_to_loop:
            clfs[layer][token] = {}

            count_table_train[layer][token] = -1
            count_table_val[layer][token] = -1
            count_table_test[layer][token] = -1

            indices_train[layer][token] = -1
            indices_val[layer][token] = -1
            indices_test[layer][token] = -1

            labels_train[layer][token] = -1
            labels_val[layer][token] = -1
            labels_test[layer][token] = -1
            
            all_preds_train[layer][token] = {}
            all_preds_val[layer][token] = {}
            all_preds_test[layer][token] = {}
            
            auc_table_train[layer][token] = {}
            auc_table_val[layer][token] = {}
            auc_table_test[layer][token] = {}
            
            for wd in wds:
                clfs[layer][token][wd] = -1

                all_preds_train[layer][token][wd] = -1
                all_preds_val[layer][token][wd] = -1
                all_preds_test[layer][token][wd] = -1

                auc_table_train[layer][token][wd] = -1
                auc_table_val[layer][token][wd] = -1
                auc_table_test[layer][token][wd] = -1
            
    return clfs, count_table_train, count_table_val, count_table_test, all_preds_train, all_preds_val, all_preds_test, auc_table_train, auc_table_val, auc_table_test, labels_train, labels_val, labels_test, indices_train, indices_val, indices_test


def load_samples(params):
    i, args, load_dir = params
    act_path = load_dir / f'activations_{args.input_output_type}_{str(i)}.pt'
    label_path = load_dir / f'label_{str(i)}.pt'

    try:
        activation = torch.load(act_path)
        label = torch.load(label_path)
    except Exception as e:
        print(f"Error loading sample {i}: {e}, probably it doesn't exist in the folder because that dataset has less samples than the requested number.")
        return None, None
    return activation, label


def collect_data(load_dir, args, n_samples=MAX_DATASET_SIZES):
    def load_wrapper(i, pbar):
        result = load_samples((i, args, load_dir))
        pbar.update(1)
        return i, result

    results = [None] * n_samples  # Preallocate list to preserve order
    # NOTE: for debugging purposes, use num_workers = 1
    num_workers = min(int(0.5*multiprocessing.cpu_count()), n_samples)
    # num_workers = 1
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        with tqdm(total=n_samples) as pbar:
            futures = [executor.submit(load_wrapper, i, pbar) for i in range(n_samples)]
            for future in as_completed(futures):
                i, result = future.result()
                results[i] = result

    gc.collect()  # Force garbage collection

    # Filter out failed samples and preserve index match between activations and labels
    filtered_results = [r for r in results if r is not None and r[0] is not None and r[1] is not None]

    if not filtered_results:
        raise RuntimeError("No valid samples were loaded. Check your data directory and sample count.")

    activations, labels = zip(*filtered_results)
    return list(activations), list(labels)
## ------------------------------------------------------------- ##

if __name__ == '__main__':
    
    
    # Parse command-line arguments
    args = parse_args_token_layers_scores()

    logger = get_logger(f"probing_{args.LLM}_{args.dataset}_{args.input_output_type}.log")
    logger.info(f"Parsed Arguments: {vars(args)}")
    load_dir = Path(args.base_raw_data_dir) / args.LLM / args.dataset
    test_dir = Path(str(load_dir) + '_test')
    
    # Collect training data
    logger.info(f"Starting data collection from {load_dir}")
    
    activations_train, train_y = collect_data(load_dir=load_dir, args=args)

    
    assert len(activations_train) == len(train_y) == TRAIN_DATASET_SIZES, f"Train data size {len(activations_train)} is not equal to the maximum dataset size {TRAIN_DATASET_SIZES}"


    logger.info(f"Starting test data collection from {test_dir}")
    n_samples_to_load = TRAIN_DATASET_SIZES if args.dataset != 'movies' else MOVIES_TEST_DATASET_SIZE
    activations_test, test_y = collect_data(load_dir=test_dir, args=args, n_samples=n_samples_to_load)

    if 'movies_test' in str(test_dir):
        assert len(activations_test) == len(test_y) == MOVIES_TEST_DATASET_SIZE, f"Movies test data size {len(activations_test)} is not equal to the maximum dataset size {MOVIES_TEST_DATASET_SIZE}"
    else:
        assert len(activations_test) == len(test_y) == MAX_DATASET_SIZES, f"Test data size {len(activations_test)} is not equal to the maximum dataset size {MAX_DATASET_SIZES}"
        


    size_limits = [10000] + SIZE_LIMITS
    
    wds = WEIGHT_DECAYS
    for size_limit in size_limits:
        logger.info(f"Processing size limit: {size_limit}")
        
        if size_limit == 10000:
            train_indices = torch.load(f"./data/dataset_splits/{args.LLM}/{args.dataset}/train_indices.pt")
            val_indices = torch.load(f"./data/dataset_splits/{args.LLM}/{args.dataset}/val_indices.pt")
        else:
            train_indices = torch.load(f"./data/dataset_splits/{args.LLM}/{args.dataset}/size_limit_{size_limit}/train_indices.pt")
            val_indices = torch.load(f"./data/dataset_splits/{args.LLM}/{args.dataset}/size_limit_{size_limit}/val_indices.pt")

        train_data_size_limit = [activations_train[i] for i in train_indices]
        train_y_size_limit = [train_y[i] for i in train_indices]
        val_data_size_limit = [activations_train[i] for i in val_indices]
        val_y_size_limit = [train_y[i] for i in val_indices]
        test_data = activations_test



        path = Path(f"./probing_baselines/{args.LLM}/{args.dataset}/{args.input_output_type}/size_limit_{size_limit}")
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving results to {path}")
        
        num_layers = train_data_size_limit[0].shape[0]
        tokens_to_loop = [0, 1, 2, -1, -2, -3]
        # Initialize count tables
        clfs, count_table_train, count_table_val, count_table_test, all_preds_train, all_preds_val, all_preds_test, auc_table_train, auc_table_val, auc_table_test, labels_train, labels_val, labels_test, indices_train, indices_val, indices_test = initialize_dicts(num_layers, tokens_to_loop, wds)
        
        start_time = time.time()
        for layer in tqdm(range(num_layers), desc=f"Processing layers for size limit {size_limit}"):
            for token in tokens_to_loop:

                logger.info(f"Processing layer {layer}, token {token}")
                # == data preparation ==
                X_train, y_train, idx_train = prepare_data_for_clf(train_data_size_limit, train_y_size_limit, layer, token)
                count_table_train[layer][token] = X_train.shape[0]
                indices_train[layer][token] = idx_train
                labels_train[layer][token] = y_train

                X_val, y_val, idx_val = prepare_data_for_clf(val_data_size_limit, val_y_size_limit, layer, token)
                count_table_val[layer][token] = X_val.shape[0]
                indices_val[layer][token] = idx_val
                labels_val[layer][token] = y_val


                X_test, y_test, idx_test = prepare_data_for_clf(test_data, test_y, layer, token)

                count_table_test[layer][token] = X_test.shape[0]
                indices_test[layer][token] = idx_test
                labels_test[layer][token] = y_test
                for wd in wds:
                    clf = LogisticRegression(C=wd)
                    clf.fit(X_train, y_train)
                    
                    clfs[layer][token][wd] = clf
                    # == Get predictions
                    probs_train = clf.predict_proba(X_train)[:, 1]
                    auc_train = roc_auc_score(y_train, probs_train)
                    all_preds_train[layer][token][wd] = probs_train
                    auc_table_train[layer][token][wd] = auc_train

                    probs_val = clf.predict_proba(X_val)[:, 1]
                    auc_val = roc_auc_score(y_val, probs_val)
                    all_preds_val[layer][token][wd] = probs_val
                    auc_table_val[layer][token][wd] = auc_val

                    probs_test = clf.predict_proba(X_test)[:, 1]
                    auc_test = roc_auc_score(y_test, probs_test)
                    all_preds_test[layer][token][wd] = probs_test
                    auc_table_test[layer][token][wd] = auc_test
                
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Time taken for wall-clock training: {total_time} seconds")
        
        save_data = save_results(
            path=path,
            logger=logger,
            clfs=clfs, 
            count_table_train=count_table_train,
            count_table_val=count_table_val, 
            count_table_test=count_table_test,
            indices_train=indices_train,
            indices_val=indices_val,
            indices_test=indices_test,
            all_preds_train=all_preds_train, 
            all_preds_val=all_preds_val, 
            all_preds_test=all_preds_test,
            auc_table_train=auc_table_train, 
            auc_table_val=auc_table_val, 
            auc_table_test=auc_table_test,
            labels_train=labels_train,
            labels_val=labels_val,
            labels_test=labels_test,
            total_time=total_time
        )
        print("Done saving all tables!")
    
        # Final cleanup of resources
        # Clear variables to free memory
        save_data.clear()
        del clfs, count_table_train, count_table_val, count_table_test
        del all_preds_train, all_preds_val, all_preds_test
        del auc_table_train, auc_table_val, auc_table_test
        del train_data_size_limit, train_y_size_limit, val_data_size_limit, val_y_size_limit, test_data
        
        # Force garbage collection
        clean_memory()
        
        logger.info("Resource cleanup completed successfully")
    

