from sklearn.metrics import auc, roc_curve
import wandb
from utils.logger import get_logger
from utils.args import parse_args_main
import os
from pathlib import Path
from utils.dataset_preprocess import *
from transformers import set_seed
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from utils.Architectures import get_model
from transformers import get_scheduler
import time
import hashlib
from collections import defaultdict
import numpy as np
from pathlib import Path
import torch

import json
import uuid
import hashlib
from pathlib import Path
from datetime import datetime, timezone



def get_train_test_datasets(args, logger):
    """Preprocesses datasets and loads them based on the task type."""
    

    logger.info(f"Starting data preparation for train'{args.train_dataset}' and test'{args.test_dataset}'.")
    

    datasets_train = []
    datasets_test = []
    preprocessed_dir = Path(args.base_pre_processed_data_dir)
    for LLM, dataset in args.train_dataset:
        preprocessed_dir_train = preprocessed_dir / LLM / dataset
        dataset_train = CustomSavedDataset(
            preprocessed_dir=preprocessed_dir_train,
            input_output_flag=args.input_output_type, 
            L_eff=args.L_eff,
            N_eff=args.N_eff,
            down_sample_strategy=args.down_sample_strategy
        )
        datasets_train.append(dataset_train)
    logger.info("Training dataset loaded successfully.")
    for LLM, dataset in args.test_dataset:
        assert "test" in dataset, "dataset should contain 'test' in the name."
        preprocessed_dir_test = preprocessed_dir / LLM / dataset
        dataset_test = CustomSavedDataset(
            preprocessed_dir=preprocessed_dir_test,
            input_output_flag=args.input_output_type, 
            L_eff=args.L_eff,
            N_eff=args.N_eff,
            down_sample_strategy=args.down_sample_strategy
        )
        datasets_test.append(dataset_test)
    logger.info("Test dataset loaded successfully.")
    
    dataset_train = CombinedCustomDataset(datasets_train)
    dataset_test = CombinedCustomDataset(datasets_test)
    
    logger.info("Dataset processing pipeline completed successfully.")  
    return dataset_train, dataset_test


def load_indices_and_create_subsets(dataset_train, dataset_test, args, logger, size_limit=10000):
    """
    Load train and validation indices from saved files for each dataset,
    combine them, and create train/val/test subsets.
    
    Args:
        dataset_train: CombinedCustomDataset object for training data
        dataset_test: CombinedCustomDataset object for test data
        args: Arguments object containing base_pre_processed_data_dir
        logger: Logger object
        
    Returns:
        tuple: (train_data, val_data, test_data) as Subset objects
    """
    logger.info("Splitting dataset into train and validation indices.")
    
    for i, dataset in enumerate(dataset_train.datasets):
        # Get the dataset name using the origins mapping
        dataset_name = dataset_train.origins[i]
        logger.info(f"Processing dataset {i}: {dataset_name}")
        
        preprocessed_path = Path(dataset.preprocessed_dir)
        base_path = Path(args.base_pre_processed_data_dir)
        relative_path = preprocessed_path.relative_to(base_path)
        
        # The last part is the dataset name, everything before is the LLM path
        path_parts = relative_path.parts
        dataset_name_only = path_parts[-1]
        LLM = str(Path(*path_parts[:-1]))  # Join all parts except the last one as string
        
        # Load train and val indices from saved files
        indices_path = f"./data/dataset_splits/{LLM}/{dataset_name_only}"
        if size_limit == 10000:
            train_indices = torch.load(f"{indices_path}/train_indices.pt")
            val_indices = torch.load(f"{indices_path}/val_indices.pt")
        else:
            train_indices = torch.load(f"{indices_path}/size_limit_{size_limit}/train_indices.pt")
            val_indices = torch.load(f"{indices_path}/size_limit_{size_limit}/val_indices.pt")
        
        dataset_train.add_indices([train_indices, val_indices], dataset_idx=i)

    train_indices = dataset_train.get_combined_indices(split='train')
    val_indices = dataset_train.get_combined_indices(split='val')
    logger.info(f"Train size: {len(train_indices)}, Validation size: {len(val_indices)}, Test size: {len(dataset_test)}")

    train_data = Subset(dataset_train, train_indices)
    val_data = Subset(dataset_train, val_indices)
    test_data = dataset_test
    
    return train_data, val_data, test_data



def prepare_run_dir(
    train_dataset,
    test_dataset,
    base_dir,
    hash_len=12,
    fail_on_collision=True,
    fine_tune_LA_only=False
):
    """
    Creates and returns a unique run directory structured as:
        base_dir/dataset_<dataset_hash>/run_<run_id>/

    Saves:
        - train_dataset.json
        - test_dataset.json
        - manifest.json
    """

    # Stable dataset serialization (include both train and test)
    dataset_dict = {
        "train_dataset": train_dataset,
        "test_dataset": test_dataset,
    }
    dataset_bytes = json.dumps(
        dataset_dict,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    dataset_hash = hashlib.blake2b(
        dataset_bytes, digest_size=16
    ).hexdigest()[:hash_len]

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    if fine_tune_LA_only:
        run_id += "_fine_tune_LA_only"
    run_dir = (
        Path(base_dir)
        / f"dataset_{dataset_hash}"
        / f"run_{run_id}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=not fail_on_collision,
    )

    # Atomic write of train dataset
    train_dataset_tmp = run_dir / "train_dataset.json.tmp"
    train_dataset_final = run_dir / "train_dataset.json"
    train_dataset_bytes = json.dumps(
        {"train_dataset": train_dataset},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    train_dataset_tmp.write_text(train_dataset_bytes.decode("utf-8"), encoding="utf-8")
    train_dataset_tmp.replace(train_dataset_final)

    # Atomic write of test dataset
    test_dataset_tmp = run_dir / "test_dataset.json.tmp"
    test_dataset_final = run_dir / "test_dataset.json"
    test_dataset_bytes = json.dumps(
        {"test_dataset": test_dataset},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    test_dataset_tmp.write_text(test_dataset_bytes.decode("utf-8"), encoding="utf-8")
    test_dataset_tmp.replace(test_dataset_final)

    # Manifest
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_hash": dataset_hash,
        "run_id": run_id,
        "num_train_examples": len(train_dataset) if hasattr(train_dataset, "__len__") else None,
        "num_test_examples": len(test_dataset) if hasattr(test_dataset, "__len__") else None,
        "path": str(run_dir),
    }

    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    return run_dir, dataset_hash, run_id


def save_best_model(logger, model, best_val_auc, best_test_auc, args):
    """Save the best model checkpoint inside the run directory."""

    run_dir = Path(args.best_model_path)

    model_path = run_dir / f"best_model_run_{args.random_number}.pth"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "best_val_auc": best_val_auc,
            "best_test_auc": best_test_auc,
            "dataset_hash": getattr(args, "dataset_hash", None),
            "run_id": args.random_number,
        },
        model_path,
    )

    logger.info(f"Best model saved at {model_path}")


def seperate_labels_predictions(labels, predictions, item_origins, origin_labels_predictions):
    item_origins = np.array(item_origins)
    unique_origins = np.unique(item_origins)
    for value in unique_origins:
        indices = np.where(item_origins == value)[0]
        origin_labels_predictions[value]['labels'].extend(labels[indices].cpu().tolist())
        origin_labels_predictions[value]['predictions'].extend(predictions[indices].detach().cpu().tolist())
    return origin_labels_predictions



def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """Trains the model for one epoch."""
    model.train()
    total_loss = 0
    all_labels, all_predictions = [], []
    origin_labels_predictions = defaultdict(lambda: {'labels': [], 'predictions': []})
    
    for item_origins, (data, labels) in tqdm(dataloader, desc="Training Progress"):
        indices = torch.tensor([list(FEATURE_DIMS.keys()).index(origin.split('_')[0]) for origin in item_origins], device=device)


        activations = data.to(device)
        optimizer.zero_grad()
        predictions = model(activations, indices).reshape(-1)
            
        all_labels.extend(labels.tolist())
        labels = labels.to(device)
        all_predictions.extend(predictions.detach().cpu().tolist())
        
        loss = criterion(predictions, labels.float())
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        # Collect labels and predictions by origin
        origin_labels_predictions = seperate_labels_predictions(labels, predictions, item_origins, origin_labels_predictions)
    fpr, tpr, _ = roc_curve(np.array(all_labels, dtype=bool), np.array(all_predictions))
    auc_score = auc(fpr, tpr)
    auc_score_by_origin = {}

    for origin, data in origin_labels_predictions.items():
        fpr, tpr, _ = roc_curve(np.array(data['labels'], dtype=bool), np.array(data['predictions']))
        auc_score_by_origin[origin] = {'auc': auc(fpr, tpr)}
    
    return total_loss / len(dataloader), auc_score, auc_score_by_origin

def evaluate(model, dataloader, criterion, device, desc="Validation"):
    """Evaluates the model on validation or test data."""
    model.eval()
    total_loss = 0
    all_labels, all_predictions = [], []
    origin_labels_predictions = defaultdict(lambda: {'labels': [], 'predictions': []})

    with torch.no_grad():
        for item_origins, (data,labels) in tqdm(dataloader, desc=f"{desc} Progress"):
            indices = torch.tensor([list(FEATURE_DIMS.keys()).index(origin.split('_')[0]) for origin in item_origins], device=device)
            
            activations = data.to(device)
            predictions = model(activations, indices).reshape(-1)
            
            all_labels.extend(labels.tolist())    
            all_predictions.extend(predictions.detach().cpu().tolist())
            
            labels = labels.to(device)
            loss = criterion(predictions, labels.float())
            total_loss += loss.item()

            # Collect labels and predictions by origin
            origin_labels_predictions = seperate_labels_predictions(labels, predictions, item_origins, origin_labels_predictions)

    # Calculate overall AUC and TPR@5%FPR
    fpr, tpr, _ = roc_curve(np.array(all_labels, dtype=bool), np.array(all_predictions))
    auc_score = auc(fpr, tpr)
    tpr_5_fpr = tpr[np.where(fpr < 0.05)[0][-1]] if np.any(fpr < 0.05) else 0

    # Calculate AUC and TPR@5%FPR for each origin
    origin_metrics = {}
    for origin, data in origin_labels_predictions.items():
        fpr, tpr, _ = roc_curve(np.array(data['labels'], dtype=bool), np.array(data['predictions']))
        origin_auc = auc(fpr, tpr)
        origin_tpr_5_fpr = tpr[np.where(fpr < 0.05)[0][-1]] if np.any(fpr < 0.05) else 0
        origin_metrics[origin] = {'auc': origin_auc, 'tpr_5_fpr': origin_tpr_5_fpr}

    return total_loss / len(dataloader), auc_score, tpr_5_fpr, origin_metrics



def train_model(logger, model, dataloader_train, dataloader_val, dataloader_test, criterion, optimizer, scheduler, args, device):
    """Trains and evaluates the model with early stopping."""
    best_val_auc, best_val_tpr_5_fpr = -1, -1
    best_test_auc, best_test_tpr_5_fpr = -1, -1
    best_val_auc_epoch, best_val_tpr_5_fpr_epoch = 0, 0
    patience, no_improve_count = args.patience, 0
    
    for epoch in range(args.num_epochs):
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
        
        train_loss, auc_train, train_auc_per_origin = train_one_epoch(model, dataloader_train, criterion, optimizer, scheduler, device)
        logger.info(f"Train Loss: {train_loss:.4f}, Train AUC: {auc_train:.4f}")
        
        # Validation
        val_loss, auc_val, tpr_5_fpr_val, val_metrics_per_origin = evaluate(model, dataloader_val, criterion, device, desc="Validation")
        logger.info(f"Val Loss: {val_loss:.4f}, Val AUC: {auc_val:.4f}, Val TPR@5%FPR: {tpr_5_fpr_val:.4f}")
        
        # Test
        test_loss, auc_test, tpr_5_fpr_test, test_metrics_per_origin = evaluate(model, dataloader_test, criterion, device, desc="Test")
        logger.info(f"Test Loss: {test_loss:.4f}, Test AUC: {auc_test:.4f}, Test TPR@5%FPR: {tpr_5_fpr_test:.4f}")
        
        # Save the best model if validation AUC improves
        if auc_val > best_val_auc:
            save_best_model(logger, model, auc_val, auc_test, args)
            best_val_auc, best_test_auc = auc_val, auc_test
            best_val_auc_epoch = epoch + 1
            no_improve_count = 0  # Reset counter
        else:
            no_improve_count += 1
            logger.info(f"No improvement for {no_improve_count} epochs.")
        
        # Update best TPR@5%FPR
        if tpr_5_fpr_val > best_val_tpr_5_fpr:
            best_val_tpr_5_fpr, best_test_tpr_5_fpr = tpr_5_fpr_val, tpr_5_fpr_test
            best_val_tpr_5_fpr_epoch = epoch + 1
        
        # Early stopping
        if no_improve_count >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs.")
            break
        
        # Logging to WandB
        logging_dict = {
            "train_loss_epoch": train_loss,
            "AUC_train_epoch": auc_train,
            "val_loss_epoch": val_loss,
            "best_val_AUC": best_val_auc,
            "best_val_tpr_5_fpr": best_val_tpr_5_fpr,
            "test_loss_epoch": test_loss,
            "best_test_AUC": best_test_auc,
            "best_test_tpr_5_fpr": best_test_tpr_5_fpr,
            "learning_rate": scheduler.get_last_lr()[0],
            "epoch": epoch + 1,
        }

        # Function to update logging_dict with metrics
        def update_logging_dict(metrics_dict, prefix):
            for origin, metrics in metrics_dict.items():
                logging_dict[f"{prefix}_auc_{origin}"] = metrics.get('auc', metrics)
                if 'tpr_5_fpr' in metrics:
                    logging_dict[f"{prefix}_tpr_5_fpr_{origin}"] = metrics['tpr_5_fpr']

        update_logging_dict(val_metrics_per_origin, "val")
        update_logging_dict(test_metrics_per_origin, "test")
        update_logging_dict(train_auc_per_origin, "train")
        
        # Log information to find the best model
        logging_dict["best_val_AUC_epoch"] = best_val_auc_epoch
        logging_dict["best_val_tpr_5_fpr_epoch"] = best_val_tpr_5_fpr_epoch
        logging_dict["best_test_AUC_at_best_val_epoch"] = best_test_auc
        logging_dict["best_test_tpr_5_fpr_at_best_val_tpr_epoch"] = best_test_tpr_5_fpr
        
        # Log the path to the best model
        best_model_path = Path(args.best_model_path) / f"best_model_run_{args.random_number}.pth"
        logging_dict["best_model_path"] = str(best_model_path)
        
        wandb.log(logging_dict)
    
    wandb.finish()
    logger.info("Training complete.")
    
  
# ---



def main():
    """Main function to preprocess data and load datasets based on task type."""
    # Initialize logger
    logger = get_logger()
    
    # Parse command-line arguments
    args = parse_args_main(foundation=True)


    logger.info("Starting the data processing pipeline.")
    logger.info(f"Parsed Arguments: {vars(args)}")
    

    logger.info(f"Loading preproccessed data")
    # Process datasets
    dataset_train, dataset_test = get_train_test_datasets(args, logger) # now this contains the combined datasets

    args.llms_in_train = dataset_train.llms
    
    train_data, val_data, test_data = load_indices_and_create_subsets(
        dataset_train, dataset_test, args, logger
    )

    set_seed(args.seed)
    device = f"cuda:{args.cuda_idx}" if torch.cuda.is_available() else "cpu"
    
    logger.info("Creating dataloaders for training, validation, and test sets.")    
    dataloader_train = DataLoader(
        train_data,          # Your dataset instance
        batch_size=args.batch_size,     # Number of samples per batch
        shuffle=True,     # Shuffle data for training
        prefetch_factor=2 if args.num_workers > 0 else None,
        num_workers=args.num_workers,    # Number of worker threads for data loading
        pin_memory=True if args.pin_memory==1 else False,
    )

    dataloader_val = DataLoader(
        val_data,          # Your dataset instance
        batch_size=args.batch_size,     # Number of samples per batch
        shuffle=False,     # Shuffle data for training
        prefetch_factor=2 if args.num_workers > 0 else None,
        num_workers=args.num_workers,    # Number of worker threads for data loading
        pin_memory=True if args.pin_memory==1 else False,
    )
    
    dataloader_test = DataLoader(
        test_data,          # Your dataset instance
        batch_size=args.batch_size,     # Number of samples per batch
        shuffle=False,     # Shuffle data for training
        prefetch_factor=2 if args.num_workers > 0 else None,
        num_workers=args.num_workers,    # Number of worker threads for data loading
        pin_memory=True if args.pin_memory==1 else False,
    )
    


    logger.info(f"Creating model for sequence length {train_data[0][1][0].shape[-2]} and feature dimension: {train_data[0][1][0].shape[-1]}")
    model = get_model(args=args,
                      input_dim=train_data[0][1][0].shape[-1],
                      input_shape=train_data[0][1][0].shape).to(device=device)

    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total number of parameters in the model: {total_params}")
    args.total_params = total_params
    
    logger.info("Creating optimizer and scheduler.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Define the number of training steps
    num_training_steps = len(dataloader_train) * args.num_epochs  # Total training steps
    logger.info(f"Total number of training steps: {num_training_steps}, and warm-up steps: {int(0.1 * num_training_steps)}")
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% of steps for warm-up

    # Create the scheduler
    scheduler = get_scheduler(
        "cosine", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    
    criterion = torch.nn.BCELoss()
    
    
    args.best_model_path, dataset_hash, args.random_number = prepare_run_dir(
        train_dataset=args.train_dataset,
        test_dataset=args.test_dataset,
        base_dir=args.best_model_path,
    )

    logger.info(
        f"Saving best model to {args.best_model_path} \n"
        f"(dataset_hash={dataset_hash}, run_id={args.random_number})"
    )
    logger.info("Starting wandb, project is ACT-ViT.")
    logger.info("Starting training loop.")
    
    wandb.init(project="ACT-ViT", config=args)
    
    train_model(logger=logger, model=model, dataloader_train=dataloader_train, dataloader_val=dataloader_val, dataloader_test=dataloader_test, criterion=criterion, optimizer=optimizer, scheduler=scheduler, args=args, device=device)
    
if __name__ == '__main__':
    main()