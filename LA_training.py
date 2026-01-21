import wandb
from foundation_main import train_model, load_indices_and_create_subsets, get_train_test_datasets, prepare_run_dir
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
import json
from ast import literal_eval
import re
from types import SimpleNamespace

def freeze_except_layer(model, layer_name):
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze parameters of the specific layer
    for name, param in model.named_parameters():
        if layer_name in name:
            param.requires_grad = True


def freeze_all_except_LA(model, wandb_config, args, logger):
    """
    Configure model parameters by freezing/unfreezing based on probe model type.
    Logs parameter information before and after freezing, and sets total parameter counts in args.
    
    Args:
        model: The model to configure
        wandb_config: Configuration object containing probe_model information
        args: Arguments namespace to store total parameter counts
        logger: Logger instance for logging information
        
    Raises:
        ValueError: If probe model is not supported or no trainable parameters found
    """
    # Debug: Print parameters before freezing
    logger.info("Parameters before freezing:")
    for name, param in model.named_parameters():
        logger.info(f"{name}: requires_grad = {param.requires_grad}")
    
    if wandb_config.probe_model == "ACT-Vit-foundation":
        logger.info("Freezing all parameters except linear_adapters for ACT-Vit-foundation")
        freeze_except_layer(model, "linear_adapters")
    elif wandb_config.probe_model == "ACT-MLP-foundation":
        logger.info("Not freezing any parameters for ACT-MLP-foundation")
    else:
        raise ValueError(f"Probe model {wandb_config.probe_model} not supported")

    # Debug: Print parameters after freezing
    logger.info("Parameters after freezing:")
    trainable_params = 0
    total_trainable_params = 0
    for name, param in model.named_parameters():
        logger.info(f"{name}: requires_grad = {param.requires_grad}")
        if param.requires_grad:
            trainable_params += 1
            total_trainable_params += param.numel()
    
    if trainable_params == 0:
        raise ValueError("No trainable parameters found after freezing! Check if 'linear_adapters' exists in model.")
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total number of parameters in the model: {total_params:,}")
    logger.info(f"Number of trainable parameters: {total_trainable_params:,} ({(total_trainable_params/total_params)*100:.2f}% of total)")
    args.total_params = total_params
    args.total_trainable_params = total_trainable_params



def get_checkpoint_path(args, logger):
    """
    Retrieves the checkpoint path from wandb sweep based on provided arguments.
    
    Args:
        args: Namespace containing wandb_link and probe_model
        logger: Logger instance for logging information
        
    Returns:
        tuple: (Path to the checkpoint file, SimpleNamespace containing wandb config)
        
    Raises:
        ValueError: If wandb link is invalid or required data is not found
    """
    pre_train_dataset = None
    raw_train_dataset = None
    raw_test_dataset = None
    randon_number_of_best_model = None
    if hasattr(args, 'wandb_link') and args.wandb_link:
        # Extract sweep path from wandb link
        # Example link format: https://wandb.ai/username/project/sweeps/sweep_id
        sweep_path = re.search(r'wandb\.ai/([^/]+/[^/]+/sweeps/[^/]+)', args.wandb_link)
        if not sweep_path:
            raise ValueError("Invalid wandb sweep link format. Expected format: https://wandb.ai/username/project/sweeps/sweep_id")
        
        api = wandb.Api()
        sweep = api.sweep(sweep_path.group(1))
        
        # Get all runs in the sweep
        runs = list(sweep.runs)
        if not runs:
            raise ValueError("No runs found in the sweep")
        
        # Filter runs by probe model
        matching_runs = [run for run in runs if run.config.get('probe_model') == args.probe_model]
        if not matching_runs:
            raise ValueError(f"No runs found with probe_model={args.probe_model}")
        
        # Find the run with best validation AUC
        best_run = max(matching_runs, key=lambda run: run.summary.get('best_val_AUC', float('-inf')))
        logger.info(f"Found best run {best_run.name} with best_val_AUC={best_run.summary.get('best_val_AUC')}")
        
        config = best_run.config
        wandb_config = SimpleNamespace(**config)  # Convert dict to object with dot notation
        raw_train_dataset = config.get('train_dataset')
        raw_test_dataset = config.get('test_dataset')
        randon_number_of_best_model = config.get('random_number')
        if not raw_train_dataset:
            raise ValueError("Could not find train_dataset in wandb config")
        if not raw_test_dataset:
            raise ValueError("Could not find test_dataset in wandb config")
            
        # Convert the train_dataset to the required format: [('llm', 'dataset'), ...]
        # Handle space-separated string of LLM:dataset pairs
        if isinstance(raw_train_dataset, str):
            try:
                # Split by spaces and then parse each LLM:dataset pair
                pairs = raw_train_dataset.split()
                pre_train_dataset = []
                for pair in pairs:
                    if ':' not in pair:
                        raise ValueError(f"Invalid format in pair: {pair}. Expected LLM:dataset format")
                    llm, dataset = pair.split(':', 1)
                    pre_train_dataset.append((llm, dataset))
            except Exception as e:
                raise ValueError(f"Could not parse train_dataset format: {raw_train_dataset}. Error: {str(e)}")
        elif isinstance(raw_train_dataset, (list, tuple)):
            pre_train_dataset = raw_train_dataset
        else:
            raise ValueError(f"Unexpected train_dataset format: {type(raw_train_dataset)}")
            
        if not pre_train_dataset:
            raise ValueError("No valid dataset pairs found in train_dataset")
            
        logger.info(f"Using pre-trained model trained on: {pre_train_dataset}")

    if not pre_train_dataset:
        raise ValueError("Could not find train_dataset in wandb config")
    
    if not randon_number_of_best_model:
        raise ValueError("Could not find random_number in wandb config")
    
    # Search for the checkpoint file by run_id since the hash computation might not match exactly
    # The structure is: base_dir / dataset_{hash} / run_{run_id} / best_model_run_{random_number}.pth
    # We'll search for the file using the run_id (random_number) which is unique
    base_path = Path(args.best_model_path)
    checkpoint_filename = f"best_model_run_{randon_number_of_best_model}.pth"
    run_dir_name = f"run_{randon_number_of_best_model}"
    
    # Search for the file in all dataset_* directories
    checkpoint_path = None
    for dataset_dir in base_path.glob("dataset_*"):
        run_dir = dataset_dir / run_dir_name
        potential_path = run_dir / checkpoint_filename
        if potential_path.exists():
            checkpoint_path = potential_path
            logger.info(f"Found checkpoint at: {checkpoint_path}")
            break
    

    assert checkpoint_path.exists(), f"Checkpoint file not found at: {checkpoint_path}"
    
    return checkpoint_path, wandb_config

def validate_dataset_no_overlap(checkpoint_path, train_dataset, test_dataset):
    """
    Validates that there's no overlap between checkpoint training datasets and the current
    train/test datasets to prevent data leakage and ensure proper adaptation.
    
    Args:
        checkpoint_path: Path to the checkpoint directory containing train_dataset.json
        train_dataset: List of (llm, dataset) tuples for current training
        test_dataset: List of (llm, dataset) tuples for current testing
        
    Raises:
        AssertionError: If there's overlap between datasets
    """
    # Load train_dataset.json from checkpoint folder
    with open(checkpoint_path.parent / 'train_dataset.json', 'r') as f:
        ckpt_train_datasets = f.read() # the string is of the form: [('llm1', 'dataset1'), ('llm2', 'dataset2')]
    ckpt_train_datasets = literal_eval(ckpt_train_datasets)['train_dataset']
    
    # Normalize both to tuples for comparison (ckpt_train_datasets may be lists, args.train_dataset may be tuples)
    ckpt_train_datasets_tuples = [tuple(item) if isinstance(item, (list, tuple)) else item for item in ckpt_train_datasets]
    train_dataset_tuples = [tuple(item) if isinstance(item, (list, tuple)) else item for item in train_dataset]
    test_dataset_tuples = [tuple(item) if isinstance(item, (list, tuple)) else item for item in test_dataset]
    
    assert not any(item in ckpt_train_datasets_tuples for item in train_dataset_tuples), \
        f"Some train datasets {train_dataset} are already in checkpoint's training datasets {ckpt_train_datasets}. So 'adapting to them' doesn't make sense."
    
    assert not any(item in ckpt_train_datasets_tuples for item in test_dataset_tuples), \
        f"There is data leakage between train and test datasets."

def main():
    """Main function to preprocess data and load datasets based on task type."""
    # Initialize logger
    logger = get_logger()
    args = parse_args_main(foundation=True, few_shot_adaptation=True)
    
    # Parse command-line arguments
    
    logger.info("Starting the data processing pipeline.")
    logger.info(f"Parsed Arguments: {vars(args)}")

    logger.info(f"Loading pre-trained model")
    # Get checkpoint path from wandb
    checkpoint_path, wandb_config = get_checkpoint_path(args, logger)




    logger.info(f"Taking patch size from wandb config: {wandb_config.patch_size}")
    # NOTE: This is redundant for ACT-MLP-foundation
    args.patch_size = wandb_config.patch_size
    
    
    
    logger.info(f"Validating no overlap/leakage between checkpoint's train datasets and current train/test datasets...")

    validate_dataset_no_overlap(checkpoint_path, args.train_dataset, args.test_dataset)
    
    logger.info(f"No overlap/leakage found between checkpoint's train datasets and current train/test datasets.")
    
    
    
    logger.info(f"Loading preproccessed data")
    # Process datasets
    dataset_train, dataset_test = get_train_test_datasets(args, logger) # now this contains the combined datasets

    args.llms_in_train = dataset_train.llms
    
    train_data, val_data, test_data = load_indices_and_create_subsets(
        dataset_train, dataset_test, args, logger, size_limit=args.size_limit
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
    
    # Load the model from the checkpoint
    model_info = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(model_info['model_state_dict'])
    
    # Configure model parameters (freeze/unfreeze)
    freeze_all_except_LA(model, wandb_config, args, logger)
    
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
        fine_tune_LA_only=True
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