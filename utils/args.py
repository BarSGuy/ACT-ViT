import argparse
from utils.constants import LIST_OF_DATASETS, LIST_OF_MODELS
from utils.datasets_helper import DatasetComboAction
from utils.constants import PROBING_ARCHITECTURES

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


########################################################################
####################### Generate Raw Datasets ##########################
########################################################################
def parse_args_raw_datasets():
    """
    Parse command-line arguments for the script.
    
    Returns:
    --------
    argparse.Namespace:
        Parsed command-line arguments with dataset, model, and split details.
    """
    parser = argparse.ArgumentParser(description="Generate model activations and labels from a specified dataset.")
    
    # Argument for selecting the model
    parser.add_argument(
        "--LLM",
        choices=LIST_OF_MODELS,
        default='meta-llama/Meta-Llama-3-8B-Instruct',
        help="Pretrained model to use for generating raw activations."
    )
    
    # Argument for selecting the dataset
    parser.add_argument(
        "--dataset",
        choices=LIST_OF_DATASETS,
        default='triviaqa',
        help="Dataset to be processed."
    )
    parser.add_argument(
        "--n_samples", 
        type=int, 
        default=10_000,
        help='number of examples to use'
        )
    parser.add_argument(
        "--chunk", 
        type=int, 
        default=1,
        help='chunk id to use when generating the raw datasets'
        )
    
    parser.add_argument(
        "--base_raw_data_dir",
        type=str,
        default='raw_data',
        help="Base directory for saving raw data."
    )

    parser.add_argument(
        "--save_activations_input",
        type=str2bool,
        default=False,
        help="A flag to save the activation tensor for the input tokens as well."
    )
    
    return parser.parse_args()

########################################################################
####################### Pre Process Raw Datasets #######################
########################################################################
def parse_args_pre_process():
    """
    Parse command-line arguments for the script.
    
    Returns:
    --------
    argparse.Namespace:
        Parsed command-line arguments with dataset, model, and split details.
    """
    parser = argparse.ArgumentParser(description="Generate model activations and labels from a specified dataset.")
    
    # Argument for selecting the model
    parser.add_argument(
        "--LLM",
        choices=LIST_OF_MODELS,
        default='meta-llama/Meta-Llama-3-8B-Instruct',
        help="Pretrained model of which its raw activations we will preprocess."
    )
    
    # Argument for selecting the dataset
    parser.add_argument(
        "--dataset",
        choices=LIST_OF_DATASETS,
        default='triviaqa',
        help="Dataset to be processed."
    )
    
    parser.add_argument(
        "--base_raw_data_dir",
        type=str,
        default='raw_data',
        help="Base directory for the raw data."
    )
    
    parser.add_argument(
        "--base_pre_processed_data_dir", 
        type=str, 
        default='preprocessed_data',
        help="Base directory for the pre-processed data."
    )
    
    parser.add_argument(
        "--input_output_type", 
        type=str, 
        default="output",
        choices=["output"],
        help="A flag to indicate whether to preprocess the activation tensor for the input or output tokens."
    )
    
        
    parser.add_argument(
        "--down_sample_strategy",
        type=str,
        default="pool",
        choices=["none", "pool"],
        help="A flag to indicate the downsampling strategy to use. We support 'pool' as in the paper, and 'none', which will not downsample the activation tensor."
    )
    
    
    parser.add_argument(
        "--L_eff", 
        type=int, 
        default=8,
        help="L_p from the paper."
    )
    
    parser.add_argument(
        "--N_eff", 
        type=int, 
        default=100,
        help="N_p from the paper."
    )
    
    return parser.parse_args()

########################################################################
####################### Create Datasets Splits #########################
########################################################################

def parse_args_create_datasets_splits():
    parser = argparse.ArgumentParser(
        description="Create train/validation/test splits for all datasets and models."
    )
    parser.add_argument(
        "--base_raw_data_dir",
        type=str,
        default='raw_data',
        help="Base directory for the raw data."
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="./data/dataset_splits",
        help="Base directory for saving the splits."
    )

    return parser.parse_args()
    

#########################################################################
############################### Probing #################################
#########################################################################

def parse_args_token_layers_scores():
    parser = argparse.ArgumentParser(
        description="Parse arguments for computing the token layers scores."
    )
    parser.add_argument(
        "--LLM",
        choices=LIST_OF_MODELS,
        default='meta-llama/Meta-Llama-3-8B-Instruct',
        help="Pretrained model of which we probe."
    )
    parser.add_argument(
        "--dataset",
        choices=LIST_OF_DATASETS,
        default='triviaqa',
        help="Dataset over which we probe."
    )
        
    parser.add_argument(
        "--base_raw_data_dir",
        type=str,
        default='raw_data',
        help="Base directory for the raw data."
    )
    parser.add_argument(
        "--input_output_type", 
        type=str, 
        default="output",
        choices=["output"],
        help="A flag to indicate whether to preprocess the activation tensor for the input or output tokens. Here we only support 'output'."
    )
    
    return parser.parse_args()


#########################################################################
############################## Foundation ###############################
#########################################################################

def parse_args_main(foundation=False, few_shot_adaptation=False):
    """
    Parses command-line arguments for training, model, dataset, and logging configurations.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Parse arguments for training and evaluation pipeline."
    )
    
    if foundation:
        parser.add_argument(
            '--train_dataset',
            action=DatasetComboAction,
            default=[
                # -- Qwen --
                ('Qwen/Qwen2.5-7B-Instruct', 'imdb'),
                ('Qwen/Qwen2.5-7B-Instruct', 'hotpotqa'),
                ('Qwen/Qwen2.5-7B-Instruct', 'triviaqa'),
                ('Qwen/Qwen2.5-7B-Instruct', 'movies'),
                ('Qwen/Qwen2.5-7B-Instruct', 'hotpotqa_with_context'),
                # -- Mistral --
                ('mistralai/Mistral-7B-Instruct-v0.2', 'imdb'),
                ('mistralai/Mistral-7B-Instruct-v0.2', 'hotpotqa'),
                ('mistralai/Mistral-7B-Instruct-v0.2', 'triviaqa'),
                ('mistralai/Mistral-7B-Instruct-v0.2', 'movies'),
                ('mistralai/Mistral-7B-Instruct-v0.2', 'hotpotqa_with_context'),
                # -- Meta --
                ('meta-llama/Meta-Llama-3-8B-Instruct', 'imdb'),
                ('meta-llama/Meta-Llama-3-8B-Instruct', 'hotpotqa'),
                ('meta-llama/Meta-Llama-3-8B-Instruct', 'triviaqa'),
                ('meta-llama/Meta-Llama-3-8B-Instruct', 'movies'),
                ('meta-llama/Meta-Llama-3-8B-Instruct', 'hotpotqa_with_context')
            ]
            ,
            help="Training LLM/dataset combinations."
        )
    
        parser.add_argument(
            '--test_dataset',
            action=DatasetComboAction,
            default=[
                # -- Qwen --
                ('Qwen/Qwen2.5-7B-Instruct', 'imdb_test'),
                ('Qwen/Qwen2.5-7B-Instruct', 'hotpotqa_test'),
                ('Qwen/Qwen2.5-7B-Instruct', 'triviaqa_test'),
                ('Qwen/Qwen2.5-7B-Instruct', 'movies_test'),
                ('Qwen/Qwen2.5-7B-Instruct', 'hotpotqa_with_context_test'),
                # -- Mistral --
                ('mistralai/Mistral-7B-Instruct-v0.2', 'imdb_test'),
                ('mistralai/Mistral-7B-Instruct-v0.2', 'hotpotqa_test'),
                ('mistralai/Mistral-7B-Instruct-v0.2', 'triviaqa_test'),
                ('mistralai/Mistral-7B-Instruct-v0.2', 'movies_test'),
                ('mistralai/Mistral-7B-Instruct-v0.2', 'hotpotqa_with_context_test'),
                # -- Meta --
                ('meta-llama/Meta-Llama-3-8B-Instruct', 'imdb_test'),
                ('meta-llama/Meta-Llama-3-8B-Instruct', 'hotpotqa_test'),
                ('meta-llama/Meta-Llama-3-8B-Instruct', 'triviaqa_test'),
                ('meta-llama/Meta-Llama-3-8B-Instruct', 'movies_test'),
                ('meta-llama/Meta-Llama-3-8B-Instruct', 'hotpotqa_with_context_test')
            ]
            ,
            help="Test LLM/dataset combinations."
        )
    else:
        parser.add_argument(
            "--LLM",
            choices=LIST_OF_MODELS,
            default='meta-llama/Meta-Llama-3-8B-Instruct',
            help="LLM of which we train on."
        )
        
        parser.add_argument(
            "--train_dataset", 
            type=str,
            default="triviaqa",
            help="Train dataset."
        )
    
        parser.add_argument(
            "--test_dataset", 
            type=str,
            default="triviaqa_test",
            help="Test dataset."
        )

    if few_shot_adaptation:
        parser.add_argument(
            "--size_limit",
            type=int,
            default=1000,
            help="Number of train samples that we use for low data adaptation (max available is 10000)."
        )
        # checkpoint_path
        parser.add_argument(
            "--wandb_link",
            type=str,
            default="PLACEHOLDER",
            help="URL of the W&B sweep used to load the best-performing checkpoint based on validation AUC). The format shoule be: https://wandb.ai/<WANDB_USERNAME>/ACT-ViT/sweeps/<SWEEP_ID>"
        )
        
    parser.add_argument(
        "--base_raw_data_dir",
        type=str,
        default='raw_data',
        help="Base directory for the raw data."
    )  

    
    parser.add_argument(
        "--base_pre_processed_data_dir", 
        type=str, 
        default='preprocessed_data',
        help="Base directory for the pre-processed data."
    )
    
    parser.add_argument(
        "--probe_model", 
        choices=PROBING_ARCHITECTURES,
        default="ACT-Vit-foundation",
        help="The probing model to use (default: 'ACT-ViT-foundation', called 'ACT-ViT' in the paper)."
    )
    
    parser.add_argument(
        "--input_output_type", 
        type=str, 
        default="output",
        choices=["output"],
        help="A flag to indicate whether to preprocess the activation tensor for the input or output tokens. Here we only support 'output'."
    )
    
    parser.add_argument(
        "--patch_size",
        type=str,
        default="(1, 1)", 
        help="The patch size to use for the ViT model."
    )

    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=0,
        help="seed."
    )

    parser.add_argument(
        "--cuda_idx", 
        type=int, 
        default=0,
        help="cuda index."
    )
        
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=64,
        help="batch size."
    )
    
    parser.add_argument(
        "--hidden_dim", 
        type=int, 
        default=128,
        help="hidden dimension of ACT-ViT/ACT-MLP."
    )
    
    parser.add_argument(
        "--heads", 
        type=int, 
        default=4,
        help="number of heads."
    )
    
    parser.add_argument(
        "--dropout", 
        type=float, 
        default=0.3,
        help="dropout."
    )
    
    parser.add_argument(
        "--num_layers", 
        type=int, 
        default=3,
        help="number of layers to use (default: 1)."
    )
    
    parser.add_argument(
        "--pool", 
        type=str, 
        default='cls',
        help="pooling token."
    )
    
    parser.add_argument(
        "--patience", 
        type=int, 
        default=30,
        help="patience."
    )
    
        
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=15,
        help="number of epochs."
    )
    
    parser.add_argument(
        "--best_model_path", 
        type=str, 
        default="saved_models",
        help="Path where we save the best model."
    )
    
    parser.add_argument(
        "--lr", 
        type=float, 
        default=0.001,
        help="Learning rate."
    )
    

    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.0,
        help="Weight decay."
    )
    
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=4,
        help="Number of workers."
    )
    parser.add_argument(
        "--pin_memory", 
        type=int, 
        default=1,
        help="pin memory."
    )
    
    parser.add_argument(
        "--down_sample_strategy",
        type=str,
        default="pool",
        choices=["none", "pool"],
        help="Flag indicating which preprocessed activation tensor should be loaded, assuming the required files are available."
    )
    
    parser.add_argument(
        "--L_eff", 
        type=int, 
        default=8,
        help="L_p from the paper."
    )

    parser.add_argument(
        "--N_eff", 
        type=int, 
        default=100,
        help="N_p from the paper."
    )


    return parser.parse_args()
