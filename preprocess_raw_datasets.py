from utils.dataset_preprocess import preprocess_data
from utils.constants import N_MAX
from pathlib import Path
from utils.logger import get_logger
from utils.args import parse_args_pre_process


def process_datasets(args, logger):
    """Preprocesses datasets and loads them based on the task type."""
    
    logger.info(f"Initializing dataset processing, loading raw data, and saving the {args.input_output_type}.")
    logger.info(f"Loading raw data from: {args.base_raw_data_dir}")
    
    # Define output directories
    output_dir = Path(args.base_pre_processed_data_dir) / args.LLM / args.dataset

    logger.info(f"Preprocessed data will be stored at: {output_dir}")
    
    logger.info(f"Using {args.L_eff} layers and {args.N_eff} tokens for '{args.dataset}' vit the {args.down_sample_strategy} strategy.")

    
    logger.info(f"Starting data preparation for model '{args.LLM}' using training dataset '{args.dataset}'.")

    logger.info(f"Preprocessing dataset: {args.dataset}")
    preprocess_data(
        args=args,
        raw_data_dir=args.base_raw_data_dir,
        LLM=args.LLM,
        dataset_name=args.dataset,
        output_dir=output_dir,
        N_max=N_MAX,
        input_output_flag=args.input_output_type,
        logger=logger)
    logger.info("Dataset preprocessing completed successfully.")


    



if __name__ == "__main__":
    """Main function to preprocess data and load datasets based on task type."""
    
    # Parse command-line arguments
    args = parse_args_pre_process()
    
    # Sanitize model name by replacing forward slashes with hyphens for log file name
    sanitized_llm = args.LLM.replace("/", "-")
    logger = get_logger(f"preprocess_raw_datasets_of_{args.dataset}_from_{sanitized_llm}.log")
    
    logger.info("Starting the data processing pipeline.")
    logger.info(f"Parsed Arguments: {vars(args)}")
    
    logger.info(f"Preparing data for model '{args.LLM}' using training dataset '{args.dataset}'.")
    # Process datasets
    process_datasets(args, logger)
    
    
    