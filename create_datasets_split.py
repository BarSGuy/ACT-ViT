from utils.dataset_preprocess import stratified_split, get_train_val_test_indices
from pathlib import Path
import torch
from tqdm import tqdm

from utils.logger import get_logger
from utils.constants import LIST_OF_DATASETS, LIST_OF_MODELS
from utils.args import parse_args_create_datasets_splits
from utils.constants import TRAIN_DATASET_SIZE, RANDOM_STATE, SIZE_LIMITS, TRAIN_VAL_RATIO







def load_label(params):
    i, load_dir = params
    label_path = load_dir / f'label_{str(i)}.pt'
    label = torch.load(label_path)
    return label

# NOTE: this code assumes that all our train datasets are composed of 10000 sampels.
def collect_data(load_dir: Path, n_samples: int = TRAIN_DATASET_SIZE):
    labels = [load_label((i, load_dir)) for i in tqdm(range(n_samples), desc="Collecting labels")]
    return labels

### ------------------------------------------------------------- ###
### Main Function ###
### ------------------------------------------------------------- ###

def main():
    """Main function to create splits for all datasets and models."""

    args = parse_args_create_datasets_splits()
    
    # Setup logger
    logger = get_logger("create_datasets_split.log")
    logger.info(f"Parsed Arguments: {vars(args)}")
    
    # Convert paths to Path objects
    base_raw_data_dir = Path(args.base_raw_data_dir)
    output_base_dir = Path(args.output_base_dir)
    
    logger.info(f"Base raw data directory: {base_raw_data_dir}")
    logger.info(f"Output base directory: {output_base_dir}")
    
    # Determine which datasets and models to process
    datasets_to_process = LIST_OF_DATASETS
    models_to_process = LIST_OF_MODELS
    
    logger.info(f"Datasets to process: {datasets_to_process}")
    logger.info(f"Models to process: {models_to_process}")
    logger.info(f"Total combinations: {len(datasets_to_process) * len(models_to_process)}")
    

    for llm in models_to_process:
        for dataset in datasets_to_process:
            if "test" in dataset:
                logger.info(f"Skipping {llm} / {dataset} because it is a test dataset")
                continue
            logger.info(f"Processing: {llm} / {dataset}")
            logger.info("-" * 80)

            logger.info(f"Collecting data from {base_raw_data_dir / llm / dataset}")
            labels = collect_data(load_dir=base_raw_data_dir / llm / dataset)
            full_train_TRAIN_DATASET_SIZE = len(labels)
            logger.info(f"This (train) dataset size is {full_train_TRAIN_DATASET_SIZE}")
            
            # NOTE: using 5 folds and taking the first one, just to be consistant with previous work: https://github.com/BarSGuy/Beyond-next-token-probabilities
            splits = stratified_split([[l] for l in labels], percentage= 1 - TRAIN_VAL_RATIO, random_state=RANDOM_STATE)
            train_indices, val_indices, test_indices = get_train_val_test_indices(splits=splits)
            
            train_indices =  [train_indices[0], val_indices[0]]
            train_indices = [idx for sublist in train_indices for idx in sublist]
            val_indices = [test_indices[0]]
            val_indices = [idx for sublist in val_indices for idx in sublist]
            logger.info(f"Train size: {len(train_indices)}")
            logger.info(f"Val size: {len(val_indices)}")
            logger.info(f"Total size: {len(train_indices) + len(val_indices)}")
            assert len(train_indices) + len(val_indices) == full_train_TRAIN_DATASET_SIZE, "Train and val indices do not match the number of train and validation labels"
            train_set = set(train_indices)
            val_set = set(val_indices)
            assert len(train_set & val_set) == 0, f"Train and val indices have non-zero intersection: {train_set & val_set}"
            expected_union = set(range(full_train_TRAIN_DATASET_SIZE))
            actual_union = train_set | val_set
            assert actual_union == expected_union, f"Union of train and val indices does not equal all indices. Missing: {expected_union - actual_union}, Extra: {actual_union - expected_union}"


            logger.info(f"Saving train and val indices to {output_base_dir / llm / dataset}")
            save_dir = output_base_dir / llm / dataset
            save_dir.mkdir(parents=True, exist_ok=True)
            torch.save(train_indices, save_dir / "train_indices.pt")
            torch.save(val_indices, save_dir / "val_indices.pt")
            
            assert full_train_TRAIN_DATASET_SIZE == TRAIN_DATASET_SIZE, "Full train dataset size is not 10000"
            size_limits = SIZE_LIMITS
            for size_limit in size_limits:
                logger.info(f"----> Processing size limit: {size_limit}")
                logger.info(f"Saving train and val indices to {output_base_dir / llm / dataset / f'size_limit_{size_limit}'}")
                save_dir = output_base_dir / llm / dataset / f'size_limit_{size_limit}'
                save_dir.mkdir(parents=True, exist_ok=True)
                assert size_limit <= full_train_TRAIN_DATASET_SIZE, "Size limit is greater than the number of train and validation labels"
 
                train_labels_for_size_limit = [labels[i] for i in train_indices]
                val_labels_for_size_limit = [labels[i] for i in val_indices]

                
                train_splits = stratified_split([[l] for l in tqdm(train_labels_for_size_limit, desc="Creating train splits")], percentage=TRAIN_VAL_RATIO*(size_limit)/len(train_labels_for_size_limit), random_state=RANDOM_STATE)
                train_indices_for_size_limit = [train_indices[i] for i in train_splits[0]]
                val_splits = stratified_split([[l] for l in tqdm(val_labels_for_size_limit, desc="Creating val splits")], percentage=(1-TRAIN_VAL_RATIO)*(size_limit)/len(val_labels_for_size_limit), random_state=RANDOM_STATE)
                val_indices_for_size_limit = [val_indices[i] for i in val_splits[0]]
                
                
                train_set_size_limit_set = set(train_indices_for_size_limit)
                val_set_size_limit_set = set(val_indices_for_size_limit)
                assert len(train_set_size_limit_set & val_set_size_limit_set) == 0, f"Train and val indices for size_limit {size_limit} have non-zero intersection: {train_set_size_limit_set & val_set_size_limit_set}"
    
                assert len(train_set_size_limit_set | val_set_size_limit_set) == size_limit, f"Union of train and val indices for size_limit {size_limit} does not equal size_limit: {len(train_set_size_limit_set | val_set_size_limit_set)} != {size_limit}"


                torch.save(train_indices_for_size_limit, save_dir / "train_indices.pt")
                torch.save(val_indices_for_size_limit, save_dir / "val_indices.pt")

if __name__ == '__main__':
    main()
