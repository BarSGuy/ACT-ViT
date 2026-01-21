from pathlib import Path
from typing import List
import torch.nn.functional as F
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import multiprocessing
from sklearn.model_selection import StratifiedKFold
import numpy as np
from utils.constants import FEATURE_DIMS
from torch.utils.data import Dataset


def stratified_split(dataset, percentage=1/5, random_state=42):
    """
    Splits the dataset into stratified chunks based on the given percentage.

    Args:
        dataset The dataset object.
        percentage (float): Fraction representing the split size (e.g., 1/5).
        random_state (int): Random state for reproducibility.

    Returns:
        list of arrays: List of index arrays corresponding to the splits.
    """
    if not (0 < percentage <= 1):
        raise ValueError("Percentage must be a value between 0 and 1.")

    # Calculate the number of splits
    num_splits = int(1 / percentage)

    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Cannot perform stratified split.")

    # Extract labels from the dataset for stratification
    labels = [dataset[i][-1] for i in tqdm(range(len(dataset)), desc="Extracting labels")]

    # Perform stratified splitting
    skf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    splits = [indices for _, indices in skf.split(np.zeros(len(labels)), labels)]

    return splits
    
def get_train_val_test_indices(splits):
    """
    Generates train, validation, and test indices from stratified splits.

    Args:
        splits (list of arrays): List of index arrays corresponding to the stratified splits.

    Returns:
        tuple: Three lists containing train_indices, val_indices, and test_indices.
    """
    num_chunks = len(splits)
    train_indices = []
    val_indices = []
    test_indices = []

    for i in tqdm(range(num_chunks), desc="Processing folds"):
        train_chunks = [splits[j] for j in range(num_chunks) if j != i and j != (i + 1) % num_chunks]
        val_chunk = splits[(i + 1) % num_chunks]
        test_chunk = splits[i]

        train_indices.append(np.concatenate(train_chunks))
        val_indices.append(val_chunk)
        test_indices.append(test_chunk)

    return train_indices, val_indices, test_indices


    
def pad_activations_tensor(tensor, L_max, N_max, pad_value):
    """
    Pads a given [L, N, d] tensor to [L_max, N_max, d] with a specified value.

    Parameters:
        tensor (torch.Tensor): The input tensor of shape [L, N, d].
        L_max (int): The desired number of layers.
        N_max (int): The desired sequence length.
        pad_value (float or int): The value to pad with.

    Returns:
        torch.Tensor: The padded tensor of shape [L_max, N_max, d].
    """
    # Get the current shape of the tensor
    L, N, d = tensor.shape
    
    # Create the padded tensor filled with pad_value
    padded_tensor = torch.full((L_max, N_max, d), pad_value, dtype=tensor.dtype, device=tensor.device)
    
    # Determine the valid slice range for copying the original tensor
    L_copy = min(L, L_max)
    N_copy = min(N, N_max)
    
    # Copy the relevant part of the tensor into the padded tensor
    padded_tensor[:L_copy, :N_copy, :] = tensor[:L_copy, :N_copy, :]
    
    return padded_tensor
    
def pad_tensor(tensor, N_max, pad_value):
    """
    Pads a given [N, d] tensor to [N_max, d] with a specified value.

    Parameters:
        tensor (torch.Tensor): The input tensor of shape [N, d].
        N_max (int): The desired number of rows (N_max).
        pad_value (float or int): The value to pad with.

    Returns:
        torch.Tensor: The padded tensor of shape [N_max, d].
    """
    # Get the current shape of the tensor
    N, d = tensor.shape

    # Ensure N_max is not smaller than N
    if N_max <= N:
        # Slice the tensor if N_max is less than or equal to N
        return tensor[:N_max, :]

    # Create the padded tensor with the padding value
    padded_tensor = torch.full((N_max, d), pad_value, dtype=tensor.dtype, device=tensor.device)

    # Copy the original tensor into the padded tensor
    padded_tensor[:N, :] = tensor

    return padded_tensor


def patch_down_sample(tensor: torch.Tensor, L_new: int, N_new: int, method: str = "interpolation"):
    """
    Transforms a tensor of shape [L, N, d] to [L_new, N_new, d] using the specified method.
    If necessary, the tensor is padded before applying pooling.

    Parameters:
        tensor (torch.Tensor): Input tensor of shape [L, N, d].
        L_new (int): Target height.
        N_new (int): Target width.
        method (str): Transformation method, one of ['sum_pool', 'mean_pool', 'interpolation'].

    Returns:
        torch.Tensor: Transformed tensor of shape [L_new, N_new, d].
    """
    L, N, d = tensor.shape
    tensor = tensor.permute(2, 0, 1)  # Change to [d, L, N] for processing
    
    if method in ["sum_pool", "mean_pool", "max_pool"]:
        # Compute the required padding to make L and N divisible by L_new and N_new
        L_pad = (L_new - (L % L_new)) % L_new + L
        N_pad = (N_new - (N % N_new)) % N_new + N

        # Pad the tensor
        pad_L = L_pad - L
        pad_N = N_pad - N
        tensor_pad = F.pad(tensor, (0, pad_N, 0, pad_L), mode="replicate")  # Pad along spatial dims

        # Compute new dimensions
        factor_L = L_pad // L_new
        factor_N = N_pad // N_new

        # Reshape into patches and apply pooling
        tensor_pad = tensor_pad.unfold(1, factor_L, factor_L).unfold(2, factor_N, factor_N)
        if method == "sum_pool":
            tensor_pad = tensor_pad.sum(dim=(-1, -2))  # Sum pooling
        elif method == "mean_pool":
            tensor_pad = tensor_pad.mean(dim=(-1, -2))  # Mean pooling
        elif method == "max_pool":
            tensor_pad = tensor_pad.max(dim=-1)[0].max(dim=-1)[0]  # Max pooling



    else:
        raise ValueError("Unsupported method. Choose from 'sum_pool', 'mean_pool', or 'interpolation'.")
    
    return tensor_pad.permute(1, 2, 0)  # Back to [L_new, N_new, d]



def process_file(params):
    file_idx, data_dir, output_dir, N_max, input_output_flag, down_sample_strategy, args, logger = params
    
    # Load label
    label_path = os.path.join(data_dir, f'label_{file_idx}.pt')
    
    if os.path.exists(label_path):
        label = torch.load(label_path)

    
    output_paths = [      
        os.path.join(output_dir, f'label_{file_idx}.pt'),
    ]
    

    if down_sample_strategy == "pool":
        output_paths.append(os.path.join(output_dir, f"activations/{down_sample_strategy}", f"activations_{input_output_flag}_||Downsample__L_{str(args.L_eff)}_N_{str(args.N_eff)}||{file_idx}.pt"))
    elif down_sample_strategy == "none":
        output_paths.append(os.path.join(output_dir, f"activations/{down_sample_strategy}", f"activations_{input_output_flag}_{file_idx}.pt"))
    else:
        raise ValueError("Unsupported down_sample_strategy. Choose from 'pool', 'logistic_regression', or 'none'.")

    




    if (not os.path.exists(output_paths[-1])):
        activations_path = os.path.join(data_dir, f'activations_{input_output_flag}_{file_idx}.pt')
        activations = torch.load(activations_path)
        
        if down_sample_strategy == "pool":
            # setting L_for_pad
            # NOTE: potentially change the row below to be with the if statement as well (as for N)
            L_for_pad = (int(activations.shape[0] / args.L_eff) + 1) * args.L_eff
            if (N_max % args.N_eff) != 0:
                N_for_pad = (int(N_max / args.N_eff) + 1) * args.N_eff 
            else:
                N_for_pad = N_max
            
            
            activations = pad_activations_tensor(activations, L_max=L_for_pad, N_max=N_for_pad, pad_value=0)
            activations = patch_down_sample(activations, L_new=args.L_eff, N_new=args.N_eff, method='max_pool')                   
        elif down_sample_strategy == "none":
            activations = pad_activations_tensor(activations, L_max=activations.shape[0], N_max=N_max, pad_value=0)

        else:
            raise ValueError("Unsupported down_sample_strategy. Choose from 'pool', 'logistic_regression', or 'none'.")
        
        act_dir = os.path.join(output_dir, f"activations/{down_sample_strategy}")
        os.makedirs(act_dir, exist_ok=True)  # Creates "act" if it doesn't exist    
        
        torch.save(activations.to(torch.float16), output_paths[1])
        
    else:
        print(f"Activations data for Sample {file_idx} already exists or we don't need it, skipping.")
    torch.save(label, output_paths[0])


def preprocess_data(args, raw_data_dir, LLM, dataset_name, output_dir, N_max=100, input_output_flag = 'input', logger=None):
    assert input_output_flag in ['input', 'output'], "input_output_flag must be 'input' or 'output'"
    data_dir = os.path.join(raw_data_dir, LLM, dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    
    files = os.listdir(data_dir)
    file_indices = sorted(
        set(int(f.split('_')[-1].split('.')[0]) for f in files)
    )
    logger.info(f"Found {len(file_indices)} files to process.")
    # Create parameter list for each file to process
    params_list = [(idx, data_dir, output_dir, N_max, input_output_flag, 
                   args.down_sample_strategy, args, logger) for idx in file_indices]

        
    num_workers = min(int(0.2*multiprocessing.cpu_count()), len(file_indices))
    # NOTE: for debugging purposes, set num_workers=1 as below
    # num_workers = 1
    with multiprocessing.Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_file, params_list), total=len(file_indices), desc="Preprocessing files"))

    
class CustomSavedDataset(Dataset):
    def __init__(self, preprocessed_dir, input_output_flag='output', L_eff=8, N_eff=100, down_sample_strategy="pool"):
        """
        Initialize the dataset.

        Args:
            base_dir (str): Base directory containing the saved data.
            LLM (str): Name of the language model directory.
            dataset_name (str): Name of the dataset directory.
        """
        self.preprocessed_dir = preprocessed_dir
        self.file_indices = self._get_indices()
        self.L_eff = L_eff
        self.N_eff = N_eff
        self.file_paths = {}
        for idx in self.file_indices:
            base_dict = {
                'label': os.path.join(self.preprocessed_dir, f'label_{idx}.pt'),
            }

            if down_sample_strategy == "pool":
                base_dict['activations'] = os.path.join(
                    self.preprocessed_dir, f"activations/{down_sample_strategy}",
                    f"activations_{input_output_flag}_||Downsample__L_{str(self.L_eff)}_N_{str(self.N_eff)}||{idx}.pt"
                )
                
            elif down_sample_strategy == "none":
                base_dict['activations'] = os.path.join(
                    self.preprocessed_dir, f"activations/{down_sample_strategy}",
                    f"activations_{input_output_flag}_{idx}.pt"
                )
            else:
                raise ValueError("Unsupported down_sample_strategy. Choose from 'pool', 'logistic_regression', or 'none'.")

            self.file_paths[idx] = base_dict

    def _get_indices(self):
        """Find all unique indices by looking for logits_canonized files."""
        files = os.listdir(self.preprocessed_dir)
        indices = sorted(
            set(int(f.split('_')[-1].split('.')[0])
                for f in files if f.startswith('') and 'label' in f)
        )
        return indices

    def __len__(self):
        """Return the number of samples."""
        return len(self.file_indices)

    def __getitem__(self, idx):
        """
        Load and return the sample at the given index.

        Args:
            idx (int): Index of the sample to load.

        Returns:
            dict: A dictionary with 'logits_canonized', 'label', 
                  'activations_input', and optionally 'activations_output'.
        """
     

        file_idx = self.file_indices[idx]
        paths = self.file_paths[file_idx]


        label = torch.load(paths['label'], weights_only=False) # relevant for torch > 2.6

        try: 
            activations = torch.load(paths['activations'])
        except FileNotFoundError:
            print(f"File not found: {paths['activations']}. Maybe {self.L_eff=} or {self.N_eff=} is the problem?")
            raise
        return (activations, label)
        
        

class CombinedCustomDataset(Dataset):
    def __init__(self, datasets: List[CustomSavedDataset]):
        """
        Combines multiple CustomSavedDataset objects into a single dataset.
        
        Args:
            datasets (list): List of CustomSavedDataset objects to combine
        """
        self.datasets = datasets

        
        # Calculate cumulative lengths for indexing
        self.cumulative_lengths = [0]
        for dataset in datasets:
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + len(dataset))
        
        # Extract origin information (LLM and dataset type) from each dataset
        self.origins = [f"{Path(dataset.preprocessed_dir).parts[-2]}_{Path(dataset.preprocessed_dir).parts[-1]}" for dataset in datasets]

        self.llms = list(set(Path(dataset.preprocessed_dir).parts[-2] for dataset in datasets))
        # NOTE: self.d_max is the maximum feature dimension of FEATURE_DIMS in constants. For LLMs that their feature dimention is less than self.d_max, we pad with 0.
        self.d_max = max(FEATURE_DIMS.values())
    
    def __len__(self):
        """Return the total length of the combined dataset."""
        return self.cumulative_lengths[-1]
    def add_indices(self, indices: List[List[int]], dataset_idx: int):
        """
        Add indices to the dataset.

        Args:
            indices (List[List[int]]): A list containing two lists of indices. 
            list[0] = train_indices (single list, no folds)
            list[1] = val_indices (single list, no folds)
        """
        if not hasattr(self, 'indices'):
            self.indices = [[], []]  # train and val only, no folds
        offset = self.cumulative_lengths[dataset_idx]
        for i, split_indices in enumerate(indices):
            offset_indices = [idx + offset for idx in split_indices]
            self.indices[i].extend(offset_indices)
                
    def get_combined_indices(self, split: str = None):
        """
        Get the combined indices for the specified split.
        
        Args:
            split (str): Either 'train' or 'val'
            
        Returns:
            List[int]: List of indices for the specified split
        """
        split_map = {'train': 0, 'val': 1}
        if split not in split_map:
            raise ValueError("Invalid split type. Choose from 'train' or 'val'.")
        
        split_idx = split_map[split]
        return self.indices[split_idx]
    
        
    def __getitem__(self, idx):
        """
        Get an item from the appropriate dataset based on the index.
        
        Args:
            idx (int): Global index into the combined dataset
            
        Returns:
            tuple: (origin, items) where origin is (llm, dataset_type) and items is the data from the dataset
        """
        # Find which dataset the index belongs to
        dataset_idx = 0
        while idx >= self.cumulative_lengths[dataset_idx + 1]:
            dataset_idx += 1
        
        # Calculate the local index within the selected dataset
        local_idx = idx - self.cumulative_lengths[dataset_idx]
        
        # Get the origin information for this dataset
        origin = self.origins[dataset_idx]
        
        # Get the item from the dataset
        (item, label) = self.datasets[dataset_idx][local_idx]
        
        # pad item to the max sequence length
        N, L, d = item.shape
        padded_item = torch.full((N, L, self.d_max), 0, dtype=item.dtype, device=item.device)
        padded_item[..., :d] = item
        item = padded_item
        # Return the origin and item
        return origin, (item, label)