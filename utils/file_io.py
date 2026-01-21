import os
import torch
from pathlib import Path


def save_raw_data(
    LLM: str,
    dataset_name: str,
    idx: int,
    label: torch.Tensor,
    activations_input: torch.Tensor = None,
    activations_output: torch.Tensor = None,
    base_dir: str = "./data",
    prompt: str = None,
    answer: str = None
):
    """
    Save model results (PyTorch tensors) and labels to a specified directory structure.

    Args:
        LLM (str): Name of the language model.
        dataset_name (str): Name of the dataset.
        label (torch.Tensor): Label data to save.
        idx (int): Index of the sample, used for file naming.
        activations_input (torch.Tensor): Input activations tensor to save.
        activations_output (torch.Tensor, optional): Output activations tensor to save.
        base_dir (str, optional): Base directory for saving results (default: './data').
        prompt (str): Prompt to save.
        answer (str): Answer to save.
    """

    # Construct save directory path
    save_dir = Path(base_dir) / LLM / dataset_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Function to save tensor safely
    def save_tensor(tensor, filename):
        if tensor is not None:
            try:
                torch.save(tensor, save_dir / filename)
            except Exception as e:
                print(f"Error saving {filename}: {e}")

    # Save label
    save_tensor(label, f'label_{idx}.pt')

    # Save activations
    if activations_input is not None:
        save_tensor(activations_input, f'activations_input_{idx}.pt')
        
    save_tensor(activations_output, f'activations_output_{idx}.pt')
    
    # Save txt file with prompt, response, and label
    if prompt is not None or answer is not None:
        label_value = label.item() if isinstance(label, torch.Tensor) and label.numel() == 1 else label
        txt_file_path = save_dir / f'data_{idx}.txt'
        with open(txt_file_path, 'w', encoding='utf-8') as f:
            f.write(f"prompt: {prompt or 'N/A'}\n")
            f.write(f"response: {answer or 'N/A'}\n")
            f.write(f"label: {label_value}\n")
