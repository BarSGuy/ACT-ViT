from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_model_and_validate_gpu(model_path, tokenizer_path=None):
    if tokenizer_path is None:
        tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print("Started loading model")
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto',
                                                 torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, output_hidden_states=True)
    assert ('cpu' not in model.hf_device_map.values())
    return model, tokenizer