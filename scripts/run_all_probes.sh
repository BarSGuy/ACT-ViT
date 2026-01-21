#!/bin/bash

# ./scripts/get_all_token_layer_scores.sh

# Default base directory for raw data
DEFAULT_BASE_RAW_DATA_DIR='raw_data'

# Use provided argument as base directory, otherwise use default
BASE_RAW_DATA_DIR=${1:-$DEFAULT_BASE_RAW_DATA_DIR}

# Define datasets
DATASETS=("imdb" "movies" "hotpotqa" "triviaqa" "hotpotqa_with_context")

# Define models for each dataset
declare -A MODELS

MODELS[imdb]="mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen2.5-7B-Instruct"
MODELS[movies]="mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen2.5-7B-Instruct"
MODELS[hotpotqa]="mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen2.5-7B-Instruct"
MODELS[triviaqa]="mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen2.5-7B-Instruct"
MODELS[hotpotqa_with_context]="mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen2.5-7B-Instruct"



echo "Starting to analyze token layer scores..."
echo "---------------------------------------------"
echo "Datasets: ${DATASETS[*]}"
echo "Models: ${MODELS[*]}"
echo "---------------------------------------------"



# Loop through datasets and models

for DATASET in "${DATASETS[@]}"; do
  for MODEL in ${MODELS[$DATASET]}; do
    printf "Running dataset %s over %s with model %s...\n" "$DATASET" "$MODEL"
    # Run the process in the background and log both stdout and stderr
    python probing.py \
      --dataset "$DATASET" \
      --LLM "$MODEL" \
      --base_raw_data_dir "$BASE_RAW_DATA_DIR"
  done
done


echo "All token layer scores have been computed sucessfully."
