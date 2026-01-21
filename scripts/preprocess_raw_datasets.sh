#!/bin/bash


# Default base directory for raw data
DEFAULT_BASE_RAW_DATA_DIR="raw_data"
# Default base directory for pre-processed data
DEFAULT_BASE_PRE_PROCESSED_DATA_DIR="preprocessed_data"
DEFAULT_MAX_PARALLEL_JOBS=3

# Use provided arguments as base directories, otherwise use defaults
BASE_RAW_DATA_DIR=${1:-$DEFAULT_BASE_RAW_DATA_DIR}
BASE_PRE_PROCESSED_DATA_DIR=${2:-$DEFAULT_BASE_PRE_PROCESSED_DATA_DIR}

# Allow specifying the number of parallel chunks (default to 8)
MAX_PARALLEL_JOBS=${3:-$DEFAULT_MAX_PARALLEL_JOBS}

# Define datasets
DATASETS=("imdb" "imdb_test" "movies" "movies_test" "hotpotqa" "hotpotqa_test" "hotpotqa_with_context" "hotpotqa_with_context_test" "triviaqa" "triviaqa_test")

STRATEGIES=("pool")

# Define models for each dataset
declare -A MODELS
MODELS[imdb]="mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen2.5-7B-Instruct"
MODELS[imdb_test]="mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen2.5-7B-Instruct"
MODELS[movies]="mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen2.5-7B-Instruct"
MODELS[movies_test]="mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen2.5-7B-Instruct"
MODELS[hotpotqa]="mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen2.5-7B-Instruct"
MODELS[hotpotqa_test]="mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen2.5-7B-Instruct"
MODELS[triviaqa]="mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen2.5-7B-Instruct"
MODELS[triviaqa_test]="mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen2.5-7B-Instruct"
MODELS[hotpotqa_with_context]="mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen2.5-7B-Instruct"
MODELS[hotpotqa_with_context_test]="mistralai/Mistral-7B-Instruct-v0.2 meta-llama/Meta-Llama-3-8B-Instruct Qwen/Qwen2.5-7B-Instruct"



declare -A INPUT_OUTPUT_TYPES
INPUT_OUTPUT_TYPES[imdb]="output"
INPUT_OUTPUT_TYPES[imdb_test]="output"
INPUT_OUTPUT_TYPES[movies]="output"
INPUT_OUTPUT_TYPES[movies_test]="output"
INPUT_OUTPUT_TYPES[hotpotqa]="output"
INPUT_OUTPUT_TYPES[hotpotqa_test]="output"
INPUT_OUTPUT_TYPES[triviaqa]="output"
INPUT_OUTPUT_TYPES[triviaqa_test]="output"
INPUT_OUTPUT_TYPES[hotpotqa_with_context]="output"
INPUT_OUTPUT_TYPES[hotpotqa_with_context_test]="output"


# Track running jobs
RUNNING_JOBS=0


echo "Starting dataset preprocessing process..."
echo "---------------------------------------------"
echo "Datasets: ${DATASETS[*]}"
echo "Models: ${MODELS[*]}"
echo "Types: ${INPUT_OUTPUT_TYPES[*]}"
echo "---------------------------------------------"


# Loop through datasets and models
for DATASET in "${DATASETS[@]}"; do
  for MODEL in ${MODELS[$DATASET]}; do
    for STRATEGY in "${STRATEGIES[@]}"; do
      printf "Running preprocessing for dataset %s with model %s and strategy %s...\n" "$DATASET" "$MODEL" "$STRATEGY"
      python preprocess_raw_datasets.py \
        --LLM "$MODEL" \
        --dataset "$DATASET" \
        --base_raw_data_dir "$BASE_RAW_DATA_DIR" \
        --base_pre_processed_data_dir "$BASE_PRE_PROCESSED_DATA_DIR" \
        --input_output_type "${INPUT_OUTPUT_TYPES[$DATASET]}" \
        --down_sample_strategy "$STRATEGY" \
        --N_eff 100 \
        --L_eff 8 &

      ((RUNNING_JOBS++))

      # If the number of jobs reaches the limit, wait for the first one to finish
      if ((RUNNING_JOBS >= MAX_PARALLEL_JOBS)); then
        wait -n  # Waits for ANY one job to finish
        ((RUNNING_JOBS--))  # Reduce the running jobs counter
        printf "Finished preprocessing for dataset %s with model %s...\n" "$DATASET" "$MODEL"
      fi
    done
  done
done

# Ensure any remaining processes finish
wait

echo "All preprocessing tasks have been completed successfully."
