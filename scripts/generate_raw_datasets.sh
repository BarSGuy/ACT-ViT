#!/bin/bash

# sleep 10100 && ./scripts/generate_raw_datasets.sh

# Default base directory for raw data
DEFAULT_BASE_RAW_DATA_DIR="raw_data"
DEFAULT_MAX_PARALLEL_JOBS=3

# Use provided argument as base directory, otherwise use default
BASE_RAW_DATA_DIR=${1:-$DEFAULT_BASE_RAW_DATA_DIR}

# Allow specifying the number of parallel chunks (default to 8)
MAX_PARALLEL_JOBS=${2:-$DEFAULT_MAX_PARALLEL_JOBS}

# Define datasets
DATASETS=("imdb" "imdb_test" "movies" "movies_test" "hotpotqa" "hotpotqa_test" "hotpotqa_with_context" "hotpotqa_with_context_test" "triviaqa" "triviaqa_test")

# Define models for each dataset
MODELS=("mistralai/Mistral-7B-Instruct-v0.2" "meta-llama/Meta-Llama-3-8B-Instruct" "Qwen/Qwen2.5-7B-Instruct")

# Define chunks
CHUNKS=(1 2 3 4 5 6 7 8 9 10)

# Track running jobs
RUNNING_JOBS=0


echo "Starting dataset creation process..." 
echo "---------------------------------------------"
echo "Datasets: ${DATASETS[*]}"
echo "Models: ${MODELS[*]}"
echo "Chunks: ${CHUNKS[*]}"
echo "---------------------------------------------"



# Loop through datasets and models
for DATASET in "${DATASETS[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    for CHUNK in "${CHUNKS[@]}"; do
      printf "Running chunk %d of dataset %s with model %s...\n" "$CHUNK" "$DATASET" "$MODEL"

      # Run the process in the background and log both stdout and stderr
      python create_raw_datasets.py \
        --dataset "$DATASET" \
        --LLM "$MODEL" \
        --chunk "$CHUNK" \
        --base_raw_data_dir "$BASE_RAW_DATA_DIR" \
        --n_samples 10000 &

      ((RUNNING_JOBS++))

      # If the number of jobs reaches the limit, wait for the first one to finish
      if ((RUNNING_JOBS >= MAX_PARALLEL_JOBS)); then
        wait -n  # Waits for ANY one job to finish
        ((RUNNING_JOBS--))  # Reduce the running jobs counter
        printf "Chunk %d of dataset %s with model %s has finished.\n" "$CHUNK" "$DATASET" "$MODEL"
      fi
    done
  done
done

# Ensure any remaining processes finish
wait

echo "All datasets have been created successfully."
