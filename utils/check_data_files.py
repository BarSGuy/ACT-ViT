
import os
from tqdm import tqdm
import csv
import argparse
from pathlib import Path

NUM_SAMPLES = {
    "hotpotqa": 10000,
    "hotpotqa_test": 10000,
    "hotpotqa_with_context": 10000,
    "hotpotqa_with_context_test": 10000,
    "imdb": 10000,
    "imdb_test": 10000,
    "movies": 10000,
    "movies_test": 7856,
    "triviaqa": 10000,
    "triviaqa_test": 10000
}

def get_chunk_from_index(index):
    """
    Determine which chunk an index belongs to.
    Chunks are defined as:
    - Chunk 1: 0-999
    - Chunk 2: 1000-1999
    - Chunk 3: 2000-2999
    - Chunk 4: 3000-3999
    - Chunk 5: 4000-4999
    - Chunk 6: 5000-5999
    - Chunk 7: 6000-6999
    - Chunk 8: 7000-7999
    - Chunk 9: 8000-8999
    - Chunk 10: 9000-9999
    """
    return (index // 1000) + 1

def get_chunks_from_indices(indices):
    """
    Get the set of chunks that need to be rerun based on missing indices.
    """
    chunks = set()
    for idx in indices:
        chunks.add(get_chunk_from_index(idx))
    return sorted(chunks)

               

def check_raw_data_exists_and_save_csv(
    csv_path="missing_raw_data_summary.csv",
    root_dir_qwen="raw_data/Qwen/Qwen2.5-7B-Instruct",
    root_dir_mistral="raw_data/mistralai/Mistral-7B-Instruct-v0.2",
    root_dir_meta="raw_data/meta-llama/Meta-Llama-3-8B-Instruct"
):
    root_dirs = [root_dir_qwen, root_dir_mistral, root_dir_meta]
    subdirs = [
        "hotpotqa",
        "hotpotqa_test",
        "hotpotqa_with_context",
        "hotpotqa_with_context_test",
        "imdb",
        "imdb_test",
        "movies",
        "movies_test",
        "triviaqa",
        "triviaqa_test"
    ]

    # Prepare list of rows for the CSV
    csv_rows = [("Root Directory", "Subdir", "Status", "Missing Count", "Chunks to Rerun")]

    for root_dir in root_dirs:
        print(f"\nChecking for subdirs in {root_dir}:\n\n")

        missing_labels = {}

        for subdir in subdirs:
            full_path = os.path.join(root_dir, subdir)
            if not os.path.exists(full_path):
                missing_labels[subdir] = "Directory does not exist"
                csv_rows.append((root_dir, subdir, "Directory does not exist", "", ""))
                continue

            missing = []
            missing_indices = []
            
            for i in tqdm(range(NUM_SAMPLES[subdir]), desc=subdir):
                file_name = f"activations_output_{i}.pt"
                if not os.path.exists(os.path.join(full_path, file_name)):
                    missing.append(file_name)
                    missing_indices.append(i)

            if missing:
                missing_labels[subdir] = missing
                chunks_to_rerun = get_chunks_from_indices(missing_indices)
                chunks_str = ", ".join(map(str, chunks_to_rerun))
                csv_rows.append((root_dir, subdir, "Missing files", len(missing), chunks_str))
            else:
                missing_labels[subdir] = "✅ All labels present"
                csv_rows.append((root_dir, subdir, "✅ All labels present", 0, ""))

        # Pretty print the results
        print("\nSummary of Missing Labels:\n" + "=" * 30)
        for subdir, result in missing_labels.items():
            print(f"\n{subdir}:")
            if isinstance(result, list):
                print(f"  Missing {len(result)} labels:")
                if subdir == "movies_test":
                    print("Note -- here we have less than 10_000 labels! so it might be fine")
                print("   ", ", ".join(result[:10]), end="")
                if len(result) > 10:
                    print(f", ... (+{len(result) - 10} more)")
                else:
                    print()
            else:
                print(f"  {result}")

    # Write to CSV
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    print(f"\n✅ Results saved to: {csv_path}")




def check_preprocess_data_exists(
    type="pool",
    output_csv="preprocessed_missing_labels_summary",
    root_dir_qwen="preprocessed_data/Qwen/Qwen2.5-7B-Instruct",
    root_dir_mistral="preprocessed_data/mistralai/Mistral-7B-Instruct-v0.2",
    root_dir_meta="preprocessed_data/meta-llama/Meta-Llama-3-8B-Instruct"
):
    output_csv = output_csv + f"_{type}.csv"
    root_dirs = [root_dir_qwen, root_dir_mistral, root_dir_meta]
    subdirs = [
        "hotpotqa", "hotpotqa_test", "hotpotqa_with_context", "hotpotqa_with_context_test",
        "imdb", "imdb_test", "movies", "movies_test", "triviaqa", "triviaqa_test"
    ]

    # Collect all summary data here
    csv_data = [("Root Directory", "Subdir", "Status", "Missing Count", "Sample Missing Files")]

    for root_dir in root_dirs:
        print(f"\nChecking for subdirs in {root_dir}:\n\n")
        missing_labels = {}

        for subdir in subdirs:
            full_path = os.path.join(root_dir, subdir, "activations", "pool")
            if not os.path.exists(full_path):
                status = "❌ Directory does not exist"
                missing_labels[subdir] = status
                csv_data.append((root_dir, subdir, status, "", ""))
                continue

            missing = []
            for i in tqdm(range(NUM_SAMPLES[subdir]), desc=f"Checking {root_dir}/{subdir}"):
                file_name = f"activations_output_||Downsample__L_8_N_100||{i}.pt"
                if not os.path.exists(os.path.join(full_path, file_name)):
                    missing.append(file_name)

            if missing:
                status = "⚠️ Missing files"
                sample = ", ".join(missing[:10])
                if len(missing) > 10:
                    sample += f", ... (+{len(missing) - 10} more)"
                csv_data.append((root_dir, subdir, status, len(missing), sample))
                missing_labels[subdir] = missing
            else:
                status = "✅ All labels present"
                csv_data.append((root_dir, subdir, status, 0, ""))
                missing_labels[subdir] = status

        # Print summary
        print("\nSummary of Missing Labels:\n" + "=" * 30)
        for subdir, result in missing_labels.items():
            print(f"\n{subdir}:")
            if isinstance(result, list):
                print(f"  Missing {len(result)} labels:")
                if subdir == "movies_test":
                    print("Note -- here we have less than 10,000 labels! So it might be fine.")
                print("   ", ", ".join(result[:10]), end="")
                if len(result) > 10:
                    print(f", ... (+{len(result) - 10} more)")
                else:
                    print()
            else:
                print(f"  {result}")

    # Write to CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(csv_data)

    print(f"\n✅ CSV summary saved to: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check data files existence and generate summary CSV")
    parser.add_argument(
        "--function",
        choices=["raw", "pool", "LG"],
        help="Which function to run: 'raw' for raw data check, 'pool' or 'LG' for preprocessed data check"
    )
    parser.add_argument(
        "--root_dir_qwen",
        type=str,
        default="raw_data/Qwen/Qwen2.5-7B-Instruct",
        help="Root directory for Qwen model raw data"
    )
    parser.add_argument(
        "--root_dir_mistral",
        type=str,
        default="raw_data/mistralai/Mistral-7B-Instruct-v0.2",
        help="Root directory for Mistral model raw data"
    )
    parser.add_argument(
        "--root_dir_meta",
        type=str,
        default="raw_data/meta-llama/Meta-Llama-3-8B-Instruct",
        help="Root directory for Meta-Llama model raw data"
    )
    parser.add_argument(
        "--preprocessed_root_dir_qwen",
        type=str,
        default="/home/guy_b/ACT-ViT/pre_processed_data/Qwen/Qwen2.5-7B-Instruct",
        help="Root directory for Qwen model preprocessed data"
    )
    parser.add_argument(
        "--preprocessed_root_dir_mistral",
        type=str,
        default="/home/guy_b/ACT-ViT/pre_processed_data/mistralai/Mistral-7B-Instruct-v0.2",
        help="Root directory for Mistral model preprocessed data"
    )
    parser.add_argument(
        "--preprocessed_root_dir_meta",
        type=str,
        default="/home/guy_b/ACT-ViT/pre_processed_data/meta-llama/Meta-Llama-3-8B-Instruct",
        help="Root directory for Meta-Llama model preprocessed data"
    )

    args = parser.parse_args()
    
    # Get project root directory (parent of utils directory)
    PROJECT_ROOT = Path(__file__).parent.parent
    # Ensure data folder exists in project root
    data_dir = PROJECT_ROOT / "data"
    os.makedirs(data_dir, exist_ok=True)
    
    if args.function == "raw":
        csv_path = data_dir / "missing_raw_data_summary.csv"
        check_raw_data_exists_and_save_csv(
            csv_path=str(csv_path),
            root_dir_qwen=args.root_dir_qwen,
            root_dir_mistral=args.root_dir_mistral,
            root_dir_meta=args.root_dir_meta
        )
    elif args.function in ["pool"]:
        output_csv = str(data_dir / "missing_pooled_data_summary.csv")
        check_preprocess_data_exists(
            type=args.function,
            output_csv=output_csv,
            root_dir_qwen=args.preprocessed_root_dir_qwen,
            root_dir_mistral=args.preprocessed_root_dir_mistral,
            root_dir_meta=args.preprocessed_root_dir_meta
        )
    
