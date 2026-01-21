
############### MAX LAYERS AND TOKENS ###############

L_MAX = 50
N_MAX = 100

############### HD ###############

LIST_OF_MODELS = ['mistralai/Mistral-7B-Instruct-v0.2',
'meta-llama/Meta-Llama-3-8B-Instruct', 
'Qwen/Qwen2.5-7B-Instruct']

PROBING_ARCHITECTURES = ['ACT-Vit-foundation', 'ACT-MLP-foundation']

LIST_OF_DATASETS = ['imdb', 'imdb_test',
                       'movies', 'movies_test',
                       'hotpotqa', 'hotpotqa_test',
                       'triviaqa', 'triviaqa_test',
                       'hotpotqa_with_context', 'hotpotqa_with_context_test',]

##################################

FEATURE_DIMS = {
    "Mistral-7B-Instruct-v0.2": 4096,
    "Meta-Llama-3-8B-Instruct": 4096,
    "Qwen2.5-7B-Instruct": 3584
}


##################################
# NOTE: We assume that all our train datasets are composed of 10000 sampels maximum.
MAX_DATASET_SIZES = 10000
TRAIN_DATASET_SIZES = 10000
MOVIES_TEST_DATASET_SIZE = 7856
RANDOM_STATE = 42
SIZE_LIMITS = [500, 1000, 2000, 5000]
TRAIN_VAL_RATIO = 4 / 5  # 80% train, 20% val
WEIGHT_DECAYS = [0.0001, 0.01, 1, 100, 10000]