# NOTE: Most of these routines are taken from https://github.com/technion-cs-nlp/LLMsKnow, with some adaptations

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import json
import pandas as pd
import argparse
from utils.constants import LIST_OF_MODELS

LIST_OF_DATASETS = ['triviaqa', 'triviaqa_test',
                    'imdb', 'imdb_test',
                    'winobias', 'winobias_test',
                    'hotpotqa', 'hotpotqa_test',
                    'hotpotqa_with_context', 'hotpotqa_with_context_test',
                    'math', 'math_test',
                    'movies', 'movies_test',
                    'mnli', 'mnli_test',
                    'natural_questions_with_context', 'natural_questions_with_context_test',
                    'winogrande', 'winogrande_test',
                    ] 



class DatasetComboAction(argparse.Action):
    """Custom action to parse LLM:dataset combinations from a single string"""
    def __call__(self, parser, namespace, values, option_string=None):
        # Expect a single string value
        if not isinstance(values, str):
            raise argparse.ArgumentError(
                self, "Input must be a single quoted string (e.g., 'llm1:data1 data2 llm2:data3')")
        
        dataset_list = []
        current_llm = None
        tokens = values.split()  # Split the string on whitespace
        for value in tokens:
            if ':' in value:
                # New LLM specification
                if value.endswith(':'):
                    raise argparse.ArgumentError(
                        self, "Invalid format: LLM specification cannot end with a colon.")
                llm, dataset = value.split(':', 1)
                if llm not in LIST_OF_MODELS:
                    raise argparse.ArgumentError(
                        self, f"Invalid LLM: {llm}. Must be one of: {', '.join(LIST_OF_MODELS)}")
                current_llm = llm
                dataset_list.append((current_llm, dataset))
            else:
                # Dataset using the current LLM
                if current_llm is None:
                    raise argparse.ArgumentError(
                        self, "Must specify LLM before datasets (use 'llm:dataset' format first)")
                dataset_list.append((current_llm, value))
        setattr(namespace, self.dest, dataset_list)


# NOTE: for all datasets we use 10_000 samples max, the same way as in https://github.com/technion-cs-nlp/LLMsKnow
def imdb_preprocess(args, model_name, reviews, labels):

    prompts = []
    labels_to_name = ['negative', 'positive']

    review1 = None
    label1 = None

    if 'phi' in model_name.lower():
        for review, label in zip(reviews, labels):
            prompt = f"""
            Review: I would put this at the top of my list of films in the category of unwatchable trash! There are films that are bad, but the worst kind are the ones that are unwatchable but you are suppose to like them because they are supposed to be good for you! The sex sequences, so shocking in its day, couldn't even arouse a rabbit. The so called controversial politics is strictly high school sophomore amateur night Marxism. The film is self-consciously arty in the worst sense of the term. The photography is in a harsh grainy black and white. Some scenes are out of focus or taken from the wrong angle. Even the sound is bad! And some people call this art?<br /><br />
            Label: negative
            Review: Zentropa is the most original movie I've seen in years. If you like unique thrillers that are influenced by film noir, then this is just the right cure for all of those Hollywood summer blockbusters clogging the theaters these days. Von Trier's follow-ups like Breaking the Waves have gotten more acclaim, but this is really his best work. It is flashy without being distracting and offers the perfect combination of suspense and dark humor. It's too bad he decided handheld cameras were the wave of the future. It's hard to say who talked him away from the style he exhibits here, but it's everyone's loss that he went into his heavily theoretical dogma direction instead.
            Label: positive
            Review: {review}
            Label:"""
            prompts.append(prompt)

    else:
        for review, label in zip(reviews, labels):
            if review1 is None:
                review1 = review
                label1 = labels_to_name[label]
            answer_first_prompt = "Start with either 'positive' or 'negative' first, then elaborate."
            example_prompt = ''
            if 'llama-3' not in model_name.lower():
                answer_first_prompt = ''
                example_prompt = f"""
                Example:
                Review: {review1}
                Label: {label1}
                Review: {review}
                Label: """


            prompt = f"""Classify the following movie reviews as either "positive" or "negative". {answer_first_prompt}
            {example_prompt}
            Review: {review}
            Label:"""
            prompts.append(prompt)

    return prompts

def load_data_imdb(split):
    dataset = load_dataset("imdb")


    indices = np.arange(0, len(dataset[split]))
    np.random.shuffle(indices)

    reviews = dataset[split][indices[:10000]]['text']
    labels = dataset[split][indices[:10000]]['label']
    return reviews, labels


def load_data_triviaqa(test=False, legacy=False):
    if legacy:
        with open('../data/verified-web-dev.json') as f:
            data_verified = json.load(f)
            data_verified = data_verified['Data']
        with open('../data/web-dev.json') as f:
            data = json.load(f)
            data = data['Data']
        questions_from_verified = [x['Question'] for x in data_verified]
        data_not_verified = []
        for x in data:
            if x['Question'] in questions_from_verified:
                pass
            else:
                data_not_verified.append(x)

        print("Length of not verified data: ", len(data_not_verified))
        print("Length of verified data: ", len(data_verified))

        if test:
            return [ex['Question'] for ex in data_verified], [ex['Answer']['Aliases'] for ex in data_verified]
        else:
            return [ex['Question'] for ex in data_not_verified], [ex['Answer']['Aliases'] for ex in data_not_verified]
    else:
        if test:
            file_path = '../data/triviaqa-unfiltered/unfiltered-web-dev.json'
        else:
            file_path = '../data/triviaqa-unfiltered/unfiltered-web-train.json'
        with open(file_path) as f:
            data = json.load(f)
            data = data['Data']
        data, _ = train_test_split(data, train_size=10000, random_state=42)
        return [ex['Question'] for ex in data], [ex['Answer']['Aliases'] for ex in data]

def triviaqa_postprocess(model_name, raw_answers):
    model_answers = []
    if 'instruct' in model_name.lower():
        model_answers = raw_answers
    else:
        for ans in raw_answers:
            model_answer = ans.strip().split('\n')[0]
            model_answers.append(model_answer)
    return raw_answers, model_answers


def load_hotpotqa(args, split, with_context):

    dataset = load_dataset("hotpot_qa", 'distractor')
    subset_indices = np.random.randint(0, len(dataset[split]), 10000)
    all_questions = [dataset[split][int(x)]['question'] for x in subset_indices]
    labels = [dataset[split][int(x)]['answer'] for x in subset_indices]
    if with_context:
        all_questions = []

        for idx in subset_indices:
            prompt = ""
            for evidence in dataset[split][int(idx)]['context']['sentences']:
                for sentence in evidence:
                    prompt += sentence + '\n'
            prompt += dataset[split][int(idx)]['question']
            all_questions.append(prompt)

    return all_questions, labels

def triviqa_preprocess(args, model_name, all_questions, labels):
    prompts = []
    if 'instruct' in model_name.lower():
        prompts = all_questions
    else:
        for q in all_questions:
            prompts.append(f'''Q: {q}
        A:''')
    return prompts

def load_data_math(test=False):
    if test:
        data = pd.read_csv("./data/AnswerableMath_test.csv")
    else:
        data = pd.read_csv("./data/AnswerableMath.csv")

    questions = data['question']
    answers = data['answer'].map(lambda x: eval(x)[0])
    return questions, answers

def math_preprocess(args, model_name, all_questions, labels):
    prompts = []

    if 'instruct' in model_name.lower():
        for q in all_questions:
            prompts.append(q + " Answer shortly.")
    else:
        for q in all_questions:
            prompts.append(f'''Q: {q}
            A:''')
    return prompts

def load_winobias(dev_or_test):
    data = pd.read_csv(f'./data/winobias_{dev_or_test}.csv')
    return (data['sentence'], data['q'], data['q_instruct']), data['answer'], data['incorrect_answer'], data['stereotype'], data['type']

def winobias_preprocess(args, model_name, all_questions, labels):
    sentences, q, q_instruct = all_questions
    if 'instruct' in model_name.lower():
        prompts = [x + ' ' + y for x, y in zip(sentences, q_instruct)]
    else:
        prompts = [x + ' ' + y for x, y in zip(sentences, q)]

    return prompts

def load_data_triviaqa(test=False, legacy=False):
    if legacy:
        with open('./data/verified-web-dev.json') as f:
            data_verified = json.load(f)
            data_verified = data_verified['Data']
        with open('./data/web-dev.json') as f:
            data = json.load(f)
            data = data['Data']
        questions_from_verified = [x['Question'] for x in data_verified]
        data_not_verified = []
        for x in data:
            if x['Question'] in questions_from_verified:
                pass
            else:
                data_not_verified.append(x)

        print("Length of not verified data: ", len(data_not_verified))
        print("Length of verified data: ", len(data_verified))

        if test:
            return [ex['Question'] for ex in data_verified], [ex['Answer']['Aliases'] for ex in data_verified]
        else:
            return [ex['Question'] for ex in data_not_verified], [ex['Answer']['Aliases'] for ex in data_not_verified]
    else:
        if test:
            file_path = './data/triviaqa-unfiltered/unfiltered-web-dev.json'
        else:
            file_path = './data/triviaqa-unfiltered/unfiltered-web-train.json'
        with open(file_path) as f:
            data = json.load(f)
            data = data['Data']
        data, _ = train_test_split(data, train_size=10000, random_state=42)
        return [ex['Question'] for ex in data], [ex['Answer']['Aliases'] for ex in data]

def load_data_movies(test=False):
    file_name = 'movie_qa'
    if test:
        file_path = f'./data/{file_name}_test.csv'
    else: # train
        file_path = f'./data/{file_name}_train.csv'
    import os
    if not os.path.exists(file_path):
        # split into train and test
        data = pd.read_csv(f"./data//{file_name}.csv")
        # spit into test and train - 50% each
        train, test = train_test_split(data, train_size=10000, random_state=42)
        train.to_csv(f"./data/{file_name}_train.csv", index=False)
        test.to_csv(f"./data/{file_name}_test.csv", index=False)

    data = pd.read_csv(file_path)
    questions = data['Question']
    answers = data['Answer']
    return questions, answers

def load_data(args, dataset_name):
    # NOTE: using max_new_tokens = 100 (which are the max generated tokens) for all datasets except math
    max_new_tokens = 100
    context, origin, stereotype, type_, wrong_labels = None, None, None, None, None
    if dataset_name == 'imdb':
        all_questions, labels = load_data_imdb('train')
        preprocess_fn = imdb_preprocess
    elif dataset_name == 'imdb_test':
        all_questions, labels = load_data_imdb('test')
        preprocess_fn = imdb_preprocess
    elif dataset_name == 'hotpotqa':
        all_questions, labels = load_hotpotqa(args, 'train', with_context=False)
        preprocess_fn = triviqa_preprocess
    elif dataset_name == 'hotpotqa_test':
        all_questions, labels = load_hotpotqa(args, 'validation', with_context=False)
        preprocess_fn = triviqa_preprocess
    elif dataset_name == 'hotpotqa_with_context':
        all_questions, labels = load_hotpotqa(args, 'train', with_context=True)
        preprocess_fn = triviqa_preprocess
    elif dataset_name == 'hotpotqa_with_context_test':
        all_questions, labels = load_hotpotqa(args, 'validation', with_context=True)
        preprocess_fn = triviqa_preprocess
    elif dataset_name == 'triviaqa':
        all_questions, labels = load_data_triviaqa(False)
        preprocess_fn = triviqa_preprocess
    elif dataset_name == 'triviaqa_test':
        all_questions, labels = load_data_triviaqa(True)
        preprocess_fn = triviqa_preprocess
    elif dataset_name == 'math':
        all_questions, labels = load_data_math(test=False)
        preprocess_fn = math_preprocess
        max_new_tokens = 200
    elif dataset_name == 'math_test':
        all_questions, labels = load_data_math(test=True)
        preprocess_fn = math_preprocess
        max_new_tokens = 200
    elif dataset_name == 'winobias':
        all_questions, labels, wrong_labels, stereotype, type_ = load_winobias('dev')
        preprocess_fn = winobias_preprocess
    elif dataset_name == 'winobias_test':
        all_questions, labels, wrong_labels, stereotype, type_ = load_winobias('test')
        preprocess_fn = winobias_preprocess
    elif dataset_name == 'movies':
        all_questions, labels = load_data_movies(test=False)
        preprocess_fn = triviqa_preprocess
    elif dataset_name == 'movies_test':
        all_questions, labels = load_data_movies(test=True)
        preprocess_fn = triviqa_preprocess
    else:
        raise TypeError("data type is not supported")
    return all_questions, context, labels, max_new_tokens, origin, preprocess_fn, stereotype, type_, wrong_labels

