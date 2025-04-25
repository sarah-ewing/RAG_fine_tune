###########################################################################
#### rouge Bilingual Evaluation Understudy - torchmetrics
###########################################################################
import evaluate
import pandas as pd

## rogue1 - refers to overlap of unigrams between the system summary and reference summary

## rouge2 - refers to the overlap of bigrams between the system and reference summaries

## rogueL - Longest Common Subsequence(LCS)

## rougeLsum
## The ROUGE-Lsum is related to the ROUGE-L metric but applies a slightly different calculation method. 
## It applies the ROUGE-L calculation method at the sentence level and then aggregates all the results for the final score. 
## This metric is seen as more suitable for tasks where sentence level extraction is valuable such as extractive summarization tasks.
## In simpler terms, whereas ROUGE-L looks at the summary as a whole, ROUGE-Lsum considers sentence-level information, potentially providing more granularity in some use cases.
## ROUGE-L ignores newlines and computes the LCS for the entire text. ROUGE-Lsum splits the text into sentences based on newlines and computes the LCS for each pair of sentences and take the average score for all sentences.

def rouge_SIM(A, B):
    rouge = evaluate.load('rouge')
    predictions = [B]
    references = [A]
    results = rouge.compute(predictions=predictions, references=references)
    return results

def rouge_scores_row_wise(row):
    """Calculates ROUGE scores for a given row."""
    rouge = evaluate.load('rouge')
    context = row['context']
    answer = row['answer']
    results = rouge.compute(predictions=[answer], references=[context])
    return pd.Series({
        'rouge1': results['rouge1'],
        'rouge2': results['rouge2'],
        'rougeL': results['rougeL'],
        'rougeLsum': results['rougeLsum']
    })
###########################################################################
import evaluate
import pandas as pd
import numpy as np
from tqdm import tqdm

def batched_rouge_scores(df, context_col='context', answer_col='answer', batch_size=200):
    """Calculates ROUGE scores in batches for a DataFrame with NaN handling."""
    rouge = evaluate.load('rouge')
    all_results = []
    total_rows = len(df)

    with tqdm(total=total_rows, desc="Processing ROUGE") as pbar:
        for i in range(0, total_rows, batch_size):
            batch_df = df.iloc[i:i + batch_size]
            predictions = batch_df[answer_col].tolist()
            references = batch_df[context_col].tolist()

            for j in range(len(batch_df)):
                prediction = predictions[j]
                reference = references[j]

                if pd.isna(prediction) or pd.isna(reference):
                    results = {'rouge1': np.nan, 'rouge2': np.nan, 'rougeL': np.nan, 'rougeLsum': np.nan}
                else:
                    try:
                        results = rouge.compute(predictions=[prediction], references=[reference])
                    except ValueError as e:
                        print(f"Error calculating ROUGE for row {df.index[i+j]}: {e}")
                        print(f"Prediction: '{prediction}'")
                        print(f"Reference: '{reference}'")
                        results = {'rouge1': np.nan, 'rouge2': np.nan, 'rougeL': np.nan, 'rougeLsum': np.nan}

                all_results.append(results)
                pbar.update(1)

    results_df = pd.DataFrame(all_results)
    return pd.concat([df.reset_index(drop=True), results_df], axis=1)

###########################################################################
#### sacrebleu Charcter F & BLEU
###########################################################################
from sacrebleu.metrics import BLEU, CHRF
def BLEU_SIM(A, B):
    refs = [[A]]
    sys = [B]

    bleu = BLEU()
    results = bleu.corpus_score(sys, refs)
    results = str(results)
    results = results[6:12]
    return(results)


def FSCORE_SIM(A, B):
    refs = [[A]]
    sys = [B]

    ## CHaRacter-level F-score
    chrf = CHRF()
    results = chrf.corpus_score(sys, refs)
    results = str(results)
    results=results[8:]
    return(results)

###########################################################################
#### fuzzy string similarity
###########################################################################
from fuzzywuzzy import fuzz

def calculate_fuzzy_similarity(response, context):
    return fuzz.partial_ratio(response, context)  # Or use other fuzzywuzzy functions

###########################################################################
#### TFID vectorizer & cosine similarity
###########################################################################
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_tfidf_cosine_similarity(response, context):
    """Calculates TF-IDF cosine similarity between response and context."""

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([response, context])  # Fit and transform both strings

    similarity = cosine_similarity(vectors[0], vectors[1])[0][0] # cosine_similarity returns a 2d array
    return similarity


###########################################################################
#### NLTK BLEU
###########################################################################
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk  # Make sure you have NLTK installed
nltk.download('punkt', quiet=True) # Download punkt tokenizer if you haven't already

def calculate_bleu_score(response, context):
    """Calculates BLEU score between response and context."""

    reference = context.split()  # Split the context into words
    candidate = response.split()  # Split the response into words

    # Use a smoothing function to handle cases where n-grams are not present
    smoothing = SmoothingFunction().method4  # Or another smoothing method

    bleu = sentence_bleu([reference], candidate, smoothing_function=smoothing)
    return bleu

###########################################################################
#### sBERT
###########################################################################
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load the BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def sBERT(tokens1, tokens2):
    tokens1 = tokens1.split()
    tokens2 = tokens2.split()
    # Convert tokens to input IDs
    input_ids1 = torch.tensor(tokenizer.convert_tokens_to_ids(tokens1)).unsqueeze(0)  # Batch size 1
    input_ids2 = torch.tensor(tokenizer.convert_tokens_to_ids(tokens2)).unsqueeze(0)  # Batch size 1

    # Obtain the BERT embeddings
    with torch.no_grad():
        outputs1 = model(input_ids1)
        outputs2 = model(input_ids2)
        embeddings1 = outputs1.last_hidden_state[:, 0, :]  # [CLS] token
        embeddings2 = outputs2.last_hidden_state[:, 0, :]  # [CLS] token

    # Calculate similarity
    similarity_score = cosine_similarity(embeddings1, embeddings2)
    return similarity_score[0][0]

###########################################################################
#### BATCH sBERT
###########################################################################
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np # Import numpy

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# if torch.cuda.is_available():
print(f"Number of CUDA devices available: {torch.cuda.device_count()}")
print(f"Current CUDA device: {torch.cuda.current_device()}")
print(f"Name of the CUDA device: {torch.cuda.get_device_name(0)}")
device = torch.device("cuda")
# else:
#     print("CUDA is not available. Using CPU.")
#     device = torch.device("cpu")

print("device:", device)
model.to(device)

def sBERT_batched(pairs):
    # print("Inside sBERT_batched function:")
    answers = []
    contexts = []
    for i, pair in enumerate(pairs):
        # print(f"Type of pair at index {i}: {type(pair)}, Length: {len(pair)}")
        try:
            answer, context = pair
            # print(f"Answer (index {i}): '{answer[:50]}...'")
            # print(f"Context (index {i}): '{context[:50]}...'")
            answers.append(answer)
            contexts.append(context)
        except Exception as e:
            print(f"Error processing pair at index {i}: {pair}, Error: {e}")
            raise

    # print("First answer:", answers[0] if answers else None)
    # print("First context:", contexts[0] if contexts else None)
    # print("Number of answers:", len(answers))
    # print("Number of contexts:", len(contexts))

    inputs1 = tokenizer(answers, padding=True, truncation=True, return_tensors='pt').to(device)
    # print("Shape of inputs1['input_ids']:", inputs1['input_ids'].shape)
    # print("Shape of inputs1['attention_mask']:", inputs1['attention_mask'].shape)

    inputs2 = tokenizer(contexts, padding=True, truncation=True, return_tensors='pt').to(device)
    # print("Shape of inputs2['input_ids']:", inputs2['input_ids'].shape)
    # print("Shape of inputs2['attention_mask']:", inputs2['attention_mask'].shape)

    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)
        embeddings1 = outputs1.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings2 = outputs2.last_hidden_state[:, 0, :].cpu().numpy()

    similarity_scores = cosine_similarity(embeddings1, embeddings2)
    return np.diagonal(similarity_scores)