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