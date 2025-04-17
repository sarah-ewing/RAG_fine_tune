import warnings


warnings.filterwarnings(
    "ignore",
    message="Using the latest cached version of the module.*",
    category=UserWarning  # Or possibly a more specific warning category
)

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--break-system-packages"])

# install('evaluate')
# install('sacrebleu')
# install('swifter')
import evaluate
import swifter

from dotenv import load_dotenv
import os
load_dotenv()
ASU_key = os.environ.get("ASU_key") 
endpoint_url = os.environ.get("endpoint_url") 

import requests

import functions as fun

headers = {
    "Authorization": f"Bearer {ASU_key}",
    "Content-Type": "application/json"
}

#################################   
import pandas as pd

df = pd.read_csv(r'C:\\programming_projects\\RAG_fine_tune\\RAG_pipeline_ASU_website\\data\\silver_data_4_16_2025.csv',
                 dtype={'title': 'str', 
                        'url': 'str', 
                        'document_type': 'str',
                        'chunked_word_count':'int64',
                        'orig_word_count':'int64',
                        'contex':'str',
                        'question':'str',
                        'answer':'str',
                        'filename':'str',
                        'section':'str'})
df = df.reset_index(drop=True)
df.rename( columns={'contex': 'context'},inplace=True)

df['question'] = df['question'].str.replace('\n', ' ')
df['answer'] = df['answer'].str.replace('\n', ' ')
df['context'] = df['context'].str.replace('\n', ' ')

# Identify the rows where 'context' is numeric and remove them
numeric_context_mask = pd.to_numeric(df['context'], errors='coerce').notna()
# Filter out those rows
df = df[~numeric_context_mask].copy()

df['similarity_score'] = 0.0
df['NLTK_bleu'] = 0.0
df['sacrebleu_bleu'] = 0.0
df['Cosine'] = 0.0
df['rouge1'] = 0.0
df['rouge2'] = 0.0
df['rougeL'] = 0.0
df['rougeLsum'] = 0.0
df['sBURT'] = 0.0

df['ragas_bleu'] = 0.0
df['ragas_precision_reference'] = 0.0
df['ragas_faith'] = 0.0
df['ragas_fact'] = 0.0
df['regas_sem_sim'] = 0.0
df['regas_rogue'] = 0.0

df = df[['title', 'url', 'document_type', 'chunked_word_count', 'orig_word_count', 'context', 'question', 'answer', 'filename']]

print('--------------------------------------------------------')
print(df.shape, df.columns)
print('--------------------------------------------------------')

import datetime

print(datetime.datetime.now(), "similarity_score")
df['similarity_score'] = df.swifter.apply(lambda row: fun.calculate_fuzzy_similarity(str(row['answer']), str(row['context'])), axis=1)
df.to_csv('C:\\programming_projects\\RAG_fine_tune\\RAG_pipeline_ASU_website\\data\\5_context_question_answer.csv')

print(datetime.datetime.now(), 'tfidf_similarity')
df['Cosine'] = df.swifter.apply(lambda row: fun.calculate_tfidf_cosine_similarity(str(row['answer']), str(row['context'])), axis=1)
df.to_csv('C:\\programming_projects\\RAG_fine_tune\\RAG_pipeline_ASU_website\\data\\5_context_question_answer.csv')

print(datetime.datetime.now(), 'sacrebleu_bleu')
df['sacrebleu_bleu'] = df.swifter.apply(lambda row: fun.BLEU_SIM(str(row['answer']), str(row['context'])), axis=1)
df.to_csv('C:\\programming_projects\\RAG_fine_tune\\RAG_pipeline_ASU_website\\data\\5_context_question_answer.csv')

print(datetime.datetime.now(), 'NLTK_bleu')
df['NLTK_bleu'] = df.swifter.apply(lambda row: fun.calculate_bleu_score(str(row['answer']), str(row['context'])), axis=1)
df.to_csv('C:\\programming_projects\\RAG_fine_tune\\RAG_pipeline_ASU_website\\data\\5_context_question_answer.csv')

print(datetime.datetime.now(), 'FSCORE_SIM')
df['FSCORE_SIM'] = df.swifter.apply(lambda row: fun.FSCORE_SIM(str(row['answer']), str(row['context'])), axis=1)
df.to_csv('C:\\programming_projects\\RAG_fine_tune\\RAG_pipeline_ASU_website\\data\\5_context_question_answer.csv')

print(datetime.datetime.now(), 'sBURT')
df['sBURT'] = df.swifter.apply(lambda row: fun.sBERT(str(row['answer']), str(row['context'])), axis=1)
df.to_csv('C:\\programming_projects\\RAG_fine_tune\\RAG_pipeline_ASU_website\\data\\5_context_question_answer.csv')

print(datetime.datetime.now(), 'Rogue')
df[['rouge1', 'rouge2', 'rougeL', 'rougeLsum']] = df.swifter.apply(fun.rouge_scores_row_wise, axis=1)
df.to_csv('C:\\programming_projects\\RAG_fine_tune\\RAG_pipeline_ASU_website\\data\\5_context_question_answer.csv')