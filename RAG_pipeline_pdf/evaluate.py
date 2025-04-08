
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# install('fuzzywuzzy')
# install('python-Levenshtein')
#################################   
import pandas as pd

df = pd.read_csv("silver_data.csv")
df = df[(df['section']!= 'References') & (df['section']!= 'Appendix')]
df = df.reset_index(drop=True)
df['context'] = df['contex']

df['question'] = df['question'].str.replace('\n', '')
df['answer'] = df['answer'].str.replace('\n', '')
df['context'] = df['context'].str.replace('\n', '')

df['similarity_score'] = 0.0
df['BLEU'] = 0.0
df['Cosine'] = 0.0

df = df[['section', 'title', 'file_name', 'document_type', 'page', 'total_pages', 'context', 'question', 'answer', 'similarity_score', 'BLEU', 'Cosine']]
#################################
from fuzzywuzzy import fuzz

def calculate_fuzzy_similarity(response, context):
    return fuzz.partial_ratio(response, context)  # Or use other fuzzywuzzy functions

#################################
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_tfidf_cosine_similarity(response, context):
    """Calculates TF-IDF cosine similarity between response and context."""

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([response, context])  # Fit and transform both strings

    similarity = cosine_similarity(vectors[0], vectors[1])[0][0] # cosine_similarity returns a 2d array
    return similarity


#################################
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

#################################
for i in range(0, len(df['question'])):
    user_input = df['question'].iloc[i]
    response = df['answer'].iloc[i]
    retrieved_context = df['context'].iloc[i]

    similarity_score = calculate_fuzzy_similarity(response, retrieved_context)
    df.loc[i, 'similarity_score'] = similarity_score
    # print(f"Fuzzy Similarity Score: {similarity_score}")

    tfidf_similarity = calculate_tfidf_cosine_similarity(response, retrieved_context)
    df.loc[i, 'Cosine'] = tfidf_similarity
    # print(f"TF-IDF Cosine Similarity: {tfidf_similarity:.2f}")

    bleu_score = calculate_bleu_score(response, retrieved_context)
    df.loc[i, 'BLEU'] = bleu_score
    # print(f"BLEU Score: {bleu_score:.2f}")

# print(df.head())

df.to_csv('silver_data_graded.csv', index=False)


### Half Golden
df = pd.read_csv("silver_data_graded.csv")
df = df[(df['similarity_score'] > 30) & (df['BLEU'] > 0.006) & (df['Cosine'] > 0.46)]


print('Half Golden:', df.shape)

df = df.reset_index(drop=True)
df[['section', 'title', 'file_name', 'document_type', 'page', 'total_pages', 'context', 'question', 'answer', 'similarity_score', 'BLEU', 'Cosine']].to_csv('half_golden.csv', index=False)