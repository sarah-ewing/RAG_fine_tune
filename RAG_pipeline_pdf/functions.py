import subprocess
import sys
import matplotlib.pyplot as plt

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--break-system-packages"])

# install('fuzzywuzzy')
install('python-Levenshtein')
# install('scikit-learn')
# install('nltk')

##################################################################
import os

def list_files_in_directory(dir_path="."):
    """Lists all files in the specified directory."""
    try:
        files = os.listdir(dir_path)
        return files
    except FileNotFoundError:
        return f"Error: Directory '{dir_path}' not found."
    except NotADirectoryError:
        return f"Error: '{dir_path}' is not a directory."
    
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