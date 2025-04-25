### turn off warnings

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings(
    "ignore",
    message="Using the latest cached version of the module.*",
    category=UserWarning  # Or possibly a more specific warning category
)

### install missing packages
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--break-system-packages"])

# install('evaluate')
# install('sacrebleu')
# install('swifter')
# install('rouge_score')

####### env vars
from dotenv import load_dotenv
import os
load_dotenv()
ASU_key = os.environ.get("ASU_key") 
endpoint_url = os.environ.get("endpoint_url") 

## is this the first run? then set to True
First_run = False

## if First_run False tell me what file i should be running off of
FILE_NAME = '5_context_question_answer_2025_04_20_22.csv'

## where do you want to start in the bert loop?
B_start_ii = 354000
## where do you want to start in the rogue loop?
R_start_ii = 0

## what scores do you want? set them to True
OLD_SCHOOL=False
## set old school to do the following tests
SIM_SCORE = False
TFIDF_SIM = False
SAC_BLEU = False
NLTK_BLEU = False
F_SCORE_SIM = False

## newer old school - these methods need the df chunked
input_file = f'C:\\programming_projects\\RAG_fine_tune\\RAG_pipeline_ASU_website\\data\\score\\{FILE_NAME}'  # Your original large file
output_dir = f'C:\\programming_projects\\RAG_fine_tune\\RAG_pipeline_ASU_website\\data\\score\\split_files'  # Directory to save the smaller files
chunk_size = 5000
SBURT = False
ROGUE = True

####### packages for AI stuffs
import requests
import evaluate
import swifter
import datetime
import functions as fun
import pandas as pd

headers = {
    "Authorization": f"Bearer {ASU_key}",
    "Content-Type": "application/json"
}



###############################################################################################
def save_file(df, 
              how_many_rows,
              directory = 'C:\\programming_projects\\RAG_fine_tune\\RAG_pipeline_ASU_website\\data\\score\\',
              file_prefix = '5_context_question_answer_'):
    now = datetime.datetime.now()
    date_string_with_hour = now.strftime("%Y_%m_%d_%H")

    if not os.path.exists(directory): os.makedirs(directory, exist_ok=True)

    if df.shape[0] == how_many_rows:
        print('Saving here:', f'{directory}{file_prefix}{date_string_with_hour}.csv')
        df.to_csv(f'{directory}{file_prefix}{date_string_with_hour}.csv', index=False)

    if df.shape[0] != how_many_rows:
        print("HELP!! THERE IS AN ERROR YOU LOST ROWS!")
        df.to_csv(f'{directory}{file_prefix}{date_string_with_hour}.csv', index=False)
        print('--------------------------------------------------------')
        print(df.shape, df.columns)
        print('--------------------------------------------------------')

###############################################################################################
if OLD_SCHOOL==True:
    ############################################################################################### 
    if First_run == True:
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
        df = df[['title', 'url', 'document_type', 'chunked_word_count', 'orig_word_count', 'context', 'question', 'answer', 'filename']]

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

    if First_run == False:
        df = pd.read_csv(f'C:\\programming_projects\\RAG_fine_tune\\RAG_pipeline_ASU_website\\data\\score\\{FILE_NAME}')
        try:
            df = df.drop(columns=['Unnamed: 0'])
        except:
            print("Unnamed: 0 is not in the columns. congrats!")

    print('--------------------------------------------------------')
    print(df.shape, df.columns)
    print('--------------------------------------------------------')
    ###############################################################################################
    if df.shape[0] < 1980000:
        how_many_rows = 1980000
    if df.shape[0] > 1980000:
        how_many_rows = df.shape[0]

    if SIM_SCORE == True:
        print(datetime.datetime.now(), "similarity_score")
        df['similarity_score'] = df.swifter.apply(lambda row: fun.calculate_fuzzy_similarity(str(row['answer']), str(row['context'])), axis=1)
        save_file(df = df, how_many_rows = how_many_rows)

    if TFIDF_SIM == True:
        print(datetime.datetime.now(), 'tfidf_similarity')
        df['Cosine'] = df.swifter.apply(lambda row: fun.calculate_tfidf_cosine_similarity(str(row['answer']), str(row['context'])), axis=1)
        save_file(df = df, how_many_rows = how_many_rows)

    if SAC_BLEU == True:
        print(datetime.datetime.now(), 'sacrebleu_bleu')
        df['sacrebleu_bleu'] = df.swifter.apply(lambda row: fun.BLEU_SIM(str(row['answer']), str(row['context'])), axis=1)
        save_file(df = df, how_many_rows = how_many_rows)
    if NLTK_BLEU == True:
        print(datetime.datetime.now(), 'NLTK_bleu')
        df['NLTK_bleu'] = df.swifter.apply(lambda row: fun.calculate_bleu_score(str(row['answer']), str(row['context'])), axis=1)
        save_file(df = df, how_many_rows = how_many_rows)

    if F_SCORE_SIM == True:
        print(datetime.datetime.now(), 'FSCORE_SIM')
        df['FSCORE_SIM'] = df.swifter.apply(lambda row: fun.FSCORE_SIM(str(row['answer']), str(row['context'])), axis=1)
        save_file(df = df, how_many_rows = how_many_rows)
###############################################################################################
if not os.path.exists(output_dir):

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file in chunks
    reader = pd.read_csv(input_file, chunksize=chunk_size)
    file_number = 1

    for chunk in reader:
        output_file = os.path.join(output_dir, f'data_part_{file_number}.csv')
        chunk.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")
        file_number += 1

    print("Splitting complete!")
###############################################################################################
def list_files_in_directory(directory):
  """Lists all files in the specified directory.

  Args:
    directory: The path to the directory.

  Returns:
    A list of strings, where each string is the name of a file in the directory.
    Returns an empty list if the directory does not exist or if there are no files.
  """
  try:
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    return files
  except FileNotFoundError:
    print(f"Error: Directory not found: {directory}")
    return []
  except Exception as e:
    print(f"An error occurred: {e}")
    return []

directory_path = r'C:\\programming_projects\\RAG_fine_tune\\RAG_pipeline_ASU_website\\data\\score\\split_files'
file_list = list_files_in_directory(directory_path)

# if file_list:
#   print(f"Files in '{directory_path}':")
#   for file_name in file_list:
#     print(directory_path+'\\'+file_name)
# else:
#   print(f"No files found in '{directory_path}'.")
################################
import re  # For regular expressions

def sort_key(filename):
    """Extracts the numerical part of the filename for sorting."""
    match = re.search(r'_(\d+)\.', filename)
    if match:
        return int(match.group(1))
    return filename  # Return original if no number found

# Sort the file list using the custom sort key
sorted_file_list = sorted(file_list, key=sort_key)

# print("sorted_file_list:", sorted_file_list)
###############################################################################################
sBERT_path = r'C:\\programming_projects\\RAG_fine_tune\\RAG_pipeline_ASU_website\\data\\score\\sBERT\\'  
if (SBURT == True) and file_list:
    for file_name in sorted_file_list[0:]:
        print("SBURT (Batched):", datetime.datetime.now(), "--", file_name)
        df = pd.read_csv(os.path.join(directory_path, file_name))
        how_many_rows = df.shape[0]
        batch_size = 100  # Adjust based on your memory

        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i + batch_size].copy()
            batch_df['answer'] = batch_df['answer'].astype(str)
            batch_df['context'] = batch_df['context'].astype(str)
            pairs = list(zip(batch_df['answer'], batch_df['context']))
            try:
                # print(f"Processing batch starting at index: {batch_df.index[0] if not batch_df.empty else None}")
                tokens_list1 = []
                tokens_list2 = []
                for j, pair in enumerate(pairs):
                    answer_str, context_str = pair
                    # print(f"Pair at index {j}: {pair}")
                    tokens1 = answer_str.split()
                    tokens2 = context_str.split()
                    tokens_list1.append(tokens1)
                    tokens_list2.append(tokens2)
                # print("Length of tokens_list1:", [len(t) for t in tokens_list1])
                # print("Length of tokens_list2:", [len(t) for t in tokens_list2])

                similarities = fun.sBERT_batched(pairs) # Comment out the function call
                df.loc[batch_df.index, 'sBURT'] = similarities

                if i % 1000 == 0:
                    print("SBURT (Batch Progress)", datetime.datetime.now(), "--", i)
            except Exception as e:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                print(f"{timestamp}: An error occurred during processing at batch starting index {batch_df.index[0] if not batch_df.empty else None}: {e}")
                break

        save_file(df=df,
                  how_many_rows=how_many_rows,
                  directory=sBERT_path,
                  file_prefix='5_sBERT_' + file_name[:-4]+'_Date_')
# ###############################################################################################
ROGUE_path = r'C:\\programming_projects\\RAG_fine_tune\\RAG_pipeline_ASU_website\\data\\score\\ROGUE\\'
file_list = list_files_in_directory(sBERT_path)
sorted_file_list = sorted(file_list, key=sort_key)

if ROGUE == True:
        for file_name in sorted_file_list[130:]:
            print("ROGUE (Batched):", datetime.datetime.now(), "--", file_name)
            df = pd.read_csv(os.path.join(sBERT_path, file_name))
            df.drop(columns=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], inplace=True)
            how_many_rows = df.shape[0]

            print("ROGUE (Batch Processing):", datetime.datetime.now(), "-- Processing all rows")
            rouge_df = fun.batched_rouge_scores(df) # Process all at once

            # Concatenate the ROUGE scores with the original DataFrame
            df = pd.concat([df.reset_index(drop=True), rouge_df], axis=1)
            print(df.columns)

            save_file(df=df,
                      how_many_rows=how_many_rows,
                      directory=ROGUE_path,
                      file_prefix='5_ROGUE_' + file_name[8:-4]+'_Date_')