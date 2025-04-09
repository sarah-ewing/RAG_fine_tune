import pandas as pd

from dotenv import load_dotenv

import os

load_dotenv()
ASU_key = os.environ.get("ASU_key")  
file_name = os.environ.get("file_name")  

directory_path = 'C:\\programming_projects\\RAG_fine_tune\\web_crawl\\'
df = pd.read_csv(directory_path + file_name)
df.rename(columns={'word_count': 'orig_word_count', 'char_count': 'orig_char_count'}, inplace = True)

#######################################################################################################
def chunk_text_with_overlap(df, text_column, chunk_size=500, overlap=20):
    """
    Breaks a column of text in a pandas DataFrame into chunks of a specified size
    with a given overlap. Keeps all other columns and their original values.

    Args:
        df (pd.DataFrame): The input DataFrame.
        text_column (str): The name of the column containing the text to chunk.
        chunk_size (int): The desired size of each text chunk in words.
        overlap (int): The number of overlapping words between consecutive chunks.

    Returns:
        pd.DataFrame: A new DataFrame where each row represents a chunk of the
                      original text, along with the values from the other columns
                      and the original full text.
    """
    new_rows = []
    for index, row in df.iterrows():
        text = row[text_column]
        words = text.split()
        n_words = len(words)
        start = 0
        while start < n_words:
            end = min(start + chunk_size, n_words)
            chunk = " ".join(words[start:end])
            new_row = row.to_dict()
            new_row['chunked_text'] = chunk
            new_rows.append(new_row)
            start += (chunk_size - overlap)
            if start < overlap:  # Ensure we don't go negative
                start = 0

    new_df = pd.DataFrame(new_rows)
    return new_df
#######################################################################################################

chunked_df = chunk_text_with_overlap(df.copy(), 'cleaned_text')

chunked_df['chunked_word_count'] = chunked_df['chunked_text'].apply(lambda x: len(str(x).split()))
chunked_df['chunked_char_count'] = chunked_df['chunked_text'].str.len()

# print(chunked_df.head())
print(f"\nShape of the original DataFrame: {df.shape}")
print(f"Shape of the chunked DataFrame: {chunked_df.shape}")

directory_path = 'C:\\programming_projects\\RAG_fine_tune\\RAG_pipeline_ASU_website\\data\\'
print(directory_path + 'chunked_' +file_name)
chunked_df.to_csv(directory_path + 'chunked_' +file_name, index=False)
