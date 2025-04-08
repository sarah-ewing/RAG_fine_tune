import os
import pandas as pd
import PyPDF2
import re
import requests

### text needs to be cleaned
def clean_string(text):
    text = re.sub(r'[^A-Za-z0-9., ]', ' ', text)
    text = re.sub(r'  ', ' ', text)
    text = re.sub(r'  ', ' ', text)
    return text

## parsing the text
from typing import List

def parse_string_with_overlap(text: str, chunk_length: int, overlap_words: int) -> List[str]:
    """
    Parses a string into chunks with a specified length and word overlap.

    Args:
        text: The input string.
        chunk_length: The desired length of each chunk (in words).
        overlap_words: The number of words to overlap between chunks.

    Returns:
        A list of string chunks.  Returns an empty list if the input is invalid.
    """

    if not isinstance(text, str) or not isinstance(chunk_length, int) or not isinstance(overlap_words, int):
        raise TypeError("Input types must be: str, int, int")

    if chunk_length <= 0 or overlap_words < 0 or overlap_words >= chunk_length:
        raise ValueError("chunk_length must be > 0, overlap_words must be >= 0 and < chunk_length")


    words = text.split()
    num_words = len(words)

    if num_words == 0:  # Handle empty string
        return []

    chunks = []
    start_index = 0

    while start_index < num_words:
        end_index = min(start_index + chunk_length, num_words)  # Don't exceed string length
        chunk = " ".join(words[start_index:end_index])
        chunks.append(chunk)

        start_index += (chunk_length - overlap_words)  # Move starting point

    return chunks

# Specify the folder path
folder_path = "C:/programming_projects/ASU/sarah_pub/" 

##########################################################
# List all files and directories in the folder
files = os.listdir(folder_path)

### some of the files start with . and are un-readable
def remove_if_starts_with(string_list, char):
    new_list = []
    for string in string_list:
        if not string.startswith(char):
            new_list.append(string)      # Add original string
    return new_list

files = remove_if_starts_with(files, ".")
df_out = pd.DataFrame([])
##########################################################
from dotenv import load_dotenv

import os

load_dotenv()
ASU_key = os.environ.get("ASU_key")  
##########################################################

### files
for ii in range(0, len(files)):
    title = re.sub(r'.pdf', ' ', files[ii])

    print(ii, files[ii])
    

    # Open the PDF in read-binary mode
    with open(folder_path+files[ii], "rb") as pdf_file:
        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Get the number of pages
        total_page = len(pdf_reader.pages)

    ### pages
    for iii in range(0, total_page):
        page_no = iii
        print('page number' , iii)

        # Open the PDF in read-binary mode
        with open(folder_path+files[ii], "rb") as pdf_file:
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(pdf_file)

            # Extract text from the first page
            page = pdf_reader.pages[page_no]
            text = page.extract_text()

        cleaned_string = clean_string(text)
        
        word_count = len(cleaned_string.split())

        if word_count > 1000:
            try:
                chunks = parse_string_with_overlap(cleaned_string, int(word_count/4), 15)
            except ValueError as e:
                chunks = cleaned_string
                print(f"Error: {e}")
        if word_count <= 1000:
            try:
                chunks = parse_string_with_overlap(cleaned_string, word_count, 0)
            except:
                print("word_count:", word_count,"\ncleaned_string:", cleaned_string, "\n\n")

        ### some pages dont have any text and we dont care about that!
        if word_count > 5:
            ##########################################
            ## section type
            api_url = 'https://api-dev-poc.aiml.asu.edu/queryV2'
            bearer_token = ASU_key
            json_payload = {
                "query": "what part of a document is the following text from in a academic paper {cleaned_string}? only respond with the section type, no other text.".format(cleaned_string=cleaned_string),
                "model_provider": "gcp-deepmind",
                "model_name": "geminiflash1_5",
            }
            headers = {
                "Authorization": f"Bearer {bearer_token}",
                "Content-Type": "application/json"
            }
            try:
                response = requests.post(api_url, headers=headers, json=json_payload)
                response.raise_for_status()
                result_document_section = response.json().get("response")
                # print("result:", result_document_section)
            except requests.exceptions.RequestException as e:
                print(f"API request error: {e}")
            except Exception as e:
                print(f"Unexpected error: {e}")
            
            for i, chunk in enumerate(chunks):
                ##########################################
                ## questions

                query = """given that the following text from the document {title} on page {page_no} of total page {total_page}, here is the text:\n {chunk}\n\n 
                            what are some good questions to ask about the {section} section? Please respond with question and answers for 3 questions.

                            the questions need to be well defined. Try to use the text as much as possible when crafting the answer. Answers need to be at least 2 sentences long.

                            Please use the following format for the response:

                            **Question 1:**
                            **Answer 1:**

                            **Question 2:**
                            **Answer 2:**

                            **Question 3:**
                            **Answer 3:**
                            """.format(
                                page_no = page_no,
                                total_page = total_page,
                                chunk=chunk,
                                section=result_document_section,
                                title=title)

                json_payload = {
                    "query": query,
                    "model_provider": "gcp-deepmind",
                    "model_name": "geminiflash1_5",
                }
                headers = {
                    "Authorization": f"Bearer {bearer_token}",
                    "Content-Type": "application/json"
                }
                try:
                    response = requests.post(api_url, headers=headers, json=json_payload)
                    response.raise_for_status()
                    result = response.json().get("response")
                    # print("result:", result)
                except requests.exceptions.RequestException as e:
                    print(f"API request error: {e}")
                except Exception as e:
                    print(f"Unexpected error: {e}")
                
                ###################################################
                ## save out
                parts = result.split("**")  # Split the string at **
                parts_no = [[2, 4], [6, 8], [10, 12]]


                for i in range(0, 3):
                    try:
                        pt1 = parts_no[i][0]
                        pt2 = parts_no[i][1]

                        Q1 = pd.DataFrame(data={'section':[result_document_section],
                                                'title':[title],
                                                'file_name':[files[ii]],
                                                'document_type':['academic paper'],
                                                'subject':['science, chemistry, materials science'],
                                                'page': [page_no],
                                                'pg_word_ct':[word_count],
                                                'total_pages': [total_page],
                                                'contex': [chunks],
                                                'page_whole': [cleaned_string],
                                                'question':[parts[pt1]],
                                                'answer':[parts[pt2]]
                                                })
                        df_out = pd.concat([Q1, df_out], ignore_index=True)
                        df_out.to_csv('silver_data.csv', index=False)

                    except Exception as e:
                        print(f"Unexpected error: {e}")
                        # print("result:", result)
