import os
import pandas as pd
import requests
import datetime
import re
##########################################################
from dotenv import load_dotenv
import os

load_dotenv()
ASU_key = os.environ.get("ASU_key")
file_name = os.environ.get("file_name")
LLM_url = os.environ.get("LLM_url")

now = datetime.datetime.now()
formatted_datetime = now.strftime("%Y_%m_%d_%H")

##########################################################

directory_path = 'C:\\programming_projects\\RAG_fine_tune\\RAG_pipeline_ASU_website\\data\\'
print(directory_path + 'chunked_' +file_name)
chunked_df = pd.read_csv(directory_path + 'chunked_' +file_name, nrows=100)

##########################################################
df_out = pd.DataFrame([])
for i in range(0, 12): ##len(chunked_df['chunked_text'])):
    print(i, datetime.datetime.now())

    cleaned_string = chunked_df['cleaned_text'].loc[i]

    chunk = chunked_df['chunked_text'].loc[i]
    title = chunked_df['title'].loc[i]
    url = chunked_df['url'].loc[i]
    chunked_word_count = chunked_df['chunked_word_count'].loc[i]
    orig_word_count = chunked_df['orig_word_count'].loc[i]
    ##########################################
    ## text type
    bearer_token = ASU_key
    json_payload = {
        "query": "what is the topic of the following text from: {cleaned_string}? only respond with the topic, no other text. Please make the topic 3 words or less".format(cleaned_string=cleaned_string),
        "model_provider": "gcp-deepmind",
        "model_name": "geminiflash2",
    }
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(LLM_url, headers=headers, json=json_payload)
        response.raise_for_status()
        result_document_section = response.json().get("response")
        # print("result:", result_document_section)
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    ##########################################
    ## questions

    query = """given that the following text from the webpage {title} on url {url}, here is a text chunk limited to 500 words:\n {chunk}\n\n 
                what are some good questions to ask about the text chunk? Please respond with a question and 3 different answers for each question.  There should be a total of 3 questions, with having 3 different answers (for a total of 9 unique answers).

                the questions need to be well defined. Try to use the text as much as possible when crafting the answer. Answers need to be at least 2 sentences long. Do not use the phrase "The text," and avoid similar language. Rephrase the question in the answer.

                Please use the following format for the response:

                **Question 1:**
                **Question 1 Answer 1:**
                **Question 1 Answer 2:**
                **Question 1 Answer 3:**

                **Question 2:**
                **Question 2 Answer 1:**
                **Question 2 Answer 2:**
                **Question 2 Answer 3:**

                **Question 3:**
                **Question 3 Answer 1:**
                **Question 3 Answer 2:**
                **Question 3 Answer 3:**
                """.format(
                    title = title,
                    url = url,
                    chunk=chunk)

    json_payload = {
        "query": query,
        "model_provider": "gcp-deepmind",
        "model_name": "geminiflash2",
    }
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(LLM_url, headers=headers, json=json_payload)
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
    parts_no = [[2, 4], [2, 6], [2,8], [10, 12], [10, 14], [10, 16], [18, 20], [18, 22], [18, 24]]


    for jj in range(0, 9):
        try:
            pt1 = parts_no[jj][0]
            pt2 = parts_no[jj][1]

            Question = re.sub(r'[^a-zA-Z0-9.,!?\s]', ' ', str(parts[pt1]))
            Question = Question.replace(r'\s+', ' ').strip()
            Answer = re.sub(r'[^a-zA-Z0-9.,!?\s]', ' ', str(parts[pt2]))
            Answer = Answer.replace(r'\s+', ' ').strip()


            Q1 = pd.DataFrame(data={'section':[result_document_section],
                                    'title':[title],
                                    'url': [url],
                                    'document_type':['web page'],
                                    'chunked_word_count':[chunked_word_count],
                                    'orig_word_count':[orig_word_count],
                                    'contex': [chunk],
                                    'question':[Question],
                                    'answer':[Answer]
                                    })
            df_out = pd.concat([Q1, df_out], ignore_index=True)



        except Exception as e:
            print(f"Unexpected error: {e}")
            # print("result:", result)

    if (i % 100 == 0) & (i != 0):
        df_out.to_csv(directory_path+'silver_data\\'+f'silver_data_{formatted_datetime}__{i}.csv', index=False)
        df_out = pd.DataFrame([])
