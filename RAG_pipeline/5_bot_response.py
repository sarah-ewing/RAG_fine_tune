import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--break-system-packages"])

# install('python-dotenv')

################################################
from dotenv import load_dotenv
import time
import os

load_dotenv()
PROJECT_TOKEN = os.environ.get("PROJECT_TOKEN")
PROJECT_KEY = os.environ.get("PROJECT_KEY") 

MODEL = os.environ.get("MODEL")
R_TYPE = os.environ.get("R_TYPE")
TOP_K = os.environ.get("TOP_K")
#################################################################
## get the token to run against the project env
import requests

# Define the endpoint URL
url = "https://api-main-poc.aiml.asu.edu/token"

# Set your project key as the Authorization Bearer Token
headers = {
    "Authorization": PROJECT_TOKEN,
    "Content-Type": "application/json"
}

# Define the request payload
payload = {
    "method": "generate_token",
    "details": {
        "asurite": "sewing12"
    }
}

# Send the POST request
response = requests.post(url, headers=headers, json=payload)

# Check the response
if response.status_code == 200:
    print("Token generated successfully!")
    # print("Response:", response.json())
    json_string = response.json()
    made_USER_TOKEN = json_string['token']
    
else:
    print("Failed to generate token.")
    print("Status Code:", response.status_code)
    print("Response:", response.text)
#################################################################
## load in the 'golden' data set and create space to record
import pandas as pd
half_g = pd.read_csv('half_golden.csv')
half_g = half_g[['section', 'title', 'file_name', 'document_type', 'page', 'total_pages', 'context', 'question', 'answer']]
half_g.rename(columns={'answer':'golden_answer'}, inplace = True)
half_g['model'] = MODEL
half_g['Retrieval_Type'] = R_TYPE ## Neighbor, Chunk, Document
half_g['Top_K'] = TOP_K
half_g['Response'] = ""
#################################################################
import requests
file_name = '5_exp_pipeline_{model}_{Retrieval_Type}_{top_k}.csv'.format(model=MODEL,
                                                                    Retrieval_Type=R_TYPE,
                                                                    top_k=TOP_K)

for ii in range(0, len(half_g['question'])):
    # Define the endpoint URL
    url = "https://api-main-poc.aiml.asu.edu/queryV2"

    # Set your Authorization Bearer Token (use the project token if you have it)
    headers = {
        "Authorization": made_USER_TOKEN,
        "Content-Type": "application/json"
    }

    # Define the request payload
    payload = {
        "project_id": PROJECT_KEY,
        "query": half_g['question'].iloc[ii],
        "enable_search": True,
        "enable_history": False,
        "history": []
    }

    # Send the POST request
    response = requests.post(url, headers=headers, json=payload)

    # Check the response
    if response.status_code == 200:
        print("   ", ii, "- Request successful!")
        # print("Response:", response.json())
    else:
        print("Failed to make request.")
        print("Status Code:", response.status_code)
        print("Response:", response.text)
    #################################################################
    ## record the answers
    question_response = response.json()
    # print(question_response['response'])
    half_g.loc[ii, 'Response'] = question_response['response']

    half_g.to_csv(file_name, index=False)
    time.sleep(1)