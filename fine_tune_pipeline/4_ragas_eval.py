from dotenv import load_dotenv

import os

load_dotenv()
ASU_key = os.environ.get("ASU_key") 

import requests
endpoint_url = 'https://api-dev-poc.aiml.asu.edu/eval'
headers = {
    "Authorization": f"Bearer {ASU_key}",
    "Content-Type": "application/json"
}

#############################################################################
import pandas as pd
# df = pd.read_csv('qa_results_evaluated_v2.csv')
# df.head()
# df['ragas_bleu'] = 0.0
# df['ragas_precision_reference'] = 0.0
# df['ragas_faith'] = 0.0
# df['ragas_fact'] = 0.0
# df['regas_sem_sim'] = 0.0
# df['regas_rogue'] = 0.0
start_ii = 1434
if start_ii > 0:
    df = pd.read_csv('qa_results_RAGAS_evaluated.csv')

for ii in range(start_ii, 6635):
    print(ii)
    questiony = df['question'].iloc[ii]
    answery = df['answer'].iloc[ii]
    contexty = df['page_text'].iloc[ii]

    ##################################
    ### Bluescore
    payload_blue = {
        "model_provider": "openai",
        "model_name": "gpt-4o",
        "metric": "bleuscore",
        "parameters": {
            "user_input": questiony,
            "response": answery,
            "reference": contexty,
        }
    }
    try:
        response = requests.post(endpoint_url, headers=headers, json=payload_blue)
        response.raise_for_status()
        result = response.json()
        print("result:", result)
        #### SAVE BLEU score
        df.loc[ii,'ragas_bleu'] = result['score']
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    ##################################
    ### Context Precision With Reference
    # payload_precision_reference = {
    #     "model_provider": "openai",
    #     "model_name": "gpt-4o",
    #     "metric": "context_precision_with_reference",
    #     "parameters": {
    #         "user_input": questiony,
    #         "reference": answery,
    #         "retrieved_contexts": [
    #             contexty,
    #         ]
    #     }
    # }
    # try:
    #     response = requests.post(endpoint_url, headers=headers, json=payload_precision_reference)
    #     response.raise_for_status()
    #     result = response.json()
    #     print("result:", result)
    #     #### SAVE Precision With Reference score
    #     df.loc[ii,'ragas_precision_reference'] = result['score']
    # except requests.exceptions.RequestException as e:
    #     print(f"API request error: {e}")
    # except Exception as e:
    #     print(f"Unexpected error: {e}")

    ##################################
    # ### RAGAS Faithfulness
    # payload_ragas_faith = {
    #     "model_provider": "openai",
    #     "model_name": "gpt-4o",
    #     "metric": "faithfulness",
    #     "parameters": {
    #         "user_input": questiony,
    #         "response": answery,
    #         "retrieved_contexts": [
    #             contexty,
    #         ]
    #     }
    # }
    # try:
    #     response = requests.post(endpoint_url, headers=headers, json=payload_ragas_faith)
    #     response.raise_for_status()
    #     result = response.json()
    #     print("result:", result)
    #     #### SAVE RAGAS Faithfulness score
    #     df.loc[ii,'ragas_faith'] = result['score']
    # except requests.exceptions.RequestException as e:
    #     print(f"API request error: {e}")
    # except Exception as e:
    #     print(f"Unexpected error: {e}")

    ##################################
    ### Factual correctness
    # print("Factual Correctness")
    payload_fact = {
        "model_provider": "openai",
        "model_name": "gpt-4o",
        "metric": "factual_correctness",
        "parameters": {
            "response": str(answery),
            "reference": str(contexty)
        }
    }
    try:
        response = requests.post(endpoint_url, headers=headers, json=payload_fact)
        response.raise_for_status()
        result = response.json()
        print("result:", result)
        #### SAVE Factual correctness score
        df.loc[ii,'ragas_fact'] = result['score']
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    ##################################
    ### Semantic similarity
    payload_sem_sim = {
        "model_provider": "openai",
        "model_name": "gpt-4o",
        "metric": "semantic_similarity",
        "parameters": {
            "response": answery,
            "reference": contexty
        }
    }
    try:
        response = requests.post(endpoint_url, headers=headers, json=payload_sem_sim)
        response.raise_for_status()
        result = response.json()
        print("result:", result)
        #### SAVE Factual correctness score
        df.loc[ii,'regas_sem_sim'] = result['score']
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

    ##################################
    ### Rouge score
    payload_rogue = {
        "model_provider": "openai",
        "model_name": "gpt-4o",
        "metric": "rouge_score",
        "parameters": {
            "response": answery,
            "reference": contexty
        }
    }
    try:
        response = requests.post(endpoint_url, headers=headers, json=payload_rogue)
        response.raise_for_status()
        result = response.json()
        print("result:", result)
        #### SAVE Factual correctness score
        df.loc[ii,'regas_rogue'] = result['score']
    except requests.exceptions.RequestException as e:
        print(f"API request error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


    df.to_csv('qa_results_RAGAS_evaluated.csv')