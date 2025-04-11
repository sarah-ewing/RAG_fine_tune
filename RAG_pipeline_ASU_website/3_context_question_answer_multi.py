import threading
import time
import pandas as pd
import os
from dotenv import load_dotenv
from queue import Queue, Empty
import datetime
import requests
import re

from dotenv import load_dotenv
load_dotenv()

ASU_key = os.environ.get("ASU_key")
LLM_url = os.environ.get("LLM_url")
print(LLM_url)

def make_llm_request(query, api_key, api_url):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    json_payload = {"query": query, "model_provider": "gcp-deepmind", "model_name": "geminiflash2"}
    try:
        response = requests.post(api_url, headers=headers, json=json_payload)
        response.raise_for_status()
        return response.json().get("response")
    except requests.exceptions.RequestException as e:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"{timestamp} API request error: {e}")
        return None
    except Exception as e:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"{timestamp} Unexpected error: {e}")
        return None

def parse_cqa_response(llm_response, title, url, chunked_word_count, orig_word_count, chunk):
    df_out = pd.DataFrame([])
    if llm_response:
        parts = llm_response.split("**")
        parts_no = [[2, 4], [2, 6], [2,8], [10, 12], [10, 14], [10, 16], [18, 20], [18, 22], [18, 24]]
        for jj in range(0, 9):
            try:
                pt1 = parts_no[jj][0]
                pt2 = parts_no[jj][1]
                Question = re.sub(r'[^a-zA-Z0-9.,!?\s]', ' ', str(parts[pt1])).replace(r'\s+', ' ').strip()
                Answer = re.sub(r'[^a-zA-Z0-9.,!?\s]', ' ', str(parts[pt2])).replace(r'\s+', ' ').strip()
                Q1 = pd.DataFrame(data={'title':[title], 'url': [url], 'document_type':['web page'], 'chunked_word_count':[chunked_word_count], 'orig_word_count':[orig_word_count], 'contex': [chunk], 'question':[Question], 'answer':[Answer]})
                df_out = pd.concat([Q1, df_out], ignore_index=True)
            except Exception as e:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                print(f"{timestamp} Error parsing response: {e}")
    return df_out

def cqa_api(chunked_df, i, ASU_key, LLM_url):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"{timestamp} Processing row index: {i}")
    chunk = chunked_df['chunked_text'].iloc[i]
    title = chunked_df['title'].iloc[i]
    url = chunked_df['url'].iloc[i]
    chunked_word_count = chunked_df['chunked_word_count'].iloc[i]
    orig_word_count = chunked_df['orig_word_count'].iloc[i]

    qa_query = f"""given that the following text from the webpage {title} on url {url}, here is a text chunk limited to 500 words:\n {chunk}\n\n 
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
    llm_response = make_llm_request(qa_query, ASU_key, LLM_url)

    return parse_cqa_response(llm_response, title, url, chunked_word_count, orig_word_count, chunk)

def worker(thread_id, all_data, task_queue, output_queues, cqa_api, ASU_key, LLM_url):
    """Worker function that gets tasks from the queue and processes them."""
    while True:
        try:
            start_index, end_index = task_queue.get(timeout=1)  # Unpack the tuple

            if start_index >= len(all_data):
                task_queue.task_done()
                break

            # The rest of your worker function remains the same
            thread_data_chunk = all_data.iloc[start_index:end_index].copy()
            all_processed_dfs = []
            for i in thread_data_chunk.index: # Iterate through the rows of the chunk
                output_df = cqa_api(all_data, i, ASU_key, LLM_url)
                if not output_df.empty:
                    all_processed_dfs.append(output_df)

            if all_processed_dfs:
                combined_df = pd.concat(all_processed_dfs, ignore_index=True)
                output_queues[thread_id - 1].put((start_index, combined_df))
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                print(f"{timestamp} Thread {thread_id}: Processed rows {start_index} to {end_index - 1} and added {len(combined_df)} rows to queue.")
            else:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                print(f"{timestamp} Thread {thread_id}: Processed rows {start_index} to {end_index - 1}, no output generated.")

            task_queue.task_done()
            time.sleep(0.1)
        except Empty:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            print(f"{timestamp} Thread {thread_id}: Task queue is empty, exiting.")
            break
        except Exception as e:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            print(f"{timestamp} Thread {thread_id}: An error occurred in worker: {e}")
            break

def writer_worker(queue, thread_id, all_data):
    """Worker function that takes dataframes and their start index from a queue and appends to a local dataframe until 10 original rows are processed, then writes to CSV."""
    local_df = pd.DataFrame()
    processed_original_rows = set()
    while True:
        try:
            start_index, thread_data = queue.get(timeout=1)
            if thread_data.empty:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                print(f"{timestamp} Writer Thread {thread_id}: Received stop signal. Processed {len(processed_original_rows)} original rows.")
                break

            local_df = pd.concat([local_df, thread_data], ignore_index=True)
            for i in range(start_index, start_index + (len(thread_data) // 9)): # Assuming 9 output rows per input row
                processed_original_rows.add(i)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            print(f"{timestamp} Writer Thread {thread_id}: Received {len(thread_data)} rows, now processed {len(processed_original_rows)} original rows.")
            queue.task_done()

            if len(processed_original_rows) >= 10:
                output_start_index = min(processed_original_rows)
                filename = directory_path+'silver_data\\'+f"processed_data_thread_{thread_id}_start_{output_start_index}_10rows.csv"
                local_df.to_csv(filename, index=False)
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                print(f"{timestamp} Writer Thread {thread_id}: Saved {len(local_df)} rows (from 10 original) to {filename}.")
                local_df = pd.DataFrame()
                processed_original_rows = set()

        except Empty:
            pass
        except Exception as e:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            print(f"{timestamp} Writer Thread {thread_id}: An error occurred in writer: {e}")
            break

    # Save any remaining data if the stop signal is received before 10 rows
    if not local_df.empty:
        output_start_index = min(processed_original_rows) if processed_original_rows else "partial"
        filename = directory_path+'silver_data\\'+f"processed_data_thread_{thread_id}_start_{output_start_index}_partial.csv"
        local_df.to_csv(filename, index=False)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"{timestamp} Writer Thread {thread_id}: Saved remaining {len(local_df)} rows to {filename}.")

if __name__ == "__main__":
    threads = []
    writer_threads = []
    num_threads = 3
    rows_per_thread = 100  # Process 100 rows at a time
    original_rows_per_writer_file = 10

    load_dotenv()
    file_name = os.environ.get("file_name")
    directory_path = 'C:\\programming_projects\\RAG_fine_tune\\RAG_pipeline_ASU_website\\data\\'
    file_path = directory_path + 'chunked_' + file_name
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"{timestamp} {file_path}")

    try:
        all_data = pd.read_csv(file_path, nrows=5900) # Increased nrows for testing
        total_rows = len(all_data)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"{timestamp} Total rows: {total_rows}, using {num_threads} processing threads and {num_threads} writer threads.")

        start_processing_row = 5500
        task_queue = Queue()
        output_queues = [Queue() for _ in range(num_threads)]
        chunk_size = rows_per_thread  # Use the defined rows_per_thread

        # Populate the task queue with chunks of 100 rows
        for i in range(start_processing_row, total_rows, chunk_size):
            end_index = min(i + chunk_size, total_rows)
            task_queue.put((i, end_index))

        # Create and start the processing threads
        for i in range(num_threads):
            thread = threading.Thread(target=worker, args=(i + 1, all_data, task_queue, output_queues, cqa_api, ASU_key, LLM_url))
            threads.append(thread)
            thread.daemon = True
            thread.start()

        # Create and start the writer threads
        for i in range(num_threads):
            writer_thread = threading.Thread(target=writer_worker, args=(output_queues[i], i + 1, all_data))
            writer_threads.append(writer_thread)
            writer_thread.daemon = True
            writer_thread.start()

        # Wait for all tasks to be processed
        task_queue.join()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"{timestamp} All tasks in the queue have been processed by worker threads.")

        # Signal writer threads to stop
        for q in output_queues:
            q.put((None, pd.DataFrame()))

        # Wait for writer threads to finish
        for wt in writer_threads:
            wt.join()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"{timestamp} All writer threads have finished.")

    except FileNotFoundError:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"{timestamp} Error: File not found at {file_path}")
    except Exception as e:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"{timestamp} An error occurred in main: {e}")