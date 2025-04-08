import requests
from lxml import html
import pandas as pd
import time
import threading
from urllib.parse import urljoin, urlparse
from collections import deque
from urllib.robotparser import RobotFileParser
import os
import pickle
import random
import datetime
import signal
import sys

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

now = datetime.datetime.now()
formatted_datetime = now.strftime("%Y_%m_%d_%H")

print(formatted_datetime)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Windows NT 10.0; Trident/7.0; rv:11.0) like Gecko",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    'Mozilla/5.0 (Linux; U; Android 4.0.4; en-gb; GT-I9300 Build/IMM76D) AppleWebKit/534.30 (KHTML, like Gecko) Version/4.0 Mobile Safari/534.30',
    'Mozilla/5.0 (iPad; U; CPU iPhone OS 3_2 like Mac OS X; en-us) AppleWebKit/531.21.10 (KHTML, like Gecko) Version/4.0.4 Mobile/7B314 Safari/531.21.10',
    "Mozilla/5.0 (X11; Linux i686 on x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2909.25 Safari/537.36"
]

REQUEST_DELAY = 1
SAVE_FREQUENCY = 100
NUM_THREADS = 5

from readability import Document

stop_threads = False

def signal_handler(sig, frame):
    global stop_threads
    print('You pressed Ctrl+C! Exiting Gracefully...')
    stop_threads = True
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

class TokenBucket:
    def __init__(self, rate, capacity):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.time()

    def consume(self, tokens=1):
        now = time.time()
        self.tokens = min(self.capacity, self.tokens + (now - self.last_refill) * self.rate)
        self.last_refill = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def wait_for_tokens(self, tokens=1):
        while not self.consume(tokens):
            time.sleep(0.1)

def extract_main_content(html_content):
    if html_content:
        try:
            doc = Document(html_content)
            summary = doc.summary()
            text = html.fromstring(summary).text_content().strip()
            return text
        except Exception as e:
            print(f"Error extracting content: {e}")
            return ""
    else:
        return ""

def process_url(url, depth, max_depth, visited, results, url_queue, lock, bucket):
    if depth > max_depth:
        return

    with lock:
        if url in visited:
            return
        visited.add(url)

    try:
        bucket.wait_for_tokens()
        user_agent = random.choice(USER_AGENTS)
        headers = {"User-Agent": user_agent}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        if response.content:
            try:
                tree = html.fromstring(response.content)
                page_text = extract_main_content(response.content)
                title_element = tree.xpath('//title/text()')
                title = title_element[0].strip() if title_element else "Untitled"

                with lock:
                    results.append({
                        'url': url,
                        'depth': depth,
                        'title': title,
                        'topic': None,
                        'word_count': None,
                        'char_count': None,
                        'page_text': page_text
                    })

                links = tree.xpath('//a/@href')
                for link in links:
                    try:
                        absolute_link = urljoin(url, link)
                        parsed_link = urlparse(absolute_link)

                        if "asu.edu" in parsed_link.netloc and "#" not in absolute_link:
                            with lock:
                                url_queue.append((absolute_link, depth + 1))
                    except ValueError as e:
                        print(f"Skipping invalid URL: {link}, error: {e}")
                        continue

                time.sleep(REQUEST_DELAY)
            except Exception as e:
                print(f"Error parsing HTML for {url}: {e}")
        else:
            print(f"Empty content received for {url}")

    except requests.exceptions.RequestException as e:
        print(f"Error accessing {url}: {e}")

results = [] #global result list.

def save_results(results, save_count, formatted_datetime):
    thread_id = threading.get_ident()
    df = pd.DataFrame(results)
    df.to_csv(f"C:\\programming_projects\\ASU\\web_crawl\\web_data\\webpage_analysis_v3_{formatted_datetime}_thread_{thread_id}_part_{save_count}.csv", index=False)
    print(f"Saved {len(results)} results from thread {thread_id} to CSV part {save_count}.")


def get_url_tree_multithreaded(start_url, max_depth=3, resume_file="crawl_state.pkl"):
    visited = set()
    url_queue = deque([(start_url, 0)])
    lock = threading.Lock()
    threads = []
    total_processed = [0]
    save_count = 0
    bucket = TokenBucket(rate=2, capacity=5)

    if os.path.exists(resume_file):
        try:
            with open(resume_file, "rb") as f:
                saved_state = pickle.load(f)
                visited = saved_state["visited"]
                url_queue = deque(saved_state["url_queue"])
                total_processed[0] = len(visited)
                print(f"Resuming crawl from saved state. Processed {total_processed[0]} URLs.")
        except Exception as e:
            print(f"Error loading saved state: {e}")

    def worker(bucket):
        global stop_threads
        thread_results = [] # thread local result list
        while not stop_threads:
            with lock:
                if not url_queue:
                    break
                url, depth = url_queue.popleft()
                total_processed[0] += 1
            process_url(url, depth, max_depth, visited, thread_results, url_queue, lock, bucket)
        if len(thread_results) > 0:
            save_results(list(thread_results), save_count, formatted_datetime)
        print("worker thread finished")

    for _ in range(NUM_THREADS):
        t = threading.Thread(target=worker, args=(bucket,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    if len(results) > 0:
        save_results(list(results), save_count, formatted_datetime)

    return results, total_processed[0]

def ticker(total_processed_variable, interval):
    start_time = time.time()
    last_count = 0
    while not stop_threads:
        current_count = total_processed_variable[0]
        if current_count != last_count:
            elapsed_time = time.time() - start_time
            print(f"Processed {current_count} URLs. Elapsed time: {elapsed_time:.2f} seconds", end='\r')
            last_count = current_count
        time.sleep(interval)

def main():
    start_url = "https://www.asu.edu/"
    max_depth = 9

    total_processed_tracker = [0]
    ticker_thread = threading.Thread(target=ticker, args=(total_processed_tracker, 2))
    ticker_thread.daemon = True
    ticker_thread.start()

    try:
        url_data, total_processed = get_url_tree_multithreaded(start_url, max_depth=max_depth)
        total_processed_tracker[0] = total_processed
    except SystemExit:
        print("Program exiting")

    df = pd.DataFrame(url_data)
    df.to_csv(f"webpage_analysis_v3_{datetime.datetime.now().strftime('%Y_%m_%d_%H')}_final.csv", index=False)

if __name__ == "__main__":
    main()