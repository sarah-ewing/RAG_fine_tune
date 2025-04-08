import requests
from lxml import html
import pandas as pd
import time
import threading
from urllib.parse import urljoin, urlparse
from collections import deque
import os
import pickle
import random
import datetime
import signal
import sys
import gc
import logging
import time

from readability import Document

# --- Global Variables ---
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
NUM_THREADS = 5
SAVE_FREQUENCY = 50
stop_threads = False

def ticker(total_processed_variable, interval):
    """
    Periodically logs the number of processed URLs and elapsed time.

    Args:
        total_processed_variable: A list containing the number of processed URLs.
        interval: The time interval (in seconds) between log messages.
    """
    start_time = time.time()
    last_count = 0
    while not stop_threads:
        current_count = total_processed_variable[0]
        if current_count != last_count:
            elapsed_time = time.time() - start_time
            logging.info(f"Processed {current_count} URLs. Elapsed time: {elapsed_time:.2f} seconds")
            last_count = current_count
        time.sleep(interval)

# --- Request Manager Class ---
class RequestManager:
    def __init__(self, user_agents, request_delay, bucket):
        self.user_agents = user_agents
        self.request_delay = request_delay
        self.bucket = bucket

    def get_page(self, url):
        self.bucket.wait_for_tokens()
        user_agent = random.choice(self.user_agents)
        headers = {"User-Agent": user_agent}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        return response

    def delay(self):
        time.sleep(self.request_delay)

# --- Page Processor Class ---
class PageProcessor:
    @staticmethod
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

    def process_url(self, url, depth, max_depth, visited, results, url_queue, lock, request_manager):
        if depth > max_depth:
            return

        with lock:
            if url in visited:
                return
            visited.add(url)

        try:
            response = request_manager.get_page(url)

            if response.content:
                try:
                    tree = html.fromstring(response.content)
                    page_text = self.extract_main_content(response.content)
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

                    request_manager.delay()
                except Exception as e:
                    print(f"Error parsing HTML for {url}: {e}")
            else:
                print(f"Empty content received for {url}")

        except requests.exceptions.RequestException as e:
            print(f"Error accessing {url}: {e}")

# --- Result Saver Class ---
class ResultSaver:
    @staticmethod
    def save_results(results, save_count, formatted_datetime):
        thread_id = threading.get_ident()
        df = pd.DataFrame(results)
        df.to_csv(f"webpage_analysis_v3_{formatted_datetime}_thread_{thread_id}_part_{save_count}.csv", index=False)
        print(f"Saved {len(results)} results from thread {thread_id} to CSV part {save_count}.")

# --- Token Bucket Class ---
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

# --- Crawler Class ---
class Crawler:
    def __init__(self, start_url, max_depth, num_threads, save_frequency, resume_file="crawl_state.pkl"):
        self.start_url = start_url
        self.max_depth = max_depth
        self.num_threads = num_threads
        self.save_frequency = save_frequency
        self.resume_file = resume_file
        self.visited = set()
        self.url_queue = deque([(start_url, 0)])
        self.lock = threading.Lock()
        self.threads = []
        self.total_processed = [0]
        self.save_count = 0
        self.formatted_datetime = datetime.datetime.now().strftime("%Y_%m_%d_%H")
        self.bucket = TokenBucket(rate=2, capacity=5)
        self.request_manager = RequestManager(USER_AGENTS, REQUEST_DELAY, self.bucket)
        self.page_processor = PageProcessor()

    def load_state(self):
        if os.path.exists(self.resume_file):
            try:
                with open(self.resume_file, "rb") as f:
                    saved_state = pickle.load(f)
                    self.visited = saved_state["visited"]
                    self.url_queue = deque(saved_state["url_queue"])
                    self.total_processed[0] = len(self.visited)
                    print(f"Resuming crawl from saved state. Processed {self.total_processed[0]} URLs.")
            except Exception as e:
                print(f"Error loading saved state: {e}")

    def worker(self):
        thread_results = []
        while not stop_threads:
            with self.lock:
                if not self.url_queue:
                    break
                url, depth = self.url_queue.popleft()
                self.total_processed[0] += 1
            self.page_processor.process_url(url, depth, self.max_depth, self.visited, thread_results, self.url_queue, self.lock, self.request_manager)
            if len(thread_results) >= self.save_frequency:
                ResultSaver.save_results(thread_results, self.save_count, self.formatted_datetime)
                with self.lock:
                    thread_results.clear()
                gc.collect()
        if len(thread_results) > 0:
            ResultSaver.save_results(thread_results, self.save_count, self.formatted_datetime)
        print("worker thread finished")

    def run(self):
        self.load_state()
        for i in range(self.num_threads):
            try:
                t = threading.Thread(target=self.worker)
                self.threads.append(t)
                t.start()
                logging.info(f"Thread {i+1} started.")
            except Exception as e:
                logging.error(f"Error starting thread {i+1}: {e}")

        for t in self.threads:
            while t.is_alive():
                t.join(1)
                if stop_threads:
                    logging.info("Force exiting")
                    sys.exit()
        logging.info("All threads finished.")
        logging.info("Program exiting.")

