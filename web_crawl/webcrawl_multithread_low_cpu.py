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
import random  # Import the random module

import datetime

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

from readability import Document

def extract_main_content(html_content):
    if html_content:  # check for empty content.
        try:
            doc = Document(html_content)
            summary = doc.summary()
            text = html.fromstring(summary).text_content().strip()
            return text
        except Exception as e:
            print(f"Error extracting content: {e}")
            return ""
    else:
        return ""  # return empty string if html_content is empty.

def get_url_tree(start_url, max_depth=3, resume_file="crawl_state.pkl"):
    visited = set()
    results = []
    url_queue = deque([(start_url, 0)])
    base_url_netloc = urlparse(start_url).netloc
    run_count = 0

    if os.path.exists(resume_file):
        try:
            with open(resume_file, "rb") as f:
                saved_state = pickle.load(f)
                visited = saved_state["visited"]
                url_queue = saved_state["url_queue"]
                run_count = saved_state.get("run_count", 0)
                print(f"Resuming crawl from saved state. Processed {len(visited)} URLs.")
        except Exception as e:
            print(f"Error loading saved state: {e}")

    while url_queue:
        url, depth = url_queue.popleft()

        if depth > max_depth:
            continue

        if url in visited:
            continue

        visited.add(url)
        # print(url, datetime.datetime.now())

        excluded_extensions = {
        '.jpg', '.jpeg', '.png', '.gif', '.tif', '.stl', '.blend',
        '.mp3', '.wav', '.ogg', '.mp4', '.avi', '.mov', '.zip', '.rar',
        '.7z', '.dwg', '.dxf', '.obj', '.exe', '.dmg', '.apk', '.ipa',  '.app', '.msi'} ##'.pdf'

        try:
            # Skip large files
            parsed_url = urlparse(url)
            if parsed_url.path.lower().endswith(tuple(excluded_extensions)):
                print(f"Skipping large file: {url}")
                continue
            user_agent = random.choice(USER_AGENTS)  # Select a random user agent
            headers = {"User-Agent": user_agent}  # Create headers with the random user agent
            response = requests.get(url, headers=headers, timeout=10)  # Use the random user agent
            response.raise_for_status()

            if response.content:  # Add this check
                try:
                    tree = html.fromstring(response.content)

                    page_text = extract_main_content(response.content)

                    title_element = tree.xpath('//title/text()')
                    title = title_element[0].strip() if title_element else "Untitled"

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
                                url_queue.append((absolute_link, depth + 1))
                        except ValueError as e:
                            print(f"Skipping invalid URL: {link}, error: {e}")
                            continue  # skip the rest of the loop, and continue to the next link.

                    time.sleep(REQUEST_DELAY)
                except Exception as e:
                    print(f"Error parsing HTML for {url}: {e}")
            else:
                print(f"Empty content received for {url}")  # print when a page has no content.

        except requests.exceptions.RequestException as e:
            print(f"Error accessing {url}: {e}")

        run_count += 1
        if run_count % SAVE_FREQUENCY == 0:
            with open(resume_file, "wb") as f:
                pickle.dump({
                    "visited": visited,
                    "url_queue": url_queue,
                    "run_count": run_count,
                }, f)

            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y_%m_%d_%H")
            print(f"Saved crawl state after {run_count} runs. {now}")

            df = pd.DataFrame(results)
            formatted_datetime = now.strftime("%Y_%m_%d_%H")
            df.to_csv(rf"C:\programming_projects\ASU\web_crawl\web_data\webpage_analysis_run_{run_count}_{formatted_datetime}.csv", index=False)
            results = []  # Clear results after saving to CSV

    return results

def ticker(results, interval):
    start_time = time.time()
    last_count = 0
    while True:
        current_count = len(results)
        if current_count != last_count:
            elapsed_time = time.time() - start_time
            print(f"Processed {current_count} URLs. Elapsed time: {elapsed_time:.2f} seconds", end='\r')
            last_count = current_count
        time.sleep(interval)

def can_fetch(url, user_agent="*"):
    parsed_url = urlparse(url)
    robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        return rp.can_fetch(user_agent, url)
    except:
        return True

def main():
    start_url = "https://www.asu.edu/"
    max_depth = 9
    results = []

    ticker_thread = threading.Thread(target=ticker, args=(results, 2))
    ticker_thread.daemon = True
    ticker_thread.start()

    url_data = get_url_tree(start_url, max_depth=max_depth)

    time.sleep(1)

    df = pd.DataFrame(url_data)
    df.to_csv("webpage_analysis_v2.csv", index=False)

if __name__ == "__main__":
    main()
