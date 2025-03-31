import requests
from lxml import html
import pandas as pd
import time
from urllib.parse import urljoin, urlparse
from collections import deque
from urllib.robotparser import RobotFileParser
from transformers import pipeline
import os
import pickle

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

REQUEST_DELAY = 1

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

TOPIC_CATEGORIES = [
    "Academic Programs & Courses",
    "Admissions & Application Process",
    "Scholarships & Financial Aid",
    "Research & Innovation at ASU",
    "Student Life & Campus Activities",
    "ASU's Global & Online Education",
    "ASU's Commitment to Sustainability",
    "International Student Support",
    "ASU's AI & Tech Initiatives",
    "Sun Devil Athletics & Sports",
    "ASU's History & Rankings",
    "ASU Library & Research Resources",
    "Career Services & Job Support",
    "Housing & Campus Life",
    "Health, Wellness & Counseling Services",
    "ASU Welbeing and Security"
]

from readability import Document

def extract_main_content(html_content):
    try:
        doc = Document(html_content)
        summary = doc.summary()
        text = html.fromstring(summary).text_content().strip()
        return text
    except Exception as e:
        print(f"Error extracting content: {e}")
        return ""

def classify_topic(text):
    if not text.strip():
        return "Unknown"
    try:
        result = classifier(text, TOPIC_CATEGORIES)
        return result["labels"][0]
    except Exception as e:
        print(f"Error in topic classification: {e}")
        return "Unknown"

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
                results = saved_state.get("results", [])
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

        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            response.raise_for_status()
            tree = html.fromstring(response.text)

            page_text = extract_main_content(response.text)
            num_words = len(page_text.split())
            num_chars = len(page_text)

            title_element = tree.xpath('//title/text()')
            title = title_element[0].strip() if title_element else "Untitled"

            topic = classify_topic(page_text)

            results.append({
                'url': url,
                'depth': depth,
                'title': title,
                'topic': topic,
                'word_count': num_words,
                'char_count': num_chars,
                'page_text': page_text
            })

            links = tree.xpath('//a/@href')
            for link in links:
                absolute_link = urljoin(url, link)
                parsed_link = urlparse(absolute_link)

                if "asu.edu" in parsed_link.netloc and "#" not in absolute_link:
                    url_queue.append((absolute_link, depth + 1))

            time.sleep(REQUEST_DELAY)

        except requests.exceptions.RequestException as e:
            print(f"Error accessing {url}: {e}")

        run_count += 1
        if run_count % 2 == 0:
            with open(resume_file, "wb") as f:
                pickle.dump({
                    "visited": visited,
                    "url_queue": url_queue,
                    "results": results,
                    "run_count": run_count,
                }, f)
            print(f"Saved crawl state after {run_count} runs.")
            df = pd.DataFrame(results)
            df.to_csv(rf"C:\programming_projects\ASU\web_crawl\web_data\webpage_analysis_run_{run_count}.csv", index=False)
            results = []
            ticker(results, 0) # run ticker during loop

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
        if len(results) > 0:
            break

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

    ticker(results, 2) # ticker called directly.

    url_data = get_url_tree(start_url, max_depth=max_depth)

    time.sleep(1)

    df = pd.DataFrame(url_data)
    df.to_csv("webpage_analysis_v2.csv", index=False)

if __name__ == "__main__":
    main()