import requests
from lxml import html
import pandas as pd
import time
import threading
import csv

def get_urls(url, indent, max_depth, results_file):
    if max_depth is not None and indent > max_depth:
        return

    try:
        response = requests.get(url, timeout=10) #added timeout
        response.raise_for_status()
        tree = html.fromstring(response.text)
        links = tree.xpath('//a/@href')

        with open(results_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([url, indent])

        for link in links:
            absolute_link = link if link.startswith('http') else f"{url.split('/')[0]}//{url.split('/')[2]}{link}"
            get_urls(absolute_link, indent + 1, max_depth, results_file)

    except requests.exceptions.RequestException as e:
        with open(results_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([url, indent, f"Error: {e}"])
        print(f"Error accessing {url}: {e}")
    except Exception as e:
        with open(results_file, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([url, indent, f"Error: {e}"])
        print(f"Error processing {url}: {e}")

def ticker(results_file, interval):
    start_time = time.time()
    last_count = 0
    while True:
        try:
            with open(results_file, 'r', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                current_count = sum(1 for row in reader)
            if current_count != last_count:
                elapsed_time = time.time() - start_time
                print(f"Processed {current_count} URLs. Elapsed time: {elapsed_time:.2f} seconds", end='\r')
                last_count = current_count
        except FileNotFoundError:
            pass #file not created yet.
        time.sleep(interval)

def main():
    start_url = "https://www.asu.edu/"
    results_file = "url_data.csv"
    ticker_interval = 2
    max_depth = 3

    with open(results_file, 'w', newline='', encoding='utf-8') as csvfile: #create or clear file
        writer = csv.writer(csvfile)
        writer.writerow(["url", "depth", "error"])

    ticker_thread = threading.Thread(target=ticker, args=(results_file, ticker_interval))
    ticker_thread.daemon = True
    ticker_thread.start()

    get_urls(start_url, 0, max_depth, results_file)

    time.sleep(ticker_interval + 0.5)

    df = pd.read_csv(results_file)
    print("\n\nFinal Results:")
    print(df.head())
    print(df.shape)

if __name__ == "__main__":
    main()