import requests
from lxml import html
import pandas as pd
import time
import threading

def get_url_tree(url, indent=0, results=None, ticker_interval=5, max_depth=None):
    """
    Recursively fetches URLs and their depths, storing them in a list of dictionaries.
    Includes a ticker to show progress and a maximum depth limit.

    Args:
        url (str): The starting URL.
        indent (int): The current depth (indentation level).
        results (list): A list to store the results (URLs and depths).
        ticker_interval (int): Interval (in seconds) to display the ticker.
        max_depth (int, optional): The maximum depth to crawl. If None, crawls without limit.

    Returns:
        list: A list of dictionaries, each containing 'url' and 'depth'.
    """
    if results is None:
        results = []

    if max_depth is not None and indent > max_depth:
        return results  # Stop recursion if max depth is exceeded

    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        tree = html.fromstring(response.text)
        links = tree.xpath('//a/@href')

        results.append({'url': url, 'depth': indent})  # Store the current URL and depth

        for link in links:
            absolute_link = link if link.startswith('http') else f"{url.split('/')[0]}//{url.split('/')[2]}{link}"
            get_url_tree(absolute_link, indent + 1, results, ticker_interval, max_depth)

    except requests.exceptions.RequestException as e:
        results.append({'url': url, 'depth': indent, 'error': f"Error accessing {url}: {e}"})
        print('  ' * indent + f"Error accessing {url}: {e}")
    except Exception as e:
        results.append({'url': url, 'depth': indent, 'error': f"Error processing {url}: {e}"})
        print('  ' * indent + f"Error processing {url}: {e}")

    return results

def ticker(results, interval):
    """Displays a ticker showing the number of URLs processed."""
    start_time = time.time()
    last_count = 0
    while True:
        current_count = len(results)
        if current_count != last_count:
            elapsed_time = time.time() - start_time
            print(f"Processed {current_count} URLs. Elapsed time: {elapsed_time:.2f} seconds", end='\r')
            last_count = current_count
        time.sleep(interval)

def main():
    start_url = "https://www.asu.edu/"
    results = []
    ticker_interval = 2
    max_depth = 3  # Set the maximum depth here

    # Start the ticker thread
    ticker_thread = threading.Thread(target=ticker, args=(results, ticker_interval))
    ticker_thread.daemon = True
    ticker_thread.start()

    url_data = get_url_tree(start_url, results=results, ticker_interval=ticker_interval, max_depth=max_depth)

    # Allow the ticker to display the final count before exiting
    time.sleep(ticker_interval + 0.5)

    # Create a Pandas DataFrame from the results
    df = pd.DataFrame(url_data)

    # Print the DataFrame
    print("\n\nFinal Results:")
    print(df)
    df.to_csv("url_tree.csv", index=False)

if __name__ == "__main__":
    main()