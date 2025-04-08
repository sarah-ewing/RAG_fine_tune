import threading
import signal
import sys
from functions import Crawler, ticker # Import classes and functions from functions.py
import logging

# --- Signal Handler ---
def signal_handler(sig, frame):
    global stop_threads
    logging.info('You pressed Ctrl+C! Exiting Gracefully...')
    stop_threads = True
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# --- Main Function ---
def main():
    start_url = "https://www.asu.edu/"
    max_depth = 9

    total_processed_tracker = [0]
    ticker_thread = threading.Thread(target=ticker, args=(total_processed_tracker, 2))
    ticker_thread.daemon = True
    ticker_thread.start()

    crawler = Crawler(start_url, max_depth, NUM_THREADS, SAVE_FREQUENCY)
    crawler.run()

if __name__ == "__main__":
    main()