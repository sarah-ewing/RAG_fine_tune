import pandas as pd
from send2trash import send2trash
import glob
import datetime

def merge_and_delete(file_list, output_file):
    """Merges a list of CSV files, drops duplicates, and deletes originals."""
    all_data = pd.DataFrame([])
    for file in file_list:
        try:
            df_read = pd.read_csv(file)
            all_data = pd.concat([all_data, df_read])
        except Exception as e:
            print(f"Error reading {file}: {e}")
            if "No columns to parse from file" in str(e):
                print(f'Deleting: {file}')
                send2trash(file)
    all_data.drop_duplicates(subset=['title', 'page_text'], inplace=True)  # Remove duplicate rows
    all_data.to_csv(output_file, index=False)
    print(F"output file made {output_file} - ", datetime.datetime.now())

    for file in file_list:
        try:
            send2trash(file)
        except Exception as e:
            print(f"Error reading {file}: {e}")

def process_batch(file_pattern, batch_size, output_prefix):
    """Processes files in batches, merging and deleting."""
    files = glob.glob(file_pattern)
    for i in range(0, len(files), batch_size):
        batch = files[i:i + batch_size]
        output_file = f"C:\\programming_projects\\ASU\\web_crawl\\web_data_merged\\{output_prefix}_batch_{i // batch_size}.csv"
        
        merge_and_delete(batch, output_file)

# Example usage:
file_pattern = "C:\\programming_projects\\ASU\\web_crawl\\web_data\\*.csv"  # Adjust to your file pattern (e.g., "data_*.csv")
batch_size = 500
output_prefix = "merged"

process_batch(file_pattern, batch_size, output_prefix)