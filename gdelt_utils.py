import os
import subprocess
import zipfile
import pandas as pd
import urllib.request
import time
from datetime import datetime, timedelta

from config import TEMP_DIR, GDELT_BASE_URL, GDELT_EVENT_COLUMNS

def check_internet_connection(url='http://www.google.com/', timeout=5):
    """Checks for a valid internet connection by making a request to a reliable server."""
    try:
        urllib.request.urlopen(url, timeout=timeout)
        return True
    except (urllib.error.URLError, ConnectionResetError):
        return False

def download_gdelt_file(timestamp_str: str) -> tuple[str, str]:
    """
    Downloads a GDELT 15-minute export.CSV.zip file.
    Returns (temp_zip_path, temp_csv_path) if successful, otherwise raises an exception.
    """
    file_name = f"{timestamp_str}.export.CSV.zip"
    url = f"{GDELT_BASE_URL}{file_name}"
    temp_zip_path = os.path.join(TEMP_DIR, file_name)
    temp_csv_path = temp_zip_path.replace('.zip', '')

    os.makedirs(TEMP_DIR, exist_ok=True)

    try:
        # Use silent curl with timeouts
        subprocess.run(["curl", "-L", "-s", "--connect-timeout", "10", "--max-time", "30", "-o", temp_zip_path, url], check=True)
        return temp_zip_path, temp_csv_path
    except subprocess.CalledProcessError as e:
        raise Exception(f"Download failed for {url}: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred during download of {url}: {e}")

def extract_gdelt_csv(temp_zip_path: str, temp_csv_path: str) -> pd.DataFrame | None:
    """
    Extracts the CSV from the zip file and reads it into a pandas DataFrame.
    Cleans up the temporary files. Returns None if the file is empty.
    """
    try:
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(TEMP_DIR)
        
        # Check for empty file before parsing
        if os.path.getsize(temp_csv_path) == 0:
            print(f"  - File for interval is empty. Skipping.")
            return None

        df = pd.read_csv(
            temp_csv_path,
            sep='\t',
            header=None,
            dtype=str,
            encoding='latin1',
            usecols=GDELT_EVENT_COLUMNS['indices']
        )
        df.columns = GDELT_EVENT_COLUMNS['names']
        
        # Drop rows where event_root_code is NaN or empty, as these are not valid events
        df.dropna(subset=['event_root_code'], inplace=True)
        df = df[df['event_root_code'] != '']

        return df
    except pd.errors.EmptyDataError:
        print(f"  - No columns to parse, file is effectively empty. Skipping.")
        return None
    except zipfile.BadZipFile:
        raise Exception(f"Downloaded file {temp_zip_path} is not a valid zip file.")
    except Exception as e:
        raise Exception(f"An unexpected error occurred during file extraction or parsing: {e}")
    finally:
        # Clean up temp files
        if os.path.exists(temp_zip_path): os.remove(temp_zip_path)
        if os.path.exists(temp_csv_path): os.remove(temp_csv_path)

def get_gdelt_data_for_interval(current_time: datetime) -> pd.DataFrame | None:
    """
    Attempts to download and parse GDELT data for a given 15-minute interval.
    Handles internet connection issues and retries on transient errors.
    Returns a DataFrame if successful, None if the interval should be skipped.
    """
    timestamp_str = current_time.strftime('%Y%m%d%H%M%S')
    
    while True:
        # First, check for a general internet connection.
        if not check_internet_connection():
            print("No internet connection detected. Pausing for 5 minutes before retrying...")
            time.sleep(300)
            continue

        try:
            # Step 1: Attempt to download the file.
            temp_zip_path, temp_csv_path = download_gdelt_file(timestamp_str)
            
            # Step 2: Attempt to extract and parse the file.
            # This function will return None if the file is empty, which is a valid skip condition.
            # It will raise a BadZipFile exception if the zip is corrupt, which we'll catch below.
            df = extract_gdelt_csv(temp_zip_path, temp_csv_path)
            
            # If we get here, the process for this interval is complete (either with data or a clean skip).
            return df

        except Exception as e:
            # This block now catches download failures or unhandled extraction errors.

            # If the file is corrupt, it's not a transient error. We should skip it.
            if "BadZipFile" in str(e) or "not a valid zip file" in str(e):
                 print(f"  - Skipping interval {timestamp_str} due to corrupt or invalid zip file.")
                 return None

            # For any other error, including "Download failed", we assume it might be a
            # temporary network issue and should be retried.
            print(f"  - An error occurred for interval {timestamp_str}: {e}. Retrying in 1 minute...")
            time.sleep(60)
            # The 'while True' loop will then cause the process to repeat. download
