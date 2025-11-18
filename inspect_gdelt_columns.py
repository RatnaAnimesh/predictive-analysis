import pandas as pd
import zipfile
import os
import subprocess

INSPECT_DIR = "/Users/ashishmishra/geopolitical_predictor/inspect"
os.makedirs(INSPECT_DIR, exist_ok=True)

URL = "http://data.gdeltproject.org/gdeltv2/20251113120000.export.CSV.zip"
ZIP_FILE = os.path.join(INSPECT_DIR, "20251113120000.export.CSV.zip")
CSV_FILE = os.path.join(INSPECT_DIR, "20251113120000.export.CSV")

print(f"--- Checking {URL} ---")

try:
    # Download the zip file
    subprocess.run(["curl", "-L", "-s", "--connect-timeout", "10", "--max-time", "30", "-o", ZIP_FILE, URL], check=True)

    if not os.path.exists(ZIP_FILE) or os.path.getsize(ZIP_FILE) == 0:
        print("Downloaded zip file is empty or not found.")
    else:
        try:
            with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
                zip_ref.extractall(INSPECT_DIR)
            
            if os.path.exists(CSV_FILE):
                # Read the CSV with pandas and count columns
                df = pd.read_csv(CSV_FILE, sep='\t', header=None, dtype=str, encoding='latin1') # Changed sep to '\t'
                print(f"Number of columns (pandas): {len(df.columns)}")
                print("First 5 columns:")
                for i, col in enumerate(df.columns[:5]):
                    print(f"  {i}: {df.iloc[0, i]}")
                print("Last 5 columns:")
                for i, col in enumerate(df.columns[-5:]):
                    print(f"  {len(df.columns) - 5 + i}: {df.iloc[0, len(df.columns) - 5 + i]}")
            else:
                print("CSV file not found after unzipping.")
        except zipfile.BadZipFile:
            print("Downloaded file is not a valid zip file.")
        except pd.errors.ParserError as e:
            print(f"Pandas parsing error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred during file processing: {e}")

except subprocess.CalledProcessError:
    print("Download failed.")
except Exception as e:
    print(f"An unexpected error occurred during download: {e}")
finally:
    # Clean up
    if os.path.exists(ZIP_FILE): os.remove(ZIP_FILE)
    if os.path.exists(CSV_FILE): os.remove(CSV_FILE)
