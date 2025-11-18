import os
import json
import time
import pandas as pd
from datetime import datetime, timedelta, timezone
import subprocess
import zipfile
import math

# --- Setup ---
MODEL_STATE_FILE = "/Users/ashishmishra/geopolitical_predictor/model_state.json"
TEMP_DIR = "/Users/ashishmishra/geopolitical_predictor/temp_data"

# --- Configuration for this Run ---
# These should ideally be passed as arguments or configured externally for a real-time system
START_TIME = datetime(2023, 11, 5, 0, 0, 0, tzinfo=timezone.utc)
END_TIME = datetime(2023, 11, 6, 0, 0, 0, tzinfo=timezone.utc)


# --- State Management Functions ---
def load_model_state():
    """Loads the model's statistical state from a JSON file."""
    if os.path.exists(MODEL_STATE_FILE):
        with open(MODEL_STATE_FILE, 'r') as f:
            state = json.load(f)
            if "event_metrics" not in state:
                raise ValueError(f"Model state file {MODEL_STATE_FILE} does not contain 'event_metrics'. Please run Pass 2 first.")
    else:
        raise FileNotFoundError(f"Model state file {MODEL_STATE_FILE} not found. Please run Pass 2 first to build the model state.")
    
    return state



# --- Main Detector Logic ---
def run_detection_run():
    """Runs the anomaly detection loop for a specific historical period to find anomalies."""
    print(f"--- Starting Anomaly Detection Run from {START_TIME} to {END_TIME} ---")
    os.makedirs(TEMP_DIR, exist_ok=True)
    model_state = load_model_state()
    
    current_time = START_TIME
    total_intervals = (END_TIME - START_TIME) // timedelta(minutes=15)
    processed_count = 0

    # Define the same columns as in historical_analyzer.py for consistency
    use_cols_indices = [1, 5, 6, 15, 16, 26, 27, 28, 29, 30, 31, 34]
    col_names = [
        'sqldate', 'actor1_code', 'actor1_name', 'actor2_code', 'actor2_name',
        'event_code', 'event_base_code', 'event_root_code', 'quad_class',
        'goldstein_scale', 'num_mentions', 'avg_tone'
    ]

    while current_time < END_TIME:
        timestamp_str = current_time.strftime('%Y%m%d%H%M%S')
        processed_count += 1
        print(f"\n--- Detecting for Interval {processed_count}/{total_intervals}: {timestamp_str} ---")

        file_name = f"{timestamp_str}.export.CSV.zip" # Changed to export.CSV.zip
        url = f"http://data.gdeltproject.org/gdeltv2/{file_name}"
        temp_zip_path = os.path.join(TEMP_DIR, file_name)
        temp_csv_path = temp_zip_path.replace('.zip', '')

        articles_df = pd.DataFrame()
        try:
            print(f"Downloading {url}...")
            # Use silent curl with timeouts
            subprocess.run(["curl", "-L", "-s", "--connect-timeout", "10", "--max-time", "30", "-o", temp_zip_path, url], check=True)

            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(TEMP_DIR)
            
            # Read only the necessary columns
            articles_df = pd.read_csv(temp_csv_path, sep='\t', header=None, dtype=str, encoding='latin1', usecols=use_cols_indices)
            articles_df.columns = col_names
            
            # Drop rows where event_root_code is NaN or empty, as these are not valid events
            articles_df.dropna(subset=['event_root_code'], inplace=True)
            articles_df = articles_df[articles_df['event_root_code'] != '']

            print(f"Loaded {len(articles_df)} events.")

        except subprocess.CalledProcessError:
            print(f"No data file for {timestamp_str}. Skipping.")
        except (IndexError, FileNotFoundError, zipfile.BadZipFile, pd.errors.ParserError) as e:
            print(f"Skipping interval {timestamp_str} due to processing error: {e}")
        except Exception as e:
            print(f"An unexpected error occurred at interval {timestamp_str}: {e}")
        finally:
            if os.path.exists(temp_zip_path): os.remove(temp_zip_path)
            if os.path.exists(temp_csv_path): os.remove(temp_csv_path)

        # --- Anomaly Detection ---
        if not articles_df.empty:
            # Group by event_root_code and quad_class to get counts for the current interval
            current_event_counts = articles_df.groupby(['event_root_code']).size().reset_index(name='count')
            current_quad_counts = articles_df.groupby(['quad_class']).size().reset_index(name='count')

            # Check for anomalies in event_root_code
            for index, row in current_event_counts.iterrows():
                event_type_key = f"event_root_code_{row['event_root_code']}"
                current_count = row['count']
                
                if event_type_key in model_state['event_metrics']:
                    stats = model_state['event_metrics'][event_type_key]
                    mean_val = stats['mean']
                    std_dev_val = stats['std_dev']

                    if std_dev_val > 0:
                        z_score = (current_count - mean_val) / std_dev_val
                        if abs(z_score) > 3.0: # Anomaly threshold
                            print(f"  >> ALERT: Anomaly in '{event_type_key}' (Z-Score: {z_score:.2f}). Current: {current_count}, Baseline Mean: {mean_val:.2f}")
            
            # Check for anomalies in quad_class
            for index, row in current_quad_counts.iterrows():
                event_type_key = f"quad_class_{row['quad_class']}"
                current_count = row['count']

                if event_type_key in model_state['event_metrics']:
                    stats = model_state['event_metrics'][event_type_key]
                    mean_val = stats['mean']
                    std_dev_val = stats['std_dev']

                    if std_dev_val > 0:
                        z_score = (current_count - mean_val) / std_dev_val
                        if abs(z_score) > 3.0: # Anomaly threshold
                            print(f"  >> ALERT: Anomaly in '{event_type_key}' (Z-Score: {z_score:.2f}). Current: {current_count}, Baseline Mean: {mean_val:.2f}")
        
        current_time += timedelta(minutes=15)

    print("\n--- Anomaly Detection Run Finished ---")

if __name__ == "__main__":
    run_detection_run()
