import os
import json
import pandas as pd
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
import time

# Import centralized configuration and utilities
import config
import gdelt_utils

def run_pass1_discovery():
    """
    Pass 1: Downloads GDELT data and saves each 15-minute interval as a separate
    CSV file in the designated output directory for the graph builder to consume.
    """
    os.makedirs(config.TEMP_DIR, exist_ok=True)
    os.makedirs(config.PASS1_OUTPUT_DIR, exist_ok=True)

    # --- Resume Logic ---
    start_time = config.START_TIME
    if os.path.exists(config.PASS1_STATE_FILE):
        with open(config.PASS1_STATE_FILE, 'r') as f:
            state = json.load(f)
            last_processed_str = state.get("last_processed_timestamp")
            if last_processed_str:
                start_time = datetime.fromisoformat(last_processed_str)
                print(f"--- Resuming Pass 1 Discovery from {start_time} ---")
    
    if start_time == config.START_TIME:
        print(f"--- Starting Pass 1 Discovery from {start_time} to {config.END_TIME} ---")
        # Clean up previous intermediate files if any to ensure a fresh start
        if os.path.exists(config.PASS1_OUTPUT_DIR):
            for f in os.listdir(config.PASS1_OUTPUT_DIR):
                os.remove(os.path.join(config.PASS1_OUTPUT_DIR, f))
        else:
            os.makedirs(config.PASS1_OUTPUT_DIR)

    # --- Progress Bar Setup ---
    total_intervals = (config.END_TIME - start_time) // timedelta(minutes=15)
    
    # --- Main Processing Loop ---
    skipped_count = 0
    with tqdm(total=total_intervals, desc="Pass 1 Discovery") as overall_progress:
        current_time = start_time
        while current_time < config.END_TIME:
            timestamp_str = current_time.strftime('%Y%m%d%H%M%S')
            overall_progress.set_postfix_str(f"Skipped: {skipped_count}, Current Date: {current_time.date()}")

            # Use the utility function to get data for the interval
            df = gdelt_utils.get_gdelt_data_for_interval(current_time)

            if df is not None and not df.empty:
                # Add the interval timestamp to the DataFrame
                df['interval_timestamp'] = current_time.isoformat()
                
                # Reorder columns to have interval_timestamp first
                cols = ['interval_timestamp'] + [col for col in df if col != 'interval_timestamp']
                df = df[cols]

                # Write to Intermediate Log
                intermediate_log_path = os.path.join(config.PASS1_OUTPUT_DIR, f"{timestamp_str}.csv")
                df.to_csv(intermediate_log_path, header=True, index=False)
            else:
                skipped_count += 1

            current_time += timedelta(minutes=15)
            overall_progress.update(1)

            # Save progress periodically
            if overall_progress.n > 0 and overall_progress.n % 10 == 0: # Save every 10 intervals
                with open(config.PASS1_STATE_FILE, 'w') as f:
                    json.dump({"last_processed_timestamp": current_time.isoformat()}, f)

    # --- Finalization ---
    print("\n--- Pass 1 Discovery Finished ---")
    
    with open(config.PASS1_STATE_FILE, 'w') as f:
        json.dump({"last_processed_timestamp": config.END_TIME.isoformat()}, f)
    print("Pass 1 discovery complete. Intermediate files are in the output directory for graph processing.")

if __name__ == "__main__":
    run_pass1_discovery()