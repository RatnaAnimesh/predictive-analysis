import pandas as pd
import json
from datetime import datetime, timedelta, timezone
import math
from tqdm import tqdm

# --- Configuration ---
# Pass 2: Analyze Log and Build Final Model State

ENTITY_LOG_FILE = "/Users/ashishmishra/geopolitical_predictor/discovered_entities.csv"
MODEL_STATE_FILE = "/Users/ashishmishra/geopolitical_predictor/model_state.json"

# To be considered "significant", an event type must appear at least this many times over the entire dataset.
MIN_OCCURRENCE_THRESHOLD = 500 


# --- Welford's Algorithm for Stable Statistics ---
def update_stats(stats, new_value):
    """Updates the running statistics using Welford's algorithm."""
    stats['count'] += 1
    delta = new_value - stats['mean']
    stats['mean'] += delta / stats['count']
    delta2 = new_value - stats['mean']
    stats['m2'] += delta * delta2

def get_std_dev(stats):
    """Calculates the standard deviation from the given stats dictionary."""
    if stats['count'] < 2:
        return 0.0
    variance = stats['m2'] / (stats['count'] - 1)
    return math.sqrt(variance)

# --- Main "Pass 2" Logic ---
def run_pass2_analysis():
    """
    Pass 2: Reads the raw event log, identifies significant event types,
    and builds the final statistical model state.
    """
    print(f"--- Starting Pass 2: Building Model State from {ENTITY_LOG_FILE} ---")

    # --- Step 1: Determine Date Range ---
    try:
        with open(PASS1_STATE_FILE, 'r') as f:
            pass1_state = json.load(f)
            start_dt_str = pass1_state.get("start_time", "2015-02-01T00:00:00Z") # Default start
            end_dt_str = pass1_state.get("last_processed_timestamp")
            if not end_dt_str:
                raise ValueError("Pass 1 state file is incomplete.")
            start_dt = datetime.fromisoformat(start_dt_str.replace('Z', '+00:00'))
            end_dt = datetime.fromisoformat(end_dt_str.replace('Z', '+00:00'))
    except FileNotFoundError:
        print(f"ERROR: {PASS1_STATE_FILE} not found. Please run Pass 1 first.")
        return

    full_date_range = pd.date_range(start=start_dt, end=end_dt, freq='15min')
    print(f"Analyzing data from {start_dt} to {end_dt} ({len(full_date_range)} intervals).")

    # --- Step 2: Read and Prepare Data ---
    print("Reading full event log into memory...")
    # Read all columns as we will use multiple for analysis
    df = pd.read_csv(ENTITY_LOG_FILE, parse_dates=['interval_timestamp'])
    df['interval_timestamp'] = df['interval_timestamp'].dt.floor('15min')
    
    # Drop rows where event_root_code is NaN or empty, as these are not valid events
    df.dropna(subset=['event_root_code'], inplace=True)
    df = df[df['event_root_code'] != '']

    # --- Step 3: Identify Significant Event Types and Calculate Statistics ---
    print(f"Step 3: Identifying significant event types and calculating statistical baseline (threshold: >{MIN_OCCURRENCE_THRESHOLD} occurrences)...")

    model_state = {"event_metrics": {}}

    # Define the event types we want to track statistics for
    # For now, let's focus on event_root_code and quad_class
    event_type_columns = ['event_root_code', 'quad_class']
    
    for col in event_type_columns:
        print(f"  Processing '{col}'...")
        # Count occurrences of each unique value in the column
        counts = df.groupby([col, 'interval_timestamp']).size().reset_index(name='count')
        
        # Identify significant event types based on total occurrences
        total_counts = counts.groupby(col)['count'].sum()
        significant_types = total_counts[total_counts > MIN_OCCURRENCE_THRESHOLD].index.tolist()

        # Ensure some core event types are included if desired (can be expanded)
        # For now, we'll rely on the threshold for event types.

        for event_type in tqdm(significant_types, desc=f"  Building model for {col}"):
            type_df = counts[counts[col] == event_type]
            
            # Create a complete time series for this event type
            ts_df = pd.DataFrame(index=full_date_range)
            ts_df.index.name = 'interval_timestamp'
            
            # Merge the actual counts into the full time series
            merged_df = ts_df.merge(type_df.set_index('interval_timestamp')['count'], how='left', left_index=True, right_index=True)
            merged_df['count'] = merged_df['count'].fillna(0) # Fill non-occurrences with 0
            
            # Calculate stats
            mean_val = merged_df['count'].mean()
            std_dev_val = merged_df['count'].std()
            total_occurrences = merged_df['count'].sum()

            model_state["event_metrics"][f"{col}_{event_type}"] = {
                "mean": mean_val,
                "std_dev": std_dev_val,
                "total_occurrences": int(total_occurrences)
            }

    # --- Step 4: Save the Final Model State ---
    print(f"Step 4: Saving final model state to {MODEL_STATE_FILE}...")
    with open(MODEL_STATE_FILE, 'w') as f:
        json.dump(model_state, f, indent=4)

    print("\n--- Pass 2 Analysis Finished ---")
    print("Final model state has been built successfully.")

if __name__ == "__main__":
    run_pass2_analysis()
