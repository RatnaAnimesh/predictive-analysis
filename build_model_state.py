import pandas as pd
import json
from datetime import datetime, timedelta, timezone
import math
from tqdm import tqdm
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Configuration ---
# Pass 2: Analyze Log and Build Final Model State

ENTITY_LOG_FILE = "/Users/ashishmishra/geopolitical_predictor/discovered_entities.csv"
MODEL_STATE_FILE = "/Users/ashishmishra/geopolitical_predictor/model_state.json"

# To be considered "significant", an entity must appear at least this many times over the entire dataset.
MIN_OCCURRENCE_THRESHOLD = 1000 

# Define a small list of core entities we want to ensure are included, even if they don't meet the threshold.
CORE_ENTITIES = [
    "China", "Russia", "United States", "USA", "Ukraine", "Taiwan", "Israel", "Iran", "Saudi Arabia",
    "Apple", "Microsoft", "Google", "Amazon", "Nvidia", "Tesla",
    "United Nations", "NATO", "OPEC", "Federal Reserve"
]

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
    Pass 2: Reads the raw entity log, identifies significant entities,
    and builds the final statistical model state.
    """
    print(f"--- Starting Pass 2: Building Model State from {ENTITY_LOG_FILE} ---")

    # --- Step 1: Identify Significant Entities ---
    print(f"Step 1: Identifying significant entities (threshold: >{MIN_OCCURRENCE_THRESHOLD} mentions)...")
    # Use chunking to handle potentially massive log file
    chunk_iter = pd.read_csv(ENTITY_LOG_FILE, chunksize=1_000_000, usecols=['entity_text'])
    entity_counts = pd.concat([chunk['entity_text'].value_counts() for chunk in tqdm(chunk_iter, desc="Counting entities")])
    entity_counts = entity_counts.groupby(entity_counts.index).sum()
    
    significant_entities = entity_counts[entity_counts > MIN_OCCURRENCE_THRESHOLD].index.tolist()
    
    # Ensure core entities are included
    for core_entity in CORE_ENTITIES:
        if core_entity not in significant_entities:
            significant_entities.append(core_entity)
    
    print(f"Found {len(significant_entities)} significant entities.")

    # --- Step 2: Calculate Statistics for Significant Entities ---
    print("Step 2: Calculating statistical baseline for significant entities...")

    # We need to know the full date range to calculate total intervals
    # A bit of a hack: assume the Pass 1 script ran and the state file exists
    # In a real pipeline, this would be passed as an argument.
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

    # Create a full date range of all possible 15-minute intervals
    full_date_range = pd.date_range(start=start_dt, end=end_dt, freq='15min')

    # Read the entire log into memory. For truly massive files, this would need
    # to be a database or a more advanced chunking strategy.
    print("Reading full entity log into memory...")
    df = pd.read_csv(ENTITY_LOG_FILE, parse_dates=['timestamp'])
    df['timestamp'] = df['timestamp'].dt.floor('15min')

    # Group by entity and timestamp to get counts per interval
    print("Grouping entities by time interval...")
    entity_interval_counts = df.groupby(['entity_text', 'timestamp']).size().reset_index(name='count')

    model_state = {"entity_metrics": {}}

    print("Calculating final statistics for each significant entity...")
    for entity in tqdm(significant_entities, desc="Building final model"):
        entity_df = entity_interval_counts[entity_interval_counts['entity_text'] == entity]
        
        # Create a complete time series for this entity
        ts_df = pd.DataFrame(index=full_date_range)
        ts_df.index.name = 'timestamp'
        
        # Merge the actual counts into the full time series
        merged_df = ts_df.merge(entity_df.set_index('timestamp')['count'], how='left', left_index=True, right_index=True)
        merged_df['count'] = merged_df['count'].fillna(0) # Fill non-mentions with 0
        
        # Calculate stats using pandas built-in, correct functions
        mean_val = merged_df['count'].mean()
        std_dev_val = merged_df['count'].std()
        total_mentions = merged_df['count'].sum()

        model_state["entity_metrics"][entity] = {
            "counts": {
                "mean": mean_val,
                "std_dev": std_dev_val,
                "total_mentions": int(total_mentions)
            },
            # Sentiment can be added here in a future version
            "sentiment": {"mean": 0.0, "std_dev": 0.0}
        }


    # --- Step 3: Save the Final Model State ---
    print(f"Step 3: Saving final model state to {MODEL_STATE_FILE}...")
    with open(MODEL_STATE_FILE, 'w') as f:
        json.dump(model_state, f, indent=4)

    print("\n--- Pass 2 Analysis Finished ---")
    print("Final model state has been built successfully.")

if __name__ == "__main__":
    run_pass2_analysis()
