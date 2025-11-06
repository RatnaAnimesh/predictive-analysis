
import os
import json
import glob

# --- Reducer Script for Parallel Processing ---

MODEL_STATE_FILE = "/Users/ashishmishra/geopolitical_predictor/model_state.json"
PARTIAL_RESULTS_PATTERN = "/Users/ashishmishra/geopolitical_predictor/partial_results_*.json"

# Define the entities to track (must match mapper)
TRACKED_ENTITIES = [
    "Joe Biden", "Vladimir Putin", "Xi Jinping",
    "Apple", "Google", "Microsoft", "Amazon", "Tesla",
    "ExxonMobil", "Saudi Aramco",
    "TSMC", "Samsung", "Intel", "Nvidia",
    "Boeing", "Lockheed Martin",
    "United Nations", "World Health Organization",
    "Kyiv", "Moscow", "Beijing", "Washington DC"
]

def load_or_initialize_global_state():
    """Loads the existing global model state, or creates a fresh one."""
    if os.path.exists(MODEL_STATE_FILE):
        print(f"Loading existing global state from {MODEL_STATE_FILE}")
        with open(MODEL_STATE_FILE, 'r') as f:
            state = json.load(f)
    else:
        print("No global state file found. Initializing a new one.")
        state = {
            "interval_article_count": {"count": 0, "mean": 0.0, "m2": 0.0},
            "entity_metrics": {}
        }
        for entity in TRACKED_ENTITIES:
            state["entity_metrics"][entity] = {
                "counts": {"count": 0, "mean": 0.0, "m2": 0.0},
                "sentiment": {"count": 0, "mean": 0.0, "m2": 0.0}
            }
    return state

def combine_stats(global_stats, partial_stats):
    """Combines two sets of Welford's algorithm stats into one."""
    # See: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    count_a, mean_a, m2_a = global_stats['count'], global_stats['mean'], global_stats['m2']
    count_b, mean_b, m2_b = partial_stats['count'], partial_stats['mean'], partial_stats['m2']

    new_count = count_a + count_b
    if new_count == 0:
        return # Nothing to combine

    delta = mean_b - mean_a
    
    new_mean = (count_a * mean_a + count_b * mean_b) / new_count
    new_m2 = m2_a + m2_b + (delta**2 * count_a * count_b) / new_count

    global_stats['count'] = new_count
    global_stats['mean'] = new_mean
    global_stats['m2'] = new_m2

def run_reducer():
    """Combines all partial results into the global model state."""
    print("--- Starting Reducer --- ")
    global_state = load_or_initialize_global_state()
    partial_files = glob.glob(PARTIAL_RESULTS_PATTERN)

    if not partial_files:
        print("No partial result files found. Exiting.")
        return

    print(f"Found {len(partial_files)} partial result files to combine.")

    for file_path in partial_files:
        print(f"  Combining {os.path.basename(file_path)}...")
        with open(file_path, 'r') as f:
            partial_state = json.load(f)
        
        # Combine the interval article count stats
        combine_stats(global_state['interval_article_count'], partial_state['interval_article_count'])

        # Combine stats for each entity
        for entity in TRACKED_ENTITIES:
            combine_stats(global_state['entity_metrics'][entity]['counts'], partial_state['entity_metrics'][entity]['counts'])
            combine_stats(global_state['entity_metrics'][entity]['sentiment'], partial_state['entity_metrics'][entity]['sentiment'])

    # Save the final combined state
    with open(MODEL_STATE_FILE, 'w') as f:
        json.dump(global_state, f, indent=4)
    
    print(f"--- Reducer Finished. Global model state saved to {MODEL_STATE_FILE} ---")

if __name__ == "__main__":
    run_reducer()
