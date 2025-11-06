import os
import json
import time
import pandas as pd
from datetime import datetime, timedelta, timezone
import subprocess
import zipfile
import math
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import xml.etree.ElementTree as ET

# --- Setup ---
MODEL_STATE_FILE = "/Users/ashishmishra/geopolitical_predictor/model_state.json"
TEMP_DIR = "/Users/ashishmishra/geopolitical_predictor/temp_data"

# --- Configuration for this Run ---
START_TIME = datetime(2023, 11, 5, 0, 0, 0, tzinfo=timezone.utc)
END_TIME = datetime(2023, 11, 6, 0, 0, 0, tzinfo=timezone.utc)


# Download VADER lexicon if not already present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    print("Downloading VADER sentiment lexicon...")
    nltk.download('vader_lexicon')

# Define the entities to track
TRACKED_ENTITIES = [
    "Joe Biden", "Vladimir Putin", "Xi Jinping",
    "Apple", "Google", "Microsoft", "Amazon", "Tesla",
    "ExxonMobil", "Saudi Aramco",
    "TSMC", "Samsung", "Intel", "Nvidia",
    "Boeing", "Lockheed Martin",
    "United Nations", "World Health Organization",
    "Kyiv", "Moscow", "Beijing", "Washington DC"
]

# --- State Management Functions ---
def load_model_state():
    """Loads the model's statistical state from a JSON file."""
    if os.path.exists(MODEL_STATE_FILE):
        with open(MODEL_STATE_FILE, 'r') as f:
            state = json.load(f)
    else:
        # If no state file, create a fresh one. This should not happen after historical run.
        print("Warning: Model state file not found. Starting with a fresh state.")
        state = {}

    # Ensure all tracked features are initialized
    if "interval_article_count" not in state:
        state["interval_article_count"] = {"count": 0, "mean": 0.0, "m2": 0.0}
    
    if "entity_metrics" not in state:
        state["entity_metrics"] = {}

    for entity in TRACKED_ENTITIES:
        if entity not in state["entity_metrics"]:
            state["entity_metrics"][entity] = {
                "counts": {"count": 0, "mean": 0.0, "m2": 0.0},
                "sentiment": {"count": 0, "mean": 0.0, "m2": 0.0}
            }
    
    return state

def save_model_state(state):
    """Saves the model's statistical state to a JSON file."""
    with open(MODEL_STATE_FILE, 'w') as f:
        json.dump(state, f, indent=4)

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

def extract_title_from_xml(xml_string):
    """Parses the GDELT Extras XML to find the page title."""
    try:
        if not xml_string or pd.isna(xml_string):
            return ""
        root = ET.fromstring(f"<root>{xml_string}</root>")
        title_element = root.find(".//PAGE_TITLE")
        if title_element is not None and title_element.text:
            return title_element.text
    except ET.ParseError:
        return ""
    return ""

# --- Main Detector Logic ---
def run_detection_run():
    """Runs the anomaly detection loop for a specific historical period to find anomalies."""
    print(f"--- Starting Anomaly Detection Run from {START_TIME} to {END_TIME} ---")
    os.makedirs(TEMP_DIR, exist_ok=True)
    state = load_model_state()
    sid = SentimentIntensityAnalyzer()
    
    current_time = START_TIME
    total_intervals = (END_TIME - START_TIME) // timedelta(minutes=15)
    processed_count = 0

    while current_time < END_TIME:
        timestamp_str = current_time.strftime('%Y%m%d%H%M%S')
        processed_count += 1
        print(f"\n--- Detecting for Interval {processed_count}/{total_intervals}: {timestamp_str} ---")

        file_name = f"{timestamp_str}.gkg.csv.zip"
        url = f"http://data.gdeltproject.org/gdeltv2/{file_name}"
        temp_zip_path = os.path.join(TEMP_DIR, file_name)
        temp_csv_path = temp_zip_path.replace('.zip', '')

        articles_df = pd.DataFrame()
        try:
            print(f"Downloading {url}...")
            subprocess.run(["curl", "-L", "-s", "-f", "-o", temp_zip_path, url], check=True)

            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(TEMP_DIR)
            
            articles_df = pd.read_csv(temp_csv_path, sep='\t', header=None, dtype=str, encoding='latin1')
            print(f"Loaded {len(articles_df)} articles.")

        except subprocess.CalledProcessError:
            print(f"No data file for {timestamp_str}. Skipping.")
        except Exception as e:
            print(f"Error processing interval {timestamp_str}: {e}")
        finally:
            if os.path.exists(temp_zip_path): os.remove(temp_zip_path)
            if os.path.exists(temp_csv_path): os.remove(temp_csv_path)

        # --- Anomaly Detection ---
        if not articles_df.empty:
            # Select columns after loading
            try:
                df_processed = articles_df.iloc[:, [19, 21, 23, -1]]
                df_processed.columns = ['Organizations', 'Persons', 'Locations', 'Extras']
            except IndexError:
                print("  Skipping file due to unexpected column structure.")
                current_time += timedelta(minutes=15)
                continue

            for entity in TRACKED_ENTITIES:
                pattern = r'\b' + re.escape(entity) + r'\b'
                mask = (
                    df_processed['Organizations'].str.contains(pattern, case=False, na=False) |
                    df_processed['Persons'].str.contains(pattern, case=False, na=False) |
                    df_processed['Locations'].str.contains(pattern, case=False, na=False)
                )
                entity_mentions_df = df_processed[mask]
                entity_count = len(entity_mentions_df)

                # --- Check for Count Anomaly ---
                stats_count = state['entity_metrics'][entity]['counts']
                mean_count = stats_count['mean']
                std_dev_count = get_std_dev(stats_count)
                if std_dev_count > 0:
                    z_score_count = (entity_count - mean_count) / std_dev_count
                    if abs(z_score_count) > 3.0: # Higher threshold for significance
                        print(f"  >> ALERT: Mention count for '{entity}' is anomalous (Z-Score: {z_score_count:.2f}). Mentions: {entity_count}, Baseline Mean: {mean_count:.2f}")

                # --- Check for Sentiment Anomaly ---
                avg_sentiment = 0.0
                if not entity_mentions_df.empty:
                    titles = entity_mentions_df['Extras'].apply(extract_title_from_xml)
                    sentiments = titles.apply(lambda x: sid.polarity_scores(x)['compound'])
                    valid_sentiments = sentiments[sentiments != 0.0]
                    if not valid_sentiments.empty:
                        avg_sentiment = valid_sentiments.mean()
                
                if avg_sentiment != 0.0:
                    stats_sentiment = state['entity_metrics'][entity]['sentiment']
                    mean_sentiment = stats_sentiment['mean']
                    std_dev_sentiment = get_std_dev(stats_sentiment)
                    if std_dev_sentiment > 0.01: # Avoid alerting on minor fluctuations
                        z_score_sentiment = (avg_sentiment - mean_sentiment) / std_dev_sentiment
                        if abs(z_score_sentiment) > 3.0:
                            print(f"  >> ALERT: Sentiment for '{entity}' is anomalous (Z-Score: {z_score_sentiment:.2f}). Sentiment: {avg_sentiment:.3f}, Baseline Mean: {mean_sentiment:.3f}")
        
        current_time += timedelta(minutes=15)

    print("\n--- Anomaly Detection Run Finished ---")

if __name__ == "__main__":
    run_detection_run()
