
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

# --- Configuration ---
# Set the start and end dates for the historical run
# Format: YYYY, M, D, H, M, S
START_TIME = datetime(2015, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
END_TIME = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


# Download VADER lexicon if not already present
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except nltk.downloader.DownloadError:
    print("Downloading VADER sentiment lexicon...")
    nltk.download('vader_lexicon')

import spacy
from collections import Counter

# --- NLP Model Setup ---
SPACY_MODEL_NAME = "en_core_web_sm"

# --- New Configuration ---
# A smaller, focused list of critical entities for stable tracking
CORE_ENTITIES = [
    # Key World Leaders & Countries
    "Joe Biden", "Vladimir Putin", "Xi Jinping",
    "China", "Russia", "United States", "USA", "Ukraine", "Taiwan", "Israel", "Iran", "Saudi Arabia", "United Kingdom", "Germany", "France", "Japan", "India",

    # Key Companies & Financial Institutions
    "Apple", "Microsoft", "Google", "Amazon", "Nvidia", "Tesla", "Meta Platforms",
    "TSMC", "Samsung", "Intel",
    "JPMorgan Chase", "Bank of America", "Goldman Sachs",
    "ExxonMobil", "Saudi Aramco",

    # Key Organizations
    "United Nations", "NATO", "OPEC", "World Health Organization", "Federal Reserve",

    # Key Locations
    "Kyiv", "Moscow", "Beijing", "Washington DC",
]

# Pre-compile regex for the core, high-signal entities for speed
CORE_ENTITY_PATTERNS = {entity: re.compile(r'\b' + re.escape(entity) + r'\b', re.IGNORECASE) for entity in CORE_ENTITIES}

# --- Dynamic Discovery Configuration ---
# An entity must appear this many times in a batch to be added to the discovered list
DISCOVERY_THRESHOLD = 10 
# We will ignore these noisy or irrelevant spaCy entity types
IGNORED_ENTITY_LABELS = ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']

# --- State Management Functions ---
def load_model_state():
    """Loads the model's statistical state from a JSON file."""
    if os.path.exists(MODEL_STATE_FILE):
        with open(MODEL_STATE_FILE, 'r') as f:
            state = json.load(f)
    else:
        state = {}

    # Initialize top-level structure
    if "interval_article_count" not in state: state["interval_article_count"] = {"count": 0, "mean": 0.0, "m2": 0.0}
    if "core_metrics" not in state: state["core_metrics"] = {}
    if "discovered_metrics" not in state: state["discovered_metrics"] = {}
    
    # Ensure all core entities are initialized
    for entity in CORE_ENTITIES:
        if entity not in state["core_metrics"]:
            state["core_metrics"][entity] = {
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
    if stats['count'] < 2: return 0.0
    variance = stats['m2'] / (stats['count'] - 1)
    return math.sqrt(variance)

def extract_title_from_xml(xml_string):
    """Parses the GDELT Extras XML to find the page title."""
    try:
        if not xml_string or pd.isna(xml_string): return ""
        root = ET.fromstring(f"<root>{xml_string}</root>")
        title_element = root.find(".//PAGE_TITLE")
        if title_element is not None and title_element.text: return title_element.text
    except ET.ParseError: return ""
    return ""

def is_valid_entity(ent):
    """Helper function to filter out noisy entities."""
    if ent.label_ in IGNORED_ENTITY_LABELS: return False
    if len(ent.text) < 3: return False # Ignore very short entities
    return True

# --- Main Analyzer Logic ---
from tqdm import tqdm

BATCH_SIZE_MINUTES = 15 # Process data in 15-minute chunks (one file at a time)

def run_historical_analyzer():
    """
    Runs the historical analysis using a hybrid approach:
    1. Fast regex matching for a small list of CORE entities.
    2. Slower, dynamic NER discovery for new, emerging entities.
    This process is resumable.
    """
    # --- Model and State Initialization ---
    os.makedirs(TEMP_DIR, exist_ok=True)
    state = load_model_state()
    sid = SentimentIntensityAnalyzer()

    # Download and load spaCy model
    try:
        nlp = spacy.load(SPACY_MODEL_NAME)
        print(f"Successfully loaded spaCy model '{SPACY_MODEL_NAME}'.")
    except OSError:
        print(f"Spacy model '{SPACY_MODEL_NAME}' not found. Downloading...")
        spacy.cli.download(SPACY_MODEL_NAME)
        nlp = spacy.load(SPACY_MODEL_NAME)

    # --- Resume Logic ---
    last_processed_str = state.get("last_processed_timestamp")
    if last_processed_str:
        start_time = datetime.fromisoformat(last_processed_str)
        print(f"--- Resuming Historical Analyzer from {start_time} ---")
    else:
        start_time = START_TIME
        print(f"--- Starting Historical Analyzer from {start_time} to {END_TIME} ---")

    # --- Main Processing Loop ---
    batch_start_time = start_time
    while batch_start_time < END_TIME:
        batch_end_time = batch_start_time + timedelta(minutes=BATCH_SIZE_MINUTES)
        if batch_end_time > END_TIME: batch_end_time = END_TIME

        print(f"\n--- Processing Batch: {batch_start_time} to {batch_end_time} ---")
        total_intervals_in_batch = (batch_end_time - batch_start_time) // timedelta(minutes=15)
        interval_iterator = tqdm(range(total_intervals_in_batch), desc=f"Batch starting {batch_start_time.strftime('%Y-%m-%d %H:%M')}")

        current_time = batch_start_time
        for _ in interval_iterator:
            timestamp_str = current_time.strftime('%Y%m%d%H%M%S')
            interval_iterator.set_postfix_str(f"Interval: {timestamp_str}")

            file_name = f"{timestamp_str}.gkg.csv.zip"
            url = f"http://data.gdeltproject.org/gdeltv2/{file_name}"
            temp_zip_path = os.path.join(TEMP_DIR, file_name)
            temp_csv_path = temp_zip_path.replace('.zip', '')

            try:
                subprocess.run(["curl", "-L", "-s", "-f", "-o", temp_zip_path, url], check=True)
                with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref: zip_ref.extractall(TEMP_DIR)
                
                articles_df = pd.read_csv(temp_csv_path, sep='\t', header=None, dtype=str, encoding='latin1')
                df_processed = articles_df.iloc[:, [3, 19, 21, 23, -1]] # Add column 3 for article text
                df_processed.columns = ['ArticleText', 'Organizations', 'Persons', 'Locations', 'Extras']

                update_stats(state['interval_article_count'], len(df_processed))

                # --- 1. Core Entity Processing (Fast) ---
                for entity, pattern in CORE_ENTITY_PATTERNS.items():
                    mask = (
                        df_processed['Organizations'].str.contains(pattern, na=False) |
                        df_processed['Persons'].str.contains(pattern, na=False) |
                        df_processed['Locations'].str.contains(pattern, na=False)
                    )
                    entity_mentions_df = df_processed[mask]
                    update_stats(state['core_metrics'][entity]['counts'], len(entity_mentions_df))
                    
                    avg_sentiment = 0.0
                    if not entity_mentions_df.empty:
                        titles = entity_mentions_df['Extras'].apply(extract_title_from_xml)
                        sentiments = titles.apply(lambda x: sid.polarity_scores(x)['compound'])
                        valid_sentiments = sentiments[sentiments != 0.0]
                        if not valid_sentiments.empty: avg_sentiment = valid_sentiments.mean()
                    update_stats(state['core_metrics'][entity]['sentiment'], avg_sentiment)

                # --- 2. Dynamic Entity Discovery (Slower) ---
                # Use a counter to find the most common entities in this batch
                discovered_entity_counts = Counter()
                all_titles = df_processed['Extras'].apply(extract_title_from_xml)
                
                # Process documents in batches with spaCy for efficiency
                for doc in nlp.pipe(all_titles, disable=["parser", "lemmatizer"]):
                    for ent in doc.ents:
                        if is_valid_entity(ent):
                            discovered_entity_counts[ent.text.strip()] += 1
                
                # Gatekeeping: Only add/update stats for significant discovered entities
                for entity, count in discovered_entity_counts.items():
                    if count >= DISCOVERY_THRESHOLD and entity not in CORE_ENTITIES:
                        # This is a significant new or existing discovered entity
                        if entity not in state["discovered_metrics"]:
                            # Initialize stats for a brand new entity
                            state["discovered_metrics"][entity] = {
                                "counts": {"count": 0, "mean": 0.0, "m2": 0.0},
                                "sentiment": {"count": 0, "mean": 0.0, "m2": 0.0}
                            }
                        
                        # Since we already counted it, we can just update the stats
                        update_stats(state["discovered_metrics"][entity]['counts'], count)
                        # Note: A full sentiment analysis for discovered entities would be too slow here.
                        # We can add this later if needed. For now, we focus on discovery and counts.
                        update_stats(state["discovered_metrics"][entity]['sentiment'], 0.0)

            except subprocess.CalledProcessError: pass
            except (IndexError, FileNotFoundError, zipfile.BadZipFile, pd.errors.ParserError) as e:
                tqdm.write(f"  Skipping interval {timestamp_str} due to processing error: {e}")
            except Exception as e:
                tqdm.write(f"An unexpected error occurred at interval {timestamp_str}: {e}")
            finally:
                if os.path.exists(temp_zip_path): os.remove(temp_zip_path)
                if os.path.exists(temp_csv_path): os.remove(temp_csv_path)

            current_time += timedelta(minutes=15)

        # --- Batch End: Save Progress ---
        print(f"--- Finished Batch. Saving state... ---")
        state["last_processed_timestamp"] = batch_end_time.isoformat()
        save_model_state(state)
        
        batch_start_time = batch_end_time

    print("\n--- Historical Analysis Finished ---")
    print("Final model state saved.")
    print("Done.")

if __name__ == "__main__":
    run_historical_analyzer()
