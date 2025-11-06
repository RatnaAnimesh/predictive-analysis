
import os
import json
import sys
import pandas as pd
from datetime import datetime, timedelta, timezone
import subprocess
import zipfile
import math
import re
import nltk
from tqdm import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import xml.etree.ElementTree as ET

# --- Mapper Script for Parallel Processing ---

TEMP_DIR = "/Users/ashishmishra/geopolitical_predictor/temp_data"

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

# Pre-compile regex for efficiency
ENTITY_PATTERNS = {entity: re.compile(r'\b' + re.escape(entity) + r'\b', re.IGNORECASE) for entity in TRACKED_ENTITIES}

def initialize_state():
    """Initializes a fresh state dictionary for this mapper instance."""
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

def update_stats(stats, new_value):
    """Updates the running statistics using Welford's algorithm."""
    stats['count'] += 1
    delta = new_value - stats['mean']
    stats['mean'] += delta / stats['count']
    delta2 = new_value - stats['mean']
    stats['m2'] += delta * delta2

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

import traceback

def run_mapper(year):
    # Add a top-level try/except to catch any unexpected errors and log them
    try:
        _run_mapper_logic(year)
    except Exception as e:
        # Log the full traceback to the log file
        log_path = f"/Users/ashishmishra/geopolitical_predictor/{year}.log"
        with open(log_path, 'a') as f:
            f.write("\n--- UNHANDLED EXCEPTION ---\n")
            f.write(traceback.format_exc())
        # Exit with an error code so the launcher knows it failed
        sys.exit(1)

def _run_mapper_logic(year):
    """Processes all GDELT data for a single given year."""
    START_TIME = datetime(year, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    END_TIME = datetime(year + 1, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    
    print(f"--- Starting Mapper for Year: {year} ---")
    os.makedirs(TEMP_DIR, exist_ok=True)
    state = initialize_state()
    sid = SentimentIntensityAnalyzer()
    
    current_time = START_TIME
    total_intervals = (END_TIME - START_TIME) // timedelta(minutes=15)

    # Wrap the main loop with tqdm for a progress bar
    for _ in tqdm(range(total_intervals), desc=f"Mapper for Year {year}"):
        timestamp_str = current_time.strftime('%Y%m%d%H%M%S')

        file_name = f"{timestamp_str}.gkg.csv.zip"
        url = f"http://data.gdeltproject.org/gdeltv2/{file_name}"
        temp_zip_path = os.path.join(TEMP_DIR, f"{file_name}_{year}") # Add year to avoid filename collision
        temp_csv_path = temp_zip_path.replace('.zip', '')

        try:
            subprocess.run(["curl", "-L", "-s", "-f", "-o", temp_zip_path, url], check=True)
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(TEMP_DIR)
            
            articles_df = pd.read_csv(temp_csv_path, sep='\t', header=None, dtype=str, encoding='latin1')
            df_processed = articles_df.iloc[:, [19, 21, 23, -1]]
            df_processed.columns = ['Organizations', 'Persons', 'Locations', 'Extras']

            # --- Feature Extraction ---
            update_stats(state['interval_article_count'], len(df_processed))

            for entity, pattern in ENTITY_PATTERNS.items():
                mask = (
                    df_processed['Organizations'].str.contains(pattern, na=False) |
                    df_processed['Persons'].str.contains(pattern, na=False) |
                    df_processed['Locations'].str.contains(pattern, na=False)
                )
                entity_mentions_df = df_processed[mask]
                entity_count = len(entity_mentions_df)
                update_stats(state['entity_metrics'][entity]['counts'], entity_count)

                avg_sentiment = 0.0
                if not entity_mentions_df.empty:
                    titles = entity_mentions_df['Extras'].apply(extract_title_from_xml)
                    sentiments = titles.apply(lambda x: sid.polarity_scores(x)['compound'])
                    valid_sentiments = sentiments[sentiments != 0.0]
                    if not valid_sentiments.empty:
                        avg_sentiment = valid_sentiments.mean()
                update_stats(state['entity_metrics'][entity]['sentiment'], avg_sentiment)

        except subprocess.CalledProcessError:
            pass # Fail silently if file doesn't exist
        except (IndexError, FileNotFoundError, zipfile.BadZipFile, pd.errors.ParserError) as e:
            # Catch common file-related processing errors
            print(f"  Skipping interval {timestamp_str} due to processing error: {e}")
        finally:
            if os.path.exists(temp_zip_path): os.remove(temp_zip_path)
            if os.path.exists(temp_csv_path): os.remove(temp_csv_path)

        current_time += timedelta(minutes=15)

    # Save the partial state for this year
    output_filename = f"/Users/ashishmishra/geopolitical_predictor/partial_results_{year}.json"
    with open(output_filename, 'w') as f:
        json.dump(state, f, indent=4)
    
    print(f"--- Mapper for Year {year} Finished. Results saved to {output_filename} ---")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python mapper.py <year>")
        sys.exit(1)
    
    try:
        year_to_process = int(sys.argv[1])
        # Download VADER lexicon if not already present
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except nltk.downloader.DownloadError:
            print("Downloading VADER sentiment lexicon...")
            nltk.download('vader_lexicon', quiet=True)
        run_mapper(year_to_process)
    except ValueError:
        print("Error: Year must be an integer.")
        sys.exit(1)
