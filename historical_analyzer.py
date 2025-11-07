
import os
import json
import pandas as pd
from datetime import datetime, timedelta, timezone
import subprocess
import zipfile
import re
import spacy
import xml.etree.ElementTree as ET
from tqdm import tqdm

# --- Configuration ---
# Pass 1: Discover and Log Entities

# New simple state file for resuming Pass 1
PASS1_STATE_FILE = "/Users/ashishmishra/geopolitical_predictor/pass1_state.json"
# Raw output log file for all discovered entities
ENTITY_LOG_FILE = "/Users/ashishmishra/geopolitical_predictor/discovered_entities.csv"
TEMP_DIR = "/Users/ashishmishra/geopolitical_predictor/temp_data"

START_TIME = datetime(2015, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
END_TIME = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

SPACY_MODEL_NAME = "en_core_web_sm"

# We will ignore these noisy or irrelevant spaCy entity types
IGNORED_ENTITY_LABELS = ['DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL']

# --- Utility Functions ---

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

def is_valid_entity(ent):
    """Helper function to filter out noisy entities."""
    if ent.label_ in IGNORED_ENTITY_LABELS:
        return False
    if not ent.text.strip():
        return False
    if len(ent.text.strip()) < 2:
        return False
    return True

# --- Main "Pass 1" Logic ---

def run_pass1_discovery():
    """
    Pass 1: Downloads GDELT data, runs NER, and logs all discovered entities to a CSV file.
    This process is fast, memory-efficient, and its speed does not degrade.
    """
    os.makedirs(TEMP_DIR, exist_ok=True)

    # --- Resume Logic ---
    start_time = START_TIME
    if os.path.exists(PASS1_STATE_FILE):
        with open(PASS1_STATE_FILE, 'r') as f:
            state = json.load(f)
            last_processed_str = state.get("last_processed_timestamp")
            if last_processed_str:
                start_time = datetime.fromisoformat(last_processed_str)
                print(f"--- Resuming Pass 1 Discovery from {start_time} ---")
    
    if start_time == START_TIME:
        print(f"--- Starting Pass 1 Discovery from {start_time} to {END_TIME} ---")
        # Write header for new log file
        with open(ENTITY_LOG_FILE, 'w') as f:
            f.write("timestamp,entity_text,entity_label\n")

    # --- NLP Model Loading ---
    try:
        nlp = spacy.load(SPACY_MODEL_NAME)
        print(f"Successfully loaded spaCy model '{SPACY_MODEL_NAME}'.")
    except OSError:
        print(f"Spacy model '{SPACY_MODEL_NAME}' not found. Downloading...")
        spacy.cli.download(SPACY_MODEL_NAME)
        nlp = spacy.load(SPACY_MODEL_NAME)

    # --- Progress Bar Setup ---
    total_intervals = (END_TIME - START_TIME) // timedelta(minutes=15)
    completed_intervals = (start_time - START_TIME) // timedelta(minutes=15)

    # --- Main Processing Loop ---
    with tqdm(total=total_intervals, initial=completed_intervals, desc="Pass 1 Discovery") as overall_progress:
        current_time = start_time
        while current_time < END_TIME:
            timestamp_str = current_time.strftime('%Y%m%d%H%M%S')
            overall_progress.set_postfix_str(f"Current Date: {current_time.date()}")

            file_name = f"{timestamp_str}.gkg.csv.zip"
            url = f"http://data.gdeltproject.org/gdeltv2/{file_name}"
            temp_zip_path = os.path.join(TEMP_DIR, file_name)
            temp_csv_path = temp_zip_path.replace('.zip', '')

            try:
                subprocess.run(["curl", "-L", "-s", "-f", "-o", temp_zip_path, url], check=True)
                with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                    zip_ref.extractall(TEMP_DIR)
                
                articles_df = pd.read_csv(temp_csv_path, sep='\t', header=None, dtype=str, encoding='latin1')
                # We only need the 'Extras' column containing the XML with the title
                titles = articles_df.iloc[:, -1].dropna().apply(extract_title_from_xml)

                # --- Entity Extraction and Logging ---
                discovered_entities = []
                for doc in nlp.pipe(titles, disable=["parser", "lemmatizer"]):
                    for ent in doc.ents:
                        if is_valid_entity(ent):
                            discovered_entities.append([
                                current_time.isoformat(), 
                                ent.text.strip().replace(",", ""), # Basic cleaning
                                ent.label_
                            ])
                
                # Append discoveries to the log file
                if discovered_entities:
                    pd.DataFrame(discovered_entities).to_csv(ENTITY_LOG_FILE, mode='a', header=False, index=False)

            except subprocess.CalledProcessError:
                pass # This is normal, just means no file for this 15-min interval
            except (IndexError, FileNotFoundError, zipfile.BadZipFile, pd.errors.ParserError) as e:
                tqdm.write(f"  Skipping interval {timestamp_str} due to processing error: {e}")
            except Exception as e:
                tqdm.write(f"An unexpected error occurred at interval {timestamp_str}: {e}")
            finally:
                # CRITICAL: Clean up temp files after each interval
                if os.path.exists(temp_zip_path): os.remove(temp_zip_path)
                if os.path.exists(temp_csv_path): os.remove(temp_csv_path)

            current_time += timedelta(minutes=15)
            overall_progress.update(1)

            # Save progress periodically
            if overall_progress.n > 0 and overall_progress.n % 10 == 0: # Save every 10 intervals
                with open(PASS1_STATE_FILE, 'w') as f:
                    json.dump({"last_processed_timestamp": current_time.isoformat()}, f)

    # --- Finalization ---
    print("\n--- Pass 1 Discovery Finished ---")
    with open(PASS1_STATE_FILE, 'w') as f:
        json.dump({"last_processed_timestamp": END_TIME.isoformat()}, f)
    print(f"All discovered entities have been logged to {ENTITY_LOG_FILE}")
    print("You can now run the Pass 2 script to build the final model state.")

if __name__ == "__main__":
    run_pass1_discovery()
