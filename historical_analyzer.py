
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

# Define the entities to track
TRACKED_ENTITIES = [
    # Countries
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia", "Australia", "Austria", "Azerbaijan",
    "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso", "Burundi",
    "Cabo Verde", "Cambodia", "Cameroon", "Canada", "Central African Republic", "Chad", "Chile", "China", "Colombia", "Comoros", "Congo", "Costa Rica", "Côte d'Ivoire", "Croatia", "Cuba", "Cyprus", "Czech Republic",
    "Denmark", "Djibouti", "Dominica", "Dominican Republic",
    "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia",
    "Fiji", "Finland", "France",
    "Gabon", "Gambia", "Georgia", "Germany", "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana",
    "Haiti", "Holy See", "Honduras", "Hungary",
    "Iceland", "India", "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy",
    "Jamaica", "Japan", "Jordan",
    "Kazakhstan", "Kenya", "Kiribati", "Kuwait", "Kyrgyzstan",
    "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya", "Liechtenstein", "Lithuania", "Luxembourg",
    "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali", "Malta", "Marshall Islands", "Mauritania", "Mauritius", "Mexico", "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco", "Mozambique", "Myanmar",
    "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Korea", "North Macedonia", "Norway",
    "Oman",
    "Pakistan", "Palau", "Palestine", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal",
    "Qatar",
    "Romania", "Russia", "Rwanda",
    "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa", "San Marino", "Sao Tome and Principe", "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia", "Solomon Islands", "Somalia", "South Africa", "South Korea", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland", "Syria",
    "Taiwan", "Tajikistan", "Tanzania", "Thailand", "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu",
    "Uganda", "Ukraine", "United Arab Emirates", "United Kingdom", "United States", "Uruguay", "Uzbekistan",
    "Vanuatu", "Venezuela", "Vietnam",
    "Yemen",
    "Zambia", "Zimbabwe", "Kosovo",

    # S&P 500 Companies (selection)
    "3M", "Abbott Laboratories", "AbbVie", "Accenture", "Adobe", "Aflac", "Agilent Technologies", "Air Products & Chemicals", "Alphabet", "Altria Group", "Amazon", "American Electric Power", "American Express", "Amgen", "Analog Devices", "Apollo Global Management", "Apple", "Applied Materials", "Arch Capital Group", "Arthur J. Gallagher", "AT&T", "Autozone", "Axon Enterprise", "Bank of America", "Becton, Dickinson", "Berkshire Hathaway", "BlackRock", "Blackstone", "Booking Holdings", "Boston Scientific", "Bristol-Myers Squibb", "Broadcom", "Bunge Global", "BXP", "C.H. Robinson", "Cadence Design Systems", "Capital One", "Caterpillar", "Cencora", "Chipotle Mexican Grill", "Cisco Systems", "Citigroup", "CME Group", "Costco Wholesale", "Digital Realty Trust", "Ecolab", "Eli Lilly", "Elevance Health", "Emerson Electric", "Eog Resources", "Fiserv", "Fortinet", "Freeport-Mcmoran", "Ge Vernova", "Globe Life", "Goldman Sachs", "Hca Healthcare", "Hilton Worldwide Holdings", "Home Depot", "Honeywell International", "Howmet Aerospace", "Illinois Tool Works", "Intercontinental Exchange", "Intuit", "Intuitive Surgical", "Johnson & Johnson", "JPMorgan Chase", "Kinder Morgan", "Kkr & Co", "Lockheed Martin", "Lowe's Companies", "Marsh & Mclennan", "Mastercard", "Merck & Co", "Meta Platforms", "Micron Technology", "Molson Coors Beverage", "Motorola Solutions", "Netflix", "Nextra Energy", "Nike", "Norfolk Southern", "Northrop Grumman", "Nxp Semiconductors", "O'reilly Automotive", "Oneok", "Oracle", "Palo Alto Networks", "Paypal", "Pepsico", "Pfizer", "Philip Morris International", "Palantir Technologies", "Prologis", "Quanta Services", "Republic Services", "Roper Technologies", "Salesforce", "Servicenow", "S&P Global", "Stanley Black & Decker", "Stryker", "T-Mobile Us", "Tesla", "TJX Companies", "Travelers Companies", "Thermo Fisher Scientific", "Uber Technologies", "Union Pacific", "Viatris", "Visa", "Vistra", "Walmart", "Waste Management", "Welltower", "Wells Fargo", "Williams Companies", "Zoetis",

    # Other Key Entities
    "Joe Biden", "Vladimir Putin", "Xi Jinping",
    "ExxonMobil", "Saudi Aramco",
    "TSMC", "Samsung", "Intel", "Nvidia",
    "Boeing",
    "United Nations", "World Health Organization",

    # Key Cities
    "Kyiv", "Moscow", "Beijing", "Washington DC"
]

# Pre-compile regex for efficiency
ENTITY_PATTERNS = {entity: re.compile(r'\b' + re.escape(entity) + r'\b', re.IGNORECASE) for entity in TRACKED_ENTITIES}

# --- State Management Functions ---
def load_model_state():
    """Loads the model's statistical state from a JSON file."""
    if os.path.exists(MODEL_STATE_FILE):
        with open(MODEL_STATE_FILE, 'r') as f:
            state = json.load(f)
    else:
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

# --- Main Analyzer Logic ---
from tqdm import tqdm

# --- Configuration ---
BATCH_SIZE_MINUTES = 15 # Process data in 15-minute chunks (one file at a time)

def run_historical_analyzer():
    """
    Runs the historical analysis in manageable batches to build the statistical baseline.
    This process is resumable.
    """
    os.makedirs(TEMP_DIR, exist_ok=True)
    state = load_model_state()
    sid = SentimentIntensityAnalyzer()

    # --- Resume Logic ---
    last_processed_str = state.get("last_processed_timestamp")
    if last_processed_str:
        start_time = datetime.fromisoformat(last_processed_str)
        print(f"--- Resuming Historical Analyzer from {start_time} ---")
    else:
        start_time = START_TIME
        print(f"--- Starting Historical Analyzer from {start_time} to {END_TIME} ---")

    # The main loop iterates through the date range in batches
    batch_start_time = start_time
    while batch_start_time < END_TIME:
        batch_end_time = batch_start_time + timedelta(minutes=BATCH_SIZE_MINUTES)
        if batch_end_time > END_TIME:
            batch_end_time = END_TIME

        print(f"\n--- Processing Batch: {batch_start_time} to {batch_end_time} ---")
        
        total_intervals_in_batch = (batch_end_time - batch_start_time) // timedelta(minutes=15)
        
        # Use tqdm for a progress bar within each batch
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

        # --- Batch End: Save Progress ---
        print(f"--- Finished Batch. Saving state... ---")
        state["last_processed_timestamp"] = batch_end_time.isoformat()
        save_model_state(state)
        
        batch_start_time = batch_end_time # Move to the next batch

    print("\n--- Historical Analysis Finished ---")
    print("Final model state saved.")
    print("Done.")

if __name__ == "__main__":
    run_historical_analyzer()
