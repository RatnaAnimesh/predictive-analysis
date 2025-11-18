import os
from datetime import datetime, timezone

# --- Project Root ---
# Use an environment variable for the root, or fall back to a default
# This makes the project portable
PROJECT_ROOT = os.getenv("GEOPRED_PROJECT_ROOT", os.path.dirname(os.path.abspath(__file__)))

# --- Data Directories ---
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
TEMP_DIR = os.path.join(DATA_DIR, "temp_data")
PASS1_OUTPUT_DIR = os.path.join(DATA_DIR, "pass1_output")
FINANCIAL_DATA_DIR = os.path.join(DATA_DIR, "financial_data")

# --- State and Log Files ---
PASS1_STATE_FILE = os.path.join(PROJECT_ROOT, "pass1_state.json")

# --- GDELT Configuration ---
START_TIME = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
END_TIME = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
GDELT_BASE_URL = "http://data.gdeltproject.org/gdeltv2/"

# --- GDELT V2 Event Database Schema ---
GDELT_EVENT_COLUMNS = {
    'indices': [1, 5, 6, 15, 16, 26, 27, 28, 29, 30, 31, 34],
    'names': [
        'sqldate', 'actor1_code', 'actor1_name', 'actor2_code', 'actor2_name',
        'event_code', 'event_base_code', 'event_root_code', 'quad_class',
        'goldstein_scale', 'num_mentions', 'avg_tone'
    ]
}

# --- Graph Database Configuration ---
# Use environment variables for credentials, with sensible defaults for local dev
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# --- DBpedia Configuration ---
DBPEDIA_URL = "http://dbpedia.org/sparql"
