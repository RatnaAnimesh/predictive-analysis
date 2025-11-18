# Geopolitical Predictor

This project implements a real-time pipeline to ingest geopolitical event data from the GDELT Project and construct a knowledge graph. This graph enables sophisticated analysis of the relationships and interactions between global actors.

## Table of Contents

- [Project Goals](#project-goals)
- [Architecture](#architecture)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Project Goals

The ultimate aim is to leverage the knowledge graph to achieve the following:

1.  **Anomaly Detection:** Identify unusual spikes in the frequency or nature of events involving specific actors or locations.
2.  **Predictive Analysis:** Analyze event sequences and actor interactions to forecast the probability of future events.
3.  **Corroboration Analysis:** Assess the probability of a user-supplied hypothetical event based on historical patterns in the graph.

## Architecture

The project uses a real-time, producer-consumer architecture:

1.  **Producer (`historical_analyzer.py`):** This script continuously downloads 15-minute event data files from GDELT and places them as individual CSVs into a staging directory (`data/pass1_output`).
2.  **Consumer (`graph_builder.py`):** This script runs in parallel, monitoring the staging directory. As new CSV files appear, it immediately processes them, converts the events into nodes and relationships, and loads them into a graph database (Neo4j or a compatible alternative like Memgraph).

This decoupled architecture allows for robust, continuous data ingestion.

## Setup

To get this project up and running, follow these steps:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/RatnaAnimesh/predictive-analysis.git
    cd predictive-analysis
    ```

2.  **Set up Graph Database:**
    A graph database compatible with the Neo4j Bolt protocol is required. The easiest way to get started is with the official Neo4j Docker container:
    ```bash
    docker run \
        --name neo4j-geopred \
        -p 7474:7474 -p 7687:7687 \
        -d \
        -e NEO4J_AUTH=neo4j/password \
        --rm \
        neo4j:latest
    ```
    This will start a Neo4j instance with the username `neo4j` and password `password`.

3.  **Configure Environment Variables (Optional):**
    The application is configured via `config.py`. For security and portability, you can override the database credentials by setting environment variables:
    ```bash
    export NEO4J_URI="bolt://localhost:7687"
    export NEO4J_USER="neo4j"
    export NEO4J_PASSWORD="password"
    ```

4.  **Create and Activate Virtual Environment:**
    ```bash
    python3 -m venv geo_venv
    source geo_venv/bin/activate
    ```

5.  **Install Dependencies:**
    ```bash
    pip install pandas tqdm watchdog neo4j SPARQLWrapper
    ```

## Usage

The pipeline is designed to be run as parallel processes.

### Step 1: Populate Foundational Data (One-Time Setup)

First, populate the graph with a canonical list of countries from DBpedia. This provides a clean, foundational layer for your graph.
```bash
python populate_graph_from_dbpedia.py
```

### Step 2: Start the Graph Builder (Consumer)

In a terminal window, start the graph builder. It will begin watching the output directory for new files to process.
```bash
python graph_builder.py
```

### Step 3: Start the Historical Analyzer (Producer)

In a *separate* terminal window, start the historical data analyzer. It will begin downloading GDELT data from 2015 to the present and feeding it to the graph builder.
```bash
python historical_analyzer.py
```
You will see both terminals processing data. The analyzer will download data, and the builder will ingest it into the graph.

### Step 4: Explore the Graph

You can use the `graph_explorer.py` script to run basic queries and inspect the state of your knowledge graph.
```bash
# Get basic stats
python graph_explorer.py

# Clear the entire database (use with caution!)
python graph_explorer.py --clear
```

## Project Structure

```
.
├── config.py                   # Centralized configuration for paths, credentials, and parameters.
├── gdelt_utils.py              # Utility functions for downloading and parsing GDELT data.
├── historical_analyzer.py      # The "producer": downloads GDELT data into the staging directory.
├── graph_builder.py            # The "consumer": processes staged files and builds the graph.
├── populate_graph_from_dbpedia.py # One-time script to enrich the graph with country data.
├── graph_explorer.py           # A CLI tool for inspecting and querying the graph.
├── pass1_state.json            # State file for resuming the historical_analyzer.
└── data/
    └── pass1_output/           # Staging directory for CSV files between the analyzer and builder.
```