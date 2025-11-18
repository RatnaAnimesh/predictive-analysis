# Geopolitical Analysis Co-Pilot

This project is a sophisticated, multi-layered data pipeline and analytical engine designed to function as a "co-pilot" for geopolitical and financial analysis. It ingests real-time global event data, fuses it with financial market data, and uses a local Large Language Model (LLM) to provide nuanced, context-aware answers to complex, natural language queries.

## Table of Contents

- [Core Objective](#core-objective)
- [Architecture](#architecture)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Core Objective

The goal is to move beyond simple data retrieval and create a system that can synthesize information to answer complex, forward-looking questions. By providing a local LLM with a rich, structured, real-time "briefing" of both geopolitical events and financial data, the system can help analyze scenarios, assess risks, and explore potential second-order effects of global events.

## Architecture

The system is composed of three distinct layers:

1.  **Layer 1: The GDELT Event Ingestion Engine**
    *   A real-time, producer-consumer pipeline (`historical_analyzer.py` and `graph_builder.py`) that continuously downloads global event data from the GDELT project.
    *   This data is processed and stored permanently in a **Graph Database** (e.g., Neo4j/Memgraph), creating a rich, interconnected knowledge graph of "who did what to whom."

2.  **Layer 2: The Financial Data Fusion Engine**
    *   A data ingestor (`financial_data_ingestor.py`) fetches historical and daily stock price data from `yfinance`.
    *   This quantitative data is stored locally and is designed to be fused with the qualitative, event-driven data from the knowledge graph.

3.  **Layer 3: The LLM Application & Analysis Engine**
    *   The main application (`main_app.py`, to be built) serves as the user interface and orchestrator.
    *   It takes a user's natural language query (e.g., "What are the current risks for NVIDIA given the situation in Taiwan?").
    *   It parses the query and intelligently fetches relevant data from both the Graph Database and the financial data files.
    *   It formats this fused data into a detailed prompt (an "intelligence briefing").
    *   It sends this prompt to a **locally running LLM**, which then performs the final synthesis and generates a human-readable answer.

## Setup

1.  **Local LLM:** You must have a local LLM running with an accessible API endpoint (e.g., via Ollama, LM Studio).
2.  **Graph Database:** Set up a Neo4j or Memgraph instance, preferably using Docker.
    ```bash
    docker run --name neo4j-geopred -p 7474:7474 -p 7687:7687 -d -e NEO4J_AUTH=neo4j/password --rm neo4j:latest
    ```
3.  **Clone Repository:**
    ```bash
    git clone https://github.com/RatnaAnimesh/predictive-analysis.git
    cd predictive-analysis
    ```
4.  **Configure Environment:** The application is configured via `config.py`. You can override settings (like database credentials or the LLM endpoint) using environment variables.
5.  **Install Dependencies:** Create and activate a virtual environment, then run:
    ```bash
    pip install -r requirements.txt 
    # (Note: A requirements.txt file will be created for all dependencies)
    ```

## Usage

The project is used in stages: Data Population, followed by Analysis.

### Stage 1: Data Population (Run as needed)

1.  **Populate Graph with Countries (One-Time):**
    ```bash
    python populate_graph_from_dbpedia.py
    ```
2.  **Populate Financial Data:**
    ```bash
    python financial_data_ingestor.py
    ```
3.  **Populate GDELT Event Graph:** Run these two scripts in parallel in separate terminals. The analyzer will feed data to the builder.
    ```bash
    # Terminal 1
    python graph_builder.py
    
    # Terminal 2
    python historical_analyzer.py
    ```

### Stage 2: Analysis (Main Application)

Once the data sources are populated, you can use the main application to ask questions.
```bash
# (This is the future application we will build)
python main_app.py "Your natural language question here..."
```

## Project Structure

```
.
├── config.py                   # Centralized configuration for all layers.
├── gdelt_utils.py              # Utilities for the GDELT ingestion pipeline.
├── historical_analyzer.py      # (Layer 1) Producer: downloads GDELT data.
├── graph_builder.py            # (Layer 1) Consumer: populates the graph database.
├── financial_data_ingestor.py  # (Layer 2) Ingests stock data from yfinance.
├── main_app.py                 # (Layer 3) The main LLM orchestrator (to be built).
├── graph_anomaly_detector.py   # Example of a data analysis tool.
├── graph_explorer.py           # Utility to inspect the graph database.
├── populate_graph_from_dbpedia.py # Utility to enrich the graph with country data.
└── data/
    ├── pass1_output/           # Staging directory for GDELT CSVs.
    └── financial_data/         # Storage for financial CSVs.
```