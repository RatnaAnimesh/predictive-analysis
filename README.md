# Geopolitical Predictor

This project implements a system for geopolitical anomaly detection and aims to extend into predictive and corroboration analysis based on news data. It processes large volumes of GDELT news data to identify unusual patterns in entity mentions and sentiment.

## Table of Contents

- [Project Goals](#project-goals)
- [Setup](#setup)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)

## Project Goals

1.  **Anomaly Detection:** Figure out if something is showing up in the news more to then check what's happening. Alerts if Z-score of occurrence is higher than 3.
2.  **Predictive Analysis:** Look at the latest news and try to figure out what *could* happen and assign an exact probability to the event. Should display major geopolitical or financial events occurring with probabilities more than 50%.
3.  **Corroboration Analysis:** Say I tell the model something may happen, its goal is to then assign the event an exact probability of occurring based on all of its training.

## Setup

To get this project up and running on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/RatnaAnimesh/predictive-analysis.git
    cd predictive-analysis
    ```

2.  **Create and activate a virtual environment:**
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    python3 -m venv geo_venv
    source geo_venv/bin/activate
    ```

3.  **Install dependencies:**
    Install the required Python packages.
    ```bash
    pip install pandas numpy nltk gdeltdoc tqdm lxml spacy
    ```

4.  **Download NLP Models:**
    The project uses NLTK for sentiment and spaCy for entity recognition.
    ```bash
    python -c "import nltk; nltk.download('vader_lexicon')"
    python -m spacy download en_core_web_sm
    ```

## Usage

The project now operates in a two-pass architecture to build the historical baseline, followed by the online anomaly detection.

### Phase 1: Discovery (Pass 1)

This phase processes all historical GDELT data to discover and log every entity mention. This is a long-running but fast, non-degrading process.

1.  **Run the Pass 1 discovery script:**
    This will create a large `discovered_entities.csv` file containing raw entity data.
    ```bash
    python historical_analyzer.py
    ```

### Phase 2: Analysis (Pass 2)

This phase runs *after* Pass 1 is complete. It analyzes the raw log file to build the final statistical baseline.

1.  **Run the Pass 2 analysis script:**
    This reads the `discovered_entities.csv` and generates the final `model_state.json`.
    ```bash
    python build_model_state.py
    ```

### Phase 3: Online Anomaly Detection

After building the `model_state.json` baseline, you can run the anomaly detector to monitor current news.

1.  **Run the online anomaly detector:**
    ```bash
    python online_anomaly_detector.py
    ```

## Data Sources

*   **GDELT Project:** The primary source for global news event data. The project downloads GKG 2.0 data in 15-minute intervals.

## Project Structure

```
.
├── .gitignore
├── historical_analyzer.py      # Pass 1: Fast, non-degrading script to discover and log all entities.
├── build_model_state.py        # Pass 2: Analyzes the log file to build the final statistical model.
├── online_anomaly_detector.py  # Phase 3: Detects anomalies in real-time GDELT data against the baseline.
├── discovered_entities.csv     # Output of Pass 1: A large log file of all discovered entities.
├── model_state.json            # Output of Pass 2: The final statistical baseline.
├── pass1_state.json            # Simple state file for resuming Pass 1.
└── geo_venv/                   # Python virtual environment
```