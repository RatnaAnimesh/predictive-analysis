import yfinance as yf
import os
import pandas as pd
from datetime import datetime

import config

def ingest_financial_data(tickers: list[str]):
    """
    Fetches historical stock data for a list of tickers using yfinance
    and saves it to CSV files.
    """
    os.makedirs(config.FINANCIAL_DATA_DIR, exist_ok=True)
    
    print(f"--- Starting Financial Data Ingestion from {config.START_TIME} ---")

    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        try:
            # Download data from START_TIME to now
            data = yf.download(ticker, start=config.START_TIME, end=datetime.now())
            
            if not data.empty:
                file_path = os.path.join(config.FINANCIAL_DATA_DIR, f"{ticker}.csv")
                data.to_csv(file_path)
                print(f"Successfully saved data for {ticker} to {file_path}")
            else:
                print(f"No data found for {ticker} in the specified period.")
        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")

    print("--- Financial Data Ingestion Finished ---")

if __name__ == "__main__":
    # Example tickers. This list can be expanded or loaded from a file.
    example_tickers = ["NVDA", "ORCL", "TSM", "AAPL", "MSFT", "GOOGL"] 
    ingest_financial_data(example_tickers)
