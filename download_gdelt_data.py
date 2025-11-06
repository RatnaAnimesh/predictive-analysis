from gdeltdoc import GdeltDoc, Filters
from datetime import date, timedelta
import pandas as pd
import os
import time

def download_gdelt_baseline(year=2024):
    print(f"Downloading GDELT data for the entire year {year} to build a baseline...")

    gd = GdeltDoc()
    output_dir = "/Users/ashishmishra/geopolitical_predictor/data/gdelt_baseline"
    os.makedirs(output_dir, exist_ok=True)

    start_date = date(year, 1, 1)
    end_date = date(year, 12, 31)

    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        file_path = os.path.join(output_dir, f"gdelt_articles_{date_str}.csv")

        if os.path.exists(file_path):
            print(f"Skipping {date_str}: file already exists.")
            current_date += timedelta(days=1)
            continue

        print(f"Downloading data for {date_str}...")
        f = Filters(
            start_date = date_str,
            end_date = date_str
        )

        try:
            articles = gd.article_search(f)
            if not articles.empty:
                articles.to_csv(file_path, index=False)
                print(f"Downloaded {len(articles)} articles for {date_str}.")
            else:
                print(f"No articles found for {date_str}.")
        except Exception as e:
            print(f"Error downloading data for {date_str}: {e}")
        
        current_date += timedelta(days=1)
        time.sleep(1) # Be polite to the GDELT server

    print(f"\nGDELT baseline data download for {year} complete. Saved to {output_dir}")

if __name__ == "__main__":
    download_gdelt_baseline(year=2024)
