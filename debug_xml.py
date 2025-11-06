import os
import pandas as pd
import subprocess
import zipfile
import re

# --- Setup ---
TEMP_DIR = "/Users/ashishmishra/geopolitical_predictor/temp_data"

# Define the same entities as the main script
TRACKED_ENTITIES = [
    "Joe Biden", "Vladimir Putin", "Xi Jinping",
    "Apple", "Google", "Microsoft", "Amazon", "Tesla",
    "ExxonMobil", "Saudi Aramco",
    "TSMC", "Samsung", "Intel", "Nvidia",
    "Boeing", "Lockheed Martin",
    "United Nations", "World Health Organization",
    "Kyiv", "Moscow", "Beijing", "Washington DC"
]

def debug_gdelt_xml():
    """Downloads one GDELT file and prints the Extras XML for the first tracked entity found."""
    print("--- Starting GDELT XML Debugger ---")
    os.makedirs(TEMP_DIR, exist_ok=True)

    timestamp_str = "20231104114500"
    file_name = f"{timestamp_str}.gkg.csv.zip"
    url = f"http://data.gdeltproject.org/gdeltv2/{file_name}"
    temp_zip_path = os.path.join(TEMP_DIR, file_name)
    temp_csv_path = temp_zip_path.replace('.zip', '')

    try:
        print(f"Downloading {url}...")
        subprocess.run(["curl", "-L", "-o", temp_zip_path, url], check=True, capture_output=True)

        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(TEMP_DIR)
        
        articles_df = pd.read_csv(temp_csv_path, sep='\t', header=None, dtype=str)
        print(f"Loaded {len(articles_df)} articles.")

        # --- Find First Entity and Print XML ---
        entity_found = False
        for entity in TRACKED_ENTITIES:
            pattern = r'\b' + re.escape(entity) + r'\b'
            
            # Check all relevant columns
            orgs_col = articles_df.iloc[:, 19] if 19 < articles_df.shape[1] else pd.Series('', index=articles_df.index)
            pers_col = articles_df.iloc[:, 21] if 21 < articles_df.shape[1] else pd.Series('', index=articles_df.index)
            locs_col = articles_df.iloc[:, 23] if 23 < articles_df.shape[1] else pd.Series('', index=articles_df.index)

            mask = (
                orgs_col.str.contains(pattern, case=False, na=False) |
                pers_col.str.contains(pattern, case=False, na=False) |
                locs_col.str.contains(pattern, case=False, na=False)
            )
            entity_mentions_df = articles_df[mask]

            if not entity_mentions_df.empty:
                print(f"Found {len(entity_mentions_df)} mentions of '{entity}'.")
                print("--- Displaying raw XML from 'Extras' column (index 29) for first 5 mentions: ---")
                
                extras_col = entity_mentions_df.iloc[:, 29] if 29 < entity_mentions_df.shape[1] else None

                if extras_col is None:
                    print("Column 29 does not exist.")
                    return

                for i, xml_content in enumerate(extras_col.head(5)):
                    print(f"\n--- Article {i+1} ---")
                    # Handle potential NaN values before printing
                    if pd.notna(xml_content):
                        print(xml_content)
                    else:
                        print("(No XML content)")
                
                entity_found = True
                break # Exit after finding the first entity

        if not entity_found:
            print("No mentions of any tracked entities found in this file.")

    except subprocess.CalledProcessError as e:
        print(f"Download failed: {e.stderr.decode()}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if os.path.exists(temp_zip_path): os.remove(temp_zip_path)
        if os.path.exists(temp_csv_path): os.remove(temp_csv_path)
        print("\n--- Debugger Finished ---")

if __name__ == "__main__":
    debug_gdelt_xml()