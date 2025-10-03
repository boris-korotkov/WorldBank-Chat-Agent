import pandas as pd
import json
import os

# --- Configuration ---
METADATA_SOURCE_IS_CSV = False
METADATA_CSV_FILE = 'Country - Metadata.csv'
METADATA_EXCEL_FILE = 'A51-80.xlsx'
METADATA_SHEET_NAME = 'Country - Metadata'
OUTPUT_JSON_FILE = 'countries_map.json'

def generate_map_from_rich_metadata():
    """
    Reads the rich "Country - Metadata" sheet/CSV to generate a comprehensive
    map that standardizes on the 'Table Name' for data consistency.
    """
    print(f"Reading rich country metadata...")

    try:
        if METADATA_SOURCE_IS_CSV:
            if not os.path.exists(METADATA_CSV_FILE):
                print(f"FATAL: Source file not found at '{METADATA_CSV_FILE}'")
                return
            df = pd.read_csv(METADATA_CSV_FILE)
        else:
            df = pd.read_excel(METADATA_EXCEL_FILE, sheet_name=METADATA_SHEET_NAME)
    except Exception as e:
        print(f"Error reading source file: {e}")
        return

    df.dropna(subset=['Code'], inplace=True)
    print(f"Found {len(df)} total entries (countries and aggregates). Processing...")
    
    country_map = []
    
    for index, row in df.iterrows():
        long_name = row.get('Long Name')
        short_name = row.get('Short Name')
        # --- THIS IS THE KEY CHANGE ---
        # The 'Table Name' is the name used in the actual data files.
        # We MUST use this as our canonical name for filtering.
        table_name = row.get('Table Name')
        
        # Use Table Name as the primary canonical name.
        # Fall back to Short Name or Long Name if Table Name is missing.
        if pd.notna(table_name):
            canonical_name = table_name
        elif pd.notna(short_name):
            canonical_name = short_name
        else:
            canonical_name = long_name

        # Create a rich set of common names for robust matching by the LLM
        common_names = {short_name, long_name, table_name}
        
        entry = {
            "canonicalName": canonical_name,
            "iso3": row.get('Code'),
            "commonNames": sorted([name for name in common_names if pd.notna(name) and name]),
            "incomeGroup": row.get('Income Group') if pd.notna(row.get('Income Group')) else "Aggregate",
            "region": row.get('Region') if pd.notna(row.get('Region')) else "Aggregate"
        }
        
        country_map.append(entry)

    with open(OUTPUT_JSON_FILE, 'w') as f:
        json.dump(country_map, f, indent=2)
        
    print(f"\nSuccessfully created '{OUTPUT_JSON_FILE}' with {len(country_map)} entries.")
    print("The 'canonicalName' now matches the 'Table Name' used in the data files.")

if __name__ == '__main__':
    generate_map_from_rich_metadata()