import pandas as pd
import numpy as np
import re
import os
import argparse

# --- Configuration ---
S3_BUCKET = 'gen-ai-data-agent-bucket-us-west-2'
S3_BASE_PATH = 'processed'

def process_and_upload_data(excel_file, sheet_name, source_name, mode):
    """
    Reads the data sheet, cleans it, and either creates a new partitioned
    Parquet dataset or updates an existing one.
    """
    print(f"--- Starting data processing in '{mode.upper()}' MODE ---")
    
    # ... (The entire data reading and cleaning section remains exactly the same) ...
    try:
        print(f"Reading new data from '{excel_file}', sheet '{sheet_name}'...")
        new_df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
        
        first_empty_row_index = new_df[new_df.isnull().all(axis=1)].index.min()
        new_df.columns = new_df.iloc[0]
        new_df = new_df.iloc[1:first_empty_row_index].reset_index(drop=True)
        
        def clean_year_column(col):
            if isinstance(col, str):
                match = re.search(r'\b(\d{4})\b', col)
                return match.group(1) if match else col
            return col
        new_df.rename(columns=clean_year_column, inplace=True)

        rename_map = {'Series Code': 'index_code', 'Country Name': 'country_name', 'Country Code': 'country_code'}
        new_df.rename(columns=rename_map, inplace=True)
        
        year_columns = [col for col in new_df.columns if str(col).isdigit() and len(str(col)) == 4]
        for col in year_columns:
            new_df[col] = pd.to_numeric(new_df[col], errors='coerce')

        if 'Series Name' in new_df.columns:
            new_df.drop(columns=['Series Name'], inplace=True)
            
    except Exception as e:
        print(f"FATAL: Could not read or process the source Excel file. Error: {e}")
        return

    s3_source_path = f"s3://{S3_BUCKET}/{S3_BASE_PATH}/{source_name}"
    
    if mode == 'create':
        print(f"\nCreating new partitioned dataset at {s3_source_path}...")
        try:
            new_df.to_parquet(path=s3_source_path, engine='pyarrow', compression='snappy', partition_cols=['index_code'])
            print("Successfully created new dataset in S3.")
        except Exception as e:
            print(f"ERROR during 'create' upload: {e}")
            
    elif mode == 'update':
        print(f"\nUpdating existing partitioned dataset at {s3_source_path}...")
        
        for index_code, group_df in new_df.groupby('index_code'):
            print(f"  - Processing updates for index: {index_code}")
            
            partition_path = f"{s3_source_path}/index_code={index_code}"
            
            try:
                print(f"    Reading existing data from {partition_path}...")
                old_partition_df = pd.read_parquet(partition_path)
                
                id_cols = ['country_name', 'country_code']
                old_partition_df.set_index(id_cols, inplace=True)
                group_df.set_index(id_cols, inplace=True)
                
                old_partition_df.update(group_df)
                
                merged_df = old_partition_df.reset_index()
                print("    Successfully merged new data.")

            except Exception as e:
                print(f"    WARNING: Could not read existing partition. Assuming it's new. Error: {e}")
                # We need to drop the index_code from the new group to match the schema of a partition
                merged_df = group_df.reset_index().drop(columns=['index_code'], errors='ignore')
            
            # --- THIS IS THE FIX ---
            # Re-introduce the 'index_code' column into the final DataFrame before writing.
            # This ensures the schema is correct for the to_parquet function.
            merged_df['index_code'] = index_code
            
            try:
                # We now write the single, updated partition back to S3
                # This is more efficient than writing the entire dataset again
                merged_df.to_parquet(
                    path=s3_source_path,
                    engine='pyarrow',
                    compression='snappy',
                    partition_cols=['index_code']
                )
                print(f"    Successfully wrote updated data for partition {index_code}.")
            except Exception as e:
                print(f"    ERROR: Could not write updated partition for {index_code}. Error: {e}")

# ... (the __main__ block remains exactly the same) ...
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create or update partitioned Parquet data from Excel.")
    parser.add_argument('--file', required=True, help="Path to the source Excel file.")
    parser.add_argument('--sheet', required=True, help="Name of the data sheet in the Excel file.")
    parser.add_argument('--source', required=True, help="Name of the data source (e.g., 'world-bank').")
    parser.add_argument('--mode', required=True, choices=['create', 'update'], help="Mode: 'create' a new dataset or 'update' an existing one.")
    
    args = parser.parse_args()
    
    process_and_upload_data(
        excel_file=args.file,
        sheet_name=args.sheet,
        source_name=args.source,
        mode=args.mode
    )