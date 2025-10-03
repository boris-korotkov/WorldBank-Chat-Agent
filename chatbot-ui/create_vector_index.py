import pandas as pd
import numpy as np
import faiss
import boto3
import json
import os
import argparse # We will use argparse to handle command-line flags

# --- Configuration ---
# Your S3 and Bedrock configs are now defined here
S3_BUCKET = 'gen-ai-data-agent-bucket-us-west-2' # Using your new bucket
S3_METADATA_FOLDER = 'metadata'
BEDROCK_REGION = 'us-west-2'
EMBEDDING_MODEL_ID = 'amazon.titan-embed-text-v2:0'

# File names for our index and map
INDEX_FILE_NAME = 'index.faiss'
MAP_FILE_NAME = 'faiss_id_to_metadata.json'

# --- AWS Clients ---
s3_client = boto3.client('s3')
bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name=BEDROCK_REGION)

def get_embedding(text):
    """Generates an embedding for a given text using Amazon Titan."""
    body = json.dumps({"inputText": text})
    response = bedrock_runtime.invoke_model(
        body=body, modelId=EMBEDDING_MODEL_ID, contentType='application/json', accept='application/json'
    )
    response_body = json.loads(response['body'].read())
    return response_body['embedding']

def process_metadata_sheet(file_path, sheet_name):
    """Reads and processes the metadata sheet from an Excel file."""
    print(f"Reading metadata from '{file_path}', sheet '{sheet_name}'...")
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    rename_map = {
        'Code': 'index_code', 'Indicator Name': 'indicator_name', 'Long definition': 'long_definition',
        'Topic': 'topic', 'Source': 'source'
    }
    df.rename(columns=rename_map, inplace=True, errors='ignore')
    df.dropna(subset=['index_code', 'indicator_name'], inplace=True)
    
    df['combined_text'] = df.apply(
        lambda row: f"Indicator: {row.get('indicator_name', '')}. "
                    f"Topic: {row.get('topic', '')}. "
                    f"Definition: {row.get('long_definition', '')}",
        axis=1
    )
    return df

def run_update_process(excel_file, sheet_name, mode):
    """
    Main function to either create a new FAISS index or append to an existing one.
    
    Args:
        excel_file (str): Path to the Excel file with new metadata.
        sheet_name (str): Name of the metadata sheet in the Excel file.
        mode (str): Either 'create' or 'append'.
    """
    
    # --- Load or Initialize Index and Map ---
    if mode == 'append':
        print("--- APPEND MODE ---")
        try:
            # Download existing files from S3 to a temporary local location
            print("Downloading existing index and map from S3...")
            s3_client.download_file(S3_BUCKET, f"{S3_METADATA_FOLDER}/{INDEX_FILE_NAME}", f"./{INDEX_FILE_NAME}")
            s3_client.download_file(S3_BUCKET, f"{S3_METADATA_FOLDER}/{MAP_FILE_NAME}", f"./{MAP_FILE_NAME}")
            
            # Load the downloaded files
            faiss_index = faiss.read_index(INDEX_FILE_NAME)
            with open(MAP_FILE_NAME, 'r') as f:
                id_to_metadata_map = {int(k): v for k, v in json.load(f).items()}
            
            print(f"Successfully loaded existing index with {faiss_index.ntotal} vectors.")
            
        except Exception as e:
            print(f"ERROR: Could not download or load existing index files from S3: {e}")
            print("Please ensure the files exist or run in 'create' mode first.")
            return
    else: # mode == 'create'
        print("--- CREATE MODE ---")
        # Initialize empty index and map
        faiss_index = None
        id_to_metadata_map = {}

    # --- Process New Data ---
    new_metadata_df = process_metadata_sheet(excel_file, sheet_name)
    
    # Filter out indicators that are already in our map
    existing_codes = {v['index_code'] for v in id_to_metadata_map.values()}
    indicators_to_add_df = new_metadata_df[~new_metadata_df['index_code'].isin(existing_codes)]
    
    if indicators_to_add_df.empty:
        print("No new indicators found to add. Exiting.")
        return

    print(f"Found {len(indicators_to_add_df)} new indicators to process.")
    
    # --- Generate Embeddings for NEW data only ---
    print("Generating embeddings for new indicators (this may take a while)...")
    indicators_to_add_df['embedding'] = indicators_to_add_df['combined_text'].apply(get_embedding)
    
    new_embeddings = np.array(indicators_to_add_df['embedding'].tolist()).astype('float32')
    
    # --- Update Index and Map ---
    if faiss_index is None: # First time running in 'create' mode
        dimension = new_embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(dimension)

    # Add the new vectors to the FAISS index
    faiss_index.add(new_embeddings)
    
    # Update the JSON map
    next_id = max(id_to_metadata_map.keys()) + 1 if id_to_metadata_map else 0
    for idx, row in indicators_to_add_df.iterrows():
        id_to_metadata_map[next_id] = {
            'index_code': row['index_code'],
            'indicator_name': row['indicator_name']
        }
        next_id += 1
        
    print(f"Index updated. New total vector count: {faiss_index.ntotal}")

    # --- Save and Upload Final Files ---
    print("Saving updated index and map files locally...")
    faiss.write_index(faiss_index, INDEX_FILE_NAME)
    with open(MAP_FILE_NAME, "w") as f:
        json.dump(id_to_metadata_map, f)
        
    try:
        print("Uploading updated files to S3 (overwriting previous versions)...")
        s3_client.upload_file(INDEX_FILE_NAME, S3_BUCKET, f"{S3_METADATA_FOLDER}/{INDEX_FILE_NAME}")
        s3_client.upload_file(MAP_FILE_NAME, S3_BUCKET, f"{S3_METADATA_FOLDER}/{MAP_FILE_NAME}")
        print("\nSuccessfully updated index and map files in S3!")
    except Exception as e:
        print(f"\nError uploading files to S3: {e}")
    finally:
        # Clean up local copies
        if os.path.exists(INDEX_FILE_NAME): os.remove(INDEX_FILE_NAME)
        if os.path.exists(MAP_FILE_NAME): os.remove(MAP_FILE_NAME)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create or update the FAISS index for indicators.")
    parser.add_argument('--file', required=True, help="Path to the source Excel file.")
    parser.add_argument('--sheet', required=True, help="Name of the metadata sheet in the Excel file.")
    parser.add_argument('--mode', required=True, choices=['create', 'append'], help="Mode of operation: 'create' a new index or 'append' to an existing one.")
    
    args = parser.parse_args()
    
    run_update_process(excel_file=args.file, sheet_name=args.sheet, mode=args.mode)