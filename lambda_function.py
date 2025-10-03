import json
import os
import boto3
import faiss
import numpy as np
import pandas as pd
from datetime import datetime
import re
import uuid
import random

# --- CONFIGURATION ---
S3_BUCKET_NAME = 'gen-ai-data-agent-bucket-us-west-2'
MAP_S3_KEY = 'metadata/faiss_id_to_metadata.json'
INDEX_S3_KEY = 'metadata/index.faiss'
COUNTRIES_MAP_S3_KEY = 'metadata/countries_map.json'
PROCESSED_DATA_S3_PATH = 'processed/world-bank'
REPORTS_S3_PATH = 'reports'

# Bedrock Model and Region Configuration
EMBEDDING_MODEL_ID = 'amazon.titan-embed-text-v2:0'
# --- THIS LINE IS NOW CORRECTED ---
# REASONING_MODEL_ID = 'anthropic.claude-3-5-sonnet-20241022-v2:0' 
REASONING_MODEL_ID = 'meta.llama3-1-8b-instruct-v1:0' 

AWS_REGION = 'us-west-2'

# --- Global variables, AWS Clients, load_index_and_metadata ---
faiss_index, id_to_metadata_map, countries_map = None, None, None
s3_client = boto3.client('s3')
bedrock_runtime = boto3.client(service_name='bedrock-runtime', region_name=AWS_REGION)

def format_lex_html_response_and_clear_context(message_html):
    """Closes the intent and clears session attributes."""
    return {
        "sessionState": {
            "dialogAction": {"type": "Close"},
            "intent": {"name": "QueryData", "state": "Fulfilled"},
            "sessionAttributes": {} # This clears the context
        },
        "messages": [{"contentType": "CustomPayload", "content": message_html}]
    }

def format_lex_elicit_intent_and_clear_context(message_text):
    """Elicits a new intent and clears session attributes."""
    return {
        "sessionState": {
            "dialogAction": {"type": "ElicitIntent"},
            "sessionAttributes": {} # This clears the context
        },
        "messages": [{"contentType": "PlainText", "content": message_text}]
    }

def load_index_and_metadata():
    global faiss_index, id_to_metadata_map, countries_map
    local_index_path, local_map_path, local_countries_path = '/tmp/index.faiss', '/tmp/faiss_id_to_metadata.json', '/tmp/countries_map.json'
    if not os.path.exists(local_index_path): s3_client.download_file(S3_BUCKET_NAME, INDEX_S3_KEY, local_index_path)
    if not os.path.exists(local_map_path): s3_client.download_file(S3_BUCKET_NAME, MAP_S3_KEY, local_map_path)
    if not os.path.exists(local_countries_path): s3_client.download_file(S3_BUCKET_NAME, COUNTRIES_MAP_S3_KEY, local_countries_path)
    faiss_index = faiss.read_index(local_index_path)
    with open(local_map_path, 'r') as f: id_to_metadata_map = {int(k): v for k, v in json.load(f).items()}
    with open(local_countries_path, 'r') as f: countries_map = json.load(f)
    print("Initialization complete.")

def plan_query_with_llm(user_query, candidate_metadata, country_list):
    print(f"Planning query with model: {REASONING_MODEL_ID}...")
    candidate_info = ""
    for metadata in candidate_metadata:
        candidate_info += f"- Code: {metadata.get('index_code')}, Name: {metadata.get('indicator_name')}\n"
    current_year = datetime.now().year
    
    # --- THIS PROMPT IS UPDATED WITH A NEW RULE FOR "ALL INDICATORS" ---
    prompt = f"""
You are a data query planner that ONLY outputs a single, valid JSON object. Your entire response must be the JSON object itself and nothing else.

**RULES:**
- Do NOT add any reasoning, explanations, or conversational text.
- Do NOT add any text before or after the JSON object.
- Your response must be in one of the formats specified below.

**OUTPUT FORMATS:**
1. For specific data queries, use this format:
   {{
     "indicators": ["<indicator_code_1>", "<indicator_code_2>"],
     "countries": ["<country_name>"],
     "start_year": <YYYY>,
     "end_year": <YYYY>
   }}
2. If the user asks for a list of available indicators, use this format:
   {{
     "action": "list_indicators"
   }}
3. If the user asks for "all indicators" or "all data" for a specific country/region, use the special value ["all"] for the indicators field. Extract the countries and years as normal.
   {{
     "indicators": ["all"],
     "countries": ["<country_name>"],
     "start_year": <YYYY>,
     "end_year": <YYYY>
   }}
---
**TASK:**
Now, analyze the following query and generate the JSON plan.

Candidate Indicators (for specific data queries):
{candidate_info}
Valid Countries and Aggregates (for data queries):
{country_list}
Current year for reference: {current_year}.

User's Query: "{user_query}"
"""
    
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    body = json.dumps({
        "prompt": formatted_prompt,
        "max_gen_len": 1024,
        "temperature": 0.0,
    })

    response = bedrock_runtime.invoke_model(
        modelId=REASONING_MODEL_ID, body=body, contentType='application/json', accept='application/json'
    )
    
    response_body = json.loads(response.get('body').read())
    response_text = response_body.get('generation')
    print(f"LLM query plan raw response: {response_text}")

    # The rest of the function remains the same...
    try:
        json_start_index = response_text.find('{')
        if json_start_index == -1: return None
        brace_level = 0
        json_end_index = -1
        for i in range(json_start_index, len(response_text)):
            char = response_text[i]
            if char == '{': brace_level += 1
            elif char == '}': brace_level -= 1
            if brace_level == 0:
                json_end_index = i + 1
                break
        if json_end_index == -1: return None
        json_str = response_text[json_start_index:json_end_index]
        print(f"Successfully extracted query plan JSON: {json_str}")
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"ERROR: Could not decode JSON. Error: {e}. Raw text: {response_text}")
        return None

# --- All other functions (lambda_handler, etc.) remain exactly the same ---
def generate_and_upload_indicator_list():
    print("Generating full indicator list text file...")
    indicator_lines = [f"{v['indicator_name']} (Code: {v['index_code']})" for k, v in id_to_metadata_map.items()]
    content = "\n".join(sorted(indicator_lines))
    file_name = f"indicator-list-{uuid.uuid4()}.txt"
    local_path = f"/tmp/{file_name}"
    s3_key = f"{REPORTS_S3_PATH}/{file_name}"
    with open(local_path, 'w', encoding='utf-8') as f:
        f.write(content)
    s3_client.upload_file(local_path, S3_BUCKET_NAME, s3_key)
    url = s3_client.generate_presigned_url('get_object', Params={'Bucket': S3_BUCKET_NAME, 'Key': s3_key}, ExpiresIn=600)
    print(f"Successfully generated and uploaded indicator list to S3. URL: {url}")
    return url

def lambda_handler(event, context):
    if event.get('source') == 'lambda-warmer':
        print("Warming Lambda function...")
        load_index_and_metadata()
        return {'statusCode': 200, 'body': 'Warmed'}

    if faiss_index is None:
        load_index_and_metadata()

    user_query = event.get('inputTranscript', '').strip()
    session_attributes = event.get('sessionState', {}).get('sessionAttributes', {}) or {}

    # The entire logic is now wrapped in a single try...except block for robustness.
    try:
        # This block now uses the new functions to clear the session context after completion.
        if session_attributes.get('intentContext') == 'confirmIndicatorListDownload':
            affirmative_responses = ['yes', 'sure', 'ok', 'yeah', 'yep', 'please']
            if any(word in user_query.lower() for word in affirmative_responses):
                if id_to_metadata_map is None:
                    load_index_and_metadata()
                download_url = generate_and_upload_indicator_list()
                response_html = f"I've prepared the full list for you. <a href=\"{download_url}\" target=\"_blank\">Click here to download the list of all indicators.</a>"
                # --- FIX: Use the new function to clear the context ---
                return format_lex_html_response_and_clear_context(response_html)
            else:
                # --- FIX: Use the new function to clear the context ---
                return format_lex_elicit_intent_and_clear_context("No problem. What data can I look up for you?")

        if not user_query:
            return format_lex_text_response("I'm sorry, I didn't understand. Could you please rephrase?")

        query_embedding = get_embedding(user_query)
        candidate_metadata = search_faiss_index(query_embedding, top_k=20)
        country_names_list = [c['canonicalName'] for c in countries_map]
        query_plan = plan_query_with_llm(user_query, candidate_metadata, country_names_list)

        if not query_plan:
             return format_lex_text_response("I'm sorry, I had trouble understanding your request. Could you please rephrase?")

        if query_plan.get('action') == 'list_indicators':
            all_indicators = list(id_to_metadata_map.values())
            sample_indicators = random.sample(all_indicators, k=min(3, len(all_indicators)))
            sample_names = [f"'{ind['indicator_name']}'" for ind in sample_indicators]
            prompt_message = (
                f"I can provide data on over 100 indicators. For example: {', '.join(sample_names)}. "
                "Would you like a full list of all available indicators as a downloadable text file?"
            )
            new_session_attributes = {"intentContext": "confirmIndicatorListDownload"}
            return format_lex_elicit_intent_with_context(prompt_message, new_session_attributes)
        elif query_plan.get('indicators') == ["all"] and query_plan.get('countries'):
            print("Handling 'all indicators' request. Building full indicator list.")
            countries = query_plan.get('countries', [])
            
            # Manually build the list of all available indicator codes
            all_indicator_codes = [meta['index_code'] for meta in id_to_metadata_map.values() if 'index_code' in meta]
            
            if not all_indicator_codes:
                return format_lex_text_response("I'm sorry, I couldn't retrieve the master list of indicators.")

            # Fetch the data for all indicators
            data_df = fetch_multi_indicator_data(all_indicator_codes, query_plan['start_year'], query_plan['end_year'], countries)

            if data_df is None or data_df.empty:
                return format_lex_text_response(f"I understood your request, but couldn't find any data for the specified criteria.")
            
            # Generate and return the Excel report
            download_url = generate_and_upload_report(data_df)
            response_html = f"I have prepared the data for all available indicators for {', '.join(countries)}. <a href=\"{download_url}\" target=\"_blank\">Click here to download the report.</a>"
            return format_lex_html_response(response_html)

        # Handle "normal" multi-indicator/multi-country requests
        elif query_plan.get('indicators') and query_plan.get('countries'):
            indicators = query_plan.get('indicators', [])
            countries = query_plan.get('countries', [])
            data_df = fetch_multi_indicator_data(indicators, query_plan['start_year'], query_plan['end_year'], countries)
            
            if data_df is None or data_df.empty:
                return format_lex_text_response(f"I understood your request, but couldn't find any data for the specified criteria.")
        

            is_simple_query = (len(indicators) == 1 and len(countries) == 1 and query_plan['start_year'] == query_plan['end_year'])
            is_multi_year_simple_query = (len(indicators) == 1 and len(countries) == 1 and query_plan['start_year'] != query_plan['end_year'])

            if is_simple_query:
                country = data_df['country_name'].iloc[0]; year = str(query_plan['start_year']); indicator_name = data_df.columns[2]; value = data_df.iloc[0, 2]
                formatted_value = f"{value:,.2f}" if pd.notna(value) and isinstance(value, (int, float)) else "not available"
                response_text = f"The value for '{indicator_name}' in {country} for the year {year} was {formatted_value}."
                return format_lex_text_response(response_text)
            elif is_multi_year_simple_query:
                country = data_df['country_name'].iloc[0]; indicator_name = data_df.columns[2]; table_df = data_df[['year', indicator_name]].copy()
                table_df.rename(columns={'year': 'Year', indicator_name: indicator_name.title()}, inplace=True)
                value_col_name = table_df.columns[1]
                table_df[value_col_name] = table_df[value_col_name].apply(lambda x: f"{x:,.2f}" if pd.notna(x) and isinstance(x, (int, float)) else "N/A")
                html_table = table_df.to_html(index=False, na_rep="N/A", justify='left', border=0)
                html_table = html_table.replace('<table border="0" class="dataframe">', '<table style="border-collapse: collapse; width: 100%;">')
                html_table = html_table.replace('<th>', '<th style="text-align: left; padding: 5px; border-bottom: 1px solid #ddd;">')
                html_table = html_table.replace('<td>', '<td style="text-align: left; padding: 5px;">')
                response_html = f"Here is the data for '{indicator_name}' in {country}:<br>{html_table}"
                return format_lex_html_response(response_html)
            else:
                download_url = generate_and_upload_report(data_df)
                response_html = f"I have prepared the data you requested. <a href=\"{download_url}\" target=\"_blank\">Click here to download the report.</a>"
                return format_lex_html_response(response_html)
        
        else:
            return format_lex_text_response("I'm sorry, I couldn't understand the details of your request. Could you please be more specific?")

    except Exception as e:
        print(f"An unhandled error occurred: {e}")
        import traceback
        traceback.print_exc()
        return format_lex_text_response("I'm sorry, an internal error occurred. Please try again later.")

def fetch_multi_indicator_data(indicator_codes, start_year, end_year, countries):
    print(f"Fetching data for indicators {indicator_codes} for countries: {countries}...")
    all_indicator_data = []
    for code in indicator_codes:
        partition_path = f"s3://{S3_BUCKET_NAME}/{PROCESSED_DATA_S3_PATH}/index_code={code}"
        try:
            df = pd.read_parquet(partition_path)
            df = df[df['country_name'].isin(countries)]
            if df.empty: continue
            indicator_name = "Unknown"
            for mid, mdata in id_to_metadata_map.items():
                if mdata.get('index_code') == code: indicator_name = mdata.get('indicator_name'); break
            id_vars = ['country_name', 'country_code']
            value_vars = [str(y) for y in range(start_year, end_year + 1) if str(y) in df.columns]
            if not value_vars: continue
            melted_df = df.melt(id_vars=id_vars, value_vars=value_vars, var_name='year', value_name='value')
            melted_df['indicator'] = indicator_name
            all_indicator_data.append(melted_df)
        except Exception as e:
            print(f"Warning: Could not read or process data for indicator {code}. Error: {e}")
    if not all_indicator_data: return None
    combined_df = pd.concat(all_indicator_data, ignore_index=True)
    final_df = combined_df.pivot_table(index=['country_name', 'year'], columns='indicator', values='value').reset_index()
    final_df.columns.name = None
    final_df['year'] = pd.to_numeric(final_df['year'])
    final_df.sort_values(by=['country_name', 'year'], inplace=True)
    return final_df

def generate_and_upload_report(df):
    file_name = f"report-{uuid.uuid4()}.xlsx"; local_path = f"/tmp/{file_name}"; s3_key = f"{REPORTS_S3_PATH}/{file_name}"
    df.to_excel(local_path, index=False, engine='xlsxwriter')
    s3_client.upload_file(local_path, S3_BUCKET_NAME, s3_key)
    url = s3_client.generate_presigned_url('get_object', Params={'Bucket': S3_BUCKET_NAME, 'Key': s3_key}, ExpiresIn=600)
    return url
    
def get_embedding(text):
    body = json.dumps({"inputText": text}); response = bedrock_runtime.invoke_model(body=body, modelId=EMBEDDING_MODEL_ID, contentType='application/json', accept='application/json'); response_body = json.loads(response['body'].read()); return response_body['embedding']

def search_faiss_index(query_embedding, top_k=20):
    query_vector = np.array([query_embedding]).astype('float32'); distances, indices = faiss_index.search(query_vector, top_k); results_indices = indices[0]; return [id_to_metadata_map.get(i) for i in results_indices if id_to_metadata_map.get(i) is not None]

def format_lex_text_response(message_text):
    return {"sessionState": {"dialogAction": {"type": "Close"}, "intent": {"name": "QueryData", "state": "Fulfilled"}}, "messages": [{"contentType": "PlainText", "content": message_text}]}

def format_lex_html_response(message_html):
    return {"sessionState": {"dialogAction": {"type": "Close"}, "intent": {"name": "QueryData", "state": "Fulfilled"}}, "messages": [{"contentType": "CustomPayload", "content": message_html}]}

def format_lex_elicit_intent_response(message_text):
    return {"sessionState": {"dialogAction": {"type": "ElicitIntent"}}, "messages": [{"contentType": "PlainText", "content": message_text}]}

def format_lex_elicit_intent_with_context(message_text, session_attributes):
    return {"sessionState": {"dialogAction": {"type": "ElicitIntent"}, "sessionAttributes": session_attributes}, "messages": [{"contentType": "PlainText", "content": message_text}]}