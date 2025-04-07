# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.
import re
import os
import uuid
import argparse

from llama_stack_client.types import Document

from llama_stack.apis.tools import RAGQueryConfig
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Search script with database and search-modes")

    parser.add_argument(
        "--vector_db_id",
        type=str,
        required=False,
        help="Name of the database provider id to use",
        default="sqlite-vec"
    )

    parser.add_argument(
        "--search-modes",
        type=str,
        nargs='+',
        required=False,
        help="Search mode(s) to use. Provide a single string or multiple strings separated by space. Accepted args vector, keyword",
        default=["vector"]
    )

    return parser.parse_args()

def create_http_client():
    from llama_stack_client import LlamaStackClient

    return LlamaStackClient(base_url=f"http://localhost:{os.environ['LLAMA_STACK_PORT']}")

def create_library_client(template="ollama"):
    from llama_stack import LlamaStackAsLibraryClient

    client = LlamaStackAsLibraryClient(template)
    client.initialize()
    return client

# Function to create a histogram based on precision values
def create_histogram(precisions_df, search_modes, db_provider_id):
    import matplotlib.pyplot as plt

    for search_mode in search_modes:
        precisions = precisions_df[precisions_df["search_mode"] == search_mode]["precision"].tolist()
        plt.xlabel('Precision')
        plt.title(f'Precision per Query - CISI {db_provider_id} {search_mode}')

        plt.hist(precisions)
        plt.savefig(f'histogram_cisi_ds_{db_provider_id}_{search_mode}.png') 


# Returns a list of document ids from the results
def extract_document_ids(input_string):
    import re
    import json

    # Regex pattern to find document_ids array
    pattern = r'"document_ids"\s*:\s*\[([^\]]*)\]'
    match = re.search(pattern, input_string)
    
    if match:
        try:
            # Extract the array content and parse it
            ids_str = match.group(1)
            # Use json.loads to handle parsing, including quotes
            ids = json.loads(f'[{ids_str}]')
            return ids
        except (json.JSONDecodeError, ValueError):
            return []
    
    return []

def calculate_metrics(doc_ids, true_doc_ids, precisions_df, search_mode):
    # Calculate precision
    retrieved_ids = doc_ids
    sot_ids = true_doc_ids
    
    str_list = []
    for doc in sot_ids:
        str_list.append(str(doc)) # need to convert doc ids to string for below calculation
    
    matching_doc_ids = sum(1 for doc_id in retrieved_ids if doc_id in str_list)
    precision = matching_doc_ids / len(retrieved_ids) if retrieved_ids else 0
    
    data = {
        "search_mode": search_mode,
        "precision": precision
    }
    precisions_df.loc[len(precisions_df)] = data

client = create_http_client() # or create_library_client() depending on the environment you picked

def run_benchmark(db_provider_id: str, search_modes: list):
    benchmark_dir = os.path.dirname(os.path.abspath(__file__))
    # The following are CSV files created from the CISI dataset https://www.kaggle.com/datasets/dmaso01dsta/cisi-a-dataset-for-information-retrieval/code
    cisi_df = pd.read_csv(benchmark_dir + "/cisi-content.csv") # DATA DOC (Based on CISI.ALL)
    queries_df = pd.read_csv(benchmark_dir + "/cisi-queries.csv") # Queries doc (based on CISIS.QUERY)
    sot_docs_df = pd.read_csv(benchmark_dir + "/cisi-sot.csv") # Source of Truth doc ids & query ids (based on CISI.REL)

    # Create a list of Documents containing the content of CISI.ALL and associated Document IDs
    documents = [
        Document(
            document_id=str(cisi_df["doc_id"][i]),
            content=cisi_df["content"][i],
            mime_type="text/plain",
            metadata={},
        )
        for i, _ in enumerate(cisi_df["doc_id"])
    ]

    # Register a database
    vector_db_id = f"test-vector-db-{uuid.uuid4().hex}"
    client.vector_dbs.register(
        provider_id=db_provider_id,
        vector_db_id=vector_db_id,
        embedding_model="all-MiniLM-L6-v2",
        embedding_dimension=384,
    )

    # Insert the documents into the vector database
    client.tool_runtime.rag_tool.insert(
        documents=documents,
        vector_db_id=vector_db_id,
        chunk_size_in_tokens=512,
        timeout=600
    )

    query_ids = []
    # Setup a dataframe that will house all precision values for each search mode type
    precisions_df = pd.DataFrame(columns=["search_mode", "precision"])

    for query in queries_df["query"]:
        # Get the query ID for this specific run
        query_id = queries_df.loc[queries_df["query"] == query, 'query_id'].values[0] # ID of current query
        query_ids.append(int(query_id))

        # Filter the sorce of truth doc ids by the query id and extract doc_id values as a list
        filtered_df = sot_docs_df[sot_docs_df['query_id'] == int(query_id)]
        true_doc_ids = filtered_df['doc_id'].tolist()

        run_rag_query(vector_db_id, query, search_modes, true_doc_ids, precisions_df)

    create_histogram(precisions_df, search_modes, db_provider_id)

def run_rag_query(vector_db_id: str, query: str, search_modes: list, true_doc_ids: list, precisions_df: pd.DataFrame):
    for search_mode in search_modes:
        if search_mode == "keyword":
            # Sanitise the query to be accepted by Keyword search
            clean_query = re.sub(r'[^a-zA-Z0-9\s]', ' ', query)
            query = ' '.join(clean_query.split())

        query_config = RAGQueryConfig(max_chunks=6, search_mode=search_mode).model_dump()
        results = client.tool_runtime.rag_tool.query(
            vector_db_ids=[vector_db_id], content=query, query_config=query_config
        )

        calculate_metrics(extract_document_ids(results.to_json()), true_doc_ids, precisions_df, search_mode)

def main():
    args = parse_args()
    run_benchmark(args.vector_db_id, args.search_modes)

if __name__ == "__main__":
    main()
