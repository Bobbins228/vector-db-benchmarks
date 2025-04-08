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
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

benchmark_dir = os.path.dirname(os.path.abspath(__file__))
Path(benchmark_dir + "/images").mkdir(parents=True, exist_ok=True) # Create images dir if it doesn't exist for charts

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
def create_histogram(evaluations_df: pd.DataFrame, search_modes: list, db_provider_id: str):
    for search_mode in search_modes:
        precisions = evaluations_df[evaluations_df["search_mode"] == search_mode]["precision"].tolist()
        plt.figure()
        plt.xlabel('Precision')
        plt.title(f'Precision per Query - CISI {db_provider_id} {search_mode}')

        plt.hist(precisions)
        plt.savefig(f'images/histogram_cisi_ds_{db_provider_id}_{search_mode}.png') 

# Function to create a Radar Chart based on precision, recall and f1 score values
def create_radar_chart(evaluations_df: pd.DataFrame, search_modes: list, db_provider_id: str):
    # Radar chart setup
    labels = ['Precision', 'Recall', 'F1 Score']
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    fig, axes = plt.subplots(subplot_kw=dict(polar=True))

    for search_mode in search_modes:
        precision = evaluations_df.loc[evaluations_df["search_mode"] == search_mode, "precision"].values.tolist()
        recall = evaluations_df.loc[evaluations_df["search_mode"] == search_mode, "recall"].values.tolist()
        f1 = evaluations_df.loc[evaluations_df["search_mode"] == search_mode, "f1_score"].values.tolist()

        averages = [
            np.mean(precision),
            np.mean(recall),
            np.mean(f1)
        ]

        averages += averages[:1]

        axes.plot(angles, averages, label=f'{search_mode.capitalize()} Search', linewidth=2)
        axes.fill(angles, averages, alpha=0.25)

    axes.set_xticks(angles[:-1])
    axes.set_xticklabels(labels)
    axes.set_title('Search Method Comparison')
    axes.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.savefig(f"images/{db_provider_id}_radar_chart.png")

# Function to create a Bar Chart based on precision, recall and f1 score values
def create_bar_chart(evaluations_df: pd.DataFrame, search_modes: list, db_provider_id: str):
    x = np.arange(len(search_modes))
    width = 0.25

    precision_averages=[]
    recall_averages=[]
    f1_score_averages=[]
    for search_mode in search_modes:
        precision = evaluations_df.loc[evaluations_df["search_mode"] == search_mode, "precision"].values.tolist()
        recall = evaluations_df.loc[evaluations_df["search_mode"] == search_mode, "recall"].values.tolist()
        f1 = evaluations_df.loc[evaluations_df["search_mode"] == search_mode, "f1_score"].values.tolist()

        precision_averages.append(np.mean(precision))
        recall_averages.append(np.mean(recall))
        f1_score_averages.append(np.mean(f1))
    
    fig, axes = plt.subplots(figsize=(10, 6))
    axes.bar(x - width, precision_averages, width, label='Precision')
    axes.bar(x, recall_averages, width, label='Recall', color='orange')
    axes.bar(x + width, f1_score_averages, width, label='F1-Score', color='green')

    # Labels and title
    axes.set_ylabel('Score')
    axes.set_title(f'Benchmark: {db_provider_id}')
    axes.set_xticks(x)
    axes.set_xticklabels(search_modes)
    axes.legend()
    plt.savefig(f"images/{db_provider_id}_evaluation_bar_chart")


# Returns a list of document ids from the results
def extract_document_ids(input_string: str):
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

def calculate_metrics(doc_ids, sot_doc_ids, evaluations_df, search_mode):
    retrieved_ids = doc_ids
    sot_ids = [str(i) for i in sot_doc_ids] # Convert the source of truth docs to strings for comparison
    
    # Get true positives and calculate percision
    true_positives = sum(1 for doc_id in retrieved_ids if doc_id in sot_ids)
    precision = true_positives / len(retrieved_ids) if retrieved_ids else 0.0

    # Calculate recall
    recall = true_positives / len(sot_ids) if sot_ids else 0.0

    # Calculate f1 score
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)

    data = {
        "search_mode": search_mode,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }
    evaluations_df.loc[len(evaluations_df)] = data


# Connect to the Llama Stack Server
client = create_http_client() # or create_library_client() depending on the environment you picked

def run_benchmark(db_provider_id: str, search_modes: list):
    # The following are CSV files created from the CISI dataset https://www.kaggle.com/datasets/dmaso01dsta/cisi-a-dataset-for-information-retrieval/code
    cisi_df = pd.read_csv(benchmark_dir + "/data/cisi-content.csv") # DATA DOC (Based on CISI.ALL)
    queries_df = pd.read_csv(benchmark_dir + "/data/cisi-queries.csv") # Queries doc (based on CISIS.QUERY)
    sot_docs_df = pd.read_csv(benchmark_dir + "/data/cisi-sot.csv") # Source of Truth doc ids & query ids (based on CISI.REL)

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
    # Setup a dataframe that will house all precision, recall and f1 score values for each search mode type
    evaluations_df = pd.DataFrame(columns=["search_mode", "precision", "recall", "f1_score"])

    for query in queries_df["query"]:
        # Get the query ID for this specific run
        query_id = queries_df.loc[queries_df["query"] == query, 'query_id'].values[0] # ID of current query
        query_ids.append(int(query_id))

        # Filter the sorce of truth doc ids by the query id and extract doc_id values as a list
        filtered_df = sot_docs_df[sot_docs_df['query_id'] == int(query_id)]
        sot_doc_ids = filtered_df['doc_id'].tolist()

        run_rag_query(vector_db_id, query, search_modes, sot_doc_ids, evaluations_df)

    create_histogram(evaluations_df, search_modes, db_provider_id)
    create_radar_chart(evaluations_df, search_modes, db_provider_id)
    create_bar_chart(evaluations_df, search_modes, db_provider_id)
    evaluations_df.to_csv(f"data/{db_provider_id}_evaluations.csv") # Save a record of evaluations

def run_rag_query(vector_db_id: str, query: str, search_modes: list, sot_doc_ids: list, evaluations_df: pd.DataFrame):
    for search_mode in search_modes:
        if search_mode == "keyword":
            # Sanitise the query to be accepted by Keyword search
            clean_query = re.sub(r'[^a-zA-Z0-9\s]', ' ', query)
            query = ' '.join(clean_query.split())

        query_config = RAGQueryConfig(max_chunks=6, search_mode=search_mode).model_dump()
        results = client.tool_runtime.rag_tool.query(
            vector_db_ids=[vector_db_id], content=query, query_config=query_config
        )

        calculate_metrics(extract_document_ids(results.to_json()), sot_doc_ids, evaluations_df, search_mode)

def main():
    args = parse_args()
    run_benchmark(args.vector_db_id, args.search_modes)

if __name__ == "__main__":
    main()
