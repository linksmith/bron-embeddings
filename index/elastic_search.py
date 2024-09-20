# index/elastic_search.py

import os
import sys
import re
import time
import logging
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    SparseVectorParams,
    VectorParams,
    PointStruct
)
from .utils import (
    format_time,
    suppress_stdout_stderr,
    remove_processing_instructions,
    process_document,
    batch_iterator,
    make_qdrant_points,
    upsert_with_progress,
    parse_date,
    run_ner_pipeline
)
from vector_store.qdrant import make_dense_embedding, make_sparse_embedding
import dateparser
import pynvml  # For GPU temperature monitoring
import multiprocessing

# Constants
NUM_WORKERS = 16
BATCH_SIZE = 10000
SLEEP_TIME = 0
DENSE_BATCH_SIZE = 512
NER_BATCH_SIZE = 512

def es(what_to_index='3_gemeentes'):
    # Define the path to your models directory
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['TRANSFORMERS_CACHE'] = models_dir

    # Connect to Elasticsearch
    es = Elasticsearch("http://localhost:9200")
    index_name = "jodal_documents7"

    # Define your queries
    queries = {
        '3_gemeentes': {
            "query": {
                "bool": {
                    "must": [
                        {"exists": {"field": "description"}}
                    ],
                    "minimum_should_match": 1,
                    "should": [
                        {"match_phrase": {"location": "GM0141"}},
                        {"match_phrase": {"location": "GM1896"}},
                        {"match_phrase": {"location": "GM0180"}}
                    ]
                }
            }
        },
        'overijssel': {
            "query": {
                "bool": {
                    "must": [
                        {"exists": {"field": "description"}}
                    ],
                    "minimum_should_match": 1,
                    "should": [
                        {"match_phrase": {"location": "GM0193"}},
                        {"match_phrase": {"location": "GM0141"}},
                        # ... (other match_phrase queries)
                        {"match_phrase": {"location": "GM1773"}}
                    ]
                }
            }
        },
        'all': {
            "query": {
                "bool": {
                    "must": [
                        {"exists": {"field": "description"}}
                    ]
                }
            }
        }
    }

    # Determine which query to use
    query = queries.get(what_to_index, queries['all'])

    # Determine collection name
    collection_name = f"{what_to_index}"

    # Initialize Qdrant client
    qdrant_client = QdrantClient(host="localhost", port=6333)

    # Create collection if it doesn't exist
    collections = qdrant_client.get_collections().collections
    if collection_name not in [col.name for col in collections]:
        qdrant_client.create_collection(
            collection_name,
            vectors_config={
                "text-dense": VectorParams(
                    size=1024,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "text-sparse": SparseVectorParams(
                    index={
                        "type": "IVF_FLAT",
                        "params": {
                            "nlist": 100
                        }
                    }
                )
            },
        )

    # Use scan to retrieve documents
    es_scan = scan(client=es, index=index_name, query=query)
    # Get total document count for progress bar
    total_docs = es.count(index=index_name, body=query)['count']
    batch_size = BATCH_SIZE
    batches = batch_iterator(es_scan, batch_size)
    total_points_processed = 0

    total_errors = 0  # Initialize total error counter

    # Start time for the entire process
    start_time = time.time()
    total_batches = (total_docs + batch_size - 1) // batch_size

    print(f"Processing {total_batches} batches of {batch_size}")
    with tqdm(total=total_docs, desc=f"Total progress", position=0, dynamic_ncols=True) as pbar:
        for batch_num, batch in enumerate(batches):
            batch_start_time = time.time()

            # Process documents in the batch
            max_workers = multiprocessing.cpu_count()
            with multiprocessing.Pool(processes=max_workers) as pool:
                processed_docs_iter = pool.imap_unordered(process_document, batch, chunksize=100)
                processed_docs = []
                batch_errors = 0  # Error count for the batch

                # Sub-progress bar for processing documents
                with tqdm(total=len(batch), desc="Processing documents", position=1, leave=False) as doc_pbar:
                    for result in processed_docs_iter:
                        doc_pbar.update(1)
                        if result is not None:
                            docs_list, error_count = result
                            batch_errors += error_count
                            if docs_list:
                                processed_docs.extend(docs_list)
                        else:
                            batch_errors += 1

            # Update total error count
            total_errors += batch_errors

            # Create DataFrame
            df = pd.DataFrame(processed_docs)
            # Remove problematic columns
            for col in df.columns:
                if df[col].apply(lambda x: isinstance(x, dict) and not x).all():
                    df.drop(columns=[col], inplace=True)

            # Process the batch
            num_points_in_batch = len(processed_docs)
            vector_store_main = __import__('vector_store.main', fromlist=['make_dense_embedding', 'make_sparse_embedding'])
            process_batch(df, collection_name, total_points_processed, qdrant_client)
            total_points_processed += num_points_in_batch

            # Update progress bar
            pbar.update(len(batch))

            # Time calculations
            elapsed_time = time.time() - start_time
            batches_processed = batch_num + 1
            avg_time_per_batch = elapsed_time / batches_processed
            estimated_total_time = avg_time_per_batch * total_batches
            remaining_time_estimate = estimated_total_time - elapsed_time

            # Update progress bar with time estimates
            pbar.set_postfix({
                'Elapsed': format_time(elapsed_time),
                'ETA': format_time(remaining_time_estimate)
            })

            # Sleep between batches (set to zero)
            time.sleep(SLEEP_TIME)  # No sleep needed as GPUs are underutilized

    # After all batches are processed, print total errors and total time
    total_elapsed_time = time.time() - start_time
    print(f"\nProcessing completed with {total_errors} errors encountered.")
    print(f"Total processing time: {format_time(total_elapsed_time)}")

def process_batch(df, collection_name, starting_id, qdrant_client):
    texts = df["text"].tolist()

    # Run embeddings
    df["dense_embedding"] = make_dense_embedding(texts)
    df["sparse_embedding"] = make_sparse_embedding(texts)
    df["entities"] = run_ner_pipeline(texts)

    # Create Qdrant points with unique IDs
    points = make_qdrant_points(df, starting_id)

    # Upsert points to Qdrant
    upsert_with_progress(qdrant_client, collection_name, points, batch_size=1000)
