import sys
import re
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import pandas as pd
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
import pyarrow as pa
import pyarrow.parquet as pq
from unstructured.partition.html import partition_html
from unstructured.cleaners.core import (
    clean,
    clean_non_ascii_chars,
    replace_unicode_quotes,
    group_broken_paragraphs
)
from lxml import etree

def remove_processing_instructions(html_text):
    """Remove processing instructions from HTML content using lxml."""
    try:
        parser = etree.HTMLParser(remove_pis=True)
        tree = etree.fromstring(html_text.encode('utf-8'), parser)
        return etree.tostring(tree, encoding='unicode', method='html')
    except Exception as e:
        # If parsing fails, return the original text
        tqdm.write(f"Error removing processing instructions: {e}", file=sys.stderr)
        return html_text

def process_document(doc):
    try:
        # Extract the 'description' field
        description = doc.get('_source', {}).get('description', '')
        description = remove_processing_instructions(description)

        # Process 'description' field using unstructured.io
        try:
            elements = partition_html(
                text=description, 
                chunking_strategy='by_title', 
                combine_text_under_n_chars=1024,
                max_characters=4096,
                new_after_n_chars=3072,
                overlap=256
            )
        except Exception as e:
            tqdm.write(f"Exception in partition_html for doc {doc.get('_id')}: {e}", file=sys.stderr)
            return None

        docs = []
        chunk_count = len(elements)

        for i, element in enumerate(elements):
            try:
                # Skip elements that are processing instructions
                if hasattr(element, 'element') and isinstance(element.element, etree._ProcessingInstruction):
                    continue

                if hasattr(element, 'text') and element.text:
                    cleaned_text = clean(
                        element.text,
                        extra_whitespace=True,
                        dashes=True,
                        bullets=True,
                        trailing_punctuation=False,
                        lowercase=False
                    )
                    cleaned_text = clean_non_ascii_chars(cleaned_text)
                    cleaned_text = replace_unicode_quotes(cleaned_text)
                    cleaned_text = group_broken_paragraphs(cleaned_text)

                    docs.append({
                        "text": cleaned_text,
                        "es_id": doc["_id"],
                        "title": doc["_source"].get("title", ""),
                        "location": doc["_source"].get("location", ""),
                        "location_name": doc["_source"].get("location_name", ""),
                        "modified": doc["_source"].get("modified", ""),
                        "published": doc["_source"].get("published", ""),
                        "source": doc["_source"].get("source", ""),
                        "type": doc["_source"].get("type", ""),
                        "identifier": doc["_source"].get("identifier", ""),
                        "url": doc["_source"].get("url", ""),
                        "chunk_index": i,
                        "chunk_count": chunk_count
                    })
                else:
                    # Skip elements without text
                    continue
            except Exception as e:
                tqdm.write(f"Exception processing element in doc {doc.get('_id')}: {e}", file=sys.stderr)
                continue

        return docs if docs else None
    except Exception as e:
        # Log exception and return None
        tqdm.write(f"Exception processing doc {doc.get('_id')}: {e}", file=sys.stderr)
        return None

def main(what_to_index='3_gemeentes'):
    # Connect to Elasticsearch
    es = Elasticsearch("http://localhost:9200")
    index_name = "jodal_documents7"
    query_3_gemeentes = {
        "query": {
            "bool": {
                "must": [
                    {"exists": {"field": "description"}}
                ],
                "minimum_should_match": 1,
                "should": [
                    {
                        "match_phrase": {
                        "location": "GM0141"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM1896"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0180"
                        }
                    }
                ]
            }
        }
    }
    query_overijssel = {
        "query": {
            "bool": {
                "must": [
                    {"exists": {"field": "description"}}
                ],
                "minimum_should_match": 1,
                "should": [
                    {
                        "match_phrase": {
                        "location": "GM0193"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0141"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0166"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM1774"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0183"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0173"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0150"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM1700"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0164"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0153"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0148"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM1708"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0168"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0160"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0189"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0177"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM1742"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0180"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM1896"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0175"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM1735"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0147"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0163"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM0158"
                        }
                    },
                    {
                        "match_phrase": {
                        "location": "GM1773"
                        }
                    }
                ]
            }
        }
    }

    query_all = {
        "query": {
            "bool": {
                "must": [
                    {"exists": {"field": "description"}}
                ]
            }
        }
    }

    if what_to_index == '3_gemeentes':
        query = query_3_gemeentes
    elif what_to_index == 'overijssel':
        query = query_overijssel
    else:
        query = query_all        

    # Use scan to retrieve documents
    es_scan = scan(client=es, index=index_name, query=query)

    # Get total document count for progress bar
    total_docs = es.count(index=index_name, body=query)['count']

    max_workers = multiprocessing.cpu_count()
    pool = Pool(processes=max_workers)
    processed_docs_iter = pool.imap_unordered(process_document, es_scan, chunksize=100)

    processed_docs = []
    batch_size = 1000  # Adjust based on your memory capacity
    batch_num = 0

    # Use a unique directory for batch Parquet files
    output_dir = 'output_parquet_batches'
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with tqdm(total=total_docs) as pbar:
        for docs_list in processed_docs_iter:
            # Update the progress bar for each Elasticsearch document processed
            pbar.update(1)
            if docs_list:
                processed_docs.extend(docs_list)
                if len(processed_docs) >= batch_size:
                    df = pd.DataFrame(processed_docs)
                    # Remove columns that are problematic for Parquet
                    for col in df.columns:
                        if df[col].apply(lambda x: isinstance(x, dict) and not x).all():
                            df.drop(columns=[col], inplace=True)
                    # Write each batch to a separate Parquet file
                    batch_file = os.path.join(output_dir, f'output_part{batch_num}.parquet')
                    df.to_parquet(batch_file, engine='pyarrow', index=False)
                    processed_docs = []
                    batch_num += 1
    # Write any remaining documents
    if processed_docs:
        df = pd.DataFrame(processed_docs)
        # Remove columns that are problematic for Parquet
        for col in df.columns:
            if df[col].apply(lambda x: isinstance(x, dict) and not x).all():
                df.drop(columns=[col], inplace=True)
        batch_file = os.path.join(output_dir, f'output_part{batch_num}.parquet')
        df.to_parquet(batch_file, engine='pyarrow', index=False)
        processed_docs = []

    pool.close()
    pool.join()

    # Combine all the Parquet files into a single dataset
    import glob
    all_files = glob.glob(os.path.join(output_dir, 'output_part*.parquet'))
    tables = [pq.read_table(f) for f in all_files]
    # Update the following line to address the FutureWarning
    combined_table = pa.concat_tables(tables, promote_options='default')
    parquet_file = f"{what_to_index}.parquet"
    pq.write_table(combined_table, parquet_file)
    
    # Optionally, clean up batch files
    for f in all_files:
        os.remove(f)
    os.rmdir(output_dir)

import jsonlines
import time
from typing import List, Tuple
from qdrant_client import QdrantClient
from fastembed.late_interaction import LateInteractionTextEmbedding
from fastembed.sparse import SparseTextEmbedding, SparseEmbedding
from fastembed.text import TextEmbedding
import pandas as pd

from qdrant_client.models import (
    Distance,
    NamedSparseVector,
    NamedVector,
    SparseVector,
    PointStruct,
    SearchRequest,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
    ScoredPoint,
    MultiVectorConfig,
    MultiVectorComparator
)
import concurrent.futures
import onnxruntime as ort

from transformers import BertTokenizer, BertForTokenClassification
from transformers import pipeline
import torch
import spacy
from tqdm import tqdm
import dateparser

NUM_WORKERS = 16

# Initialize Qdrant client
qdrant_client = QdrantClient(
    host="localhost",
    port=6333
)

# Create two separate inference sessions, one for each GPU
session_options_0 = ort.SessionOptions()
session_options_0.execution_mode = ort.ExecutionMode.ORT_PARALLEL
session_options_0.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options_0.intra_op_num_threads = NUM_WORKERS

session_options_1 = ort.SessionOptions()
session_options_1.execution_mode = ort.ExecutionMode.ORT_PARALLEL
session_options_1.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options_1.intra_op_num_threads = NUM_WORKERS

session_options_2 = ort.SessionOptions()
session_options_2.execution_mode = ort.ExecutionMode.ORT_PARALLEL
session_options_2.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options_2.intra_op_num_threads = NUM_WORKERS

session_options_3 = ort.SessionOptions()
session_options_3.execution_mode = ort.ExecutionMode.ORT_PARALLEL
session_options_3.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options_3.intra_op_num_threads = NUM_WORKERS


# Initialize the models on different GPUs
dense_document_embedder_0 = TextEmbedding(
    model_name="intfloat/multilingual-e5-large",
    session_options=session_options_0,
    providers=[("CUDAExecutionProvider", {"device_id": 0})]
)

dense_document_embedder_1 = TextEmbedding(
    model_name="intfloat/multilingual-e5-large",
    session_options=session_options_1,
    providers=[("CUDAExecutionProvider", {"device_id": 1})]
)

sparse_document_embedder_0 = SparseTextEmbedding(
    model_name="Qdrant/bm25",
    session_options=session_options_2,
    providers=[("CUDAExecutionProvider", {"device_id": 0})]
)

sparse_document_embedder_1 = SparseTextEmbedding(
    model_name="Qdrant/bm25",
    session_options=session_options_3,
    providers=[("CUDAExecutionProvider", {"device_id": 1})]
)

# late_interaction_document_embedder = LateInteractionTextEmbedding(
#     model_name="colbert-ir/colbertv2.0",
#     session_options=session_options_2,
#     providers=[("CUDAExecutionProvider", {"device_id": 1})]    
# )

# Load the small Dutch model

# def make_late_interaction_embedding(texts: List[str]):
#     return list(late_interaction_document_embedder.embed(texts))

# def make_sparse_embedding(texts: List[str]):
#     return list(sparse_document_embedder.embed(texts))

# def embed_documents_on_gpu(embedder, documents, progress_bar):
#     embeddings = []
#     for doc in documents:
#         embedding = embedder.embed(doc)  # Replace with actual embedding generation logic
#         embeddings.append(embedding)
#         progress_bar.update(1)
#     return embeddings

def embed_documents_on_gpu(embedder, documents, progress_bar):
    # 'embedder.embed(documents)' returns a generator over embeddings
    embedding_gen = embedder.embed(documents)
    embeddings = []
    for embedding in embedding_gen:
        embeddings.append(embedding)
        progress_bar.update(1)
    return embeddings

def make_sparse_embedding(texts: List[str]) -> List[SparseEmbedding]:
    total_docs = len(texts)
    
    # Split documents between the two GPUs
    mid_point = total_docs // 2
    docs_0 = texts[:mid_point]
    docs_1 = texts[mid_point:]

    with tqdm(total=total_docs, desc="Generating sparse embeddings") as progress_bar:
        # Run inference on both GPUs in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_0 = executor.submit(embed_documents_on_gpu, sparse_document_embedder_0, docs_0, progress_bar)
            future_1 = executor.submit(embed_documents_on_gpu, sparse_document_embedder_1, docs_1, progress_bar)
            
            embeddings_0 = future_0.result()
            embeddings_1 = future_1.result()

    embeddings = embeddings_0 + embeddings_1
    return embeddings

# Function to embed documents on a specific embedder
# def embed_documents_on_gpu(embedder, documents):
#     return list(embedder.embed(documents))

def make_dense_embedding(texts: List[str]):    
    # Split documents between the two GPUs
    mid_point = len(texts) // 2
    docs_0 = texts[:mid_point]
    docs_1 = texts[mid_point:]

    # Run inference on both GPUs in parallel
    total_docs = len(texts)
    with tqdm(total=total_docs, desc="Generating dense embeddings") as progress_bar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_0 = executor.submit(embed_documents_on_gpu, dense_document_embedder_0, docs_0, progress_bar)
            future_1 = executor.submit(embed_documents_on_gpu, dense_document_embedder_1, docs_1, progress_bar)
            
            embeddings_0 = future_0.result()
            embeddings_1 = future_1.result()

    # Combine results
    embeddings = embeddings_0 + embeddings_1
    return embeddings

def parse_date(input_string):
    date = dateparser.parse(input_string, languages=['nl'])
    if date is None:
        return None
    else:
        return date.strftime('%Y-%m-%d')
    
def make_qdrant_points(df: pd.DataFrame) -> List[PointStruct]:
    print("Creating Qdrant points")
    sparse_vectors = df["sparse_embedding"].tolist()
    combined_text = df["combined_text"].tolist()
    dense_vectors = df["dense_embedding"].tolist()
    # late_interaction_vector = df["late_interaction"].tolist()
    rows = df.to_dict(orient="records")
    points = []
    for idx, (combined_text, sparse_vector, dense_vector) in enumerate(
        zip(combined_text, sparse_vectors, dense_vectors)
    ):
        # sparse_vector = SparseVector(
        #     indices=sparse_vector.indices.tolist(), 
        #     values=sparse_vector.values.tolist()
        # )
        # Convert sparse_vector to dictionary if it's a generator
        if isinstance(sparse_vector, (list, tuple)):
            sparse_vector = {
                "indices": sparse_vector.indices.tolist(), 
                "values": sparse_vector.values.tolist()
            }
        else:
            sparse_vector = list(sparse_vector)
            sparse_vector = {
                "indices": [sv.indices.tolist() for sv in sparse_vector],  # Convert to list
                "values": [sv.values.tolist() for sv in sparse_vector]  # Convert to list
            }
                    
        # prepare the entities for the qdrant points
        entities, dates, locations, people, organisations, times = [], [], [], [], [], []
        entity_types = ["DATE", "GPE", "PERSON", "ORG", "TIME"]
        for entity in rows[idx]["entities"].ents:
            if entity.label_ in entity_types:
                entities.append({
                    "text": entity.text,  # Use dot notation to access attributes
                    "label": entity.label_,
                    "start": entity.start_char,
                    "end": entity.end_char,
                })

                if entity.label_ == "DATE":
                    if entity.text not in dates:
                        date = parse_date(entity.text)
                        if date is not None:
                            dates.append(date)
                elif entity.label_ == "GPE":
                    if entity.text not in locations:
                        locations.append(entity.text)                
                elif entity.label_ == "PERSON":
                    if entity.text not in people:
                        people.append(entity.text)
                elif entity.label_ == "ORG":
                    if entity.text not in organisations:
                        organisations.append(entity.text)

        point = PointStruct(
            id=idx,
            payload={
                "title": rows[idx]["title"],
                "location_name": rows[idx]["location_name"],
                "location": rows[idx]["location"],
                "text": combined_text,
                "source": rows[idx]["source"],
                "es_id": rows[idx]["es_id"],
                "modified": rows[idx]["modified"],
                "published": rows[idx]["published"],
                "type": rows[idx]["type"],
                "identifier": rows[idx]["identifier"],
                "url": rows[idx]["url"],
                "ner_entities": entities,
                "ner_dates": dates,
                "ner_locations": locations,
                "ner_people": people,
                "ner_organisations": organisations,
                "ner_times": times,
                "chunk_index": rows[idx]["chunk_index"],
                "chunk_count": rows[idx]["chunk_count"],
            },  # Add any additional payload if necessary
            vector={
                "text-sparse": sparse_vector,
                # "text-late-interaction": late_interaction_vector,      
                "text-dense": dense_vector,          
            },
        )
        points.append(point)

    return points

def ner_pipeline(docs, gpu_id, progress_bar, batch_size=512):
    entities = []
    torch.cuda.empty_cache()
    torch.cuda.set_device(gpu_id)
    spacy.require_gpu()
    nlp = spacy.load("nl_core_news_lg")
    
    for doc in nlp.pipe(docs, batch_size=batch_size):
        entities.append(doc)
        progress_bar.update(1)
    
    return entities

def run_ner_pipeline(texts: List[str], batch_size=512):
    print( "Running NER pipeline")
    # Split documents between the two GPUs
    mid_point = len(texts) // 2
    docs_0 = texts[:mid_point]
    docs_1 = texts[mid_point:]

    # Run inference on both GPUs in parallel
    total_docs = len(texts)
    with tqdm(total=total_docs, desc="Running NER pipeline") as progress_bar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_0 = executor.submit(ner_pipeline, docs_0, 0, progress_bar, batch_size)
            future_1 = executor.submit(ner_pipeline, docs_1, 1, progress_bar, batch_size)
            
            entities_0 = future_0.result()
            entities_1 = future_1.result()

    # Combine results
    entities = entities_0 + entities_1
    return entities
    

def upsert_with_progress(collection_name, points, batch_size=1000):
    qdrant_client.create_collection(
        collection_name,
        vectors_config={
            "text-dense": VectorParams(
                size=1024,  # OpenAI Embeddings
                distance=Distance.COSINE,
            ),
            # "text-late-interaction": VectorParams(
            #     size=5, 
            #     distance=Distance.COSINE,
            #     multivector_config=MultiVectorConfig(
            #         comparator=MultiVectorComparator.MAX_SIM
            #     ),
            # ),
        },
        sparse_vectors_config={
            "text-sparse": SparseVectorParams(
                index=SparseIndexParams(
                    on_disk=False,
                )
            )
        },
    )

    total_points = len(points)
    with tqdm(total=total_points, desc="Upserting points") as progress_bar:
        for i in range(0, total_points, batch_size):
            batch = points[i:i + batch_size]
            qdrant_client.upsert(collection_name, batch)
            progress_bar.update(len(batch))

# load the data from parquet
def load_data(df, collection_name):
    # increment collection name based on existing collections
    from qdrant_client.models import CollectionsResponse
    collections : CollectionsResponse = qdrant_client.get_collections()
    collection_name = f"{collection_name}-{len(collections.collections)}"    
    
    # create a new field which combines the fields text, title and location_name into into a single text  like this f'Titel: {value}. Gemeente:  {value}. Tekst: {value}'
    df["combined_text"] = df.apply(lambda x: f'Titel: {x["title"]} \n Gemeente: {x["location_name"]} \n Tekst: \n\n {x["text"]}', axis=1)

    combined_texts = df["combined_text"].tolist()

    # if f"dense_embedding_{collection_name}.parquet" exits, load it and return it, otherwise make the dense embedding
    try:
        df = pd.read_parquet(f"dense_embedding_{collection_name}.parquet", engine="pyarrow")
        print("Loaded dense embedding from parquet")
    except:
        df["dense_embedding"] = make_dense_embedding(combined_texts)
        df.to_parquet(f"dense_embedding_{collection_name}.parquet")
    
    try:
        df["sparse_embedding"] = df["sparse_embedding"].apply(lambda x: {"indices": x.indices.tolist(), "values": x.values.tolist()})
        df = pd.read_parquet(f"dense_and_sparse_embedding_{collection_name}.parquet", engine="pyarrow")
        print("Loaded sparse embedding from parquet")
    except:        
        df["sparse_embedding"] = make_sparse_embedding(combined_texts)
        df.to_parquet(f"dense_and_sparse_embedding_{collection_name}.parquet")

    try:
        df = pd.read_parquet(f"dense_and_sparse_embedding_and_entities_{collection_name}.parquet", engine="pyarrow")
        print("Loaded entities embedding from parquet")
    except:        
        df["entities"] = run_ner_pipeline(combined_texts)
        df.to_parquet(f"dense_and_sparse_embedding_and_entities_{collection_name}.parquet")

    points = make_qdrant_points(df)

    upsert_with_progress(collection_name, points)

    df.to_parquet(f"all_{collection_name}.parquet")

    # delete the temp parquet files
    import os
    os.remove(f"dense_embedding_{collection_name}.parquet")
    os.remove(f"dense_and_sparse_embedding_{collection_name}.parquet")
    os.remove(f"dense_and_sparse_embedding_and_entities_{collection_name}.parquet")

    return df

def run(collection_name = '3_gemeentes', docs_to_load = None):
    collection_name = f"{collection_name}.parquet"
    df = pd.read_parquet(collection_name, engine="pyarrow")

    if docs_to_load is not None:
        df = df[:docs_to_load]

    df = load_data(df, collection_name)

    return df

import sys

if __name__ == "__main__":
    print("Starting the script")
    # Clear
    print("Clearing CUDA memory")

    torch.cuda.empty_cache()

    # sys.argv[0] is the script name
    # sys.argv[1] is the first argument
    if len(sys.argv) > 1:
        what_to_index = sys.argv[1]
        print(f"Indexing {what_to_index} documents in Qdrant.")
    else:
        print("No command-line arguments were provided.")
        print("Please provide the name of the Parquet file to index.")
        print("Example: python es_to_qdrant.py 3_gemeentes")
        sys.exit(1)

    if what_to_index not in ["3_gemeentes", "overijssel", "all"]:
        print("Invalid argument. Please provide one of the following:")
        print("3_gemeentes, overijssel, all")
        sys.exit(1)

    main(what_to_index)
    run(what_to_index)
