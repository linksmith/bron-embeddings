import sys
import re
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor  # Use ProcessPoolExecutor for multiprocessing
import pyarrow as pa
from unstructured.partition.html import partition_html
from unstructured.cleaners.core import (
    clean,
    clean_non_ascii_chars,
    replace_unicode_quotes,
    group_broken_paragraphs
)
from lxml import etree
import time
import os
import glob
import torch
import concurrent.futures
import onnxruntime as ort
import spacy
from typing import List, Tuple
from qdrant_client import QdrantClient
from fastembed.late_interaction import LateInteractionTextEmbedding
from fastembed.sparse import SparseTextEmbedding, SparseEmbedding
from fastembed.text import TextEmbedding
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
    MultiVectorComparator,
    CollectionsResponse
)
import dateparser
import pynvml  # For GPU temperature monitoring
import contextlib  # For suppressing stdout and stderr
import logging  # For adjusting logging levels
from joblib import Parallel, delayed
from math import ceil
from tqdm_joblib import tqdm_joblib  # Import tqdm_joblib


# Define the path to your models directory
models_dir = os.path.join(os.path.dirname(__file__), 'models')

# Suppress specific FutureWarning from transformers library
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.utils.generic")
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Suppress logging messages from the 'unstructured' library
logging.getLogger('unstructured').setLevel(logging.ERROR)
logging.getLogger('lxml').setLevel(logging.ERROR)

# Constants
NUM_WORKERS = 16
BATCH_SIZE = 1000  # Number of documents to process from Elasticsearch in each batch
SLEEP_TIME = 0     # Set sleep time to zero since GPUs are underutilized
DENSE_BATCH_SIZE = 512   # Increased batch size for dense embeddings per GPU
NER_BATCH_SIZE = 512    # Increased batch size for NER per GPU

# Initialize NVML for GPU temperature monitoring (optional, can be removed if not needed)
pynvml.nvmlInit()

def format_time(seconds):
    """Format time in seconds to HH:MM:SS."""
    hours = int(seconds) // 3600
    minutes = (int(seconds) % 3600) // 60
    seconds = int(seconds) % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

@contextlib.contextmanager
def suppress_stdout_stderr():
    """Suppress stdout and stderr."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

def remove_processing_instructions(html_text):
    """Remove processing instructions from HTML content using lxml."""
    try:
        with suppress_stdout_stderr():
            parser = etree.HTMLParser(remove_pis=True)
            tree = etree.fromstring(html_text.encode('utf-8'), parser)
            return etree.tostring(tree, encoding='unicode', method='html')
    except Exception:
        # If parsing fails, return the original text
        # Error is handled silently
        return html_text

def process_document(doc):
    error_count = 0  # Initialize error counter
    try:
        # Extract the 'description' field
        description = doc.get('_source', {}).get('description', '')
        try:
            description = remove_processing_instructions(description)
        except Exception:
            # If parsing fails, keep the original description
            error_count += 1

        # Process 'description' field using unstructured.io
        try:
            with suppress_stdout_stderr():
                elements = partition_html(
                    text=description,
                    chunking_strategy='by_title',
                    combine_text_under_n_chars=1024,
                    max_characters=4096,
                    new_after_n_chars=3072,
                    overlap=256
                )
        except Exception:
            # Error in partition_html
            error_count += 1
            return None, error_count  # Return None and error count

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

                    title = doc["_source"].get("title", "")
                    location_name = doc["_source"].get("location_name", "")
                    cleaned_text = f'Titel: {title} \n Gemeente: {location_name} \n Tekst: \n\n {cleaned_text}'

                    docs.append({
                        "text": cleaned_text,
                        "es_id": doc["_id"],
                        "title": title,
                        "location": doc["_source"].get("location", ""),
                        "location_name": location_name,
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
            except Exception:
                # Error processing element
                error_count += 1
                continue

        return (docs if docs else None), error_count
    except Exception:
        # Log exception and return None
        error_count += 1
        return None, error_count

def batch_iterator(iterator, batch_size):
    """Yield batches of documents from an iterator."""
    batch = []
    for item in iterator:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def parse_date(input_string):
    date = dateparser.parse(input_string, languages=['nl'])
    if date is None:
        return None
    else:
        return date.strftime('%Y-%m-%d')
    
def prepare_entities_for_qdrant(rows, idx):
    """Prepare the entities for the qdrant points."""
    entities, dates, locations, people, organisations, times = [], [], [], [], [], []
    entity_types = ["DATE", "GPE", "PERSON", "ORG", "TIME"]
    for entity in rows[idx]["entities"].ents:
        if entity.label_ in entity_types:
            entities.append({
                "text": entity.text,
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
    return entities, dates, locations, people, organisations, times

def make_qdrant_points(df: pd.DataFrame, starting_id) -> List[PointStruct]:
    sparse_vectors = df["sparse_embedding"].tolist()
    text = df["text"].tolist()
    dense_vectors = df["dense_embedding"].tolist()
    rows = df.to_dict(orient="records")
    points = []
    for idx, (text, sparse_vector, dense_vector) in enumerate(
        zip(text, sparse_vectors, dense_vectors)
    ):
        # Convert sparse_vector to dictionary
        if isinstance(sparse_vector, SparseEmbedding):
            sparse_vector = {
                "indices": sparse_vector.indices.tolist(),
                "values": sparse_vector.values.tolist()
            }
        else:
            # Handle list of SparseEmbeddings
            sparse_vector = {
                "indices": [sv.indices.tolist() for sv in sparse_vector],
                "values": [sv.values.tolist() for sv in sparse_vector]
            }

        # Use the new function in the existing code
        # entities, dates, locations, people, organisations, times = prepare_entities_for_qdrant(rows, idx)

        point = PointStruct(
            id=starting_id + idx,  # Ensure unique IDs across batches
            payload={
                "title": rows[idx]["title"],
                "location_name": rows[idx]["location_name"],
                "location": rows[idx]["location"],
                "text": text,
                "source": rows[idx]["source"],
                "es_id": rows[idx]["es_id"],
                "modified": rows[idx]["modified"],
                "published": rows[idx]["published"],
                "type": rows[idx]["type"],
                "identifier": rows[idx]["identifier"],
                "url": rows[idx]["url"],
                # "ner_entities": entities,
                # "ner_dates": dates,
                # "ner_locations": locations,
                # "ner_people": people,
                # "ner_organisations": organisations,
                # "ner_times": times,
                "chunk_index": rows[idx]["chunk_index"],
                "chunk_count": rows[idx]["chunk_count"],
            },
            vector={
                "text-sparse": sparse_vector,
                "text-dense": dense_vector,
            },
        )
        points.append(point)

    return points

def upsert_with_progress(collection_name, points, batch_size=1000):
    total_points = len(points)
    with tqdm(total=total_points, desc="Upserting points", position=2, leave=False) as progress_bar:
        for i in range(0, total_points, batch_size):
            batch = points[i:i + batch_size]
            qdrant_client.upsert(collection_name, batch)
            progress_bar.update(len(batch))

def process_batch(df, collection_name, starting_id):
    texts = df["text"].tolist()

    # Run embeddings
    df["dense_embedding"] = make_dense_embedding(texts)
    df["sparse_embedding"] = make_sparse_embedding(texts)
    # # time the difference:
    # start = time.time()

    # df["entities_0"] = run_spacy_ner_pipeline(texts)
    # end = time.time()
    # print(f"spaCy time taken: {end - start}")

    # start = time.time()
    # df["entities_1"] = run_flair_ner_pipeline(texts)
    # end = time.time()
    # print(f"Flair time taken: {end - start}")

    # Create Qdrant points with unique IDs
    points = make_qdrant_points(df, starting_id)

    # Upsert points to Qdrant
    upsert_with_progress(collection_name, points, batch_size=1000)

def main(what_to_index='3_gemeentes'):
    # Connect to Elasticsearch
    es = Elasticsearch("http://localhost:9200")
    index_name = "jodal_documents7"

    query_1_gemeente = {
        "query": {
            "bool": {
                "must": [
                    {"exists": {"field": "description"}}
                ],
                "minimum_should_match": 1,
                "should": [
                    {"match_phrase": {"location": "GM0141"}},
                ]
            }
        }
    }

    # Define your queries
    query_3_gemeentes = {
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
    }
    query_overijssel = {
        "query": {
            "bool": {
                "must": [
                    {"exists": {"field": "description"}}
                ],
                "minimum_should_match": 1,
                "should": [
                    {"match_phrase": {"location": "GM0193"}},
                    {"match_phrase": {"location": "GM0141"}},
                    {"match_phrase": {"location": "GM0166"}},
                    {"match_phrase": {"location": "GM1774"}},
                    {"match_phrase": {"location": "GM0183"}},
                    {"match_phrase": {"location": "GM0173"}},
                    {"match_phrase": {"location": "GM0150"}},
                    {"match_phrase": {"location": "GM1700"}},
                    {"match_phrase": {"location": "GM0164"}},
                    {"match_phrase": {"location": "GM0153"}},
                    {"match_phrase": {"location": "GM0148"}},
                    {"match_phrase": {"location": "GM1708"}},
                    {"match_phrase": {"location": "GM0168"}},
                    {"match_phrase": {"location": "GM0160"}},
                    {"match_phrase": {"location": "GM0189"}},
                    {"match_phrase": {"location": "GM0177"}},
                    {"match_phrase": {"location": "GM1742"}},
                    {"match_phrase": {"location": "GM0180"}},
                    {"match_phrase": {"location": "GM1896"}},
                    {"match_phrase": {"location": "GM0175"}},
                    {"match_phrase": {"location": "GM1735"}},
                    {"match_phrase": {"location": "GM0147"}},
                    {"match_phrase": {"location": "GM0163"}},
                    {"match_phrase": {"location": "GM0158"}},
                    {"match_phrase": {"location": "GM1773"}}
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

    # Determine which query to use    
    if what_to_index == '1_gemeente':
        query = query_1_gemeente
    if what_to_index == '3_gemeentes':
        query = query_3_gemeentes
    elif what_to_index == 'overijssel':
        query = query_overijssel
    else:
        query = query_all

    # Determine collection name
    collection_name = f"{what_to_index}"

    # Initialize Qdrant client
    global qdrant_client
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
                    index=SparseIndexParams(on_disk=False)
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
            process_batch(df, collection_name, total_points_processed)
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

def embed_documents_on_gpu(embedder, documents, progress_bar, batch_size, gpu_index, position):
    embeddings = []
    total_docs = len(documents)
    for i in range(0, total_docs, batch_size):
        batch_docs = documents[i:i + batch_size]
        # 'embedder.embed(batch_docs)' returns a generator over embeddings
        embedding_gen = embedder.embed(batch_docs)
        for embedding in embedding_gen:
            embeddings.append(embedding)
            progress_bar.update(1)
    return embeddings

def make_dense_embedding(texts: List[str]):
    batch_size_per_gpu = DENSE_BATCH_SIZE
    # Split documents between the two GPUs
    mid_point = len(texts) // 2
    docs_0 = texts[:mid_point]
    docs_1 = texts[mid_point:]

    # Run inference on both GPUs in parallel
    total_docs = len(texts)
    with tqdm(total=total_docs, desc="Generating dense embeddings", position=1, leave=False) as progress_bar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_0 = executor.submit(embed_documents_on_gpu, dense_document_embedder_0, docs_0, progress_bar, batch_size_per_gpu, 0, 2)
            future_1 = executor.submit(embed_documents_on_gpu, dense_document_embedder_1, docs_1, progress_bar, batch_size_per_gpu, 1, 2)

            embeddings_0 = future_0.result()
            embeddings_1 = future_1.result()

    # Combine results
    embeddings = embeddings_0 + embeddings_1
    return embeddings

def make_sparse_embedding(texts: List[str]) -> List[SparseEmbedding]:
    total_docs = len(texts)

    # Split documents between the two GPUs
    mid_point = total_docs // 2
    docs_0 = texts[:mid_point]
    docs_1 = texts[mid_point:]

    batch_size_per_gpu = 512  # You can adjust this if needed

    with tqdm(total=total_docs, desc="Generating sparse embeddings", position=1, leave=False) as progress_bar:
        # Run inference on both GPUs in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_0 = executor.submit(embed_documents_on_gpu, sparse_document_embedder_0, docs_0, progress_bar, batch_size_per_gpu, 0, 2)
            future_1 = executor.submit(embed_documents_on_gpu, sparse_document_embedder_1, docs_1, progress_bar, batch_size_per_gpu, 1, 2)

            embeddings_0 = future_0.result()
            embeddings_1 = future_1.result()

    embeddings = embeddings_0 + embeddings_1
    return embeddings

def split_list(lst: List[str], n: int) -> List[List[str]]:
    from math import ceil
    """
    Split a list into n approximately equal parts.

    Args:
        lst (List[str]): The list to split.
        n (int): Number of sublists.

    Returns:
        List[List[str]]: A list containing n sublists.
    """
    avg = ceil(len(lst) / n)
    return [lst[i * avg : (i + 1) * avg] for i in range(n)]

def load_nlp():
    """
    Function to load the NLP model.
    This will be called in each worker process.
    """
    return spacy.load("nl_core_news_lg")  # Replace with your model


def ner_pipeline(docs):
    nlp = load_nlp()  # Load the NLP model in each worker
    entities = []
    for doc in nlp.pipe(docs, batch_size=512):  # Adjust batch_size as needed
        entities.append(doc)
    return entities

def chunker(lst: List[str], total: int, chunksize: int) -> List[List[str]]:
    """Yield successive chunks from lst of size chunksize."""
    for i in range(0, total, chunksize):
        yield lst[i:i + chunksize]

def flatten(list_of_lists: List[List[str]]) -> List[str]:
    """Flatten a list of lists into a single list."""
    return [item for sublist in list_of_lists for item in sublist]

def run_spacy_ner_pipeline(texts: List[str]) -> List[str]:
    """
    Process a list of texts using NER in parallel with a progress bar.

    Args:
        texts (List[str]): The list of texts to process.

    Returns:
        List[str]: The list of processed texts.
    """
    batch_size = NER_BATCH_SIZE
    total_texts = len(texts)
    max_workers = multiprocessing.cpu_count()

    # Calculate the number of chunks
    num_chunks = ceil(total_texts / batch_size)

    # Initialize the progress bar using tqdm_joblib
    with tqdm_joblib(tqdm(total=num_chunks, desc="Processing spaCy NER pipeline", unit="chunk")):
        # Set up the Parallel executor
        executor = Parallel(
            n_jobs=max_workers,
            backend='multiprocessing',
            prefer="processes",
            verbose=0  # Set to 0 to avoid joblib's own progress messages
        )
        do = delayed(ner_pipeline)
        tasks = (do(chunk) for chunk in chunker(texts, total_texts, chunksize=batch_size))
        result = executor(tasks)

    # Flatten the list of results
    return flatten(result)

def run_flair_nlp(run_flair_nlp, texts: List[str], progress_bar, batch_size, gpu_index):
    import flair
    from flair.data import Sentence
    from flair.nn import Classifier
    from flair.splitter import SegtokSentenceSplitter

    docs = []
    flair.device = torch.device(gpu_index)  # cuda:0
    # initialize sentence splitter
    splitter = SegtokSentenceSplitter()

    for text in texts:
        entities = []

        # use splitter to split text into list of sentences
        sentences = splitter.split(text)
        run_flair_nlp.predict(sentences, mini_batch_size=batch_size)

        # iterate through sentences and print predicted labels
        for sentence in sentences:
            entities.append(sentence.to_dict(tag_type='ner'))

        docs.append(entities)           

        progress_bar.update(1) 

    return docs

def run_flair_ner_pipeline(texts: List[str]):
    total_docs = len(texts)

    # Split documents between the two GPUs
    mid_point = total_docs // 2
    docs_0 = texts[:mid_point]
    docs_1 = texts[mid_point:]

    batch_size_per_gpu = 512  # You can adjust this if needed

    with tqdm(total=total_docs, desc="Generating NER entities", position=1, leave=False) as progress_bar:
        # Run inference on both GPUs in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_0 = executor.submit(run_flair_nlp, flair_ner_0, texts, progress_bar, batch_size_per_gpu, 0)
            # future_1 = executor.submit(run_flair_nlp, flair_ner_1, docs_1, progress_bar, batch_size_per_gpu, 1)

            entities_0 = future_0.result()
            # entities_1 = future_1.result()

    return entities_0
    # entities = entities_0 + entities_1
    # return entities

# Initialize the models on different GPUs
def initialize_models():
    from flair.nn import Classifier

    global dense_document_embedder_0, dense_document_embedder_1, flair_ner_0
    global sparse_document_embedder_0, sparse_document_embedder_1, flair_ner_1

    # Create two separate inference sessions, one for each GPU
    session_options_0 = ort.SessionOptions()
    session_options_0.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    session_options_0.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options_0.intra_op_num_threads = NUM_WORKERS

    session_options_1 = ort.SessionOptions()
    session_options_1.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    session_options_1.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options_1.intra_op_num_threads = NUM_WORKERS

    # Initialize the models on different GPUs
    dense_document_embedder_0 = TextEmbedding(
        cache_dir=models_dir,
        model_name="intfloat/multilingual-e5-large",
        session_options=session_options_0,
        providers=[("CUDAExecutionProvider", {"device_id": 0})]
    )

    dense_document_embedder_1 = TextEmbedding(
        cache_dir=models_dir,
        model_name="intfloat/multilingual-e5-large",
        session_options=session_options_1,
        providers=[("CUDAExecutionProvider", {"device_id": 1})]
    )

    sparse_document_embedder_0 = SparseTextEmbedding(
        cache_dir=models_dir,
        model_name="Qdrant/bm25",
        session_options=session_options_0,
        providers=[("CUDAExecutionProvider", {"device_id": 0})]
    )

    sparse_document_embedder_1 = SparseTextEmbedding(
        cache_dir=models_dir,
        model_name="Qdrant/bm25",
        session_options=session_options_1,
        providers=[("CUDAExecutionProvider", {"device_id": 1})]
    )

    flair_ner_0 = Classifier.load('nl-ner-large')
    # flair_ner_1 = Classifier.load('nl-ner-large')

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    print("Starting the script")

    # Initialize models
    initialize_models()

    if len(sys.argv) > 1:
        what_to_index = sys.argv[1]
        print(f"Indexing {what_to_index} documents in Qdrant.")
    else:
        print("No command-line arguments were provided.")
        print("Please provide the name of the dataset to index.")
        print("Example: python es_to_qdrant.py 3_gemeentes")
        sys.exit(1)

    if what_to_index not in ["3_gemeentes", "overijssel", "all"]:
        print("Invalid argument. Please provide one of the following:")
        print("3_gemeentes, overijssel, all")
        sys.exit(1)

    try:
        main(what_to_index)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Terminating processes...")
        sys.exit(0)
