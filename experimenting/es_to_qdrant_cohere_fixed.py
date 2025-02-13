import sys
import os
import io
import warnings
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
from elasticsearch.exceptions import NotFoundError
import pandas as pd
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor  # Use ProcessPoolExecutor for multiprocessing
from unstructured.partition.html import partition_html
from unstructured.partition.text import partition_text
from unstructured.cleaners.core import (
    clean,
    clean_non_ascii_chars,
    replace_unicode_quotes,
    group_broken_paragraphs
)
from lxml import etree
import time
from typing import List, Tuple

import contextlib  # For suppressing stdout and stderr
import logging  # For adjusting logging levels
from joblib import Parallel, delayed
from math import ceil
import warnings
import cohere
import httpx
from bs4 import BeautifulSoup
import re

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    NamedSparseVector,
    NamedVector,
    SparseVector,  # Make sure this is imported
    PointStruct,
    SearchRequest,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
    ScoredPoint,
    MultiVectorConfig,
    MultiVectorComparator,
    Datatype,
    HnswConfigDiff
)
import concurrent
from fastembed.late_interaction import LateInteractionTextEmbedding
from fastembed.sparse import SparseTextEmbedding, SparseEmbedding
from fastembed.text import TextEmbedding
import onnxruntime as ort
import uuid

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)

# os.environ["COHERE_API_KEY"] = "RU9eGeOrKo0jD2Z6kAqOJAw2RpOmF4jGgO9ZAGQT" # Linksmith Trial API Key
# os.environ["COHERE_API_KEY"] = "leBUANLdJzox27RHfrolRkiCzWIEMmyeBTeTKmsE" # Linksmith Production API Key
# os.environ["COHERE_API_KEY"] = "u0uqZMNOUfnrgZbNm0Y3IaraJu0uhzR7JKVnooF5" # Open State Trial API Key
os.environ["COHERE_API_KEY"] = "mpEie8xjPjKIhHz7wCbwxzxWaMEkojhc6ZhO8U82" # Open State Production API Key


# Define the path to your models directory
models_dir = os.path.join(os.path.dirname(__file__), 'models')

# Constants
BATCH_SIZE = 500  # Increased due to high memory availability
SLEEP_TIME = 0     # Set sleep time to zero since GPUs are underutilized
DENSE_BATCH_SIZE = 96  # Larger batches for embedding
SPARSE_BATCH_SIZE = 1000
NUM_WORKERS = max_workers = 12  # Leave 4 cores for system processes

# Unstructured Chunking Parameters
COMBINE_TEXT_UNDER_N_CHARS=800
MAX_CHARACTERS=1000
NEW_AFTER_N_CHARS=1000
MAX_PARTITION=1000
OVERLAP=75

HUMAN_READABLE_SOURCES = {
    "openbesluitvorming": "Raadstuk of bijlage",
    "poliflw": "Politiek nieuwsbericht",
    "openspending": "Begrotingsdata",
    "woogle": "Woo-verzoek",
    "obk": "Officiële bekendmaking",
    "cvdr": "Lokale wet- en regelgeving",
    "oor": "Rapport",
}

# Initialize NVML for GPU temperature monitoring (optional, can be removed if not needed)
# pynvml.nvmlInit()
from functools import lru_cache
from threading import Lock

# Replace the global variable with a singleton pattern
class CohereClientManager:
    _instance = None
    _lock = Lock()
    
    @classmethod
    def get_client(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:  # Double-check pattern
                    cls._instance = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
        return cls._instance

class QdrantClientManager:
    _instance = None
    _lock = Lock()
    
    @classmethod
    def get_client(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = QdrantClient(host="localhost", port=6333)
        return cls._instance
    
class SparseEmbedderManager:
    _instance = None
    _lock = Lock()
    
    @classmethod
    def get_embedder(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    session_options = ort.SessionOptions()
                    session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
                    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    session_options.intra_op_num_threads = 14
                    
                    cls._instance = SparseTextEmbedding(
                        model_name="Qdrant/bm25",
                        cache_dir=models_dir,
                        session_options=session_options,
                        providers=["CUDAExecutionProvider"],
                        threads=14
                    )
        return cls._instance
    
class ElasticsearchManager:
    _instance = None
    _lock = Lock()
    
    @classmethod
    def get_client(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = Elasticsearch(
                        "http://localhost:9200",
                        max_retries=3,
                        retry_on_timeout=True,
                        request_timeout=30
                    )
        return cls._instance   
    
def is_empty(text):    
    if len(text) == 0:
        logger.debug(f"is_empty: {text}")
        return True
    
    return False

# Pattern to match "Pagina X van Y" where X and Y are numbers
pagina_pattern_1 = r'Pagina \d+ van \d+'


def get_human_readable_source(source: str) -> str: 
    return HUMAN_READABLE_SOURCES.get(source, source)

def clean_page_numbers(text):
    return re.sub(pagina_pattern_1, '', text)
 
def has_page_numbers(text):    
    return bool(re.search(pagina_pattern_1, text))
 
def clean_extra_whitespace(text):
    return " ".join(text.split())

def html_table_to_markdown(html_table):
    """Convert HTML table to markdown format.
    
    Args:
        html_table (str): HTML string containing a table
        
    Returns:
        str: Markdown formatted table
    """
    # Parse HTML
    soup = BeautifulSoup(html_table, 'html.parser')
    table = soup.find('table')
    
    if not table:
        return ""
    
    markdown_lines = []
    
    # Process header row if exists
    headers = []
    header_row = table.find('thead')
    if header_row:
        headers = [cell.get_text(strip=True) for cell in header_row.find_all(['th', 'td'])]
    else:
        # Try to get headers from first row
        first_row = table.find('tr')
        if first_row:
            headers = [cell.get_text(strip=True) for cell in first_row.find_all(['th', 'td'])]
    
    if headers:
        # Add header row
        markdown_lines.append('| ' + ' | '.join(headers) + ' |')
        # Add separator row
        markdown_lines.append('|' + '|'.join(['---' for _ in headers]) + '|')
    
    # Process data rows
    rows = table.find_all('tr')
    start_index = 1 if headers and not header_row else 0  # Skip first row if used as header
    
    for row in rows[start_index:]:
        cells = row.find_all(['td', 'th'])
        # Clean and format cell text
        cell_text = []
        for cell in cells:
            text = cell.get_text(strip=True)
            # Replace newlines with spaces
            text = re.sub(r'\s+', ' ', text)
            cell_text.append(text)
        
        if any(cell_text):  # Only add row if it contains any text
            markdown_lines.append('| ' + ' | '.join(cell_text) + ' |')
    
    return '\n'.join(markdown_lines)

def element_to_markdown(element):
    if has_page_numbers(element.text):
        logger.debug(f"has_page_numbers: {element.text}")
        return ""
    
    text = ""
    if element.category == 'Title':
        if element.metadata.category_depth == 0:
            text = f"## {element.text}\n\n"
        elif element.metadata.category_depth == 1:
            text = f"### {element.text}\n\n"
        elif element.metadata.category_depth == 2:
            text = f"#### {element.text}\n\n"
        elif element.metadata.category_depth == 3:
            text = f"##### {element.text}\n\n"
        elif element.metadata.category_depth == 4:
            text = f"###### {element.text}\n\n"
        else:
            text = f"## {element.text}\n\n"
    elif element.category == 'Header':
        text = f"## {element.text}\n\n"
    elif element.category == 'SubHeader':
        text = f"### {element.text}\n\n"
    elif element.category == 'ListItem':
        text = f"- {element.text}\n"
    elif element.category == 'Paragraph':
        text = f"{element.text}\n\n"
    elif element.category == 'NarrativeText':
        text = f"{element.text}\n\n"        
    elif element.category == 'CompositeElement':   
        text = f"{element.text}\n\n" 
    elif element.category == 'UncategorizedText':  
        text = f"{element.text}\n\n" 
    elif element.category == 'Table':
        text = f"{element.text}\n\n" 
        # text = html_table_to_markdown(element.metadata.text_as_html)
    else:
        print(f"Other:{element.category} - {element.text}")
        text = f"{element.text}\n" 
        
    logging.debug(f"category:{element.category} text:{text}")    

    return text

def elements_to_markdown(elements):
    # Convert all elements to Markdown
    return "".join([element_to_markdown(el) for el in elements])

def custom_clean(text):
    text = clean_non_ascii_chars(text)
    text = text.replace("•\n", "")
    text = text.replace("- \n", "- ")
    text = replace_unicode_quotes(text)
    text = text.rstrip("\n")
    return text


def html_partition(doc):
    markdown_chunks = []    
     
    narative_text_elements = partition_html(
        text=doc,
        chunking_strategy='by_title',
        paragraph_grouper=group_broken_paragraphs,
        combine_text_under_n_chars=COMBINE_TEXT_UNDER_N_CHARS,
        max_characters=MAX_CHARACTERS,
        max_partition=MAX_PARTITION,
        overlap=OVERLAP
    )
    
    for composite_element in narative_text_elements:
        markdown = elements_to_markdown(composite_element.metadata.orig_elements)
        cleaned_markdown = custom_clean(markdown)
        
        if not is_empty(cleaned_markdown):
            markdown_chunks.append(cleaned_markdown)
        
    return markdown_chunks


def txt_partition(doc):
    markdown_chunks = []

    narative_text_elements = partition_text(
        text=doc,
        chunking_strategy='by_title',
        paragraph_grouper=group_broken_paragraphs,
        combine_text_under_n_chars=COMBINE_TEXT_UNDER_N_CHARS,
        max_characters=MAX_CHARACTERS,
        max_partition=MAX_PARTITION,
        overlap=OVERLAP
    )
    
    for composite_element in narative_text_elements:
        markdown = elements_to_markdown(composite_element.metadata.orig_elements)
        cleaned_markdown = custom_clean(markdown)
        if not is_empty(cleaned_markdown):
            markdown_chunks.append(cleaned_markdown)
    
    return markdown_chunks  


def html_txt_partition(doc):
    markdown_chunks = []
    
    #     narative_text_elements = partition_html(
    #     text=doc,
    #     paragraph_grouper=group_broken_paragraphs,
    # )

    # for element in narative_text_elements:
    #     narative_text_elements = partition_text(
    #         text=element.text,
    # Strip all <p> and </p> tags from the document
    doc = doc.replace("<p>", "").replace("</p>", "\n\n").replace("<body>", "").replace("</body>", "").replace("<html>", "").replace("</html>", "")

    narative_text_elements = partition_text(
        text=doc,
        chunking_strategy='by_title',
        paragraph_grouper=group_broken_paragraphs,
        combine_text_under_n_chars=COMBINE_TEXT_UNDER_N_CHARS,
        max_characters=MAX_CHARACTERS,
        new_after_n_chars=NEW_AFTER_N_CHARS,
        max_partition=MAX_PARTITION,
        overlap=OVERLAP
    )
    
    for composite_element in narative_text_elements:
        markdown = elements_to_markdown(composite_element.metadata.orig_elements)
        cleaned_markdown = custom_clean(markdown)        
        if not is_empty(cleaned_markdown):
            markdown_chunks.append(cleaned_markdown)
        
    return markdown_chunks                  

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


def prepare_qdrant_payload(args):    
    doc, existing_ids = args  # Unpack the arguments
    error_count = 0  # Initialize error counter
    
    if doc['_id'] in existing_ids:    
        logger.info(f"Skipping {doc['_id']} because it already exists in Qdrant")    
        return None, error_count
    
    try:
        # Extract the 'description' field
        description = doc.get('_source', {}).get('description', '')
        
        try:
            description = remove_processing_instructions(description)
        except Exception:
            # If parsing fails, keep the original description
            error_count += 1

        source = doc.get('_source', {}).get('source', '')        
        title = doc["_source"].get("title", "")
        location_name = doc["_source"].get("location_name", "")
        doc_url = doc["_source"].get("doc_url", "")
        location = doc["_source"].get("location", "")
        modified = doc["_source"].get("modified", "")
        published = doc["_source"].get("published", "")
        type = doc["_source"].get("type", "")
        identifier = doc["_source"].get("identifier", "")
        url = doc["_source"].get("url", "")
        doc_url = doc["_source"].get("doc_url", "")
        source = doc["_source"].get("source", "")
        source_id = doc["_id"] 
        
        markdown_chunks = []
            
        try:
            with suppress_stdout_stderr():        
                if source == "cvdr" or source == "poliflow":
                    markdown_chunks = html_partition(description)
                elif source == "oor" or source == "woogle" or source == "obk":
                    markdown_chunks = txt_partition(description)
                elif source == "openbesluitvorming":
                    markdown_chunks = html_txt_partition(description)
        except Exception:
            # Error in partition_html
            error_count += 1
            return None, error_count  # Return None and error count

        qdrant_payload = []
        chunk_count = len(markdown_chunks)       

        for i, markdown_chunk in enumerate(markdown_chunks):
            try: 
                qdrant_payload.append({
                    "content": markdown_chunk,
                    "meta": {
                        "title": title,
                        "location": location,
                        "location_name": location_name,
                        "modified": modified,
                        "published": published,
                        "type": type,
                        "identifier": identifier,
                        "url": url,
                        "doc_url": doc_url,
                        "source": source,
                        "source_id": source_id,
                        "page_number": i,
                        "page_count": chunk_count
                    }
                })
            except Exception:
                # Error processing element
                error_count += 1
                continue

        return (qdrant_payload if qdrant_payload else None), error_count
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

def generate_dense_embeddings_in_pool(args):
    docs = args
    try:
        return generate_dense_embeddings(docs)    
    except Exception as e:
        logging.error(f"Error in process_docs_in_pool: {type(e).__name__}: {str(e)}")
        return []  # Return an empty list if any unexpected error occurs

def generate_dense_embeddings(texts, retries=3, delay=5):    
    cohere_client = CohereClientManager.get_client()
    
    for attempt in range(retries):
        try:
            return cohere_client.embed(
                texts=texts, 
                input_type="search_document", 
                model="embed-multilingual-v3.0",
                embedding_types=["uint8"]
            ).embeddings.uint8
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed with error: {type(e).__name__}: {str(e)}")
            if attempt < retries - 1:
                time.sleep(delay)
    return None  # Return None to indicate failure after all retries

def upsert_with_progress(collection_name, points, batch_size=100):
    qdrant_client = QdrantClientManager.get_client()
    
    total_points = len(points)
    with tqdm(total=total_points, desc="Upserting points", position=1, leave=False) as progress_bar:
        for i in range(0, total_points, batch_size):
            batch = points[i:i + batch_size]
            qdrant_client.upsert(collection_name, batch)
            progress_bar.update(len(batch))
            
def embed_documents_on_gpu(embedder, texts, progress_bar, batch_size):
    embeddings = []
    total_docs = len(texts)
    for i in range(0, total_docs, batch_size):
        batch_docs = texts[i:i + batch_size]
        # 'embedder.embed(batch_docs)' returns a generator over embeddings
        embedding_gen = embedder.embed(batch_docs)
        for embedding in embedding_gen:
            embeddings.append(embedding)
            progress_bar.update(1)
    return embeddings

def generate_sparse_embedding(texts: List[str]) -> List[SparseEmbedding]:
    total_texts = len(texts)

    batch_size = SPARSE_BATCH_SIZE  # You can adjust this if needed
    
    sparse_embedder = SparseEmbedderManager.get_embedder()

    with tqdm(total=total_texts, desc="Generating sparse embeddings", position=1, leave=False) as progress_bar:
        # Run inference on both GPUs in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(embed_documents_on_gpu, sparse_embedder, texts, progress_bar, batch_size)
            embeddings = future.result()

    return embeddings

def make_qdrant_points(payloads, dense_vectors, sparse_vectors) -> List[PointStruct]:
    points = []
    
    for idx, (payload, dense_vector, sparse_vector) in enumerate(
        zip(payloads, dense_vectors, sparse_vectors)
    ):
        if isinstance(sparse_vector, SparseEmbedding):
            sparse_dict = SparseVector(
                indices=sparse_vector.indices.tolist(),
                values=[float(x) for x in sparse_vector.values.tolist()]
            )
        else:
            sparse_dict = SparseVector(
                indices=[int(x) for x in sparse_vector["indices"]],
                values=[float(x) for x in sparse_vector["values"]]
            )

        point = PointStruct(
            id=str(uuid.uuid4()),
            payload=payload,
            vector={  # Changed from vector to vectors
                "text-dense": dense_vector,
                "text-sparse": sparse_dict
            }
        )
        points.append(point)

    return points

def process_es_batch(es, query, batch_size, scroll_id=None, max_retries=3):
    """Process a batch of documents from Elasticsearch with retry logic."""
    for attempt in range(max_retries):
        try:
            if scroll_id:
                response = es.scroll(scroll_id=scroll_id, scroll='30m')
            else:
                response = es.search(
                    index="bron_2025_02_01",
                    query=query,
                    scroll='30m',
                    size=batch_size,
                    timeout='30s'
                )
            return response, response['_scroll_id']
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logging.warning(f"Attempt {attempt + 1} failed. Retrying in 5 seconds... Error: {str(e)}")
            time.sleep(5)
            
            
def main(what_to_index='3_gemeentes'):
    qdrant_client = QdrantClientManager.get_client()    
    es = ElasticsearchManager.get_client()
    
    # index_name = "jodal_documents7"
    index_name = "bron_2025_02_01"

    # Determine collection name
    collection_name = f"{what_to_index}_2025_02_01_cohere"

    # qdrant_client.create_collection(
    #     collection_name=collection_name,
    #     vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    #     sparse_vectors_config=SparseVectorParams(size=384, distance=Distance.COSINE, vector_name="bm25"),
    #     on_disk_payload=True
    # )
    collections = qdrant_client.get_collections().collections
    if collection_name not in [col.name for col in collections]:
        qdrant_client.create_collection(
            collection_name,
            vectors_config={
                "text-dense": VectorParams(
                    size=1024,
                    distance=Distance.COSINE,
                    datatype=Datatype.UINT8,
                    on_disk=True
                ),
            },
            sparse_vectors_config={
                "text-sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=True)
                )
            },
            hnsw_config=HnswConfigDiff(
                on_disk=True, 
                m=32, 
                ef_construct=100
            ),
            on_disk_payload=True
        )        
    
    
    logging.info(f"Getting elastic IDs from Qdrant collection '{collection_name}'.")
    cache_file = f"source_ids.txt"
    existing_elastic_ids = get_elastic_ids_from_qdrant(collection_name, cache_file)
    print(f"Retrieved {len(existing_elastic_ids)} existing elastic IDs from Qdrant collection '{collection_name}'. These will be skipped.")
    
    # Base query
    base_query = {
        "bool": {
            "must": [
                {"exists": {"field": "description"}}
            ],
        }
    }

    # Define your queries
    query_1_gemeente = {
        "query": {
            "bool": {
                "should": [
                    {"term": {"_id": "d2ed4a79c4c22be7931b36ef8fe15af935bdccae"}},
                    {"term": {"_id": "6a230233acab98ee05123e8246d37c2d28ddc338"}},
                    {"term": {"_id": "451e2f5866fe5c80d12875ecc95b3107e6c4a0b8"}},
                ],
                "minimum_should_match": 1
            }
        }
    }

    # query_1_gemeente = {
    #     "query": {
    #         "bool": {
    #             "must": [base_query],
    #             "should": [
    #                 {"match_phrase": {"location": "GM0383"}},
    #             ],
    #             "minimum_should_match": 1
    #         }
    #     }
    # }

    query_3_gemeentes = {
        "query": {
            "bool": {
                "must": [base_query],
                "should": [
                    {"match_phrase": {"location": "GM0141"}},
                    {"match_phrase": {"location": "GM1896"}},
                    {"match_phrase": {"location": "GM0180"}}
                ],
                "minimum_should_match": 1
            }
        }
    }

    query_overijssel = {
        "query": {
            "bool": {
                "must": [base_query],
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
                ],
                "minimum_should_match": 1
            }
        }
    }

    query_all = {
        "query": base_query
    }

    # Determine which query to use    
    if what_to_index == '1_gemeente':
        query = query_1_gemeente
    if what_to_index == '3_gemeentes':
        query = query_3_gemeentes
    elif what_to_index == 'overijssel':
        query = query_overijssel
    elif what_to_index == 'nederland':
        query = query_all
    
    # Get total document count for progress bar
    total_docs = es.count(index=index_name, body=query)['count']

    total_docs_to_process = total_docs - len(existing_elastic_ids)
    logging.info(f"Total documents to process: {total_docs_to_process}.")
    
    batch_size = BATCH_SIZE
    total_points_processed = 0
    total_errors = 0  # Initialize total error counter

    # Start time for the entire process
    start_time = time.time()
    total_batches = (total_docs_to_process + batch_size - 1) // batch_size

    print(f"Processing {total_batches} batches of each {batch_size}")
    try:
        with tqdm(total=total_docs, desc="Total progress", position=0, dynamic_ncols=True) as pbar:
            while True:
                try:
                    # Use scan to retrieve documents
                    es_scan = scan(client=es, index=index_name, query=query, scroll='30m', size=BATCH_SIZE)
                    document_batch = batch_iterator(es_scan, batch_size)

                    for batch_num, es_docs in enumerate(document_batch):
                        try:
                            # Process the documents in the batch
                            # Process documents in parallel using multiprocessing
                            with multiprocessing.Pool(processes=max_workers) as pool:
                                # Create a list of tuples containing both arguments
                                pool_args = [(doc, existing_elastic_ids) for doc in es_docs]
                                qdrant_payload_list = pool.map(prepare_qdrant_payload, pool_args)
                            
                            # Combine results, filtering out None values
                            qdrant_payloads = []
                            for doc, _ in qdrant_payload_list:
                                if doc:
                                    qdrant_payloads.extend(doc)

                            if qdrant_payloads:
                                texts_to_embed = []
                                for doc in qdrant_payloads:
                                    try:
                                        source = get_human_readable_source(doc['meta'].get('source', ''))
                                        location = doc['meta'].get('location_name', '')
                                        title = doc['meta'].get('title', '')
                                        content = doc.get('content', '')
                                        
                                        text = ""
                                        if source and source != '':
                                            text += f"{source} van "
                                        if location and location != '':
                                            text += f"{location} \n"
                                        if title and title != '':
                                            text += f"{title} \n"
                                        if content and content != '':
                                            text += f"{content}"
                                            
                                        texts_to_embed.append(text)
                                    except Exception as e:
                                        logger.error(f"Error creating text to embed: {str(e)}")
                                        continue
                                
                                # Process dense embeddings
                                with multiprocessing.Pool(processes=max_workers) as pool:
                                    pool_args = [(texts_to_embed[i:i+DENSE_BATCH_SIZE]) 
                                                 for i in range(0, len(texts_to_embed), DENSE_BATCH_SIZE)]
                                    dense_embeddings_list = pool.map(generate_dense_embeddings_in_pool, pool_args)
                                                                  
                                dense_embeddings = [doc for sublist in dense_embeddings_list for doc in sublist if sublist]    
                                
                                # Process sparse embeddings
                                sparse_embeddings = generate_sparse_embedding(texts_to_embed)  
                                points = make_qdrant_points(qdrant_payloads, dense_embeddings, sparse_embeddings)                                
                                upsert_with_progress(collection_name, points)
                                
                            # Update progress bar
                            pbar.update(len(es_docs))

                        except Exception as e:
                            logging.error(f"Error processing batch {batch_num}: {type(e).__name__}: {str(e)}")
                            continue  # Move to the next batch

                    # If we've processed all batches without error, break the loop
                    break

                except NotFoundError:
                    print("Search context expired. Restarting from the beginning.")

    except ImportError:
        print("An ImportError occurred. This is likely due to Python shutting down.")
    except Exception as e:
        print(f"An unexpected error occurred: {type(e).__name__}: {str(e)}")
    finally:
        # Perform any necessary cleanup
        es.clear_scroll(scroll_id='_all')  # Clear all scroll contexts

    # After all batches are processed, print total errors and total time
    total_elapsed_time = time.time() - start_time
    print(f"\nProcessing completed with {total_errors} errors encountered.")
    print(f"Total processing time: {format_time(total_elapsed_time)}")

def cleanup_multiprocessing():
    for p in multiprocessing.active_children():
        p.terminate()
    # Remove the following line:
    # multiprocessing.current_process()._cleanup()

def get_elastic_ids_from_qdrant(collection_name, cache_file):          
    if os.path.exists(cache_file):
        print(f"Loading cached elastic IDs from {cache_file}")
        with open(cache_file, 'r') as f:
            return set(line.strip() for line in f)
    
    qdrant_client = QdrantClientManager.get_client()
    
    if not qdrant_client.collection_exists(collection_name=collection_name):
        print(f"Collection '{collection_name}' does not exist.")
        return []
    
    doc_count = qdrant_client.count(collection_name).count
    print(f"Total documents in collection '{collection_name}': {doc_count}.")

    document_ids = []
    offset = None
    limit = 10000  # Adjust this based on your system's capabilities
    
    pbar = tqdm(desc="Retrieving points", unit="pts", dynamic_ncols=True, total=doc_count)

    while True:
        scroll_result = qdrant_client.scroll(
            collection_name=collection_name,
            limit=limit,
            offset=offset,
            with_payload=['meta.source_id'],
            with_vectors=False
        )
        
        batch, next_offset = scroll_result
        
        if not batch:
            break

        document_ids.extend(doc.payload['meta']['source_id'] for doc in batch)
        
        pbar.update(len(batch)) 
        
        if next_offset is None:
            break
        
        offset = next_offset
    
    # Remove duplicates
    document_ids = list(set(document_ids))

    print(f"\nRetrieved {len(document_ids)} unique document IDs, and wrote to {cache_file}.")
    
    if document_ids:
        with open(cache_file, 'a') as f:
            for doc_id in document_ids:
                f.write(f"{doc_id}\n")
                
    return document_ids
    
if __name__ == "__main__":
    import multiprocessing
    import argparse
    multiprocessing.set_start_method('spawn')
    print("Starting the script")

    parser = argparse.ArgumentParser(description="Index documents in Qdrant.")
    parser.add_argument("--what_to_index", choices=["1_gemeente", "3_gemeentes", "overijssel", "nederland"],
                        help="Name of the dataset to index")
    args = parser.parse_args()

    what_to_index = args.what_to_index

    print(f"Indexing {what_to_index} documents in Qdrant.")
    
    try:
        main(what_to_index)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Terminating processes...")
        sys.exit(0)
    finally:
        cleanup_multiprocessing()
