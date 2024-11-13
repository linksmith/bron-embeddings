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

from haystack_integrations.components.embedders.fastembed import FastembedSparseDocumentEmbedder
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack.components.preprocessors import DocumentSplitter
from haystack_integrations.components.embedders.cohere import CohereDocumentEmbedder
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import NLTKDocumentSplitter
from haystack.utils import Secret
from qdrant_client import QdrantClient
import contextlib  # For suppressing stdout and stderr
import logging  # For adjusting logging levels
from joblib import Parallel, delayed
from math import ceil
import warnings
import cohere
import httpx
from bs4 import BeautifulSoup
import re

# os.environ["COHERE_API_KEY"] = "RU9eGeOrKo0jD2Z6kAqOJAw2RpOmF4jGgO9ZAGQT" # Linksmith Trial API Key
# os.environ["COHERE_API_KEY"] = "leBUANLdJzox27RHfrolRkiCzWIEMmyeBTeTKmsE" # Linksmith Production API Key
# os.environ["COHERE_API_KEY"] = "u0uqZMNOUfnrgZbNm0Y3IaraJu0uhzR7JKVnooF5" # Open State Trial API Key
os.environ["COHERE_API_KEY"] = "mpEie8xjPjKIhHz7wCbwxzxWaMEkojhc6ZhO8U82" # Open State Production API Key


# Define the path to your models directory
models_dir = os.path.join(os.path.dirname(__file__), 'models')

# # Suppress specific warnings and deprecation messages
# warnings.filterwarnings("ignore", category=DeprecationWarning, module="haystack")
# warnings.filterwarnings("ignore", category=DeprecationWarning, message="PipelineMaxLoops is deprecated and will be removed in version '2.7.0'; use PipelineMaxComponentRuns instead.")
# warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.utils.generic")
# warnings.filterwarnings("ignore", category=DeprecationWarning, module="haystack.core.errors")

# # Suppress the specific SageMaker log message
# class SageMakerFilter(logging.Filter):
#     def filter(self, record):
#         return not (record.name == 'sagemaker.config' and 'Not applying SDK defaults' in record.msg)

# # Redirect stderr to capture and filter SageMaker messages
# class FilteredStderr(io.IOBase):
#     def __init__(self, stderr):
#         self.stderr = stderr

#     def write(self, message):
#         if 'sagemaker.config' not in message or 'Not applying SDK defaults' not in message:
#             self.stderr.write(message)

#     def flush(self):
#         self.stderr.flush()

# # Apply the filter to the SageMaker logger
# sagemaker_logger = logging.getLogger('sagemaker.config')
# sagemaker_logger.addFilter(SageMakerFilter())

# # Redirect stderr to use the filtered stderr
# sys.stderr = FilteredStderr(sys.stderr)

# # Set logging levels for specific modules
# logging.getLogger('unstructured').setLevel(logging.ERROR)
# logging.getLogger('lxml').setLevel(logging.ERROR)
# logging.getLogger('sagemaker.config').setLevel(logging.ERROR)

# # Disable all warnings
# warnings.filterwarnings("ignore")

# # Disable all logging
# logging.disable(logging.CRITICAL)
# logging.basicConfig(level=logging.ERROR)
# # Set environment variable to ignore deprecation warnings
# os.environ["PYTHONWARNINGS"] = "ignore"


def suppress_specific_haystack_warning():
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        message=r"PipelineMaxLoops is deprecated and will be removed in version '2.7.0'; use PipelineMaxComponentRuns instead.",
        module="haystack.core.errors"
    )

def patch_logging_config():
    # Set logging level for sagemaker and haystack to suppress INFO messages and deprecation warnings
    for module_name in ["sagemaker.config", "haystack"]:
        logger = logging.getLogger(module_name)
        logger.setLevel(logging.ERROR)  # Set to ERROR to suppress INFO and WARNING messages
        logger.propagate = False

def suppress_sagemaker_warning():    
    sagemaker_logger = logging.getLogger('sagemaker.config')
    sagemaker_logger.setLevel(logging.ERROR)  # Suppress INFO messages

    # Disable propagation to prevent the message from appearing in parent loggers
    sagemaker_logger.propagate = False

def filtered_stderr():    
    class FilteredStderr(io.TextIOBase):
        def __init__(self, original_stderr):
            self.original_stderr = original_stderr

        def write(self, message):
            if "PipelineMaxLoops is deprecated" not in message:
                self.original_stderr.write(message)

        def flush(self):
            self.original_stderr.flush()

    sys.stderr = FilteredStderr(sys.stderr)

def suppress_warnings_and_logging():
    suppress_specific_haystack_warning()
    suppress_sagemaker_warning()
    patch_logging_config()
    filtered_stderr()    
    
    # Suppress warnings globally
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppresses TensorFlow logs

    # Set logging level for specific modules to ERROR or higher
    for module_name in ["unstructured", "lxml", "sagemaker.config"]:
        logging.getLogger(module_name).setLevel(logging.ERROR)

    # Optionally, disable all logging
    logging.disable(logging.WARNING)

# Constants
BATCH_SIZE = 5000  # Increased due to high memory availability
SLEEP_TIME = 0     # Set sleep time to zero since GPUs are underutilized
DENSE_BATCH_SIZE = 1024  # Larger batches for embedding
NER_BATCH_SIZE = 512    # Increased batch size for NER per GPU
NUM_WORKERS = max_workers = 14  # Leave 2 cores for system processes
COMBINE_TEXT_UNDER_N_CHARS=800
MAX_CHARACTERS=1000
NEW_AFTER_N_CHARS=1000
MAX_PARTITION=1000
OVERLAP=100

# Initialize NVML for GPU temperature monitoring (optional, can be removed if not needed)
# pynvml.nvmlInit()


def is_empty(text):    
    if len(text) == 0:
      return True
    
    return False
# Pattern to match "Pagina X van Y" where X and Y are numbers
pagina_pattern_1 = r'Pagina \d+ van \d+'

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
        text = html_table_to_markdown(element.metadata.text_as_html)
    else:
        print(f"Other:{element.category} - {element.text}")
        text = f"{element.text}\n" 

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


def txt_partition(doc):
    markdown_chunks = []
    
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


def html_txt_partition(doc):
    markdown_chunks = []
    
    narative_text_elements = partition_html(
        text=doc,
        paragraph_grouper=group_broken_paragraphs,
    )
    
    for element in narative_text_elements:
        narative_text_elements = partition_text(
            text=element.text,
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

        docs = []
        chunk_count = len(markdown_chunks)       

        for i, markdown_chunk in enumerate(markdown_chunks):
            try:                  
                document = Document(
                    content=markdown_chunk,
                    meta={
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
                )                    

                docs.append(document)
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

def process_docs_in_pool(args):
    embedder, docs = args
    try:
        return add_embeddings_to_documents(embedder, docs)
    except Exception as e:
        logging.error(f"Error in process_docs_in_pool: {type(e).__name__}: {str(e)}")
        return []  # Return an empty list if any unexpected error occurs

def add_embeddings_to_documents(embedder, documents, retries=3, delay=5):
    for attempt in range(retries):
        try:
            return embedder.run(documents=documents)["documents"]
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed with error: {type(e).__name__}: {str(e)}")
            if attempt < retries - 1:
                time.sleep(delay)
    return None  # Return None to indicate failure after all retries
    
def write_documents_to_qdrant(document_store, documents):  
    batch_size = 1000
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        document_store.write_documents(batch)

def main(what_to_index='3_gemeentes'):
    # Connect to Elasticsearch
    es = Elasticsearch("http://localhost:9200")
    # index_name = "jodal_documents7"
    index_name = "bron_2024_10_11"

    # Determine collection name
    collection_name = f"{what_to_index}_cohere"

    document_store = QdrantDocumentStore(
        host="localhost",
        port=6333,
        index=collection_name,
        embedding_dim=384,
        use_sparse_embeddings=True    
    )    
    
    logging.info(f"Getting elastic IDs from Qdrant collection '{collection_name}'.")
    elastic_ids = get_elastic_ids_from_qdrant(collection_name)
    logging.info(f"Retrieved {len(elastic_ids)} elastic IDs from Qdrant collection '{collection_name}'.")
    
    # Base query
    base_query = {
        "bool": {
            "must": [
                {"exists": {"field": "description"}}
            ],
            "must_not": [
                {"ids": {"values": elastic_ids}},
                {"match": {"source": "openspending"}}
            ]
        }
    }

    # Define your queries
    query_1_gemeente = {
        "query": {
            "bool": {
                "must": [base_query],
                "should": [
                    {"match_phrase": {"location": "GM0141"}},
                ],
                "minimum_should_match": 1
            }
        }
    }

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
    logging.info(f"Total documents to process: {total_docs}.")
    
    batch_size = BATCH_SIZE
    total_points_processed = 0
    total_errors = 0  # Initialize total error counter

    # Start time for the entire process
    start_time = time.time()
    total_batches = (total_docs + batch_size - 1) // batch_size

    print(f"Processing {total_batches} batches of each {batch_size}")
    try:
        with tqdm(total=total_docs, desc=f"Total progress", position=0, dynamic_ncols=True) as pbar:
            while True:
                try:
                    # Use scan to retrieve documents
                    es_scan = scan(client=es, index=index_name, query=query, scroll='15m', size=BATCH_SIZE)
                    document_batch = batch_iterator(es_scan, batch_size)

                    for batch_num, es_doc in enumerate(document_batch):
                        try:
                            # Process the documents in the batch
                            # Process documents in parallel using multiprocessing
                            with multiprocessing.Pool(processes=max_workers) as pool:
                                processed_results = pool.map(process_document, es_doc)
                            
                            # Combine results, filtering out None values
                            processed_docs = []
                            for doc, _ in processed_results:
                                if doc:
                                    processed_docs.extend(doc)

                            if processed_docs:
                                # Process dense embeddings
                                with multiprocessing.Pool(processes=max_workers) as pool:
                                    pool_args = [(dense_document_embedder, processed_docs[i:i+DENSE_BATCH_SIZE]) 
                                                 for i in range(0, len(processed_docs), DENSE_BATCH_SIZE)]
                                    documents_w_dense_embeddings = pool.map(process_docs_in_pool, pool_args)

                                # Flatten the list of lists
                                documents_w_dense_embeddings = [doc for sublist in documents_w_dense_embeddings for doc in sublist if sublist]

                                if documents_w_dense_embeddings:
                                    # Process sparse embeddings
                                    documents_w_dense_and_sparse_embeddings = add_embeddings_to_documents(sparse_document_embedder, documents_w_dense_embeddings)
                                    if documents_w_dense_and_sparse_embeddings:
                                        write_documents_to_qdrant(document_store, documents_w_dense_and_sparse_embeddings)

                            # Update progress bar
                            pbar.update(len(es_doc))

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

# Initialize the models on different GPUs
def initialize_models():
    global sparse_document_embedder, dense_document_embedder
    
    dense_document_embedder = CohereDocumentEmbedder(
        model= "embed-multilingual-light-v3.0",
        meta_fields_to_embed=["title", "location_name"],
        use_async_client = True,
        batch_size=1024  # Add batch size parameter
    ) 
    
    sparse_document_embedder = FastembedSparseDocumentEmbedder(
        model = "Qdrant/bm25",
        meta_fields_to_embed=["title", "location_name"],
        parallel=14,  # Match NUM_WORKERS
        threads=28,   # Allow 2 threads per core
        batch_size=1024
    )   
    
    sparse_document_embedder.warm_up()

def cleanup_multiprocessing():
    for p in multiprocessing.active_children():
        p.terminate()
    # Remove the following line:
    # multiprocessing.current_process()._cleanup()

def get_elastic_ids_from_qdrant(collection_name):  
    qdrant_client = QdrantClient(host="localhost", port=6333)    
    
    if not qdrant_client.collection_exists(collection_name=collection_name):
        print(f"Collection '{collection_name}' does not exist.")
        return []
    
    doc_count = qdrant_client.count(collection_name).count
    print(f"Total documents in collection '{collection_name}': {doc_count}.")

    document_ids = []
    offset = None
    limit = 10000  # Adjust this based on your system's capabilities

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
        
        if next_offset is None:
            break
        
        offset = next_offset
    
    # Remove duplicates
    document_ids = list(set(document_ids))

    print(f"Retrieved {len(document_ids)} unique document IDs.")
    return document_ids
    
if __name__ == "__main__":
    suppress_warnings_and_logging()
    import multiprocessing
    import argparse
    multiprocessing.set_start_method('spawn')
    print("Starting the script")

    # Initialize models
    initialize_models()

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
