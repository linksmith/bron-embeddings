# app.py

import sys
import multiprocessing
from index.elastic_search import es
from vector_store.qdrant import initialize_models
import logging

def setup_logging():
    # Suppress logging messages from specific libraries
    logging.getLogger('unstructured').setLevel(logging.ERROR)
    logging.getLogger('lxml').setLevel(logging.ERROR)

def parse_arguments():
    if len(sys.argv) > 1:
        what_to_index = sys.argv[1]
        print(f"Indexing {what_to_index} documents in Qdrant.")
    else:
        print("No command-line arguments were provided.")
        print("Please provide the name of the dataset to index.")
        print("Example: python app.py 3_gemeentes")
        sys.exit(1)

    if what_to_index not in ["3_gemeentes", "overijssel", "all"]:
        print("Invalid argument. Please provide one of the following:")
        print("3_gemeentes, overijssel, all")
        sys.exit(1)
    
    return what_to_index

def app():
    print("Starting the script")
    # setup_logging()
    
    # Initialize multiprocessing
    multiprocessing.set_start_method('spawn')
    
    # Initialize models
    initialize_models()
    
    # Parse command-line arguments
    what_to_index = parse_arguments()
    
    try:
        es(what_to_index)
    except KeyboardInterrupt:
        print("\nInterrupted by user. Terminating processes...")
        sys.exit(0)

if __name__ == "__main__":
    app()
