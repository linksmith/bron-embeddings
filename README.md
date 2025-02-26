# Bron Embeddings

This repository contains experimental code for creating and working with embeddings for Bron chat. The primary goal is to explore different embedding techniques and implementations to enhance the search and retrieval capabilities of the Bron chat system.

## Overview

This is a repository for experimentation with various embedding approaches, vector databases, and search techniques. The code here demonstrates the process of:

1. Extracting data from Elasticsearch
2. Processing and cleaning text documents
3. Generating embeddings using Cohere's API
4. Storing the embeddings in Qdrant vector database
5. Implementing hybrid search with both dense and sparse embeddings

## Main Implementation

The final, production-ready implementation can be found in:

```
experimenting/es_to_qdrant_cohere_fixed.py
```

This script handles the complete pipeline from extracting documents from Elasticsearch to generating embeddings and storing them in Qdrant. It includes optimizations for performance and reliability, such as:

- Batch processing for efficient resource usage
- Error handling and retry mechanisms
- Multiprocessing for parallel embedding generation
- Caching to avoid reprocessing documents
- Support for both dense and sparse embeddings

## Running the Script

You can run the main script from the command line with various parameters to control its behavior:

```bash
python experimenting/es_to_qdrant_cohere_fixed.py --what_to_index [DATASET] --index_name [ES_INDEX_NAME]
```

### Command-line Parameters

- `--what_to_index`: Specifies which dataset to index. Options include:
  - `1_gemeente`: Index documents from a single municipality
  - `3_gemeentes`: Index documents from three municipalities
  - `overijssel`: Index documents from all municipalities in Overijssel province
  - `nederland`: Index documents from all of the Netherlands

- `--update_processed_flags`: (Flag) Update the `is_processed` flags in the Elasticsearch index
  
- `--index_name`: Name of the Elasticsearch index to use (default: "bron_2025_02_01v3")

### Examples

Index documents from three municipalities:
```bash
python experimenting/es_to_qdrant_cohere_fixed.py --what_to_index 3_gemeentes
```

Index documents from all of the Netherlands:
```bash
python experimenting/es_to_qdrant_cohere_fixed.py --what_to_index nederland
```

Update processed flags in Elasticsearch:
```bash
python experimenting/es_to_qdrant_cohere_fixed.py --update_processed_flags
```

## Features

- Hybrid search combining dense and sparse embeddings
- Document chunking and cleaning
- Efficient processing of large document collections
- Integration with Elasticsearch and Qdrant
- Singleton pattern for resource management
- Progress tracking and logging

## Requirements

The code requires several dependencies including:
- Elasticsearch
- Qdrant
- Cohere API
- FastEmbed
- ONNX Runtime
- Various text processing libraries

## Usage

The main script can be configured through command-line arguments to specify:
- What content to index
- Cache file location
- Elasticsearch index name
- Qdrant collection name

## License

This project is intended for internal use and experimentation. 