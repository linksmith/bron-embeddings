import cohere
import uuid
from typing import List, Dict
from unstructured.partition.html import partition_html
from unstructured.chunking.title import chunk_by_title
from multiprocessing import Pool
from qdrant_client import QdrantClient
from fastembed.sparse import SparseTextEmbedding

co = cohere.Client("u0uqZMNOUfnrgZbNm0Y3IaraJu0uhzR7JKVnooF5") # Get your API key here: https://dashboard.cohere.com/api-keys
qdrant_client = QdrantClient()

class Vectorstore:
    """
    A class representing a collection of documents indexed into a vectorstore.

    Parameters:
    raw_documents (list): A list of dictionaries representing the sources of the raw documents. Each dictionary should have 'title' and 'url' keys.
    batch_size (int): The size of each batch for processing documents.

    Attributes:
    raw_documents (list): A list of dictionaries representing the raw documents.
    docs (list): A list of dictionaries representing the chunked documents, with 'title', 'text', and 'url' keys.
    docs_embs (list): A list of the associated embeddings for the document chunks.
    docs_len (int): The number of document chunks in the collection.
    batch_size (int): The size of each batch for processing documents.

    Methods:
    load_and_chunk(): Loads the data from the sources and partitions the HTML content into chunks.
    embed(): Embeds the document chunks using the Cohere API.
    index(): Indexes the document chunks for efficient retrieval.
    retrieve(): Retrieves document chunks based on the given query.
    """

    def __init__(self, raw_documents: List[Dict[str, str]], batch_size: int = 1000):
        self.raw_documents = raw_documents
        self.docs = []
        self.docs_embs = []
        self.retrieve_top_k = 10
        self.rerank_top_k = 3
        self.batch_size = batch_size
        self.load_and_chunk()
        self.embed()
        self.index()

    def load_and_chunk(self) -> None:
        """
        Loads the text from the sources and chunks the HTML content in batches.
        """
        print("Loading documents in batches...")

        with Pool(processes=16) as p:
            for i in range(0, len(self.raw_documents), self.batch_size):
                batch = self.raw_documents[i : min(i + self.batch_size, len(self.raw_documents))]
                results = p.map(self._process_batch, batch)
                for result in results:
                    self.docs.extend(result)

    def _process_batch(self, raw_document_batch: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Processes a batch of raw documents, partitioning the HTML content into chunks.
        """
        batch_docs = []
        for raw_document in raw_document_batch:
            elements = partition_html(url=raw_document["url"])
            chunks = chunk_by_title(elements)
            for chunk in chunks:
                batch_docs.append(
                    {
                        "title": raw_document["title"],
                        "text": str(chunk),
                        "url": raw_document["url"],
                    }
                )
        return batch_docs

    def embed(self) -> None:
        """
        Embeds the document chunks using the Cohere API in batches.
        """
        print("Embedding document chunks in batches...")

        with Pool(processes=16) as p:
            for i in range(0, self.docs_len, self.batch_size):
                batch = self.docs[i : min(i + self.batch_size, self.docs_len)]
                texts = [item["text"] for item in batch]
                docs_embs_batch = co.embed(
                    texts=texts, model="embed-english-v3.0", input_type="search_document"
                ).embeddings
                self.docs_embs.extend(docs_embs_batch)

    def index(self) -> None:
        """
        Indexes the document chunks for efficient retrieval.
        """
        print("Indexing document chunks...")

        self.idx = qdrant_client.create_collection("my_collection")
        self.idx.upsert(self.docs_embs, self.docs)

        print(f"Indexing complete with {len(self.docs)} document chunks.")

    def retrieve(self, query: str) -> List[Dict[str, str]]:
        """
        Retrieves document chunks based on the given query.

        Parameters:
        query (str): The query to retrieve document chunks for.

        Returns:
        List[Dict[str, str]]: A list of dictionaries representing the retrieved document chunks, with 'title', 'text', and 'url' keys.
        """

        # Dense retrieval
        query_emb_dense = co.embed(
            texts=[query], model="embed-english-v3.0", input_type="search_query"
        ).embeddings
        
        doc_ids_dense = self.idx.knn_query(query_emb_dense, k=self.retrieve_top_k)[0][0]

        # Sparse retrieval
        query_emb_sparse = SparseTextEmbedding(text=query, model="Qdrant/bm25")
        
        doc_ids_sparse = self.idx.knn_query(query_emb_sparse, k=self.retrieve_top_k)[0][0]

        # Combine dense and sparse results
        doc_ids_combined = list(set(doc_ids_dense) | set(doc_ids_sparse))

        # Reranking
        rank_fields = ["title", "text"] # We'll use the title and text fields for reranking

        docs_to_rerank = [self.docs[doc_id] for doc_id in doc_ids_combined]
        rerank_results = co.rerank(
            query=query,
            documents=docs_to_rerank,
            top_n=self.rerank_top_k,
            model="rerank-english-v3.0",
            rank_fields=rank_fields
        )

        doc_ids_reranked = [doc_ids_combined[result.index] for result in rerank_results.results]

        docs_retrieved = []
        for doc_id in doc_ids_reranked:
            docs_retrieved.append(
                {
                    "title": self.docs[doc_id]["title"],
                    "text": self.docs[doc_id]["text"],
                    "url": self.docs[doc_id]["url"],
                }
            )

        return docs_retrieved