# vector_store/qdrant.py

from typing import List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from fastembed.text import TextEmbedding
from fastembed.sparse import SparseTextEmbedding
import onnxruntime as ort

# Constants
DENSE_BATCH_SIZE = 512
NER_BATCH_SIZE = 512
NUM_WORKERS = 16

# Global variables for embedder instances
dense_document_embedder_0 = None
dense_document_embedder_1 = None
sparse_document_embedder_0 = None
sparse_document_embedder_1 = None

def initialize_models():
    global dense_document_embedder_0, dense_document_embedder_1
    global sparse_document_embedder_0, sparse_document_embedder_1

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
        session_options=session_options_0,
        providers=[("CUDAExecutionProvider", {"device_id": 0})]
    )

    sparse_document_embedder_1 = SparseTextEmbedding(
        model_name="Qdrant/bm25",
        session_options=session_options_1,
        providers=[("CUDAExecutionProvider", {"device_id": 1})]
    )

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
        with ThreadPoolExecutor() as executor:
            future_0 = executor.submit(embed_documents_on_gpu, dense_document_embedder_0, docs_0, progress_bar, batch_size_per_gpu, 0, 2)
            future_1 = executor.submit(embed_documents_on_gpu, dense_document_embedder_1, docs_1, progress_bar, batch_size_per_gpu, 1, 2)

            embeddings_0 = future_0.result()
            embeddings_1 = future_1.result()

    # Combine results
    embeddings = embeddings_0 + embeddings_1
    return embeddings

def make_sparse_embedding(texts: List[str]) -> List:
    total_docs = len(texts)

    # Split documents between the two GPUs
    mid_point = total_docs // 2
    docs_0 = texts[:mid_point]
    docs_1 = texts[mid_point:]

    batch_size_per_gpu = 512  # You can adjust this if needed

    with tqdm(total=total_docs, desc="Generating sparse embeddings", position=1, leave=False) as progress_bar:
        # Run inference on both GPUs in parallel
        with ThreadPoolExecutor() as executor:
            future_0 = executor.submit(embed_documents_on_gpu, sparse_document_embedder_0, docs_0, progress_bar, batch_size_per_gpu, 0, 2)
            future_1 = executor.submit(embed_documents_on_gpu, sparse_document_embedder_1, docs_1, progress_bar, batch_size_per_gpu, 1, 2)

            embeddings_0 = future_0.result()
            embeddings_1 = future_1.result()

    embeddings = embeddings_0 + embeddings_1
    return embeddings
