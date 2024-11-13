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
import tqdm
from transformers import BertTokenizer, BertForTokenClassification
from transformers.onnx import OnnxConfig
import torch
from transformers.onnx import export
from pathlib import Path

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
nlp = spacy.load("nl_core_news_md")

# def make_late_interaction_embedding(texts: List[str]):
#     return list(late_interaction_document_embedder.embed(texts))

# def make_sparse_embedding(texts: List[str]):
#     return list(sparse_document_embedder.embed(texts))

def embed_documents_on_gpu(embedder, documents, progress_bar):
    embeddings = []
    for doc in documents:
        embedding = embedder.embed(doc)  # Replace with actual embedding generation logic
        embeddings.append(embedding)
        progress_bar.update(1)
    return embeddings

def make_sparse_embedding(texts: List[str]):
    print("Creating sparse embeddings")

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

    # Combine results
    return embeddings_0 + embeddings_1

# Function to embed documents on a specific embedder
# def embed_documents_on_gpu(embedder, documents):
#     return list(embedder.embed(documents))

def make_dense_embedding(texts: List[str]):
    print("Creating dense embeddings")
    
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
    return embeddings_0 + embeddings_1

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
        sparse_vector = SparseVector(
            indices=sparse_vector.indices.tolist(), 
            values=sparse_vector.values.tolist()
        )
        
        # prepare the entities for the qdrant points
        entities = []
        for entity in rows[idx]["entities"].ents:
            entities.append({
                "text": entity.text,  # Use dot notation to access attributes
                "label": entity.label_,
                "start": entity.start_char,
                "end": entity.end_char,
            })

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
                "entities": entities,
                "chunk_index": rows[idx]["chunk_index"],
                "chunk_count": rows[idx]["chunk_count"],
            },  # Add any additional payload if necessary
            vector={
                "text-sparse": sparse_vector,
                # "text-late-interaction": late_interaction_vector,      
                "text-dense": dense_vector.tolist(),          
            },
        )
        points.append(point)

    return points

def run_ner_pipeline(texts: List[str]):
    print( "Running NER pipeline")
    return list(tqdm(nlp.pipe(texts, batch_size=100), total=len(texts)))

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

    df["dense_embedding"] = make_dense_embedding(combined_texts)
    df["sparse_embedding"] = make_sparse_embedding(combined_texts)
    df["entities"] = run_ner_pipeline(combined_texts)

    points = make_qdrant_points(df)

    upsert_with_progress(collection_name, points)

    return df

def run(collection_name, parquet_file, docs_to_load = None):
    df = pd.read_parquet(parquet_file, engine="pyarrow")

    if docs_to_load is not None:
        df = df[:docs_to_load]

    df = load_data(df, collection_name)

    return df

# Run the script with main function
if __name__ == "__main__":
    collection_name ="Overijssel"
    parquet_file = "collected_texts.parquet"
    df = run(collection_name, parquet_file, docs_to_load=None)
