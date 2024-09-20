# ner/main.py

from typing import List
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
import spacy

def ner_pipeline(docs, gpu_id, batch_size):
    torch.cuda.empty_cache()
    torch.cuda.set_device(gpu_id)
    spacy.require_gpu()
    nlp = spacy.load("nl_core_news_lg")

    entities = []
    try:
        for doc in nlp.pipe(docs, batch_size=batch_size):
            entities.append(doc)
    finally:
        # Release GPU resources
        torch.cuda.empty_cache()
        del nlp
    return entities



def run_ner_pipeline(texts: List[str]):
    from vector_store.qdrant import NER_BATCH_SIZE  # Import batch size from vector_store or define here

    batch_size_per_gpu = NER_BATCH_SIZE
    # Split documents between the two GPUs
    mid_point = len(texts) // 2
    docs_0 = texts[:mid_point]
    docs_1 = texts[mid_point:]

    # Run inference on both GPUs in parallel using multiprocessing
    total_docs = len(texts)
    entities = []
    with tqdm(total=total_docs, desc="Running NER pipeline", position=1, leave=False) as progress_bar:
        with ProcessPoolExecutor(max_workers=2) as executor:
            future_0 = executor.submit(ner_pipeline, docs_0, 0, batch_size_per_gpu)
            future_1 = executor.submit(ner_pipeline, docs_1, 1, batch_size_per_gpu)

            # Use as_completed to update progress bar
            for future in as_completed([future_0, future_1]):
                result = future.result()
                progress_bar.update(len(result))
                entities.extend(result)

    return entities
