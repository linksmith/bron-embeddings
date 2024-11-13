
from custom_haystack import Pipeline
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack_integrations.components.embedders.cohere import CohereTextEmbedder
from haystack_integrations.components.generators.cohere import CohereGenerator
from haystack_integrations.components.rankers.cohere import CohereRanker
from haystack.components.preprocessors import DocumentCleaner
from haystack_integrations.components.retrievers.qdrant import QdrantHybridRetriever
from haystack_integrations.components.embedders.fastembed import FastembedSparseTextEmbedder	
import logging
from custom_haystack import component
from custom_haystack import Document
from typing import List, Dict, Any
from pprint import pprint
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from custom_haystack.custom_components import ContentNormalizer

import os
os.environ["COHERE_API_KEY"] = "RU9eGeOrKo0jD2Z6kAqOJAw2RpOmF4jGgO9ZAGQT"

document_store = QdrantDocumentStore(
    host="localhost",
    port=6333,
    index="1_gemeente_remote",
    embedding_dim=384,
    use_sparse_embeddings=True    
)


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# cohere_api_key = "RU9eGeOrKo0jD2Z6kAqOJAw2RpOmF4jGgO9ZAGQT"
# cohere_api_key = "leBUANLdJzox27RHfrolRkiCzWIEMmyeBTeTKmsE"
import os
os.environ["COHERE_API_KEY"] = "u0uqZMNOUfnrgZbNm0Y3IaraJu0uhzR7JKVnooF5"

question = "Klimaatbeleid Almelo"

logger.info(f"Vraag: {question}")
	
template = """
You will be provided with Dutch language docunents. Create informative answer in Dutch for a given question based solely on the provided documents. Use citations to support your answer.

\nDocuments:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

\nQuestion: {{question}};
\nAnswer (include citations):
"""

rag_pipe = Pipeline("rag_pipe")

rag_pipe.add_component("dense_text_embedder", CohereTextEmbedder(model="embed-multilingual-light-v3.0"))
rag_pipe.add_component("sparse_text_embedder", FastembedSparseTextEmbedder(model="Qdrant/bm25"))
rag_pipe.add_component("hybrid_retriever", QdrantHybridRetriever(document_store=document_store, top_k=50))
rag_pipe.add_component("reranker", CohereRanker(model="rerank-multilingual-v3.0", top_k=20))
rag_pipe.add_component("prompt_builder", PromptBuilder(template=template))
rag_pipe.add_component("llm", CohereGenerator(model="command-r-plus"))
rag_pipe.add_component("content_normalizer", ContentNormalizer())

rag_pipe.connect("sparse_text_embedder.sparse_embedding", "hybrid_retriever.query_sparse_embedding")
rag_pipe.connect("dense_text_embedder.embedding", "hybrid_retriever.query_embedding")
rag_pipe.connect("hybrid_retriever.documents", "content_normalizer.documents")
rag_pipe.connect("content_normalizer.documents", "reranker.documents")
rag_pipe.connect("reranker", "prompt_builder")
rag_pipe.connect("prompt_builder", "llm")

with open("haystack/rap_pipe.yml", "w") as file:
  rag_pipe.dump(file)

result = rag_pipe.run({
    "dense_text_embedder": {"text": question},
    "sparse_text_embedder": {"text": question},
    # "hybrid_retriever": {"top_k": 50},
    "reranker": {"query": question},
    "prompt_builder": {"question": question}
})

pprint(result['llm'])