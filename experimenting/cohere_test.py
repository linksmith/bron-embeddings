#!/usr/bin/env python

import sys
import os
import re
from pprint import pprint
import json
import requests
from urllib.parse import urlencode, quote
import cohere
import qdrant_client
from qdrant_client import QdrantClient
from fastembed.sparse import SparseTextEmbedding, SparseEmbedding
from fastembed.text import TextEmbedding

def get_bron_documents_from_elasticsearch(query):
    params = {
        'query': query,
        #'filter': 'source:openbesluitvorming,woo,poliflw',
        'filter': 'source:openbesluitvorming,woo,poliflw',
        'excludes': '',
        'limit': 350,
        'default_operator': 'and'
    }
    query_string = urlencode(params)
    url = 'https://api.bron.live/documents/search?' + query_string
    resp = requests.get(url)
    resp.raise_for_status()

    results = []
    for i in resp.json()['hits']['hits']:
            results += [{
                'title': i['_source']['title'],
                'snippet': d
            } for d in i['highlight'].get('description', [])]
    # results = [
    #     {'title': i['_source']['title'], 'snippet': "\n".join(
    #         i['highlight'].get('description') or
    #         i['highlight'].get('title', ''))
    #     } for i in resp.json()['hits']['hits']
    # ]
    return results


models_dir = os.path.join(os.path.dirname(__file__), 'models')

sparse_document_embedder_0 = SparseTextEmbedding(
    cache_dir=models_dir,
    model_name="Qdrant/bm25",
    providers=[("CUDAExecutionProvider", {"device_id": 0})]
)

def generate_sparse_vector(query):
    return sparse_document_embedder_0.embed(query)
    
def get_bron_documents_from_qdrant(cohere_client, query, limit=50):   
    global qdrant_client
    try:
        qdrant_client = QdrantClient(host="localhost", port=6333)
        collection_name="1_gemeente_remote"

        # Generate dense embedding
        dense_vector = cohere_client.embed(
            texts=[query], 
            input_type="search_query", 
            model="embed-multilingual-light-v3.0",
            embedding_types=["float"]
        ).embeddings.float[0]
    except Exception as e:
        print(f"Error creating dense vector from query using Cohere: {e}")
        return None

    # Generate sparse embedding (you'll need to implement this)
    # sparse_vector = generate_sparse_vector(query)
        
    try:
        qdrant_documents = qdrant_client.search(
            query_vector=("text-dense", dense_vector),
            collection_name=collection_name,            
            limit=limit   
        )
        
        # print(f"Documenten gevonden in Qdrant:")
        # for i, doc in enumerate(qdrant_documents):
        #     snippet = doc.payload['text'][:50].replace('\n', ' ')
        #     print(f"Index: {i}, ID: {doc.id}, Score: {doc.score}, Snippet: {snippet}")
        
        # print("\n" + "-"*100 + "\n")
        return qdrant_documents    
    except Exception as e:
        print(f"Error retrieving documents from Qdrant: {e}")   
        return None

def main(argv):
    #qst = "Hoe ziet het bestuurlijke apparaat van NL er uit?"
    #qst = "Hoe worden vluchtelingen opgevangen de respectievelijke gemeenten?"
    #qst = "Hoe gaan gemeenten om met windmolens en andere vormen van schone energie?"
    #qst = "hoe staat het met de parken in almelo?"
    question = 'x'

    while question != 'exit':
        question = input('Stel een vraag: ')
        if question.strip() == '':
            question = "Om vluchtelingen beter op te vangen worden er in gemeenten lokale voorzieningen getroffen om vluchtelingen te helpen. Welke voorzieningen voor vluchtelingen zijn er in almelo?"
        cohere_client = cohere.ClientV2(api_key="RU9eGeOrKo0jD2Z6kAqOJAw2RpOmF4jGgO9ZAGQT")        
        # cohere_client = cohere.ClientV2(api_key="leBUANLdJzox27RHfrolRkiCzWIEMmyeBTeTKmsE") # production key

        print(f"Vraag: {question}")

        if question.strip() == 'exit':
            continue

        qdrant_documents = get_bron_documents_from_qdrant(
            cohere_client=cohere_client,
            query=question, 
            limit=100
        )

        # Add a check for None before accessing length
        if qdrant_documents is not None and len(qdrant_documents) > 0:                        
            # convert Qdrant ScoredPoint to Cohere RerankDocument
            document_texts = [document.payload['text'] for document in qdrant_documents]
            reranked_documents = cohere_client.rerank(
                query = question,
                documents = document_texts,
                top_n = 20,
                model = 'rerank-multilingual-v3.0',
                return_documents=True
            )
            
            # print(f"Documenten na reranking meegestuurd:")
            # for doc in reranked_documents.results:
            #     print(f"Index: {doc.index}, Score: {doc.relevance_score}")
            # print("\n" + "-"*100 + "\n")
        else:
            print("No documents retrieved from Qdrant or an error occurred.")
            # Handle the error case appropriately
            break
        
        # Sort the qdrant documents by relevance_score of the reranked_documents using the index
        # sorted_documents = sorted(
        #     qdrant_documents,
        #     key=lambda doc: next(
        #         (reranked_doc.relevance_score 
        #          for reranked_doc in reranked_documents.results 
        #          if reranked_doc.index == qdrant_documents.index(doc)),
        #         0  # Default score if not found in reranked_documents
        #     ),
        #     reverse=True
        # )
        qdrant_docs_reranked = [qdrant_documents[result.index] for result in reranked_documents.results]           
                
        # print(f"Top 20 documents after reranking:")
        # for i, doc in enumerate(qdrant_docs_reranked):
        #     snippet = doc.payload['text'][:50].replace('\n', ' ')
        #     print(f"Index: {i}, ID: {doc.id}, Score: {doc.score}, Snippet: {snippet}")
        
        # print("\n" + "-"*100 + "\n")
                
        documents = [ 
                        { 
                            'id': f'{doc.id}',
                            'data': { 
                                'es_id': doc.payload['es_id'],
                                'url': doc.payload['url'],
                                'title': doc.payload['title'],
                                'snippet': doc.payload['text'] 
                            } 
                        } for doc in qdrant_docs_reranked 
                    ]
        
        # print(documents[0])
        
        chat_response = cohere_client.chat_stream(
            model="command-r-plus",
            messages=[{
                "role": "user",
                "content": question
            }],
            documents=documents
        )
                    
        for event in chat_response:
            # if event has property type
            if hasattr(event, 'type'):
                if event.type == "content-delta":
                    print(event.delta.message.content.text, end='')
                    
                elif event.type == "citation-start":            
                    # print(f"\n{event.delta.message.citations}")
                    citation = event.delta.message.citations
                    text = citation.text.replace('\n', ' ')
                    print(f"\n[{citation.start}-{citation.end}] {text}")
                    for source in citation.sources:
                        snippet = source.document['snippet'][:100].replace('\n', ' ')
                        print(f"ID: {source.id}, ES ID: {source.document['es_id']}, Snippet: {snippet}...")  # Print first 100 characters of snippet
                    
        print(f"\n{'-'*100}\n")
        return 0

if __name__ == '__main__':
    sys.exit(main(sys.argv))
