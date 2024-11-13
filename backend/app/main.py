import os
from typing import List, Dict
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from qdrant_client import QdrantClient
from cohere import Client as CohereClient
import asyncio
import json

# ... (previous imports)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ... (rest of the code remains the same)

# Initialize clients
qdrant_client = QdrantClient(os.getenv("QDRANT_URL"))
cohere_client = CohereClient(os.getenv("COHERE_API_KEY"))

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: bool = True

async def retrieve_relevant_documents(query: str) -> List[Dict]:
    # Implement retrieval logic using Qdrant
    # This is a placeholder - replace with actual retrieval logic
    results = qdrant_client.search(
        collection_name="your_collection",
        query_vector=cohere_client.embed(texts=[query]).embeddings[0],
        limit=5
    )
    return [{"id": hit.id, "content": hit.payload["content"]} for hit in results]

async def generate_response(messages: List[ChatMessage], relevant_docs: List[Dict]):
    # Prepare the prompt with chat history and relevant documents
    prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
    prompt += "\n\nRelevant information:\n"
    prompt += "\n".join([doc["content"] for doc in relevant_docs])
    prompt += "\n\nAssistant: "

    # Generate response using Cohere
    response = cohere_client.chat(
        message=prompt,
        model="command",
        stream=True,
        citation_quality="accurate",
        documents=relevant_docs
    )
    
    return response

@app.post("/chat")
async def chat(request: ChatRequest):
    relevant_docs = await retrieve_relevant_documents(request.messages[-1].content)
    response = await generate_response(request.messages, relevant_docs)
    
    if request.stream:
        return StreamingResponse(response, media_type="text/event-stream")
    else:
        full_response = "".join([event.text for event in response])
        return {"response": full_response}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    while True:
        try:
            data = await websocket.receive_text()
            request = ChatRequest.parse_raw(data)
            
            relevant_docs = await retrieve_relevant_documents(request.messages[-1].content)
            response = await generate_response(request.messages, relevant_docs)
            
            for event in response:
                await websocket.send_text(json.dumps({
                    "text": event.text,
                    "citations": event.citations
                }))
            
            await websocket.send_text(json.dumps({"end": True}))
        except Exception as e:
            await websocket.send_text(json.dumps({"error": str(e)}))
            break

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)