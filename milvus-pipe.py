"""
title: milvus rag pipeline
author: forcewake
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using thepymilvus.
requirements: pymilvus, requests, sentence-transformers>=2.2.0
"""

from typing import List, Union, Generator, Iterator
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import requests

class Pipeline:
    class Valves(BaseModel):
        MILVUS_HOST: str
        MILVUS_PORT: str
        MILVUS_COLLECTIONS: List[str]
        OLLAMA_URL: str
        OLLAMA_NAME_MODEL: str 

    def __init__(self):
        self.name = "Milvus Pipeline"
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Initialize
        self.valves = self.Valves(
            **{
                "pipelines": ["*"],                                                           # Connect to all pipelines
                "MILVUS_HOST": os.getenv("MILVUS_HOST", "host.docker.internal"),                          
                "MILVUS_PORT": os.getenv("MILVUS_PORT", 19530),                                  
                "MILVUS_COLLECTIONS": os.getenv("MILVUS_COLLECTIONS", ["docs"]),                                
                "OLLAMA_URL": os.getenv("OLLAMA_URL", "http://host.docker.internal:11434/api/generate"), # Make sure to update with the URL of your Ollama host, such as http://localhost:11434 or remote server address
                "OLLAMA_NAME_MODEL": os.getenv("OLLAMA_NAME_MODEL", "llama3.1:latest")        # Model to use for text-to-SQL generation      
            }
        )

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    def connect_to_milvus(self):
        connections.connect(host=self.valves.MILVUS_HOST, port=self.valves.MILVUS_PORT)
        print("Connected to Milvus")

    def retrieve_from_milvus(self, collection_name, query, top_k=3):
        collection = Collection(collection_name)
        collection.load()

        query_embedding = self.model.encode([query])[0].tolist()

        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 10},
        }
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["text"]
        )

        retrieved_docs = [hit.entity.get('text') for hit in results[0]]
        return retrieved_docs

    def format_context(self, retrieved_docs):
        return "\n\n".join(retrieved_docs)

    def query_ollama(self, prompt, context):
        data = {
            "model": self.valves.OLLAMA_NAME_MODEL,
            "prompt": f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:",
            "stream": False
        }
        response = requests.post(self.valves.OLLAMA_URL, json=data)
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error: {response.status_code} - {response.text}"

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")

        if "user" in body:
            print(f'User: {body["user"]["name"]} ({body["user"]["id"]})')
            print(f"Message: {user_message}")

        # Connect to Milvus
        self.connect_to_milvus()

        # Retrieve relevant documents from both Milvus collections
        all_retrieved_docs = []
        for collection_name in self.valves.MILVUS_COLLECTIONS:
            retrieved_docs = self.retrieve_from_milvus(collection_name=collection_name, query=user_message)
            all_retrieved_docs.extend(retrieved_docs)

        # Format context for Ollama
        context = self.format_context(all_retrieved_docs)

        # Query Ollama for the final answer
        answer = self.query_ollama(user_message, context)

        # Return answer
        return answer
