from typing import List, Union, Generator, Iterator
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
import requests

class Pipeline:
    def __init__(self):
        self.name = "Milvus Pipeline"
        self.milvus_host = 'host.docker.internal'
        self.milvus_port = '19530'
        self.ollama_url = "http://host.docker.internal:11434/api/generate"
        self.model_name = "llama3"
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.collections = ["faqs", "wikidocs"]  # Add your second collection name here

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    def connect_to_milvus(self):
        connections.connect(host=self.milvus_host, port=self.milvus_port)
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
            "model": self.model_name,
            "prompt": f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:",
            "stream": False
        }
        response = requests.post(self.ollama_url, json=data)
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
        for collection_name in self.collections:
            retrieved_docs = self.retrieve_from_milvus(collection_name=collection_name, query=user_message)
            all_retrieved_docs.extend(retrieved_docs)

        # Format context for Ollama
        context = self.format_context(all_retrieved_docs)

        # Query Ollama for the final answer
        answer = self.query_ollama(user_message, context)

        # Return answer
        return answer