import os
from mem0 import MemoryClient
from typing import List, Dict

# Use the key provided by the user
MEM0_API_KEY = "m0-dFFPdnL0iQP7DvUkRFMPTSk75TdcxH4DdyperOTF"

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

class Mem0Backend:
    def __init__(self, user_id="benchmark_user_1"):
        self.client = MemoryClient(api_key=MEM0_API_KEY)
        self.user_id = user_id
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self.openai_client = OpenAI(api_key=api_key)
        else:
            self.openai_client = None

    def ingest_documents(self, docs: List[Dict]):
        # Mem0 Platform 'add'
        messages = []
        for doc in docs:
            # Mem0 Platform expects 'messages' format usually or text?
            # Reading docs (assumed): client.add(messages=[...], user_id=...)
            # Or just text. Let's try passing text directly if supported, or list of messages.
            # Usually: client.add(messages=[{"role": "user", "content": doc["content"]}], user_id=...)
            
            # The add method typically takes: messages, user_id, etc.
            self.client.add(messages=[{"role": "user", "content": doc["content"]}], user_id=self.user_id)
        print(f"Mem0: Ingested {len(docs)} documents.")

    def search(self, query: str):
        # Mem0 search requires filters for user_id in v2
        results = self.client.search(query=query, filters={"user_id": self.user_id})
        return results

    def generate_response(self, query: str):
        # 1. Retrieve from Mem0
        results = self.search(query)
        
        # Mem0 results usually contain 'memory' text.
        # Structure of results: [{'memory': '...', 'score': ...}, ...]
        context_list = [r['memory'] for r in results] if results else []
        context = "\n\n".join(context_list)

        # 2. Generate (similar to RAG, we use the retrieve context)
        # Or does Mem0 have a generation endpoint? 
        # Usually Mem0 is Memory-as-a-Service, so we still generate with an LLM.
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant. Answer the user info based ONLY on the provided context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        answer = response.choices[0].message.content
        return answer, results
