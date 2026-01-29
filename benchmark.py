import time
import json
import pandas as pd
import numpy as np # For percentile calc
from rag_backend import RAGBackend
from mem0_backend import Mem0Backend
import os

# Load Data
if not os.path.exists("knowledge_base.json"):
    print("Error: knowledge_base.json not found. Run generate_data.py first.")
    exit(1)

with open("knowledge_base.json", "r") as f:
    docs = json.load(f)

with open("benchmark_queries.json", "r") as f:
    queries = json.load(f)

# Create lookup map for verification
doc_map = {d["id"]: d for d in docs}

def calculate_metrics(results_list):
    df = pd.DataFrame(results_list)
    return {
        "p50_latency": df["latency"].quantile(0.50),
        "p95_latency": df["latency"].quantile(0.95),
        "p99_latency": df["latency"].quantile(0.99),
        "hit_rate": df["hit"].mean(),
        "total_queries": len(df)
    }

def run_benchmark():
    print(f"Starting Benchmark with {len(docs)} Docs and {len(queries)} Queries...")
    
    # -----------------
    # 1. RAG Benchmark
    # -----------------
    print("\n--- Testing RAG System ---")
    rag = RAGBackend()
    
    start_time = time.time()
    rag.ingest_documents(docs)
    ingest_time = time.time() - start_time
    print(f"RAG Ingestion: {ingest_time:.4f}s")
    
    rag_results = []
    
    for i, q in enumerate(queries):
        start_q = time.time()
        
        # Simulate passing user context to RAG (often ignored/hard to implement)
        context = q.get("user_context", {})
        res = rag.search(q["query_text"], user_context=context, top_k=1)
        
        lat = time.time() - start_q
        
        # Strict Verification
        retrieved_ids = res['ids'][0] if res['ids'] else []
        hit = q["target_doc_id"] in retrieved_ids
        
        rag_results.append({
            "query_id": q["query_id"],
            "latency": lat,
            "hit": hit
        })
        
        if i % 100 == 0: print(f"RAG: Processed {i} queries...")

    # -----------------
    # 2. Mem0 Benchmark
    # -----------------
    print("\n--- Testing Mem0 Platform ---")
    # For Mem0, we must assume the user_id matches the context in the query
    # In a real app, this is "user_123". Here we simulate one user per department context for fairness
    # Or just one global user if we want to test "can it figure it out?"
    # To match the article's "User Specific" claim, we should use a distinct user_id per department logic
    # But for simplicity and clean comparison, we'll use one user ID and rely on Mem0's memory of that user.
    # WAIT: The article says Mem0 wins because of User Scoping. 
    # So we should simulate "User A from HR" vs "User B from Engineering".
    
    # We will instantiate Mem0 backend dynamically per user context to be extremely fair to the "Pro" case
    # Actually, simpler: Pass user_id to search.
    
    # Let's stick to the Mem0Backend class as defined, but assume 'mem0_backend.py' handles the client.
    # We will modifying the loop to init/filter properly.
    
    # Re-init Mem0 for clean state
    # We will just use one instance and rely on 'search(..., user_id=...)' if the backend supports it.
    # checking mem0_backend.py... it takes user_id in __init__.
    # So strictly, we should re-instantiate or assume one user. 
    # Let's assume one active user "benchmark_user" who has ALL this memory.
    
    mem0 = Mem0Backend(user_id="benchmark_user_hq")
    
    start_time = time.time()
    mem0.ingest_documents(docs) # Ingest all policies for this user
    ingest_time = time.time() - start_time
    print(f"Mem0 Ingestion: {ingest_time:.4f}s")
    
    mem0_results = []
    
    for i, q in enumerate(queries):
        start_q = time.time()
        
        # Mem0 search
        res = mem0.search(q["query_text"])
        lat = time.time() - start_q
        
        # Verification: Mem0 returns text chunks.
        # We check if the KEY FACT value (e.g. "3000") is present in the answer
        # This is a robust check.
        
        hit = False
        expected_val = str(q["expected_fact"])
        
        # Parse Mem0 result list
        # result structure is usually [{"memory": "...", "score": ...}]
        if res and "results" in res:
             for r in res["results"]:
                 if expected_val in r.get("memory", ""):
                     hit = True
                     break
        
        mem0_results.append({
            "query_id": q["query_id"],
            "latency": lat,
            "hit": hit
        })
        
        if i % 100 == 0: print(f"Mem0: Processed {i} queries...")

    # -----------------
    # Reporting
    # -----------------
    metrics_rag = calculate_metrics(rag_results)
    metrics_mem0 = calculate_metrics(mem0_results)
    
    print("\nXXX BENCHMARK REPORT XXX")
    print("-" * 30)
    print(f"{'Metric':<20} | {'RAG (Local)':<15} | {'Mem0 (Cloud)':<15}")
    print("-" * 30)
    print(f"{'Hit Rate':<20} | {metrics_rag['hit_rate']:.2%}           | {metrics_mem0['hit_rate']:.2%}")
    print(f"{'P50 Latency':<20} | {metrics_rag['p50_latency']:.4f}s         | {metrics_mem0['p50_latency']:.4f}s")
    print(f"{'P95 Latency':<20} | {metrics_rag['p95_latency']:.4f}s         | {metrics_mem0['p95_latency']:.4f}s")
    print("-" * 30)
    
    # Save detailed CSV
    df = pd.DataFrame(rag_results)
    df["system"] = "RAG"
    df2 = pd.DataFrame(mem0_results)
    df2["system"] = "Mem0"
    pd.concat([df, df2]).to_csv("benchmark_results_detailed.csv", index=False)
    print("Detailed results saved to benchmark_results_detailed.csv")

if __name__ == "__main__":
    run_benchmark()
