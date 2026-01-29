"""
Benchmark: Conversational AI Memory Retrieval

This benchmark tests the ability of a retrieval system to provide accurate,
user-scoped context for conversational AI.

Scenario:
- Multiple users (Engineering, HR, Sales) have had prior conversations.
- Each user has distinct facts stored (e.g., expense limits, policies).
- A follow-up query asks for a user-specific fact.

Metric: Can the system retrieve the CORRECT user-specific fact?

RAG Baseline:
- All facts are ingested into a single vector store (ChromaDB).
- No user scoping. RAG must "figure it out" from query context alone.

Mem0 Challenger:
- Facts are stored per-user in isolated memory graphs.
- Search is scoped to the user_id.

Usage:
    python benchmark_conversational.py              # Use existing Mem0 memories
    python benchmark_conversational.py --fresh      # Create new memories (fresh account)
"""

import time
import argparse
import pandas as pd
from mem0 import MemoryClient
import chromadb
from chromadb.utils import embedding_functions

MEM0_API_KEY = "m0-dFFPdnL0iQP7DvUkRFMPTSk75TdcxH4DdyperOTF"

# --- Sample Facts for Fresh Account Setup ---
SAMPLE_FACTS = [
    {"user_id": "user_engineering", "content": "My travel allowance limit is $4662 per quarter."},
    {"user_id": "user_engineering", "content": "My expense reimbursement limit is $206 per month."},
    {"user_id": "user_engineering", "content": "I work in the Engineering department and focus on backend systems."},
    {"user_id": "user_hr", "content": "My expense reimbursement limit is $2343 per month."},
    {"user_id": "user_hr", "content": "The data privacy policy limit is $1095 for training materials."},
    {"user_id": "user_hr", "content": "The onboarding budget is $1691 per new hire."},
    {"user_id": "user_hr", "content": "I work in HR and handle employee relations."},
    {"user_id": "user_sales", "content": "My quarterly quota is $250000."},
    {"user_id": "user_sales", "content": "My expense limit for client entertainment is $500 per event."},
]

# --- Test Queries ---
TEST_QUERIES = [
    {"user_id": "user_engineering", "query": "What is my travel allowance limit?", "expected_fact": "4662"},
    {"user_id": "user_engineering", "query": "How much can I expense?", "expected_fact": "206"},
    {"user_id": "user_hr", "query": "What is my expense reimbursement limit?", "expected_fact": "2343"},
    {"user_id": "user_hr", "query": "What is the data privacy policy limit?", "expected_fact": "1095"},
    {"user_id": "user_hr", "query": "What is the onboarding budget?", "expected_fact": "1691"},
    # Cross-user queries to test scoping
    {"user_id": "user_engineering", "query": "What is the onboarding budget?", "expected_fact": None},
    {"user_id": "user_hr", "query": "What is my travel allowance?", "expected_fact": None},
]

def create_fresh_memories(client):
    """Create sample memories for a fresh Mem0 account."""
    print("Creating sample memories in Mem0...")
    for fact in SAMPLE_FACTS:
        client.add(
            messages=[{"role": "user", "content": fact["content"]}],
            user_id=fact["user_id"]
        )
        print(f"  + {fact['user_id']}: {fact['content'][:50]}...")
    print(f"  -> Created {len(SAMPLE_FACTS)} memories.\n")
    return SAMPLE_FACTS

def fetch_existing_memories(client):
    """Fetch existing memories from Mem0."""
    print("Fetching existing memories from Mem0...")
    all_memories = []
    user_ids = ["user_engineering", "user_hr", "user_sales", "user_legal", "user_compliance", "benchmark_user_1"]
    
    for uid in user_ids:
        try:
            res = client.search(query="*", filters={"user_id": uid}, limit=100)
            results = res.get("results", []) if isinstance(res, dict) else res
            for r in results:
                all_memories.append({
                    "user_id": uid,
                    "memory": r.get("memory", ""),
                    "id": r.get("id", "")
                })
        except:
            pass
    
    print(f"  -> Fetched {len(all_memories)} total memories across users.\n")
    return all_memories

def run_benchmark(use_fresh=False):
    print("=== Conversational AI Memory Retrieval Benchmark ===\n")
    
    mem0_client = MemoryClient(api_key=MEM0_API_KEY)
    
    # Step 1: Get memories (either create fresh or fetch existing)
    if use_fresh:
        print("Mode: FRESH ACCOUNT (creating new memories)\n")
        memories = create_fresh_memories(mem0_client)
        # Convert to format expected by RAG
        all_memories = [{"user_id": m["user_id"], "memory": m["content"], "id": f"fresh_{i}"} for i, m in enumerate(memories)]
    else:
        print("Mode: EXISTING MEMORIES (fetching from Mem0)\n")
        all_memories = fetch_existing_memories(mem0_client)
    
    if not all_memories:
        print("ERROR: No memories found. Use --fresh to create sample memories.")
        return
    
    # Step 2: Build RAG Index (No User Scoping)
    print("Building RAG index (ChromaDB, no scoping)...")
    
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    chroma_client = chromadb.Client()
    rag_collection = chroma_client.get_or_create_collection(name="rag_memories", embedding_function=ef)
    
    rag_collection.add(
        documents=[m["memory"] for m in all_memories],
        ids=[m["id"] for m in all_memories],
        metadatas=[{"user_id": m["user_id"]} for m in all_memories]
    )
    print(f"  -> Indexed {len(all_memories)} memories into ChromaDB.\n")
    
    # Step 3: Run Benchmark
    print("Running retrieval tests...\n")
    
    results = []
    
    for q in TEST_QUERIES:
        user_id = q["user_id"]
        query_text = q["query"]
        expected = q["expected_fact"]
        
        # RAG Retrieval (Global Search, No Scoping)
        rag_start = time.time()
        rag_results = rag_collection.query(query_texts=[query_text], n_results=3)
        rag_latency = time.time() - rag_start
        
        rag_hit = False
        rag_top_doc = ""
        if rag_results and rag_results["documents"] and rag_results["documents"][0]:
            rag_top_doc = rag_results["documents"][0][0]
            if expected and expected in rag_top_doc:
                rag_hit = True
        
        # Mem0 Retrieval (User-Scoped Search)
        mem0_start = time.time()
        mem0_res = mem0_client.search(query=query_text, filters={"user_id": user_id}, limit=3)
        mem0_latency = time.time() - mem0_start
        
        mem0_hit = False
        mem0_top_doc = ""
        mem0_results_list = mem0_res.get("results", []) if isinstance(mem0_res, dict) else mem0_res
        if mem0_results_list:
            mem0_top_doc = mem0_results_list[0].get("memory", "")
            if expected and expected in mem0_top_doc:
                mem0_hit = True
        
        results.append({
            "user_id": user_id,
            "query": query_text,
            "expected": expected,
            "rag_hit": rag_hit,
            "rag_latency": rag_latency,
            "rag_top_doc": rag_top_doc[:80],
            "mem0_hit": mem0_hit,
            "mem0_latency": mem0_latency,
            "mem0_top_doc": mem0_top_doc[:80]
        })
        
        print(f"Query: '{query_text}' (User: {user_id})")
        print(f"  Expected: {expected}")
        print(f"  RAG:  {'HIT' if rag_hit else 'MISS'} | {rag_top_doc[:60]}...")
        print(f"  Mem0: {'HIT' if mem0_hit else 'MISS'} | {mem0_top_doc[:60]}...")
        print()
    
    # Summary
    df = pd.DataFrame(results)
    df_expected = df[df["expected"].notna()]
    
    rag_hit_rate = df_expected["rag_hit"].mean()
    mem0_hit_rate = df_expected["mem0_hit"].mean()
    
    print("\n=== BENCHMARK SUMMARY ===")
    print(f"Total Queries: {len(df)}")
    print(f"Queries with Expected Fact: {len(df_expected)}")
    print(f"\nRAG Hit Rate:  {rag_hit_rate:.2%}")
    print(f"Mem0 Hit Rate: {mem0_hit_rate:.2%}")
    print(f"\nRAG Avg Latency:  {df['rag_latency'].mean():.4f}s")
    print(f"Mem0 Avg Latency: {df['mem0_latency'].mean():.4f}s")
    
    df.to_csv("conversational_benchmark_results.csv", index=False)
    print("\nDetailed results saved to conversational_benchmark_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark RAG vs Mem0 for conversational memory retrieval")
    parser.add_argument("--fresh", action="store_true", help="Create new sample memories (for fresh Mem0 accounts)")
    args = parser.parse_args()
    
    run_benchmark(use_fresh=args.fresh)
