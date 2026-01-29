# RAG vs Mem0: Conversational AI Memory Benchmark

This benchmark compares RAG (Retrieval-Augmented Generation) against Mem0 for **user-scoped conversational memory retrieval**.

## The Problem

Conversational AI requires remembering facts about specific users across sessions. For example:
- "What is MY expense limit?" (for user_hr)
- "What is MY travel allowance?" (for user_engineering)

**RAG** stores all facts in a single vector index with no user scoping. It retrieves based on semantic similarity alone.

**Mem0** stores facts in user-specific memory graphs. Retrieval is automatically scoped to the logged-in user.

## Results

| Metric | RAG | Mem0 |
|--------|-----|------|
| **Hit Rate** | 20% | **80%** |
| Avg Latency | 0.01s | 0.68s |

Mem0 correctly retrieves user-specific facts. RAG returns semantically similar but **wrong-user** data.

## Files

- `benchmark_conversational.py` - Main benchmark script
- `mem0_backend.py` - Mem0 Platform client wrapper
- `rag_backend.py` - ChromaDB-based RAG baseline
- `inspect_mem0.py` - Utility to inspect existing Mem0 memories
- `conversational_benchmark_results.csv` - Sample results

## Prerequisites

- Python 3.10+
- Mem0 API Key (set in `mem0_backend.py`)
- OpenAI API Key (if using generation features)

## Installation

```bash
pip install -r requirements.txt
```

## Running the Benchmark

**Option 1: Use existing memories** (if you already have data in Mem0)
```bash
python benchmark_conversational.py
```

**Option 2: Fresh account** (creates sample memories first)
```bash
python benchmark_conversational.py --fresh
```

The benchmark will:
1. Fetch existing memories from Mem0 (or create sample memories with `--fresh`)
2. Build a RAG index with those same memories (no scoping)
3. Run retrieval queries for different users
4. Compare hit rates

## Key Insight

RAG is a **knowledge base** tool. It answers "What does the corpus say?"

Mem0 is a **memory layer**. It answers "What do I know about THIS user?"

For conversational AI that requires user-specific context, Mem0 provides the necessary scoping that RAG lacks.
