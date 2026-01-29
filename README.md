# Mem0 Benchmark: RAG vs Memory Layer

This repository contains the official benchmark code for the article **"Benchmarking RAG vs Mem0: A rigorous technical analysis of stateful AI memory"**.

It provides a head-to-head comparison between a standard Production-Grade RAG implementation (using ChromaDB) and the Mem0 Platform.

## Experiment Design

The benchmark is designed to isolate **Retrieval Latency** and **Hit Rate** using a synthetic but high-fidelity dataset.

### The Dataset
We generate 50 complex "Corporate Policy" documents using `generate_data.py`. 
- **Chaos**: We inject version conflicts (v1 vs v2), departmental overlap (Engineering vs Sales policies), and noise.
- **The Needle**: Each document contains a hidden, numeric `KEY_FACT` (e.g., spending limit).
- **The Query**: We generate 1000 context-aware queries (e.g., "As an engineer, what can I spend?").

### The Competitors
1. **Baseline**: `rag_backend.py` - Uses `chromadb` with metadata filtering and persistent storage.
2. **Challenger**: `mem0_backend.py` - Uses Mem0 SDK with `user_id` scoping.

## How to Run

### Prerequisites
- Python 3.10+
- A Mem0 API Key (Get one free at [mem0.ai](https://mem0.ai))

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/mem0ai/mem0-benchmark
   cd mem0-benchmark
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Benchmark

1. **Generate the Data**:
   This script creates the `knowledge_base.json` and `benchmark_queries.json` files.
   ```bash
   python generate_data.py
   ```

2. **Run the Test**:
   Set your API key and execute the runner.
   ```bash
   export MEM0_API_KEY="m0-xxx-your-key-xxx"
   # Optional: export OPENAI_API_KEY="sk-..." if using OpenAI features in RAG extension
   
   python benchmark.py
   ```

### Results

The script will output a summary to the console:

```text
XXX BENCHMARK REPORT XXX
------------------------------
Metric               | RAG (Local)     | Mem0 (Cloud)   
------------------------------
Hit Rate             | 30.00%          | 70.00%
P50 Latency          | 0.0500s         | 0.4800s
P95 Latency          | 0.0900s         | 0.8500s
------------------------------
Detailed results saved to benchmark_results_detailed.csv
```

You can inspect `benchmark_results_detailed.csv` to analyze specific failure cases.

## Contributing

If you believe the RAG implementation is unfair, please open a PR! We welcome `HybridRAGBackend` or `RerankedRAGBackend` implementations to see how they stack up against the Memory Layer.
