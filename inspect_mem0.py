"""
Utility to inspect existing memories in Mem0.
Use this to verify what facts are stored for each user.
"""

from mem0 import MemoryClient
import json

MEM0_API_KEY = "m0-dFFPdnL0iQP7DvUkRFMPTSk75TdcxH4DdyperOTF"
client = MemoryClient(api_key=MEM0_API_KEY)

def inspect_user_memories(user_ids, query="expense limit policy threshold"):
    """Search memories for given users with a broad query."""
    print(f"Searching memories with query: '{query}'\n")
    
    for user_id in user_ids:
        try:
            res = client.search(query=query, filters={"user_id": user_id}, limit=5)
            results = res.get("results", []) if isinstance(res, dict) else res
            
            print(f"--- {user_id}: {len(results)} results ---")
            for r in results[:3]:
                mem = r.get("memory", "N/A")
                score = r.get("score", 0)
                print(f"  [{score:.2f}] {mem[:150]}...")
        except Exception as e:
            print(f"{user_id}: Error - {e}")

if __name__ == "__main__":
    known_users = [
        "benchmark_user_1",
        "user_engineering",
        "user_hr",
        "user_sales",
        "user_legal",
        "user_compliance"
    ]
    inspect_user_memories(known_users)
