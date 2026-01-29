import json
import random
from faker import Faker
import uuid
from datetime import datetime, timedelta

fake = Faker()

# Real-world corporate categories
DEPARTMENTS = ["Engineering", "HR", "Sales", "Legal", "Compliance"]
DOC_TYPES = ["Policy", "Standard Operating Procedure", "Meeting Notes", "Proposal"]
STATUSES = ["Draft", "Final", "Archived", "Deprecated"]

def generate_complex_kb(num_docs=50):
    docs = []
    
    # We want to create "conflicting" information to test memory/context.
    # e.g. "2023 Remote Work Policy" vs "2024 Remote Work Policy"
    
    topics = [
        "Remote Work", "Expense Reimbursement", "Cloud Security", 
        "Hiring Process", "Code Review", "incident Response", 
        "Data Privacy", "Travel Allowance", "Procurement", "Onboarding"
    ]
    
    for _ in range(num_docs):
        topic = random.choice(topics)
        dept = random.choice(DEPARTMENTS)
        doc_type = random.choice(DOC_TYPES)
        status = random.choice(STATUSES)
        
        # Create a "Key Fact" that changes based on version/date to challenge retrieval
        # e.g. older docs have lower limits.
        limit_val = random.randint(100, 5000)
        
        title = f"{dept} {topic} {doc_type}"
        
        # Simulate real markdown structure
        content = f"""
# {title}
**ID**: {uuid.uuid4()}
**Department**: {dept}
**Status**: {status}
**Last Updated**: {fake.date_between(start_date='-2y', end_date='today')}
**Author**: {fake.name()}

## 1. Executive Summary
{fake.paragraph(nb_sentences=3)}

## 2. {topic} Guidelines
The objective of this {doc_type} is to define the boundaries for {dept} regarding {topic}.
{fake.paragraph(nb_sentences=5)}

### Key Thresholds & Limits
> **CRITICAL**: The current approved limit for {topic} is **${limit_val}** (or equivalent units). This supersedes all previous memos.

## 3. Compliance and Exceptions
Any exceptions to the ${limit_val} rule must be approved by the VP of {dept}.
{fake.paragraph(nb_sentences=4)}

## 4. References
- Internal Wiki Link: {fake.uri()}
- Slack Channel: #{dept.lower()}-{topic.lower().replace(' ', '-')}
"""
        
        docs.append({
            "id": str(uuid.uuid4()),
            "title": title,
            "content": content,
            "topic": topic,
            "department": dept,
            "status": status,
            "key_fact_value": limit_val,
            "metadata": {
                "source": "internal_wiki",
                "access_level": "internal",
                "version": random.choice(["1.0", "1.1", "2.0"])
            }
        })
        
    return docs

def generate_realistic_queries(docs, num_queries=1000):
    queries = []
    
    for _ in range(num_queries):
        target_doc = random.choice(docs)
        
        # We generate queries that imply a specific user context or constraint
        # "I am in Engineering, what is my expense limit?"
        
        q_templates = [
            f"What is the {target_doc['topic']} limit for {target_doc['department']}?",
            f"Current threshold for {target_doc['topic']} in {target_doc['department']}",
            f"As a {target_doc['department']} employee, how much can I spend on {target_doc['topic']}?",
            f"Show me the {target_doc['status']} policy for {target_doc['topic']}"
        ]
        
        query_text = random.choice(q_templates)
        
        queries.append({
            "query_id": str(uuid.uuid4()),
            "query_text": query_text,
            "target_doc_id": target_doc["id"],
            "expected_fact": target_doc["key_fact_value"],
            "user_context": {
                "department": target_doc["department"],
                "role": "Employee"
            }
        })
        
    return queries

if __name__ == "__main__":
    print("Generating High-Fidelity Knowledge Base...")
    kb = generate_complex_kb(50)
    with open("knowledge_base.json", "w") as f:
        json.dump(kb, f, indent=2)
        
    print(f"Generated {len(kb)} documents.")
    
    print("Generating Context-Aware Queries...")
    qs = generate_realistic_queries(kb, 1000)
    with open("benchmark_queries.json", "w") as f:
        json.dump(qs, f, indent=2)
        
    print(f"Generated {len(qs)} queries.")
