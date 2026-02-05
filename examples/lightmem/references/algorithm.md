# LightMem Algorithm Reference

Detailed algorithm documentation from the LightMem paper and implementation.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Fact Extraction](#fact-extraction)
3. [Memory Update Logic](#memory-update-logic)
4. [Retrieval Pipeline](#retrieval-pipeline)
5. [Data Model](#data-model)

---

## Architecture Overview

LightMem follows a modular pipeline:

```
Input Messages
     ↓
[Pre-Compression] (optional)
     ↓
[Topic Segmentation] (optional)
     ↓
[Fact Extraction] (LLM-based)
     ↓
[Embedding + Storage] (Qdrant)
     ↓
[Retrieval] (Vector search)
```

### Components

| Component | Purpose | Backends |
|-----------|---------|----------|
| PreCompressor | Reduce token count | llmlingua-2, entropy_compress |
| TopicSegmenter | Split by topic | llmlingua-2 |
| MemoryManager | LLM for extraction/update | openai, deepseek, ollama, vllm, transformers |
| TextEmbedder | Generate embeddings | huggingface |
| EmbeddingRetriever | Vector storage/search | qdrant |
| ContextRetriever | Keyword search | BM25 |

---

## Fact Extraction

The core of LightMem is extracting standalone facts from conversations.

### Extraction Prompt

```
You are a Personal Information Extractor.
Your task is to extract **all possible facts or information** about the user from a conversation.

Input format:
--- Topic X ---
[timestamp, weekday] source_id.SpeakerName: message
...

Important Instructions:
1. Process messages strictly in ascending source_id order
2. For each message, decide if it contains factual information
   - If yes → extract and rephrase as standalone sentence
   - If no (greeting, filler) → skip
3. Use light contextual completion for clarity

Output format:
{
  "data": [
    {"source_id": "<id>", "fact": "<complete fact>"}
  ]
}
```

### Extraction Examples

**Input:**
```
--- Topic 1 ---
[2022-03-20T13:21:00.000, Sun] 0.User: My name is Alice and I work as a teacher.
[2022-03-20T13:21:00.500, Sun] 1.User: My favourite movies are Inception and Interstellar.
```

**Output:**
```json
{"data": [
  {"source_id": 0, "fact": "User's name is Alice."},
  {"source_id": 0, "fact": "User works as a teacher."},
  {"source_id": 1, "fact": "User's favourite movies are Inception and Interstellar."}
]}
```

### Extraction Guidelines

1. **Be exhaustive** - Extract even minor details
2. **Preserve specifics** - Full names, locations, numbers
3. **Contextual completion** - Make facts standalone
4. **Time handling** - Note relative times with reference date

---

## Memory Update Logic

After extraction, similar memories are consolidated to reduce redundancy.

### Update Queue Construction

For each memory entry:
1. Find top-K similar entries (by embedding)
2. Filter to entries with earlier timestamps
3. Keep top-N as update candidates

```python
def construct_update_queue(entry, top_k=20, keep_top_n=10):
    vec = entry.vector
    ts = entry.float_time_stamp

    # Find similar entries before this one
    hits = retriever.search(
        query_vector=vec,
        limit=top_k,
        filters={"float_time_stamp": {"lte": ts}}
    )

    # Exclude self, keep top N
    candidates = [h for h in hits if h.id != entry.id]
    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates[:keep_top_n]
```

### Update Prompt

```
You are a memory management assistant.
Decide whether the target memory should be updated, deleted, or ignored
based on candidate source memories.

Decision rules:
1. Update: Same fact with refinements → integrate information
2. Delete: Direct conflict with newer info → delete target
3. Ignore: Unrelated memories → no action

Output:
{
  "action": "update" | "delete" | "ignore",
  "new_memory": "..."  // only for update
}
```

### Update Examples

**Example 1: Update (refinement)**
```
Target: "The user likes coffee."
Candidates:
- "The user prefers cappuccino in the mornings."
- "The user avoids decaf."

Output: {
  "action": "update",
  "new_memory": "The user likes coffee, especially cappuccino in the morning, and avoids decaf."
}
```

**Example 2: Delete (conflict)**
```
Target: "The user lives in New York."
Candidates:
- "The user moved to San Francisco in 2023."

Output: {"action": "delete"}
```

**Example 3: Ignore (unrelated)**
```
Target: "The user is learning Italian cooking."
Candidates:
- "The user started practicing yoga."

Output: {"action": "ignore"}
```

---

## Retrieval Pipeline

### Embedding-Based Retrieval

```python
def retrieve(query: str, limit: int = 10):
    # 1. Embed query
    query_vector = text_embedder.embed(query)

    # 2. Vector search
    results = embedding_retriever.search(
        query_vector=query_vector,
        limit=limit,
        return_full=True
    )

    # 3. Format results
    formatted = []
    for r in results:
        payload = r["payload"]
        formatted.append(
            f"{payload['time_stamp']} {payload['weekday']} {payload['memory']}"
        )

    return "\n".join(formatted)
```

### Hybrid Retrieval

Combines embedding search with BM25 keyword matching:

1. BM25 retrieves top candidates by keywords
2. Embedding search reranks by semantic similarity
3. Final results merged and deduplicated

---

## Data Model

### MemoryEntry

```python
@dataclass
class MemoryEntry:
    id: str                    # Unique identifier
    time_stamp: str            # ISO timestamp
    float_time_stamp: float    # Unix timestamp for filtering
    weekday: str               # Day of week
    topic_id: int              # Topic segment ID
    topic_summary: str         # Summary of topic
    category: str              # Memory category
    subcategory: str           # Memory subcategory
    memory_class: str          # Classification
    memory: str                # Extracted fact
    original_memory: str       # Original message
    compressed_memory: str     # Compressed version
    speaker_id: int            # Speaker identifier
    speaker_name: str          # Speaker name
```

### Vector Payload (Qdrant)

```json
{
    "time_stamp": "2024-01-15T10:30:00.000",
    "float_time_stamp": 1705315800.0,
    "weekday": "Mon",
    "topic_id": 1,
    "topic_summary": "User preferences",
    "category": "User-related",
    "subcategory": "Preferences",
    "memory_class": "fact",
    "memory": "User's favorite color is blue.",
    "original_memory": "My favorite color is blue.",
    "compressed_memory": "favorite color blue",
    "speaker_id": 0,
    "speaker_name": "User",
    "update_queue": [
        {"id": "abc123", "score": 0.85},
        {"id": "def456", "score": 0.72}
    ]
}
```

### Token Statistics

LightMem tracks API usage:

```python
{
    "summary": {
        "total_llm_calls": 10,
        "total_llm_tokens": 5000,
        "total_embedding_calls": 50,
        "total_embedding_tokens": None  # Local models
    },
    "llm": {
        "add_memory": {
            "calls": 5,
            "prompt_tokens": 2000,
            "completion_tokens": 500,
            "total_tokens": 2500
        },
        "update": {
            "calls": 5,
            "prompt_tokens": 2000,
            "completion_tokens": 500,
            "total_tokens": 2500
        }
    }
}
```

---

## Performance Benchmarks

From the paper (LoCoMo dataset, GPT-4o-mini backbone):

| Method | Accuracy | Total Tokens (k) | Runtime (s) |
|--------|----------|------------------|-------------|
| FullText | 73.83% | 54,884 | 6,971 |
| NaiveRAG | 63.64% | 3,870 | 1,884 |
| A-MEM | 64.16% | 21,665 | 67,084 |
| Mem0 | 36.49% | 25,793 | 120,175 |
| LightMem | **TBD** | **Low** | **Fast** |

LightMem achieves competitive accuracy with significantly lower token usage and runtime compared to other memory systems.
