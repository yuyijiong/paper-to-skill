---
name: general-agentic-memory
description: >
  General Agentic Memory (GAM) via deep research. Implements Just-in-Time memory
  optimization with dual-agent architecture: MemoryAgent constructs structured abstracts,
  ResearchAgent performs iterative retrieval-reflection loops at query time. Supports
  keyword (BM25), vector (dense), and page index retrieval. Use when building agents that
  need deep research over stored memories, implementing multi-hop QA systems, or creating
  assistants that reason iteratively over context. Trigger: "deep research memory",
  "GAM setup", "iterative retrieval", "multi-hop reasoning".
---

# General Agentic Memory (GAM)

Deep research-powered memory for AI agents. Uses dual-agent architecture for memory construction and Just-in-Time retrieval with iterative reflection.

## Key Concepts

| Concept | Description |
|---------|-------------|
| **Just-in-Time (JIT)** | Intensive research at query time, not preprocessing |
| **MemoryAgent** | Constructs abstracts from documents |
| **ResearchAgent** | Iterative search-integrate-reflect loop |
| **Deep Research** | Multi-round retrieval with reflection |

## Prerequisites

```bash
# Clone and install
git clone https://github.com/VectorSpaceLab/general-agentic-memory.git
cd general-agentic-memory
pip install -e .

# Required models
# - Dense retriever: BAAI/bge-m3
```

**Environment variables:**
- `OPENAI_API_KEY` (required for OpenAI backend)

## Quick Start

### 1. Setup Generator and Stores

```python
import os
from gam import (
    MemoryAgent, ResearchAgent,
    OpenAIGenerator, OpenAIGeneratorConfig,
    InMemoryMemoryStore, InMemoryPageStore,
    IndexRetriever, IndexRetrieverConfig,
    BM25Retriever, BM25RetrieverConfig,
    DenseRetriever, DenseRetrieverConfig
)

# Generator config
gen_config = OpenAIGeneratorConfig(
    model_name="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.3,
    max_tokens=256
)
generator = OpenAIGenerator.from_config(gen_config)

# Stores
memory_store = InMemoryMemoryStore()
page_store = InMemoryPageStore()
```

### 2. Build Memory with MemoryAgent

```python
memory_agent = MemoryAgent(
    generator=generator,
    memory_store=memory_store,
    page_store=page_store
)

# Memorize documents
documents = [
    "Machine Learning is a subset of AI that enables learning from data.",
    "Deep Learning uses neural networks with multiple layers.",
    "Transformers revolutionized NLP with attention mechanisms."
]

for doc in documents:
    memory_agent.memorize(doc)

# Check memory state
state = memory_store.load()
print(f"Built {len(state.abstracts)} memory abstracts")
```

### 3. Setup Retrievers

```python
retrievers = {}

# BM25 keyword retriever
bm25_config = BM25RetrieverConfig(index_dir="./index/bm25", threads=1)
bm25_retriever = BM25Retriever(bm25_config.__dict__)
bm25_retriever.build(page_store)
retrievers["keyword"] = bm25_retriever

# Dense vector retriever
dense_config = DenseRetrieverConfig(
    index_dir="./index/dense",
    model_name="BAAI/bge-m3"
)
dense_retriever = DenseRetriever(dense_config.__dict__)
dense_retriever.build(page_store)
retrievers["vector"] = dense_retriever

# Page index retriever
index_config = IndexRetrieverConfig(index_dir="./index/page")
index_retriever = IndexRetriever(index_config.__dict__)
index_retriever.build(page_store)
retrievers["page_index"] = index_retriever
```

### 4. Research with ResearchAgent

```python
research_agent = ResearchAgent(
    page_store=page_store,
    memory_store=memory_store,
    retrievers=retrievers,
    generator=generator,
    max_iters=5  # Max reflection iterations
)

# Perform research
result = research_agent.research(
    request="What is the relationship between ML and Deep Learning?"
)

print(f"Iterations: {len(result.raw_memory['iterations'])}")
print(f"Answer: {result.integrated_memory}")
```

## Architecture

### Dual-Agent Design

```
Documents → [MemoryAgent] → Abstracts + Pages
                              ↓
Query → [ResearchAgent] → Planning → Search → Integrate → Reflect
                              ↑___________________________|
                                    (iterate until enough)
```

### ResearchAgent Loop

1. **Planning**: Analyze query, generate search plan
   - Identify info needs
   - Select tools (keyword, vector, page_index)
   - Generate queries

2. **Search**: Execute planned searches
   - Keyword (BM25) for exact entities
   - Vector for semantic similarity
   - Page index for known relevant pages

3. **Integrate**: Combine results with LLM
   - Deduplicate hits
   - Synthesize factual summary

4. **Reflect**: Check completeness
   - Enough info? → Return result
   - Not enough? → Generate follow-up questions, iterate

## Configuration Options

### Generator (OpenAI)

```python
OpenAIGeneratorConfig(
    model_name="gpt-4o-mini",      # Model to use
    api_key="...",                  # API key
    base_url="https://api.openai.com/v1",
    temperature=0.3,                # Lower = more deterministic
    max_tokens=256                  # Max output tokens
)
```

### Generator (vLLM for local)

```python
from gam import VLLMGenerator, VLLMGeneratorConfig

vllm_config = VLLMGeneratorConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    tensor_parallel_size=1
)
generator = VLLMGenerator.from_config(vllm_config)
```

### ResearchAgent

```python
ResearchAgent(
    page_store=page_store,
    memory_store=memory_store,
    retrievers=retrievers,
    generator=generator,
    max_iters=5,                    # Max reflection iterations
    system_prompts={                # Custom prompts (optional)
        "planning": "...",
        "integration": "...",
        "reflection": "..."
    }
)
```

## TTL (Time-To-Live) for Production

Prevent unbounded memory growth:

```python
from gam import TTLMemoryStore, TTLPageStore

# Auto-cleanup memories older than 30 days
memory_store = TTLMemoryStore(
    dir_path="./data",
    ttl_days=30,
    enable_auto_cleanup=True
)
page_store = TTLPageStore(
    dir_path="./data",
    ttl_days=30,
    enable_auto_cleanup=True
)

# Check statistics
stats = memory_store.get_stats()
print(f"Total: {stats['total']}, Valid: {stats['valid']}, Expired: {stats['expired']}")

# Manual cleanup
removed = memory_store.cleanup_expired()
```

## Persistent Storage

```python
# File-based storage
memory_store = InMemoryMemoryStore(dir_path="./my_agent/memory")
page_store = InMemoryPageStore(dir_path="./my_agent/pages")

# Saves to:
# ./my_agent/memory/memory_state.json
# ./my_agent/pages/pages.json
```

## Algorithm Details

For the full planning, integration, and reflection prompts, see [references/algorithm.md](references/algorithm.md).

## Benchmarks

GAM achieves state-of-the-art on:

| Dataset | Task | Metric |
|---------|------|--------|
| LoCoMo | Conversation Memory | F1, BLEU-1 |
| HotpotQA | Multi-hop QA | F1 |
| RULER | Long Context | Accuracy |
| NarrativeQA | Narrative QA | F1 |
