---
name: lightmem
description: >
  Lightweight memory-augmented generation using the LightMem framework. Provides long-term
  memory for conversations with topic segmentation, fact extraction, and semantic retrieval.
  Use when building agents that need persistent memory across sessions, implementing personal
  assistants that remember user preferences, or adding memory capabilities to chatbots.
  Supports OpenAI, DeepSeek, Ollama, and local models. Trigger: "add memory to agent",
  "remember conversations", "build personal assistant with memory", "LightMem setup".
---

# LightMem: Memory-Augmented Generation

Lightweight framework for adding long-term memory to LLM applications. Extracts facts from conversations, stores in vector DB, retrieves relevant memories for context.

## Prerequisites

```bash
# Install LightMem
pip install lightmem

# Or from source
git clone https://github.com/zjunlp/LightMem.git
cd LightMem && pip install -e .

# Download required models
# 1. LLMLingua-2: microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank
# 2. Embedding: sentence-transformers/all-MiniLM-L6-v2
```

**Environment variables:**
- `OPENAI_API_KEY` (if using OpenAI backend)

## Quick Start

### Minimal Setup

```python
from lightmem.memory.lightmem import LightMemory

config = {
    "pre_compress": False,
    "topic_segment": False,
    "metadata_generate": True,
    "text_summary": True,
    "memory_manager": {
        "model_name": "openai",
        "configs": {
            "model": "gpt-4o-mini",
            "api_key": "your-api-key",
        }
    },
    "index_strategy": "embedding",
    "text_embedder": {
        "model_name": "huggingface",
        "configs": {
            "model": "/path/to/all-MiniLM-L6-v2",
            "embedding_dims": 384,
        },
    },
    "retrieve_strategy": "embedding",
    "embedding_retriever": {
        "model_name": "qdrant",
        "configs": {
            "collection_name": "my_memory",
            "embedding_model_dims": 384,
            "path": "./qdrant_data/my_memory",
        }
    },
    "update": "offline",
}

lightmem = LightMemory.from_config(config)
```

### Full Setup (with compression and segmentation)

```python
config = {
    "pre_compress": True,
    "pre_compressor": {
        "model_name": "llmlingua-2",
        "configs": {
            "llmlingua_config": {
                "model_name": "/path/to/llmlingua-2-model",
                "device_map": "cuda",
                "use_llmlingua2": True,
            },
            "compress_config": {"rate": 0.6}
        }
    },
    "topic_segment": True,
    "precomp_topic_shared": True,
    "topic_segmenter": {"model_name": "llmlingua-2"},
    "messages_use": "user_only",
    "metadata_generate": True,
    "text_summary": True,
    "memory_manager": {
        "model_name": "openai",  # or "ollama", "deepseek", "vllm"
        "configs": {
            "model": "gpt-4o-mini",
            "api_key": "your-api-key",
            "max_tokens": 16000,
        }
    },
    "extract_threshold": 0.1,
    "index_strategy": "embedding",
    "text_embedder": {
        "model_name": "huggingface",
        "configs": {
            "model": "/path/to/all-MiniLM-L6-v2",
            "embedding_dims": 384,
            "model_kwargs": {"device": "cuda"},
        },
    },
    "retrieve_strategy": "embedding",
    "embedding_retriever": {
        "model_name": "qdrant",
        "configs": {
            "collection_name": "my_memory",
            "embedding_model_dims": 384,
            "path": "./qdrant_data/my_memory",
        }
    },
    "update": "offline",
}

lightmem = LightMemory.from_config(config)
```

## Core Workflow

### Phase 1: Add Memory

Store conversation turns with timestamps:

```python
messages = [
    {"role": "user", "content": "My favorite color is blue.", "time_stamp": "2024-01-15T10:30:00"},
    {"role": "assistant", "content": "Got it, blue is a nice color.", "time_stamp": "2024-01-15T10:30:00"},
]

result = lightmem.add_memory(
    messages=messages,
    force_segment=True,   # Force topic segmentation
    force_extract=True    # Force fact extraction
)
```

**What happens:**
1. Messages normalized with timestamps
2. (Optional) Pre-compressed to reduce tokens
3. (Optional) Segmented by topic
4. Facts extracted by LLM (e.g., "User's favorite color is blue")
5. Stored in vector database

### Phase 2: Offline Update (Optional)

Consolidate related memories after batch additions:

```python
# Build update queue (find related memories)
lightmem.construct_update_queue_all_entries(top_k=20, keep_top_n=10)

# Update/delete redundant memories
lightmem.offline_update_all_entries(score_threshold=0.8)
```

### Phase 3: Retrieve

Search for relevant memories:

```python
query = "What is the user's favorite color?"
memories = lightmem.retrieve(query, limit=5)
print(memories)
# Output: "2024-01-15T10:30:00 Mon User's favorite color is blue."
```

### Phase 4: Use in Generation

Inject memories into LLM context:

```python
from openai import OpenAI

client = OpenAI()
memories = lightmem.retrieve(user_question, limit=10)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": f"Relevant memories:\n{memories}"},
        {"role": "user", "content": user_question}
    ]
)
```

## Configuration Reference

| Option | Values | Description |
|--------|--------|-------------|
| `pre_compress` | true/false | Enable token compression |
| `topic_segment` | true/false | Enable topic-based segmentation |
| `messages_use` | user_only/assistant_only/hybrid | Which messages to extract facts from |
| `metadata_generate` | true/false | Extract metadata (keywords, entities) |
| `text_summary` | true/false | Generate text summaries |
| `index_strategy` | embedding/context/hybrid | Indexing method |
| `retrieve_strategy` | embedding/context/hybrid | Retrieval method |
| `update` | online/offline | When to consolidate memories |

## Backend Options

### OpenAI
```python
"memory_manager": {
    "model_name": "openai",
    "configs": {"model": "gpt-4o-mini", "api_key": "..."}
}
```

### Ollama (Local)
```python
"memory_manager": {
    "model_name": "ollama",
    "configs": {"model": "llama3:latest", "host": "http://localhost:11434"}
}
```

### DeepSeek
```python
"memory_manager": {
    "model_name": "deepseek",
    "configs": {"model": "deepseek-chat", "api_key": "...", "deepseek_base_url": "..."}
}
```

## MCP Server

LightMem provides an MCP server for integration:

```bash
# Install MCP dependencies
pip install 'lightmem[mcp]'

# Run server
cd LightMem
fastmcp run mcp/server.py:mcp --transport http --port 8000
```

**Available tools:**
- `add_memory`: Add conversation to memory
- `retrieve_memory`: Search memories
- `offline_update`: Consolidate memories
- `get_timestamp`: Get current timestamp
- `show_lightmem_instance`: Show configuration

## Algorithm Details

For the full extraction prompts, update logic, and retrieval algorithms, see [references/algorithm.md](references/algorithm.md).
