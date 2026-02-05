# Memory System Patterns Reference

Common implementation patterns for memory and learning systems in AI agents.

## Table of Contents

1. [Storage Patterns](#storage-patterns)
2. [Retrieval Patterns](#retrieval-patterns)
3. [Weight/Quality Systems](#weightquality-systems)
4. [Feedback Mechanisms](#feedback-mechanisms)
5. [Decay and Pruning](#decay-and-pruning)
6. [Common Data Models](#common-data-models)

---

## Storage Patterns

### JSONL File Storage

Simple, append-friendly format for experience records.

```python
# Append new record
with open("memory.jsonl", "a") as f:
    f.write(json.dumps(record) + "\n")

# Load all records
records = []
with open("memory.jsonl", "r") as f:
    for line in f:
        records.append(json.loads(line))
```

**Pros**: Simple, human-readable, easy to debug
**Cons**: Full scan for updates, no indexing

### SQLite Storage

Better for larger databases with complex queries.

```python
import sqlite3
conn = sqlite3.connect("memory.db")
conn.execute("""
    CREATE TABLE IF NOT EXISTS experiences (
        id TEXT PRIMARY KEY,
        task TEXT,
        lesson TEXT,
        embedding BLOB,
        weight REAL DEFAULT 1.0
    )
""")
```

**Pros**: Indexed queries, atomic updates
**Cons**: More complex, embedding storage requires serialization

---

## Retrieval Patterns

### Embedding-Based Semantic Search

Most common for memory systems. Uses vector similarity.

```python
from openai import OpenAI

def embed(text: str) -> list:
    client = OpenAI()
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def cosine_similarity(a: list, b: list) -> float:
    dot = sum(x*y for x,y in zip(a,b))
    norm_a = sum(x*x for x in a) ** 0.5
    norm_b = sum(x*x for x in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0

def search(query: str, experiences: list, top_k: int = 5) -> list:
    query_emb = embed(query)
    scored = []
    for exp in experiences:
        sim = cosine_similarity(query_emb, exp["embedding"])
        scored.append((exp, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
```

### Weighted Semantic Search

Combines embedding similarity with quality weights.

```python
def weighted_search(query: str, experiences: list, top_k: int = 5) -> list:
    query_emb = embed(query)
    scored = []
    for exp in experiences:
        sim = cosine_similarity(query_emb, exp["embedding"])
        weight = exp.get("weight", 1.0)
        weighted_score = sim * weight  # Key difference
        scored.append((exp, weighted_score, sim))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]
```

### Hybrid Search (Keywords + Embeddings)

Combines exact matching with semantic similarity.

```python
def hybrid_search(query: str, experiences: list) -> list:
    # Keyword matches get boosted
    keywords = query.lower().split()
    query_emb = embed(query)

    scored = []
    for exp in experiences:
        sim = cosine_similarity(query_emb, exp["embedding"])

        # Keyword boost
        text = f"{exp['task']} {exp['lesson']}".lower()
        keyword_matches = sum(1 for kw in keywords if kw in text)
        boost = 1 + (keyword_matches * 0.1)

        scored.append((exp, sim * boost))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
```

---

## Weight/Quality Systems

### Simple Bounded Weights

Basic weight system with min/max bounds.

```python
INITIAL_WEIGHT = 1.0
MIN_WEIGHT = 0.1
MAX_WEIGHT = 2.0
INCREASE_RATE = 0.15
DECREASE_RATE = 0.10

def update_weight(exp: dict, success: bool) -> float:
    old_weight = exp.get("weight", INITIAL_WEIGHT)
    if success:
        new_weight = min(old_weight + INCREASE_RATE, MAX_WEIGHT)
    else:
        new_weight = max(old_weight - DECREASE_RATE, MIN_WEIGHT)
    exp["weight"] = new_weight
    return new_weight
```

### ELO-Style Rating

For competitive/comparative scenarios.

```python
def elo_update(winner_rating: float, loser_rating: float, k: float = 32) -> tuple:
    expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
    expected_loser = 1 - expected_winner

    new_winner = winner_rating + k * (1 - expected_winner)
    new_loser = loser_rating + k * (0 - expected_loser)
    return new_winner, new_loser
```

### Bayesian Confidence

Track uncertainty alongside quality.

```python
def bayesian_update(prior_success: int, prior_failure: int, outcome: bool) -> tuple:
    if outcome:
        return prior_success + 1, prior_failure
    else:
        return prior_success, prior_failure + 1

def confidence_score(successes: int, failures: int) -> float:
    # Beta distribution mean
    return (successes + 1) / (successes + failures + 2)
```

---

## Feedback Mechanisms

### Binary Feedback (Success/Failure)

Simplest feedback model.

```python
def handle_feedback(exp_ids: list, outcome: str, experiences: list):
    for exp in experiences:
        if exp["id"] in exp_ids:
            success = (outcome == "success")
            update_weight(exp, success)
            exp["use_count"] = exp.get("use_count", 0) + 1
            if success:
                exp["success_count"] = exp.get("success_count", 0) + 1
```

### Graded Feedback

More nuanced feedback with multiple levels.

```python
FEEDBACK_MULTIPLIERS = {
    "very_helpful": 0.20,
    "helpful": 0.10,
    "neutral": 0.0,
    "unhelpful": -0.05,
    "harmful": -0.15,
}

def graded_feedback(exp: dict, feedback_level: str):
    delta = FEEDBACK_MULTIPLIERS.get(feedback_level, 0)
    exp["weight"] = max(MIN_WEIGHT, min(MAX_WEIGHT, exp["weight"] + delta))
```

### Failure Experience Creation

Learn from misapplied experiences.

```python
def create_failure_experience(original_task: str, applied_exp_ids: list,
                              failure_reason: str) -> dict:
    return {
        "id": generate_id(),
        "task": f"[FAILURE CASE] {original_task}",
        "lesson": f"LESSON FROM FAILURE: {failure_reason}",
        "is_failure_experience": True,
        "weight": 0.8,  # Start lower than normal
        "related_experiences": applied_exp_ids,
    }
```

---

## Decay and Pruning

### Time-Based Decay

Reduce weights over time for unused experiences.

```python
from datetime import datetime, timedelta

def time_decay(experiences: list, half_life_days: int = 30):
    now = datetime.now()
    for exp in experiences:
        created = datetime.fromisoformat(exp["created_at"])
        age_days = (now - created).days
        decay_factor = 0.5 ** (age_days / half_life_days)
        exp["weight"] *= decay_factor
```

### Performance-Based Decay

Only decay underperforming experiences.

```python
def selective_decay(experiences: list, decay_factor: float = 0.95):
    for exp in experiences:
        use_count = exp.get("use_count", 0)
        success_count = exp.get("success_count", 0)

        # Only decay if poor performance or never used
        if use_count == 0 or (success_count / use_count) < 0.5:
            exp["weight"] = max(MIN_WEIGHT, exp["weight"] * decay_factor)
```

### Pruning Low-Quality Experiences

Remove experiences below threshold.

```python
def prune(experiences: list, min_weight: float = 0.2) -> list:
    return [exp for exp in experiences if exp.get("weight", 1.0) > min_weight]
```

---

## Common Data Models

### Minimal Experience Record

```json
{
  "id": "abc12345",
  "task": "Description of the original task",
  "lesson": "Actionable lesson learned",
  "embedding": [0.1, 0.2, ...],
  "weight": 1.0,
  "created_at": "2024-01-15T10:30:00"
}
```

### Full Experience Record

```json
{
  "id": "abc12345",
  "task": "Description of the original task",
  "category": "coding",
  "tags": ["python", "async", "database"],
  "failure_reason": "Why the initial approach failed",
  "lesson": "Actionable lesson for similar tasks",
  "context": "Project/environment context",
  "weight": 1.0,
  "embedding": [0.1, 0.2, ...],
  "created_at": "2024-01-15T10:30:00",
  "use_count": 5,
  "success_count": 4,
  "is_failure_experience": false
}
```

### Hierarchical Memory (Multi-Level)

```json
{
  "short_term": [
    {"task": "...", "lesson": "...", "ttl": 3600}
  ],
  "working_memory": [
    {"task": "...", "lesson": "...", "session_id": "xyz"}
  ],
  "long_term": [
    {"task": "...", "lesson": "...", "weight": 1.5, "embedding": [...]}
  ]
}
```
