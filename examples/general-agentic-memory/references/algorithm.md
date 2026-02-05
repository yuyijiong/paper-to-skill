# GAM Algorithm Reference

Detailed algorithm documentation from the General Agentic Memory paper and implementation.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [MemoryAgent: Abstraction](#memoryagent-abstraction)
3. [ResearchAgent: Deep Research Loop](#researchagent-deep-research-loop)
4. [Prompts](#prompts)
5. [Data Model](#data-model)

---

## Architecture Overview

GAM implements Just-in-Time (JIT) memory optimization through a dual-agent architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                      MEMORY CONSTRUCTION                     │
│  Documents → MemoryAgent → Abstracts (MemoryStore)          │
│                         → Pages (PageStore)                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                      QUERY-TIME RESEARCH                     │
│  Query → ResearchAgent ──┐                                   │
│           ↓              │                                   │
│     1. Planning    ←─────┤                                   │
│           ↓              │                                   │
│     2. Search      ←─────┤  (iterate until enough)           │
│           ↓              │                                   │
│     3. Integrate   ←─────┤                                   │
│           ↓              │                                   │
│     4. Reflect     ──────┘                                   │
│           ↓                                                  │
│        Result                                                │
└─────────────────────────────────────────────────────────────┘
```

### Key Difference from AOT Systems

| Ahead-of-Time (AOT) | Just-in-Time (JIT) |
|---------------------|--------------------|
| Pre-process documents into facts | Store full pages with abstracts |
| Fixed structure at indexing | Dynamic research at query time |
| May lose context | Preserves full fidelity |
| Fast retrieval | Deeper understanding |

---

## MemoryAgent: Abstraction

The MemoryAgent generates abstracts from documents and stores both the abstract and original content.

### Workflow

```python
def memorize(message: str) -> MemoryUpdate:
    # 1. Load current memory state
    state = memory_store.load()

    # 2. Generate abstract with context
    abstract, header, decorated_page = _decorate(message, state)

    # 3. Add abstract to memory (deduplicated)
    memory_store.add(abstract)

    # 4. Store page with header
    page = Page(header=header, content=message, meta={...})
    page_store.add(page)

    return MemoryUpdate(new_state=..., new_page=...)
```

### Memory Prompt

```
You are the MemoryAgent. Write one concise abstract for long-term memory.

MEMORY_CONTEXT:
{existing abstracts as "Page 0: ...", "Page 1: ...", etc.}

INPUT_MESSAGE:
{new document}

TASK:
1. Extract memory-relevant information:
   - plans, goals, decisions, preferences
   - actions, assignments, responsibilities
   - problems, blockers, questions
   - specific facts (names, dates, numbers)

2. Use MEMORY_CONTEXT for:
   - resolve entities and terminology
   - keep naming consistent

3. Output ONE concise paragraph. No bullets. No meta phrases.
```

---

## ResearchAgent: Deep Research Loop

The ResearchAgent performs iterative retrieval-reflection at query time.

### Main Loop

```python
def research(request: str) -> ResearchOutput:
    result = Result()
    next_request = request

    for step in range(max_iters):
        # Load current memory state
        memory_state = memory_store.load()

        # 1. Planning
        plan = _planning(next_request, memory_state)

        # 2. Search + Integrate
        result = _search(plan, result, request)

        # 3. Reflection
        decision = _reflection(request, result)

        # Check if done
        if decision.enough:
            break

        # Generate follow-up
        next_request = decision.new_request or request

    return ResearchOutput(integrated_memory=result.content, ...)
```

### Planning Phase

```python
def _planning(request: str, memory_state: MemoryState) -> SearchPlan:
    # Build memory context from abstracts
    memory_context = "\n".join([
        f"Page {i}: {abstract}"
        for i, abstract in enumerate(memory_state.abstracts)
    ])

    # LLM generates search plan
    response = generator.generate(Planning_PROMPT.format(
        request=request,
        memory=memory_context
    ))

    return SearchPlan(
        info_needs=["What is X?", "How does Y work?"],
        tools=["keyword", "vector"],
        keyword_collection=["X", "Y component"],
        vector_queries=["How does X relate to Y?"],
        page_index=[0, 2]  # Known relevant pages
    )
```

### Search Tools

| Tool | What It Does | When to Use |
|------|--------------|-------------|
| `keyword` | BM25 exact match | Entity names, function names, specific attributes |
| `vector` | Semantic similarity | Conceptual questions, "how/why" queries |
| `page_index` | Direct page access | When memory context points to specific pages |

### Integration Phase

After search, results are integrated:

```python
def _integrate(hits: List[Hit], result: Result, question: str) -> Result:
    # Format evidence
    evidence = "\n".join([
        f"{i}. [{hit.source}]({hit.page_id}) {hit.snippet}"
        for i, hit in enumerate(hits, 1)
    ])

    # LLM synthesizes
    response = generator.generate(Integrate_PROMPT.format(
        question=question,
        evidence_context=evidence,
        result=result.content
    ))

    return Result(content=response["content"], sources=response["sources"])
```

### Reflection Phase

Two-step reflection:

```python
def _reflection(request: str, result: Result) -> ReflectionDecision:
    # Step 1: Check completeness
    check = generator.generate(InfoCheck_PROMPT.format(
        request=request,
        result=result.content
    ))

    if check["enough"]:
        return ReflectionDecision(enough=True, new_request=None)

    # Step 2: Generate follow-up questions
    follow_up = generator.generate(GenerateRequests_PROMPT.format(
        request=request,
        result=result.content
    ))

    return ReflectionDecision(
        enough=False,
        new_request=" ".join(follow_up["new_requests"])
    )
```

---

## Prompts

### Planning Prompt

```
You are the PlanningAgent. Generate a retrieval plan for the QUESTION.

QUESTION: {request}
MEMORY: {memory context}

PLANNING PROCEDURE:
1. Identify what info is needed
2. Break into concrete sub-questions
3. For each, decide which tools:
   - "keyword" for exact entities
   - "vector" for conceptual understanding
   - "page_index" if memory points to relevant pages
4. Build plan with:
   - info_needs: list of sub-questions
   - tools: which tools to use
   - keyword_collection: keyword queries
   - vector_queries: semantic queries
   - page_index: page numbers to read

OUTPUT JSON:
{
  "info_needs": [...],
  "tools": ["keyword", "vector"],
  "keyword_collection": [...],
  "vector_queries": [...],
  "page_index": [...]
}
```

### Integration Prompt

```
You are the IntegrateAgent. Build a factual summary for QUESTION.

QUESTION: {question}
EVIDENCE_CONTEXT: {retrieved hits}
RESULT: {current summary}

TASK:
1. Keep useful info from RESULT
2. Add relevant facts from EVIDENCE_CONTEXT
3. Remove off-topic content
4. Produce merged factual summary

OUTPUT JSON:
{
  "content": "...",
  "sources": [...]
}
```

### Info Check Prompt

```
You are the InfoCheckAgent. Judge if RESULT has enough info to answer REQUEST.

REQUEST: {request}
RESULT: {current summary}

PROCEDURE:
1. Decompose REQUEST into required info pieces
2. Check if RESULT covers each piece
3. "enough" = true only if all covered

OUTPUT JSON:
{
  "enough": true/false
}
```

### Generate Requests Prompt

```
You are the FollowUpRequestAgent. Propose follow-up questions.

REQUEST: {original question}
RESULT: {current summary}

TASK:
1. Identify what's still missing
2. Generate 1-5 retrieval questions to fill gaps
3. Rank by importance

OUTPUT JSON:
{
  "new_requests": ["...", "..."]
}
```

---

## Data Model

### MemoryState

```python
class MemoryState(BaseModel):
    abstracts: List[str]  # List of memory abstracts
```

### Page

```python
class Page(BaseModel):
    header: str           # "[ABSTRACT] ..."
    content: str          # Original document
    meta: Dict[str, Any]  # Metadata including decorated version
```

### SearchPlan

```python
class SearchPlan(BaseModel):
    info_needs: List[str]        # Sub-questions to answer
    tools: List[str]             # ["keyword", "vector", "page_index"]
    keyword_collection: List[str] # Keyword queries
    vector_queries: List[str]     # Semantic queries
    page_index: List[int]         # Page IDs to fetch
```

### Hit

```python
class Hit(BaseModel):
    page_id: str          # Page identifier
    snippet: str          # Matched content
    source: str           # "keyword", "vector", "page_index"
    meta: Dict[str, Any]  # Score and other metadata
```

### Result

```python
class Result(BaseModel):
    content: str          # Integrated summary
    sources: List[str]    # Page IDs used
```

### ResearchOutput

```python
class ResearchOutput(BaseModel):
    integrated_memory: str           # Final answer
    raw_memory: Dict[str, Any]       # Debug info with iterations
```

### ReflectionDecision

```python
class ReflectionDecision(BaseModel):
    enough: bool                     # Is info sufficient?
    new_request: Optional[str]       # Follow-up query if not enough
```

---

## Comparison with Other Systems

| Feature | GAM | LightMem | Live-Evo |
|---------|-----|----------|----------|
| **Memory Type** | Abstracts + Pages | Facts | Experiences |
| **Query-Time Processing** | Deep research loop | Direct retrieval | Weighted retrieval |
| **Iteration** | Yes (reflection) | No | No |
| **Multi-hop Reasoning** | Yes | No | No |
| **Storage** | JSON files | Qdrant + JSONL | JSONL |
| **Retrieval** | BM25 + Dense + Index | Embedding | Embedding |
