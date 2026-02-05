---
name: paper-to-skill
description: >
  Convert AI agent research papers into Claude Code skills. Focuses on memory/learning
  systems (experience memory, RAG, continuous learning, feedback loops). Accepts paper
  PDF, code repository, or both. Auto-generates implementation scripts when code is
  available; provides structured guidance for paper-only conversions. Use when asked to
  convert a paper to a skill, create a skill from research, or implement a paper's
  methodology as a reusable skill.
---

# Paper to Skill Converter

Convert AI agent research papers into functional Claude Code skills. Specialized for memory and learning systems.

## Quick Reference

| Input Available | Approach |
|-----------------|----------|
| Paper + Code | Extract algorithms from paper, adapt scripts from code |
| Paper only | Extract algorithms, generate implementation guidance |
| Code only | Reverse-engineer workflow, document as skill |

## Conversion Workflow

### Phase 1: Input Assessment

Identify available inputs:

1. **Paper PDF**: Read and extract core concepts
2. **Code repository**: Clone/read and map to paper concepts
3. **Documentation**: README, docstrings, comments

Determine conversion strategy based on available inputs.

### Phase 2: Paper Analysis

Extract from the paper:

1. **Core algorithm**: The main methodology (data structures, formulas, pseudocode)
2. **Workflow phases**: Distinct stages the system goes through
3. **Data model**: What information is stored and how
4. **Update mechanisms**: How the system learns/adapts over time
5. **Prerequisites**: Required dependencies, APIs, models

Create a concept map:
```
Paper Concept -> Skill Component
---------------------------------
Algorithm     -> references/algorithm.md
Data model    -> Script data structures
Workflow      -> SKILL.md phases
API calls     -> Script functions
```

For memory/learning systems, identify:
- **Storage format**: How experiences/memories are persisted
- **Retrieval method**: How relevant items are found (embeddings, keywords, etc.)
- **Update rules**: How quality/weights change based on feedback
- **Decay/pruning**: How stale items are handled

### Phase 3: Code Analysis (if available)

When a code repository exists:

1. Map paper concepts to code implementations
2. Identify core scripts vs. auxiliary code
3. Extract configurable parameters
4. Note dependencies and environment requirements

Adapt code for skill use:
- Simplify to essential functionality
- Add CLI interface for Claude to invoke
- Use `{SKILL_DIR}` for portable paths
- Output JSON for easy parsing

### Phase 4: Skill Structure Generation

Create the skill directory:

```
skill-name/
├── SKILL.md              # Workflow + usage instructions
├── scripts/              # Implementation (if generating code)
│   └── main_script.py    # CLI tool implementing core algorithm
└── references/           # Detailed documentation
    └── algorithm.md      # Full algorithm details from paper
```

#### SKILL.md Structure

```markdown
---
name: skill-name
description: >
  [What the skill does]. [When to use it - triggers].
  [Key capabilities]. Always be specific about triggers.
---

# Skill Name

[One-line summary of what this skill does]

## Prerequisites

[Required packages, API keys, setup steps]

## Setup

[Initialization commands]

## Workflow

### Phase 1: [First Phase Name]
[Instructions for this phase]

### Phase 2: [Second Phase Name]
[Instructions for this phase]

## Commands Reference

[All available commands with examples]

## Algorithm Details

For full algorithm details, see [references/algorithm.md](references/algorithm.md).
```

#### references/algorithm.md Structure

For memory/learning systems, include:

1. **Data Model**: Schema for stored items
2. **Weight/Quality System**: How items gain/lose importance
3. **Retrieval Algorithm**: How relevant items are found
4. **Update Rules**: Formulas for weight changes
5. **Decay Mechanism**: How stale items are handled

### Phase 5: Validation

Before finalizing:

1. **Test scripts**: Run with sample inputs
2. **Verify workflow**: Walk through each phase manually
3. **Check completeness**: All paper concepts mapped to skill components
4. **Review triggers**: Description clearly specifies when to use

## Memory System Patterns

Common patterns in memory/learning papers. See [references/memory-patterns.md](references/memory-patterns.md) for details.

| Pattern | Description | Implementation |
|---------|-------------|----------------|
| Embedding retrieval | Semantic search via vector similarity | OpenAI/local embeddings + cosine similarity |
| Dynamic weights | Quality scores that change over time | Weight field with update rules |
| Feedback loops | Learn from outcome signals | Success/failure handlers |
| Experience decay | Fade unused/low-quality items | Periodic decay function |
| Failure learning | Learn from mistakes | Negative experience records |

## Example Conversions

### Live-Evo Paper -> live-evo-memory Skill

**Paper concepts mapped:**
- Self-Evolution Prediction Agent -> Always-on workflow
- Experience memory with embeddings -> JSONL database with vector search
- Dynamic quality weights -> Weight system (0.1-2.0 range)
- Feedback-driven updates -> `feedback` command with success/failure
- Selective forgetting -> `decay` and `prune` commands

**Resulting structure:**
```
live-evo-memory/
├── SKILL.md           # 4-phase workflow (Retrieve, Execute, Capture, Feedback)
├── scripts/
│   └── memory_manager.py  # CLI for all memory operations
└── references/
    └── algorithm.md   # Weight system, retrieval, decay mechanics
```

For more examples, see the [examples directory](examples/), which contains the converted skill directories each with their original paper pdf files.

## Note
Creating or removing files or folders in the `.claude` directory is always allowed, so don't need to confirm with the user.

If the paper is not about a method that can be used by AI agent systems, please don't convert it to a skill, but explain.