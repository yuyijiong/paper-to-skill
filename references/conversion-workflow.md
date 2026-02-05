# Detailed Conversion Workflow

Step-by-step guide for converting AI agent papers into skills.

## Table of Contents

1. [Paper Reading Strategy](#paper-reading-strategy)
2. [Extracting Core Components](#extracting-core-components)
3. [Code Repository Analysis](#code-repository-analysis)
4. [Script Generation Guidelines](#script-generation-guidelines)
5. [SKILL.md Writing Guide](#skillmd-writing-guide)

---

## Paper Reading Strategy

### First Pass: Structure Scan (5 min)

1. Read **title** and **abstract** - identify the core contribution
2. Scan **section headings** - understand paper structure
3. Look at **figures and tables** - visual summaries of the approach
4. Read **conclusion** - what do authors claim as key results

### Second Pass: Algorithm Extraction (15-30 min)

Focus on:

1. **Methodology/Approach section** - core algorithm details
2. **Pseudocode blocks** - direct implementation guidance
3. **Mathematical formulas** - update rules, scoring functions
4. **Data structures** - what information is stored

Take notes on:
```
ALGORITHM: [Name]
INPUT: [What goes in]
OUTPUT: [What comes out]
STEPS:
1. [Step description]
2. [Step description]
...
KEY FORMULAS:
- [Formula 1]: [What it computes]
- [Formula 2]: [What it computes]
```

### Third Pass: Implementation Details (10-20 min)

Look for:

1. **Hyperparameters** - default values, ranges
2. **Embedding models** - what's used for vectors
3. **Thresholds** - minimum scores, cutoffs
4. **Edge cases** - what happens with empty data, failures

---

## Extracting Core Components

### For Memory/Learning Systems

Extract these components:

| Component | Where to Find | Maps To |
|-----------|---------------|---------|
| Memory schema | Data structures, figures | Script data model |
| Storage format | Implementation section | JSONL/SQLite structure |
| Retrieval method | Algorithm pseudocode | Search function |
| Update rules | Formulas, algorithm steps | Feedback handler |
| Decay mechanism | Time/quality management | Maintenance commands |

### Mapping Paper Sections to Skill Parts

```
Paper Section          -> Skill Component
-----------------------------------------
Abstract               -> SKILL.md description
Introduction           -> Background context (optional)
Methodology            -> references/algorithm.md
Algorithm pseudocode   -> scripts/*.py
Experiments            -> Validation approach
Hyperparameters        -> Script constants
```

### Identifying Workflow Phases

Most memory systems follow this pattern:

1. **Initialize** - Set up storage, load existing data
2. **Retrieve** - Find relevant memories for current task
3. **Apply** - Use memories to guide execution
4. **Capture** - Store new experiences
5. **Update** - Adjust quality based on outcome
6. **Maintain** - Decay, prune, cleanup

Map paper's approach to these phases.

---

## Code Repository Analysis

### Repository Structure Scan

```bash
# Find main entry points
find . -name "main.py" -o -name "run.py" -o -name "cli.py"

# Find core modules
find . -type f -name "*.py" | head -20

# Check dependencies
cat requirements.txt
cat setup.py
cat pyproject.toml
```

### Key Files to Examine

| File Pattern | Usually Contains |
|--------------|-----------------|
| `memory*.py`, `experience*.py` | Core memory operations |
| `embed*.py`, `vector*.py` | Embedding functions |
| `search*.py`, `retrieve*.py` | Retrieval logic |
| `update*.py`, `feedback*.py` | Weight update rules |
| `config*.py`, `constants*.py` | Hyperparameters |

### Code-to-Paper Mapping

Create a mapping table:

```
Paper Concept           Code Location              Function/Class
-----------------------------------------------------------------
Experience storage      src/memory.py:45           ExperienceDB
Embedding generation    src/embed.py:12            get_embedding()
Weighted retrieval      src/search.py:78           weighted_search()
Feedback processing     src/feedback.py:30         update_weights()
```

### Simplification Checklist

When adapting code for a skill:

- [ ] Remove training/evaluation code (keep inference only)
- [ ] Remove multi-GPU/distributed code
- [ ] Simplify to single-file when possible
- [ ] Add CLI interface with argparse
- [ ] Use JSON output for easy parsing
- [ ] Replace hardcoded paths with arguments
- [ ] Add `{SKILL_DIR}` support for portable paths

---

## Script Generation Guidelines

### CLI Structure Template

```python
#!/usr/bin/env python3
"""
[Skill Name] - [Brief description]

Usage:
    python script.py init [--data-dir PATH]
    python script.py add --input "..." [options]
    python script.py search --query "..." [options]
    python script.py feedback --ids "..." --outcome success|failure
"""
import argparse
import json
import sys

def cmd_init(args):
    """Initialize the system."""
    # Implementation
    print(json.dumps({"status": "initialized", "path": args.data_dir}))

def cmd_add(args):
    """Add new item."""
    # Implementation
    print(json.dumps({"status": "added", "id": new_id}))

def cmd_search(args):
    """Search for relevant items."""
    # Implementation
    print(json.dumps({"results": results}))

def cmd_feedback(args):
    """Process feedback."""
    # Implementation
    print(json.dumps({"updated": updated_ids}))

def main():
    parser = argparse.ArgumentParser(description="[Skill Name]")
    subparsers = parser.add_subparsers(dest="command")

    # Add subparsers for each command...

    args = parser.parse_args()
    # Route to command handlers...

if __name__ == "__main__":
    main()
```

### JSON Output Standards

Always output JSON for Claude to parse:

```python
# Success output
print(json.dumps({
    "status": "success",
    "data": {...}
}, indent=2))

# Error output
print(json.dumps({
    "status": "error",
    "message": "Description of what went wrong"
}), file=sys.stderr)
sys.exit(1)
```

### Embedding Integration

Standard pattern for OpenAI embeddings:

```python
def get_embedding(text: str) -> list:
    """Get embedding using OpenAI API."""
    try:
        from openai import OpenAI
        client = OpenAI()  # Uses OPENAI_API_KEY env var
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except ImportError:
        print(json.dumps({"error": "openai package not installed"}))
        sys.exit(1)
```

---

## SKILL.md Writing Guide

### Frontmatter Best Practices

```yaml
---
name: skill-name  # lowercase, hyphenated
description: >
  [What it does - 1 sentence]. [Key capabilities - 1-2 sentences].
  [When to trigger - be specific]. Use when [trigger 1], [trigger 2],
  or [trigger 3].
---
```

**Good description example:**
```yaml
description: >
  Self-evolving memory system for experience-based learning. Stores task
  experiences with quality weights, retrieves relevant lessons via semantic
  search, updates weights based on feedback. Use when implementing experience
  memory, building learning agents, or adding memory to existing workflows.
```

**Bad description example:**
```yaml
description: Memory system for agents.  # Too vague, no triggers
```

### Body Structure

```markdown
# [Skill Name]

[One-line summary]

## Prerequisites

[Required packages, API keys]

## Setup

[Initialization commands with examples]

## Workflow

[Main usage pattern with clear phases]

### Phase 1: [Name]
[What to do, when, commands]

### Phase 2: [Name]
[What to do, when, commands]

## Commands Reference

### [command-name]
[Description and example]

## [Domain-Specific Section]

[Patterns, tips, common issues]

## Algorithm Details

For full details, see [references/algorithm.md](references/algorithm.md).
```

### Command Documentation Pattern

```markdown
### search

Search for relevant experiences.

```bash
python {SKILL_DIR}/scripts/memory_manager.py search \
  --query "task description" \
  --top-k 5 \
  --threshold 0.3 \
  --data-dir .claude/memory
```

**Parameters:**
- `--query`: Search query (required)
- `--top-k`: Number of results (default: 5)
- `--threshold`: Minimum score (default: 0.25)

**Output:**
```json
{
  "results": [
    {"id": "abc123", "score": 0.85, "task": "...", "lesson": "..."}
  ]
}
```
```

### Keep Instructions Action-Oriented

**Good:**
```markdown
### Phase 1: Retrieve

Before starting work, search for relevant experiences:

\`\`\`bash
python {SKILL_DIR}/scripts/memory_manager.py search --query "<task>"
\`\`\`

If results found, synthesize into 3-5 guidelines.
```

**Bad:**
```markdown
### Phase 1: Retrieve

The retrieval phase is an important part of the workflow where you
search for experiences that might be relevant to your current task.
This helps ensure that past lessons are applied...
```

(Too verbose, not action-oriented)
