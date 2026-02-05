---
name: live-evo-memory
description: >
  Self-evolving memory system for Claude Code based on the Live-Evo algorithm. Stores task
  experiences with dynamic quality weights, retrieves relevant lessons via embedding-based
  semantic search, and synthesizes them into task-specific guidelines. Experiences that help
  gain weight; those that hurt lose weight and generate failure lessons. This skill is
  ALWAYS ACTIVE: before every non-trivial task, search memory for relevant experiences and
  apply them as guidelines. After task completion or failure, capture lessons learned. Use
  this skill for all tasks - coding, debugging, configuration, deployment, refactoring, etc.
  Trigger: every task interaction in a project that has .claude/memory/ initialized.
---

# Live-Evo Memory

Self-evolving experience memory. Before each task, retrieve and apply past lessons. After each task, capture new lessons. Experiences gain or lose weight based on outcomes.

## Prerequisites

Requires `openai` Python package and `OPENAI_API_KEY` environment variable for embeddings.

```bash
pip install openai
```

## Setup

Initialize memory for the current project:

```bash
python {SKILL_DIR}/scripts/memory_manager.py init --memory-dir .claude/memory
```

This creates `.claude/memory/experience_db.jsonl` and `.claude/memory/weight_history.jsonl`.

## Always-On Workflow

Follow this workflow for EVERY non-trivial task. Skip only for single-line fixes or trivial changes.

### Phase 1: Retrieve (Before Task)

Search memory for relevant experiences before starting work.

```bash
python {SKILL_DIR}/scripts/memory_manager.py search --query "<task description>" --memory-dir .claude/memory
```

The script returns JSON with results ranked by `weighted_score = similarity * weight`. If results are found:

1. Read the returned experiences (task, lesson, failure_reason, weight, is_failure_experience)
2. Synthesize into 3-5 actionable guidelines for the current task:
   - Extract **methodological** lessons (how to approach), not specific conclusions
   - Flag task-type mismatches: if a past experience is from a different domain/context, note reduced applicability
   - Pay special attention to failure experiences (`is_failure_experience: true`) - these warn about common misapplications
3. Apply guidelines during task execution, but trust fresh research over stale patterns

If no results or the DB is empty, proceed normally.

### Phase 2: Execute (During Task)

Execute the task, keeping retrieved guidelines in mind. Note:
- Which guidelines were applied
- Which experience IDs were used (from the search results)
- Whether the approach succeeded or required changes

### Phase 3: Capture (After Task)

After task completion, determine if a lesson should be saved. Save an experience when:
- A non-obvious approach was needed
- A mistake was made and corrected
- An initial approach failed before finding the right one
- A tricky edge case or environment-specific issue was encountered
- Knowledge was gained that would help with similar future tasks

```bash
python {SKILL_DIR}/scripts/memory_manager.py add \
  --task "Brief description of the task" \
  --lesson "Actionable lesson for similar future tasks" \
  --failure-reason "What went wrong (if applicable)" \
  --tags "python,async,database" \
  --category "coding" \
  --memory-dir .claude/memory
```

**Do NOT save experiences for:**
- Routine tasks that went smoothly with no new insights
- Trivial fixes (typos, simple syntax errors)
- Lessons that are general programming knowledge

### Phase 4: Feedback (After Outcome Known)

When experiences were used (Phase 1 returned results) and the outcome is clear:

```bash
# If the retrieved experiences helped:
python {SKILL_DIR}/scripts/memory_manager.py feedback --ids "id1,id2" --outcome success --memory-dir .claude/memory

# If the retrieved experiences did not help or hurt:
python {SKILL_DIR}/scripts/memory_manager.py feedback --ids "id1,id2" --outcome failure --memory-dir .claude/memory
```

On failure feedback, also capture a failure experience explaining WHY the past experience misapplied:

```bash
python {SKILL_DIR}/scripts/memory_manager.py add \
  --task "[FAILURE CASE] Original task description" \
  --lesson "LESSON FROM FAILURE: Why the past experience did not apply here..." \
  --failure-reason "Task-type mismatch / over-generalization / context difference" \
  --is-failure \
  --memory-dir .claude/memory
```

## Guideline Synthesis Rules

When synthesizing retrieved experiences into guidelines:

1. **Applicability check**: Assess whether each experience matches the current task type. A debugging lesson may not apply to a deployment task.
2. **Weight as confidence**: Higher-weight experiences (>1.2) have proven track records. Lower-weight (<0.8) may be unreliable.
3. **Failure experiences**: These carry warnings. If a failure experience is retrieved, explicitly note what NOT to do.
4. **Concise bullets**: Produce 3-5 bullet points. Each should be specific and actionable, not generic advice.
5. **Methodology over conclusions**: Extract the approach (e.g., "check for circular imports first") not the specific fix.

## Maintenance Commands

### List all experiences
```bash
python {SKILL_DIR}/scripts/memory_manager.py list --sort weight --memory-dir .claude/memory
```

### View statistics
```bash
python {SKILL_DIR}/scripts/memory_manager.py stats --memory-dir .claude/memory
```

### Prune low-quality experiences
```bash
# Preview what would be removed
python {SKILL_DIR}/scripts/memory_manager.py prune --min-weight 0.2 --dry-run --memory-dir .claude/memory

# Actually remove
python {SKILL_DIR}/scripts/memory_manager.py prune --min-weight 0.2 --memory-dir .claude/memory
```

### Apply weight decay
Run periodically (e.g., weekly or when the DB grows past 50 experiences) to fade out stale, unproven experiences:
```bash
python {SKILL_DIR}/scripts/memory_manager.py decay --factor 0.95 --memory-dir .claude/memory
```

### Export for review
```bash
python {SKILL_DIR}/scripts/memory_manager.py export --format json --memory-dir .claude/memory
```

## Categories

Use consistent category values: `coding`, `debugging`, `config`, `deployment`, `refactoring`, `testing`, `performance`, `security`, `documentation`, `general`.

## Algorithm Details

For the full weight system, retrieval algorithm, failure learning, and decay mechanics, see [references/algorithm.md](references/algorithm.md).
