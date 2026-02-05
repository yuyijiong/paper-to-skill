# Live-Evo Algorithm Reference

This documents the core algorithm from the Live-Evo paper (Self-Evolution Prediction Agent), adapted for general-purpose Claude Code memory management.

## Table of Contents

1. [Experience Data Model](#experience-data-model)
2. [Weight System](#weight-system)
3. [Retrieval: Weighted Semantic Search](#retrieval-weighted-semantic-search)
4. [Active Exploration](#active-exploration)
5. [Guideline Synthesis](#guideline-synthesis)
6. [Weight Update Rules](#weight-update-rules)
7. [Failure Experience Creation](#failure-experience-creation)
8. [Weight Decay (Forgetting)](#weight-decay-forgetting)
9. [Experience Refinement](#experience-refinement)

---

## Experience Data Model

Each experience record:

```json
{
  "id": "8-char UUID",
  "task": "Description of the original task",
  "category": "coding|debugging|config|deployment|refactoring|general|...",
  "tags": ["python", "async", "database"],
  "failure_reason": "Why the approach failed",
  "lesson": "Actionable improvement for similar tasks",
  "context": "Project/environment context",
  "weight": 1.0,
  "embedding": [0.1, 0.2, ...],
  "created_at": "ISO timestamp",
  "use_count": 0,
  "success_count": 0,
  "is_failure_experience": false
}
```

## Weight System

Experiences carry a quality weight reflecting historical usefulness.

| Parameter | Value |
|-----------|-------|
| Initial weight | 1.0 |
| Min weight | 0.1 |
| Max weight | 2.0 |
| Increase rate (per success) | +0.15 |
| Decrease rate (per failure) | -0.10 |
| Decay factor | 0.95 per cycle |

Weight determines retrieval priority: `weighted_score = cosine_similarity * weight`.

High-weight experiences surface first; low-weight experiences fade out over time.

## Retrieval: Weighted Semantic Search

1. Embed the query using `text-embedding-3-small`
2. Compute cosine similarity between query embedding and each experience's embedding
3. Multiply similarity by experience weight: `weighted_score = similarity * weight`
4. Filter by threshold (default 0.25)
5. Return top-K results (default 5), sorted by weighted_score descending

This ensures high-quality experiences (proven helpful) rank above mediocre ones even with slightly lower raw similarity.

## Active Exploration

Instead of simple keyword matching, generate targeted search queries:

1. Analyze the current task to determine what kind of past experience might help
2. Generate 2-3 search queries with different angles:
   - **Task-similar**: Find experiences from similar tasks
   - **Lesson-similar**: Find experiences with relevant lessons regardless of task similarity
3. Execute each query against the experience DB
4. Deduplicate results by experience ID (keep highest score)
5. Return top-5 unique experiences

## Guideline Synthesis

Raw experiences are not injected directly. Instead, synthesize them into focused, task-specific guidelines:

1. Retrieve top relevant experiences
2. For each experience, note: task context, lesson, failure reason, weight, match type
3. Synthesize into 3-5 concise, actionable bullet points tailored to the current task
4. Include applicability warnings:
   - Flag task-type mismatches between past experience and current task
   - Warn against over-generalizing from different contexts
   - Distinguish methodological lessons (HOW to analyze) from conclusions (WHAT to conclude)

## Weight Update Rules

After task completion, update weights based on outcome:

**Success (experience helped):**
```
new_weight = min(old_weight + INCREASE_RATE, MAX_WEIGHT)
success_count += 1
```

**Failure (experience did not help or hurt):**
```
new_weight = max(old_weight - DECREASE_RATE, MIN_WEIGHT)
```

Additionally, create a failure experience capturing WHY the experience misapplied (see below).

## Failure Experience Creation

When applying past experiences leads to a worse outcome, create a special "failure experience":

1. Record the original task and which experiences were applied
2. Analyze WHY the experiences hurt:
   - Task-type mismatch?
   - Over-generalization?
   - Context difference?
3. Store as a new experience with:
   - `is_failure_experience: true`
   - `weight: 0.8` (starts lower)
   - Lesson captures the meta-lesson about misapplication
4. Prefix the task with `[FAILURE CASE]` for identification

This enables the system to learn not just from failures, but from failures-of-learning.

## Weight Decay (Forgetting)

Periodically apply decay to prevent stale experiences from dominating:

```
For each experience where success_rate < 50% or never used:
    new_weight = max(old_weight * DECAY_FACTOR, MIN_WEIGHT)
```

Decay is selective: frequently successful experiences are immune to decay.

Run decay when the experience DB grows large or after extended periods.

## Experience Refinement

When an initial lesson from a failure analysis isn't specific enough:

1. Test the experience on the same or similar task
2. If insufficient improvement, refine the experience:
   - Analyze why the previous lesson was too vague/generic
   - Generate a more specific, actionable lesson
   - Replace the old lesson with the refined version
3. Repeat up to 3 iterations
4. Validate that the refined experience provides measurable improvement

This iterative refinement ensures experience quality improves over time.
