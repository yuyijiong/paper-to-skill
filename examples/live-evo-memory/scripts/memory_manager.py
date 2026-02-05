#!/usr/bin/env python3
"""
Memory Manager for Live-Evo Memory Skill.

Manages experience database with embedding-based retrieval and dynamic weight system.
Adapted from Live-Evo (Self-Evolution Prediction Agent) for general Claude Code tasks.

Usage:
    python memory_manager.py init [--memory-dir PATH]
    python memory_manager.py add --task "..." --lesson "..." [--failure-reason "..."] [--tags "t1,t2"] [--category "..."]
    python memory_manager.py search --query "..." [--top-k N] [--threshold F]
    python memory_manager.py update-weight --id ID --delta FLOAT [--reason "..."]
    python memory_manager.py feedback --ids "id1,id2" --outcome success|failure
    python memory_manager.py list [--category CAT] [--sort weight|date|use_count]
    python memory_manager.py stats
    python memory_manager.py prune [--min-weight FLOAT] [--dry-run]
    python memory_manager.py decay [--factor FLOAT]
    python memory_manager.py export [--format json|jsonl]
"""
import sys
import os
import json
import argparse
import uuid
import pickle
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INITIAL_WEIGHT = 1.0
MIN_WEIGHT = 0.1
MAX_WEIGHT = 2.0
WEIGHT_INCREASE_RATE = 0.15
WEIGHT_DECREASE_RATE = 0.10
DEFAULT_DECAY = 0.95
DEFAULT_THRESHOLD = 0.25
DEFAULT_TOP_K = 5

DEFAULT_MEMORY_DIR = os.path.join(".claude", "memory")

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _get_openai_client():
    """Get OpenAI client. Requires OPENAI_API_KEY env var."""
    try:
        from openai import OpenAI
        return OpenAI()
    except ImportError:
        logger.error("openai package not installed. Run: pip install openai")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to create OpenAI client: {e}")
        sys.exit(1)


def _embed_text(text: str, client=None, cache: dict = None, cache_path: str = None) -> list:
    """Embed text using OpenAI text-embedding-3-small. Returns list of floats."""
    if cache and text in cache:
        return cache[text]

    if client is None:
        client = _get_openai_client()

    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    embedding = response.data[0].embedding

    if cache is not None:
        cache[text] = embedding
        if cache_path:
            _save_embedding_cache(cache, cache_path)

    return embedding


def _cosine_similarity(a: list, b: list) -> float:
    """Cosine similarity between two vectors."""
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _load_embedding_cache(cache_path: str) -> dict:
    """Load embedding cache from pickle file."""
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return {}
    return {}


def _save_embedding_cache(cache: dict, cache_path: str):
    """Save embedding cache to pickle file."""
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
    except Exception as e:
        logger.warning(f"Failed to save embedding cache: {e}")

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _db_path(memory_dir: str) -> str:
    return os.path.join(memory_dir, "experience_db.jsonl")


def _weight_history_path(memory_dir: str) -> str:
    return os.path.join(memory_dir, "weight_history.jsonl")


def _embedding_cache_path(memory_dir: str) -> str:
    return os.path.join(memory_dir, "embedding_cache.pkl")


def _load_experiences(memory_dir: str) -> List[Dict]:
    path = _db_path(memory_dir)
    experiences = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        experiences.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return experiences


def _save_experiences(experiences: List[Dict], memory_dir: str):
    path = _db_path(memory_dir)
    with open(path, "w", encoding="utf-8") as f:
        for exp in experiences:
            f.write(json.dumps(exp, default=str) + "\n")


def _append_experience(exp: Dict, memory_dir: str):
    path = _db_path(memory_dir)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(exp, default=str) + "\n")


def _log_weight_change(memory_dir: str, exp_id: str, old_w: float, new_w: float, reason: str):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "experience_id": exp_id,
        "old_weight": old_w,
        "new_weight": new_w,
        "reason": reason,
    }
    path = _weight_history_path(memory_dir)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_init(args):
    """Initialize memory directory."""
    memory_dir = args.memory_dir
    os.makedirs(memory_dir, exist_ok=True)

    db = _db_path(memory_dir)
    if not os.path.exists(db):
        Path(db).touch()

    wh = _weight_history_path(memory_dir)
    if not os.path.exists(wh):
        Path(wh).touch()

    print(f"Memory initialized at: {os.path.abspath(memory_dir)}")
    print(f"  experience_db.jsonl: {db}")
    print(f"  weight_history.jsonl: {wh}")


def cmd_add(args):
    """Add a new experience to the database."""
    memory_dir = args.memory_dir

    # Ensure memory dir exists
    if not os.path.exists(_db_path(memory_dir)):
        os.makedirs(memory_dir, exist_ok=True)
        Path(_db_path(memory_dir)).touch()

    tags = [t.strip() for t in args.tags.split(",")] if args.tags else []

    exp = {
        "id": str(uuid.uuid4())[:8],
        "task": args.task,
        "category": args.category or "general",
        "tags": tags,
        "failure_reason": args.failure_reason or "",
        "lesson": args.lesson,
        "context": args.context or "",
        "weight": INITIAL_WEIGHT,
        "created_at": datetime.now().isoformat(),
        "use_count": 0,
        "success_count": 0,
        "is_failure_experience": bool(args.is_failure),
    }

    # Compute embedding for the task + lesson combined text
    embed_text = f"{exp['task']} {exp['lesson']}"
    cache_path = _embedding_cache_path(memory_dir)
    cache = _load_embedding_cache(cache_path)
    client = _get_openai_client()

    try:
        exp["embedding"] = _embed_text(embed_text, client=client, cache=cache, cache_path=cache_path)
    except Exception as e:
        logger.warning(f"Failed to compute embedding: {e}. Storing without embedding.")
        exp["embedding"] = []

    _append_experience(exp, memory_dir)
    print(json.dumps({
        "status": "added",
        "id": exp["id"],
        "task": exp["task"],
        "weight": exp["weight"],
    }, indent=2))


def cmd_search(args):
    """Search for relevant experiences."""
    memory_dir = args.memory_dir
    experiences = _load_experiences(memory_dir)

    if not experiences:
        print(json.dumps({"results": [], "message": "No experiences in database"}))
        return

    cache_path = _embedding_cache_path(memory_dir)
    cache = _load_embedding_cache(cache_path)
    client = _get_openai_client()

    query_embedding = _embed_text(args.query, client=client, cache=cache, cache_path=cache_path)

    # Score each experience
    scored = []
    for exp in experiences:
        exp_emb = exp.get("embedding", [])
        if not exp_emb:
            continue
        sim = _cosine_similarity(query_embedding, exp_emb)
        weight = exp.get("weight", INITIAL_WEIGHT)
        weighted_score = sim * weight
        scored.append((exp, weighted_score, sim))

    # Sort by weighted score
    scored.sort(key=lambda x: x[1], reverse=True)

    threshold = args.threshold
    top_k = args.top_k
    results = []
    for exp, w_score, sim in scored:
        if w_score < threshold:
            continue
        results.append({
            "id": exp["id"],
            "task": exp["task"],
            "lesson": exp["lesson"],
            "failure_reason": exp.get("failure_reason", ""),
            "category": exp.get("category", ""),
            "tags": exp.get("tags", []),
            "weight": exp.get("weight", 1.0),
            "weighted_score": round(w_score, 4),
            "similarity": round(sim, 4),
            "use_count": exp.get("use_count", 0),
            "success_count": exp.get("success_count", 0),
            "is_failure_experience": exp.get("is_failure_experience", False),
        })
        if len(results) >= top_k:
            break

    # Update use_count for returned results
    if results:
        result_ids = {r["id"] for r in results}
        for exp in experiences:
            if exp["id"] in result_ids:
                exp["use_count"] = exp.get("use_count", 0) + 1
        _save_experiences(experiences, memory_dir)

    print(json.dumps({"results": results, "total_experiences": len(experiences)}, indent=2))


def cmd_feedback(args):
    """Update weights based on outcome feedback (success/failure)."""
    memory_dir = args.memory_dir
    experiences = _load_experiences(memory_dir)

    ids = [i.strip() for i in args.ids.split(",")]
    outcome = args.outcome  # "success" or "failure"

    updates = []
    for exp in experiences:
        if exp["id"] in ids:
            old_w = exp.get("weight", INITIAL_WEIGHT)
            if outcome == "success":
                new_w = min(old_w + WEIGHT_INCREASE_RATE, MAX_WEIGHT)
                exp["success_count"] = exp.get("success_count", 0) + 1
                reason = "positive feedback: experience helped"
            else:
                new_w = max(old_w - WEIGHT_DECREASE_RATE, MIN_WEIGHT)
                reason = "negative feedback: experience did not help"
            exp["weight"] = new_w
            _log_weight_change(memory_dir, exp["id"], old_w, new_w, reason)
            updates.append({
                "id": exp["id"],
                "old_weight": round(old_w, 4),
                "new_weight": round(new_w, 4),
                "reason": reason,
            })

    _save_experiences(experiences, memory_dir)
    print(json.dumps({"updates": updates}, indent=2))


def cmd_update_weight(args):
    """Manually update weight for a specific experience."""
    memory_dir = args.memory_dir
    experiences = _load_experiences(memory_dir)

    found = False
    for exp in experiences:
        if exp["id"] == args.id:
            old_w = exp.get("weight", INITIAL_WEIGHT)
            new_w = max(MIN_WEIGHT, min(MAX_WEIGHT, old_w + args.delta))
            exp["weight"] = new_w
            reason = args.reason or f"manual adjustment delta={args.delta}"
            _log_weight_change(memory_dir, exp["id"], old_w, new_w, reason)
            _save_experiences(experiences, memory_dir)
            print(json.dumps({
                "id": exp["id"],
                "old_weight": round(old_w, 4),
                "new_weight": round(new_w, 4),
                "reason": reason,
            }, indent=2))
            found = True
            break

    if not found:
        print(json.dumps({"error": f"Experience {args.id} not found"}))


def cmd_list(args):
    """List all experiences."""
    memory_dir = args.memory_dir
    experiences = _load_experiences(memory_dir)

    if args.category:
        experiences = [e for e in experiences if e.get("category") == args.category]

    sort_key = args.sort or "date"
    if sort_key == "weight":
        experiences.sort(key=lambda e: e.get("weight", 1.0), reverse=True)
    elif sort_key == "use_count":
        experiences.sort(key=lambda e: e.get("use_count", 0), reverse=True)
    else:  # date
        experiences.sort(key=lambda e: e.get("created_at", ""), reverse=True)

    items = []
    for exp in experiences:
        items.append({
            "id": exp["id"],
            "task": exp.get("task", ""),
            "lesson": exp.get("lesson", ""),
            "failure_reason": exp.get("failure_reason", ""),
            "category": exp.get("category", ""),
            "tags": exp.get("tags", []),
            "weight": round(exp.get("weight", 1.0), 4),
            "use_count": exp.get("use_count", 0),
            "success_count": exp.get("success_count", 0),
            "is_failure_experience": exp.get("is_failure_experience", False),
            "created_at": exp.get("created_at", ""),
        })

    print(json.dumps({"experiences": items, "count": len(items)}, indent=2))


def cmd_stats(args):
    """Show statistics about the experience database."""
    memory_dir = args.memory_dir
    experiences = _load_experiences(memory_dir)

    if not experiences:
        print(json.dumps({"total": 0, "message": "No experiences in database"}))
        return

    categories = {}
    weights = []
    total_use = 0
    total_success = 0
    failure_count = 0

    for exp in experiences:
        cat = exp.get("category", "general")
        categories[cat] = categories.get(cat, 0) + 1
        weights.append(exp.get("weight", 1.0))
        total_use += exp.get("use_count", 0)
        total_success += exp.get("success_count", 0)
        if exp.get("is_failure_experience"):
            failure_count += 1

    avg_w = sum(weights) / len(weights)
    min_w = min(weights)
    max_w = max(weights)

    top_5 = sorted(experiences, key=lambda e: e.get("weight", 1.0), reverse=True)[:5]
    bottom_5 = sorted(experiences, key=lambda e: e.get("weight", 1.0))[:5]

    stats = {
        "total_experiences": len(experiences),
        "failure_experiences": failure_count,
        "categories": categories,
        "weight_stats": {
            "mean": round(avg_w, 4),
            "min": round(min_w, 4),
            "max": round(max_w, 4),
        },
        "usage_stats": {
            "total_retrievals": total_use,
            "total_successes": total_success,
            "success_rate": round(total_success / max(total_use, 1), 4),
        },
        "top_weighted": [{"id": e["id"], "task": e.get("task", "")[:80], "weight": round(e.get("weight", 1.0), 4)} for e in top_5],
        "lowest_weighted": [{"id": e["id"], "task": e.get("task", "")[:80], "weight": round(e.get("weight", 1.0), 4)} for e in bottom_5],
    }

    print(json.dumps(stats, indent=2))


def cmd_prune(args):
    """Remove low-weight experiences."""
    memory_dir = args.memory_dir
    experiences = _load_experiences(memory_dir)
    min_weight = args.min_weight

    to_remove = [e for e in experiences if e.get("weight", 1.0) <= min_weight]
    remaining = [e for e in experiences if e.get("weight", 1.0) > min_weight]

    if args.dry_run:
        print(json.dumps({
            "dry_run": True,
            "would_remove": len(to_remove),
            "would_keep": len(remaining),
            "removals": [{"id": e["id"], "task": e.get("task", "")[:80], "weight": e.get("weight", 1.0)} for e in to_remove],
        }, indent=2))
    else:
        _save_experiences(remaining, memory_dir)
        print(json.dumps({
            "removed": len(to_remove),
            "remaining": len(remaining),
        }, indent=2))


def cmd_decay(args):
    """Apply weight decay to all experiences (forgetting mechanism)."""
    memory_dir = args.memory_dir
    experiences = _load_experiences(memory_dir)
    factor = args.factor

    updates = []
    for exp in experiences:
        old_w = exp.get("weight", INITIAL_WEIGHT)
        # Only decay if success rate < 50% or never used
        use_count = exp.get("use_count", 0)
        success_count = exp.get("success_count", 0)
        if use_count == 0 or (success_count / max(use_count, 1)) < 0.5:
            new_w = max(old_w * factor, MIN_WEIGHT)
            if new_w != old_w:
                exp["weight"] = new_w
                _log_weight_change(memory_dir, exp["id"], old_w, new_w, f"periodic_decay factor={factor}")
                updates.append({"id": exp["id"], "old": round(old_w, 4), "new": round(new_w, 4)})

    _save_experiences(experiences, memory_dir)
    print(json.dumps({"decayed": len(updates), "total": len(experiences), "factor": factor}, indent=2))


def cmd_export(args):
    """Export experiences to JSON or JSONL (without embeddings for readability)."""
    memory_dir = args.memory_dir
    experiences = _load_experiences(memory_dir)

    # Strip embeddings for export
    cleaned = []
    for exp in experiences:
        e = {k: v for k, v in exp.items() if k != "embedding"}
        cleaned.append(e)

    if args.format == "json":
        print(json.dumps(cleaned, indent=2, default=str))
    else:
        for e in cleaned:
            print(json.dumps(e, default=str))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Use a parent parser for --memory-dir so it works with any subcommand
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--memory-dir", default=DEFAULT_MEMORY_DIR,
                        help=f"Memory directory (default: {DEFAULT_MEMORY_DIR})")

    parser = argparse.ArgumentParser(description="Live-Evo Memory Manager")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # init
    subparsers.add_parser("init", parents=[parent], help="Initialize memory directory")

    # add
    p_add = subparsers.add_parser("add", parents=[parent], help="Add a new experience")
    p_add.add_argument("--task", required=True, help="Task description")
    p_add.add_argument("--lesson", required=True, help="Actionable lesson learned")
    p_add.add_argument("--failure-reason", default="", help="Why the approach failed")
    p_add.add_argument("--tags", default="", help="Comma-separated tags")
    p_add.add_argument("--category", default="general", help="Category")
    p_add.add_argument("--context", default="", help="Additional context")
    p_add.add_argument("--is-failure", action="store_true", help="Mark as failure experience")

    # search
    p_search = subparsers.add_parser("search", parents=[parent], help="Search for relevant experiences")
    p_search.add_argument("--query", required=True, help="Search query")
    p_search.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Number of results")
    p_search.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Min similarity threshold")

    # feedback
    p_fb = subparsers.add_parser("feedback", parents=[parent], help="Provide outcome feedback for experiences")
    p_fb.add_argument("--ids", required=True, help="Comma-separated experience IDs")
    p_fb.add_argument("--outcome", required=True, choices=["success", "failure"], help="Outcome")

    # update-weight
    p_uw = subparsers.add_parser("update-weight", parents=[parent], help="Manually adjust experience weight")
    p_uw.add_argument("--id", required=True, help="Experience ID")
    p_uw.add_argument("--delta", type=float, required=True, help="Weight change amount")
    p_uw.add_argument("--reason", default="", help="Reason for adjustment")

    # list
    p_list = subparsers.add_parser("list", parents=[parent], help="List all experiences")
    p_list.add_argument("--category", default="", help="Filter by category")
    p_list.add_argument("--sort", choices=["weight", "date", "use_count"], default="date", help="Sort order")

    # stats
    subparsers.add_parser("stats", parents=[parent], help="Show database statistics")

    # prune
    p_prune = subparsers.add_parser("prune", parents=[parent], help="Remove low-weight experiences")
    p_prune.add_argument("--min-weight", type=float, default=0.2, help="Remove experiences at or below this weight")
    p_prune.add_argument("--dry-run", action="store_true", help="Preview without removing")

    # decay
    p_decay = subparsers.add_parser("decay", parents=[parent], help="Apply weight decay")
    p_decay.add_argument("--factor", type=float, default=DEFAULT_DECAY, help="Decay factor (0-1)")

    # export
    p_export = subparsers.add_parser("export", parents=[parent], help="Export experiences")
    p_export.add_argument("--format", choices=["json", "jsonl"], default="jsonl", help="Export format")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    commands = {
        "init": cmd_init,
        "add": cmd_add,
        "search": cmd_search,
        "feedback": cmd_feedback,
        "update-weight": cmd_update_weight,
        "list": cmd_list,
        "stats": cmd_stats,
        "prune": cmd_prune,
        "decay": cmd_decay,
        "export": cmd_export,
    }

    cmd_func = commands.get(args.command)
    if cmd_func:
        cmd_func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
